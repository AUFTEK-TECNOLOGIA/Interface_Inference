import { createContext, useCallback, useContext, useEffect, useMemo, useState, useRef } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Handle,
  Background,
  Controls,
  MiniMap,
  reconnectEdge,
  SelectionMode,
} from "@xyflow/react";
import axios from "axios";
import "@xyflow/react/dist/style.css";
import "./App.css";

import ResultsPanel from "./components/ResultsPanel";
import AppHeader from "./components/AppHeader";
import BlocksSidebar from "./components/BlocksSidebar";
import GraphModal from "./components/GraphModal";
import ConfirmDeleteModal from "./components/ConfirmDeleteModal";
import BlockResultsModal from "./components/BlockResultsModal";
import ModelCandidatesPanel from "./components/ModelCandidatesPanel";
import DatasetSelector from "./components/DatasetSelector";
import TrainingStudio from "./components/TrainingStudio";
import TrainingModalBody from "./components/TrainingModalBody";
import { useI18n } from "./locale/i18n";
import { TRAINING_ALGO_PARAM_SCHEMA, parseExperimentIdsText as parseExperimentIdsInput, buildTrainingParamsForAlgorithm as buildTrainingParamsByAlgorithm } from "./modulos/trainingModule";
import { getFlowColorFromLabel, getBlockCardCategory as getBlockCardCategoryModule } from "./modulos/flowEditorModule";
import { sanitizeColor, resolveWorkspaceLogoSrc as resolveWorkspaceLogoSrcModule, workspaceInitials } from "./modulos/workspaceModule";
import { buildPreparedSteps as buildPreparedStepsModule, collectSimulationGraphs } from "./modulos/simulationModule";
import usePipelineStudioState from "./hooks/usePipelineStudioState";

const PipelineStudioContext = createContext(null);

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8001";

// Helper para extrair mensagem de erro (suporta erros Pydantic)
const extractErrorMessage = (err) => {
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // Pydantic validation errors
    return detail
      .map((e) => (typeof e === "object" && e.msg ? `${e.loc?.join?.(".") || ""}: ${e.msg}` : String(e)))
      .join("; ");
  }
  if (detail && typeof detail === "object" && detail.msg) {
    return detail.msg;
  }
  return err?.message || String(err);
};

// Hook para painéis redimensionáveis
const useResizable = (initialWidth, minWidth = 200, maxWidth = 600) => {
  const [width, setWidth] = useState(initialWidth);
  const isResizing = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);

  const startResize = useCallback((e, direction = 1) => {
    isResizing.current = true;
    startX.current = e.clientX;
    startWidth.current = width;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const onMouseMove = (e) => {
      if (!isResizing.current) return;
      const delta = (e.clientX - startX.current) * direction;
      const newWidth = Math.min(maxWidth, Math.max(minWidth, startWidth.current + delta));
      setWidth(newWidth);
    };

    const onMouseUp = () => {
      isResizing.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [width, minWidth, maxWidth]);

  return { width, startResize };
};

  const PipelineNode = ({ data }) => {
  const { t } = useI18n();
  const studio = useContext(PipelineStudioContext);
  // Se dataInputs estiver vazio, usar keys do inputSchema como fallback
  // Filtrar inputs com hidden: true no schema
  const rawInputs = (data.dataInputs && data.dataInputs.length > 0) 
    ? data.dataInputs 
    : Object.keys(data.inputSchema || {});
  const inputs = rawInputs.filter(key => {
    const schema = data.inputSchema?.[key];
    return !schema?.hidden;
  });
  const outputs = (data.dataOutputs && data.dataOutputs.length > 0)
    ? data.dataOutputs
    : Object.keys(data.outputSchema || {});
  const isLabelNode = data.blockName === "label";
  const nodeStyle = data.flowColor ? { "--node-flow-color": data.flowColor } : undefined;
  const getHandleDisplay = (direction, blockName, key) => {
    if (!blockName || !key) return key;
    if (blockName === "condition_branch") {
      if (direction === "out" && key === "data_if_true") return t("handles.conditionBranch.true");
      if (direction === "out" && key === "data_if_false") return t("handles.conditionBranch.false");
      if (direction === "in" && key === "data") return t("handles.common.data");
      if (direction === "in" && key === "condition") return t("handles.common.condition");
    }
    if (blockName === "value_in_list") {
      if (direction === "in" && key === "value") return t("handles.valueInList.value");
      if (direction === "out" && key === "condition") return t("handles.common.condition");
    }
    if (blockName === "numeric_compare") {
      if (direction === "in" && key === "value") return "valor";
      if (direction === "out" && key === "condition") return t("handles.common.condition");
    }
    if (["amplitude_detector", "derivative_detector", "ratio_detector"].includes(blockName)) {
      if (direction === "out" && key === "has_growth") return t("handles.detectors.hasGrowth");
    }
    if (blockName === "ml_detector") {
      if (direction === "out" && key === "detected") return t("handles.detectors.hasGrowth");
    }
    if (blockName === "sensor_fusion") {
      if (direction === "in" && key.startsWith("sensor_data_")) {
        const idx = Number(String(key).split("_").pop());
        if (Number.isFinite(idx)) return t("handles.sensorFusion.sensor", { index: idx });
      }
    }
    return key;
  };

  // Determine node category and styling
  const getNodeCategory = (blockName) => {
    const name = blockName.toLowerCase();
    if (name.includes('experiment_fetch') || name.includes('data') || name.includes('input')) return 'data';
    if (name.includes('filter') || name.includes('process') || name.includes('normalize') || name.includes('smooth') || name.includes('fusion')) return 'process';
    if (name.includes('growth') || name.includes('detect') || name.includes('analysis') || name.includes('feature')) return 'analysis';
    if (name.includes('curve') || name.includes('model') || name.includes('ml') || name.includes('predict') || name.includes('regression')) return 'ml';
    // Controle de fluxo
    if (name.includes('gate') || name.includes('branch') || name.includes('merge') || name.includes('boolean') || name.includes('condition') || name === 'label') return 'flow';
    return 'default';
  };

  const category = getNodeCategory(data.blockName);

  const categoryConfig = {
    data: {
      color: 'var(--block-data)',
      bgColor: '#f3e8ff',
      icon: "D",
      label: t("nodeCategories.data"),
    },
    process: {
      color: 'var(--block-process)',
      bgColor: '#ecfeff',
      icon: "P",
      label: t("nodeCategories.process"),
    },
    analysis: {
      color: 'var(--block-analysis)',
      bgColor: '#fffbeb',
      icon: "A",
      label: t("nodeCategories.analysis"),
    },
    ml: {
      color: 'var(--block-ml)',
      bgColor: '#fdf2f8',
      icon: "ML",
      label: t("nodeCategories.ml"),
    },
    flow: {
      color: 'var(--block-flow)',
      bgColor: '#f0fdf4',
      icon: "F",
      label: t("nodeCategories.flow"),
    },
    default: {
      color: 'var(--color-gray-400)',
      bgColor: '#f8fafc',
      icon: "B",
      label: t("nodeCategories.block"),
    }
  };

  const config = categoryConfig[category];

  return (
    <div
      className={`pipeline-node ${category} ${isLabelNode ? "pipeline-node--label" : ""} ${data.dimmed ? "is-dimmed" : ""}`}
      style={nodeStyle}
    >
      {/* Node Header */}
      <div className="node-header">
        <div
          className="node-icon"
          style={{ backgroundColor: isLabelNode && data.flowColor ? data.flowColor : config.color }}
          title={isLabelNode ? (data.flowLabel || t("flows.none")) : undefined}
        >
          {config.icon}
        </div>
        <div className="node-title-section">
          <div className="node-title-row">
            <div className="node-title" title={data.label}>
              {data.label}
            </div>
          </div>
          <div className="node-meta">
            <div className="node-category">{config.label}</div>
            <span
              className="node-flow-badge"
              title={data.flowLabel || t("flows.none")}
              style={data.flowColor ? { borderColor: data.flowColor, color: data.flowColor } : undefined}
            >
              {data.flowLabel || t("flows.none")}
            </span>
          </div>
        </div>
      </div>

      {/* Node Body */}
      <div className="node-body">
        <div className="node-description">{data.description}</div>

        {/* Input Handles */}
        {inputs.length > 0 && (
          <div className="node-section">
            <div className="section-label">Inputs</div>
            <div className="handles-list">
              {inputs.map((key, index) => (
                <div key={key} className="handle-item input-handle">
                  <Handle
                    type="target"
                    position="left"
                    id={`${data.blockName}-in-${key}`}
                    className="node-handle input"
                    style={{ left: '-6px' }}
                  />
                  <span className="handle-label">{getHandleDisplay("in", data.blockName, key)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Output Handles */}
        {outputs.length > 0 && (
          <div className="node-section">
            <div className="section-label">Outputs</div>
            <div className="handles-list">
              {outputs.map((key, index) => (
                <div key={key} className="handle-item output-handle">
                  <span className="handle-label">{getHandleDisplay("out", data.blockName, key)}</span>
                  <Handle
                    type="source"
                    position="right"
                    id={`${data.blockName}-out-${key}`}
                    className="node-handle output"
                    style={{ right: '-6px' }}
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="node-footer">
        <div className="node-actions">
          {studio?.openConfigModalForNode && (
            <button
              type="button"
              className="node-action node-config"
              aria-label={t("configuration.openLabel")}
              title={t("configuration.openLabel")}
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                studio.openConfigModalForNode(data.stepId);
              }}
            >
              C
            </button>
          )}
          {studio?.openBlockResultsModal && (
            <button
              type="button"
              className="node-action node-results"
              aria-label={t("blockResults.openLabel")}
              title={t("blockResults.openLabel")}
              disabled={!studio.simulation?.step_results?.[data.stepId]}
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                studio.openBlockResultsModal(data.stepId);
              }}
            >
              R
            </button>
          )}
          {studio?.openHelpModal && (
            <button
              type="button"
              className="node-action node-help"
              aria-label={t("helper.openLabel")}
              title={t("helper.openLabel")}
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                const blockFromLibrary = studio.library?.blocks?.find((b) => b.name === data.blockName);
                const fallbackBlock = {
                  name: data.blockName,
                  description: data.description || "",
                  data_inputs: inputs,
                  data_outputs: outputs,
                  config_inputs: [],
                  input_schema: data.inputSchema || {},
                  output_schema: data.outputSchema || {},
                };
                studio.openHelpModal(blockFromLibrary || fallbackBlock);
              }}
            >
              ?
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

const nodeTypes = {
  pipelineNode: PipelineNode,
};

const defaultLibrary = {
  blocks: [],
  filters: [],
  growth_detectors: [],
  feature_extractors: [],
  curve_models: [],
};

function App() {
  const { t } = useI18n();
  const [workspaceHomeMode, setWorkspaceHomeMode] = useState("available");
  const [isDarkTheme, setIsDarkTheme] = useState(() => {
    try {
      return localStorage.getItem("pipelineStudio.theme") === "dark";
    } catch {
      return false;
    }
  });
  const {
    helpModal, setHelpModal, helpTab, setHelpTab, blockResultsModal, setBlockResultsModal,
    resultsModalOpen, setResultsModalOpen, configModalOpen, setConfigModalOpen, valueInListDraft, setValueInListDraft,
    library, setLibrary, nodes, setNodes, defaultOnNodesChange, edges, setEdges, onEdgesChange,
    selectedNode, setSelectedNode, selectedNodes, setSelectedNodes, selectedEdge, setSelectedEdge,
    clipboard, setClipboard, configFieldErrors, setConfigFieldErrors, pipelineName, setPipelineName,
    workspace, setWorkspace, workspaceHomeOpen, setWorkspaceHomeOpen, workspaceList, setWorkspaceList,
    workspaceListLoading, setWorkspaceListLoading, workspaceActionLoading, setWorkspaceActionLoading,
    workspaceError, setWorkspaceError, newTenantName, setNewTenantName, selectedWorkspaceKey, setSelectedWorkspaceKey,
    workspaceCardMenuKey, setWorkspaceCardMenuKey, workspaceMetaDraft, setWorkspaceMetaDraft,
    duplicateModal, setDuplicateModal, editModal, setEditModal, deleteModal, setDeleteModal,
    versionsModal, setVersionsModal, versionActionsModal, setVersionActionsModal, versionLogsModal, setVersionLogsModal,
    deleteVersionModal, setDeleteVersionModal, renameVersionModal, setRenameVersionModal,
    simulation, setSimulation, trainModal, setTrainModal, trainModelsDraft, setTrainModelsDraft,
    trainBlockModal, setTrainBlockModal, candidatesModal, setCandidatesModal, datasetSelectorOpen, setDatasetSelectorOpen,
    trainingStudioOpen, setTrainingStudioOpen, graphModalOpen, setGraphModalOpen, graphModalSrc, setGraphModalSrc,
    graphModalTitle, setGraphModalTitle, graphList, setGraphList, graphIndex, setGraphIndex,
    isRunning, setIsRunning, error, setError, inspectedBlock, setInspectedBlock,
    blocksQuery, setBlocksQuery, favoriteBlocks, setFavoriteBlocks, recentBlocks, setRecentBlocks,
    contextMenu, setContextMenu, useDefaultExperiment, setUseDefaultExperiment,
    viewport, setViewport, reactFlowWrapper, reactFlowInstance,
  } = usePipelineStudioState(defaultLibrary);

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-theme", isDarkTheme ? "dark" : "light");
    try {
      localStorage.setItem("pipelineStudio.theme", isDarkTheme ? "dark" : "light");
    } catch {}
  }, [isDarkTheme]);

  // Compatibilidade: normaliza handles de edges para evitar erros quando o schema muda (ex: growth_features fit_results -> data)
  useEffect(() => {
    if (!nodes.length || !edges.length) return;

    const nodeIoById = new Map(
      (nodes || []).map((node) => {
        const blockName = node?.data?.blockName || "";
        const inputKeys = Array.isArray(node?.data?.dataInputs) ? node.data.dataInputs : [];
        const outputKeys = Array.isArray(node?.data?.dataOutputs) ? node.data.dataOutputs : [];
        const inputHandles = new Set(inputKeys.map((k) => `${blockName}-in-${k}`));
        const outputHandles = new Set(outputKeys.map((k) => `${blockName}-out-${k}`));
        return [node.id, { blockName, inputKeys, outputKeys, inputHandles, outputHandles }];
      })
    );

    const normalizeTargetHandle = (targetHandleRaw, targetNodeId) => {
      const info = nodeIoById.get(targetNodeId);
      if (!info) return targetHandleRaw;

      const targetHandle = String(targetHandleRaw || "");
      if (targetHandle && info.inputHandles.has(targetHandle)) return targetHandle;

      const fallback = () => {
        if (info.inputKeys.length === 1) return `${info.blockName}-in-${info.inputKeys[0]}`;
        if (info.inputKeys.includes("data")) return `${info.blockName}-in-data`;
        if (info.inputKeys.length) return `${info.blockName}-in-${info.inputKeys[0]}`;
        return targetHandleRaw;
      };

      const key = targetHandle.includes("-in-") ? targetHandle.split("-in-")[1] : "";
      if (info.blockName === "growth_features" && key === "fit_results") return `${info.blockName}-in-sensor_data`;
      return fallback();
    };

    const normalizeSourceHandle = (sourceHandleRaw, sourceNodeId) => {
      const info = nodeIoById.get(sourceNodeId);
      if (!info) return sourceHandleRaw;

      const sourceHandle = String(sourceHandleRaw || "");
      if (sourceHandle && info.outputHandles.has(sourceHandle)) return sourceHandle;

      if (info.outputKeys.length === 1) return `${info.blockName}-out-${info.outputKeys[0]}`;
      if (info.outputKeys.includes("data")) return `${info.blockName}-out-data`;
      if (info.outputKeys.length) return `${info.blockName}-out-${info.outputKeys[0]}`;
      return sourceHandleRaw;
    };

    let changed = false;
    const normalized = edges.map((edge) => {
      const nextSourceHandle = normalizeSourceHandle(edge.sourceHandle, edge.source);
      const nextTargetHandle = normalizeTargetHandle(edge.targetHandle, edge.target);
      if (nextSourceHandle !== edge.sourceHandle || nextTargetHandle !== edge.targetHandle) changed = true;
      return { ...edge, sourceHandle: nextSourceHandle, targetHandle: nextTargetHandle };
    });

    if (changed) setEdges(normalized);
  }, [nodes, edges, setEdges]);

  const formatDateTime = useCallback((value) => {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString("pt-BR");
  }, []);

  const parseExperimentIdsText = useCallback((value) => parseExperimentIdsInput(value), []);

  const safeParseJsonObject = useCallback((raw) => {
    const text = String(raw || "").trim();
    if (!text) return {};
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return parsed;
  }, []);

  const buildTrainingParamsForAlgorithm = useCallback((algorithmKey) => buildTrainingParamsByAlgorithm(algorithmKey), []);

  const openTrainModalForWorkspace = useCallback(() => {
    if (!workspace?.tenant || !workspace?.pipeline) return;

    const trainable = new Set(["ml_inference", "ml_inference_series", "ml_inference_multichannel", "ml_forecaster_series"]);
    const candidates = (nodes || []).filter((n) => trainable.has(n?.data?.blockName));

    const draft = {};
    candidates.forEach((node) => {
      draft[node.id] = {
        enabled: true,
        algorithms: ["ridge"],
        activeAlgorithm: "ridge",
        paramsByAlgorithm: { ridge: buildTrainingParamsForAlgorithm("ridge") },
        selectionMetric: "rmse",
        maxTrials: 60,
        expanded: false,
      };
    });

    let protocolId = "";
    try {
      protocolId = localStorage.getItem("pipelineStudio.lastTrainProtocolId") || "";
    } catch {
      protocolId = "";
    }

    setTrainModelsDraft(draft);
    setTrainModal((prev) => ({
      ...prev,
      open: true,
      step: 0,
      protocolId: prev.protocolId || protocolId,
      selectionMetric: prev.selectionMetric || "rmse",
      maxTrials: Number.isFinite(Number(prev.maxTrials)) ? prev.maxTrials : 60,
      result: null,
      error: "",
      running: false,
    }));
  }, [buildTrainingParamsForAlgorithm, nodes, workspace]);

  const getFlowColor = useCallback(
    (flowLabel) => getFlowColorFromLabel(flowLabel || t("flows.none")),
    [t]
  );

  const nodeFlowMetaById = useMemo(() => {
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const incoming = new Map();
    edges.forEach((e) => {
      const list = incoming.get(e.target) || [];
      list.push(e.source);
      incoming.set(e.target, list);
    });
    incoming.forEach((list, key) => {
      const sorted = [...list].sort((a, b) => String(a).localeCompare(String(b)));
      incoming.set(key, sorted);
    });

    const cache = new Map();
    const inProgress = new Set();

    const getLabelFromNode = (node) => {
      const configured = node?.data?.config?.label;
      if (typeof configured === "string" && configured.trim().length > 0) return configured.trim();
      return null;
    };

    const getColorFromNode = (node) => {
      const raw = node?.data?.config?.label_color;
      if (typeof raw !== "string") return null;
      const value = raw.trim();
      if (!value) return null;
      if (/^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(value)) return value;
      return null;
    };

    const compute = (nodeId) => {
      if (cache.has(nodeId)) return cache.get(nodeId);
      if (inProgress.has(nodeId)) {
        const fallback = { label: t("flows.none"), color: getFlowColor(t("flows.none")) };
        return fallback;
      }
      inProgress.add(nodeId);

      const startNode = byId.get(nodeId);
      if (!startNode) {
        const resolved = { label: t("flows.none"), color: getFlowColor(t("flows.none")) };
        cache.set(nodeId, resolved);
        inProgress.delete(nodeId);
        return resolved;
      }

      if (startNode.data?.blockName === "label") {
        const own = getLabelFromNode(startNode);
        const ownColor = getColorFromNode(startNode);
        const label = own || t("flows.none");
        const resolved = { label, color: ownColor || getFlowColor(label) };
        cache.set(nodeId, resolved);
        inProgress.delete(nodeId);
        return resolved;
      }

      const noneLabel = t("flows.none");
      const parents = incoming.get(nodeId) || [];
      const parentLabels = new Map();
      parents.forEach((parentId) => {
        const meta = compute(parentId);
        const label = meta?.label;
        if (!label || label === noneLabel) return;
        if (!parentLabels.has(label)) parentLabels.set(label, meta?.color);
      });

      if (parentLabels.size === 1) {
        const [label, color] = Array.from(parentLabels.entries())[0];
        const resolved = { label, color: color || getFlowColor(label) };
        cache.set(nodeId, resolved);
        inProgress.delete(nodeId);
        return resolved;
      }

      if (parentLabels.size > 1) {
        const resolved = { label: noneLabel, color: getFlowColor(noneLabel) };
        cache.set(nodeId, resolved);
        inProgress.delete(nodeId);
        return resolved;
      }

      const resolved = { label: t("flows.none"), color: getFlowColor(t("flows.none")) };
      cache.set(nodeId, resolved);
      inProgress.delete(nodeId);
      return resolved;
    };

    const result = {};
    nodes.forEach((n) => {
      result[n.id] = compute(n.id);
    });
    return result;
  }, [edges, getFlowColor, nodes, t]);

  const analysisIdIndex = useMemo(() => {
    if (!nodes.length || !edges.length) return new Map();
    const nodeById = new Map(nodes.map((n) => [n.id, n]));
    const outgoing = new Map();
    edges.forEach((e) => {
      const list = outgoing.get(e.source) || [];
      list.push(e.target);
      outgoing.set(e.source, list);
    });

    const noneLabel = t("flows.none");

    const seemsAnalysisRouter = (valueInListNode) => {
      const cfg = valueInListNode?.data?.config || {};
      if (String(cfg.key || "").toLowerCase().includes("analysis")) return true;
      const incomingFromFetch = edges.some((e) => {
        if (e.target !== valueInListNode.id) return false;
        const src = nodeById.get(e.source);
        if (!src) return false;
        if (src?.data?.blockName !== "experiment_fetch") return false;
        const h = String(e.sourceHandle || "").toLowerCase();
        return h.includes("analysis");
      });
      return incomingFromFetch;
    };

    const bfsDescendants = (startId, maxDepth = 12) => {
      const visited = new Set([startId]);
      const queue = [{ id: startId, depth: 0 }];
      const routingNodeIds = new Set([startId]);
      const flowLabels = new Set();

      while (queue.length) {
        const { id, depth } = queue.shift();
        if (depth >= maxDepth) continue;
        const outs = outgoing.get(id) || [];
        for (const next of outs) {
          if (visited.has(next)) continue;
          visited.add(next);
          const n = nodeById.get(next);
          if (!n) continue;

          routingNodeIds.add(next);
          const label = nodeFlowMetaById[next]?.label || noneLabel;
          if (label !== noneLabel) flowLabels.add(label);

          if (n?.data?.blockName === "label") continue;
          queue.push({ id: next, depth: depth + 1 });
        }
      }

      return { routingNodeIds, flowLabels };
    };

    const index = new Map();
    nodes
      .filter((n) => n?.data?.blockName === "value_in_list")
      .filter(seemsAnalysisRouter)
      .forEach((n) => {
        const cfg = n?.data?.config || {};
        const allowed = Array.isArray(cfg.allowed_values) ? cfg.allowed_values : [];
        const analysisIds = allowed.map((v) => String(v).trim()).filter(Boolean);
        if (!analysisIds.length) return;

        const { routingNodeIds, flowLabels } = bfsDescendants(n.id);

        analysisIds.forEach((analysisId) => {
          const current = index.get(analysisId) || {
            analysisId,
            color: getFlowColor(`analysisId:${analysisId}`),
            flowLabels: new Set(),
            routingNodeIds: new Set(),
          };
          flowLabels.forEach((l) => current.flowLabels.add(l));
          routingNodeIds.forEach((id) => current.routingNodeIds.add(id));
          index.set(analysisId, current);
        });
      });

    return index;
  }, [edges, getFlowColor, nodeFlowMetaById, nodes, t]);

  const analysisAreas = useMemo(() => {
    if (!analysisIdIndex.size) return [];
    const nodeById = new Map(nodes.map((n) => [n.id, n]));
    const incoming = new Map();
    edges.forEach((e) => {
      const list = incoming.get(e.target) || [];
      list.push(e.source);
      incoming.set(e.target, list);
    });
    const defaultWidth = 300;
    const defaultHeight = 240;
    const padX = 90;
    const padY = 70;

    const noneLabel = t("flows.none");
    const getUpstreamFlowLabels = (startId, maxDepth = 12) => {
      const labels = new Set();
      const visited = new Set([startId]);
      const queue = [{ id: startId, depth: 0 }];
      while (queue.length) {
        const { id, depth } = queue.shift();
        if (depth >= maxDepth) continue;
        const parents = incoming.get(id) || [];
        for (const parent of parents) {
          if (visited.has(parent)) continue;
          visited.add(parent);
          const label = nodeFlowMetaById[parent]?.label || noneLabel;
          if (label !== noneLabel) labels.add(label);
          const bn = nodeById.get(parent)?.data?.blockName;
          if (bn === "label") continue;
          queue.push({ id: parent, depth: depth + 1 });
        }
      }
      return labels;
    };

    const areas = [];
    Array.from(analysisIdIndex.values())
      .sort((a, b) => String(a.analysisId).localeCompare(String(b.analysisId)))
      .forEach((meta) => {
        let left = Number.POSITIVE_INFINITY;
        let top = Number.POSITIVE_INFINITY;
        let right = Number.NEGATIVE_INFINITY;
        let bottom = Number.NEGATIVE_INFINITY;

        const includeNodeBounds = (id) => {
          const n = nodeById.get(id);
          if (!n) return;
          const w = n.measured?.width || defaultWidth;
          const h = n.measured?.height || defaultHeight;
          left = Math.min(left, n.position.x);
          top = Math.min(top, n.position.y);
          right = Math.max(right, n.position.x + w);
          bottom = Math.max(bottom, n.position.y + h);
        };

        meta.routingNodeIds.forEach(includeNodeBounds);
        nodes.forEach((n) => {
          const flow = nodeFlowMetaById[n.id]?.label || t("flows.none");
          if (meta.flowLabels.has(flow)) includeNodeBounds(n.id);
        });

        // Estender para incluir a "Saída do grupo" (response_pack) que fica fora das lanes (sem flowLabel),
        // mas pertence ao grupo por estar a jusante dos mesmos labels.
        nodes
          .filter((n) => n?.data?.blockName === "response_pack")
          .forEach((packNode) => {
            const upstreamLabels = getUpstreamFlowLabels(packNode.id);
            const intersects = Array.from(upstreamLabels).some((l) => meta.flowLabels.has(l));
            if (!intersects) return;
            includeNodeBounds(packNode.id);
          });

        if (
          left === Number.POSITIVE_INFINITY ||
          top === Number.POSITIVE_INFINITY ||
          right === Number.NEGATIVE_INFINITY ||
          bottom === Number.NEGATIVE_INFINITY
        ) {
          return;
        }

        areas.push({
          analysisId: meta.analysisId,
          color: meta.color,
          left: left - padX,
          top: top - padY,
          width: right - left + padX * 2,
          height: bottom - top + padY * 2,
          flowCount: meta.flowLabels.size,
        });
      });

    return areas;
  }, [analysisIdIndex, edges, nodeFlowMetaById, nodes, t]);

  // Painéis redimensionáveis
  const leftPanel = useResizable(320, 260, 520);

  const openHelpModal = useCallback((block) => {
    if (!block) return;
    setInspectedBlock(block);
    setHelpModal({ open: true, block });
    setHelpTab("overview");
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("pipelineStudio.favoriteBlocks", JSON.stringify(favoriteBlocks));
    } catch {
      // ignore
    }
  }, [favoriteBlocks]);

  useEffect(() => {
    try {
      localStorage.setItem("pipelineStudio.recentBlocks", JSON.stringify(recentBlocks));
    } catch {
      // ignore
    }
  }, [recentBlocks]);

  const closeHelpModal = useCallback(() => {
    setHelpModal({ open: false, block: null });
    setHelpTab("overview");
  }, []);

  const openBlockResultsModal = useCallback((stepId) => {
    if (!stepId) return;
    setBlockResultsModal({ open: true, stepId });
  }, []);

  const closeBlockResultsModal = useCallback(() => {
    setBlockResultsModal({ open: false, stepId: null });
  }, []);

  const openResultsModal = useCallback(() => {
    setResultsModalOpen(true);
  }, []);

  const closeResultsModal = useCallback(() => {
    setResultsModalOpen(false);
  }, []);

  const closeConfigModal = useCallback(() => {
    setConfigModalOpen(false);
  }, []);

  const openConfigModalForNode = useCallback((nodeId) => {
    if (!nodeId) return;
    const node = nodesRef.current?.find((n) => n.id === nodeId);
    if (!node) return;
    setSelectedEdge(null);
    setSelectedNodes([node]);
    setSelectedNode(node);
    setConfigModalOpen(true);
  }, []);

  // Ref para guardar nodes atual (evita closure stale)
  const nodesRef = useRef(nodes);
  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  // (painel interativo de analysisId removido: mantemos apenas o retângulo visual no canvas)

  const selectedNodesRef = useRef(selectedNodes);
  useEffect(() => {
    selectedNodesRef.current = selectedNodes;
  }, [selectedNodes]);

  // Custom onNodesChange para sincronizar seleção do ReactFlow com nosso estado
  const onNodesChange = useCallback((changes) => {
    defaultOnNodesChange(changes);
    
    // Processar mudanças de seleção
    const selectionChanges = changes.filter(c => c.type === 'select');
    if (selectionChanges.length > 0) {
      // Atualizar lista de nós selecionados
      setNodes((currentNodes) => {
        const newSelectedNodes = currentNodes.filter(n => {
          const change = selectionChanges.find(c => c.id === n.id);
          if (change) {
            return change.selected;
          }
          return n.selected;
        });
        
        // Atualizar selectedNodes state
        setTimeout(() => {
          setSelectedNodes(newSelectedNodes);
          
          // Se apenas um nó selecionado, também setar selectedNode para o painel de config
          if (newSelectedNodes.length === 1) {
            setSelectedNode(newSelectedNodes[0]);
            setSelectedEdge(null);
          } else if (newSelectedNodes.length === 0) {
            setSelectedNode(null);
          } else {
            // Múltiplos selecionados - limpar selectedNode para não mostrar config
            setSelectedNode(null);
          }
        }, 0);
        
        return currentNodes;
      });
    }
  }, [defaultOnNodesChange, setNodes]);

  useEffect(() => {
    axios
      .get(`${API_URL}/pipelines/library`)
      .then((response) => {
        const payload = { ...defaultLibrary, ...(response.data || {}) };
        setLibrary(payload);
        if (payload.blocks?.length) {
          setInspectedBlock(payload.blocks[0]);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    // remove editor JSON: não precisamos manter draft
  }, [selectedNode]);

  useEffect(() => {
    setConfigFieldErrors({});
  }, [selectedNode?.id]);

  useEffect(() => {
    setValueInListDraft("");
  }, [selectedNode?.id]);

  // Sincronizar estado local do checkbox com selectedNode (apenas quando muda o nó selecionado)
  useEffect(() => {
    if (selectedNode?.data?.config) {
      setUseDefaultExperiment(Boolean(selectedNode.data.config.use_default_experiment));
    } else {
      setUseDefaultExperiment(false);
    }
  }, [selectedNode?.id, selectedNode?.data?.config?.use_default_experiment]);

  const onConnect = useCallback(
    (params) => {
      // criar aresta manualmente (compatível com API disponível em @xyflow/react)
      setEdges((eds) => {
        const id = `e${Date.now()}`;
        const edge = { 
          id, 
          source: params.source, 
          target: params.target, 
          sourceHandle: params.sourceHandle,  // Preservar handle de origem específico
          targetHandle: params.targetHandle,  // Preservar handle de destino específico
          animated: true 
        };
        return [...eds, edge];
      });
    },
    [setEdges]
  );

  // Ref para controlar estado de reconexão
  const edgeReconnectSuccessful = useRef(true);

  // Handler para início da reconexão
  const onReconnectStart = useCallback(() => {
    edgeReconnectSuccessful.current = false;
  }, []);

  // Handler para reconectar edge (arrastar ponta para outro nó)
  const onReconnect = useCallback(
    (oldEdge, newConnection) => {
      console.log('Edge reconnected:', oldEdge.id, '->', newConnection);
      edgeReconnectSuccessful.current = true;
      setEdges((eds) => reconnectEdge(oldEdge, newConnection, eds));
      // Limpar seleção após reconexão
      setSelectedEdge(null);
    },
    [setEdges]
  );

  // Handler para fim da reconexão (remover edge se soltar no vazio)
  const onReconnectEnd = useCallback(
    (_, edge) => {
      if (!edgeReconnectSuccessful.current) {
        // Se não conectou a nada, remover a edge
        setEdges((eds) => eds.filter((e) => e.id !== edge.id));
      }
      edgeReconnectSuccessful.current = true;
    },
    [setEdges]
  );

  // Helper para obter o centro da viewport atual
  const getViewportCenter = useCallback(() => {
    if (!reactFlowInstance.current || !reactFlowWrapper.current) {
      return { x: 200, y: 200 };
    }
    
    const { width, height } = reactFlowWrapper.current.getBoundingClientRect();
    const centerScreen = { x: width / 2, y: height / 2 };
    
    // Converter coordenadas de tela para coordenadas do flow
    try {
      const flowPosition = reactFlowInstance.current.screenToFlowPosition(centerScreen);
      return flowPosition;
    } catch (e) {
      return { x: 200, y: 200 };
    }
  }, []);

  const getBlockDisplayName = useCallback((blockName) => {
    const legacyOverrides = {
      experiment_fetch: "Experimento",
      label: "Etiqueta",

      turbidimetry_extraction: "Turbidimetria",
      nephelometry_extraction: "Nefelometria",
      fluorescence_extraction: "Fluorescência",

      temperatures_extraction: "Temperaturas",
      power_supply_extraction: "Fonte de energia",
      peltier_currents_extraction: "Correntes do Peltier",
      nema_currents_extraction: "Correntes NEMA",
      resonant_frequencies_extraction: "Frequências ressonantes",
      control_state_extraction: "Estado de controle",

      time_slice: "Corte temporal",
      outlier_removal: "Remoção de outliers",
      moving_average_filter: "Média móvel",
      savgol_filter: "Savitzky-Golay",
      median_filter: "Filtro de mediana",
      lowpass_filter: "Passa-baixa",
      exponential_filter: "Filtro exponencial",

      derivative: "Derivada",
      integral: "Integral",
      normalize: "Normalização",

      rgb_conversion: "Conversão RGB",
      cmyk_conversion: "Conversão CMYK",
      hsv_conversion: "Conversão HSV",
      hsl_conversion: "Conversão HSL",
      lab_conversion: "Conversão LAB",
      luv_conversion: "Conversão LUV",

      curve_fitting: "Ajuste de curva",
      curve_fit: "Ajuste de curva",
      curve_fit_best: "Melhor ajuste",

      statistical_features: "Features estatísticas",
      temporal_features: "Features temporais",
      shape_features: "Features de forma",
      growth_features: "Features de crescimento",
      features_merge: "Combinar features",

      amplitude_detector: "Detector de amplitude",
      derivative_detector: "Detector de derivada",
      ratio_detector: "Detector de razão",

      boolean_extractor: "Extrator booleano",
      value_in_list: "Verificar valor na lista",
      numeric_compare: "Comparar valor numérico",
      condition_gate: "Portão condicional",
      and_gate: "Porta AND",
      or_gate: "Porta OR",
      not_gate: "Porta NOT",
      condition_branch: "Ramificação condicional",
      merge: "Junção de fluxos",

      ml_inference: "Inferência ML",
      response_builder: "Construtor de resposta",
      response_pack: "Saída do grupo",
      response_merge: "Unir saídas",
      feature_extraction: "Extração de features",
    };

    // Padrão recomendado (nomes curtos, consistentes e orientados à ação).
    // Observação: não altera o identificador do bloco (blockName), apenas o nome exibido.
    const standardOverrides = {
      // Entrada / fonte
      experiment_fetch: "Carregar experimento",

      // Fluxo
      label: "Marcador de fluxo",
      value_in_list: "IN",
      numeric_compare: "CMP",
      boolean_extractor: "Converter para booleano",
      condition_gate: "Portao condicional",
      condition_branch: "Ramificacao condicional",
      and_gate: "AND",
      or_gate: "OR",
      not_gate: "NOT",
      merge: "Resolver fluxo ativo",

      // Aquisição / extração de sensores
      turbidimetry_extraction: "Turbidimetria",
      nephelometry_extraction: "Nefelometria",
      fluorescence_extraction: "Fluorescencia",
      resonant_frequencies_extraction: "Frequencia de ressonancia",
      temperatures_extraction: "Temperaturas",
      power_supply_extraction: "Fonte de alimentacao",
      peltier_currents_extraction: "Pastilha Peltier",
      nema_currents_extraction: "Agitador magnetico",
      control_state_extraction: "Estados de controle",

      // Preparação / limpeza
      time_slice: "Selecionar janela",
      sensor_fusion: "Combinar sensores",
      outlier_removal: "Remover outliers",
      moving_average_filter: "Filtro: Media movel",
      savgol_filter: "Filtro: Savitzky-Golay",
      median_filter: "Filtro: Mediana",
      lowpass_filter: "Filtro: Passa-baixa",
      exponential_filter: "Filtro: Exponencial",

      // Transformações
      derivative: "Derivada",
      integral: "Integral",
      normalize: "Normalizacao",

      // Conversões (espaço de cor)
      rgb_conversion: "RGB",
      cmyk_conversion: "CMYK",
      hsv_conversion: "HSV",
      hsl_conversion: "HSL",
      hsb_conversion: "HSB",
      lab_conversion: "LAB",
      luv_conversion: "LUV",
      xyz_conversion: "XYZ",
      xyy_conversion: "xyY",

      // Detecção / decisão
      amplitude_detector: "Detectar crescimento (amplitude)",
      derivative_detector: "Detectar crescimento (derivada)",
      ratio_detector: "Detectar crescimento (razao)",

      // Features
      statistical_features: "Features (estatisticas)",
      temporal_features: "Features (temporais)",
      shape_features: "Features (forma)",
      growth_features: "Features (crescimento)",
      feature_extraction: "Features (geral)",
      features_merge: "Unir features",

      // Modelo / predição
      curve_fit: "Ajustar curva",
      curve_fit_best: "Escolher melhor ajuste",
      ml_inference: "Inferencia ML",
      ml_inference_series: "Inferencia ML (serie)",
      ml_inference_multichannel: "Inferencia ML (multicanal)",
      ml_forecaster_series: "Forecaster ML (serie)",
      ml_transform_series: "Transformar serie (ML)",
      ml_detector: "Detector (ML)",

      // Resposta
      response_pack: "Empacotar saida do grupo",
      response_merge: "Selecionar saida ativa",
      response_builder: "Construir resposta (API)",
    };

    const overrides = { ...legacyOverrides, ...standardOverrides };

    if (overrides[blockName]) return overrides[blockName];

    return String(blockName)
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }, []);

  const getNextInstanceNumber = useCallback((nodesList, blockName) => {
    const count = (nodesList || []).filter((n) => n.data?.blockName === blockName).length;
    return count + 1;
  }, []);

  const normalizeLoadedLabel = useCallback(
    (nodesList, blockName, currentLabel) => {
      if (typeof currentLabel === "string" && currentLabel.trim().length > 0 && currentLabel !== blockName) {
        return currentLabel;
      }
      const displayName = getBlockDisplayName(blockName);
      const num = getNextInstanceNumber(nodesList, blockName);
      return `${displayName} ${num}`;
    },
    [getBlockDisplayName, getNextInstanceNumber]
  );

  const addBlockToCanvas = useCallback((block, preConfig = null, position = null) => {
    setRecentBlocks((prev) => [block.name, ...(prev || []).filter((n) => n !== block.name)].slice(0, 12));
    setNodes((nds) => {
      const id = `${block.name}_${nds.length + 1}`;
      const baseConfig = block.config || {};
      
      // Para experiment_fetch, começar com valores padrão (gráficos ativados)
      let initialConfig = { ...baseConfig };
      if (block.name === "experiment_fetch") {
        initialConfig = {
          ...initialConfig,
          experimentId: "",
          analysisId: "",
          tenant: "",
          use_default_experiment: false,
          generate_output_graphs: true,
          plot_mode: "all",
          include_legend: true,
          include_labels: true,
          time_from_zero: true,
          // selecionar todos sensores por padrão
          sensors: ["turbidimetry", "nephelometry", "fluorescence"],
          // selecionar todos os canais de plot por padrão
          plot_channel: ["f1","f2","f3","f4","f5","f6","f7","f8","clear","nir"],
        };
      }

      if (block.name === "sensor_fusion") {
        initialConfig = {
          inputs_count: 2,
          merge_mode: "intersection",
          resample_step: undefined,
          pad_value: 0,
          sources: [
            { input: "sensor_data_1", prefix: "", channels: [] },
            { input: "sensor_data_2", prefix: "", channels: [] },
          ],
          ...initialConfig,
        };
      }
      
      const mergedConfig = preConfig ? { ...initialConfig, ...preConfig } : initialConfig;

      // Usar data_inputs do bloco, ou fallback para keys do input_schema
      let resolvedDataInputs = (block.data_inputs && block.data_inputs.length > 0) 
        ? block.data_inputs 
        : Object.keys(block.input_schema || {});
      if (block.name === "sensor_fusion") {
        const raw = Number(mergedConfig.inputs_count);
        const clamped = Math.max(1, Math.min(6, Number.isFinite(raw) ? raw : 2));
        resolvedDataInputs = Array.from({ length: clamped }, (_, i) => `sensor_data_${i + 1}`);
      }
      if (["response_builder", "response_pack", "response_merge"].includes(block.name)) {
        const raw = Number(mergedConfig.inputs_count);
        if (Number.isFinite(raw)) {
          const clamped = Math.max(1, Math.min(8, raw));
          resolvedDataInputs = Array.from({ length: clamped }, (_, i) => `input_${i + 1}`);
        }
      }
      
      // Se não tem posição definida, usar centro da viewport com pequeno offset aleatório
      let nodePosition = position;
      if (!nodePosition) {
        const center = getViewportCenter();
        // Adiciona pequeno offset aleatório para não sobrepor blocos
        nodePosition = {
          x: center.x - 100 + (Math.random() * 40 - 20),
          y: center.y - 75 + (Math.random() * 40 - 20),
        };
      }
      
      const newNode = {
        id,
        type: "pipelineNode",
        position: nodePosition,
        data: {
          label: normalizeLoadedLabel(nds, block.name),
          description: block.description,
          blockName: block.name,
          config: mergedConfig,
          inputSchema: block.input_schema || {},
          outputSchema: block.output_schema || {},
          dataInputs: resolvedDataInputs,
          dataOutputs: (block.data_outputs && block.data_outputs.length > 0) 
            ? block.data_outputs 
            : Object.keys(block.output_schema || {}),
          configInputs: block.config_inputs || [],
        },
      };
      return nds.concat(newNode);
    });
  }, [setNodes, getViewportCenter, normalizeLoadedLabel]);

  const onDragStartBlock = useCallback(
    (event, block) => {
      // Selecionar o bloco ao iniciar drag
      setInspectedBlock(block);
      event.dataTransfer.setData("application/reactflow", block.name);
      event.dataTransfer.setData("text/plain", block.name);
      event.dataTransfer.effectAllowed = "move";
      console.log("[drag-start]", block.name);
    },
    [setInspectedBlock]
  );

  const getBlockCardCategory = useCallback((blockName) => getBlockCardCategoryModule(blockName), []);

  const getBlockCardTags = useCallback((blockName) => {
    const name = String(blockName || "").toLowerCase();
    // Tags curtas e consistentes (etapa + detalhe opcional).
    // Regra: no máximo 2 tags para não poluir o card.
    const tags = [];

    // Entrada
    if (name === "experiment_fetch") return ["Entrada"];

    // Resposta
    if (name === "response_builder") return ["Resposta", "API"];
    if (name.startsWith("response_")) return ["Resposta"];

    // Modelo / predição
    if (
      ["ml_inference", "ml_inference_series", "ml_inference_multichannel", "ml_forecaster_series", "ml_transform_series", "ml_detector"].includes(name) ||
      name.includes("curve_fit")
    )
      return ["Modelo"];

    // Features
    if (name.includes("features_merge")) return ["Features", "Merge"];
    if (name.includes("feature")) return ["Features"];

    // Detecção
    if (name.includes("detector") || name.includes("growth")) return ["Deteccao", "Crescimento"];

    // Sensores (extrações)
    if (name.endsWith("_extraction")) {
      // Já existe subdivisão visual; aqui só reforçamos a etapa e, quando útil, um detalhe.
      tags.push("Sensores");

      let detail = null;
      if (["turbidimetry_extraction", "nephelometry_extraction", "fluorescence_extraction", "resonant_frequencies_extraction"].includes(name)) {
        detail = "Principal";
      } else if (name === "temperatures_extraction") {
        detail = "Temperatura";
      } else if (name === "power_supply_extraction") {
        detail = "Energia";
      } else if (name === "peltier_currents_extraction") {
        detail = "Peltier";
      } else if (name === "nema_currents_extraction") {
        detail = "Agitador";
      } else if (name === "control_state_extraction") {
        detail = "Controle";
      } else {
        detail = "Outros";
      }

      tags.push(detail);
      return tags.slice(0, 2);
    }

    // Pré-processamento (limpeza / recorte / filtros)
    if (name === "time_slice") return ["Pre-processamento", "Recorte"];
    if (name.includes("outlier")) return ["Pre-processamento", "Outliers"];
    if (name.includes("filter")) return ["Pre-processamento", "Filtro"];

    // Transformações
    if (name.includes("conversion")) return ["Transformacoes", "Cor"];
    if (name.includes("normalize")) return ["Transformacoes", "Escala"];
    if (name.includes("derivative") || name.includes("integral")) return ["Transformacoes"];

    // Fluxo / condições
    if (name === "value_in_list") return ["Fluxo", "IN"];
    if (name === "numeric_compare") return ["Fluxo", "CMP"];
    if (name === "merge") return ["Fluxo", "Resolver"];
    if (name === "label") return ["Fluxo", "Tag"];
    if (name.includes("branch") || name.includes("gate") || name.includes("condition")) return ["Fluxo", "Condicao"];
    if (name.endsWith("_gate")) return ["Fluxo", "Condicao"];
    if (name.endsWith("_extractor")) return ["Fluxo"];

    return tags;
  }, []);

  const blockMatchesQuery = useCallback(
    (block, query) => {
      const q = String(query || "").trim().toLowerCase();
      if (!q) return true;
      const displayName = getBlockDisplayName(block?.name || "");
      const hay = `${block?.name || ""} ${displayName} ${block?.description || ""}`.toLowerCase();
      return hay.includes(q);
    },
    [getBlockDisplayName]
  );

  const blockCardCategoryConfig = useMemo(
    () => ({
      data: { color: "var(--block-data)", icon: "D" },
      process: { color: "var(--block-process)", icon: "P" },
      analysis: { color: "var(--block-analysis)", icon: "A" },
      ml: { color: "var(--block-ml)", icon: "ML" },
      flow: { color: "var(--block-flow)", icon: "F" },
      default: { color: "var(--color-gray-400)", icon: "B" },
    }),
    []
  );

  const getBlockHelpModel = useCallback(
    (block) => {
      if (!block) return null;

      const displayName = getBlockDisplayName(block.name);
      const category = getBlockCardCategory(block.name);
      const visual = blockCardCategoryConfig[category] || blockCardCategoryConfig.default;

      const specificKey = `help.blocks.${block.name}`;
      const specific = t(specificKey);
      const hasSpecific = specific && typeof specific === "object" && !Array.isArray(specific);

      const templates = {
        extraction: t("help.templates.extraction"),
        filter: t("help.templates.filter"),
        conversion: t("help.templates.conversion"),
      };

      const whatFallback = block.description || "";

      const model = {
        id: block.name,
        title: displayName,
        subtitle: block.version ? `${block.name} • v${block.version}` : block.name,
        color: visual.color,
        icon: visual.icon,
        what: hasSpecific && typeof specific.what === "string" ? specific.what : whatFallback,
        when: hasSpecific && Array.isArray(specific.when) ? specific.when : [],
        how: hasSpecific && Array.isArray(specific.how) ? specific.how : [],
        tips: hasSpecific && Array.isArray(specific.tips) ? specific.tips : [],
        errors: hasSpecific && Array.isArray(specific.errors) ? specific.errors : [],
      };

      const name = String(block.name || "").toLowerCase();
      if ((!model.what || model.what === whatFallback) && name.endsWith("_extraction")) {
        if (templates.extraction?.what) model.what = templates.extraction.what;
        if (templates.extraction?.how && model.how.length === 0) model.how = [templates.extraction.how];
      }
      if ((!model.what || model.what === whatFallback) && name.endsWith("_filter")) {
        if (templates.filter?.what) model.what = templates.filter.what;
        if (templates.filter?.tips && model.tips.length === 0) model.tips = [templates.filter.tips];
      }
      if ((!model.what || model.what === whatFallback) && name.endsWith("_conversion")) {
        if (templates.conversion?.what) model.what = templates.conversion.what;
        if (templates.conversion?.tips && model.tips.length === 0) model.tips = [templates.conversion.tips];
      }

      // Exemplos rápidos por tipo (UI)
      const examples = [];
      if (block.name === "experiment_fetch") {
        examples.push("[Carregar experimento] -> experiment_data -> [Sensores] -> ...");
      } else if (block.name === "label") {
        examples.push("[Carregar experimento] -> experiment_data -> [Marcador de fluxo] -> [Sensores] -> ...");
      } else if (block.name === "value_in_list") {
        examples.push("[Carregar experimento] -> analysisId -> [IN] -> condition -> [Ramificacao condicional]");
      } else if (block.name === "condition_branch") {
        examples.push("[Dados] + [Condicao] -> [Ramificacao condicional] -> data_if_true / data_if_false");
      } else if (block.name === "merge") {
        examples.push("[Ramificacao condicional] -> data_if_true/data_if_false -> [Resolver fluxo ativo] -> data");
      } else if (block.name === "time_slice") {
        examples.push("[Sensor] -> sensor_data -> [Selecionar janela] -> sensor_data");
      } else if (name.endsWith("_filter")) {
        examples.push("[Pre-processamento] -> sensor_data -> [Filtro] -> sensor_data");
      } else if (name.endsWith("_conversion")) {
        examples.push("[Sensor] -> sensor_data -> [Conversao] -> sensor_data");
      } else if (name.includes("detector")) {
        examples.push("[Sinal] -> sensor_data -> [Detector] -> has_growth");
      } else if (name.includes("feature")) {
        examples.push("[Sinal] -> sensor_data -> [Features] -> features");
      } else if (name.startsWith("response_")) {
        examples.push("[Saidas parciais] -> [Resposta] -> final_response");
      }
      model.examples = examples;

      return model;
    },
    [blockCardCategoryConfig, getBlockCardCategory, getBlockDisplayName, t]
  );

  const activeHelpModel = useMemo(() => {
    if (!helpModal?.block) return null;
    return getBlockHelpModel(helpModal.block);
  }, [helpModal, getBlockHelpModel]);

  const renderBlockCard = useCallback(
    (block) => {
      if (!block) return null;
      const displayName = getBlockDisplayName(block.name);
      const category = getBlockCardCategory(block.name);
      const visual = blockCardCategoryConfig[category] || blockCardCategoryConfig.default;
      const isActive = inspectedBlock?.name === block.name;
      const isPinned = favoriteBlocks.includes(block.name);
      const tags = getBlockCardTags(block.name);

      const handleAdd = (ev) => {
        ev?.preventDefault?.();
        addBlockToCanvas(block);
      };

      const handleHelp = (ev) => {
        ev?.preventDefault?.();
        ev?.stopPropagation?.();
        openHelpModal(block);
      };

      const handleTogglePin = (ev) => {
        ev?.preventDefault?.();
        ev?.stopPropagation?.();
        setFavoriteBlocks((prev) => {
          const set = new Set(prev || []);
          if (set.has(block.name)) set.delete(block.name);
          else set.add(block.name);
          return Array.from(set);
        });
      };

      return (
        <div
          key={block.name}
          className={`block-card${isActive ? " active" : ""}`}
          style={{ "--block-accent": visual.color }}
          draggable
          role="button"
          tabIndex={0}
          title={`${displayName} (${block.name})`}
          onClick={() => setInspectedBlock(block)}
          onDoubleClick={handleAdd}
          onDragStart={(e) => onDragStartBlock(e, block)}
          onKeyDown={(e) => {
            if (e.key === "Enter") return handleAdd(e);
            if (e.key === "?" || e.key === "F1") return handleHelp(e);
            return undefined;
          }}
        >
          <div className="block-card-icon" aria-hidden="true">
            {visual.icon}
          </div>
          <div className="block-card-content">
            <strong>{displayName}</strong>
            {block.description ? <small>{block.description}</small> : null}
            {tags.length ? (
              <div className="block-card-tags">
                {tags.map((tag) => (
                  <span key={tag} className="block-card-tag is-accent">
                    {tag}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
          <button
            className={`block-card-pin${isPinned ? " is-active" : ""}`}
            type="button"
            onClick={handleTogglePin}
            aria-label={isPinned ? t("actions.unpin") : t("actions.pin")}
            title={isPinned ? t("actions.unpin") : t("actions.pin")}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M14 3h-4v2l1 1v6l-3 3v2h8v-2l-3-3V6l1-1V3Z" />
            </svg>
          </button>
          <button
            className="block-card-help"
            type="button"
            onClick={handleHelp}
            aria-label={t("helper.openLabel")}
            title={t("helper.openLabel")}
          >
            ?
          </button>
        </div>
      );
    },
    [
      addBlockToCanvas,
      blockCardCategoryConfig,
      getBlockCardCategory,
      getBlockCardTags,
      getBlockDisplayName,
      favoriteBlocks,
      inspectedBlock,
      onDragStartBlock,
      openHelpModal,
      setInspectedBlock,
      setFavoriteBlocks,
      t,
    ]
  );

  const renderBlockCardMini = useCallback(
    (block) => {
      if (!block) return null;
      const displayName = getBlockDisplayName(block.name);
      const category = getBlockCardCategory(block.name);
      const visual = blockCardCategoryConfig[category] || blockCardCategoryConfig.default;
      const isPinned = favoriteBlocks.includes(block.name);

      const togglePin = (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        setFavoriteBlocks((prev) => {
          const set = new Set(prev || []);
          if (set.has(block.name)) set.delete(block.name);
          else set.add(block.name);
          return Array.from(set);
        });
      };

      return (
        <div
          key={block.name}
          className="block-card mini"
          style={{ "--block-accent": visual.color }}
          draggable
          role="button"
          tabIndex={0}
          title={`${displayName} (${block.name})`}
          onClick={() => setInspectedBlock(block)}
          onDoubleClick={() => addBlockToCanvas(block)}
          onDragStart={(e) => onDragStartBlock(e, block)}
        >
          <div className="block-card-icon" aria-hidden="true">
            {visual.icon}
          </div>
          <div className="block-card-content">
            <strong>{displayName}</strong>
          </div>
          <button
            className={`block-card-pin${isPinned ? " is-active" : ""}`}
            type="button"
            onClick={togglePin}
            aria-label={isPinned ? t("actions.unpin") : t("actions.pin")}
            title={isPinned ? t("actions.unpin") : t("actions.pin")}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M14 3h-4v2l1 1v6l-3 3v2h8v-2l-3-3V6l1-1V3Z" />
            </svg>
          </button>
        </div>
      );
    },
    [
      addBlockToCanvas,
      blockCardCategoryConfig,
      favoriteBlocks,
      getBlockCardCategory,
      getBlockDisplayName,
      onDragStartBlock,
      setFavoriteBlocks,
      setInspectedBlock,
      t,
    ]
  );

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    console.log("[drag-over]");
  }, []);

  const addExperimentFetchPreset = () => {
    const block = (library.blocks || []).find((b) => b.name === "experiment_fetch");
    if (!block) {
      setError("Bloco 'experiment_fetch' não disponível na biblioteca");
      return;
    }
    const preConfig = {
      experimentId: "019b221a-bfa8-705a-9b40-8b30f144ef68",
      analysisId: "68cb3fb380ac865ce0647ea8",
      tenant: "corsan",
      generate_output_graphs: true,
      plot_mode: "all",
      include_legend: true,
      include_labels: true,
      time_from_zero: true,
    };
    addBlockToCanvas(block, preConfig);
  };

  const onDrop = useCallback(
    (event) => {
    event.preventDefault();
      const blockName =
        event.dataTransfer.getData("application/reactflow") ||
        event.dataTransfer.getData("application/reactflow/block");
      if (!blockName || !reactFlowWrapper.current) {
        console.log("[drop] no blockName", blockName);
        return;
      }
      const block = (library.blocks || []).find((item) => item.name === blockName);
      if (!block) {
        console.log("[drop] block not found", blockName);
        return;
      }
      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const instance = reactFlowInstance.current;
      const position = instance?.screenToFlowPosition
        ? instance.screenToFlowPosition({ x: event.clientX, y: event.clientY })
        : instance?.project
          ? instance.project({
              x: event.clientX - reactFlowBounds.left,
              y: event.clientY - reactFlowBounds.top,
            })
          : {
              x: event.clientX - reactFlowBounds.left,
              y: event.clientY - reactFlowBounds.top,
            };
      addBlockToCanvas(block, null, position);
      console.log("[drop] added block", blockName, position);
    },
    [library.blocks, addBlockToCanvas]
  );

  // Handle context menu globally (nodes, edges e seleção no canvas)
  useEffect(() => {
    const handleContextMenu = (event) => {
      if (!reactFlowWrapper.current?.contains(event.target)) return;

      // Check if the target is a node
      const nodeElement = event.target.closest('.react-flow__node');
      if (nodeElement) {
        event.preventDefault();
        const nodeId = nodeElement.getAttribute('data-id');
        const node = nodes.find(n => n.id === nodeId);
        if (node) {
          setContextMenu({
            x: event.clientX,
            y: event.clientY,
            nodeId: node.id,
            nodeIds: null,
            edgeId: null,
            type: 'node'
          });
        }
        return;
      }
      
      // Check if the target is an edge (connection line)
      const edgeElement = event.target.closest('.react-flow__edge');
      if (edgeElement) {
        event.preventDefault();
        const edgeId = edgeElement.getAttribute('data-testid')?.replace('rf__edge-', '') 
                      || edgeElement.getAttribute('data-id');
        // ReactFlow usa data-testid com formato "rf__edge-{id}"
        // Tentar extrair o ID de diferentes formas
        let foundEdge = edges.find(e => e.id === edgeId);
        if (!foundEdge) {
          // Tentar encontrar pelo aria-label ou outras propriedades
          const ariaLabel = edgeElement.getAttribute('aria-label');
          if (ariaLabel) {
            // aria-label formato: "Edge from {source} to {target}"
            const match = ariaLabel.match(/Edge from (.*) to (.*)/);
            if (match) {
              foundEdge = edges.find(e => e.source === match[1] && e.target === match[2]);
            }
          }
        }
        if (foundEdge) {
          setContextMenu({
            x: event.clientX,
            y: event.clientY,
            nodeId: null,
            nodeIds: null,
            edgeId: foundEdge.id,
            type: 'edge'
          });
          return;
        }
      }

      // Clique com botão direito no canvas: se houver seleção, abrir menu para excluir seleção
      const selection = selectedNodesRef.current || [];
      if (selection.length > 0) {
        event.preventDefault();
        setContextMenu({
          x: event.clientX,
          y: event.clientY,
          nodeId: null,
          nodeIds: selection.map((n) => n.id),
          edgeId: null,
          type: 'selection',
        });
      }
    };

    document.addEventListener('contextmenu', handleContextMenu);
    return () => document.removeEventListener('contextmenu', handleContextMenu);
  }, [nodes, edges]);

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // Remover edge por ID (definido antes de deleteEdge que o usa)
  const removeEdge = useCallback((edgeId) => {
    if (!edgeId) return;
    setEdges((eds) => eds.filter((edge) => edge.id !== edgeId));
    if (selectedEdge?.id === edgeId) {
      setSelectedEdge(null);
    }
  }, [setEdges, selectedEdge]);

  // Deletar nó via menu de contexto
  const deleteNode = useCallback(() => {
    if (!contextMenu?.nodeId) return;
    // abrir modal de confirmação
    const isPartOfSelection = selectedNodes.length > 1 && selectedNodes.some((n) => n.id === contextMenu.nodeId);
    const ids = (isPartOfSelection ? selectedNodes.map((n) => n.id) : [contextMenu.nodeId]).map(String).filter(Boolean);
    if (!ids.length) return;
    setConfirmDelete({ open: true, nodeId: ids[0] ?? null, nodeIds: ids });
    closeContextMenu();
  }, [contextMenu, closeContextMenu, selectedNodes]);

  // Deletar edge via menu de contexto
  const deleteEdge = useCallback(() => {
    if (!contextMenu?.edgeId) return;
    removeEdge(contextMenu.edgeId);
    closeContextMenu();
  }, [contextMenu, removeEdge, closeContextMenu]);

  const removeNodes = useCallback((nodeIds) => {
    const ids = Array.isArray(nodeIds) ? nodeIds.filter(Boolean) : [];
    if (!ids.length) return;
    const idSet = new Set(ids);
    setNodes((nds) => nds.filter((node) => !idSet.has(node.id)));
    setEdges((eds) => eds.filter((edge) => !idSet.has(edge.source) && !idSet.has(edge.target)));
    if (selectedNode?.id && idSet.has(selectedNode.id)) {
      setSelectedNode(null);
    }
    setSelectedNodes((current) => current.filter((n) => !idSet.has(n.id)));
    setSelectedEdge(null);
  }, [setNodes, setEdges, selectedNode, setSelectedNode, setSelectedNodes]);

  // Estado para modal de confirmação
  const [confirmDelete, setConfirmDelete] = useState({ open: false, nodeId: null, nodeIds: [] });

  const showConfirmDelete = useCallback((nodeIdOrIds) => {
    if (!nodeIdOrIds) return;
    const ids = Array.isArray(nodeIdOrIds) ? nodeIdOrIds : [nodeIdOrIds];
    const normalized = ids.map(String).filter(Boolean);
    if (!normalized.length) return;
    setConfirmDelete({ open: true, nodeId: normalized[0] ?? null, nodeIds: normalized });
  }, []);

  const handleConfirmDelete = useCallback(() => {
    if (confirmDelete?.nodeIds?.length) {
      removeNodes(confirmDelete.nodeIds);
    }
    setConfirmDelete({ open: false, nodeId: null, nodeIds: [] });
  }, [confirmDelete, removeNodes]);

  const handleCancelDelete = useCallback(() => {
    setConfirmDelete({ open: false, nodeId: null, nodeIds: [] });
  }, []);

  // Handler para selecionar nó (clique esquerdo) - backup do onNodesChange
  const handleSelect = useCallback((event, node) => {
    // O sistema de seleção do ReactFlow já cuida da maioria dos casos via onNodesChange
    // Este handler serve como backup e para casos onde queremos selecionar diretamente
    console.log('Node clicked:', node.id);
    setSelectedNode(node);
    setSelectedEdge(null);
  }, []);

  // Handler para selecionar edge (clique esquerdo na conexão)
  const handleEdgeClick = useCallback((_, edge) => {
    console.log('Edge selected:', edge.id, edge);
    setSelectedEdge(edge);
    setSelectedNode(null); // Limpar seleção de nó quando seleciona edge
  }, []);

  // Handler para clique no canvas (desselecionar tudo)
  const handlePaneClick = useCallback(() => {
    setSelectedEdge(null);
    setSelectedNode(null); // Limpar seleção quando clica no canvas vazio
    setSelectedNodes([]); // Limpar seleção múltipla
  }, []);

  // ============ FUNÇÕES DE ALINHAMENTO ============
  
  // Alinhar nós selecionados à esquerda
  const alignNodesLeft = useCallback(() => {
    if (selectedNodes.length < 2) return;
    const minX = Math.min(...selectedNodes.map(n => n.position.x));
    setNodes(nds => nds.map(n => 
      selectedNodes.find(s => s.id === n.id) 
        ? { ...n, position: { ...n.position, x: minX } }
        : n
    ));
  }, [selectedNodes, setNodes]);

  // Alinhar nós selecionados à direita
  const alignNodesRight = useCallback(() => {
    if (selectedNodes.length < 2) return;
    const maxX = Math.max(...selectedNodes.map(n => n.position.x + (n.measured?.width || 200)));
    setNodes(nds => nds.map(n => {
      if (selectedNodes.find(s => s.id === n.id)) {
        const width = n.measured?.width || 200;
        return { ...n, position: { ...n.position, x: maxX - width } };
      }
      return n;
    }));
  }, [selectedNodes, setNodes]);

  // Alinhar nós selecionados ao topo
  const alignNodesTop = useCallback(() => {
    if (selectedNodes.length < 2) return;
    const minY = Math.min(...selectedNodes.map(n => n.position.y));
    setNodes(nds => nds.map(n => 
      selectedNodes.find(s => s.id === n.id) 
        ? { ...n, position: { ...n.position, y: minY } }
        : n
    ));
  }, [selectedNodes, setNodes]);

  // Alinhar nós selecionados abaixo
  const alignNodesBottom = useCallback(() => {
    if (selectedNodes.length < 2) return;
    const maxY = Math.max(...selectedNodes.map(n => n.position.y + (n.measured?.height || 150)));
    setNodes(nds => nds.map(n => {
      if (selectedNodes.find(s => s.id === n.id)) {
        const height = n.measured?.height || 150;
        return { ...n, position: { ...n.position, y: maxY - height } };
      }
      return n;
    }));
  }, [selectedNodes, setNodes]);

  // Alinhar nós ao centro horizontal
  const alignNodesCenterH = useCallback(() => {
    if (selectedNodes.length < 2) return;
    const centers = selectedNodes.map(n => n.position.x + (n.measured?.width || 200) / 2);
    const avgCenter = centers.reduce((a, b) => a + b, 0) / centers.length;
    setNodes(nds => nds.map(n => {
      if (selectedNodes.find(s => s.id === n.id)) {
        const width = n.measured?.width || 200;
        return { ...n, position: { ...n.position, x: avgCenter - width / 2 } };
      }
      return n;
    }));
  }, [selectedNodes, setNodes]);

  // Alinhar nós ao centro vertical
  const alignNodesCenterV = useCallback(() => {
    if (selectedNodes.length < 2) return;
    const centers = selectedNodes.map(n => n.position.y + (n.measured?.height || 150) / 2);
    const avgCenter = centers.reduce((a, b) => a + b, 0) / centers.length;
    setNodes(nds => nds.map(n => {
      if (selectedNodes.find(s => s.id === n.id)) {
        const height = n.measured?.height || 150;
        return { ...n, position: { ...n.position, y: avgCenter - height / 2 } };
      }
      return n;
    }));
  }, [selectedNodes, setNodes]);

  // Distribuir nós horizontalmente (espaçamento igual)
  const distributeNodesH = useCallback(() => {
    if (selectedNodes.length < 3) return;
    const sorted = [...selectedNodes].sort((a, b) => a.position.x - b.position.x);
    const minX = sorted[0].position.x;
    const maxX = sorted[sorted.length - 1].position.x + (sorted[sorted.length - 1].measured?.width || 200);
    const totalWidth = sorted.reduce((sum, n) => sum + (n.measured?.width || 200), 0);
    const spacing = (maxX - minX - totalWidth) / (sorted.length - 1);
    
    let currentX = minX;
    const newPositions = {};
    sorted.forEach((n, i) => {
      newPositions[n.id] = currentX;
      currentX += (n.measured?.width || 200) + spacing;
    });
    
    setNodes(nds => nds.map(n => 
      newPositions[n.id] !== undefined
        ? { ...n, position: { ...n.position, x: newPositions[n.id] } }
        : n
    ));
  }, [selectedNodes, setNodes]);

  // Distribuir nós verticalmente (espaçamento igual)
  const distributeNodesV = useCallback(() => {
    if (selectedNodes.length < 3) return;
    const sorted = [...selectedNodes].sort((a, b) => a.position.y - b.position.y);
    const minY = sorted[0].position.y;
    const maxY = sorted[sorted.length - 1].position.y + (sorted[sorted.length - 1].measured?.height || 150);
    const totalHeight = sorted.reduce((sum, n) => sum + (n.measured?.height || 150), 0);
    const spacing = (maxY - minY - totalHeight) / (sorted.length - 1);
    
    let currentY = minY;
    const newPositions = {};
    sorted.forEach((n, i) => {
      newPositions[n.id] = currentY;
      currentY += (n.measured?.height || 150) + spacing;
    });
    
    setNodes(nds => nds.map(n => 
      newPositions[n.id] !== undefined
        ? { ...n, position: { ...n.position, y: newPositions[n.id] } }
        : n
    ));
  }, [selectedNodes, setNodes]);

  const autoLayoutNodes = useCallback(() => {
    const selectedIds =
      selectedNodes.length >= 2 ? new Set(selectedNodes.map((n) => n.id)) : null;

    const edgesForLayout = selectedIds
      ? edges.filter((e) => selectedIds.has(e.source) && selectedIds.has(e.target))
      : edges;

    setNodes((currentNodes) => {
      const nodesForLayout = selectedIds
        ? currentNodes.filter((n) => selectedIds.has(n.id))
        : currentNodes;

      if (nodesForLayout.length < 2) return currentNodes;

      const defaultWidth = 300;
      const defaultHeight = 240;
      const colGap = 120;
      const rowGap = 80;
      const laneGap = 180;

      const minX = Math.min(...nodesForLayout.map((n) => n.position.x));
      const minY = Math.min(...nodesForLayout.map((n) => n.position.y));

      const nodeByIdAll = new Map(nodesForLayout.map((n) => [n.id, n]));
      const hasLabelNodes = nodesForLayout.some((n) => n?.data?.blockName === "label");
      const flowLabelById = {};
      nodesForLayout.forEach((n) => {
        flowLabelById[n.id] = nodeFlowMetaById[n.id]?.label || t("flows.none");
      });
      const noneLabel = t("flows.none");
      const distinctLabeledFlows = new Set(Object.values(flowLabelById).filter((l) => l && l !== noneLabel));
      const shouldLaneByFlow = hasLabelNodes && distinctLabeledFlows.size >= 2;

        const layoutSubset = (subsetIds, baseY) => {
          const subsetSet = new Set(subsetIds);
          const nodeById = new Map(subsetIds.map((id) => [id, nodeByIdAll.get(id)]));
          const incoming = new Map();
          const outgoing = new Map();
          const indegree = new Map();

        subsetIds.forEach((id) => {
          incoming.set(id, []);
          outgoing.set(id, []);
          indegree.set(id, 0);
        });

        edgesForLayout.forEach((e) => {
          if (!subsetSet.has(e.source) || !subsetSet.has(e.target)) return;
          incoming.get(e.target).push(e.source);
          outgoing.get(e.source).push(e.target);
          indegree.set(e.target, (indegree.get(e.target) || 0) + 1);
        });

        // Prioridade de blocos para ordenação (menor = mais à esquerda)
        const blockPriority = (id) => {
          const bn = nodeById.get(id)?.data?.blockName || "";
          if (bn === "experiment_fetch") return 0;  // Experimentos primeiro
          if (bn === "label") return 1;             // Labels em seguida
          return 10;                                // Resto depois
        };

        const queue = Array.from(indegree.entries())
          .filter(([, d]) => d === 0)
          .map(([id]) => id)
          .sort((a, b) => {
            const pa = blockPriority(a);
            const pb = blockPriority(b);
            if (pa !== pb) return pa - pb;
            return String(a).localeCompare(String(b));
          });

        const topo = [];
        while (queue.length) {
          const id = queue.shift();
          topo.push(id);
          const next = outgoing.get(id) || [];
          next.forEach((to) => {
            const nextDeg = (indegree.get(to) || 0) - 1;
            indegree.set(to, nextDeg);
            if (nextDeg === 0) {
              queue.push(to);
              queue.sort((a, b) => {
                const pa = blockPriority(a);
                const pb = blockPriority(b);
                if (pa !== pb) return pa - pb;
                return String(a).localeCompare(String(b));
              });
            }
          });
        }

        const topoSet = new Set(topo);
        const remaining = subsetIds
          .filter((id) => !topoSet.has(id))
          .sort((a, b) => {
            const na = nodeById.get(a);
            const nb = nodeById.get(b);
            return (
              na.position.x - nb.position.x ||
              na.position.y - nb.position.y ||
              String(a).localeCompare(String(b))
            );
          });

        const ordered = [...topo, ...remaining];

        // Determinar nível (coluna) de cada nó
        // experiment_fetch SEMPRE no nível 0, outros sem pais no nível 1
        const levelById = {};
        ordered.forEach((id) => {
          const node = nodeById.get(id);
          const bn = node?.data?.blockName || "";
          const parents = incoming.get(id) || [];
          
          if (bn === "experiment_fetch") {
            // Sempre na primeira coluna
            levelById[id] = 0;
          } else if (parents.length === 0) {
            // Outros sem pais: se houver experiment_fetch, ficam na coluna 1
            const hasAnyFetch = ordered.some((oid) => nodeById.get(oid)?.data?.blockName === "experiment_fetch");
            levelById[id] = hasAnyFetch ? 1 : 0;
          } else {
            const maxParent = parents.reduce((acc, p) => {
              const lv = levelById[p] ?? 0;
              return Math.max(acc, lv);
            }, 0);
            levelById[id] = maxParent + 1;
          }
        });

        const byLevel = new Map();
        ordered.forEach((id) => {
          const lv = levelById[id] ?? 0;
          if (!byLevel.has(lv)) byLevel.set(lv, []);
          byLevel.get(lv).push(id);
        });

         const levelEntries = Array.from(byLevel.entries()).sort(([a], [b]) => a - b);
         const levelMaxWidth = new Map();
         levelEntries.forEach(([lv, ids]) => {
           const maxW = Math.max(
             ...ids.map((id) => nodeById.get(id)?.measured?.width || defaultWidth)
           );
           levelMaxWidth.set(lv, maxW);
         });

        const levelX = new Map();
        let cursorX = minX;
        levelEntries.forEach(([lv]) => {
          levelX.set(lv, cursorX);
          cursorX += (levelMaxWidth.get(lv) || defaultWidth) + colGap;
        });

        const pos = new Map();
        let maxBottom = baseY;
          levelEntries.forEach(([lv, ids]) => {
          const sorted = [...ids].sort((a, b) => {
            const na = nodeById.get(a);
            const nb = nodeById.get(b);
            const aBn = na?.data?.blockName || "";
            const bBn = nb?.data?.blockName || "";
            // Prioridade vertical: experiment_fetch > label > outros
            const aPrio = aBn === "experiment_fetch" ? 0 : aBn === "label" ? 1 : 10;
            const bPrio = bBn === "experiment_fetch" ? 0 : bBn === "label" ? 1 : 10;
            if (aPrio !== bPrio) return aPrio - bPrio;
            return na.position.y - nb.position.y || String(a).localeCompare(String(b));
          });

          let cursorY = baseY;
          sorted.forEach((id) => {
            const node = nodeById.get(id);
            const height = node?.measured?.height || defaultHeight;
            pos.set(id, { x: levelX.get(lv), y: cursorY });
            cursorY += height + rowGap;
            maxBottom = Math.max(maxBottom, cursorY);
          });
        });

        return { pos, height: Math.max(0, maxBottom - baseY) };
      };

      const allPositions = new Map();
      if (!shouldLaneByFlow) {
        const subsetIds = nodesForLayout.map((n) => n.id);
        const { pos } = layoutSubset(subsetIds, minY);
        pos.forEach((p, id) => allPositions.set(id, p));
      } else {
        const groups = new Map();
        const sharedIds = [];
        nodesForLayout.forEach((n) => {
          const label = flowLabelById[n.id];
          if (!label || label === noneLabel) {
            sharedIds.push(n.id);
            return;
          }
          if (!groups.has(label)) groups.set(label, []);
          groups.get(label).push(n.id);
        });

        const sharedSet = new Set(sharedIds);
        const labeledSet = new Set(nodesForLayout.map((n) => n.id).filter((id) => !sharedSet.has(id)));

        const laneSortKey = (ids, label) => {
          const minY = Math.min(...ids.map((id) => nodeByIdAll.get(id).position.y));
          const labelNodeId = ids.find((id) => nodeByIdAll.get(id)?.data?.blockName === "label");
          if (!labelNodeId) return { splitterY: Number.POSITIVE_INFINITY, minY, label: String(label) };
          const incomingFromShared = edgesForLayout.filter(
            (e) => e.target === labelNodeId && sharedSet.has(e.source)
          );
          const splitter = incomingFromShared
            .map((e) => e.source)
            .find((sid) => {
              const bn = nodeByIdAll.get(sid)?.data?.blockName;
              return bn === "condition_branch" || bn === "condition_gate";
            });
          const splitterY = splitter ? (nodeByIdAll.get(splitter)?.position?.y ?? minY) : Number.POSITIVE_INFINITY;
          return { splitterY, minY, label: String(label) };
        };

        const entries = Array.from(groups.entries()).sort(([aLabel, aIds], [bLabel, bIds]) => {
          const aKey = laneSortKey(aIds, aLabel);
          const bKey = laneSortKey(bIds, bLabel);
          return (
            aKey.splitterY - bKey.splitterY ||
            aKey.minY - bKey.minY ||
            aKey.label.localeCompare(bKey.label)
          );
        });

        // Alguns nós (ex: response_merge / response_builder final) não recebem inputs DIRETAMENTE de nós rotulados,
        // mas ainda são downstream por estarem conectados a agregadores (response_pack). Para o auto-layout ficar
        // consistente, classificamos como downstream se houver algum ancestral rotulado.
        const incomingByTargetEarly = (() => {
          const map = new Map();
          edgesForLayout.forEach((e) => {
            if (!map.has(e.target)) map.set(e.target, []);
            map.get(e.target).push(e.source);
          });
          return map;
        })();

        const hasUpstreamLabeled = (startId, maxDepth = 10) => {
          const visited = new Set([startId]);
          const queue = [{ id: startId, depth: 0 }];
          while (queue.length) {
            const { id, depth } = queue.shift();
            if (depth >= maxDepth) continue;
            const parents = incomingByTargetEarly.get(id) || [];
            for (const parent of parents) {
              if (visited.has(parent)) continue;
              visited.add(parent);
              if (labeledSet.has(parent)) return true;
              queue.push({ id: parent, depth: depth + 1 });
            }
          }
          return false;
        };

        // Separar experiment_fetch dos outros shared (eles terão tratamento especial)
        const experimentFetchIds = [];
        const otherSharedIds = [];
        
        sharedIds.forEach((id) => {
          const bn = nodeByIdAll.get(id)?.data?.blockName;
          if (bn === "experiment_fetch") {
            experimentFetchIds.push(id);
          } else {
            otherSharedIds.push(id);
          }
        });

        const sharedUpstream = [];
        const sharedDownstream = [];
        
        otherSharedIds.forEach((id) => {
          const hasIncomingFromLabeled = edgesForLayout.some((e) => e.target === id && labeledSet.has(e.source));
          const hasOutgoingToLabeled = edgesForLayout.some((e) => e.source === id && labeledSet.has(e.target));
          const isDownstream = hasIncomingFromLabeled || hasUpstreamLabeled(id);
          if (isDownstream) sharedDownstream.push(id);
          else if (hasOutgoingToLabeled) sharedUpstream.push(id);
          else sharedUpstream.push(id);
        });

        let baseXForFlows = minX;

        const stackVertical = (ids, x, startY) => {
          const sorted = [...ids].sort((a, b) => {
            const na = nodeByIdAll.get(a);
            const nb = nodeByIdAll.get(b);
            return na.position.y - nb.position.y || String(a).localeCompare(String(b));
          });
          let cursorY = startY;
          let bottom = startY;
          sorted.forEach((id) => {
            const n = nodeByIdAll.get(id);
            const h = n?.measured?.height || defaultHeight;
            allPositions.set(id, { x, y: cursorY });
            cursorY += h + rowGap;
            bottom = Math.max(bottom, cursorY);
          });
          return Math.max(0, bottom - startY);
        };

        // Posicionar cadeias de roteamento (ex: value_in_list + condition_gate/branch) alinhadas à altura do(s) fluxos (labels)
        // que elas alimentam, para evitar empilhar tudo no topo.
        const sharedUpstreamSet = new Set(sharedUpstream);
        const sharedById = (id) => sharedUpstreamSet.has(id);

        const findLabelDescendants = (startId, maxDepth = 6) => {
          const visited = new Set([startId]);
          const queue = [{ id: startId, depth: 0 }];
          const labels = [];
          while (queue.length) {
            const { id, depth } = queue.shift();
            if (depth > maxDepth) continue;
            const node = nodeByIdAll.get(id);
            if (node?.data?.blockName === "label") {
              labels.push(id);
              continue;
            }
            const outs = edgesForLayout.filter((e) => e.source === id);
            for (const e of outs) {
              const next = e.target;
              if (visited.has(next)) continue;
              visited.add(next);
              queue.push({ id: next, depth: depth + 1 });
            }
          }
          return labels;
        };

        const routingRoots = sharedUpstream.filter((id) => {
          const bn = nodeByIdAll.get(id)?.data?.blockName;
          if (bn === "condition_branch" || bn === "condition_gate") return true;
          const outgoingToLabeled = edgesForLayout.some((e) => e.source === id && labeledSet.has(e.target));
          return outgoingToLabeled;
        });

        // experimentFetchIds já foi separado antes - usar Set para exclusão em clusters
        const experimentFetchIdsSet = new Set(experimentFetchIds);

        const buildCluster = (rootId) => {
          const cluster = new Set([rootId]);
          const stack = [rootId];
          while (stack.length) {
            const cur = stack.pop();
            const ins = edgesForLayout.filter((e) => e.target === cur && sharedById(e.source));
            for (const e of ins) {
              const parent = e.source;
              // Excluir TODOS os experiment_fetch dos clusters
              if (experimentFetchIdsSet.has(parent)) continue;
              if (!cluster.has(parent)) {
                cluster.add(parent);
                stack.push(parent);
              }
            }
          }
          return Array.from(cluster);
        };

        // Clusters upstream (entrada/roteamento) em 2 passos:
        // 1) calcular template de X/largura do roteamento
        // 2) posicionar lanes (divisórias) e só então alinhar Y dos clusters ao centro dessas lanes.
        const clustersBase = routingRoots.map((rootId) => ({ rootId, clusterIds: buildCluster(rootId) }));

        // Remover clusters duplicados (mesmo conjunto de ids)
        const seen = new Set();
        const uniqueClusters = clustersBase.filter((c) => {
          const key = c.clusterIds.slice().sort().join("|");
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });

        // Separar visualmente a hierarquia:
        // - coluna 1 (mais à esquerda): entrada principal (experiment_fetch)
        // - coluna 2: roteamento (value_in_list / condition_gate / condition_branch etc)
        const inputColumnX = minX;
        const routingColumnX = (() => {
          if (!experimentFetchIds.length) return minX;
          const maxFetchWidth = Math.max(...experimentFetchIds.map((id) => nodeByIdAll.get(id)?.measured?.width || defaultWidth));
          return minX + maxFetchWidth + 220;
        })();

        const clustersWithInput = uniqueClusters;

        const clusterTemplateByRoot = (() => {
          const map = new Map();
          clustersWithInput.forEach((c) => {
            const { pos, height } = layoutSubset(c.clusterIds, 0);
            map.set(c.rootId, { pos, height });
          });
          return map;
        })();

        const templateMaxRight = (() => {
          let maxRight = minX;
          clustersWithInput.forEach((c) => {
            const tpl = clusterTemplateByRoot.get(c.rootId);
            if (!tpl) return;
            tpl.pos.forEach((p, id) => {
              const w = nodeByIdAll.get(id)?.measured?.width || defaultWidth;
              maxRight = Math.max(maxRight, (p.x + (routingColumnX - minX)) + w);
            });
          });

          // Considerar qualquer shared upstream fora de clusters
          const inAnyCluster = new Set();
          clustersWithInput.forEach((c) => c.clusterIds.forEach((id) => inAnyCluster.add(id)));
          sharedUpstream
            .filter((id) => !inAnyCluster.has(id))
            .forEach((id) => {
              const w = nodeByIdAll.get(id)?.measured?.width || defaultWidth;
              maxRight = Math.max(maxRight, routingColumnX + w);
            });

          return maxRight;
        })();

        baseXForFlows = sharedUpstream.length ? templateMaxRight + 160 : minX;

        let laneY = minY;
        entries.forEach(([, ids]) => {
          const labelId = ids.find((id) => nodeByIdAll.get(id)?.data?.blockName === "label");
          const desiredY = labelId ? (nodeByIdAll.get(labelId)?.position?.y ?? laneY) : laneY;
          const laneStartY = Math.max(laneY, desiredY);
          const { pos, height } = layoutSubset(ids, laneStartY);
          pos.forEach((p, id) => allPositions.set(id, { x: p.x - minX + baseXForFlows, y: p.y }));
          laneY = laneStartY + height + laneGap;
        });

        const flowCenterByLabel = (() => {
          const center = new Map();
          entries.forEach(([label, ids]) => {
            let top = Number.POSITIVE_INFINITY;
            let bottom = Number.NEGATIVE_INFINITY;
            ids.forEach((id) => {
              const p = allPositions.get(id) || nodeByIdAll.get(id)?.position;
              if (!p) return;
              const h = nodeByIdAll.get(id)?.measured?.height || defaultHeight;
              top = Math.min(top, p.y);
              bottom = Math.max(bottom, p.y + h);
            });
            if (top !== Number.POSITIVE_INFINITY && bottom !== Number.NEGATIVE_INFINITY) {
              center.set(label, (top + bottom) / 2);
            }
          });
          return center;
        })();

        // Agora posiciona os clusters upstream alinhados ao centro real das lanes (pós layout)
        const desiredCenterForRoot = (rootId) => {
          const labelIds = findLabelDescendants(rootId).filter((id) => labeledSet.has(id));
          const centers = labelIds
            .map((id) => {
              const flowLabel = flowLabelById[id];
              if (flowLabel && flowLabel !== noneLabel && flowCenterByLabel.has(flowLabel)) {
                return flowCenterByLabel.get(flowLabel);
              }
              return null;
            })
            .filter((v) => typeof v === "number");

          if (!centers.length) return nodeByIdAll.get(rootId)?.position?.y ?? minY;
          centers.sort((a, b) => a - b);
          return centers.length === 1 ? centers[0] : (centers[0] + centers[centers.length - 1]) / 2;
        };

        // Posicionar TODOS os experiment_fetch na coluna de entrada (mais à esquerda)
        // Ordenados pela posição Y original, empilhados verticalmente
        const sortedFetchIds = [...experimentFetchIds].sort((a, b) => {
          const na = nodeByIdAll.get(a);
          const nb = nodeByIdAll.get(b);
          return (na?.position?.y ?? 0) - (nb?.position?.y ?? 0);
        });
        
        let fetchCursorY = minY;
        sortedFetchIds.forEach((fetchId) => {
          const fetchH = nodeByIdAll.get(fetchId)?.measured?.height || defaultHeight;
          // Tentar alinhar ao centro da lane que alimenta, mas não sobrepor
          const desiredCenterY = desiredCenterForRoot(fetchId);
          const desiredY = Math.max(fetchCursorY, desiredCenterY - fetchH / 2);
          allPositions.set(fetchId, { x: inputColumnX, y: desiredY });
          fetchCursorY = desiredY + fetchH + rowGap;
        });

        const clustersOrdered = [...clustersWithInput]
          .map((c) => ({ ...c, desiredCenterY: desiredCenterForRoot(c.rootId) }))
          .sort((a, b) => a.desiredCenterY - b.desiredCenterY || String(a.rootId).localeCompare(String(b.rootId)));

        let nextClusterY = minY;
        const placedShared = new Set();
        // Marcar todos os fetch como já posicionados
        experimentFetchIds.forEach((id) => placedShared.add(id));
        
        clustersOrdered.forEach((c) => {
          const tpl = clusterTemplateByRoot.get(c.rootId);
          if (!tpl) return;
          const desiredTop = c.desiredCenterY - tpl.height / 2;
          const startY = Math.max(desiredTop, nextClusterY);
          tpl.pos.forEach((p, id) => {
            allPositions.set(id, { x: p.x + (routingColumnX - minX), y: p.y + startY });
            placedShared.add(id);
          });
          nextClusterY = startY + tpl.height + 40;
        });

        // Qualquer shared que sobrou (não faz parte de clusters) fica empilhado após o roteamento
        // Evita reposicionar nós já colocados (ex: experiment_fetch na coluna de entrada).
        const leftoverShared = sharedUpstream.filter((id) => !placedShared.has(id) && !allPositions.has(id));
        if (leftoverShared.length) {
          stackVertical(leftoverShared, routingColumnX, nextClusterY);
          leftoverShared.forEach((id) => placedShared.add(id));
        }

        // Coluna de saída compartilhada (ex: response_builder) à direita
        const labeledRight = (() => {
          const labeledIds = Array.from(labeledSet);
          if (!labeledIds.length) return baseXForFlows;
          return Math.max(
            ...labeledIds.map((id) => {
              const p = allPositions.get(id) || nodeByIdAll.get(id)?.position || { x: baseXForFlows, y: minY };
              const w = nodeByIdAll.get(id)?.measured?.width || defaultWidth;
              return p.x + w;
            })
          );
        })();
        const outputX = labeledRight + 320;

        // `flowCenterByLabel` calculado acima para alinhar clusters upstream e downstream.

        const incomingByTarget = (() => {
          const map = new Map();
          edgesForLayout.forEach((e) => {
            if (!map.has(e.target)) map.set(e.target, []);
            map.get(e.target).push(e.source);
          });
          return map;
        })();

        const findUpstreamLabeled = (startId, maxDepth = 10) => {
          const labeled = [];
          const visited = new Set([startId]);
          const queue = [{ id: startId, depth: 0 }];
          while (queue.length) {
            const { id, depth } = queue.shift();
            if (depth >= maxDepth) continue;
            const parents = incomingByTarget.get(id) || [];
            for (const parent of parents) {
              if (visited.has(parent)) continue;
              visited.add(parent);
              if (labeledSet.has(parent)) {
                labeled.push(parent);
                continue;
              }
              queue.push({ id: parent, depth: depth + 1 });
            }
          }
          return labeled;
        };

        const stackByDesiredY = (ids, x, startY) => {
          const desiredById = new Map();
          ids.forEach((id) => {
            const direct = (incomingByTarget.get(id) || []).filter((sid) => labeledSet.has(sid));
            const incomingFromLabeled = direct.length ? direct : findUpstreamLabeled(id);

            if (!incomingFromLabeled.length) {
              desiredById.set(id, nodeByIdAll.get(id)?.position?.y ?? startY);
              return;
            }

            const ys = incomingFromLabeled.map((sid) => {
              const label = flowLabelById[sid];
              if (label && label !== noneLabel && flowCenterByLabel.has(label)) {
                return flowCenterByLabel.get(label);
              }
              return allPositions.get(sid)?.y ?? nodeByIdAll.get(sid)?.position?.y ?? startY;
            });
            ys.sort((a, b) => a - b);
            // Para agregadores (ex: response_builder) com múltiplos fluxos,
            // centraliza no meio do intervalo (min..max) para ficar visualmente "no centro" das regiões.
            const desiredCenter = ys.length <= 1 ? ys[0] : (ys[0] + ys[ys.length - 1]) / 2;
            desiredById.set(id, desiredCenter);
          });

          const sorted = [...ids].sort((a, b) => {
            const da = desiredById.get(a) ?? startY;
            const db = desiredById.get(b) ?? startY;
            return da - db || String(a).localeCompare(String(b));
          });

          let cursorY = startY;
          sorted.forEach((id) => {
            const n = nodeByIdAll.get(id);
            const h = n?.measured?.height || defaultHeight;
            const desiredCenter = desiredById.get(id) ?? (cursorY + h / 2);
            let y = desiredCenter - h / 2;
            if (y < cursorY) y = cursorY;
            allPositions.set(id, { x, y });
            cursorY = y + h + rowGap;
          });
        };

        // Tiers de saída para manter hierarquia visual:
        // response_pack (saída do grupo) -> response_merge -> response_builder final (API)
        const isResponsePack = (id) => nodeByIdAll.get(id)?.data?.blockName === "response_pack";
        const isResponseMerge = (id) => nodeByIdAll.get(id)?.data?.blockName === "response_merge";
        const isFinalResponseBuilder = (id) =>
          id === "response_builder" && nodeByIdAll.get(id)?.data?.blockName === "response_builder";

        const packIds = sharedDownstream.filter(isResponsePack);
        const mergeIds = sharedDownstream.filter(isResponseMerge);
        const finalIds = sharedDownstream.filter(isFinalResponseBuilder);
        const otherDownstream = sharedDownstream.filter(
          (id) => !isResponsePack(id) && !isResponseMerge(id) && !isFinalResponseBuilder(id)
        );

        const packX = outputX;
        const mergeX = packX + defaultWidth + 240;
        const finalX = mergeX + defaultWidth + 240;

        if (packIds.length) stackByDesiredY(packIds, packX, minY);
        if (mergeIds.length) stackByDesiredY(mergeIds, mergeX, minY);
        if (finalIds.length) stackByDesiredY(finalIds, finalX, minY);
        if (otherDownstream.length) stackByDesiredY(otherDownstream, packX, minY);
      }

      return currentNodes.map((n) => {
        if (!allPositions.has(n.id)) return n;
        return { ...n, position: allPositions.get(n.id) };
      });
    });

    setTimeout(() => {
      reactFlowInstance.current?.fitView?.({ padding: 0.2, duration: 250 });
    }, 50);
  }, [edges, nodeFlowMetaById, selectedNodes, setNodes, t]);

  // ============ COPIAR E COLAR ============
  
  // Copiar nós selecionados para clipboard
  const copySelectedNodes = useCallback(() => {
    const nodesToCopy = selectedNodes.length > 0 ? selectedNodes : (selectedNode ? [selectedNode] : []);
    if (nodesToCopy.length === 0) return;
    
    // Copiar nós com suas posições relativas
    const nodeIds = new Set(nodesToCopy.map(n => n.id));
    
    // Copiar também as edges entre os nós copiados
    const edgesToCopy = edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
    
    // Calcular offset para posição relativa (baseado no nó mais à esquerda/cima)
    const minX = Math.min(...nodesToCopy.map(n => n.position.x));
    const minY = Math.min(...nodesToCopy.map(n => n.position.y));
    
    const copiedNodes = nodesToCopy.map(n => ({
      ...n,
      relativePosition: {
        x: n.position.x - minX,
        y: n.position.y - minY
      }
    }));
    
    setClipboard({ nodes: copiedNodes, edges: edgesToCopy });
    console.log(`Copiados ${nodesToCopy.length} blocos e ${edgesToCopy.length} conexões`);
  }, [selectedNodes, selectedNode, edges]);

  // Colar nós do clipboard
  const pasteNodes = useCallback(() => {
    if (clipboard.nodes.length === 0) return;
    
    // Calcular centro da viewport para colar
    const center = getViewportCenter();
    
    // Calcular tamanho total dos blocos copiados
    const clipboardWidth = Math.max(...clipboard.nodes.map(n => n.relativePosition.x + 200)) || 200;
    const clipboardHeight = Math.max(...clipboard.nodes.map(n => n.relativePosition.y + 150)) || 150;
    
    // Posição base é o centro da viewport menos metade do tamanho total
    const baseX = center.x - clipboardWidth / 2;
    const baseY = center.y - clipboardHeight / 2;
    
    // Criar mapeamento de IDs antigos para novos
    const idMapping = {};
    const timestamp = Date.now();
    
    // Criar novos nós
    const newNodes = clipboard.nodes.map((n, index) => {
      const newId = `${n.data.blockName}_${timestamp}_${index}`;
      idMapping[n.id] = newId;
      
      return {
        ...n,
        id: newId,
        position: {
          x: baseX + n.relativePosition.x,
          y: baseY + n.relativePosition.y
        },
        selected: false,
        data: {
          ...n.data,
          config: { ...(n.data.config || {}) } // Deep copy do config
        }
      };
    });
    
    // Criar novas edges com IDs atualizados
    const newEdges = clipboard.edges.map((e, index) => ({
      ...e,
      id: `e${timestamp}_${index}`,
      source: idMapping[e.source],
      target: idMapping[e.target]
    }));
    
    // Adicionar ao canvas
    setNodes(nds => [...nds, ...newNodes]);
    setEdges(eds => [...eds, ...newEdges]);
    
    // Selecionar os nós colados
    setSelectedNodes(newNodes);
    setSelectedNode(newNodes.length === 1 ? newNodes[0] : null);
    
    console.log(`Colados ${newNodes.length} blocos e ${newEdges.length} conexões`);
  }, [clipboard, getViewportCenter, setNodes, setEdges]);

  // Duplicar nós selecionados (copiar + colar imediato)
  const duplicateSelectedNodes = useCallback(() => {
    const nodesToDuplicate = selectedNodes.length > 0 ? selectedNodes : (selectedNode ? [selectedNode] : []);
    if (nodesToDuplicate.length === 0) return;
    
    const nodeIds = new Set(nodesToDuplicate.map(n => n.id));
    const edgesToDuplicate = edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
    
    const timestamp = Date.now();
    const idMapping = {};
    
    // Criar novos nós com offset de 50px
    const newNodes = nodesToDuplicate.map((n, index) => {
      const newId = `${n.data.blockName}_${timestamp}_${index}`;
      idMapping[n.id] = newId;
      
      return {
        ...n,
        id: newId,
        position: {
          x: n.position.x + 50,
          y: n.position.y + 50
        },
        selected: false,
        data: {
          ...n.data,
          config: { ...(n.data.config || {}) }
        }
      };
    });
    
    const newEdges = edgesToDuplicate.map((e, index) => ({
      ...e,
      id: `e${timestamp}_${index}`,
      source: idMapping[e.source],
      target: idMapping[e.target]
    }));
    
    setNodes(nds => [...nds, ...newNodes]);
    setEdges(eds => [...eds, ...newEdges]);
    setSelectedNodes(newNodes);
    setSelectedNode(newNodes.length === 1 ? newNodes[0] : null);
    
    console.log(`Duplicados ${newNodes.length} blocos`);
  }, [selectedNodes, selectedNode, edges, setNodes, setEdges]);

  // Atalhos de teclado para copiar/colar
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignorar se estiver em um input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      
      if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        e.preventDefault();
        copySelectedNodes();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
        e.preventDefault();
        pasteNodes();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        duplicateSelectedNodes();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [copySelectedNodes, pasteNodes, duplicateSelectedNodes]);

  const applyConfigToSelectedNode = useCallback(
    (nextConfig) => {
      if (!selectedNode) return;
      setNodes((nds) =>
        nds.map((node) =>
          node.id === selectedNode.id
            ? { ...node, data: { ...node.data, config: nextConfig } }
            : node
        )
      );
      setSelectedNode((prev) =>
        prev && prev.id === selectedNode.id ? { ...prev, data: { ...prev.data, config: nextConfig } } : prev
      );
    },
    [selectedNode, setNodes, setSelectedNode]
  );

  const updateNodeConfigField = useCallback(
    (key, value) => {
      if (!selectedNode) return;
      const currentConfig = selectedNode.data.config || {};
      const nextConfig = { ...currentConfig };
      if (value === undefined) {
        delete nextConfig[key];
      } else {
        nextConfig[key] = value;
      }
      applyConfigToSelectedNode(nextConfig);
      setConfigFieldErrors((prev) => {
        if (!prev[key]) {
          return prev;
        }
        const next = { ...prev };
        delete next[key];
        return next;
      });
    },
    [selectedNode, applyConfigToSelectedNode]
  );

  const setFeaturesMergeInputs = useCallback(
    (nodeId, nextInputs) => {
      if (!nodeId) return;
      const inputs = Array.isArray(nextInputs) ? nextInputs.map(String).filter(Boolean) : [];

      const currentNode = nodesRef.current?.find((n) => n.id === nodeId);
      const previousInputs = (currentNode?.data?.dataInputs || []).map(String);
      const removedInputs = previousInputs.filter((key) => !inputs.includes(key));

      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== nodeId) return n;
          const currentConfig = n.data?.config || {};
          return {
            ...n,
            data: {
              ...n.data,
              dataInputs: inputs,
              config: { ...currentConfig, features_inputs: inputs },
            },
          };
        })
      );
      if (removedInputs.length > 0) {
        setEdges((eds) =>
          eds.filter((edge) => {
            if (edge.target !== nodeId) return true;
            const handle = edge.targetHandle || "";
            const inputName = handle.includes("-in-") ? handle.split("-in-")[1] : handle;
            return !removedInputs.includes(String(inputName));
          })
        );
      }
      setSelectedNode((prev) => {
        if (!prev || prev.id !== nodeId) return prev;
        const currentConfig = prev.data?.config || {};
        return {
          ...prev,
          data: {
            ...prev.data,
            dataInputs: inputs,
            config: { ...currentConfig, features_inputs: inputs },
          },
        };
      });
    },
    [setNodes, setEdges, setSelectedNode]
  );

  const setSequentialInputsCount = useCallback(
    (nodeId, nextCount, { prefix = "input_", max = 8, configKey = "inputs_count" } = {}) => {
      if (!nodeId) return;
      const parsed = Number(nextCount);
      const clamped = Math.max(1, Math.min(max, Number.isFinite(parsed) ? parsed : 1));
      const inputs = Array.from({ length: clamped }, (_, i) => `${prefix}${i + 1}`);

      const currentNode = nodesRef.current?.find((n) => n.id === nodeId);
      const previousInputs = (currentNode?.data?.dataInputs || [])
        .map(String)
        .filter((key) => key.startsWith(prefix));
      const removedInputs = previousInputs.filter((key) => !inputs.includes(key));

      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== nodeId) return n;
          const currentConfig = n.data?.config || {};

          const nextConfig = { ...currentConfig, [configKey]: clamped };
          if (Array.isArray(nextConfig.field_mappings) && prefix === "input_") {
            nextConfig.field_mappings = nextConfig.field_mappings.filter((m) => {
              const idx = Number(m?.input);
              return Number.isFinite(idx) ? idx >= 1 && idx <= clamped : true;
            });
          }

          return {
            ...n,
            data: {
              ...n.data,
              dataInputs: inputs,
              config: nextConfig,
            },
          };
        })
      );

      if (removedInputs.length > 0) {
        setEdges((eds) =>
          eds.filter((edge) => {
            if (edge.target !== nodeId) return true;
            const handle = edge.targetHandle || "";
            const inputName = handle.includes("-in-") ? handle.split("-in-")[1] : handle;
            return !removedInputs.includes(String(inputName));
          })
        );
      }

      setSelectedNode((prev) => {
        if (!prev || prev.id !== nodeId) return prev;
        const currentConfig = prev.data?.config || {};
        const nextConfig = { ...currentConfig, [configKey]: clamped };
        if (Array.isArray(nextConfig.field_mappings) && prefix === "input_") {
          nextConfig.field_mappings = nextConfig.field_mappings.filter((m) => {
            const idx = Number(m?.input);
            return Number.isFinite(idx) ? idx >= 1 && idx <= clamped : true;
          });
        }
        return {
          ...prev,
          data: {
            ...prev.data,
            dataInputs: inputs,
            config: nextConfig,
          },
        };
      });
    },
    [setNodes, setEdges, setSelectedNode]
  );

  const handleJsonFieldBlur = useCallback(
    (key, rawValue, typeLabel) => {
      if (!selectedNode) return;
      const trimmed = rawValue.trim();
      if (!trimmed.length) {
        updateNodeConfigField(key, undefined);
        return;
      }
      try {
        const parsed = JSON.parse(rawValue);
        updateNodeConfigField(key, parsed);
      } catch (err) {
        setConfigFieldErrors((prev) => ({
          ...prev,
          [key]: `JSON invÇ­lido (${typeLabel || "objeto"}): ${err.message}`,
        }));
      }
    },
    [selectedNode, updateNodeConfigField]
  );

  const updateNodeConfig = (value) => {
    // removido: editor JSON não utilizado
  };

  const preparedSteps = useMemo(() => buildPreparedStepsModule(nodes, edges), [nodes, edges]);

  const handleSimulate = async () => {
    setIsRunning(true);
    setError("");
    setSimulation(null);
    try {
      const requestBody = {
        name: pipelineName,
        description: "pipeline criado pela GUI",
        steps: preparedSteps,
        initial_state: {},
      };
      const response = await axios.post(`${API_URL}/pipelines/simulate`, requestBody);
      setSimulation(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsRunning(false);
    }
  };

  // Coletar todos os gráficos da simulação para navegação
  const allGraphs = useMemo(() => collectSimulationGraphs(simulation), [simulation]);

  const openGraphModal = (src, title) => {
    // Encontrar índice do gráfico clicado na lista
    const idx = allGraphs.findIndex(g => g.src === src);
    setGraphIndex(idx >= 0 ? idx : 0);
    setGraphList(allGraphs);
    setGraphModalSrc(src);
    setGraphModalTitle(title || "Gráfico");
    setGraphModalOpen(true);
  };

  const getStepLabel = useCallback(
    (stepId) => {
      const node = nodes.find((n) => n.id === stepId);
      return node?.data?.label || stepId;
    },
    [nodes]
  );

  const getStepFlowLabel = useCallback(
    (stepId) => nodeFlowMetaById[stepId]?.label || t("flows.none"),
    [nodeFlowMetaById, t]
  );

  const getStepFlowColor = useCallback(
    (stepId) => nodeFlowMetaById[stepId]?.color || getFlowColor(getStepFlowLabel(stepId)),
    [getFlowColor, getStepFlowLabel, nodeFlowMetaById]
  );

  const flowLanes = useMemo(() => {
    if (!nodes.length) return [];
    const hasLabelNodes = nodes.some((n) => n?.data?.blockName === "label");
    if (!hasLabelNodes) return [];
    const noneLabel = t("flows.none");

    const defaultWidth = 300;
    const defaultHeight = 240;
    const lanePadX = 70;
    const lanePadY = 50;

    const map = new Map();
    nodes.forEach((n) => {
      const meta = nodeFlowMetaById[n.id];
      const flowLabel = meta?.label || t("flows.none");
      if (flowLabel === noneLabel) return;
      const flowColor = meta?.color || getFlowColor(flowLabel);
      const width = n.measured?.width || defaultWidth;
      const height = n.measured?.height || defaultHeight;
      const minX = n.position.x;
      const minY = n.position.y;
      const maxX = minX + width;
      const maxY = minY + height;

      if (!map.has(flowLabel)) {
        map.set(flowLabel, {
          label: flowLabel,
          color: flowColor,
          minX,
          minY,
          maxX,
          maxY,
          count: 1,
        });
        return;
      }

      const lane = map.get(flowLabel);
      lane.minX = Math.min(lane.minX, minX);
      lane.minY = Math.min(lane.minY, minY);
      lane.maxX = Math.max(lane.maxX, maxX);
      lane.maxY = Math.max(lane.maxY, maxY);
      lane.count += 1;
    });

    const lanes = Array.from(map.values())
      .filter((l) => l.count >= 2)
      .sort((a, b) => a.minY - b.minY);

    if (lanes.length < 2) return [];

    return lanes.map((l) => ({
      ...l,
      left: l.minX - lanePadX,
      top: l.minY - lanePadY,
      width: l.maxX - l.minX + lanePadX * 2,
      height: l.maxY - l.minY + lanePadY * 2,
    }));
  }, [getFlowColor, nodeFlowMetaById, nodes, t]);

  // (overlay de grupos removido: usamos apenas o retângulo do analysisId e o estendemos para cobrir a "Saída do grupo")

  const closeGraphModal = () => {
    setGraphModalOpen(false);
    setGraphModalSrc(null);
    setGraphModalTitle("");
  };

  // Navegação entre gráficos
  const handleNavigate = useCallback((direction) => {
    if (graphList.length === 0) return;
    let newIndex = graphIndex + direction;
    if (newIndex < 0) newIndex = graphList.length - 1;
    if (newIndex >= graphList.length) newIndex = 0;
    
    setGraphIndex(newIndex);
    setGraphModalSrc(graphList[newIndex].src);
    setGraphModalTitle(graphList[newIndex].title);
  }, [graphList, graphIndex]);

  // Navegação por teclado no modal
  useEffect(() => {
    if (!graphModalOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === "ArrowLeft") handleNavigate(-1);
      else if (e.key === "ArrowRight") handleNavigate(1);
      else if (e.key === "Escape") closeGraphModal();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [graphModalOpen, handleNavigate]);

  // Handler de teclado para Delete/Backspace - remover edge ou nó selecionado
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignorar se estiver em campo de input/textarea
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      
      if (e.key === 'Delete' || e.key === 'Backspace') {
        // Priorizar edge selecionada
        if (selectedEdge) {
          e.preventDefault();
          removeEdge(selectedEdge.id);
        } else if (selectedNodes.length > 0) {
          e.preventDefault();
          showConfirmDelete(selectedNodes.map((n) => n.id));
        } else if (selectedNode) {
          e.preventDefault();
          showConfirmDelete(selectedNode.id);
        }
      }
      // Escape para desselecionar
      if (e.key === 'Escape') {
        setSelectedEdge(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedEdge, selectedNode, selectedNodes, removeEdge, showConfirmDelete]);

  const buildPipelineData = useCallback(() => {
    return {
      version: "1.0",
      name: pipelineName,
      savedAt: new Date().toISOString(),
      editor: {
        nodes: nodes.map((node) => ({
          id: node.id,
          type: node.type,
          position: node.position,
          data: {
            label: node.data.label,
            blockName: node.data.blockName,
            config: node.data.config,
            description: node.data.description,
          },
        })),
        edges: edges.map((edge) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle,
          targetHandle: edge.targetHandle,
          animated: edge.animated,
        })),
      },
      execution: {
        name: pipelineName,
        steps: preparedSteps,
        initial_state: {},
      },
    };
  }, [pipelineName, nodes, edges, preparedSteps]);

  const downloadPipelineFile = useCallback(() => {
    const pipelineData = buildPipelineData();
    const file = new Blob([JSON.stringify(pipelineData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${pipelineName}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [buildPipelineData, pipelineName]);

  const saveWorkspacePipeline = useCallback(async () => {
    if (!workspace?.tenant || !workspace?.pipeline) return false;
    const payload = buildPipelineData();
    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      await axios.post(`${API_URL}/pipelines/workspaces/save`, {
        tenant: workspace.tenant,
        pipeline: workspace.pipeline,
        workspace_version: workspace.version || undefined,
        data: payload,
        change_reason: "Edição no editor",
      });
      return true;
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
      return false;
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [workspace, buildPipelineData]);

  // Salvar pipeline: se estiver em um workspace, persiste em resources/<tenant>/pipeline; caso contrário, baixa arquivo
  const savePipeline = useCallback(async () => {
    if (workspace?.tenant && workspace?.pipeline) {
      const ok = await saveWorkspacePipeline();
      if (ok) return;
    }
    downloadPipelineFile();
  }, [workspace, saveWorkspacePipeline, downloadPipelineFile]);

  // Referência para input de arquivo oculto
  const fileInputRef = useRef(null);
  const workspaceLogoFileInputRef = useRef(null);
  const duplicateLogoFileInputRef = useRef(null);

  const loadPipelineFromJson = useCallback(
    (pipelineData, { tenant = null, pipeline = null, version = null } = {}) => {
      try {
        let nodesToLoad, edgesToLoad, nameToLoad;

        if (pipelineData?.editor) {
          nodesToLoad = pipelineData.editor.nodes || [];
          edgesToLoad = pipelineData.editor.edges || [];
          nameToLoad = pipelineData.name;
        } else if (pipelineData?.nodes && pipelineData?.edges) {
          nodesToLoad = pipelineData.nodes;
          edgesToLoad = pipelineData.edges;
          nameToLoad = pipelineData.name;
        } else {
          setError("Arquivo inválido: estrutura de pipeline não encontrada");
          return false;
        }

        const restoredNodes = (nodesToLoad || []).map((savedNode, idx) => {
          const blockDef = library.blocks?.find((b) => b.name === savedNode.data.blockName);
          const blockName = savedNode.data.blockName;
          const savedConfig = savedNode.data.config || {};
          const isResponseBlock = ["response_builder", "response_pack", "response_merge"].includes(blockName);
          // Usar data_inputs do bloco, ou fallback para keys do input_schema
          const defaultDataInputs = (blockDef?.data_inputs && blockDef.data_inputs.length > 0)
            ? blockDef.data_inputs
            : Object.keys(blockDef?.input_schema || {});
          const defaultDataOutputs = (blockDef?.data_outputs && blockDef.data_outputs.length > 0)
            ? blockDef.data_outputs
            : Object.keys(blockDef?.output_schema || {});
          const responseInputsCountRaw = Number(savedConfig.inputs_count);
          const responseInputsCount = Number.isFinite(responseInputsCountRaw)
            ? Math.max(1, Math.min(8, responseInputsCountRaw))
            : null;
          const resolvedResponseInputs = (() => {
            if (!isResponseBlock || !responseInputsCount) return null;
            const fromDef = defaultDataInputs.filter((k) => String(k).startsWith("input_"));
            if (fromDef.length >= responseInputsCount) return fromDef.slice(0, responseInputsCount);
            return Array.from({ length: responseInputsCount }, (_, i) => `input_${i + 1}`);
          })();
          return {
            id: savedNode.id,
            type: savedNode.type || "pipelineNode",
            position: savedNode.position,
            data: {
              label: normalizeLoadedLabel(nodesToLoad.slice(0, idx), blockName, savedNode.data.label),
              blockName,
              config: savedNode.data.config || {},
              description: savedNode.data.description || blockDef?.description || "",
              inputSchema: blockDef?.input_schema || {},
              outputSchema: blockDef?.output_schema || {},
              dataInputs:
                blockName === "features_merge" &&
                Array.isArray(savedNode.data.config?.features_inputs) &&
                savedNode.data.config.features_inputs.length
                  ? savedNode.data.config.features_inputs
                  : resolvedResponseInputs || defaultDataInputs,
              dataOutputs: defaultDataOutputs,
            },
          };
        });

        const nodeIoById = new Map(
          (restoredNodes || []).map((node) => {
            const blockName = node?.data?.blockName || node?.data?.block || "";
            const inputKeys = Array.isArray(node?.data?.dataInputs) ? node.data.dataInputs : [];
            const outputKeys = Array.isArray(node?.data?.dataOutputs) ? node.data.dataOutputs : [];
            const inputHandles = new Set(inputKeys.map((k) => `${blockName}-in-${k}`));
            const outputHandles = new Set(outputKeys.map((k) => `${blockName}-out-${k}`));
            return [node.id, { blockName, inputKeys, outputKeys, inputHandles, outputHandles }];
          })
        );

        const normalizeTargetHandle = (targetHandleRaw, targetNodeId) => {
          const info = nodeIoById.get(targetNodeId);
          if (!info) return targetHandleRaw;

          const targetHandle = String(targetHandleRaw || "");
          if (targetHandle && info.inputHandles.has(targetHandle)) return targetHandle;

          const fallback = () => {
            if (info.inputKeys.length === 1) return `${info.blockName}-in-${info.inputKeys[0]}`;
            if (info.inputKeys.includes("data")) return `${info.blockName}-in-data`;
            if (info.inputKeys.length) return `${info.blockName}-in-${info.inputKeys[0]}`;
            return targetHandleRaw;
          };

          const key = targetHandle.includes("-in-") ? targetHandle.split("-in-")[1] : "";

          // Migrações pontuais (compatibilidade retroativa)
          if (info.blockName === "growth_features" && key === "fit_results") return `${info.blockName}-in-data`;

          return fallback();
        };

        const normalizeSourceHandle = (sourceHandleRaw, sourceNodeId) => {
          const info = nodeIoById.get(sourceNodeId);
          if (!info) return sourceHandleRaw;

          const sourceHandle = String(sourceHandleRaw || "");
          if (sourceHandle && info.outputHandles.has(sourceHandle)) return sourceHandle;

          if (info.outputKeys.length === 1) return `${info.blockName}-out-${info.outputKeys[0]}`;
          if (info.outputKeys.includes("data")) return `${info.blockName}-out-data`;
          if (info.outputKeys.length) return `${info.blockName}-out-${info.outputKeys[0]}`;
          return sourceHandleRaw;
        };

        const restoredEdges = (edgesToLoad || []).map((savedEdge) => {
          const edge = {
            id: savedEdge.id,
            source: savedEdge.source,
            target: savedEdge.target,
            sourceHandle: savedEdge.sourceHandle,
            targetHandle: savedEdge.targetHandle,
            animated: savedEdge.animated ?? true,
          };

          return {
            ...edge,
            sourceHandle: normalizeSourceHandle(edge.sourceHandle, edge.source),
            targetHandle: normalizeTargetHandle(edge.targetHandle, edge.target),
          };
        });

        setNodes(restoredNodes);
        setEdges(restoredEdges);
        setPipelineName(nameToLoad || pipeline || tenant || "pipeline_carregado");
        setSelectedNode(null);
        setSelectedEdge(null);
        setError("");

        if (tenant && pipeline) {
          setWorkspace({ tenant, pipeline, version: version || null });
        } else {
          setWorkspace(null);
        }
        setWorkspaceHomeOpen(false);
        return true;
      } catch (err) {
        setError(`Erro ao carregar pipeline: ${err.message}`);
        return false;
      }
    },
    [library.blocks, normalizeLoadedLabel, setNodes, setEdges]
  );

  const runTraining = async () => {
    if (!workspace?.tenant || !workspace?.pipeline) return;

    const protocolId = String(trainModal.protocolId || "").trim();
    if (!protocolId) {
      setTrainModal((prev) => ({ ...prev, error: "Informe o protocolId para treinamento." }));
      return;
    }

    const experimentIds = parseExperimentIdsText(trainModal.experimentIdsText);
    if (!experimentIds.length) {
      setTrainModal((prev) => ({ ...prev, error: "Informe ao menos um experimentId." }));
      return;
    }

    const paramErrors = [];

    const buildGridValues = (field, row) => {
      if (field.kind === "select") {
        const options = Array.isArray(field.options) ? field.options.map((o) => String(o)) : [];
        const chosen = Array.isArray(row?.choices) ? row.choices.map((o) => String(o)) : [];
        const values = (chosen.length ? chosen : options).map((v) => String(v || "")).filter(Boolean);
        return values.length ? values : null;
      }

      const minV = Number(row?.min);
      const maxV = Number(row?.max);
      const divisionsRaw = parseInt(String(row?.divisions ?? "3"), 10);
      const divisions = Math.max(1, Math.min(25, Number.isFinite(divisionsRaw) ? divisionsRaw : 3));
      if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return null;

      if (divisions <= 1) {
        const only = field.kind === "int" ? Math.round(minV) : minV;
        return [only];
      }

      const step = (maxV - minV) / (divisions - 1);
      const values = Array.from({ length: divisions }, (_, idx) => minV + idx * step);
      if (field.kind === "int") {
        return Array.from(new Set(values.map((v) => Math.round(v))));
      }
      return values.map((v) => Number(v.toFixed(8)));
    };

    const parseFixedValue = (field, row) => {
      if (field.allowNull && row?.isNull) return null;
      const raw = row?.value;
      if (raw === "" || raw === undefined || raw === null) return undefined;
      if (field.kind === "int") {
        const v = parseInt(String(raw), 10);
        return Number.isFinite(v) ? v : undefined;
      }
      if (field.kind === "float") {
        const v = Number(raw);
        return Number.isFinite(v) ? v : undefined;
      }
      if (field.kind === "select") return String(raw || "");
      return String(raw);
    };

    let models = [];
    try {
      models = Object.entries(trainModelsDraft || {}).map(([stepId, spec]) => {
        const enabled = spec?.enabled !== false;
        const algorithms = Array.isArray(spec?.algorithms) && spec.algorithms.length ? spec.algorithms : ["ridge"];

        const paramsByAlgorithm = {};
        const paramGridByAlgorithm = {};

        algorithms.forEach((algoKey) => {
          const algo = String(algoKey || "ridge").trim().toLowerCase() || "ridge";
          const schema = TRAINING_ALGO_PARAM_SCHEMA[algo] || TRAINING_ALGO_PARAM_SCHEMA.ridge;
          const rows = spec?.paramsByAlgorithm?.[algo] || {};

          const fixedParams = {};
          const gridParams = {};

          (schema.fields || []).forEach((field) => {
            const row = rows?.[field.key];
            if (!row) return;
            const mode = row?.mode === "grid" && field.grid ? "grid" : "fixed";

            if (mode === "grid") {
              const values = buildGridValues(field, row);
              if (!values || values.length === 0) {
                paramErrors.push(`${stepId}/${algo}: grade inválida para '${field.key}'`);
                return;
              }
              gridParams[field.key] = values;
              return;
            }

            const val = parseFixedValue(field, row);
            if (val === undefined) return;
            fixedParams[field.key] = val;
          });

          if (Object.keys(fixedParams).length) paramsByAlgorithm[algo] = fixedParams;
          if (Object.keys(gridParams).length) paramGridByAlgorithm[algo] = gridParams;
        });

        const selectionMetric = String(spec?.selectionMetric || "").trim() || undefined;
        const maxTrials = Number(spec?.maxTrials);

        return {
          step_id: stepId,
          enabled,
          algorithm: String(algorithms[0] || "ridge"),
          params: {},
          grid_search: true,
          algorithms,
          params_by_algorithm: Object.keys(paramsByAlgorithm).length ? paramsByAlgorithm : undefined,
          param_grid_by_algorithm: Object.keys(paramGridByAlgorithm).length ? paramGridByAlgorithm : undefined,
          selection_metric: selectionMetric,
          max_trials: Number.isFinite(maxTrials) && maxTrials > 0 ? maxTrials : undefined,
        };
      });
    } catch (e) {
      setTrainModal((prev) => ({ ...prev, error: e?.message || t("training.validation.params") }));
      setTrainModal((prev) => ({ ...prev, error: e?.message || "Parâmetros inválidos. Use JSON válido." }));
      return;
    }

    if (paramErrors.length) {
      setTrainModal((prev) => ({ ...prev, error: `${t("training.validation.params")}\n- ${paramErrors.slice(0, 6).join("\n- ")}` }));
      return;
    }

    setTrainModal((prev) => ({ ...prev, running: true, error: "", result: null }));

    try {
      try {
        localStorage.setItem("pipelineStudio.lastTrainProtocolId", protocolId);
      } catch {}

      // =====================================================================
      // REGRESSÃO MATEMÁTICA
      // =====================================================================
      // Se useRegression está ativo, usa endpoint de regressão ao invés de ML
      // =====================================================================
      if (trainModal.useRegression && models.length > 0) {
        const allResults = [];
        const allErrors = [];
        
        const enabledModels = models.filter((m) => m.enabled !== false);
        
        for (const model of enabledModels) {
          const params = new URLSearchParams({
            tenant: workspace.tenant,
            step_id: model.step_id,
            protocolId,
            regression_type: trainModal.regressionType || "linear",
            auto_select: String(trainModal.regressionAutoSelect),
            polynomial_degree: String(trainModal.polynomialDegree ?? 3),
            apply_to_pipeline: String(trainModal.applyToPipeline),
          });
          if (workspace.version) {
            params.append("version", workspace.version);
          }
          for (const expId of experimentIds) {
            params.append("experimentIds", expId);
          }

          try {
            const pipelineData = buildPipelineData();
            const res = await axios.post(`${API_URL}/training/regression?${params.toString()}`, pipelineData, { timeout: 300000 });
            
            if (res.data.status === "skipped" || !res.data.success) {
              allResults.push({
                step_id: model.step_id,
                status: "skipped",
                reason: res.data.reason,
                n_samples: res.data.n_collected || 0,
              });
            } else {
              allResults.push({
                step_id: model.step_id,
                status: "trained",
                regression_type: res.data.regression_type,
                equation: res.data.equation,
                coefficients: res.data.coefficients,
                metrics: res.data.metrics,
                plot_data: res.data.plot_data,
                comparison: res.data.comparison,
                n_samples: res.data.metrics?.n_samples || res.data.n_collected || 0,
              });
            }
            if (res.data.errors?.length) {
              allErrors.push(...res.data.errors);
            }
          } catch (err) {
            allErrors.push(`${model.step_id}: ${extractErrorMessage(err)}`);
          }
        }

        const trainedCount = allResults.filter((r) => r.status === "trained").length;
        const skippedCount = allResults.filter((r) => r.status === "skipped").length;

        setTrainModal((prev) => ({
          ...prev,
          running: false,
          result: {
            success: trainedCount > 0,
            isRegression: true,
            regressionResults: allResults,
            trained: allResults.filter((r) => r.status === "trained"),
            skipped: allResults.filter((r) => r.status === "skipped").map((r) => ({ step_id: r.step_id, reason: r.reason })),
            errors: allErrors,
            summary: {
              trained: trainedCount,
              skipped: skippedCount,
              errors: allErrors.length,
            },
          },
          error: allErrors.length > 0 && trainedCount === 0 ? allErrors.join("; ") : "",
        }));

        // Recarregar pipeline se aplicado
        if (trainedCount > 0 && trainModal.applyToPipeline) {
          try {
            const pipelineJson = await axios.get(`${API_URL}/pipelines/workspaces/${workspace.tenant}/${workspace.pipeline}`);
            loadPipelineFromJson(pipelineJson.data, { tenant: workspace.tenant, pipeline: workspace.pipeline });
          } catch {}
        }
        return;
      }
      // =====================================================================
      // FIM: REGRESSÃO MATEMÁTICA
      // =====================================================================

      // Se gridSearchManual está ativo, usa o endpoint de grid search que salva todos os candidatos
      if (trainModal.gridSearchManual && models.length > 0) {
        const allResults = [];
        const allErrors = [];
        
        // Filtrar apenas modelos habilitados
        const enabledModels = models.filter((m) => m.enabled !== false);
        
        // Iterar sobre cada modelo habilitado
        for (const model of enabledModels) {
          const params = new URLSearchParams({
            tenant: workspace.tenant,
            step_id: model.step_id,
            protocolId,
            algorithm: model.algorithm || "ridge",
            y_transform: trainModal.yTransform || "log10p",
            selection_metric: String(trainModal.selectionMetric || "rmse"),
            max_trials: String(trainModal.maxTrials ?? 60),
            test_size: String(trainModal.testSize ?? 0.2),
          });
          if (workspace.version) {
            params.append("version", workspace.version);
          }
          for (const algo of (model.algorithms || [])) {
            params.append("algorithms", algo);
          }
          for (const expId of experimentIds) {
            params.append("experimentIds", expId);
          }
          // Param grid
          if (model.param_grid_by_algorithm) {
            params.append("param_grid", JSON.stringify(model.param_grid_by_algorithm));
          }

          try {
            // Timeout longo: grid-search pode processar múltiplos experimentos
            const pipelineData = buildPipelineData();
            const res = await axios.post(`${API_URL}/training/grid-search?${params.toString()}`, pipelineData, { timeout: 600000 }); // 10 minutos
            
            // Verificar se foi pulado por falta de dados
            if (res.data.status === "skipped") {
              allResults.push({
                step_id: model.step_id,
                status: "skipped",
                reason: res.data.reason,
                skipped_reasons: res.data.skipped_reasons || [],
                n_candidates: 0,
                n_samples: res.data.n_samples || 0,
              });
            } else {
              allResults.push({
                step_id: model.step_id,
                status: "trained",
                session_path: res.data.session_path,
                n_candidates: res.data.n_candidates,
                best_index: res.data.best_index,
                candidates: res.data.candidates,
                n_samples: res.data.n_samples,
              });
            }
            if (res.data.errors?.length) {
              allErrors.push(...res.data.errors);
            }
          } catch (err) {
            allErrors.push(`${model.step_id}: ${extractErrorMessage(err)}`);
          }
        }

        const trainedCount = allResults.filter((r) => r.status === "trained").length;
        const skippedCount = allResults.filter((r) => r.status === "skipped").length;

        setTrainModal((prev) => ({
          ...prev,
          running: false,
          result: {
            success: trainedCount > 0 || skippedCount > 0,
            gridSearchResults: allResults,
            trained: allResults.filter((r) => r.status === "trained").map((r) => ({ step_id: r.step_id, n_samples: r.n_samples })),
            skipped: allResults.filter((r) => r.status === "skipped").map((r) => ({ step_id: r.step_id, reason: r.reason })),
            errors: allErrors,
            summary: {
              trained: trainedCount,
              skipped: skippedCount,
              errors: allErrors.length,
            },
          },
          error: allErrors.length > 0 && trainedCount === 0 && skippedCount === 0 ? allErrors.join("; ") : "",
        }));
      } else {
        // Treino normal (automático)
        const body = {
          tenant: workspace.tenant,
          protocolId,
          experimentIds,
          y_transform: trainModal.yTransform || "log10p",
          test_size: Number(trainModal.testSize ?? 0.2),
          random_state: Number(trainModal.randomState ?? 42),
          perm_importance: !!trainModal.permImportance,
          perm_repeats: Number(trainModal.permRepeats ?? 10),
          selection_metric: String(trainModal.selectionMetric || "rmse"),
          max_trials: Number(trainModal.maxTrials ?? 60),
          skip_missing: true,
          apply_to_pipeline: !!trainModal.applyToPipeline,
          change_reason: String(trainModal.changeReason || "").trim() || undefined,
          models,
          version: workspace.version || undefined,
        };

        const res = await axios.post(`${API_URL}/pipelines/train`, body);
        setTrainModal((prev) => ({ ...prev, running: false, result: res.data, error: "" }));

        if (res.data?.version && trainModal.applyToPipeline) {
          const pipelineJson = await axios.get(`${API_URL}/pipelines/workspaces/${workspace.tenant}/${workspace.pipeline}`);
          loadPipelineFromJson(pipelineJson.data, { tenant: workspace.tenant, pipeline: workspace.pipeline, version: res.data.version });
        }
      }
    } catch (err) {
      setTrainModal((prev) => ({ ...prev, running: false, error: extractErrorMessage(err) }));
    }
  };



  const loadPipeline = useCallback(
    (event) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const pipelineData = JSON.parse(e.target.result);
          loadPipelineFromJson(pipelineData);
        } catch (err) {
          setError(`Erro ao carregar pipeline: ${err.message}`);
        }
      };
      reader.readAsText(file);

      event.target.value = "";
    },
    [loadPipelineFromJson]
  );

  const fetchWorkspaces = useCallback(async () => {
    setWorkspaceListLoading(true);
    setWorkspaceError("");
    try {
      const res = await axios.get(`${API_URL}/pipelines/workspaces`);
      setWorkspaceList(res.data?.pipelines || []);
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceListLoading(false);
    }
  }, []);

  useEffect(() => {
    if (workspaceHomeOpen) fetchWorkspaces();
  }, [workspaceHomeOpen, fetchWorkspaces]);

  useEffect(() => {
    if (workspaceHomeOpen) setWorkspaceHomeMode("available");
  }, [workspaceHomeOpen]);

  const selectedWorkspace = useMemo(() => {
    const key = String(selectedWorkspaceKey || "");
    if (!key.includes("/")) return null;
    const [tenant, pipeline] = key.split("/");
    return (workspaceList || []).find((w) => w.tenant === tenant && w.pipeline === pipeline) || { tenant, pipeline };
  }, [selectedWorkspaceKey, workspaceList]);

  useEffect(() => {
    if (!selectedWorkspace) return;
    setWorkspaceMetaDraft({
      title: selectedWorkspace.title || selectedWorkspace.pipeline || "",
      logo: selectedWorkspace.logo || "",
      accent_color: selectedWorkspace.accent_color || "#1e90ff",
    });
  }, [selectedWorkspace]);

  const resolveWorkspaceLogoSrc = useCallback(
    (logo, tenant, pipeline) => resolveWorkspaceLogoSrcModule({ logo, tenant, pipeline, apiUrl: API_URL }),
    []
  );

  const handleSaveWorkspaceAppearance = useCallback(async () => {
    if (!selectedWorkspace?.tenant || !selectedWorkspace?.pipeline) return;
    const tenant = selectedWorkspace.tenant;
    const pipeline = selectedWorkspace.pipeline;
    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      const res = await axios.get(`${API_URL}/pipelines/workspaces/${tenant}/${pipeline}`);
      const data = res.data || {};
      data.workspace = {
        ...(typeof data.workspace === "object" && data.workspace ? data.workspace : {}),
        title: String(workspaceMetaDraft.title || "").trim() || pipeline,
        logo: String(workspaceMetaDraft.logo || "").trim(),
        accent_color: sanitizeColor(workspaceMetaDraft.accent_color),
      };
      await axios.post(`${API_URL}/pipelines/workspaces/save`, { tenant, pipeline, data, change_reason: "Atualização de personalização" });
      await fetchWorkspaces();
      return true;
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
      return false;
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [selectedWorkspace, workspaceMetaDraft, fetchWorkspaces]);

  const handleSaveWorkspaceAppearanceFromModal = useCallback(async () => {
    const ok = await handleSaveWorkspaceAppearance();
    if (ok) setEditModal({ open: false, target: null });
  }, [handleSaveWorkspaceAppearance]);

  const handleUploadWorkspaceLogo = useCallback(
    async (file) => {
      if (!selectedWorkspace?.tenant || !selectedWorkspace?.pipeline || !file) return;
      const tenant = selectedWorkspace.tenant;
      const pipeline = selectedWorkspace.pipeline;
      setWorkspaceActionLoading(true);
      setWorkspaceError("");
      try {
        const form = new FormData();
        form.append("file", file);
        const res = await axios.post(`${API_URL}/pipelines/workspaces/${tenant}/${pipeline}/logo-upload`, form, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        const nextLogo = res.data?.asset_path || res.data?.logo || "";
        const nextAccent = sanitizeColor(res.data?.accent_color);
        setWorkspaceMetaDraft((prev) => ({ ...prev, logo: nextLogo, accent_color: nextAccent }));

        // Persistir automaticamente no JSON do pipeline
        const pipelineJson = await axios.get(`${API_URL}/pipelines/workspaces/${tenant}/${pipeline}`);
        const data = pipelineJson.data || {};
        data.workspace = {
          ...(typeof data.workspace === "object" && data.workspace ? data.workspace : {}),
          title: String(workspaceMetaDraft.title || "").trim() || pipeline,
          logo: nextLogo,
          accent_color: nextAccent,
        };
        await axios.post(`${API_URL}/pipelines/workspaces/save`, { tenant, pipeline, data });
        await fetchWorkspaces();
      } catch (err) {
        setWorkspaceError(err.response?.data?.detail || err.message);
      } finally {
        setWorkspaceActionLoading(false);
      }
    },
    [selectedWorkspace, fetchWorkspaces, sanitizeColor, workspaceMetaDraft.title]
  );

  const handleCreateWorkspace = useCallback(async () => {
    const tenant = String(newTenantName || "").trim();
    if (!tenant) return;
    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      const res = await axios.post(`${API_URL}/pipelines/workspaces/create`, { tenant });
      loadPipelineFromJson(res.data, { tenant, pipeline: tenant });
      setNewTenantName("");
      setSelectedWorkspaceKey("");
      setWorkspaceHomeMode("available");
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [newTenantName, loadPipelineFromJson]);

  const handleLoadWorkspace = useCallback(async () => {
    const key = String(selectedWorkspaceKey || "");
    if (!key.includes("/")) return;
    const [tenant, pipeline] = key.split("/");
    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      const res = await axios.get(`${API_URL}/pipelines/workspaces/${tenant}/${pipeline}`);
      loadPipelineFromJson(res.data, { tenant, pipeline });
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [selectedWorkspaceKey, loadPipelineFromJson]);

  const handleDuplicateWorkspace = useCallback(async () => {
    const source = duplicateModal?.source;
    if (!source?.tenant || !source?.pipeline) return;
    const targetTenant = String(duplicateModal?.tenant || "").trim();
    if (!targetTenant) return;

    const sourceTenant = source.tenant;
    const sourcePipeline = source.pipeline;
    const targetPipeline = targetTenant;

    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      const res = await axios.post(`${API_URL}/pipelines/workspaces/duplicate`, {
        source_tenant: sourceTenant,
        source_pipeline: sourcePipeline,
        target_tenant: targetTenant,
        target_pipeline: targetPipeline,
        target_title: targetTenant,
      });

      let payload = res.data || {};

      if (duplicateModal?.logoFile) {
        const form = new FormData();
        form.append("file", duplicateModal.logoFile);
        const up = await axios.post(`${API_URL}/pipelines/workspaces/${targetTenant}/${targetPipeline}/logo-upload`, form, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        const nextLogo = up.data?.asset_path || up.data?.logo || "";
        const nextAccent = sanitizeColor(up.data?.accent_color);
        payload.workspace = {
          ...(typeof payload.workspace === "object" && payload.workspace ? payload.workspace : {}),
          title: targetTenant,
          logo: nextLogo,
          accent_color: nextAccent,
        };
        await axios.post(`${API_URL}/pipelines/workspaces/save`, { tenant: targetTenant, pipeline: targetPipeline, data: payload });
      }

      await fetchWorkspaces();
      setDuplicateModal({ open: false, source: null, tenant: "", logoFile: null });
      if (duplicateLogoFileInputRef.current) duplicateLogoFileInputRef.current.value = "";

      loadPipelineFromJson(payload, { tenant: targetTenant, pipeline: targetPipeline });
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [duplicateModal, fetchWorkspaces, loadPipelineFromJson, sanitizeColor]);

  const handleDeleteWorkspace = useCallback(async () => {
    const target = deleteModal?.target;
    if (!target?.tenant || !target?.pipeline) return;

    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      await axios.delete(`${API_URL}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}`);
      await fetchWorkspaces();

      const deletedKey = `${target.tenant}/${target.pipeline}`;
      if (selectedWorkspaceKey === deletedKey) setSelectedWorkspaceKey("");
      if (workspace?.tenant === target.tenant && workspace?.pipeline === target.pipeline) setWorkspace(null);

      setDeleteModal({ open: false, target: null });
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [deleteModal, fetchWorkspaces, selectedWorkspaceKey, workspace]);

  const openVersionsModal = useCallback(
    async (target) => {
      if (!target?.tenant || !target?.pipeline) return;
      setVersionActionsModal({ open: false, version: null });
      setVersionLogsModal({ open: false, version: null, query: "" });
      setVersionsModal({ open: true, target, active: "", versions: [], loading: true, reasonDraft: "", page: 0, query: "" });
      setWorkspaceError("");
      try {
        const res = await axios.get(
          `${API_URL}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions`
        );
        setVersionsModal((prev) => ({
          ...prev,
          active: res.data?.active || "",
          versions: Array.isArray(res.data?.versions) ? res.data.versions : [],
          loading: false,
          page: 0,
          query: "",
        }));
      } catch (err) {
        setVersionsModal((prev) => ({ ...prev, loading: false }));
        setWorkspaceError(err.response?.data?.detail || err.message);
      }
    },
    [setVersionsModal]
  );

  const handleCreateNewVersionCopy = useCallback(async (fromVersion, options = {}) => {
    const target = versionsModal?.target;
    if (!target?.tenant || !target?.pipeline) return;
    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      const activate = Boolean(options?.activate);
      await axios.post(
        `${API_URL}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/create-copy`,
        {
          reason: String(versionsModal?.reasonDraft || "").trim(),
          from_version: fromVersion ? String(fromVersion).trim() : "",
          activate,
        }
      );
      await fetchWorkspaces();
      await openVersionsModal(target);
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [versionsModal, fetchWorkspaces, openVersionsModal]);

  const handleCreateNewVersionClean = useCallback(async () => {
    const target = versionsModal?.target;
    if (!target?.tenant || !target?.pipeline) return;
    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      await axios.post(
        `${API_URL}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/create-clean`,
        { reason: String(versionsModal?.reasonDraft || "").trim() }
      );
      await fetchWorkspaces();
      await openVersionsModal(target);
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [versionsModal, fetchWorkspaces, openVersionsModal]);

  const handleActivateVersion = useCallback(
    async (versionId) => {
      const target = versionsModal?.target;
      if (!target?.tenant || !target?.pipeline || !versionId) return;
      setWorkspaceActionLoading(true);
      setWorkspaceError("");
      try {
        await axios.post(
          `${API_URL}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/${encodeURIComponent(versionId)}/activate`,
          { reason: String(versionsModal?.reasonDraft || "").trim() }
        );
        await fetchWorkspaces();
        await openVersionsModal(target);
      } catch (err) {
        setWorkspaceError(err.response?.data?.detail || err.message);
      } finally {
        setWorkspaceActionLoading(false);
      }
    },
    [versionsModal, fetchWorkspaces]
  );

  const handleDeleteVersion = useCallback(async () => {
    const target = versionsModal?.target;
    const version = String(deleteVersionModal?.version || "").trim();
    if (!target?.tenant || !target?.pipeline || !version) return;

    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      await axios.delete(
        `${API_URL}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/${encodeURIComponent(version)}`
      );
      await fetchWorkspaces();
      setDeleteVersionModal({ open: false, version: "" });
      await openVersionsModal(target);
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [versionsModal, deleteVersionModal, fetchWorkspaces, openVersionsModal]);

  const handleRenameVersion = useCallback(async () => {
    const target = versionsModal?.target;
    const version = String(renameVersionModal?.version || "").trim();
    const name = String(renameVersionModal?.name || "").trim();
    if (!target?.tenant || !target?.pipeline || !version || !name) return;

    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      await axios.post(
        `${API_URL}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/${encodeURIComponent(version)}/rename`,
        { name, reason: String(renameVersionModal?.reason || "").trim() }
      );
      setRenameVersionModal({ open: false, version: "", name: "", reason: "" });
      await openVersionsModal(target);
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  }, [versionsModal, renameVersionModal, openVersionsModal]);

  const handleOpenVersionInEditor = useCallback(
    async (versionId) => {
      const target = versionsModal?.target;
      if (!target?.tenant || !target?.pipeline || !versionId) return;

      setWorkspaceActionLoading(true);
      setWorkspaceError("");
      try {
        const res = await axios.get(
          `${API_URL}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/${encodeURIComponent(versionId)}/load`
        );
        loadPipelineFromJson(res.data, { tenant: target.tenant, pipeline: target.pipeline, version: versionId });
        setVersionActionsModal({ open: false, version: null });
        setVersionLogsModal({ open: false, version: null, query: "" });
        setVersionsModal({ open: false, target: null, active: "", versions: [], loading: false, reasonDraft: "", page: 0, query: "" });
      } catch (err) {
        setWorkspaceError(err.response?.data?.detail || err.message);
      } finally {
        setWorkspaceActionLoading(false);
      }
    },
    [versionsModal, loadPipelineFromJson]
  );

  const versionsPageSize = 5;
  const versionsSorted = useMemo(() => {
    const list = Array.isArray(versionsModal?.versions) ? versionsModal.versions : [];
    const q = String(versionsModal?.query || "").trim().toLowerCase();
    const filtered = q
      ? list.filter((v) => {
          const name = String(v?.name || "").toLowerCase();
          const id = String(v?.id || "").toLowerCase();
          return name.includes(q) || id.includes(q);
        })
      : list;
    const getSortKey = (v) => v?.created_at || v?.updated_at || "";
    return [...filtered].sort((a, b) => {
      const activeA = a?.is_active ? 1 : 0;
      const activeB = b?.is_active ? 1 : 0;
      if (activeA !== activeB) return activeB - activeA;
      const byDate = String(getSortKey(b)).localeCompare(String(getSortKey(a)));
      if (byDate !== 0) return byDate;
      return String(a?.id || "").localeCompare(String(b?.id || ""));
    });
  }, [versionsModal?.versions, versionsModal?.query]);
  const versionsTotalPages = useMemo(() => Math.max(1, Math.ceil(versionsSorted.length / versionsPageSize)), [versionsSorted.length]);
  const versionsCurrentPage = Math.min(Math.max(0, versionsModal?.page || 0), versionsTotalPages - 1);
  const versionsPageItems = useMemo(() => {
    const start = versionsCurrentPage * versionsPageSize;
    return versionsSorted.slice(start, start + versionsPageSize);
  }, [versionsSorted, versionsCurrentPage]);
  const versionsPageStart = versionsSorted.length ? versionsCurrentPage * versionsPageSize + 1 : 0;
  const versionsPageEnd = Math.min(versionsSorted.length, (versionsCurrentPage + 1) * versionsPageSize);


  // Trigger para abrir diálogo de arquivo
  const triggerLoadPipeline = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const selectedBlockDefinition = useMemo(
    () => (library.blocks || []).find((block) => block.name === selectedNode?.data?.blockName),
    [library.blocks, selectedNode?.data?.blockName]
  );

  const configInputs = useMemo(() => selectedBlockDefinition?.config_inputs || [], [selectedBlockDefinition]);
  const effectiveConfigInputs = useMemo(() => {
    if (selectedNode?.data?.blockName !== "label") return configInputs;
    if (configInputs.includes("label_color")) return configInputs;
    const idx = configInputs.indexOf("label");
    if (idx >= 0) {
      const withColor = [...configInputs];
      withColor.splice(idx + 1, 0, "label_color");
      return withColor;
    }
    return [...configInputs, "label_color"];
  }, [configInputs, selectedNode?.data?.blockName]);

  // Detectar canais disponíveis do bloco conectado à entrada sensor_data
  const availableChannels = useMemo(() => {
    if (!selectedNode) return [];
    
    // Mapeamento de tipos de bloco para seus canais de saída
    const channelsByBlockType = {
      // Extratores de sensores - canais raw
      turbidimetry_extraction: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
      fluorescence_extraction: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
      nephelometry_extraction: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
      // Conversões de cor (nomes minúsculos como no backend)
      xyy_conversion: ["xyY_x", "xyY_y", "xyY_Y"],
      xyz_conversion: ["XYZ_X", "XYZ_Y", "XYZ_Z"],
      lab_conversion: ["LAB_L", "LAB_A", "LAB_B"],
      hsv_conversion: ["HSV_H", "HSV_S", "HSV_V"],
      hsb_conversion: ["HSB_H", "HSB_S", "HSB_B"],
      rgb_conversion: ["RGB_R", "RGB_G", "RGB_B"],
      cmyk_conversion: ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"],
      // Filtros e processamentos mantêm os canais do anterior (passthrough)
      moving_average_filter: "passthrough",
      lowpass_filter: "passthrough",
      median_filter: "passthrough",
      savgol_filter: "passthrough",
      exponential_filter: "passthrough",
      outlier_filter: "passthrough",
      time_slice: "passthrough",
      condition_branch: "passthrough",
      condition_gate: "passthrough",
      baseline_subtraction: "passthrough",
      zscore_normalizer: "passthrough",
      minmax_normalizer: "passthrough",
      reference_normalizer: "passthrough",
      curve_fit: "passthrough",
      curve_fit_best: "passthrough",
    };
    
    // Função para encontrar canais recursivamente
    const findChannelsFromSource = (nodeId, visited = new Set()) => {
      if (visited.has(nodeId)) return [];
      visited.add(nodeId);

      const node = nodes.find(n => n.id === nodeId);
      if (!node) return [];

      const stepData = simulation?.step_results?.[nodeId] || null;
      const stepSensorData = stepData?.data?.sensor_data || stepData?.sensor_data || null;
      if (stepSensorData) {
        const fromData = stepSensorData?.available_channels;
        if (Array.isArray(fromData) && fromData.length) return fromData.map(String);
        const ch = stepSensorData?.channels;
        if (ch && typeof ch === "object" && !Array.isArray(ch)) return Object.keys(ch);
      }

      const blockType = node.data?.blockName;
      const channelDef = channelsByBlockType[blockType];
      
      if (Array.isArray(channelDef)) {
        return channelDef;
      }
      
      if (channelDef === "passthrough") {
        // Procurar o bloco conectado à entrada sensor_data deste nó
        const inputEdge = edges.find(e => {
          if (e.target !== nodeId) return false;
          const handle = e.targetHandle || "";
          // Handle pode ser "blockName-in-sensor_data" ou conter "sensor_data"
          return handle.includes("sensor_data");
        });
        if (inputEdge) {
          return findChannelsFromSource(inputEdge.source, visited);
        }
        // Fallback: usar qualquer entrada se não existir sensor_data
        const anyEdge = edges.find(e => e.target === nodeId);
        if (anyEdge) {
          return findChannelsFromSource(anyEdge.source, visited);
        }
      }
      
      return [];
    };
    
    // Encontrar todas as edges conectadas ao nó selecionado
    const incomingEdges = edges.filter(e => e.target === selectedNode.id);
    
    // Procurar edge conectado à entrada sensor_data, fitted_data ou data
    const inputEdge = incomingEdges.find(e => {
      const handle = e.targetHandle || "";
      return handle.includes("sensor_data") || handle.includes("fitted_data") || handle.includes("-in-data");
    });
    
    if (!inputEdge) {
      // Fallback: usar a primeira edge se não encontrar sensor_data/fitted_data específico
      if (incomingEdges.length > 0) {
        return findChannelsFromSource(incomingEdges[0].source);
      }
      return [];
    }
    
    return findChannelsFromSource(inputEdge.source);
  }, [selectedNode, nodes, edges, simulation]);

  // Detectar canais disponíveis do bloco conectado à entrada sensor_data/fit_results/data (para growth_features e outros)
  const availableFitChannels = useMemo(() => {
    if (!selectedNode) return [];
    
    // Encontrar QUALQUER edge de entrada (não apenas sensor_data)
    const incomingEdges = edges.filter(e => e.target === selectedNode.id);
    if (incomingEdges.length === 0) return [];
    
    // Priorizar entrada sensor_data, fit_results ou data, mas aceitar qualquer uma
    const inputEdge = incomingEdges.find(e => {
      const handle = e.targetHandle || "";
      return handle.includes("sensor_data") || handle.includes("fit_results") || handle.includes("-in-data");
    }) || incomingEdges[0];
    
    const sourceHandle = inputEdge.sourceHandle || "";
    const sourceNodeId = inputEdge.source;
    const sourceNode = nodes.find(n => n.id === sourceNodeId);
    if (!sourceNode) return [];
    
    // =================================================================
    // MÉTODO 1: Buscar nos dados de simulação (mais preciso)
    // =================================================================
    if (simulation?.step_outputs) {
      const stepKey = `step_${sourceNode.id}`;
      const stepData = simulation.step_outputs[stepKey];
      
      if (stepData) {
        // Extrair chave do output do handle (ex: moving_average-out-sensor_data -> sensor_data)
        const outputKey = sourceHandle.includes("-out-") 
          ? sourceHandle.split("-out-")[1] 
          : null;
        
        // Tentar output específico primeiro, depois qualquer um com channels
        const outputsToCheck = outputKey 
          ? [stepData[outputKey], ...Object.values(stepData)]
          : Object.values(stepData);
        
        for (const outputData of outputsToCheck) {
          if (outputData?.channels && typeof outputData.channels === "object") {
            const channelKeys = Object.keys(outputData.channels);
            if (channelKeys.length > 0) return channelKeys;
          }
        }
      }
    }
    
    // =================================================================
    // MÉTODO 2: Fallback - inferir canais pelo tipo de bloco
    // =================================================================
    const findChannelsRecursively = (nodeId, visited = new Set()) => {
      if (visited.has(nodeId)) return [];
      visited.add(nodeId);
      
      const node = nodes.find(n => n.id === nodeId);
      if (!node) return [];
      
      const blockType = node.data?.blockName;
      
      // Mapa de canais conhecidos por tipo de bloco
      const channelsByBlockType = {
        turbidimetry_extraction: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        fluorescence_extraction: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        nephelometry_extraction: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        xyy_conversion: ["xyY_x", "xyY_y", "xyY_Y"],
        xyz_conversion: ["XYZ_X", "XYZ_Y", "XYZ_Z"],
        lab_conversion: ["LAB_L", "LAB_A", "LAB_B"],
        hsv_conversion: ["HSV_H", "HSV_S", "HSV_V"],
        hsb_conversion: ["HSB_H", "HSB_S", "HSB_B"],
        rgb_conversion: ["RGB_R", "RGB_G", "RGB_B"],
        cmyk_conversion: ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"],
      };
      
      // Se temos canais conhecidos para este bloco, retornar
      if (channelsByBlockType[blockType]) {
        return channelsByBlockType[blockType];
      }
      
      // Se for um bloco que apenas passa dados (passthrough), buscar no bloco anterior
      const passthroughBlocks = [
        "moving_average", "savgol_filter", "kalman_filter", "low_pass_filter",
        "high_pass_filter", "band_pass_filter", "derivative", "normalize",
        "temporal_slice", "temporal_cut", "resample", "interpolate",
        "curve_fit", "curve_fit_best", "baseline_correction", "detrend"
      ];
      
      if (passthroughBlocks.includes(blockType) || !channelsByBlockType[blockType]) {
        // Encontrar entrada sensor_data ou data deste nó
        const sensorEdge = edges.find(e => {
          if (e.target !== nodeId) return false;
          const handle = e.targetHandle || "";
          return handle.includes("sensor_data") || handle.includes("-in-data");
        });
        
        if (sensorEdge) {
          return findChannelsRecursively(sensorEdge.source, visited);
        }
      }
      
      return [];
    };
    
    return findChannelsRecursively(sourceNodeId);
  }, [selectedNode, nodes, edges, simulation]);

  const renderConfigField = (field) => {
    if (!selectedNode) return null;
    const schema = selectedBlockDefinition?.input_schema?.[field] || {};
    const normalizedType = (schema.type || "").toLowerCase();
    const value = selectedNode.data.config?.[field];
    const description = schema.description;
    const inputId = `${selectedNode.id}-${field}`;
    const fieldError = configFieldErrors[field];

    if (field === "label_color") {
      const safeValue =
        typeof value === "string" && /^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(value.trim())
          ? value.trim()
          : "#6366f1";

      return (
        <div className="config-field" key={field}>
          <label className="config-label" htmlFor={inputId}>
            {t("configuration.labelColor")}
          </label>
          <div className="config-color-row">
            <input
              id={inputId}
              type="color"
              className="config-color-input"
              value={safeValue}
              onChange={(e) => updateNodeConfigField(field, e.target.value)}
            />
            <input
              type="text"
              className="config-input config-color-text"
              value={typeof value === "string" ? value : ""}
              placeholder="#RRGGBB"
              onChange={(e) => updateNodeConfigField(field, e.target.value)}
            />
            <button
              type="button"
              className="config-color-clear"
              onClick={() => updateNodeConfigField(field, undefined)}
            >
              {t("actions.clear")}
            </button>
          </div>
          <small className="config-hint-small">{t("configuration.labelColorHint")}</small>
        </div>
      );
    }

    // Special-case: chip buttons for plot_channel
    if (field === "plot_channel") {
      const options = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"];
      const selectedValues = Array.isArray(value) ? value : value ? [value] : options.slice();
      const allSelected = selectedValues.length === options.length;
      const toggleOption = (opt) => {
        const newVals = selectedValues.includes(opt)
          ? selectedValues.filter((v) => v !== opt)
          : [...selectedValues, opt];
        updateNodeConfigField(field, newVals.length === 0 ? undefined : newVals);
      };
      const toggleAll = () => {
        updateNodeConfigField(field, allSelected ? [] : options.slice());
      };
      return (
        <div className="config-field" key={field}>
          <label>{t("common.channels")}</label>
          {description && <small>{description}</small>}
          <div className="chip-group">
            <button
              type="button"
              className={`chip chip-all ${allSelected ? "active" : ""}`}
              onClick={toggleAll}
            >
              {t("common.all")}
            </button>
            {options.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`chip ${selectedValues.includes(opt) ? "active" : ""}`}
                onClick={() => toggleOption(opt)}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>
      );
    }

    // Special-case: hierarchical sensor/channel selection for graph_config
    if (field === "graph_config") {
      // Definição de todos os sensores e seus canais
      const sensorDefinitions = {
        turbidimetry: {
          title: "Turbidimetria",
          channels: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        },
        nephelometry: {
          title: "Nefelometria",
          channels: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        },
        fluorescence: {
          title: "Fluorescência",
          channels: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        },
        temperatures: {
          title: "Temperaturas",
          channels: ["sample", "core", "ambient", "coreHeatExchanger", "sampleHeatExchanger", "heatsinkUpper", "heatsinkLower", "magneticStirrer"],
        },
        powerSupply: {
          title: "Fonte de Alimentação",
          channels: ["voltage", "current"],
        },
        peltierCurrents: {
          title: "Correntes Peltier",
          channels: ["heatExchanger", "sampleChamber"],
        },
        nemaCurrents: {
          title: "Correntes NEMA",
          channels: ["coilA", "coilB"],
        },
        ressonantFrequencies: {
          title: "Freq. Ressonantes",
          channels: ["channel0", "channel1"],
        },
        controlState: {
          title: "Estado de Controle",
          channels: ["sampleTempError", "sampleTempU", "coreTempError", "coreTempU", "heatExchangerU", "heatsinkUpperU", "heatsinkLowerU"],
        },
      };

      // Default: apenas sensores espectrais com todos os canais
      const defaultConfig = {
        turbidimetry: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        nephelometry: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
        fluorescence: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"],
      };

      // Valor atual: se não definido, começa vazio (usuário precisa selecionar)
      // Se já existe um valor (objeto), usa ele
      const currentConfig = value !== undefined && value !== null && typeof value === "object" 
        ? value 
        : {};  // Começa vazio - usuário seleciona o que quer

      // Verificar se um sensor está ativo (tem pelo menos um canal)
      const isSensorActive = (sensor) => {
        return currentConfig[sensor] && currentConfig[sensor].length > 0;
      };

      // Toggle sensor inteiro (ativa/desativa todos os canais)
      const toggleSensor = (sensor) => {
        const sensorDef = sensorDefinitions[sensor];
        const newConfig = { ...currentConfig };
        
        if (isSensorActive(sensor)) {
          // Desativar: remover do config
          delete newConfig[sensor];
        } else {
          // Ativar: adicionar todos os canais
          newConfig[sensor] = [...sensorDef.channels];
        }
        
        updateNodeConfigField(field, newConfig);
      };

      // Toggle canal individual
      const toggleChannel = (sensor, channel) => {
        const sensorDef = sensorDefinitions[sensor];
        const newConfig = { ...currentConfig };
        const currentChannels = newConfig[sensor] || [];
        
        if (currentChannels.includes(channel)) {
          // Remover canal
          const newChannels = currentChannels.filter(c => c !== channel);
          if (newChannels.length === 0) {
            delete newConfig[sensor];
          } else {
            newConfig[sensor] = newChannels;
          }
        } else {
          // Adicionar canal
          newConfig[sensor] = [...currentChannels, channel];
        }
        
        updateNodeConfigField(field, newConfig);
      };

      // Toggle todos os canais de um sensor
      const toggleAllChannels = (sensor) => {
        const sensorDef = sensorDefinitions[sensor];
        const newConfig = { ...currentConfig };
        const currentChannels = newConfig[sensor] || [];
        const allSelected = currentChannels.length === sensorDef.channels.length;
        
        if (allSelected) {
          delete newConfig[sensor];
        } else {
          newConfig[sensor] = [...sensorDef.channels];
        }
        
        updateNodeConfigField(field, newConfig);
      };

      return (
        <div className="config-field graph-config-field" key={field}>
          {Object.entries(sensorDefinitions).map(([sensorKey, sensorDef]) => {
            const isActive = isSensorActive(sensorKey);
            const selectedChannels = currentConfig[sensorKey] || [];
            const allChannelsSelected = selectedChannels.length === sensorDef.channels.length;
            
            return (
                <div key={sensorKey} className={`sensor-group ${isActive ? "active" : ""}`}>
                  {/* Header do sensor com toggle */}
                  <div className="sensor-header" onClick={() => toggleSensor(sensorKey)}>
                  <span className="sensor-checkbox" aria-hidden="true" />
                    <span className="sensor-title">{sensorDef.title}</span>
                    {isActive && (
                      <span className="sensor-count">{selectedChannels.length}/{sensorDef.channels.length}</span>
                    )}
                  </div>
                
                {/* Canais (apenas se sensor ativo) */}
                {isActive && (
                  <div className="sensor-channels">
                    <button
                      type="button"
                      className={`chip chip-all ${allChannelsSelected ? "active" : ""}`}
                      onClick={(e) => { e.stopPropagation(); toggleAllChannels(sensorKey); }}
                    >
                      {t("common.all")}
                    </button>
                    {sensorDef.channels.map((channel) => (
                      <button
                        key={channel}
                        type="button"
                        className={`chip ${selectedChannels.includes(channel) ? "active" : ""}`}
                        onClick={(e) => { e.stopPropagation(); toggleChannel(sensorKey, channel); }}
                      >
                        {channel}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      );
    }

    if (normalizedType === "bool" || normalizedType === "boolean") {
      return (
        <div className="config-field" key={field}>
          <label htmlFor={inputId}>{field}</label>
          {description && <small>{description}</small>}
          <label className="config-toggle">
            <input
              id={inputId}
              type="checkbox"
              checked={Boolean(value)}
              onChange={(event) => updateNodeConfigField(field, event.target.checked)}
            />
            <span>{Boolean(value) ? "Ativado" : "Desativado"}</span>
          </label>
        </div>
      );
    }

    if (["int", "float", "number"].includes(normalizedType)) {
      return (
        <div className="config-field" key={field}>
          <label htmlFor={inputId}>{field}</label>
          {description && <small>{description}</small>}
          <input
            id={inputId}
            type="number"
            value={value ?? ""}
            onChange={(event) => {
              if (event.target.value === "") {
                updateNodeConfigField(field, undefined);
                return;
              }
              const parsed = Number(event.target.value);
              if (Number.isNaN(parsed)) {
                return;
              }
              updateNodeConfigField(field, parsed);
            }}
          />
        </div>
      );
    }

    if (["dict", "list", "object", "array"].includes(normalizedType)) {
      const stringValue = value !== undefined ? JSON.stringify(value, null, 2) : "";
      return (
        <div className="config-field" key={field}>
          <label htmlFor={inputId}>{field}</label>
          {description && <small>{description}</small>}
          <textarea
            id={inputId}
            key={`${selectedNode.id}-${field}-${stringValue}`}
            defaultValue={stringValue}
            onBlur={(event) => handleJsonFieldBlur(field, event.target.value, normalizedType)}
            placeholder="Cole um objeto JSON válido"
          />
          <small className="config-hint">
            Use JSON para estruturar {normalizedType === "list" ? "listas" : "objetos"}.
          </small>
          {fieldError && <span className="field-error">{fieldError}</span>}
        </div>
      );
    }

    return (
      <div className="config-field" key={field}>
        <label htmlFor={inputId}>{field}</label>
        {description && <small>{description}</small>}
        <input
          id={inputId}
          type="text"
          value={value ?? ""}
          placeholder={schema.placeholder || "Informe um valor"}
          onChange={(event) => updateNodeConfigField(field, event.target.value)}
          key={`${selectedNode.id}-${field}-${value ?? ''}`}
        />
      </div>
    );
  };

  return (
    <div className="app-container">
      {workspaceHomeOpen && (
        <div className="workspace-home">
          <div className="workspace-home-bg" aria-hidden="true" />
          <div className="workspace-home-card" role="dialog" aria-modal="true">
            <div className="workspace-home-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.kicker")}</div>
                <h2 className="workspace-home-title">{t("workspace.title")}</h2>
                <div className="workspace-home-subtitle">{t("workspace.subtitle")}</div>
              </div>
              {workspace?.tenant && (
                <button className="workspace-home-close" type="button" onClick={() => setWorkspaceHomeOpen(false)}>
                  {t("actions.close")}
                </button>
              )}
            </div>

            <div className="workspace-home-grid">
              <div className="workspace-panel">
                <div className="workspace-panel-title">{workspaceHomeMode === "create" ? t("workspace.createTitle") : t("workspace.loadTitle")}</div>
                <small className="workspace-muted">{workspaceHomeMode === "create" ? t("workspace.createSubtitle") : t("workspace.availableSubtitle")}</small>

                {workspaceHomeMode === "create" ? (
                  <>
                    <div className="workspace-field workspace-create-field">
                      <label>{t("workspace.tenantLabel")}</label>
                      <input
                        value={newTenantName}
                        onChange={(e) => setNewTenantName(e.target.value)}
                        placeholder={t("workspace.tenantPlaceholder")}
                        disabled={workspaceActionLoading}
                      />
                    </div>
                    <button
                      className="workspace-primary"
                      type="button"
                      disabled={workspaceActionLoading || !String(newTenantName || "").trim()}
                      onClick={handleCreateWorkspace}
                    >
                      {workspaceActionLoading ? t("workspace.creating") : t("workspace.create")}
                    </button>
                    <div
                      className="workspace-mode-link workspace-back-action"
                      role="button"
                      tabIndex={0}
                      onClick={() => setWorkspaceHomeMode("available")}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") setWorkspaceHomeMode("available");
                      }}
                    >
                      {t("workspace.backToAvailable")}
                    </div>
                  </>
                ) : (
                  <>
                    <small className="workspace-muted">{t("workspace.pickHint")}</small>

                    <div className="workspace-pipeline-grid" role="list">
                      {workspaceList.map((w) => {
                        const key = `${w.tenant}/${w.pipeline}`;
                        const tenantId = String(w.tenant || "").trim();
                        const pipelineId = String(w.pipeline || "").trim();
                        const subtitle =
                          pipelineId && tenantId && pipelineId !== tenantId
                            ? `${t("workspace.tenantLabel")}: ${tenantId} / ${t("workspace.pipelineIdLabel")}: ${pipelineId}`
                            : `${t("workspace.tenantLabel")}: ${tenantId || pipelineId}`;
                        return (
                          <button
                            key={key}
                            type="button"
                            role="listitem"
                            className={`workspace-pipeline-card ${selectedWorkspaceKey === key ? "selected" : ""} ${workspaceCardMenuKey === key ? "menu-open" : ""}`}
                            onClick={() => {
                              setSelectedWorkspaceKey(key);
                              setWorkspaceCardMenuKey("");
                            }}
                            title={`${w.tenant} / ${w.pipeline}`}
                          >
                            <div className="workspace-pipeline-card-header">
                              <div className="workspace-pipeline-identity">
                                <div className="workspace-pipeline-logo">
                                  {w.logo ? (
                                    <img src={resolveWorkspaceLogoSrc(w.logo, w.tenant, w.pipeline)} alt={w.title || w.pipeline} />
                                  ) : (
                                    <span className="workspace-pipeline-logo-fallback">{workspaceInitials(w)}</span>
                                  )}
                                </div>
                                <div className="workspace-pipeline-main">
                                  <div className="workspace-pipeline-title">{w.title || w.pipeline}</div>
                                  <div className="workspace-pipeline-sub">{String(subtitle || "").replace(/[^\u0020-\u007E]+/g, " | ")}</div>
                                </div>
                              </div>
                              <div className="workspace-pipeline-actions" onMouseDown={(e) => e.stopPropagation()}>
                                <button
                                  type="button"
                                  className="workspace-pipeline-action-btn"
                                  title={t("workspace.editAction")}
                                  onClick={(e) => {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    setWorkspaceCardMenuKey("");
                                    setSelectedWorkspaceKey(key);
                                    setWorkspaceMetaDraft((prev) => ({
                                      ...prev,
                                      title: w.title || w.pipeline || "",
                                      logo: w.logo || "",
                                      accent_color: w.accent_color || prev.accent_color,
                                    }));
                                    setEditModal({
                                      open: true,
                                      target: { tenant: w.tenant, pipeline: w.pipeline, title: w.title || w.pipeline },
                                    });
                                  }}
                                >
                                  <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                                    <path
                                      fill="currentColor"
                                      d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25Zm2.92 2.33H5v-.92l8.06-8.06.92.92L5.92 19.58ZM20.71 7.04a1.003 1.003 0 0 0 0-1.42L18.37 3.29a1.003 1.003 0 0 0-1.42 0L15.13 5.1l3.75 3.75 1.83-1.81Z"
                                    />
                                  </svg>
                                </button>
                                <button
                                  type="button"
                                  className="workspace-pipeline-action-btn"
                                  aria-haspopup="menu"
                                  aria-expanded={workspaceCardMenuKey === key}
                                  onClick={(e) => {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    setSelectedWorkspaceKey(key);
                                    setWorkspaceCardMenuKey((prev) => (prev === key ? "" : key));
                                  }}
                                >
                                  <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                                    <path fill="currentColor" d="M6 10a2 2 0 1 0 0 4a2 2 0 0 0 0-4Zm6 0a2 2 0 1 0 0 4a2 2 0 0 0 0-4Zm6 0a2 2 0 1 0 0 4a2 2 0 0 0 0-4Z" />
                                  </svg>
                                </button>

                                {workspaceCardMenuKey === key && (
                                  <div className="workspace-pipeline-menu" role="menu" aria-label={t("workspace.cardMenuLabel")}>
                                    <button
                                      type="button"
                                      className="workspace-pipeline-menu-item"
                                      role="menuitem"
                                      onClick={(e) => {
                                        e.preventDefault();
                                        e.stopPropagation();
                                        setWorkspaceCardMenuKey("");
                                        openVersionsModal({ tenant: w.tenant, pipeline: w.pipeline, title: w.title || w.pipeline });
                                      }}
                                    >
                                      {t("workspace.versionsAction")}
                                    </button>
                                    <button
                                      type="button"
                                      className="workspace-pipeline-menu-item"
                                      role="menuitem"
                                      onClick={(e) => {
                                        e.preventDefault();
                                        e.stopPropagation();
                                        setWorkspaceCardMenuKey("");
                                        setDuplicateModal({
                                          open: true,
                                          source: { tenant: w.tenant, pipeline: w.pipeline, title: w.title || w.pipeline },
                                          tenant: "",
                                          logoFile: null,
                                        });
                                        if (duplicateLogoFileInputRef.current) duplicateLogoFileInputRef.current.value = "";
                                      }}
                                    >
                                      {t("workspace.duplicateAction")}
                                    </button>
                                    <button
                                      type="button"
                                      className="workspace-pipeline-menu-item danger"
                                      role="menuitem"
                                      onClick={(e) => {
                                        e.preventDefault();
                                        e.stopPropagation();
                                        setWorkspaceCardMenuKey("");
                                        setDeleteModal({
                                          open: true,
                                          target: { tenant: w.tenant, pipeline: w.pipeline, title: w.title || w.pipeline },
                                        });
                                      }}
                                    >
                                      {t("workspace.deleteAction")}
                                    </button>
                                  </div>
                                )}
                              </div>
                            </div>
                            <div className="workspace-pipeline-meta">
                              <div className="workspace-pipeline-tags">
                                <span className="workspace-pill">{w.active_version || "v1"}</span>
                                <span className="workspace-pill muted">{typeof w.versions_count === "number" ? `${w.versions_count} ${t("workspace.versionsLabel")}` : t("workspace.versionsLabel")}</span>
                              </div>
                            </div>
                          </button>
                        );
                      })}
                    </div>

                    <div className="workspace-actions-row">
                      <button
                        className="workspace-secondary"
                        type="button"
                        disabled={workspaceActionLoading || !selectedWorkspaceKey}
                        onClick={handleLoadWorkspace}
                      >
                        {workspaceActionLoading ? t("workspace.loading") : t("workspace.load")}
                      </button>
                      <button
                        className="workspace-tertiary"
                        type="button"
                        disabled={workspaceListLoading || workspaceActionLoading}
                        onClick={fetchWorkspaces}
                      >
                        {t("workspace.refresh")}
                      </button>
                    </div>

                    <div className="workspace-divider" />

                    <button className="workspace-tertiary" type="button" onClick={triggerLoadPipeline} disabled={workspaceActionLoading}>
                      {t("workspace.loadFromFile")}
                    </button>
                    <div
                      className="workspace-mode-link workspace-create-switch"
                      role="button"
                      tabIndex={0}
                      onClick={() => setWorkspaceHomeMode("create")}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") setWorkspaceHomeMode("create");
                      }}
                    >
                      {t("workspace.createClientCta")}
                    </div>
                  </>
                )}
              </div>
            </div>

            {workspaceError && <div className="workspace-error">{workspaceError}</div>}
          </div>
        </div>
      )}

      {duplicateModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setDuplicateModal({ open: false, source: null, tenant: "", logoFile: null })}
        >
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.duplicateTitle")}</div>
                <div className="workspace-modal-title">{duplicateModal?.source?.title || ""}</div>
              </div>
              <button
                className="workspace-home-close"
                type="button"
                onClick={() => setDuplicateModal({ open: false, source: null, tenant: "", logoFile: null })}
              >
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-field">
              <label>{t("workspace.duplicateTenantLabel")}</label>
              <input
                value={duplicateModal.tenant}
                onChange={(e) => setDuplicateModal((prev) => ({ ...prev, tenant: e.target.value }))}
                placeholder={t("workspace.duplicateTenantPlaceholder")}
                disabled={workspaceActionLoading}
                autoFocus
              />
            </div>

            <div className="workspace-field">
              <label>{t("workspace.logoLocalLabel")}</label>
              <button
                className="workspace-tertiary"
                type="button"
                disabled={workspaceActionLoading}
                onClick={() => duplicateLogoFileInputRef.current?.click()}
              >
                {t("workspace.chooseLogo")}
              </button>
              {duplicateModal?.logoFile?.name && <small className="workspace-muted">{duplicateModal.logoFile.name}</small>}
            </div>

            <div className="workspace-modal-actions three">
              <button
                className="workspace-tertiary"
                type="button"
                disabled={workspaceActionLoading}
                onClick={() => setDuplicateModal({ open: false, source: null, tenant: "", logoFile: null })}
              >
                {t("actions.cancel")}
              </button>
              <button
                className="workspace-secondary"
                type="button"
                disabled={workspaceActionLoading || !String(duplicateModal?.tenant || "").trim()}
                onClick={handleDuplicateWorkspace}
              >
                {workspaceActionLoading ? t("workspace.duplicating") : t("workspace.duplicateAction")}
              </button>
            </div>
          </div>
        </div>
      )}

      {editModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setEditModal({ open: false, target: null })}
        >
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.appearanceTitle")}</div>
                <div className="workspace-modal-title">{editModal?.target?.title || ""}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setEditModal({ open: false, target: null })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-field">
              <label>{t("workspace.titleLabel")}</label>
              <input
                value={workspaceMetaDraft.title}
                onChange={(e) => setWorkspaceMetaDraft((prev) => ({ ...prev, title: e.target.value }))}
                disabled={workspaceActionLoading}
                autoFocus
              />
            </div>

            <div className="workspace-field">
              <label>{t("workspace.logoLocalLabel")}</label>
              <button
                className="workspace-tertiary"
                type="button"
                disabled={workspaceActionLoading}
                onClick={() => workspaceLogoFileInputRef.current?.click()}
              >
                {workspaceActionLoading ? t("workspace.uploadingLogo") : t("workspace.chooseLogo")}
              </button>
              {workspaceMetaDraft.logo && <small className="workspace-muted">{t("workspace.logoSelectedHint")}</small>}
            </div>

            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => setEditModal({ open: false, target: null })}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary" type="button" disabled={workspaceActionLoading} onClick={handleSaveWorkspaceAppearanceFromModal}>
                {workspaceActionLoading ? t("workspace.savingAppearance") : t("workspace.saveAppearance")}
              </button>
            </div>
          </div>
        </div>
      )}

      {versionsModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => {
            setVersionActionsModal({ open: false, version: null });
            setVersionLogsModal({ open: false, version: null, query: "" });
            setVersionsModal({ open: false, target: null, active: "", versions: [], loading: false, reasonDraft: "", page: 0, query: "" });
          }}
        >
          <div className="workspace-modal workspace-modal-wide" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.versionsTitle")}</div>
                <div className="workspace-modal-title">{versionsModal?.target?.title || ""}</div>
              </div>
              <button
                className="workspace-home-close"
                type="button"
                onClick={() => {
                  setVersionActionsModal({ open: false, version: null });
                  setVersionLogsModal({ open: false, version: null, query: "" });
                  setVersionsModal({ open: false, target: null, active: "", versions: [], loading: false, reasonDraft: "", page: 0, query: "" });
                }}
              >
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-muted">{t("workspace.versionsHint")}</div>

            <div className="workspace-versions-toolbar">
              <div className="workspace-field">
                <label>{t("workspace.changeReasonLabel")}</label>
                <input
                  value={versionsModal?.reasonDraft || ""}
                  onChange={(e) => setVersionsModal((prev) => ({ ...prev, reasonDraft: e.target.value }))}
                  placeholder={t("workspace.changeReasonPlaceholder")}
                  disabled={workspaceActionLoading}
                />
              </div>
              <div className="workspace-field">
                <label>{t("workspace.searchVersionsLabel")}</label>
                <input
                  value={versionsModal?.query || ""}
                  onChange={(e) => setVersionsModal((prev) => ({ ...prev, query: e.target.value, page: 0 }))}
                  placeholder={t("workspace.searchVersionsPlaceholder")}
                  disabled={workspaceActionLoading}
                />
              </div>
            </div>

            {!versionsModal.loading && (
              <div className="workspace-version-summary">
                {t("workspace.versionsShowing", { start: versionsPageStart, end: versionsPageEnd, total: versionsSorted.length })}
                {versionsModal?.active ? ` • ${t("workspace.activeVersionShort", { id: versionsModal.active })}` : ""}
              </div>
            )}

            {versionsModal.loading ? (
              <div className="workspace-muted">{t("workspace.loadingVersions")}</div>
            ) : (
              <div className="workspace-version-list" role="list">
                {versionsPageItems.map((v) => {
                  const history = Array.isArray(v.history) ? v.history : [];
                  const lastChange = history.length ? history[history.length - 1] : null;

                  return (
                    <div key={v.id} className={`workspace-version-row ${v.is_active ? "active" : ""}`} role="listitem">
                      <div className="workspace-version-main">
                        <div className="workspace-version-top">
                          <div className="workspace-version-name">{String(v.name || v.id)}</div>
                          <div className="workspace-version-id">{v.id}</div>
                        </div>
                        <div className="workspace-version-meta">
                          {v.is_active ? t("workspace.activeVersionLabel") : t("workspace.inactiveVersionLabel")}
                          {v.based_on ? ` • ${t("workspace.basedOnLabel")} ${v.based_on}` : ""}
                          {v.created_at ? ` • ${t("workspace.versionCreatedAt", { date: formatDateTime(v.created_at) })}` : ""}
                          {v.updated_at ? ` • ${t("workspace.versionUpdatedAt", { date: formatDateTime(v.updated_at) })}` : ""}
                        </div>
                        {lastChange?.at && (
                          <div className="workspace-version-lastchange">
                            {t("workspace.versionLastChangeLabel", { date: formatDateTime(lastChange.at) })}
                            {lastChange?.reason ? ` — ${String(lastChange.reason).trim()}` : ""}
                          </div>
                        )}
                      </div>

                      <div className="workspace-version-actions">
                        <button
                          className="workspace-secondary"
                          type="button"
                          disabled={workspaceActionLoading || versionsModal.loading}
                          onClick={() => handleOpenVersionInEditor(v.id)}
                        >
                          {t("workspace.openVersionAction")}
                        </button>
                        <button
                          className="workspace-tertiary"
                          type="button"
                          disabled={workspaceActionLoading}
                          onClick={() => setVersionActionsModal({ open: true, version: v })}
                        >
                          {t("actions.more")}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {!versionsModal.loading && versionsSorted.length > 0 && (
              <div className="workspace-version-pagination" aria-label={t("workspace.paginationLabel")}>
                <button
                  className="workspace-tertiary"
                  type="button"
                  disabled={workspaceActionLoading || versionsCurrentPage <= 0}
                  onClick={() => setVersionsModal((prev) => ({ ...prev, page: Math.max(0, (prev.page || 0) - 1) }))}
                >
                  {t("workspace.paginationPrev")}
                </button>
                <div className="workspace-version-pagination-meta">
                  {t("workspace.paginationMeta", { page: versionsCurrentPage + 1, total: versionsTotalPages })}
                </div>
                <button
                  className="workspace-tertiary"
                  type="button"
                  disabled={workspaceActionLoading || versionsCurrentPage >= versionsTotalPages - 1}
                  onClick={() =>
                    setVersionsModal((prev) => ({
                      ...prev,
                      page: Math.min(versionsTotalPages - 1, (prev.page || 0) + 1),
                    }))
                  }
                >
                  {t("workspace.paginationNext")}
                </button>
              </div>
            )}

            <div className="workspace-modal-actions">
              <button
                className="workspace-tertiary"
                type="button"
                disabled={workspaceActionLoading}
                onClick={() => {
                  setVersionActionsModal({ open: false, version: null });
                  setVersionLogsModal({ open: false, version: null, query: "" });
                  setVersionsModal({ open: false, target: null, active: "", versions: [], loading: false, reasonDraft: "", page: 0, query: "" });
                }}
              >
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary" type="button" disabled={workspaceActionLoading || versionsModal.loading} onClick={handleCreateNewVersionClean}>
                {workspaceActionLoading ? t("workspace.creatingVersion") : t("workspace.createVersionCleanAction")}
              </button>
            </div>
          </div>
        </div>
      )}

      {versionActionsModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setVersionActionsModal({ open: false, version: null })}
        >
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.versionActionsTitle")}</div>
                <div className="workspace-modal-title">{String(versionActionsModal?.version?.name || versionActionsModal?.version?.id || "")}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setVersionActionsModal({ open: false, version: null })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-version-actions-groups">
              <div className="workspace-version-actions-group">
                <div className="workspace-version-actions-group-title">{t("workspace.versionActionsViewTitle")}</div>
                <div className="workspace-version-actions-group-buttons">
                  <button
                    className="workspace-secondary"
                    type="button"
                    disabled={workspaceActionLoading || versionsModal.loading}
                    onClick={() => handleOpenVersionInEditor(versionActionsModal.version.id)}
                  >
                    {t("workspace.openVersionAction")}
                  </button>
                  <button
                    className="workspace-tertiary"
                    type="button"
                    disabled={workspaceActionLoading}
                    onClick={() => setVersionLogsModal({ open: true, version: versionActionsModal.version, query: "" })}
                  >
                    {t("workspace.openVersionLogsAction")}
                  </button>
                </div>
              </div>

              <div className="workspace-version-actions-group">
                <div className="workspace-version-actions-group-title">{t("workspace.versionActionsEditTitle")}</div>
                <div className="workspace-version-actions-group-buttons">
                  <button
                    className="workspace-tertiary"
                    type="button"
                    disabled={workspaceActionLoading || versionsModal.loading}
                    onClick={() => handleCreateNewVersionCopy(versionActionsModal.version.id, { activate: false })}
                  >
                    {t("workspace.copyVersionAction")}
                  </button>
                  <button
                    className="workspace-tertiary"
                    type="button"
                    disabled={workspaceActionLoading}
                    onClick={() =>
                      setRenameVersionModal({
                        open: true,
                        version: versionActionsModal.version.id,
                        name: String(versionActionsModal.version.name || ""),
                        reason: String(versionsModal?.reasonDraft || "").trim(),
                      })
                    }
                  >
                    {t("workspace.renameVersionAction")}
                  </button>
                </div>
              </div>

              <div className="workspace-version-actions-group">
                <div className="workspace-version-actions-group-title">{t("workspace.versionActionsActivationTitle")}</div>
                <div className="workspace-version-actions-group-buttons">
                  <button
                    className="workspace-tertiary"
                    type="button"
                    disabled={workspaceActionLoading || Boolean(versionActionsModal?.version?.is_active)}
                    onClick={() => handleActivateVersion(versionActionsModal.version.id)}
                  >
                    {t("workspace.activateVersionAction")}
                  </button>
                </div>
              </div>

              <div className="workspace-version-actions-group danger">
                <div className="workspace-version-actions-group-title">{t("workspace.versionActionsDangerTitle")}</div>
                <div className="workspace-version-actions-group-buttons">
                  <button
                    className="workspace-tertiary danger"
                    type="button"
                    disabled={
                      workspaceActionLoading ||
                      Boolean(versionActionsModal?.version?.is_active && (versionsSorted || []).length <= 1)
                    }
                    onClick={() => setDeleteVersionModal({ open: true, version: versionActionsModal.version.id })}
                  >
                    {t("workspace.deleteVersionAction")}
                  </button>
                </div>
              </div>
            </div>

            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => setVersionActionsModal({ open: false, version: null })}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary" type="button" disabled={workspaceActionLoading} onClick={() => setVersionActionsModal({ open: false, version: null })}>
                {t("actions.close")}
              </button>
            </div>
          </div>
        </div>
      )}

      {versionLogsModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setVersionLogsModal({ open: false, version: null, query: "" })}
        >
          <div className="workspace-modal workspace-modal-wide" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.versionLogsTitle")}</div>
                <div className="workspace-modal-title">{String(versionLogsModal?.version?.name || versionLogsModal?.version?.id || "")}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setVersionLogsModal({ open: false, version: null, query: "" })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-versions-toolbar">
              <div className="workspace-field">
                <label>{t("workspace.searchLogsLabel")}</label>
                <input
                  value={versionLogsModal?.query || ""}
                  onChange={(e) => setVersionLogsModal((prev) => ({ ...prev, query: e.target.value }))}
                  placeholder={t("workspace.searchLogsPlaceholder")}
                />
              </div>
            </div>

            <div className="workspace-version-history workspace-version-history-full" aria-label={t("workspace.versionHistoryTitle")}>
              <div className="workspace-version-history-list">
                {(() => {
                  const history = Array.isArray(versionLogsModal?.version?.history) ? versionLogsModal.version.history : [];
                  const q = String(versionLogsModal?.query || "").trim().toLowerCase();
                  const filtered = q
                    ? history.filter((h) => String(h?.reason || "").toLowerCase().includes(q) || String(h?.action || "").toLowerCase().includes(q))
                    : history;
                  const items = [...filtered].reverse();
                  if (!items.length) {
                    return <div className="workspace-muted">{t("workspace.noLogs")}</div>;
                  }
                  return items.map((h, idx) => (
                    <div key={`vh-${idx}`} className="workspace-version-history-item">
                      <span className="workspace-version-history-when">{formatDateTime(h.at)}</span>
                      <span className="workspace-version-history-reason">{String(h.reason || "").trim() || "-"}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>

            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" onClick={() => setVersionLogsModal({ open: false, version: null, query: "" })}>
                {t("actions.close")}
              </button>
            </div>
          </div>
        </div>
      )}

      {renameVersionModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setRenameVersionModal({ open: false, version: "", name: "", reason: "" })}
        >
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.renameVersionTitle")}</div>
                <div className="workspace-modal-title">{renameVersionModal.version}</div>
              </div>
              <button
                className="workspace-home-close"
                type="button"
                onClick={() => setRenameVersionModal({ open: false, version: "", name: "", reason: "" })}
              >
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-field">
              <label>{t("workspace.versionNameLabel")}</label>
              <input
                value={renameVersionModal.name}
                onChange={(e) => setRenameVersionModal((prev) => ({ ...prev, name: e.target.value }))}
                placeholder={t("workspace.versionNamePlaceholder")}
                disabled={workspaceActionLoading}
                autoFocus
              />
            </div>

            <div className="workspace-field">
              <label>{t("workspace.changeReasonLabel")}</label>
              <input
                value={renameVersionModal.reason}
                onChange={(e) => setRenameVersionModal((prev) => ({ ...prev, reason: e.target.value }))}
                placeholder={t("workspace.changeReasonPlaceholder")}
                disabled={workspaceActionLoading}
              />
            </div>

            <div className="workspace-modal-actions">
              <button
                className="workspace-tertiary"
                type="button"
                disabled={workspaceActionLoading}
                onClick={() => setRenameVersionModal({ open: false, version: "", name: "", reason: "" })}
              >
                {t("actions.cancel")}
              </button>
              <button
                className="workspace-secondary"
                type="button"
                disabled={workspaceActionLoading || !String(renameVersionModal?.name || "").trim()}
                onClick={handleRenameVersion}
              >
                {workspaceActionLoading ? t("workspace.savingVersionName") : t("workspace.saveVersionNameAction")}
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteVersionModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setDeleteVersionModal({ open: false, version: "" })}
        >
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.deleteVersionTitle")}</div>
                <div className="workspace-modal-title">{deleteVersionModal.version}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setDeleteVersionModal({ open: false, version: "" })}>
                {t("actions.close")}
              </button>
            </div>
            <div className="workspace-muted">{t("workspace.deleteVersionConfirm")}</div>
            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => setDeleteVersionModal({ open: false, version: "" })}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary danger" type="button" disabled={workspaceActionLoading} onClick={handleDeleteVersion}>
                {workspaceActionLoading ? t("workspace.deletingVersion") : t("workspace.deleteVersionAction")}
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setDeleteModal({ open: false, target: null })}
        >
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.deleteTitle")}</div>
                <div className="workspace-modal-title">{deleteModal?.target?.title || ""}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setDeleteModal({ open: false, target: null })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-muted">{t("workspace.deleteConfirm")}</div>

            <div className="workspace-modal-actions">
              <button
                className="workspace-tertiary"
                type="button"
                disabled={workspaceActionLoading}
                onClick={() => setDeleteModal({ open: false, target: null })}
              >
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary danger" type="button" disabled={workspaceActionLoading} onClick={handleDeleteWorkspace}>
                {workspaceActionLoading ? t("workspace.deleting") : t("workspace.deleteAction")}
              </button>
            </div>
          </div>
        </div>
      )}

      {trainModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setTrainModal((prev) => ({ ...prev, open: false, running: false }))}
        >
          <div className="workspace-modal workspace-modal-wide training-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("actions.train")}</div>
                <div className="workspace-modal-title">{t("training.title")}</div>
                <div className="workspace-muted">{t("training.subtitle")}</div>
              </div>
              <button
                className="workspace-home-close"
                type="button"
                onClick={() => setTrainModal((prev) => ({ ...prev, open: false, running: false }))}
              >
                {t("actions.close")}
              </button>
            </div>

            <TrainingModalBody
              t={t}
              trainModal={trainModal}
              setTrainModal={setTrainModal}
              parseExperimentIdsText={parseExperimentIdsText}
              trainModelsDraft={trainModelsDraft}
              setTrainModelsDraft={setTrainModelsDraft}
              nodes={nodes}
              setDatasetSelectorOpen={setDatasetSelectorOpen}
              setTrainBlockModal={setTrainBlockModal}
              setCandidatesModal={setCandidatesModal}
              setNodes={setNodes}
              trainBlockModal={trainBlockModal}
              runTraining={runTraining}
            />
          </div>
        </div>
      )}

      {/* Modal de seleção de candidatos do grid search */}
      {candidatesModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setCandidatesModal({ open: false, sessionPath: "", stepId: "" })}
        >
          <div className="workspace-modal workspace-modal-candidates" onMouseDown={(e) => e.stopPropagation()}>
            <ModelCandidatesPanel
              tenant={workspace?.tenant}
              sessionPath={candidatesModal.sessionPath}
              stepId={candidatesModal.stepId}
              onSelect={async (candidateId) => {
                console.log("Candidato selecionado:", candidateId);
                setCandidatesModal({ open: false, sessionPath: "", stepId: "" });
                // Recarrega o pipeline para pegar o novo modelo
                if (workspace?.tenant && workspace?.pipeline) {
                  try {
                    const res = await axios.get(`${API_URL}/pipelines/workspaces/${workspace.tenant}/${workspace.pipeline}`);
                    loadPipelineFromJson(res.data, { tenant: workspace.tenant, pipeline: workspace.pipeline, version: workspace.version });
                  } catch (err) {
                    console.error("Erro ao recarregar pipeline:", err);
                  }
                }
              }}
              onBack={() => {
                // Fecha o modal de candidatos e volta para o modal de treinamento
                setCandidatesModal({ open: false, sessionPath: "", stepId: "" });
                // O trainModal já deve estar aberto, apenas garante que está visível
                setTrainModal((prev) => ({ ...prev, open: true }));
              }}
              onClose={() => setCandidatesModal({ open: false, sessionPath: "", stepId: "" })}
            />
          </div>
        </div>
      )}

      {/* Modal de seleção de dataset para treinamento */}
      {datasetSelectorOpen && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setDatasetSelectorOpen(false)}
        >
          <div className="workspace-modal workspace-modal-dataset" onMouseDown={(e) => e.stopPropagation()}>
            <DatasetSelector
              tenant={workspace?.tenant}
              protocolId={trainModal.protocolId}
              selectedExperimentIds={parseExperimentIdsText(trainModal.experimentIdsText)}
              onSelectionChange={(ids) => {
                setTrainModal((prev) => ({
                  ...prev,
                  experimentIdsText: ids.join("\n"),
                }));
              }}
              onClose={() => setDatasetSelectorOpen(false)}
              disabled={trainModal.running}
            />
          </div>
        </div>
      )}

      {/* Modal do Training Studio */}
      {trainingStudioOpen && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setTrainingStudioOpen(false)}
        >
          <div className="workspace-modal workspace-modal-training" onMouseDown={(e) => e.stopPropagation()}>
            <TrainingStudio
              tenant={workspace?.tenant}
              pipeline={workspace?.pipeline}
              pipelineData={buildPipelineData()}
              version={workspace?.version}
              nodes={nodes}
              onClose={() => setTrainingStudioOpen(false)}
              onOpenDatasetSelector={() => setDatasetSelectorOpen(true)}
              onOpenCandidates={(sessionPath, stepId) => {
                setTrainingStudioOpen(false);
                setCandidatesModal({ open: true, sessionPath, stepId });
              }}
              onReloadPipeline={async () => {
                if (workspace?.tenant && workspace?.pipeline) {
                  try {
                    const res = await axios.get(`${API_URL}/pipelines/workspaces/${workspace.tenant}/${workspace.pipeline}`);
                    loadPipelineFromJson(res.data, { tenant: workspace.tenant, pipeline: workspace.pipeline, version: workspace.version });
                  } catch (err) {
                    console.error("Erro ao recarregar pipeline:", err);
                  }
                }
              }}
              initialProtocolId={trainModal.protocolId}
              initialExperimentIds={parseExperimentIdsText(trainModal.experimentIdsText)}
            />
          </div>
        </div>
      )}

      {/* Input oculto para carregar arquivo */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={loadPipeline}
        accept=".json"
        style={{ display: "none" }}
      />
      {/* Input oculto para upload de logo */}
      <input
        type="file"
        ref={workspaceLogoFileInputRef}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleUploadWorkspaceLogo(f);
          e.target.value = "";
        }}
      />
      {/* Input oculto para upload de logo (duplicação) */}
      <input
        type="file"
        ref={duplicateLogoFileInputRef}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          setDuplicateModal((prev) => ({ ...prev, logoFile: f || null }));
        }}
      />
      
      <AppHeader
        t={t}
        workspace={workspace}
        pipelineName={pipelineName}
        setPipelineName={setPipelineName}
        onOpenWorkspace={() => setWorkspaceHomeOpen(true)}
        onSave={savePipeline}
        onDownload={downloadPipelineFile}
        onAutoLayout={autoLayoutNodes}
        onOpenResults={openResultsModal}
        hasNodes={nodes.length > 0}
        canAutoLayout={nodes.length >= 2}
        hasSimulation={!!simulation}
        isDarkTheme={isDarkTheme}
        onToggleTheme={setIsDarkTheme}
      />

      <main className="five-column-layout">
        {/* PAINEL 1: Seleção de Blocos */}
        <BlocksSidebar
          t={t}
          width={leftPanel.width}
          blocksQuery={blocksQuery}
          setBlocksQuery={setBlocksQuery}
          favoriteBlocks={favoriteBlocks}
          recentBlocks={recentBlocks}
          library={library}
          blockMatchesQuery={blockMatchesQuery}
          renderBlockCardMini={renderBlockCardMini}
          onStartResize={leftPanel.startResize}
        >

            {/* 1) Entrada / Fonte */}
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">1</span>
                  <span className="stage-title">{t("blocksPanel.stages.acquisition")}</span>
                </div>
              </summary>
              <div className="blocks-grid">
                {(library.blocks || [])
                  .filter((b) => ["experiment_fetch"].includes(b.name))
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCard)}
              </div>
            </details>

            {/* 2) Roteamento */}
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">2</span>
                  <span className="stage-title">{t("blocksPanel.stages.routing")}</span>
                </div>
              </summary>
              <div className="blocks-grid">
                {(library.blocks || [])
                  .filter((b) =>
                    [
                      "label",
                      "value_in_list",
                      "numeric_compare",
                      "condition_branch",
                      "condition_gate",
                      "and_gate",
                      "or_gate",
                      "not_gate",
                      "boolean_extractor",
                      "merge",
                    ].includes(b.name)
                  )
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCard)}
              </div>
            </details>

            {/* 3) Aquisição / Extração de sensores */}
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">3</span>
                  <span className="stage-title">{t("blocksPanel.stages.sensorExtraction")}</span>
                </div>
              </summary>

              <div className="sensor-category">
                <span className="sensor-category-label">Sensores principais</span>
                <div className="blocks-grid">
                  {(library.blocks || [])
                    .filter((b) =>
                      [
                        "turbidimetry_extraction",
                        "nephelometry_extraction",
                        "fluorescence_extraction",
                        "resonant_frequencies_extraction",
                      ].includes(b.name)
                    )
                    .filter((b) => blockMatchesQuery(b, blocksQuery))
                    .map(renderBlockCard)}
                </div>
              </div>

              <div className="sensor-category">
                <span className="sensor-category-label">Temperaturas</span>
                <div className="blocks-grid">
                  {(library.blocks || [])
                    .filter((b) => ["temperatures_extraction"].includes(b.name))
                    .filter((b) => blockMatchesQuery(b, blocksQuery))
                    .map(renderBlockCard)}
                </div>
              </div>

              <div className="sensor-category">
                <span className="sensor-category-label">Fonte de alimentacao</span>
                <div className="blocks-grid">
                  {(library.blocks || [])
                    .filter((b) => ["power_supply_extraction"].includes(b.name))
                    .filter((b) => blockMatchesQuery(b, blocksQuery))
                    .map(renderBlockCard)}
                </div>
              </div>

              <div className="sensor-category">
                <span className="sensor-category-label">Pastilha Peltier</span>
                <div className="blocks-grid">
                  {(library.blocks || [])
                    .filter((b) => ["peltier_currents_extraction"].includes(b.name))
                    .filter((b) => blockMatchesQuery(b, blocksQuery))
                    .map(renderBlockCard)}
                </div>
              </div>

              <div className="sensor-category">
                <span className="sensor-category-label">Agitador magnetico</span>
                <div className="blocks-grid">
                  {(library.blocks || [])
                    .filter((b) => ["nema_currents_extraction"].includes(b.name))
                    .filter((b) => blockMatchesQuery(b, blocksQuery))
                    .map(renderBlockCard)}
                </div>
              </div>

              <div className="sensor-category">
                <span className="sensor-category-label">Estados de controle</span>
                <div className="blocks-grid">
                  {(library.blocks || [])
                    .filter((b) => ["control_state_extraction"].includes(b.name))
                    .filter((b) => blockMatchesQuery(b, blocksQuery))
                    .map(renderBlockCard)}
                </div>
              </div>
            </details>

            {/* 4) Preparação (limpeza) */}
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">4</span>
                  <span className="stage-title">{t("blocksPanel.stages.preparation")}</span>
                </div>
              </summary>
              <div className="blocks-grid">
                {(library.blocks || [])
                  .filter((b) =>
                    [
                      "sensor_fusion",
                      "time_slice",
                      "outlier_removal",
                      "moving_average_filter",
                      "savgol_filter",
                      "median_filter",
                      "lowpass_filter",
                      "exponential_filter",
                    ].includes(b.name)
                  )
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCard)}
              </div>
            </details>

            {/* 5) Transformações */}
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">5</span>
                  <span className="stage-title">{t("blocksPanel.stages.transformations")}</span>
                </div>
              </summary>
              <div className="blocks-grid">
                {(library.blocks || [])
                  .filter((b) =>
                    [
                      "derivative",
                      "integral",
                      "normalize",
                      "xyz_conversion",
                      "rgb_conversion",
                      "lab_conversion",
                      "luv_conversion",
                      "hsv_conversion",
                      "hsl_conversion",
                      "hsb_conversion",
                      "cmyk_conversion",
                      "xyy_conversion",
                    ].includes(b.name)
                  )
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCard)}
              </div>
            </details>

            {/* 6) Detecção */}
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">6</span>
                  <span className="stage-title">{t("blocksPanel.stages.detection")}</span>
                </div>
              </summary>
              <div className="blocks-grid">
                {(library.blocks || [])
                  .filter((b) => ["amplitude_detector", "derivative_detector", "ratio_detector"].includes(b.name))
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCard)}
              </div>
            </details>

            {/* 7) Features */}
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">7</span>
                  <span className="stage-title">{t("blocksPanel.stages.features")}</span>
                </div>
              </summary>
              <div className="blocks-grid">
                {(library.blocks || [])
                  .filter((b) =>
                    [
                      "statistical_features",
                      "temporal_features",
                      "shape_features",
                      "growth_features",
                      "feature_extraction",
                      "features_merge",
                    ].includes(b.name)
                  )
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCard)}
              </div>
            </details>

            {/* 8) Modelo / Predição */}
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">8</span>
                  <span className="stage-title">{t("blocksPanel.stages.modelling")}</span>
                </div>
              </summary>
              <div className="blocks-grid">
                {(library.blocks || [])
                  .filter((b) =>
                    ["curve_fit", "curve_fit_best", "ml_inference", "ml_inference_series", "ml_inference_multichannel", "ml_forecaster_series", "ml_transform_series", "ml_detector"].includes(
                      b.name
                    )
                  )
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCard)}
              </div>
            </details>

            {/* 9) Resposta / Saídas */}
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">9</span>
                  <span className="stage-title">{t("blocksPanel.stages.response")}</span>
                </div>
              </summary>
              <div className="blocks-grid">
                {(library.blocks || [])
                  .filter((b) => ["response_pack", "response_merge", "response_builder"].includes(b.name))
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCard)}
              </div>
            </details>
        </BlocksSidebar>

        {/* PAINEL 3: Canvas */}
        <section className="canvas" ref={reactFlowWrapper}>
          <PipelineStudioContext.Provider
            value={{
              openHelpModal,
              openBlockResultsModal,
              openConfigModalForNode,
              library,
              simulation,
            }}
          >
              <ReactFlowProvider>
              <ReactFlow
              nodes={nodes.map((n) => {
                const isSelected = selectedNodes.some((s) => s.id === n.id) || selectedNode?.id === n.id;
                const noneLabel = t("flows.none");
                const flowLabel = nodeFlowMetaById[n.id]?.label || noneLabel;

                return ({
                  ...n,
                  selected: isSelected,
                  data: {
                    ...n.data,
                    stepId: n.id,
                    flowLabel,
                    flowColor: nodeFlowMetaById[n.id]?.color,
                    dimmed: false,
                  },
                });
              })}
               edges={edges.map(e => {
                 const isSelected = selectedEdge?.id === e.id;

                 const baseColor = nodeFlowMetaById[e.source]?.color || nodeFlowMetaById[e.target]?.color;

                 const activeStyle = baseColor ? {
                   stroke: baseColor,
                   strokeWidth: 2,
                   opacity: 0.9,
                 } : undefined;

                 return ({
                   ...e,
                   selected: isSelected,
                   reconnectable: true,
                   style: isSelected ? {
                     stroke: '#ef4444',
                     strokeWidth: 3,
                     opacity: 0.95,
                   } : (activeStyle || undefined),
                 });
               })}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onReconnectStart={onReconnectStart}
              onReconnect={onReconnect}
              onReconnectEnd={onReconnectEnd}
              onNodeClick={handleSelect}
              onEdgeClick={handleEdgeClick}
              onPaneClick={handlePaneClick}
              onDragOver={onDragOver}
              onDrop={onDrop}
              onInit={(instance) => {
                reactFlowInstance.current = instance;
                try {
                  setViewport(instance.getViewport());
                } catch {
                  // ignore
                }
              }}
              onMove={(_, vp) => setViewport(vp)}
              fitView
              minZoom={0.05}
              maxZoom={3}
              nodeTypes={nodeTypes}
              reconnectRadius={15}
              deleteKeyCode={null}
              selectionOnDrag={true}
              selectionMode={SelectionMode.Partial}
              panOnDrag={[1, 2]}
              elevateNodesOnSelect={true}
            >
              <Background gap={16} color="#d0d7ff" />
              <MiniMap pannable zoomable position="bottom-left" />
              <Controls />
              </ReactFlow>
            </ReactFlowProvider>
          </PipelineStudioContext.Provider>

          {analysisAreas.length > 0 && (
            <div className="analysis-areas-overlay" aria-hidden="true">
              <div
                className="analysis-areas-transform"
                style={{
                  transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
                }}
              >
                {analysisAreas.map((area) => (
                  <div
                    key={area.analysisId}
                    className="analysis-area"
                    style={{
                      left: area.left,
                      top: area.top,
                      width: area.width,
                      height: area.height,
                      "--analysis-color": area.color,
                    }}
                  >
                    <div className="analysis-area-title">
                      <span className="analysis-area-title__label">{t("analysis.badge", { id: area.analysisId })}</span>
                      <span className="analysis-area-title__meta">{t("analysis.flowsCount", { count: area.flowCount })}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {flowLanes.length > 0 && (
            <div className="flow-lanes-overlay" aria-hidden="true">
              <div
                className="flow-lanes-transform"
                style={{
                  transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
                }}
              >
                {flowLanes.map((lane) => (
                  <div
                    key={lane.label}
                    className="flow-lane"
                    style={{
                      left: lane.left,
                      top: lane.top,
                      width: lane.width,
                      height: lane.height,
                      "--lane-color": lane.color,
                    }}
                  >
                    <div className="flow-lane-title">{lane.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Selection Toolbar - shown when nodes are selected */}
          {(selectedNodes.length >= 1 || selectedNode) && (
            <div className="alignment-toolbar">
              <div className="alignment-toolbar-label">
                {(() => {
                  const count = selectedNodes.length || 1;
                  return count > 1
                    ? t("canvas.selectionPlural", { count })
                    : t("canvas.selection", { count });
                })()}
              </div>
              <div className="alignment-toolbar-buttons">
                {/* Copiar/Colar/Duplicar - sempre visível */}
                <div className="alignment-group">
                  <span className="alignment-group-label">Editar</span>
                  <button onClick={copySelectedNodes} title="Copiar (Ctrl+C)">
                    <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
                  </button>
                  <button onClick={pasteNodes} title="Colar (Ctrl+V)" disabled={clipboard.nodes.length === 0}>
                    <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M19 2h-4.18C14.4.84 13.3 0 12 0c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm7 18H5V4h2v3h10V4h2v16z"/></svg>
                  </button>
                  <button onClick={duplicateSelectedNodes} title="Duplicar (Ctrl+D)">
                    <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M11 17H4a2 2 0 0 1-2-2V3a2 2 0 0 1 2-2h12v2H4v12h7v2m11-2V7a2 2 0 0 0-2-2H8v2h12v10a2 2 0 0 0 2 2h-1v-2M9 7v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H11a2 2 0 0 0-2 2m2 0h10v10H11V7z"/></svg>
                  </button>
                </div>
                
                {/* Alinhamento - só quando 2+ nós selecionados */}
                {selectedNodes.length >= 2 && (
                  <>
                    <div className="alignment-separator" />
                    <div className="alignment-group">
                      <span className="alignment-group-label">{t("actions.autoLayout")}</span>
                      <button onClick={autoLayoutNodes} title={t("actions.autoLayoutTitle")}>
                        <svg viewBox="0 0 24 24" width="16" height="16">
                          <path fill="currentColor" d="M4 4h6v4H4V4zm10 0h6v4h-6V4zM4 10h6v4H4v-4zm10 0h6v4h-6v-4zM4 16h6v4H4v-4zm10 0h6v4h-6v-4z" />
                        </svg>
                      </button>
                    </div>
                    <div className="alignment-group">
                      <span className="alignment-group-label">Alinhar</span>
                      <button onClick={alignNodesLeft} title="Alinhar à esquerda">
                        <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M4 22H2V2h2v20zM22 7H6v3h16V7zm-6 7H6v3h10v-3z"/></svg>
                      </button>
                      <button onClick={alignNodesCenterH} title="Centralizar horizontalmente">
                        <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M11 2h2v5h8v3H13v4h6v3h-6v5h-2v-5H5v-3h6V10H3V7h8V2z"/></svg>
                      </button>
                      <button onClick={alignNodesRight} title="Alinhar à direita">
                        <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M20 2h2v20h-2V2zM2 7h16v3H2V7zm6 7h10v3H8v-3z"/></svg>
                      </button>
                      <div className="alignment-separator" />
                      <button onClick={alignNodesTop} title="Alinhar ao topo">
                        <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M22 2v2H2V2h20zM7 22V6h3v16H7zm7-6V6h3v10h-3z"/></svg>
                      </button>
                      <button onClick={alignNodesCenterV} title="Centralizar verticalmente">
                        <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M2 11v2h5v8h3V13h4v6h3v-6h5v-2h-5V5h-3v6h-4V3H7v8H2z"/></svg>
                      </button>
                      <button onClick={alignNodesBottom} title="Alinhar abaixo">
                        <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M22 22v-2H2v2h20zM7 2v16h3V2H7zm7 6v10h3V8h-3z"/></svg>
                      </button>
                    </div>
                  </>
                )}
                
                {/* Distribuir - só quando 3+ nós selecionados */}
                {selectedNodes.length >= 3 && (
                  <>
                    <div className="alignment-separator" />
                    <div className="alignment-group">
                      <span className="alignment-group-label">Distribuir</span>
                      <button onClick={distributeNodesH} title="Distribuir horizontalmente">
                        <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M4 5v14H2V5h2zm4 2v10h3V7H8zm5 2v6h3V9h-3zm5-2v10h3V7h-3zm4-2v14h2V5h-2z"/></svg>
                      </button>
                      <button onClick={distributeNodesV} title="Distribuir verticalmente">
                        <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M5 4h14V2H5v2zm2 4h10v3H7V8zm2 5h6v3H9v-3zm-2 5h10v3H7v-3zm-2 4h14v2H5v-2z"/></svg>
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Floating actions */}
          <div className="floating-actions" aria-label="Ações rápidas">
            <button
              className="floating-train-button"
              type="button"
              onClick={() => setTrainingStudioOpen(true)}
              disabled={!workspace?.tenant || !workspace?.pipeline || trainModal.running}
              title={trainModal.running ? t("actions.training") : t("actions.train")}
            >
              {trainModal.running ? (
                <span className="simulate-spinner" aria-hidden="true" />
              ) : (
                <svg className="simulate-play" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M12 3 1 9l11 6 9-4.91V17h2V9L12 3zm0 9.2L4.31 9 12 4.8 19.69 9 12 12.2zM6 12.5V16c0 2.21 2.69 4 6 4s6-1.79 6-4v-3.5l-6 3.27-6-3.27z" />
                </svg>
              )}
              {trainModal.running ? t("actions.training") : t("actions.train")}
            </button>

            <button
              className="floating-simulate-button"
              onClick={handleSimulate}
              disabled={isRunning || !nodes.length}
              title={isRunning ? t("actions.simulating") : t("actions.simulate")}
              aria-busy={isRunning ? "true" : "false"}
            >
              {isRunning ? (
                <span className="simulate-spinner" aria-hidden="true" />
              ) : (
                <svg className="simulate-play" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
              {isRunning ? t("actions.simulating") : t("actions.simulate")}
            </button>
          </div>

          {/* Context Menu */}
          {contextMenu && (
            <>
              {/* Overlay to close menu when clicking outside */}
              <div
                className="context-menu-overlay"
                onClick={closeContextMenu}
                style={{
                  position: 'fixed',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  zIndex: 999
                }}
              />
              <div
                className="context-menu"
                style={{
                  position: 'fixed',
                  left: contextMenu.x,
                  top: contextMenu.y,
                  zIndex: 1000
                }}
              >
                {contextMenu.type === 'node' ? (
                  <div className="context-menu-item" onClick={deleteNode}>
                    {(selectedNodes.length > 1 && selectedNodes.some((n) => n.id === contextMenu.nodeId))
                      ? t("actions.deleteSelection", { count: selectedNodes.length })
                      : t("actions.deleteBlock")}
                  </div>
                ) : contextMenu.type === 'selection' ? (
                  <div
                    className="context-menu-item"
                    onClick={() => {
                      const ids = (contextMenu.nodeIds || []).map(String).filter(Boolean);
                      if (!ids.length) return;
                      setConfirmDelete({ open: true, nodeId: ids[0] ?? null, nodeIds: ids });
                      closeContextMenu();
                    }}
                  >
                    {t("actions.deleteSelection", { count: (contextMenu.nodeIds || []).length })}
                  </div>
                ) : contextMenu.type === 'edge' ? (
                  <div className="context-menu-item" onClick={deleteEdge}>
                    {t("actions.removeConnection")}
                  </div>
                ) : null}
              </div>
            </>
          )}
        </section>

        {/* PAINEL 4: Configuração (modal) */}
        {configModalOpen && (selectedNode || selectedEdge) && (
          <div className="config-modal-overlay" onClick={closeConfigModal}>
            <aside
              className="panel-config panel-config--overlay"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="panel-header">
                <h2>{t("panels.configuration")}</h2>
                <button type="button" className="panel-close" onClick={closeConfigModal}>
                  {t("actions.close")}
                </button>
              </div>
              <div className="panel-content">
            {selectedNode ? (
              <>
                <div className="config-node-header">
                  <strong>{selectedNode.data.label}</strong>
                  <button 
                    className="delete-node-btn"
                    onClick={() => showConfirmDelete(selectedNode.id)}
                    aria-label={t("actions.deleteBlock")}
                    title={t("actions.deleteBlock")}
                  >
                    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">
                      <path
                        fill="currentColor"
                        d="M9 3h6l1 2h5v2H3V5h5l1-2zm1 7h2v9h-2v-9zm4 0h2v9h-2v-9zM6 8h12l-1 13H7L6 8z"
                      />
                    </svg>
                  </button>
                </div>

                {selectedBlockDefinition && (
                  <div className="node-io-summary">
                    <div className="io-column">
                      <span className="io-label">
                        {t("configuration.ioInputs", { count: selectedBlockDefinition.data_inputs?.length || 0 })}
                      </span>
                    </div>
                    <div className="io-column">
                      <span className="io-label">
                        {t("configuration.ioOutputs", { count: selectedBlockDefinition.data_outputs?.length || 0 })}
                      </span>
                    </div>
                  </div>
                )}

                <div className="config-form">
                  {selectedNode && selectedNode.data && selectedNode.data.blockName === "experiment_fetch" ? (
                    <>
                      {console.log('Rendering experiment_fetch config for node:', selectedNode.id)}
                      {/* Grupo 1: Fonte de Dados */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.dataSource")}</legend>
                        {["experimentId", "tenant"].map((f) => renderConfigField(f))}
                        
                        {/* Botão para experimento padrão */}
                        <div className="config-field">
                          <label className="config-label">
                            <input
                              type="checkbox"
                              checked={useDefaultExperiment}
                              onChange={(e) => {
                                const newValue = e.target.checked;
                                console.log('Toggle changed:', newValue);
                                setUseDefaultExperiment(newValue);
                                
                                // Atualizar todos os campos de uma vez para evitar problemas de estado
                                const currentConfig = selectedNode.data.config || {};
                                let nextConfig;
                                
                                if (newValue) {
                                  // Se ativar experimento padrão, preencher os campos obrigatórios
                                  nextConfig = {
                                    ...currentConfig,
                                    use_default_experiment: true,
                                    experimentId: "019b221a-bfa8-705a-9b40-8b30f144ef68",
                                    analysisId: "68cb3fb380ac865ce0647ea8",
                                    tenant: "corsan",
                                    generate_output_graphs: true,
                                    include_experiment_output: true,
                                    include_experiment_data_output: true,
                                    graph_config: {
                                      turbidimetry: ["f1","f2","f3","f4","f5","f6","f7","f8","clear","nir"],
                                      nephelometry: ["f1","f2","f3","f4","f5","f6","f7","f8","clear","nir"],
                                      fluorescence: ["f1","f2","f3","f4","f5","f6","f7","f8","clear","nir"],
                                    },
                                  };
                                } else {
                                  // Se desativar, limpar os campos
                                  nextConfig = {
                                    ...currentConfig,
                                    use_default_experiment: false,
                                    experimentId: "",
                                    analysisId: "",
                                    tenant: "",
                                    generate_output_graphs: false
                                  };
                                }
                                
                                applyConfigToSelectedNode(nextConfig);
                              }}
                              className="config-checkbox"
                            />
                            {t("configuration.useDemoExperiment")}
                          </label>
                          <div className="config-description">
                            Carrega dados de exemplo para testar o pipeline
                          </div>
                        </div>
                      </fieldset>

                      {/* Grupo 2: Debug */}
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_experiment_output)}
                              onChange={(e) => updateNodeConfigField("include_experiment_output", e.target.checked)}
                            />
                            <span>Mostrar metadados do experimento (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_experiment_data_output)}
                              onChange={(e) => updateNodeConfigField("include_experiment_data_output", e.target.checked)}
                            />
                            <span>Mostrar dados brutos (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.generate_output_graphs)}
                              onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)}
                            />
                            <span>Gerar gráficos de visualização</span>
                          </label>
                        </div>
                        {selectedNode.data.config?.generate_output_graphs && (
                          <div className="graph-options">
                            {renderConfigField("graph_config")}
                          </div>
                        )}
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName?.includes("_extraction") ? (
                    <>
                      {/* Blocos de extração de sensor - Layout com cards */}
                      
                      {/* Card 0: Modo de saída (apenas para sensores espectrais) */}
                      {['turbidimetry_extraction', 'nephelometry_extraction', 'fluorescence_extraction'].includes(selectedNode?.data?.blockName) && (
                        <fieldset className="config-group">
                          <legend>{t("configuration.outputMode")}</legend>
                          <div className="config-field">
                            <label className="config-label">Formato dos valores</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.output_mode || "raw"}
                              onChange={(e) => updateNodeConfigField("output_mode", e.target.value)}
                            >
                              <option value="raw">RAW (valores brutos)</option>
                              <option value="basic_counts">Basic Counts (RAW / gain × timeMs)</option>
                            </select>
                          </div>
                        </fieldset>
                      )}

                      {/* Card 1: Limpeza de Dados */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.dataCleaning")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={selectedNode.data.config?.remove_duplicates !== false}
                              onChange={(e) => updateNodeConfigField("remove_duplicates", e.target.checked)}
                            />
                            <span>Remover timestamps duplicados</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={selectedNode.data.config?.validate_data !== false}
                              onChange={(e) => updateNodeConfigField("validate_data", e.target.checked)}
                            />
                            <span>Validar dados extraídos</span>
                          </label>
                        </div>
                      </fieldset>

                      {/* Card 2: Debug/Saídas */}
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Incluir dados brutos (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.generate_output_graphs)}
                              onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)}
                            />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "time_slice" ? (
                    <>
                      {/* Bloco Time Slice - Layout com cards */}
                      
                      {/* Card 1: Configuração de Corte */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.timeCut")}</legend>
                        <div className="config-field">
                          <label className="config-label">Modo de corte</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.slice_mode || "time"}
                            onChange={(e) => updateNodeConfigField("slice_mode", e.target.value)}
                          >
                            <option value="time">Por tempo (minutos)</option>
                            <option value="index">Por índice</option>
                          </select>
                        </div>
                        
                        {(selectedNode.data.config?.slice_mode || "time") === "time" ? (
                          <>
                            <div className="config-field">
                              <label className="config-label">Tempo inicial (min)</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.start_time_min ?? 0}
                                onChange={(e) => updateNodeConfigField("start_time_min", parseFloat(e.target.value) || 0)}
                                step="0.5"
                                min="0"
                              />
                            </div>
                            <div className="config-field">
                              <label className="config-label">Tempo final (min)</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.end_time_min ?? ""}
                                onChange={(e) => updateNodeConfigField("end_time_min", e.target.value ? parseFloat(e.target.value) : null)}
                                placeholder="Até o fim"
                                step="0.5"
                                min="0"
                              />
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="config-field">
                              <label className="config-label">Índice inicial</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.start_index ?? 0}
                                onChange={(e) => updateNodeConfigField("start_index", parseInt(e.target.value) || 0)}
                                min="0"
                              />
                            </div>
                            <div className="config-field">
                              <label className="config-label">Índice final</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.end_index ?? ""}
                                onChange={(e) => updateNodeConfigField("end_index", e.target.value ? parseInt(e.target.value) : null)}
                                placeholder="Até o fim"
                                min="0"
                              />
                            </div>
                          </>
                        )}
                        
                      </fieldset>

                      {/* Card 2: Debug */}
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.generate_output_graphs)}
                              onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)}
                            />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "sensor_fusion" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.sensorFusion")}</legend>
                        <p className="config-hint">{t("configuration.sensorFusionHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.sensorFusionInputs")}</label>
                          {(() => {
                            const maxInputs = 6;
                            const currentKeys = (selectedNode?.data?.dataInputs || []).filter((k) => String(k).startsWith("sensor_data_"));
                            const configuredCount = Number(selectedNode?.data?.config?.inputs_count);
                            const count = Math.max(
                              1,
                              Math.min(maxInputs, Number.isFinite(configuredCount) ? configuredCount : (currentKeys.length || 2))
                            );
                            const keys = Array.from({ length: count }, (_, i) => `sensor_data_${i + 1}`);

                            const sources = (() => {
                              const val = selectedNode?.data?.config?.sources;
                              if (Array.isArray(val)) return val;
                              if (typeof val === "string" && val.trim().length) {
                                try {
                                  const parsed = JSON.parse(val);
                                  return Array.isArray(parsed) ? parsed : [];
                                } catch {
                                  return [];
                                }
                              }
                              return [];
                            })();

                            const updateSource = (inputKey, patch) => {
                              const next = Array.isArray(sources) ? [...sources] : [];
                              const idx = next.findIndex((s) => s && s.input === inputKey);
                              const base = idx >= 0 && next[idx] ? next[idx] : { input: inputKey };
                              const merged = { ...base, ...patch, input: inputKey };
                              if (idx >= 0) next[idx] = merged;
                              else next.push(merged);
                              updateNodeConfigField("sources", next);
                            };

                            return (
                              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                                  <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                                    <button
                                      type="button"
                                      className="btn btn-secondary"
                                      onClick={() =>
                                        setSequentialInputsCount(selectedNode.id, count - 1, {
                                          prefix: "sensor_data_",
                                          max: 6,
                                          configKey: "inputs_count",
                                        })
                                      }
                                      disabled={count <= 1}
                                    >
                                      -
                                    </button>
                                    <span style={{ fontSize: 12 }}>{t("configuration.inputsCount", { count })}</span>
                                    <button
                                      type="button"
                                      className="btn btn-secondary"
                                      onClick={() =>
                                        setSequentialInputsCount(selectedNode.id, count + 1, {
                                          prefix: "sensor_data_",
                                          max: 6,
                                          configKey: "inputs_count",
                                        })
                                      }
                                      disabled={count >= maxInputs}
                                    >
                                      +
                                    </button>
                                  </span>
                                </div>

                                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                  {keys.map((k) => {
                                    const s = sources.find((x) => x && x.input === k) || { input: k };
                                    const prefix = typeof s.prefix === "string" ? s.prefix : "";
                                    const channels = Array.isArray(s.channels) ? s.channels.join(", ") : s.channels || "";

                                    const edge = edges.find((e) => {
                                      if (e.target !== selectedNode.id) return false;
                                      const th = e.targetHandle || "";
                                      const inputName = th.includes("-in-") ? th.split("-in-")[1] : th;
                                      return inputName === k;
                                    });
                                    const sourceId = edge?.source;
                                    const sourceNode = sourceId ? nodes.find((n) => n.id === sourceId) : null;
                                    const stepData = sourceId ? simulation?.step_results?.[sourceId] : null;
                                    const stepSensorData = stepData?.data?.sensor_data || stepData?.sensor_data || null;
                                    const availableChannels = (() => {
                                      const fromData = stepSensorData?.available_channels;
                                      if (Array.isArray(fromData) && fromData.length) return fromData.map(String);
                                      const ch = stepSensorData?.channels;
                                      if (ch && typeof ch === "object" && !Array.isArray(ch)) return Object.keys(ch);

                                      const blockName = String(sourceNode?.data?.blockName || "");
                                      const spectralDefaults = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"];
                                      const conversionDefaults = {
                                        rgb_conversion: ["RGB_R", "RGB_G", "RGB_B"],
                                        hsv_conversion: ["HSV_H", "HSV_S", "HSV_V"],
                                        hsl_conversion: ["HSL_H", "HSL_S", "HSL_L"],
                                        hsb_conversion: ["HSB_H", "HSB_S", "HSB_B"],
                                        lab_conversion: ["LAB_L", "LAB_A", "LAB_B"],
                                        luv_conversion: ["LUV_L", "LUV_U", "LUV_V"],
                                        xyz_conversion: ["XYZ_X", "XYZ_Y", "XYZ_Z"],
                                        cmyk_conversion: ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"],
                                        xyy_conversion: ["xyY_x", "xyY_y", "xyY_Y"],
                                      };
                                      if (blockName in conversionDefaults) return conversionDefaults[blockName];
                                      if (blockName.endsWith("_extraction") && !blockName.includes("temperatures") && !blockName.includes("power_supply") && !blockName.includes("peltier") && !blockName.includes("nema") && !blockName.includes("control_state")) {
                                        return spectralDefaults;
                                      }
                                      return [];
                                    })();

                                    const suggestedPrefix = (() => {
                                      const sk = stepSensorData?.sensor_key;
                                      if (typeof sk === "string" && sk.trim()) return sk.trim().slice(0, 16);
                                      const sn = stepSensorData?.sensor_name;
                                      if (typeof sn === "string" && sn.trim()) return sn.trim().split("_")[0].slice(0, 16);
                                      return "";
                                    })();

                                    const selectedList = (() => {
                                      if (Array.isArray(s.channels)) return s.channels.map(String).filter(Boolean);
                                      if (typeof s.channels === "string") {
                                        return s.channels
                                          .split(",")
                                          .map((v) => v.trim())
                                          .filter(Boolean);
                                      }
                                      return [];
                                    })();

                                    const isAll = availableChannels.length > 0 && (selectedList.length === 0 || selectedList.length === availableChannels.length);

                                    return (
                                      <div
                                        key={k}
                                        style={{
                                          border: "1px solid var(--color-gray-200)",
                                          borderRadius: 12,
                                          padding: 10,
                                          background: "rgba(255,255,255,0.6)",
                                        }}
                                      >
                                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8, marginBottom: 8 }}>
                                          <div style={{ fontWeight: 600, fontSize: 12 }}>{k}</div>
                                          <div style={{ fontSize: 11, color: "var(--color-gray-500)", textAlign: "right" }}>
                                            {sourceNode?.data?.label ? (
                                              <span title={sourceNode?.data?.blockName || ""}>{sourceNode.data.label}</span>
                                            ) : (
                                              <span>{t("configuration.noConnection")}</span>
                                            )}
                                            {stepSensorData?.color_space ? <span>{` • ${stepSensorData.color_space}`}</span> : null}
                                          </div>
                                        </div>
                                        <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 8 }}>
                                          <div className="config-field" style={{ margin: 0 }}>
                                            <label className="config-label">{t("configuration.prefix")}</label>
                                            <input
                                              type="text"
                                              className="config-input"
                                              value={prefix}
                                              onChange={(e) => updateSource(k, { prefix: e.target.value })}
                                              placeholder={t("configuration.prefixPlaceholder")}
                                            />
                                          </div>
                                          <div className="config-field" style={{ margin: 0 }}>
                                            <label className="config-label">{t("configuration.channels")}</label>
                                            <input
                                              type="text"
                                              className="config-input"
                                              value={channels}
                                              onChange={(e) => {
                                                const parsed = e.target.value
                                                  .split(",")
                                                  .map((v) => v.trim())
                                                  .filter(Boolean);
                                                updateSource(k, { channels: parsed.length ? parsed : e.target.value });
                                              }}
                                              placeholder={t("configuration.sensorFusionChannelsPlaceholder")}
                                            />
                                            <small className="config-hint-small">{t("configuration.sensorFusionChannelsHint")}</small>
                                          </div>
                                        </div>

                                        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginTop: 8 }}>
                                          <button
                                            type="button"
                                            className="btn btn-secondary"
                                            onClick={() => {
                                              if (suggestedPrefix && !prefix) updateSource(k, { prefix: suggestedPrefix });
                                              updateSource(k, { channels: [] });
                                            }}
                                            disabled={!availableChannels.length && !suggestedPrefix}
                                            title={t("configuration.useSuggested")}
                                          >
                                            {t("configuration.useSuggested")}
                                          </button>
                                          {availableChannels.length > 0 && (
                                            <span style={{ fontSize: 11, color: "var(--color-gray-500)" }}>
                                              {t("configuration.availableChannels", { count: availableChannels.length })}
                                            </span>
                                          )}
                                        </div>

                                        {availableChannels.length > 0 && (
                                          <div className="chip-group" style={{ marginTop: 8 }}>
                                            <button
                                              type="button"
                                              className={`chip chip-all ${isAll ? "active" : ""}`}
                                              onClick={() => updateSource(k, { channels: [] })}
                                              title={t("configuration.allChannels")}
                                            >
                                              {t("common.all")}
                                            </button>
                                            {availableChannels.map((chName) => {
                                              const isActive = isAll ? false : selectedList.includes(chName);
                                              return (
                                                <button
                                                  key={chName}
                                                  type="button"
                                                  className={`chip ${isActive ? "active" : ""}`}
                                                  onClick={() => {
                                                    if (isAll) {
                                                      updateSource(k, { channels: [chName] });
                                                      return;
                                                    }
                                                    const nextSet = new Set(selectedList);
                                                    if (nextSet.has(chName)) nextSet.delete(chName);
                                                    else nextSet.add(chName);
                                                    updateSource(k, { channels: Array.from(nextSet) });
                                                  }}
                                                >
                                                  {chName}
                                                </button>
                                              );
                                            })}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            );
                          })()}
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.mergeMode")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.merge_mode || "intersection"}
                              onChange={(e) => updateNodeConfigField("merge_mode", e.target.value)}
                            >
                              <option value="intersection">{t("configuration.mergeModes.intersection")}</option>
                              <option value="union">{t("configuration.mergeModes.union")}</option>
                            </select>
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.resampleStep")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.resample_step ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("resample_step", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder={t("configuration.resampleStepPlaceholder")}
                              step="0.1"
                            />
                            <small className="config-hint-small">{t("configuration.resampleStepHint")}</small>
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "outlier_removal" ? (
                    <>
                      {/* Bloco Outlier Removal - Layout com cards */}
                      
                      {/* Card 1: Método de Detecção */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.outlierDetection")}</legend>
                        <div className="config-field">
                          <label className="config-label">Método</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.method || "zscore"}
                            onChange={(e) => updateNodeConfigField("method", e.target.value)}
                          >
                            <option value="zscore">Z-Score (desvio padrão)</option>
                            <option value="iqr">IQR (intervalo interquartil)</option>
                            <option value="mad">MAD (desvio mediano absoluto)</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-label">
                            Limiar ({selectedNode.data.config?.method === "iqr" ? "multiplicador IQR" : "desvios"})
                          </label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.threshold ?? (selectedNode.data.config?.method === "iqr" ? 1.5 : 3.0)}
                            onChange={(e) => updateNodeConfigField("threshold", parseFloat(e.target.value) || 3.0)}
                            step="0.1"
                            min="0.1"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Estratégia</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.replace_strategy || "remove"}
                            onChange={(e) => updateNodeConfigField("replace_strategy", e.target.value)}
                          >
                            <option value="remove">Remover linhas com outliers</option>
                            <option value="interpolate">Interpolar valores</option>
                          </select>
                        </div>
                      </fieldset>

                      {/* Card 2: Debug */}
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.generate_output_graphs)}
                              onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)}
                            />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "moving_average_filter" ? (
                    <>
                      {/* Bloco Moving Average Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.movingAverage")}</legend>
                        <div className="config-field">
                          <label className="config-label">Tamanho da janela</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.window_size ?? 5}
                            onChange={(e) => updateNodeConfigField("window_size", parseInt(e.target.value) || 5)}
                            min="1"
                            max="50"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Alinhamento</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.alignment || "center"}
                            onChange={(e) => updateNodeConfigField("alignment", e.target.value)}
                          >
                            <option value="center">Centralizado</option>
                            <option value="left">Esquerda (causal)</option>
                            <option value="right">Direita</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "savgol_filter" ? (
                    <>
                      {/* Bloco Savitzky-Golay Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.savgol")}</legend>
                        <div className="config-field">
                          <label className="config-label">Tamanho da janela (ímpar)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.window_size ?? 11}
                            onChange={(e) => updateNodeConfigField("window_size", parseInt(e.target.value) || 11)}
                            min="3"
                            max="51"
                            step="2"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Ordem do polinômio</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.poly_order ?? 3}
                            onChange={(e) => updateNodeConfigField("poly_order", parseInt(e.target.value) || 3)}
                            min="1"
                            max="10"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "median_filter" ? (
                    <>
                      {/* Bloco Median Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.medianFilter")}</legend>
                        <div className="config-field">
                          <label className="config-label">Tamanho do kernel (ímpar)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.kernel_size ?? 5}
                            onChange={(e) => updateNodeConfigField("kernel_size", parseInt(e.target.value) || 5)}
                            min="3"
                            max="31"
                            step="2"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "lowpass_filter" ? (
                    <>
                      {/* Bloco Lowpass Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.lowpass")}</legend>
                        <div className="config-field">
                          <label className="config-label">Frequência de corte (0-1)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.cutoff_freq ?? 0.1}
                            onChange={(e) => updateNodeConfigField("cutoff_freq", parseFloat(e.target.value) || 0.1)}
                            min="0.01"
                            max="0.99"
                            step="0.01"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Ordem do filtro</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.order ?? 4}
                            onChange={(e) => updateNodeConfigField("order", parseInt(e.target.value) || 4)}
                            min="1"
                            max="10"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "exponential_filter" ? (
                    <>
                      {/* Bloco Exponential Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.ema")}</legend>
                        <div className="config-field">
                          <label className="config-label">Alpha (0-1, menor = mais suave)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.alpha ?? 0.3}
                            onChange={(e) => updateNodeConfigField("alpha", parseFloat(e.target.value) || 0.3)}
                            min="0.01"
                            max="1.0"
                            step="0.05"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "derivative" ? (
                    <>
                      {/* Bloco Derivative */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.derivative")}</legend>
                        <div className="config-field">
                          <label className="config-label">Método</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.method || "gradient"}
                            onChange={(e) => updateNodeConfigField("method", e.target.value)}
                          >
                            <option value="gradient">Gradiente (numpy)</option>
                            <option value="diff">Diferença simples</option>
                            <option value="savgol">Savgol (suavizado)</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Ordem da derivada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.order || 1}
                            onChange={(e) => updateNodeConfigField("order", parseInt(e.target.value))}
                          >
                            <option value={1}>1ª derivada (velocidade)</option>
                            <option value={2}>2ª derivada (aceleração)</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "integral" ? (
                    <>
                      {/* Bloco Integral */}
                      <fieldset className="config-group">
                        <legend>∫ Integral</legend>
                        <div className="config-field">
                          <label className="config-label">Método</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.method || "trapz"}
                            onChange={(e) => updateNodeConfigField("method", e.target.value)}
                          >
                            <option value="trapz">Trapezoidal (recomendado)</option>
                            <option value="cumsum">Soma cumulativa</option>
                            <option value="simpson">Simpson</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "normalize" ? (
                    <>
                      {/* Bloco Normalize */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.normalization")}</legend>
                        <div className="config-field">
                          <label className="config-label">Método</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.method || "minmax"}
                            onChange={(e) => updateNodeConfigField("method", e.target.value)}
                          >
                            <option value="minmax">Min-Max (0 a 1)</option>
                            <option value="zscore">Z-Score (média 0, std 1)</option>
                            <option value="robust">Robust (baseado em mediana)</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "curve_fit" ? (
                    <>
                      {/* Bloco Curve Fit */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.curveFit")}</legend>
                        <p className="config-hint">Ajusta modelo matemático aos dados de crescimento</p>
                        <div className="config-field">
                          <label className="config-label">Modelo matemático</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.model || "richards"}
                            onChange={(e) => updateNodeConfigField("model", e.target.value)}
                          >
                            <option value="richards">Richards (flexível)</option>
                            <option value="gompertz">Gompertz (clássico)</option>
                            <option value="logistic">Logístico</option>
                            <option value="baranyi">Baranyi (com lag)</option>
                            <option value="auto">Automático (melhor fit)</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Todos os canais</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="ex: f1, clear, nir"
                            />
                          )}
                          <small className="config-hint-small">
                            {availableChannels.length > 0 
                              ? "Selecione um canal específico ou todos" 
                              : "Conecte um bloco para ver os canais disponíveis"}
                          </small>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Tentativas de ajuste</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.max_attempts ?? 15}
                            onChange={(e) => updateNodeConfigField("max_attempts", parseInt(e.target.value) || 15)}
                            min="1"
                            max="50"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Tolerância de erro</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.tolerance ?? 0.001}
                            onChange={(e) => updateNodeConfigField("tolerance", parseFloat(e.target.value) || 0.001)}
                            min="0.0001"
                            max="0.1"
                            step="0.0001"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group">
                        <legend>Reamostragem (ML)</legend>
                        <p className="config-hint">Padroniza saída para uso em modelos de Machine Learning</p>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input 
                              type="checkbox" 
                              checked={Boolean(selectedNode.data.config?.resample_output)} 
                              onChange={(e) => updateNodeConfigField("resample_output", e.target.checked)} 
                            />
                            <span>Reamostrar em grid regular</span>
                          </label>
                        </div>
                        {selectedNode.data.config?.resample_output && (
                          <>
                            <div className="config-field">
                              <label className="config-label">Número de pontos</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.resample_points ?? 100}
                                onChange={(e) => updateNodeConfigField("resample_points", parseInt(e.target.value) || 100)}
                                min="10"
                                max="1000"
                              />
                              <small className="config-hint-small">Quantidade de pontos na curva de saída</small>
                            </div>
                            <div className="config-field">
                              <label className="config-label">X mínimo (min)</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.resample_xmin ?? 0}
                                onChange={(e) => updateNodeConfigField("resample_xmin", parseFloat(e.target.value) || 0)}
                                min="0"
                                step="1"
                              />
                            </div>
                            <div className="config-field">
                              <label className="config-label">X máximo (min)</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.resample_xmax ?? 0}
                                onChange={(e) => updateNodeConfigField("resample_xmax", parseFloat(e.target.value) || 0)}
                                min="0"
                                step="1"
                                placeholder="0 = automático"
                              />
                              <small className="config-hint-small">0 = usa máximo do experimento. Ex: 1440 = 24h</small>
                            </div>
                          </>
                        )}
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "curve_fit_best" ? (
                    <>
                      {/* Bloco Curve Fit Best */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.bestFit")}</legend>
                        <p className="config-hint">Testa múltiplos modelos e retorna o melhor</p>
                        <div className="config-field">
                          <label className="config-label">Modelos a testar</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.models || "all"}
                            onChange={(e) => updateNodeConfigField("models", e.target.value)}
                            placeholder="all ou: richards, gompertz, logistic"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Primeiro canal disponível</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="ex: f1 (vazio = primeiro)"
                            />
                          )}
                          <small className="config-hint-small">
                            {availableChannels.length > 0 
                              ? "Selecione o canal para comparar modelos" 
                              : "Conecte um bloco para ver os canais disponíveis"}
                          </small>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "statistical_features" ? (
                    <>
                      {/* Bloco Statistical Features */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.statisticalFeatures")}</legend>
                        <p className="config-hint">Extrai estatísticas básicas dos dados</p>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Todos os canais</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="vazio = todos"
                            />
                          )}
                        </div>
                        <div className="config-field">
                          <label className="config-label">Features a extrair</label>
                          <div className="chip-group">
                            {["max", "min", "mean", "std", "range", "variance", "median", "sum"].map((feat) => {
                              const currentFeatures = selectedNode.data.config?.features || ["max", "min", "mean", "std", "range"];
                              const isSelected = currentFeatures.includes(feat);
                              return (
                                <button
                                  key={feat}
                                  type="button"
                                  className={`chip ${isSelected ? "active" : ""}`}
                                  onClick={() => {
                                    const newFeatures = isSelected
                                      ? currentFeatures.filter(f => f !== feat)
                                      : [...currentFeatures, feat];
                                    updateNodeConfigField("features", newFeatures);
                                  }}
                                >
                                  {feat}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "temporal_features" ? (
                    <>
                      {/* Bloco Temporal Features */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.temporalFeatures")}</legend>
                        <p className="config-hint">Extrai características relacionadas ao tempo</p>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Todos os canais</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="vazio = todos"
                            />
                          )}
                        </div>
                        <div className="config-field">
                          <label className="config-label">Threshold (%)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.threshold_percent ?? 50}
                            onChange={(e) => updateNodeConfigField("threshold_percent", parseFloat(e.target.value) || 50)}
                            min="0"
                            max="100"
                          />
                          <small className="config-hint-small">Percentual do máximo para time_to_threshold</small>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Features a extrair</label>
                          <div className="chip-group">
                            {["time_to_max", "time_to_min", "time_to_threshold", "time_at_max", "time_at_min", "duration"].map((feat) => {
                              const currentFeatures = selectedNode.data.config?.features || ["time_to_max", "time_to_min", "time_to_threshold"];
                              const isSelected = currentFeatures.includes(feat);
                              return (
                                <button
                                  key={feat}
                                  type="button"
                                  className={`chip ${isSelected ? "active" : ""}`}
                                  onClick={() => {
                                    const newFeatures = isSelected
                                      ? currentFeatures.filter(f => f !== feat)
                                      : [...currentFeatures, feat];
                                    updateNodeConfigField("features", newFeatures);
                                  }}
                                >
                                  {feat.replace(/_/g, " ")}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "shape_features" ? (
                    <>
                      {/* Bloco Shape Features */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.shapeFeatures")}</legend>
                        <p className="config-hint">Extrai características da geometria da curva</p>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Todos os canais</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="vazio = todos"
                            />
                          )}
                        </div>
                        <div className="config-field">
                          <label className="config-label">Janela de suavização</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.smoothing_window ?? 5}
                            onChange={(e) => updateNodeConfigField("smoothing_window", parseInt(e.target.value) || 5)}
                            min="1"
                            max="20"
                          />
                          <small className="config-hint-small">Suaviza derivadas para evitar ruído</small>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Features a extrair</label>
                          <div className="chip-group">
                            {["inflection_points", "peaks", "valleys", "zero_crossings", "slope_start", "slope_end", "auc", "max_derivative"].map((feat) => {
                              const currentFeatures = selectedNode.data.config?.features || ["inflection_points", "peaks", "auc"];
                              const isSelected = currentFeatures.includes(feat);
                              return (
                                <button
                                  key={feat}
                                  type="button"
                                  className={`chip ${isSelected ? "active" : ""}`}
                                  onClick={() => {
                                    const newFeatures = isSelected
                                      ? currentFeatures.filter(f => f !== feat)
                                      : [...currentFeatures, feat];
                                    updateNodeConfigField("features", newFeatures);
                                  }}
                                >
                                  {feat.replace(/_/g, " ")}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "growth_features" ? (
                    <>
                      {/* Bloco Growth Features */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.growthFeatures")}</legend>
                        <p className="config-hint">Extrai parâmetros microbiológicos (numéricos) das curvas</p>
                        
                        {/* Seleção de canais via chips */}
                        <div className="config-field">
                          <label className="config-label">Canais a processar</label>
                          {availableFitChannels.length > 0 ? (
                            <>
                              <div className="chip-group">
                                {(() => {
                                  const selectedChannels = selectedNode.data.config?.selected_channels || [];
                                  const normalizedSelected = Array.isArray(selectedChannels) ? selectedChannels : 
                                    (typeof selectedChannels === 'string' && selectedChannels.trim() 
                                      ? selectedChannels.split(',').map(s => s.trim()).filter(Boolean) 
                                      : []);
                                  const isAllSelected = normalizedSelected.length === 0 || normalizedSelected.length === availableFitChannels.length;
                                  
                                  return (
                                    <>
                                      <button
                                        type="button"
                                        className={`chip chip-all ${isAllSelected ? "active" : ""}`}
                                        onClick={() => updateNodeConfigField("selected_channels", [])}
                                        title="Processar todos os canais"
                                      >
                                        {t("common.all")}
                                      </button>
                                      {availableFitChannels.map((chName) => {
                                        const isActive = isAllSelected ? false : normalizedSelected.includes(chName);
                                        return (
                                          <button
                                            key={chName}
                                            type="button"
                                            className={`chip ${isActive ? "active" : ""}`}
                                            onClick={() => {
                                              if (isAllSelected) {
                                                // Se "Todos" está selecionado, clicar em um canal seleciona só ele
                                                updateNodeConfigField("selected_channels", [chName]);
                                                return;
                                              }
                                              const nextSet = new Set(normalizedSelected);
                                              if (nextSet.has(chName)) nextSet.delete(chName);
                                              else nextSet.add(chName);
                                              // Se ficou vazio ou todos selecionados, volta para []
                                              const arr = Array.from(nextSet);
                                              updateNodeConfigField("selected_channels", arr.length === availableFitChannels.length ? [] : arr);
                                            }}
                                          >
                                            {chName}
                                          </button>
                                        );
                                      })}
                                    </>
                                  );
                                })()}
                              </div>
                              <small className="config-hint-small">
                                Selecione os canais ou clique em "Todos"
                              </small>
                            </>
                          ) : (
                            <p className="config-hint" style={{ marginTop: 4 }}>
                              Conecte a um curve_fit ou load_data para ver os canais disponíveis
                            </p>
                          )}
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Features a extrair</label>
                          <div className="chip-group">
                            {["lag_time", "growth_rate", "asymptote", "doubling_time", "inflection_time", "y0", "r_squared", "rmse", "aic"].map((feat) => {
                              const currentFeatures = selectedNode.data.config?.features || ["lag_time", "growth_rate", "asymptote", "r_squared"];
                              const isSelected = currentFeatures.includes(feat);
                              return (
                                <button
                                  key={feat}
                                  type="button"
                                  className={`chip ${isSelected ? "active" : ""}`}
                                  onClick={() => {
                                    const newFeatures = isSelected
                                      ? currentFeatures.filter(f => f !== feat)
                                      : [...currentFeatures, feat];
                                    updateNodeConfigField("features", newFeatures);
                                  }}
                                >
                                  {feat === "growth_rate" ? "µmax" : feat === "lag_time" ? "λ (lag)" : feat.replace(/_/g, " ")}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "features_merge" ? (
                    <>
                      {/* Bloco Features Merge */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.combineFeatures")}</legend>
                        <p className="config-hint">Combina múltiplos blocos de features em um único output</p>
                        
                        <div className="config-info">
                          <pre className="gate-diagram">{`features_a  \\\\\nfeatures_b ---- MERGE ---- features\nfeatures_c  ///\nfeatures_d  \\\\\\n`}</pre>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.featuresInputsLabel")}</label>
                          {(() => {
                            const configured = selectedNode.data.config?.features_inputs;
                            const baseInputs = Array.isArray(configured) && configured.length
                              ? configured
                              : (selectedNode.data.dataInputs || []).filter((k) => String(k).startsWith("features_"));
                            const inputs = baseInputs.length ? baseInputs : ["features_a"];

                            return (
                              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                  {inputs.map((key) => (
                                    <span
                                      key={key}
                                      className="chip active"
                                      style={{ cursor: "default", display: "inline-flex", alignItems: "center", gap: 6 }}
                                    >
                                      <code>{key}</code>
                                      <button
                                        type="button"
                                        className="btn btn-tertiary"
                                        onClick={() => {
                                          if (inputs.length <= 1) return;
                                          setFeaturesMergeInputs(selectedNode.id, inputs.filter((k) => k !== key));
                                        }}
                                        title={t("actions.remove")}
                                        aria-label={t("actions.remove")}
                                        style={{
                                          padding: "0 6px",
                                          height: 20,
                                          lineHeight: "18px",
                                          fontSize: 12,
                                          borderRadius: 999,
                                        }}
                                      >
                                        ×
                                      </button>
                                    </span>
                                  ))}
                                </div>
                                <div>
                                  <button
                                    type="button"
                                    className="btn btn-secondary"
                                    onClick={() => {
                                      const used = new Set(inputs.map(String));
                                      let next = null;
                                      for (let i = 0; i < 26; i += 1) {
                                        const cand = `features_${String.fromCharCode(97 + i)}`;
                                        if (!used.has(cand)) {
                                          next = cand;
                                          break;
                                        }
                                      }
                                      if (!next) {
                                        next = `features_${inputs.length + 1}`;
                                      }
                                      setFeaturesMergeInputs(selectedNode.id, [...inputs, next]);
                                    }}
                                  >
                                    + {t("configuration.addInput")}
                                  </button>
                                  <div className="config-hint-small" style={{ marginTop: 6 }}>
                                    {t("configuration.featuresMergeInputsHint")}
                                  </div>
                                </div>
                              </div>
                            );
                          })()}
                        </div>
                         
                        <div className="config-field">
                          <label className="config-label">Modo de Merge</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.merge_mode || "flat"}
                            onChange={(e) => updateNodeConfigField("merge_mode", e.target.value)}
                          >
                            <option value="flat">Flat (combina por canal)</option>
                            <option value="grouped">Grouped (agrupa por fonte)</option>
                          </select>
                          <small className="config-hint-small">
                            Flat: combina todas features no mesmo canal<br/>
                            Grouped: mantém separação por bloco fonte
                          </small>
                        </div>
                        
                        <div className="config-hint" style={{marginTop: '12px'}}>
                          <strong>{t("configuration.connectFeaturesTitle")}</strong>
                          <ul style={{fontSize: '12px', margin: '4px 0 0 16px'}}>
                            <li><code>features_a</code> ← statistical_features</li>
                            <li><code>features_b</code> ← temporal_features</li>
                            <li><code>features_c</code> ← shape_features</li>
                            <li><code>features_d</code> ← growth_features</li>
                          </ul>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar features combinadas (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName?.endsWith("_conversion") ? (
                    <>
                      {/* Blocos de Conversão Espectral Individual */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.spectralConversion")}</legend>
                        <div className="config-info">
                          <strong>Espaço de cor:</strong> {selectedNode.data.blockName.replace("_conversion", "").toUpperCase()}
                        </div>

                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.calculate_matrix ?? true)} onChange={(e) => updateNodeConfigField("calculate_matrix", e.target.checked)} />
                            <span>Usar matriz de conversão (referência)</span>
                          </label>
                        </div>
                        
                        <div className="config-hint">
                          <small>
                            {t("configuration.defaultsInfoTitle")}:<br />
                            • Turbidimetria/Nefelometria: adaptação sim, luminosidade sim<br />
                            • Fluorescência: adaptação não, luminosidade não
                          </small>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_chromatic_adaptation ?? true)} onChange={(e) => updateNodeConfigField("apply_chromatic_adaptation", e.target.checked)} />
                            <span>Adaptação cromática (Bradford)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_luminosity_correction ?? true)} onChange={(e) => updateNodeConfigField("apply_luminosity_correction", e.target.checked)} />
                            <span>Correção de luminosidade</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_gamma)} onChange={(e) => updateNodeConfigField("apply_gamma", e.target.checked)} />
                            <span>Correção Gamma (sRGB)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.auto_exposure)} onChange={(e) => updateNodeConfigField("auto_exposure", e.target.checked)} />
                            <span>Auto Exposure</span>
                          </label>
                        </div>
                        {(selectedNode.data.blockName === "hsv_conversion" || selectedNode.data.blockName === "hsb_conversion") && (
                          <div className="config-field">
                            <label className="config-toggle">
                              <input type="checkbox" checked={Boolean(selectedNode.data.config?.return_hue_unwrapped ?? true)} onChange={(e) => updateNodeConfigField("return_hue_unwrapped", e.target.checked)} />
                              <span>Hue Unwrapped (contínuo)</span>
                            </label>
                          </div>
                        )}
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar Gráficos</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "spectral_conversion" ? (
                    <>
                      {/* Bloco Spectral Conversion */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.spectralConversion")}</legend>
                        <div className="config-field">
                          <label className="config-label">Espaço de Cor Alvo</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.target_color_space || "LAB"}
                            onChange={(e) => updateNodeConfigField("target_color_space", e.target.value)}
                          >
                            <option value="XYZ">XYZ</option>
                            <option value="RGB">RGB</option>
                            <option value="LAB">LAB (CIE L*a*b*)</option>
                            <option value="HSV">HSV</option>
                            <option value="HSB">HSB</option>
                            <option value="CMYK">CMYK</option>
                            <option value="xyY">xyY</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Tipo de Calibração</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.calibration_type || "standard"}
                            onChange={(e) => updateNodeConfigField("calibration_type", e.target.value)}
                          >
                            <option value="standard">Standard</option>
                            <option value="custom">Custom</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_chromatic_adaptation ?? true)} onChange={(e) => updateNodeConfigField("apply_chromatic_adaptation", e.target.checked)} />
                            <span>Aplicar adaptação cromática</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_luminosity_correction ?? true)} onChange={(e) => updateNodeConfigField("apply_luminosity_correction", e.target.checked)} />
                            <span>Aplicar correção de luminosidade</span>
                          </label>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "amplitude_detector" ? (
                    <>
                      {/* Bloco Amplitude Detector */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.amplitudeDetector")}</legend>
                        <p className="config-hint">Detecta crescimento pela amplitude relativa (max-min) em todos os canais</p>
                        <div className="config-field">
                          <label className="config-label">Amplitude mínima (%)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.min_amplitude_percent ?? 5.0}
                            onChange={(e) => updateNodeConfigField("min_amplitude_percent", parseFloat(e.target.value) || 5.0)}
                            min="0.1"
                            max="100"
                            step="0.5"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Direção esperada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.expected_direction || "auto"}
                            onChange={(e) => updateNodeConfigField("expected_direction", e.target.value)}
                          >
                            <option value="auto">Automático</option>
                            <option value="increasing">Crescente</option>
                            <option value="decreasing">Decrescente</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "derivative_detector" ? (
                    <>
                      {/* Bloco Derivative Detector */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.derivativeDetector")}</legend>
                        <p className="config-hint">Detecta crescimento pela taxa de variação em todos os canais</p>
                        <div className="config-field">
                          <label className="config-label">Amplitude mínima (%)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.min_amplitude_percent ?? 5.0}
                            onChange={(e) => updateNodeConfigField("min_amplitude_percent", parseFloat(e.target.value) || 5.0)}
                            min="0.1"
                            max="100"
                            step="0.5"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Direção esperada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.expected_direction || "auto"}
                            onChange={(e) => updateNodeConfigField("expected_direction", e.target.value)}
                          >
                            <option value="auto">Automático</option>
                            <option value="increasing">Crescente</option>
                            <option value="decreasing">Decrescente</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ratio_detector" ? (
                    <>
                      {/* Bloco Ratio Detector */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.ratioDetector")}</legend>
                        <p className="config-hint">Detecta crescimento pela razão início/fim em todos os canais</p>
                        <div className="config-field">
                          <label className="config-label">Razão mínima (max/min)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.min_growth_ratio ?? 1.2}
                            onChange={(e) => updateNodeConfigField("min_growth_ratio", parseFloat(e.target.value) || 1.2)}
                            min="1.0"
                            max="10"
                            step="0.1"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Direção esperada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.expected_direction || "auto"}
                            onChange={(e) => updateNodeConfigField("expected_direction", e.target.value)}
                          >
                            <option value="auto">Automático</option>
                            <option value="increasing">Crescente</option>
                            <option value="decreasing">Decrescente</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "boolean_extractor" ? (
                    <>
                      {/* Bloco Boolean Extractor */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.booleanExtractor")}</legend>
                        <p className="config-hint">Extrai um booleano de <strong>source_data</strong> e passa <strong>sensor_data</strong> adiante</p>
                        <div className="config-field">
                          <label className="config-label">Caminho do campo (em source_data)</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.field_path || "has_any_growth"}
                            onChange={(e) => updateNodeConfigField("field_path", e.target.value)}
                            placeholder="ex: has_any_growth"
                          />
                          <small className="config-hint">Use ponto para campos aninhados: channel_results.R.has_growth</small>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Valor padrão (se não encontrar)</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.default_value === true ? "true" : "false"}
                            onChange={(e) => updateNodeConfigField("default_value", e.target.value === "true")}
                          >
                            <option value="false">false</option>
                            <option value="true">true</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "value_in_list" ? (
                    <>
                      {/* Bloco Value In List */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.valueInList")}</legend>
                        <p className="config-hint">{t("configuration.valueInListHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.allowedValues")}</label>
                          <div className="chip-group" style={{ marginBottom: 8 }}>
                            {(() => {
                              const current = Array.isArray(selectedNode.data.config?.allowed_values)
                                ? selectedNode.data.config.allowed_values
                                : [];
                              if (!current.length) {
                                return <small className="config-hint-small">{t("configuration.noValuesConfigured")}</small>;
                              }
                              return current.map((val) => (
                                <span
                                  key={String(val)}
                                  className="chip active"
                                  style={{ cursor: "default", display: "inline-flex", alignItems: "center", gap: 6 }}
                                >
                                  <code>{String(val)}</code>
                                  <button
                                    type="button"
                                    className="btn btn-tertiary"
                                    onClick={() => {
                                      const next = current.filter((v) => String(v) !== String(val));
                                      updateNodeConfigField("allowed_values", next);
                                    }}
                                    title={t("actions.remove")}
                                    aria-label={t("actions.remove")}
                                    style={{
                                      padding: "0 6px",
                                      height: 20,
                                      lineHeight: "18px",
                                      fontSize: 12,
                                      borderRadius: 999,
                                    }}
                                  >
                                    ×
                                  </button>
                                </span>
                              ));
                            })()}
                          </div>

                          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                            <input
                              type="text"
                              className="config-input"
                              value={valueInListDraft}
                              onChange={(e) => setValueInListDraft(e.target.value)}
                              placeholder={t("configuration.valueToAdd")}
                            />
                            <button
                              type="button"
                              className="btn btn-secondary"
                              onClick={() => {
                                const nextValue = String(valueInListDraft || "").trim();
                                if (!nextValue) return;
                                const current = Array.isArray(selectedNode.data.config?.allowed_values)
                                  ? selectedNode.data.config.allowed_values.map(String)
                                  : [];
                                if (current.includes(nextValue)) {
                                  setValueInListDraft("");
                                  return;
                                }
                                updateNodeConfigField("allowed_values", [...current, nextValue]);
                                setValueInListDraft("");
                              }}
                            >
                              + {t("configuration.addInput")}
                            </button>
                          </div>

                          <div className="config-hint-small" style={{ marginTop: 6 }}>
                            {t("configuration.valueInListConnectHint")}
                          </div>
                        </div>

                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.invert)}
                              onChange={(e) => updateNodeConfigField("invert", e.target.checked)}
                            />
                            <span>{t("configuration.invertMatch")}</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.case_sensitive)}
                              onChange={(e) => updateNodeConfigField("case_sensitive", e.target.checked)}
                            />
                            <span>{t("configuration.caseSensitive")}</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={selectedNode.data.config?.trim !== false}
                              onChange={(e) => updateNodeConfigField("trim", e.target.checked)}
                            />
                            <span>{t("configuration.trimValues")}</span>
                          </label>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "numeric_compare" ? (
                    <>
                      {/* Bloco Numeric Compare */}
                      <fieldset className="config-group">
                        <legend>Comparação numérica</legend>
                        <p className="config-hint">Compara um valor numérico com um threshold. Útil para verificar se diluição != 1, por exemplo.</p>

                        <div className="config-field">
                          <label className="config-label">Operador</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.operator || "!="}
                            onChange={(e) => updateNodeConfigField("operator", e.target.value)}
                          >
                            <option value="==">Igual a (==)</option>
                            <option value="!=">Diferente de (!=)</option>
                            <option value=">">Maior que (&gt;)</option>
                            <option value=">=">Maior ou igual (&gt;=)</option>
                            <option value="<">Menor que (&lt;)</option>
                            <option value="<=">Menor ou igual (&lt;=)</option>
                          </select>
                        </div>

                        <div className="config-field">
                          <label className="config-label">Threshold (valor de comparação)</label>
                          <input
                            type="number"
                            className="config-input"
                            step="any"
                            value={selectedNode.data.config?.threshold ?? 1}
                            onChange={(e) => updateNodeConfigField("threshold", parseFloat(e.target.value) || 0)}
                          />
                          <small className="config-hint-small">
                            Exemplo: threshold=1 com operador "!=" detecta se diluição é diferente de 1
                          </small>
                        </div>

                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.invert)}
                              onChange={(e) => updateNodeConfigField("invert", e.target.checked)}
                            />
                            <span>Inverter resultado</span>
                          </label>
                        </div>

                        <div className="config-info" style={{marginTop: 8}}>
                          <pre className="gate-diagram">{`dilution_factor ---- CMP ---- condition\n                     ↑\n               threshold=${selectedNode.data.config?.threshold ?? 1}\n               operator="${selectedNode.data.config?.operator || "!="}"                     `}</pre>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "condition_gate" ? (
                    <>
                      {/* Bloco Condition Gate */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.conditionalGate")}</legend>
                        <p className="config-hint">Passa os dados somente se a condição bater com o valor esperado</p>
                        <div className="config-field">
                          <label className="config-label">Passar dados quando condição for:</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.pass_when === false ? "false" : "true"}
                            onChange={(e) => updateNodeConfigField("pass_when", e.target.value === "true")}
                          >
                            <option value="true">{t("options.trueCondition")}</option>
                            <option value="false">{t("options.falseCondition")}</option>
                          </select>
                        </div>
                        <div className="config-info">
                          <pre className="gate-diagram">{`data -----\\\\\n          ---- GATE ---- data (ou inativo)\ncondition --//\n`}</pre>
                          <p className="config-hint" style={{marginTop: '8px'}}>
                            <strong>Exemplo:</strong><br/>
                            • pass_when=TRUE + condição=true → passa dados<br/>
                            • pass_when=TRUE + condição=false → dados inativos<br/>
                            • pass_when=FALSE + condição=false → passa dados<br/>
                            • pass_when=FALSE + condição=true → dados inativos
                          </p>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "and_gate" ? (
                    <>
                      {/* Bloco AND Gate */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.andGate")}</legend>
                        <p className="config-hint">Retorna <strong>true</strong> se AMBAS condições de entrada forem true</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`condition_a --\\\\\n              ---- AND ---- result\ncondition_b --//\n`}</pre>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "or_gate" ? (
                    <>
                      {/* Bloco OR Gate */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.orGate")}</legend>
                        <p className="config-hint">Retorna <strong>true</strong> se PELO MENOS UMA condição de entrada for true</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`condition_a --\\\\\n              ---- OR ---- result\ncondition_b --//\n`}</pre>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "not_gate" ? (
                    <>
                      {/* Bloco NOT Gate */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.notGate")}</legend>
                        <p className="config-hint">Inverte o valor booleano: true→false, false→true</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`condition ---- NOT ---- result\n`}</pre>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "condition_branch" ? (
                    <>
                      {/* Bloco Condition Branch */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.conditionalBranch")}</legend>
                        <p className="config-hint">Direciona os dados para uma das duas saídas baseado na condição</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`              /-- data_if_true (se condition=true)\ndata + condition --|\n              \\\\-- data_if_false (se condition=false)\n`}</pre>
                        </div>
                        <p className="config-hint" style={{ marginTop: 8 }}>{t("configuration.inactiveOutputHint")}</p>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "merge" ? (
                    <>
                      {/* Bloco Merge */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.flowMerge")}</legend>
                        <p className="config-hint">Junta dois fluxos condicionais - passa adiante o que estiver ativo</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`data_a --\\\\\n         ---- MERGE ---- data\ndata_b --//\n`}</pre>
                        </div>
                        <p className="config-hint" style={{ marginTop: 8 }}>{t("configuration.flowMergeHint")}</p>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "label" ? (
                    <>
                      {/* Bloco Label */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.labelTag")}</legend>
                        <p className="config-hint">Adiciona uma tag/identificador aos dados para agrupamento</p>
                        <div className="config-info">
                          <pre className="gate-diagram" style={{fontSize: '11px'}}>{`experiment_data --\\\\\n                --[ label ]--+-- experiment_data (com tag)\nexperiment -------//          \\\\-- experiment (com tag)\n`}</pre>
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Identificador (Label)</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.label || ""}
                            onChange={(e) => updateNodeConfigField("label", e.target.value)}
                            placeholder="ex: ecoli, coliformes, salmonella"
                          />
                          <small className="config-hint-small">Esta tag será propagada através dos blocos seguintes</small>
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label" htmlFor={`${selectedNode.id}-label_color`}>
                            {t("configuration.labelColor")}
                          </label>
                          <div className="config-color-row">
                            <input
                              id={`${selectedNode.id}-label_color`}
                              type="color"
                              className="config-color-input"
                              value={
                                typeof selectedNode.data.config?.label_color === "string" &&
                                /^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(selectedNode.data.config.label_color.trim())
                                  ? selectedNode.data.config.label_color.trim()
                                  : "#6366f1"
                              }
                              onChange={(e) => updateNodeConfigField("label_color", e.target.value)}
                            />
                            <input
                              type="text"
                              className="config-input config-color-text"
                              value={typeof selectedNode.data.config?.label_color === "string" ? selectedNode.data.config.label_color : ""}
                              placeholder="#RRGGBB"
                              onChange={(e) => updateNodeConfigField("label_color", e.target.value)}
                            />
                            <button
                              type="button"
                              className="config-color-clear"
                              onClick={() => updateNodeConfigField("label_color", undefined)}
                            >
                              {t("actions.clear")}
                            </button>
                          </div>
                          <small className="config-hint-small">{t("configuration.labelColorHint")}</small>
                        </div>

                        <p className="config-hint" style={{marginTop: 8, background: '#d4edda', padding: 8, borderRadius: 4, border: '1px solid #28a745'}}>
                          <strong>{t("configuration.positionLabel")}:</strong> Entre experiment_fetch e xxx_extraction<br/>
                          <code style={{fontSize: 10}}>[experiment_fetch] → [label] → [fluorescence_extraction] → ...</code>
                        </p>
                        
                        <p className="config-hint" style={{marginTop: 8, background: '#e7f3ff', padding: 8, borderRadius: 4}}>
                          <strong>{t("configuration.connectionsLabel")}:</strong><br/>
                          • <code>experiment_data</code> do fetch → entrada <code>experiment_data</code> do label<br/>
                          • Saída <code>experiment_data</code> do label → xxx_extraction
                        </p>
                        
                        <p className="config-hint" style={{marginTop: 8}}>
                          <strong>Exemplo de saída do response_builder:</strong>
                          <code style={{display: 'block', marginTop: 4, fontSize: 11, background: '#f5f5f5', padding: 4}}>
                            {`{
  "presence_ecoli": true,
  "predict_nmp_ecoli": 2.99,
  "presence_coliformes": true,
  "predict_nmp_coliformes": 1.66
}`}
                          </code>
                        </p>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_inference" ? (
                    <>
                      {/* Bloco ML Inference */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlInference")}</legend>
                        <p className="config-hint">Executa predição usando modelos ONNX treinados</p>
                        
                        <div className="config-field">
                          <label className="config-label">Arquivo do modelo (.onnx)</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                          <small className="config-hint-small">Caminho do arquivo do modelo no servidor</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">Arquivo do scaler (.joblib)</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">Obrigatório quando model_path estiver preenchido</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">Unidade de saída</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.output_unit || ""}
                            onChange={(e) => updateNodeConfigField("output_unit", e.target.value)}
                            placeholder="NMP/100mL ou UFC/mL"
                          />
                          <small className="config-hint-small">Será propagada para os resultados e para o response_builder</small>
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Feature de Entrada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.input_feature || "growth_rate"}
                            onChange={(e) => updateNodeConfigField("input_feature", e.target.value)}
                          >
                            <optgroup label="Growth Features (do curve_fit)">
                              <option value="growth_rate">µmax (growth_rate)</option>
                              <option value="asymptote">Assíntota (A)</option>
                              <option value="lag_time">Lag Time (λ)</option>
                              <option value="inflection_time">Tempo de Inflexão (T)</option>
                              <option value="doubling_time">Tempo de Duplicação</option>
                            </optgroup>
                            <optgroup label="Statistical Features">
                              <option value="max">Máximo</option>
                              <option value="min">Mínimo</option>
                              <option value="mean">Média</option>
                              <option value="std">Desvio Padrão</option>
                              <option value="range">Range (max-min)</option>
                            </optgroup>
                            <optgroup label="Shape Features">
                              <option value="auc">Área sob a Curva (AUC)</option>
                              <option value="slope_start">Inclinação Inicial</option>
                              <option value="slope_end">Inclinação Final</option>
                              <option value="max_derivative">Derivada Máxima</option>
                            </optgroup>
                            <optgroup label="Temporal Features">
                              <option value="time_to_max">Tempo até Máximo</option>
                              <option value="time_to_threshold">Tempo até Threshold</option>
                              <option value="duration">Duração Total</option>
                            </optgroup>
                          </select>
                          <small className="config-hint-small">Valor que será enviado ao modelo</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.channel || ""}
                            onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            placeholder="vazio = primeiro disponível"
                          />
                          <small className="config-hint-small">Canal das features para usar (ex: R, f1, X)</small>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group">
                        <legend>{t("configuration.output")}</legend>
                        <div className="config-info">
                          <p><strong>Saída:</strong> <code>prediction</code></p>
                          <ul style={{fontSize: '12px', marginLeft: '16px', marginTop: '4px'}}>
                            <li><code>value</code> - Valor predito</li>
                            <li><code>unit</code> - Unidade (NMP/100mL ou UFC/mL)</li>
                            <li><code>input_feature</code> - Feature usada</li>
                            <li><code>input_value</code> - Valor da feature</li>
                          </ul>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar detalhes da inferência (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_inference_series" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlInferenceSeries")}</legend>
                        <p className="config-hint">{t("configuration.mlSeriesHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.modelPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.scalerPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">{t("configuration.scalerOptionalHint")}</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.channel")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.channel || ""}
                            onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            placeholder={t("configuration.channelPlaceholder")}
                          />
                          <small className="config-hint-small">{t("configuration.channelHint")}</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.inputLayout")}</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.input_layout || "sequence"}
                            onChange={(e) => updateNodeConfigField("input_layout", e.target.value)}
                          >
                            <option value="sequence">{t("configuration.layouts.sequence")}</option>
                            <option value="flat">{t("configuration.layouts.flat")}</option>
                            <option value="channels_first">{t("configuration.layouts.channelsFirst")}</option>
                          </select>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.maxLength")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.max_length ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("max_length", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="ex: 2048"
                              min="1"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.align")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.align || "end"}
                              onChange={(e) => updateNodeConfigField("align", e.target.value)}
                            >
                              <option value="end">{t("configuration.alignOptions.end")}</option>
                              <option value="start">{t("configuration.alignOptions.start")}</option>
                            </select>
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group">
                        <legend>{t("configuration.output")}</legend>
                        <div className="config-field">
                          <label className="config-label">{t("configuration.outputUnit")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.output_unit || ""}
                            onChange={(e) => updateNodeConfigField("output_unit", e.target.value)}
                            placeholder="ex: NMP/100mL, UFC/mL"
                          />
                          <small className="config-hint-small">{t("configuration.outputUnitHint")}</small>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_inference_multichannel" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlInferenceMultichannel")}</legend>
                        <p className="config-hint">{t("configuration.mlMultichannelHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.modelPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.scalerPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">{t("configuration.scalerOptionalHint")}</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.channels")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={
                              Array.isArray(selectedNode.data.config?.channels)
                                ? selectedNode.data.config.channels.join(", ")
                                : selectedNode.data.config?.channels || ""
                            }
                            onChange={(e) => {
                              const raw = e.target.value;
                              const parsed = raw
                                .split(",")
                                .map((s) => s.trim())
                                .filter(Boolean);
                              updateNodeConfigField("channels", parsed.length ? parsed : raw);
                            }}
                            placeholder="ex: f1, f2, clear"
                          />
                          <small className="config-hint-small">{t("configuration.channelsHint")}</small>
                        </div>
                        {availableChannels.length > 0 && (() => {
                          const selectedList = Array.isArray(selectedNode.data.config?.channels)
                            ? selectedNode.data.config.channels.map(String).filter(Boolean)
                            : typeof selectedNode.data.config?.channels === "string"
                              ? selectedNode.data.config.channels
                                  .split(",")
                                  .map((v) => v.trim())
                                  .filter(Boolean)
                              : [];
                          const isAll =
                            selectedList.length === 0 || selectedList.length === availableChannels.length;
                          return (
                            <>
                              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
                                <span style={{ fontSize: 11, color: "var(--color-gray-500)" }}>
                                  {t("configuration.availableChannels", { count: availableChannels.length })}
                                </span>
                              </div>
                              <div className="chip-group" style={{ marginTop: 6 }}>
                                <button
                                  type="button"
                                  className={`chip chip-all ${isAll ? "active" : ""}`}
                                  onClick={() => updateNodeConfigField("channels", [])}
                                  title={t("configuration.allChannels")}
                                >
                                  {t("common.all")}
                                </button>
                                {availableChannels.map((chName) => {
                                  const isActive = isAll ? false : selectedList.includes(chName);
                                  return (
                                    <button
                                      key={chName}
                                      type="button"
                                      className={`chip ${isActive ? "active" : ""}`}
                                      onClick={() => {
                                        if (isAll) {
                                          updateNodeConfigField("channels", [chName]);
                                          return;
                                        }
                                        const nextSet = new Set(selectedList);
                                        if (nextSet.has(chName)) nextSet.delete(chName);
                                        else nextSet.add(chName);
                                        updateNodeConfigField("channels", Array.from(nextSet));
                                      }}
                                    >
                                      {chName}
                                    </button>
                                  );
                                })}
                              </div>
                            </>
                          );
                        })()}

                        <div className="config-field">
                          <label className="config-label">{t("configuration.inputLayout")}</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.input_layout || "time_channels"}
                            onChange={(e) => updateNodeConfigField("input_layout", e.target.value)}
                          >
                            <option value="time_channels">{t("configuration.layouts.timeChannels")}</option>
                            <option value="channels_time">{t("configuration.layouts.channelsTime")}</option>
                          </select>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.maxLength")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.max_length ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("max_length", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="ex: 2048"
                              min="1"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.align")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.align || "end"}
                              onChange={(e) => updateNodeConfigField("align", e.target.value)}
                            >
                              <option value="end">{t("configuration.alignOptions.end")}</option>
                              <option value="start">{t("configuration.alignOptions.start")}</option>
                            </select>
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group">
                        <legend>{t("configuration.output")}</legend>
                        <div className="config-field">
                          <label className="config-label">{t("configuration.outputUnit")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.output_unit || ""}
                            onChange={(e) => updateNodeConfigField("output_unit", e.target.value)}
                            placeholder="ex: NMP/100mL, UFC/mL"
                          />
                          <small className="config-hint-small">{t("configuration.outputUnitHint")}</small>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_transform_series" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlTransformSeries")}</legend>
                        <p className="config-hint">{t("configuration.mlTransformHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.modelPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.scalerPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">{t("configuration.scalerOptionalHint")}</small>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.channel")}</label>
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder={t("configuration.channelPlaceholder")}
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.outputChannel")}</label>
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.output_channel || "ml"}
                              onChange={(e) => updateNodeConfigField("output_channel", e.target.value)}
                              placeholder="ml"
                            />
                            <small className="config-hint-small">{t("configuration.outputChannelHint")}</small>
                          </div>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.inputLayout")}</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.input_layout || "sequence"}
                            onChange={(e) => updateNodeConfigField("input_layout", e.target.value)}
                          >
                            <option value="sequence">{t("configuration.layouts.sequence")}</option>
                            <option value="flat">{t("configuration.layouts.flat")}</option>
                            <option value="channels_first">{t("configuration.layouts.channelsFirst")}</option>
                          </select>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.maxLength")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.max_length ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("max_length", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="ex: 2048"
                              min="1"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.align")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.align || "end"}
                              onChange={(e) => updateNodeConfigField("align", e.target.value)}
                            >
                              <option value="end">{t("configuration.alignOptions.end")}</option>
                              <option value="start">{t("configuration.alignOptions.start")}</option>
                            </select>
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_detector" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlDetector")}</legend>
                        <p className="config-hint">{t("configuration.mlDetectorHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.modelPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.scalerPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">{t("configuration.scalerOptionalHint")}</small>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.channel")}</label>
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder={t("configuration.channelPlaceholder")}
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.inputLayout")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.input_layout || "sequence"}
                              onChange={(e) => updateNodeConfigField("input_layout", e.target.value)}
                            >
                              <option value="sequence">{t("configuration.layouts.sequence")}</option>
                              <option value="flat">{t("configuration.layouts.flat")}</option>
                              <option value="channels_first">{t("configuration.layouts.channelsFirst")}</option>
                            </select>
                          </div>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.threshold")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.threshold ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("threshold", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0.5"
                              step="0.01"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.operator")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.operator || ">="}
                              onChange={(e) => updateNodeConfigField("operator", e.target.value)}
                            >
                              <option value=">=">≥</option>
                              <option value=">">&gt;</option>
                              <option value="<=">≤</option>
                              <option value="<">&lt;</option>
                            </select>
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.align")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.align || "end"}
                              onChange={(e) => updateNodeConfigField("align", e.target.value)}
                            >
                              <option value="end">{t("configuration.alignOptions.end")}</option>
                              <option value="start">{t("configuration.alignOptions.start")}</option>
                            </select>
                          </div>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.maxLength")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.max_length ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("max_length", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="ex: 2048"
                              min="1"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : ["response_builder", "response_pack"].includes(selectedNode?.data?.blockName) ? (
                    <>
                      {/* Bloco Response Builder / Response Pack */}
                      <fieldset className="config-group">
                        <legend>
                          {selectedNode?.data?.blockName === "response_pack"
                            ? t("configuration.responsePack")
                            : t("configuration.responseBuilder")}
                        </legend>
                        <p className="config-hint">
                          {selectedNode?.data?.blockName === "response_pack"
                            ? t("configuration.responsePackHint")
                            : "Monta o JSON de resposta final conectando múltiplas entradas"}
                        </p>
                        
                        <div className="config-info" style={{marginBottom: '12px'}}>
                          <pre className="gate-diagram" style={{fontSize: '11px'}}>{`input_1 (condition) --\\\\\ninput_2 (predict_nmp) --+-- response\ninput_3 (predict_ufc) --//\n`}</pre>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.responseInputsLabel")}</label>
                          {(() => {
                            const maxInputs = 8;
                            const currentKeys = (selectedNode?.data?.dataInputs || []).filter((k) => String(k).startsWith("input_"));
                            const configuredCount = Number(selectedNode?.data?.config?.inputs_count);
                            const count = Math.max(
                              1,
                              Math.min(maxInputs, Number.isFinite(configuredCount) ? configuredCount : (currentKeys.length || 1))
                            );
                            const keys = Array.from({ length: count }, (_, i) => `input_${i + 1}`);
                            return (
                              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                  {keys.map((key, idx) => (
                                    <span
                                      key={key}
                                      className="chip active"
                                      style={{ cursor: "default", display: "inline-flex", alignItems: "center", gap: 6 }}
                                    >
                                      <code>{key}</code>
                                      <button
                                        type="button"
                                        className="btn btn-tertiary"
                                        onClick={() => {
                                          if (count <= 1) return;
                                          setSequentialInputsCount(selectedNode.id, count - 1, { prefix: "input_", max: 8, configKey: "inputs_count" });
                                        }}
                                        title={t("actions.remove")}
                                        aria-label={t("actions.remove")}
                                        style={{
                                          padding: "0 6px",
                                          height: 20,
                                          lineHeight: "18px",
                                          fontSize: 12,
                                          borderRadius: 999,
                                          opacity: idx === count - 1 ? 1 : 0.65,
                                        }}
                                        disabled={idx !== count - 1 || count <= 1}
                                      >
                                        -
                                      </button>
                                    </span>
                                  ))}
                                </div>
                                <div>
                                  <button
                                    type="button"
                                    className="btn btn-secondary"
                                    onClick={() => {
                                      if (count >= maxInputs) return;
                                      setSequentialInputsCount(selectedNode.id, count + 1, { prefix: "input_", max: 8, configKey: "inputs_count" });
                                    }}
                                    disabled={count >= maxInputs}
                                  >
                                    + {t("configuration.addInput")}
                                  </button>
                                  <div className="config-hint-small" style={{ marginTop: 6 }}>
                                    {t("configuration.responseInputsHint")}
                                  </div>
                                </div>
                              </div>
                            );
                          })()}
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Mapeamentos de Campos</label>
                          <small className="config-hint-small" style={{display: 'block', marginBottom: '8px'}}>
                            Define como extrair valores de cada entrada
                          </small>
                          
                          {/* Editor de mapeamentos */}
                          {(() => {
                            const inputCount = Math.max(
                              1,
                              Math.min(
                                8,
                                Number.isFinite(Number(selectedNode?.data?.config?.inputs_count))
                                  ? Number(selectedNode?.data?.config?.inputs_count)
                                  : ((selectedNode?.data?.dataInputs || []).filter((k) => String(k).startsWith("input_")).length || 1)
                              )
                            );
                            const mappings = (() => {
                              try {
                                const val = selectedNode.data.config?.field_mappings;
                                if (Array.isArray(val)) return val;
                                return JSON.parse(val || "[]");
                              } catch { return []; }
                            })();
                            
                            return (
                              <div className="mappings-editor">
                                {mappings.map((m, idx) => (
                                  <div key={idx} className="mapping-row" style={{
                                    display: 'grid',
                                    gridTemplateColumns: '60px 1fr 80px 30px',
                                    gap: '4px',
                                    marginBottom: '4px',
                                    alignItems: 'center'
                                  }}>
                                    <select
                                      className="config-select"
                                      style={{padding: '4px', fontSize: '11px'}}
                                      value={m.input || 1}
                                      onChange={(e) => {
                                        const newMappings = [...mappings];
                                        newMappings[idx] = {...m, input: parseInt(e.target.value)};
                                        updateNodeConfigField("field_mappings", newMappings);
                                      }}
                                    >
                                      {Array.from({ length: inputCount }, (_, i) => i + 1).map((i) => (
                                        <option key={i} value={i}>input_{i}</option>
                                      ))}
                                    </select>
                                    <input
                                      type="text"
                                      className="config-input"
                                      style={{padding: '4px', fontSize: '11px'}}
                                      value={m.field || ""}
                                      onChange={(e) => {
                                        const newMappings = [...mappings];
                                        newMappings[idx] = {...m, field: e.target.value};
                                        updateNodeConfigField("field_mappings", newMappings);
                                      }}
                                      placeholder="nome_campo"
                                    />
                                    <input
                                      type="text"
                                      className="config-input"
                                      style={{padding: '4px', fontSize: '11px'}}
                                      value={m.path || ""}
                                      onChange={(e) => {
                                        const newMappings = [...mappings];
                                        newMappings[idx] = {...m, path: e.target.value};
                                        updateNodeConfigField("field_mappings", newMappings);
                                      }}
                                      placeholder="path"
                                    />
                                    <button
                                      type="button"
                                      className="btn-icon"
                                      style={{padding: '2px 6px', fontSize: '14px'}}
                                      onClick={() => {
                                        const newMappings = mappings.filter((_, i) => i !== idx);
                                        updateNodeConfigField("field_mappings", newMappings);
                                      }}
                                    >
                                      ×
                                    </button>
                                  </div>
                                ))}
                                
                                <button
                                  type="button"
                                  className="btn-secondary"
                                  style={{marginTop: '4px', padding: '4px 8px', fontSize: '11px'}}
                                  onClick={() => {
                                    const newMappings = [...mappings, {input: 1, field: "", path: ""}];
                                    updateNodeConfigField("field_mappings", newMappings);
                                  }}
                                >
                                  + Adicionar Campo
                                </button>
                              </div>
                            );
                          })()}
                          
                          <small className="config-hint-small" style={{marginTop: '8px', display: 'block'}}>
                            <strong>Paths comuns:</strong><br/>
                            • <code>.</code> ou vazio → valor direto<br/>
                            • <code>value</code> → prediction.value<br/>
                            • <code>success</code> → condition boolean
                          </small>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group">
                        <legend>{t("configuration.staticFields")}</legend>
                        <div className="config-field">
                          <label className="config-label">Campos fixos (JSON)</label>
                          <textarea
                            className="config-textarea"
                            style={{minHeight: '60px', fontFamily: 'monospace', fontSize: '11px'}}
                            value={(() => {
                              try {
                                const val = selectedNode.data.config?.static_fields;
                                if (typeof val === 'object') return JSON.stringify(val, null, 2);
                                return val || '{\n  "analysis_mode": "prediction"\n}';
                              } catch { return '{}'; }
                            })()}
                            onChange={(e) => {
                              try {
                                const parsed = JSON.parse(e.target.value);
                                updateNodeConfigField("static_fields", parsed);
                              } catch {
                                updateNodeConfigField("static_fields", e.target.value);
                              }
                            }}
                            placeholder='{"analysis_mode": "prediction"}'
                          />
                          <small className="config-hint-small">Campos que serão adicionados à resposta</small>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group">
                        <legend>{t("configuration.groupByLabel")}</legend>
                        <p className="config-hint">Se os dados vierem de blocos com label, agrupa automaticamente</p>
                        
                        <div className="config-field">
                          <label className="config-toggle">
                            <input 
                              type="checkbox" 
                              checked={selectedNode.data.config?.group_by_label !== false} 
                              onChange={(e) => updateNodeConfigField("group_by_label", e.target.checked)} 
                            />
                            <span>Agrupar resultados por label</span>
                          </label>
                          <small className="config-hint-small">Detecta automaticamente dados vindos de blocos label</small>
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Formato de Agrupamento</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.flatten_labels !== false ? "flat" : "nested"}
                            onChange={(e) => updateNodeConfigField("flatten_labels", e.target.value === "flat")}
                          >
                            <option value="flat">Prefixo (presence_ecoli, predict_nmp_ecoli)</option>
                            <option value="nested">Aninhado (ecoli: {"{"} presence, predict_nmp {"}"})</option>
                          </select>
                          <small className="config-hint-small">Como os campos com label aparecerão na resposta</small>
                        </div>
                        
                        <div className="config-info" style={{marginTop: '8px', background: '#f0f8ff', padding: '8px', borderRadius: '4px', fontSize: '11px'}}>
                          <strong>Exemplo (modo prefixo):</strong>
                          <code style={{display: 'block', marginTop: '4px', whiteSpace: 'pre', background: '#f5f5f5', padding: '4px'}}>
{`{
  "presence_ecoli": true,
  "predict_nmp_ecoli": 2.99,
  "presence_coliformes": true,
  "predict_nmp_coliformes": 1.66
}`}
                          </code>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar resposta montada (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "response_merge" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.responseMerge")}</legend>
                        <p className="config-hint">{t("configuration.responseMergeHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.responseInputsLabel")}</label>
                          {(() => {
                            const maxInputs = 8;
                            const currentKeys = (selectedNode?.data?.dataInputs || []).filter((k) => String(k).startsWith("input_"));
                            const configuredCount = Number(selectedNode?.data?.config?.inputs_count);
                            const count = Math.max(
                              1,
                              Math.min(maxInputs, Number.isFinite(configuredCount) ? configuredCount : (currentKeys.length || 1))
                            );
                            const keys = Array.from({ length: count }, (_, i) => `input_${i + 1}`);
                            return (
                              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                  {keys.map((key, idx) => (
                                    <span
                                      key={key}
                                      className="chip active"
                                      style={{ cursor: "default", display: "inline-flex", alignItems: "center", gap: 6 }}
                                    >
                                      <code>{key}</code>
                                      <button
                                        type="button"
                                        className="btn btn-tertiary"
                                        onClick={() => {
                                          if (count <= 1) return;
                                          setSequentialInputsCount(selectedNode.id, count - 1, { prefix: "input_", max: 8, configKey: "inputs_count" });
                                        }}
                                        title={t("actions.remove")}
                                        aria-label={t("actions.remove")}
                                        style={{
                                          padding: "0 6px",
                                          height: 20,
                                          lineHeight: "18px",
                                          fontSize: 12,
                                          borderRadius: 999,
                                          opacity: idx === count - 1 ? 1 : 0.65,
                                        }}
                                        disabled={idx !== count - 1 || count <= 1}
                                      >
                                        -
                                      </button>
                                    </span>
                                  ))}
                                </div>
                                <div>
                                  <button
                                    type="button"
                                    className="btn btn-secondary"
                                    onClick={() => {
                                      if (count >= maxInputs) return;
                                      setSequentialInputsCount(selectedNode.id, count + 1, { prefix: "input_", max: 8, configKey: "inputs_count" });
                                    }}
                                    disabled={count >= maxInputs}
                                  >
                                    + {t("configuration.addInput")}
                                  </button>
                                  <div className="config-hint-small" style={{ marginTop: 6 }}>
                                    {t("configuration.responseInputsHint")}
                                  </div>
                                </div>
                              </div>
                            );
                          })()}
                        </div>

                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.fail_if_multiple)}
                              onChange={(e) => updateNodeConfigField("fail_if_multiple", e.target.checked)}
                            />
                            <span>Falhar se mais de uma saída estiver preenchida</span>
                          </label>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar detalhes do merge (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : effectiveConfigInputs.length ? (
                    effectiveConfigInputs.map((field) => renderConfigField(field))
                  ) : (
                    <p className="no-config">Este bloco não possui configurações</p>
                  )}
                </div>

                {/* Editor JSON removido conforme solicitado (pesado) */}
              </>
            ) : selectedEdge ? (
              <div className="selected-edge-info">
                <div className="edge-info-header">
                  <span className="edge-icon" aria-hidden="true" />
                  <strong>{t("configuration.selectedConnection")}</strong>
                </div>
                <div className="edge-info-details">
                  <p><strong>{t("configuration.from")}:</strong> {selectedEdge.source}</p>
                  <p><strong>{t("configuration.to")}:</strong> {selectedEdge.target}</p>
                  {selectedEdge.sourceHandle && <p><strong>{t("configuration.output")}:</strong> {selectedEdge.sourceHandle}</p>}
                  {selectedEdge.targetHandle && <p><strong>{t("configuration.input")}:</strong> {selectedEdge.targetHandle}</p>}
                </div>
                <div className="edge-actions">
                  <button 
                    className="delete-edge-btn"
                    onClick={() => removeEdge(selectedEdge.id)}
                  >
                    {t("actions.removeConnection")}
                  </button>
                </div>
                <div className="edge-hints">
                  <p className="edge-hint">{t("actions.reconnectHint")}</p>
                  <p className="edge-hint"><kbd>Delete</kbd> / <kbd>Backspace</kbd> — {t("actions.deleteKeyHint")}</p>
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <span aria-hidden="true" />
                <p>{t("configuration.empty")}</p>
              </div>
            )}

          

            {error && (
              <div className="error-section">
                <strong>{t("configuration.errorTitle")}</strong>
                <p>{error}</p>
              </div>
            )}
              </div>
            </aside>
          </div>
        )}

      </main>

      {resultsModalOpen && (
        <div className="helper-modal" onClick={closeResultsModal} role="dialog" aria-modal="true">
          <div className="helper-modal-inner" onClick={(e) => e.stopPropagation()}>
            <div className="helper-modal-header">
              <div>
                <h4>{t("panels.results")}</h4>
              </div>
              <button className="helper-modal-close" type="button" onClick={closeResultsModal}>
                {t("actions.close")}
              </button>
            </div>
            <div style={{ marginTop: 12 }}>
              <ResultsPanel
                simulation={simulation}
                onGraphClick={openGraphModal}
                getStepLabel={getStepLabel}
                getStepFlowLabel={getStepFlowLabel}
                getStepFlowColor={getStepFlowColor}
              />
            </div>
          </div>
        </div>
      )}

      {helpModal.open && helpModal.block && (
        <div className="helper-modal" onClick={closeHelpModal} role="dialog" aria-modal="true">
          <div className="helper-modal-inner helper-modal-inner--help" onClick={(e) => e.stopPropagation()}>
            <div className="help-modal-header">
              <div className="help-modal-title">
                <div className="help-modal-icon" style={activeHelpModel?.color ? { background: activeHelpModel.color } : undefined}>
                  {activeHelpModel?.icon || "B"}
                </div>
                <div>
                  <div className="help-modal-title-text">{activeHelpModel?.title || helpModal.block.name}</div>
                  <div className="help-modal-subtitle">{activeHelpModel?.subtitle || helpModal.block.name}</div>
                </div>
              </div>
              <div className="help-modal-actions">
                <button className="btn" type="button" onClick={() => addBlockToCanvas(helpModal.block)}>
                  {t("actions.add")}
                </button>
                <button className="helper-modal-close" type="button" onClick={closeHelpModal}>
                  {t("actions.close")}
                </button>
              </div>
            </div>

            <div className="help-tabs" role="tablist" aria-label={t("helper.title")}>
              {[
                { id: "overview", label: t("helper.tabs.overview") },
                { id: "io", label: t("helper.tabs.io") },
                { id: "config", label: t("helper.tabs.config") },
                { id: "examples", label: t("helper.tabs.examples") },
                { id: "troubleshooting", label: t("helper.tabs.troubleshooting") },
              ].map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  className={`help-tab${helpTab === tab.id ? " active" : ""}`}
                  role="tab"
                  aria-selected={helpTab === tab.id}
                  onClick={() => setHelpTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {helpTab === "overview" && (
              <div className="help-content">
                <div className="help-card">
                  <div className="help-card-title">{t("helper.sections.what")}</div>
                  <div className="help-card-text">{activeHelpModel?.what || helpModal.block.description}</div>
                </div>

                {activeHelpModel?.when?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.when")}</div>
                    <ul className="help-list">
                      {activeHelpModel.when.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {activeHelpModel?.how?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.how")}</div>
                    <ol className="help-list">
                      {activeHelpModel.how.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ol>
                  </div>
                ) : null}

                {activeHelpModel?.tips?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.tips")}</div>
                    <ul className="help-list">
                      {activeHelpModel.tips.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            )}

            {helpTab === "io" && (
              <div className="help-content">
                <div className="help-card">
                  <div className="help-card-title">{t("helper.sections.inputs")}</div>
                  <ul className="schema-list">
                    {helpModal.block.data_inputs?.length ? (
                      helpModal.block.data_inputs.map((key) => (
                        <li key={key}>
                          <span className="schema-key">{key}</span>
                          <small>{helpModal.block.input_schema?.[key]?.type || "any"}</small>
                        </li>
                      ))
                    ) : (
                      <li className="empty">{t("summary.noneInput")}</li>
                    )}
                  </ul>
                </div>

                <div className="help-card">
                  <div className="help-card-title">{t("helper.sections.outputs")}</div>
                  <ul className="schema-list">
                    {helpModal.block.data_outputs?.length ? (
                      helpModal.block.data_outputs.map((key) => (
                        <li key={key}>
                          <span className="schema-key">{key}</span>
                          <small>{helpModal.block.output_schema?.[key]?.type || "any"}</small>
                        </li>
                      ))
                    ) : (
                      <li className="empty">{t("summary.noneOutput")}</li>
                    )}
                  </ul>
                </div>
              </div>
            )}

            {helpTab === "config" && (
              <div className="help-content">
                <div className="help-card">
                  <div className="help-card-title">{t("helper.sections.configs")}</div>
                  <ul className="schema-list">
                    {helpModal.block.config_inputs?.length ? (
                      helpModal.block.config_inputs.map((key) => (
                        <li key={key}>
                          <span className="schema-key">{key}</span>
                          <small>
                            {helpModal.block.input_schema?.[key]?.description || helpModal.block.input_schema?.[key]?.type || ""}
                          </small>
                        </li>
                      ))
                    ) : (
                      <li className="empty">{t("summary.noneConfig")}</li>
                    )}
                  </ul>
                </div>
              </div>
            )}

            {helpTab === "examples" && (
              <div className="help-content">
                {activeHelpModel?.examples?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.examples")}</div>
                    {activeHelpModel.examples.map((ex) => (
                      <pre className="help-diagram" key={ex}>
                        {ex}
                      </pre>
                    ))}
                  </div>
                ) : null}

                {helpModal.block.name === "signal_filters" && library.filters?.length > 0 && (
                  <div className="help-card">
                    <div className="help-card-title">{t("summary.availableFilters")}</div>
                    <p className="component-hint">{t("summary.componentHint")}</p>
                    <ul className="component-list">
                      {library.filters.map((f) => (
                        <li
                          key={f.name}
                          role="button"
                          tabIndex={0}
                          onClick={() => addBlockToCanvas(helpModal.block, { filter_type: f.name })}
                          onKeyDown={(e) => e.key === "Enter" && addBlockToCanvas(helpModal.block, { filter_type: f.name })}
                        >
                          <span>{f.name}</span>
                          <small>{f.description}</small>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {helpModal.block.name === "curve_fitting" && library.curve_models?.length > 0 && (
                  <div className="help-card">
                    <div className="help-card-title">{t("summary.availableModels")}</div>
                    <p className="component-hint">{t("summary.componentHint")}</p>
                    <ul className="component-list">
                      {library.curve_models.map((m) => (
                        <li
                          key={m.name}
                          role="button"
                          tabIndex={0}
                          onClick={() => addBlockToCanvas(helpModal.block, { model_type: m.name })}
                          onKeyDown={(e) => e.key === "Enter" && addBlockToCanvas(helpModal.block, { model_type: m.name })}
                        >
                          <span>{m.name}</span>
                          <small>{m.description}</small>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {helpModal.block.name === "feature_extraction" && library.feature_extractors?.length > 0 && (
                  <div className="help-card">
                    <div className="help-card-title">{t("summary.availableExtractors")}</div>
                    <p className="component-hint">{t("summary.componentHint")}</p>
                    <ul className="component-list">
                      {library.feature_extractors.map((e) => (
                        <li
                          key={e.name}
                          role="button"
                          tabIndex={0}
                          onClick={() => addBlockToCanvas(helpModal.block, { extractor_type: e.name })}
                          onKeyDown={(ev) => ev.key === "Enter" && addBlockToCanvas(helpModal.block, { extractor_type: e.name })}
                        >
                          <span>{e.name}</span>
                          <small>{e.description}</small>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {helpTab === "troubleshooting" && (
              <div className="help-content">
                {activeHelpModel?.errors?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.errors")}</div>
                    <ul className="help-list">
                      {activeHelpModel.errors.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.errors")}</div>
                    <div className="help-card-text">{t("results.enableDebugHint")}</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      <BlockResultsModal
        open={blockResultsModal.open}
        t={t}
        onClose={closeBlockResultsModal}
        stepId={blockResultsModal.stepId}
        getStepLabel={getStepLabel}
        simulation={simulation}
        onGraphClick={openGraphModal}
      />

      <GraphModal
        open={graphModalOpen}
        onClose={closeGraphModal}
        graphList={graphList}
        graphIndex={graphIndex}
        graphModalTitle={graphModalTitle}
        graphModalSrc={graphModalSrc}
        onNavigate={handleNavigate}
      />
      <ConfirmDeleteModal
        open={confirmDelete.open}
        confirmDelete={confirmDelete}
        onCancel={handleCancelDelete}
        onConfirm={handleConfirmDelete}
      />
    </div>
  );
}

export default App;
