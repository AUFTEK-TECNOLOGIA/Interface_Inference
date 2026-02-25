import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import { reconnectEdge } from "@xyflow/react";
import axios from "axios";
import "@xyflow/react/dist/style.css";
import "./App.css";

import AppHeader from "./components/AppHeader";
import PipelineBlocksPanel from "./components/PipelineBlocksPanel";
import PipelineGlobalModals from "./components/PipelineGlobalModals";
import PipelineSupportModals from "./components/PipelineSupportModals";
import BlockCard from "./components/BlockCard";
import WorkspaceQuickModals from "./components/WorkspaceQuickModals";
import WorkspaceManagementModals from "./components/WorkspaceManagementModals";
import PipelineTrainingModals from "./components/PipelineTrainingModals";
import PipelineCanvasSection from "./components/PipelineCanvasSection";
import WorkspaceHomeModal from "./components/WorkspaceHomeModal";
import NodeConfigFormContent from "./components/NodeConfigFormContent";
import { useI18n } from "./locale/i18n";
import { TRAINING_ALGO_PARAM_SCHEMA, parseExperimentIdsText as parseExperimentIdsInput, buildTrainingParamsForAlgorithm as buildTrainingParamsByAlgorithm } from "./modulos/trainingModule";
import { getFlowColorFromLabel, getBlockCardCategory as getBlockCardCategoryModule } from "./modulos/flowEditorModule";
import { sanitizeColor, resolveWorkspaceLogoSrc as resolveWorkspaceLogoSrcModule, workspaceInitials } from "./modulos/workspaceModule";
import { buildPreparedSteps as buildPreparedStepsModule, collectSimulationGraphs } from "./modulos/simulationModule";
import usePipelineStudioState from "./hooks/usePipelineStudioState";


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
        <BlockCard
          block={block}
          displayName={displayName}
          visual={visual}
          isPinned={isPinned}
          isActive={isActive}
          tags={tags}
          onInspect={() => setInspectedBlock(block)}
          onAdd={handleAdd}
          onHelp={handleHelp}
          onTogglePin={handleTogglePin}
          onDragStart={(e) => onDragStartBlock(e, block)}
          t={t}
        />
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
        <BlockCard
          block={block}
          displayName={displayName}
          visual={visual}
          isPinned={isPinned}
          isMini
          onInspect={() => setInspectedBlock(block)}
          onAdd={() => addBlockToCanvas(block)}
          onTogglePin={togglePin}
          onDragStart={(e) => onDragStartBlock(e, block)}
          t={t}
        />
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
      <WorkspaceHomeModal
        t={t}
        workspaceHomeOpen={workspaceHomeOpen}
        workspace={workspace}
        setWorkspaceHomeOpen={setWorkspaceHomeOpen}
        workspaceHomeMode={workspaceHomeMode}
        setWorkspaceHomeMode={setWorkspaceHomeMode}
        workspaceListLoading={workspaceListLoading}
        workspaceList={workspaceList}
        workspaceActionLoading={workspaceActionLoading}
        workspaceError={workspaceError}
        newTenantName={newTenantName}
        setNewTenantName={setNewTenantName}
        handleCreateWorkspace={handleCreateWorkspace}
        handleLoadWorkspace={handleLoadWorkspace}
        fetchWorkspaces={fetchWorkspaces}
        selectedWorkspaceKey={selectedWorkspaceKey}
        setSelectedWorkspaceKey={setSelectedWorkspaceKey}
        workspaceCardMenuKey={workspaceCardMenuKey}
        setWorkspaceCardMenuKey={setWorkspaceCardMenuKey}
        openVersionsModal={openVersionsModal}
        setDuplicateModal={setDuplicateModal}
        duplicateLogoFileInputRef={duplicateLogoFileInputRef}
        setDeleteModal={setDeleteModal}
        setWorkspaceMetaDraft={setWorkspaceMetaDraft}
        setEditModal={setEditModal}
        triggerLoadPipeline={triggerLoadPipeline}
        resolveWorkspaceLogoSrc={resolveWorkspaceLogoSrc}
        workspaceInitials={workspaceInitials}
      />

      <WorkspaceQuickModals
        t={t}
        duplicateModal={duplicateModal}
        setDuplicateModal={setDuplicateModal}
        workspaceActionLoading={workspaceActionLoading}
        duplicateLogoFileInputRef={duplicateLogoFileInputRef}
        handleDuplicateWorkspace={handleDuplicateWorkspace}
        editModal={editModal}
        setEditModal={setEditModal}
        workspaceMetaDraft={workspaceMetaDraft}
        setWorkspaceMetaDraft={setWorkspaceMetaDraft}
        workspaceLogoFileInputRef={workspaceLogoFileInputRef}
        handleSaveWorkspaceAppearanceFromModal={handleSaveWorkspaceAppearanceFromModal}
      />

      <WorkspaceManagementModals
        t={t}
        versionsModal={versionsModal}
        setVersionsModal={setVersionsModal}
        versionActionsModal={versionActionsModal}
        setVersionActionsModal={setVersionActionsModal}
        versionLogsModal={versionLogsModal}
        setVersionLogsModal={setVersionLogsModal}
        renameVersionModal={renameVersionModal}
        setRenameVersionModal={setRenameVersionModal}
        deleteVersionModal={deleteVersionModal}
        setDeleteVersionModal={setDeleteVersionModal}
        deleteModal={deleteModal}
        setDeleteModal={setDeleteModal}
        workspaceActionLoading={workspaceActionLoading}
        versionsPageStart={versionsPageStart}
        versionsPageEnd={versionsPageEnd}
        versionsSorted={versionsSorted}
        versionsPageItems={versionsPageItems}
        versionsCurrentPage={versionsCurrentPage}
        versionsTotalPages={versionsTotalPages}
        formatDateTime={formatDateTime}
        handleOpenVersionInEditor={handleOpenVersionInEditor}
        handleCreateNewVersionClean={handleCreateNewVersionClean}
        handleCreateNewVersionCopy={handleCreateNewVersionCopy}
        handleActivateVersion={handleActivateVersion}
        handleRenameVersion={handleRenameVersion}
        handleDeleteVersion={handleDeleteVersion}
        handleDeleteWorkspace={handleDeleteWorkspace}
      />

      <PipelineTrainingModals
        t={t}
        trainModal={trainModal}
        setTrainModal={setTrainModal}
        trainModelsDraft={trainModelsDraft}
        setTrainModelsDraft={setTrainModelsDraft}
        parseExperimentIdsText={parseExperimentIdsText}
        nodes={nodes}
        setNodes={setNodes}
        setDatasetSelectorOpen={setDatasetSelectorOpen}
        setTrainBlockModal={setTrainBlockModal}
        setCandidatesModal={setCandidatesModal}
        trainBlockModal={trainBlockModal}
        runTraining={runTraining}
        candidatesModal={candidatesModal}
        datasetSelectorOpen={datasetSelectorOpen}
        trainingStudioOpen={trainingStudioOpen}
        setTrainingStudioOpen={setTrainingStudioOpen}
        workspace={workspace}
        loadPipelineFromJson={loadPipelineFromJson}
        buildPipelineData={buildPipelineData}
      />

      {/* Input oculto para carregar arquivo */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={loadPipeline}
        accept=".json"
        className="visually-hidden-input"
      />
      {/* Input oculto para upload de logo */}
      <input
        type="file"
        ref={workspaceLogoFileInputRef}
        accept="image/*"
        className="visually-hidden-input"
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
        className="visually-hidden-input"
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
        <PipelineBlocksPanel
          t={t}
          width={leftPanel.width}
          blocksQuery={blocksQuery}
          setBlocksQuery={setBlocksQuery}
          favoriteBlocks={favoriteBlocks}
          recentBlocks={recentBlocks}
          library={library}
          blockMatchesQuery={blockMatchesQuery}
          renderBlockCardMini={renderBlockCardMini}
          renderBlockCard={renderBlockCard}
          onStartResize={leftPanel.startResize}
        />

        {/* PAINEL 3: Canvas */}
        <PipelineCanvasSection
          t={t}
          nodes={nodes}
          edges={edges}
          selectedNodes={selectedNodes}
          selectedNode={selectedNode}
          selectedEdge={selectedEdge}
          nodeFlowMetaById={nodeFlowMetaById}
          library={library}
          simulation={simulation}
          openHelpModal={openHelpModal}
          openBlockResultsModal={openBlockResultsModal}
          openConfigModalForNode={openConfigModalForNode}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onReconnectStart={onReconnectStart}
          onReconnect={onReconnect}
          onReconnectEnd={onReconnectEnd}
          handleSelect={handleSelect}
          handleEdgeClick={handleEdgeClick}
          handlePaneClick={handlePaneClick}
          onDragOver={onDragOver}
          onDrop={onDrop}
          reactFlowInstance={reactFlowInstance}
          setViewport={setViewport}
          reactFlowWrapper={reactFlowWrapper}
          analysisAreas={analysisAreas}
          viewport={viewport}
          flowLanes={flowLanes}
          clipboard={clipboard}
          copySelectedNodes={copySelectedNodes}
          pasteNodes={pasteNodes}
          duplicateSelectedNodes={duplicateSelectedNodes}
          autoLayoutNodes={autoLayoutNodes}
          alignNodesLeft={alignNodesLeft}
          alignNodesCenterH={alignNodesCenterH}
          alignNodesRight={alignNodesRight}
          alignNodesTop={alignNodesTop}
          alignNodesCenterV={alignNodesCenterV}
          alignNodesBottom={alignNodesBottom}
          distributeNodesH={distributeNodesH}
          distributeNodesV={distributeNodesV}
          setTrainingStudioOpen={setTrainingStudioOpen}
          workspace={workspace}
          trainModal={trainModal}
          isRunning={isRunning}
          runSimulation={runSimulation}
          contextMenu={contextMenu}
          closeContextMenu={closeContextMenu}
          deleteNode={deleteNode}
          setConfirmDelete={setConfirmDelete}
          deleteEdge={deleteEdge}
        />

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

                <NodeConfigFormContent
                  selectedNode={selectedNode}
                  t={t}
                  renderConfigField={renderConfigField}
                  useDefaultExperiment={useDefaultExperiment}
                  setUseDefaultExperiment={setUseDefaultExperiment}
                  setNodes={setNodes}
                  edges={edges}
                  updateNodeConfigField={updateNodeConfigField}
                  applyConfigToSelectedNode={applyConfigToSelectedNode}
                  setFeaturesMergeInputs={setFeaturesMergeInputs}
                  setSequentialInputsCount={setSequentialInputsCount}
                  setValueInListDraft={setValueInListDraft}
                  valueInListDraft={valueInListDraft}
                  effectiveConfigInputs={effectiveConfigInputs}
                  availableChannels={availableChannels}
                  availableFitChannels={availableFitChannels}
                  simulation={simulation}
                  trainModal={trainModal}
                  setTrainModal={setTrainModal}
                  trainModelsDraft={trainModelsDraft}
                  setTrainModelsDraft={setTrainModelsDraft}
                  nodes={nodes}
                  setDatasetSelectorOpen={setDatasetSelectorOpen}
                  setTrainBlockModal={setTrainBlockModal}
                  setCandidatesModal={setCandidatesModal}
                  trainBlockModal={trainBlockModal}
                  runTraining={runTraining}
                />

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

      <PipelineSupportModals
        t={t}
        resultsModalOpen={resultsModalOpen}
        closeResultsModal={closeResultsModal}
        simulation={simulation}
        openGraphModal={openGraphModal}
        getStepLabel={getStepLabel}
        getStepFlowLabel={getStepFlowLabel}
        getStepFlowColor={getStepFlowColor}
        helpModal={helpModal}
        closeHelpModal={closeHelpModal}
        activeHelpModel={activeHelpModel}
        addBlockToCanvas={addBlockToCanvas}
        helpTab={helpTab}
        setHelpTab={setHelpTab}
        library={library}
      />

      <PipelineGlobalModals
        t={t}
        blockResultsModal={blockResultsModal}
        closeBlockResultsModal={closeBlockResultsModal}
        getStepLabel={getStepLabel}
        simulation={simulation}
        openGraphModal={openGraphModal}
        graphModalOpen={graphModalOpen}
        closeGraphModal={closeGraphModal}
        graphList={graphList}
        graphIndex={graphIndex}
        graphModalTitle={graphModalTitle}
        graphModalSrc={graphModalSrc}
        handleNavigate={handleNavigate}
        confirmDelete={confirmDelete}
        handleCancelDelete={handleCancelDelete}
        handleConfirmDelete={handleConfirmDelete}
      />
    </div>
  );
}

export default App;
