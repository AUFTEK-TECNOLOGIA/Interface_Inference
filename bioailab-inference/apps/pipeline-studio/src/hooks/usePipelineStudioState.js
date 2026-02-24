import { useEffect, useRef, useState } from "react";
import { useEdgesState, useNodesState } from "@xyflow/react";

export default function usePipelineStudioState(defaultLibrary) {
  const [helpModal, setHelpModal] = useState({ open: false, block: null });
  const [helpTab, setHelpTab] = useState("overview");
  const [blockResultsModal, setBlockResultsModal] = useState({ open: false, stepId: null });
  const [resultsModalOpen, setResultsModalOpen] = useState(false);
  const [configModalOpen, setConfigModalOpen] = useState(false);
  const [valueInListDraft, setValueInListDraft] = useState("");

  const [library, setLibrary] = useState(defaultLibrary);
  const [nodes, setNodes, defaultOnNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [clipboard, setClipboard] = useState({ nodes: [], edges: [] });
  const [configFieldErrors, setConfigFieldErrors] = useState({});
  const [pipelineName, setPipelineName] = useState("pipeline_visual");
  const [workspace, setWorkspace] = useState(null);
  const [workspaceHomeOpen, setWorkspaceHomeOpen] = useState(true);
  const [workspaceList, setWorkspaceList] = useState([]);
  const [workspaceListLoading, setWorkspaceListLoading] = useState(false);
  const [workspaceActionLoading, setWorkspaceActionLoading] = useState(false);
  const [workspaceError, setWorkspaceError] = useState("");
  const [newTenantName, setNewTenantName] = useState("");
  const [selectedWorkspaceKey, setSelectedWorkspaceKey] = useState("");
  const [workspaceCardMenuKey, setWorkspaceCardMenuKey] = useState("");
  const [workspaceMetaDraft, setWorkspaceMetaDraft] = useState({ title: "", logo: "", accent_color: "#1e90ff" });
  const [duplicateModal, setDuplicateModal] = useState({ open: false, source: null, tenant: "", logoFile: null });
  const [editModal, setEditModal] = useState({ open: false, target: null });
  const [deleteModal, setDeleteModal] = useState({ open: false, target: null });
  const [versionsModal, setVersionsModal] = useState({
    open: false,
    target: null,
    active: "",
    versions: [],
    loading: false,
    reasonDraft: "",
    page: 0,
    query: "",
  });
  const [versionActionsModal, setVersionActionsModal] = useState({ open: false, version: null });
  const [versionLogsModal, setVersionLogsModal] = useState({ open: false, version: null, query: "" });
  const [deleteVersionModal, setDeleteVersionModal] = useState({ open: false, version: "" });
  const [renameVersionModal, setRenameVersionModal] = useState({ open: false, version: "", name: "", reason: "" });
  const [simulation, setSimulation] = useState(null);
  const [trainModal, setTrainModal] = useState({
    open: false,
    step: 0,
    protocolId: "",
    experimentIdsText: "",
    yTransform: "log10p",
    advancedOpen: false,
    testSize: 0.2,
    randomState: 42,
    permImportance: false,
    permRepeats: 10,
    selectionMetric: "rmse",
    maxTrials: 60,
    applyToPipeline: true,
    gridSearchManual: false,
    changeReason: "",
    running: false,
    error: "",
    result: null,
    useRegression: false,
    regressionType: "linear",
    regressionAutoSelect: false,
    polynomialDegree: 3,
  });
  const [trainModelsDraft, setTrainModelsDraft] = useState({});
  const [trainBlockModal, setTrainBlockModal] = useState({ open: false, stepId: "" });
  const [candidatesModal, setCandidatesModal] = useState({ open: false, sessionPath: "", stepId: "" });
  const [datasetSelectorOpen, setDatasetSelectorOpen] = useState(false);
  const [trainingStudioOpen, setTrainingStudioOpen] = useState(false);
  const [graphModalOpen, setGraphModalOpen] = useState(false);
  const [graphModalSrc, setGraphModalSrc] = useState(null);
  const [graphModalTitle, setGraphModalTitle] = useState("");
  const [graphList, setGraphList] = useState([]);
  const [graphIndex, setGraphIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");
  const [inspectedBlock, setInspectedBlock] = useState(null);
  const [blocksQuery, setBlocksQuery] = useState("");
  const [favoriteBlocks, setFavoriteBlocks] = useState(() => {
    try {
      const raw = localStorage.getItem("pipelineStudio.favoriteBlocks");
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });
  const [recentBlocks, setRecentBlocks] = useState(() => {
    try {
      const raw = localStorage.getItem("pipelineStudio.recentBlocks");
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });
  const [contextMenu, setContextMenu] = useState(null);
  const [useDefaultExperiment, setUseDefaultExperiment] = useState(false);
  const [viewport, setViewport] = useState({ x: 0, y: 0, zoom: 1 });
  const reactFlowWrapper = useRef(null);
  const reactFlowInstance = useRef(null);

  useEffect(() => {
    if (!workspaceCardMenuKey) return undefined;

    const onMouseDown = (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest(".workspace-pipeline-actions")) return;
      setWorkspaceCardMenuKey("");
    };

    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [workspaceCardMenuKey]);

  return {
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
  };
}
