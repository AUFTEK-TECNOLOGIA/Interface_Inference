import { useMemo } from "react";
import { ReactFlow, ReactFlowProvider, Background, SelectionMode } from "@xyflow/react";

import PipelineNode from "./PipelineNode";
import PipelineCanvasOverlays from "./PipelineCanvasOverlays";
import CanvasSelectionToolbar from "./CanvasSelectionToolbar";
import useFlowCanvasViewModel from "../hooks/useFlowCanvasViewModel";

export default function PipelineCanvasSection({
  t,
  nodes,
  edges,
  selectedNodes,
  selectedNode,
  selectedEdge,
  nodeFlowMetaById,
  library,
  simulation,
  openHelpModal,
  openBlockResultsModal,
  openConfigModalForNode,
  onNodesChange,
  onEdgesChange,
  onConnect,
  onReconnectStart,
  onReconnect,
  onReconnectEnd,
  handleSelect,
  handleEdgeClick,
  handlePaneClick,
  onDragOver,
  onDrop,
  reactFlowInstance,
  setViewport,
  reactFlowWrapper,
  analysisAreas,
  viewport,
  flowLanes,
  clipboard,
  copySelectedNodes,
  pasteNodes,
  duplicateSelectedNodes,
  autoLayoutNodes,
  alignNodesLeft,
  alignNodesCenterH,
  alignNodesRight,
  alignNodesTop,
  alignNodesCenterV,
  alignNodesBottom,
  distributeNodesH,
  distributeNodesV,
  setTrainingStudioOpen,
  workspace,
  trainModal,
  isRunning,
  runSimulation,
  contextMenu,
  closeContextMenu,
  deleteNode,
  setConfirmDelete,
  deleteEdge,
}) {
  const { canvasNodes, canvasEdges } = useFlowCanvasViewModel({
    nodes,
    edges,
    selectedNodes,
    selectedNode,
    selectedEdge,
    nodeFlowMetaById,
    noneLabel: t("flows.none"),
  });

  const pipelineStudioContextValue = useMemo(
    () => ({ openHelpModal, openBlockResultsModal, openConfigModalForNode, library, simulation }),
    [openHelpModal, openBlockResultsModal, openConfigModalForNode, library, simulation]
  );

  const nodeTypes = useMemo(
    () => ({
      pipelineNode: (props) => <PipelineNode {...props} studio={pipelineStudioContextValue} />,
    }),
    [pipelineStudioContextValue]
  );

  return (
    <section className="canvas" ref={reactFlowWrapper}>
      <ReactFlowProvider>
        <ReactFlow
          nodes={canvasNodes}
          edges={canvasEdges}
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
        </ReactFlow>
      </ReactFlowProvider>

      <PipelineCanvasOverlays analysisAreas={analysisAreas} viewport={viewport} t={t} flowLanes={flowLanes} />

      <CanvasSelectionToolbar
        t={t}
        selectedNodes={selectedNodes}
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
      />

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

        <button className="floating-simulate-button" onClick={runSimulation} disabled={isRunning || nodes.length === 0}>
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

      {contextMenu && (
        <>
          <div
            className="context-menu-overlay"
            onClick={closeContextMenu}
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 999,
            }}
          />
          <div
            className="context-menu"
            style={{
              position: "fixed",
              left: contextMenu.x,
              top: contextMenu.y,
              zIndex: 1000,
            }}
          >
            {contextMenu.type === "node" ? (
              <div className="context-menu-item" onClick={deleteNode}>
                {selectedNodes.length > 1 && selectedNodes.some((n) => n.id === contextMenu.nodeId)
                  ? t("actions.deleteSelection", { count: selectedNodes.length })
                  : t("actions.deleteBlock")}
              </div>
            ) : contextMenu.type === "selection" ? (
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
            ) : contextMenu.type === "edge" ? (
              <div className="context-menu-item" onClick={deleteEdge}>
                {t("actions.removeConnection")}
              </div>
            ) : null}
          </div>
        </>
      )}
    </section>
  );
}
