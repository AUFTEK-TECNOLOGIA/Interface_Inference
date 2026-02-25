import { useMemo } from "react";

const SELECTED_EDGE_STYLE = {
  stroke: "#ef4444",
  strokeWidth: 3,
  opacity: 0.95,
};

const getDefaultEdgeStyle = (edge, nodeFlowMetaById) => {
  const baseColor = nodeFlowMetaById[edge.source]?.color || nodeFlowMetaById[edge.target]?.color;

  if (!baseColor) {
    return undefined;
  }

  return {
    stroke: baseColor,
    strokeWidth: 2,
    opacity: 0.9,
  };
};

export default function useFlowCanvasViewModel({
  nodes,
  edges,
  selectedNodes,
  selectedNode,
  selectedEdge,
  nodeFlowMetaById,
  noneLabel,
}) {
  const canvasNodes = useMemo(() => {
    return nodes.map((node) => {
      const isSelected = selectedNodes.some((selected) => selected.id === node.id) || selectedNode?.id === node.id;
      const flowLabel = nodeFlowMetaById[node.id]?.label || noneLabel;

      return {
        ...node,
        selected: isSelected,
        data: {
          ...node.data,
          stepId: node.id,
          flowLabel,
          flowColor: nodeFlowMetaById[node.id]?.color,
          dimmed: false,
        },
      };
    });
  }, [nodes, selectedNodes, selectedNode, nodeFlowMetaById, noneLabel]);

  const canvasEdges = useMemo(() => {
    return edges.map((edge) => {
      const isSelected = selectedEdge?.id === edge.id;
      const style = isSelected ? SELECTED_EDGE_STYLE : getDefaultEdgeStyle(edge, nodeFlowMetaById);

      return {
        ...edge,
        selected: isSelected,
        reconnectable: true,
        style,
      };
    });
  }, [edges, selectedEdge, nodeFlowMetaById]);

  return {
    canvasNodes,
    canvasEdges,
  };
}
