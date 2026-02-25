import { Controls, MiniMap } from "@xyflow/react";

const MINIMAP_DEFAULT_NODE_COLOR = "#c7ced9";
const MINIMAP_SELECTED_NODE_COLOR = "#2563eb";

const getMiniMapNodeColor = (node) => {
  if (node.selected) {
    return MINIMAP_SELECTED_NODE_COLOR;
  }

  return node?.data?.flowColor || MINIMAP_DEFAULT_NODE_COLOR;
};

export default function FlowViewportNavigator() {
  return (
    <>
      <MiniMap
        className="studio-minimap"
        position="bottom-left"
        pannable
        zoomable
        nodeColor={getMiniMapNodeColor}
        nodeStrokeWidth={3}
        maskColor="rgba(15, 23, 42, 0.12)"
      />
      <Controls
        className="studio-map-controls"
        position="bottom-left"
        showInteractive={false}
      />
    </>
  );
}
