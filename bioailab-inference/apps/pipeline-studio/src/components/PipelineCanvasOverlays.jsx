export default function PipelineCanvasOverlays({ analysisAreas, viewport, t, flowLanes }) {
  return (
    <>
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
    </>
  );
}
