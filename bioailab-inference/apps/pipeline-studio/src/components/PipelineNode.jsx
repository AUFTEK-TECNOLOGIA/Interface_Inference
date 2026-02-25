import { Handle } from "@xyflow/react";
import { useI18n } from "../locale/i18n";

const getNodeCategory = (blockName = "") => {
  const name = String(blockName).toLowerCase();
  if (name.includes("experiment_fetch") || name.includes("data") || name.includes("input")) return "data";
  if (name.includes("filter") || name.includes("process") || name.includes("normalize") || name.includes("smooth") || name.includes("fusion")) return "process";
  if (name.includes("growth") || name.includes("detect") || name.includes("analysis") || name.includes("feature")) return "analysis";
  if (name.includes("curve") || name.includes("model") || name.includes("ml") || name.includes("predict") || name.includes("regression")) return "ml";
  if (name.includes("gate") || name.includes("branch") || name.includes("merge") || name.includes("boolean") || name.includes("condition") || name === "label") return "flow";
  return "default";
};

export default function PipelineNode({ data, studio }) {
  const { t } = useI18n();

  const rawInputs = data.dataInputs?.length ? data.dataInputs : Object.keys(data.inputSchema || {});
  const inputs = rawInputs.filter((key) => !data.inputSchema?.[key]?.hidden);
  const outputs = data.dataOutputs?.length ? data.dataOutputs : Object.keys(data.outputSchema || {});
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
    if (["amplitude_detector", "derivative_detector", "ratio_detector"].includes(blockName) && direction === "out" && key === "has_growth") return t("handles.detectors.hasGrowth");
    if (blockName === "ml_detector" && direction === "out" && key === "detected") return t("handles.detectors.hasGrowth");
    if (blockName === "sensor_fusion" && direction === "in" && key.startsWith("sensor_data_")) {
      const idx = Number(String(key).split("_").pop());
      if (Number.isFinite(idx)) return t("handles.sensorFusion.sensor", { index: idx });
    }
    return key;
  };

  const category = getNodeCategory(data.blockName);
  const categoryConfig = {
    data: { color: "var(--block-data)", icon: "D", label: t("nodeCategories.data") },
    process: { color: "var(--block-process)", icon: "P", label: t("nodeCategories.process") },
    analysis: { color: "var(--block-analysis)", icon: "A", label: t("nodeCategories.analysis") },
    ml: { color: "var(--block-ml)", icon: "ML", label: t("nodeCategories.ml") },
    flow: { color: "var(--block-flow)", icon: "F", label: t("nodeCategories.flow") },
    default: { color: "var(--color-gray-400)", icon: "B", label: t("nodeCategories.block") },
  };
  const config = categoryConfig[category];

  return (
    <div className={`pipeline-node ${category} ${isLabelNode ? "pipeline-node--label" : ""} ${data.dimmed ? "is-dimmed" : ""}`} style={nodeStyle}>
      <div className="node-header">
        <div className="node-icon" style={{ backgroundColor: isLabelNode && data.flowColor ? data.flowColor : config.color }} title={isLabelNode ? (data.flowLabel || t("flows.none")) : undefined}>{config.icon}</div>
        <div className="node-title-section">
          <div className="node-title-row"><div className="node-title" title={data.label}>{data.label}</div></div>
          <div className="node-meta">
            <div className="node-category">{config.label}</div>
            <span className="node-flow-badge" title={data.flowLabel || t("flows.none")} style={data.flowColor ? { borderColor: data.flowColor, color: data.flowColor } : undefined}>{data.flowLabel || t("flows.none")}</span>
          </div>
        </div>
      </div>

      <div className="node-body">
        <div className="node-description">{data.description}</div>
        {inputs.length > 0 && (
          <div className="node-section">
            <div className="section-label">Inputs</div>
            <div className="handles-list">
              {inputs.map((key) => (
                <div key={key} className="handle-item input-handle">
                  <Handle type="target" position="left" id={`${data.blockName}-in-${key}`} className="node-handle input" style={{ left: "-6px" }} />
                  <span className="handle-label">{getHandleDisplay("in", data.blockName, key)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {outputs.length > 0 && (
          <div className="node-section">
            <div className="section-label">Outputs</div>
            <div className="handles-list">
              {outputs.map((key) => (
                <div key={key} className="handle-item output-handle">
                  <span className="handle-label">{getHandleDisplay("out", data.blockName, key)}</span>
                  <Handle type="source" position="right" id={`${data.blockName}-out-${key}`} className="node-handle output" style={{ right: "-6px" }} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="node-footer">
        <div className="node-actions">
          {studio?.openConfigModalForNode && <button type="button" className="node-action node-config" aria-label={t("configuration.openLabel")} title={t("configuration.openLabel")} onMouseDown={(e) => e.stopPropagation()} onClick={(e) => { e.stopPropagation(); studio.openConfigModalForNode(data.stepId); }}>C</button>}
          {studio?.openBlockResultsModal && <button type="button" className="node-action node-results" aria-label={t("blockResults.openLabel")} title={t("blockResults.openLabel")} disabled={!studio.simulation?.step_results?.[data.stepId]} onMouseDown={(e) => e.stopPropagation()} onClick={(e) => { e.stopPropagation(); studio.openBlockResultsModal(data.stepId); }}>R</button>}
          {studio?.openHelpModal && (
            <button type="button" className="node-action node-help" aria-label={t("helper.openLabel")} title={t("helper.openLabel")} onMouseDown={(e) => e.stopPropagation()} onClick={(e) => {
              e.stopPropagation();
              const blockFromLibrary = studio.library?.blocks?.find((b) => b.name === data.blockName);
              const fallbackBlock = { name: data.blockName, description: data.description || "", data_inputs: inputs, data_outputs: outputs, config_inputs: [], input_schema: data.inputSchema || {}, output_schema: data.outputSchema || {} };
              studio.openHelpModal(blockFromLibrary || fallbackBlock);
            }}>?</button>
          )}
        </div>
      </div>
    </div>
  );
}
