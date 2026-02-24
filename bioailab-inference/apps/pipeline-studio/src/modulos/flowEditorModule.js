export const hashString = (value) => {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
};

export const getFlowColorFromLabel = (flowLabel) => {
  const label = String(flowLabel || "");
  const hue = hashString(label) % 360;
  return `hsl(${hue} 70% 42%)`;
};

export const getBlockCardCategory = (blockName) => {
  const name = String(blockName || "").toLowerCase();
  if (name.startsWith("response_")) return "ml";
  if (name === "experiment_fetch" || name.endsWith("_extraction") || name.includes("fetch")) return "data";
  if (name.includes("time_slice") || name.includes("outlier") || name.includes("filter") || name.includes("derivative") || name.includes("integral") || name.includes("normalize") || name.includes("conversion")) return "process";
  if (name.includes("detector") || name.includes("growth") || name.includes("feature") || name.includes("curve")) return "analysis";
  if (name.includes("ml") || name.includes("inference") || name.includes("model") || name.includes("predict")) return "ml";
  if (name.includes("gate") || name.includes("branch") || name === "merge" || name === "value_in_list" || name === "numeric_compare" || name.includes("boolean") || name.includes("condition") || name === "label") return "flow";
  return "default";
};

export const getBlockCardTags = (blockName) => {
  const name = String(blockName || "").toLowerCase();
  const tags = [];
  if (name === "experiment_fetch") return ["Entrada"];
  if (name === "response_builder") return ["Resposta", "API"];
  if (name.startsWith("response_")) return ["Resposta"];
  if (["ml_inference", "ml_inference_series", "ml_inference_multichannel", "ml_forecaster_series", "ml_transform_series", "ml_detector"].includes(name) || name.includes("curve_fit")) return ["Modelo"];
  if (name.includes("feature")) return ["Features"];
  if (name.includes("detector")) return ["Detecção"];
  if (name.includes("filter")) return ["Filtro"];
  if (name.includes("normalize")) return ["Normalização"];
  if (name.includes("gate") || name.includes("branch") || name === "label" || name === "value_in_list") return ["Fluxo"];
  if (name.endsWith("_extraction")) return ["Extração"];
  if (name.includes("conversion")) return ["Conversão"];
  if (name.includes("merge")) return ["Composição"];
  return tags;
};
