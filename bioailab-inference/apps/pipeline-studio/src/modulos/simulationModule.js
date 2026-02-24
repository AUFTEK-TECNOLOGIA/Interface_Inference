export const buildPreparedSteps = (nodes, edges) =>
  nodes.map((node) => {
    const incomingEdges = edges.filter((edge) => edge.target === node.id);
    const inputMapping = {};

    incomingEdges.forEach((edge) => {
      const sourceHandle = edge.sourceHandle || "";
      const targetHandle = edge.targetHandle || "";
      const outputName = sourceHandle.includes("-out-") ? sourceHandle.split("-out-")[1] : sourceHandle;
      const inputName = targetHandle.includes("-in-") ? targetHandle.split("-in-")[1] : outputName;
      if (inputName && outputName) {
        inputMapping[inputName] = `${edge.source}.${outputName}`;
      }
    });

    return {
      step_id: node.id,
      block_name: node.data.blockName,
      block_config: node.data.config || {},
      depends_on: incomingEdges.map((edge) => edge.source),
      input_mapping: inputMapping,
    };
  });

export const collectSimulationGraphs = (simulation) => {
  if (!simulation?.step_results) return [];
  const graphs = [];
  Object.entries(simulation.step_results).forEach(([stepId, stepData]) => {
    const outputGraphs = stepData?.output_graphs || stepData?.data?.output_graphs || {};
    Object.entries(outputGraphs).forEach(([name, src]) => {
      graphs.push({ src, title: `${stepId} - ${name}`, stepId, name });
    });
  });
  return graphs;
};
