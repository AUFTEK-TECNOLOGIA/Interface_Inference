import axios from "axios";

import TrainingModalBody from "./TrainingModalBody";
import ModelCandidatesPanel from "./ModelCandidatesPanel";
import DatasetSelector from "./DatasetSelector";
import TrainingStudio from "./TrainingStudio";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8001";

export default function PipelineTrainingModals({
  t,
  trainModal,
  setTrainModal,
  trainModelsDraft,
  setTrainModelsDraft,
  parseExperimentIdsText,
  nodes,
  setNodes,
  setDatasetSelectorOpen,
  setTrainBlockModal,
  setCandidatesModal,
  trainBlockModal,
  runTraining,
  candidatesModal,
  datasetSelectorOpen,
  trainingStudioOpen,
  setTrainingStudioOpen,
  workspace,
  loadPipelineFromJson,
  buildPipelineData,
}) {
  return (
    <>
      {trainModal?.open && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={() => setTrainModal((prev) => ({ ...prev, open: false, running: false }))}>
          <div className="workspace-modal workspace-modal-wide training-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("actions.train")}</div>
                <div className="workspace-modal-title">{t("training.title")}</div>
                <div className="workspace-muted">{t("training.subtitle")}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setTrainModal((prev) => ({ ...prev, open: false, running: false }))}>
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

      {candidatesModal?.open && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={() => setCandidatesModal({ open: false, sessionPath: "", stepId: "" })}>
          <div className="workspace-modal workspace-modal-candidates" onMouseDown={(e) => e.stopPropagation()}>
            <ModelCandidatesPanel
              tenant={workspace?.tenant}
              sessionPath={candidatesModal.sessionPath}
              stepId={candidatesModal.stepId}
              onSelect={async (candidateId) => {
                setCandidatesModal({ open: false, sessionPath: "", stepId: "" });
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
                setCandidatesModal({ open: false, sessionPath: "", stepId: "" });
                setTrainModal((prev) => ({ ...prev, open: true }));
              }}
              onClose={() => setCandidatesModal({ open: false, sessionPath: "", stepId: "" })}
            />
          </div>
        </div>
      )}

      {datasetSelectorOpen && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={() => setDatasetSelectorOpen(false)}>
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

      {trainingStudioOpen && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={() => setTrainingStudioOpen(false)}>
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
    </>
  );
}
