import "./PipelineSupportModals.css";
import ResultsHelperModal from "./ResultsHelperModal";
import HelpHelperModal from "./HelpHelperModal";

export default function PipelineSupportModals({
  t,
  resultsModalOpen,
  closeResultsModal,
  simulation,
  openGraphModal,
  getStepLabel,
  getStepFlowLabel,
  getStepFlowColor,
  helpModal,
  closeHelpModal,
  activeHelpModel,
  addBlockToCanvas,
  helpTab,
  setHelpTab,
  library,
}) {
  return (
    <>
      <ResultsHelperModal
        t={t}
        resultsModalOpen={resultsModalOpen}
        closeResultsModal={closeResultsModal}
        simulation={simulation}
        openGraphModal={openGraphModal}
        getStepLabel={getStepLabel}
        getStepFlowLabel={getStepFlowLabel}
        getStepFlowColor={getStepFlowColor}
      />
      <HelpHelperModal
        t={t}
        helpModal={helpModal}
        closeHelpModal={closeHelpModal}
        activeHelpModel={activeHelpModel}
        addBlockToCanvas={addBlockToCanvas}
        helpTab={helpTab}
        setHelpTab={setHelpTab}
        library={library}
      />
    </>
  );
}
