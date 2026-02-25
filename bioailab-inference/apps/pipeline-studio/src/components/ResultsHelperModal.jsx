import { memo } from "react";

import ResultsPanel from "./ResultsPanel";

function ResultsHelperModal({
  t,
  resultsModalOpen,
  closeResultsModal,
  simulation,
  openGraphModal,
  getStepLabel,
  getStepFlowLabel,
  getStepFlowColor,
}) {
  if (!resultsModalOpen) return null;

  return (
    <div className="helper-modal" onClick={closeResultsModal} role="dialog" aria-modal="true">
      <div className="helper-modal-inner" onClick={(e) => e.stopPropagation()}>
        <div className="helper-modal-header">
          <div>
            <h4>{t("panels.results")}</h4>
          </div>
          <button className="helper-modal-close" type="button" onClick={closeResultsModal}>
            {t("actions.close")}
          </button>
        </div>
        <div className="pipeline-support-modal-results-content">
          <ResultsPanel
            simulation={simulation}
            onGraphClick={openGraphModal}
            getStepLabel={getStepLabel}
            getStepFlowLabel={getStepFlowLabel}
            getStepFlowColor={getStepFlowColor}
          />
        </div>
      </div>
    </div>
  );
}

export default memo(ResultsHelperModal);
