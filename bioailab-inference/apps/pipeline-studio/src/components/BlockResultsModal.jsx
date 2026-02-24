import StepResultCard from "./StepResultCard";

export default function BlockResultsModal({ open, t, onClose, stepId, getStepLabel, simulation, onGraphClick }) {
  if (!open) return null;

  return (
    <div className="helper-modal" onClick={onClose} role="dialog" aria-modal="true">
      <div className="helper-modal-inner" onClick={(e) => e.stopPropagation()}>
        <div className="helper-modal-header">
          <div>
            <h4>{t("blockResults.title")}</h4>
            <div className="helper-modal-subtitle">{getStepLabel(stepId)}</div>
          </div>
          <button className="helper-modal-close" type="button" onClick={onClose}>
            {t("actions.close")}
          </button>
        </div>

        {!simulation ? (
          <p className="helper-modal-desc">{t("blockResults.noSimulation")}</p>
        ) : simulation?.step_results?.[stepId] ? (
          <div style={{ marginTop: 12 }}>
            <StepResultCard
              stepId={stepId}
              stepLabel={getStepLabel(stepId)}
              stepData={simulation.step_results[stepId]}
              onGraphClick={onGraphClick}
              defaultExpanded={true}
              hideHeader={true}
            />
          </div>
        ) : (
          <p className="helper-modal-desc">{t("blockResults.noResults")}</p>
        )}
      </div>
    </div>
  );
}
