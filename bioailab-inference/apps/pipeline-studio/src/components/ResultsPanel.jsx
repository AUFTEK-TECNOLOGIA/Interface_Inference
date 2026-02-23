import React from 'react';
import StepResultCard from './StepResultCard';
import { useI18n } from "../locale/i18n";

const ResultsPanel = ({ simulation, onGraphClick, getStepLabel, getStepFlowLabel, getStepFlowColor }) => {
  const { t } = useI18n();

  if (!simulation) {
    return (
      <div className="results-empty">
        <span className="results-empty-icon" aria-hidden="true" />
        <p>{t("results.empty")}</p>
      </div>
    );
  }

  const { success, duration_ms, errors, step_results } = simulation;
  const stepEntries = Object.entries(step_results || {});

  return (
    <div className="results-container">
      {/* Status da Simulação */}
      <fieldset className="results-group">
        <legend>{t("results.status")}</legend>
        <div className="results-status-row">
          <span className={`results-badge ${success ? 'results-badge--success' : 'results-badge--error'}`}>
            {success ? t("results.success") : t("results.failed")}
          </span>
          <span className="results-badge results-badge--info">
            {t("results.duration")}: {duration_ms?.toFixed(2)} ms
          </span>
        </div>
      </fieldset>

      {/* Erros */}
      {errors && errors.length > 0 && (
        <fieldset className="results-group results-group--error">
          <legend>{t("results.errors")}</legend>
          <ul className="results-error-list">
            {errors.map((error, index) => (
              <li key={index}>{error}</li>
            ))}
          </ul>
        </fieldset>
      )}

      {/* Resultados por Passo */}
      <fieldset className="results-group">
        <legend>{t("results.stepsExecuted")}</legend>
        <div className="results-steps">
          {!getStepFlowLabel && (
            <>
              {stepEntries.map(([stepId, stepData]) => (
                <StepResultCard
                  key={stepId}
                  stepId={stepId}
                  stepLabel={getStepLabel ? getStepLabel(stepId) : stepId}
                  stepData={stepData}
                  onGraphClick={onGraphClick}
                />
              ))}
            </>
          )}

          {getStepFlowLabel && (
            <>
              {(() => {
                const groups = new Map();
                stepEntries.forEach(([stepId, stepData]) => {
                  const flowLabel = getStepFlowLabel(stepId) || t("flows.none");
                  if (!groups.has(flowLabel)) groups.set(flowLabel, []);
                  groups.get(flowLabel).push([stepId, stepData]);
                });

                return Array.from(groups.entries()).map(([flowLabel, steps]) => {
                  const flowColor = getStepFlowColor ? getStepFlowColor(steps[0]?.[0]) : undefined;
                  return (
                    <div className="results-flow-group" key={flowLabel}>
                      <div className="results-flow-header">
                        <span
                          className="results-flow-badge"
                          title={flowLabel}
                          style={flowColor ? { borderColor: flowColor, color: flowColor } : undefined}
                        >
                          {t("flows.groupTitle", { label: flowLabel })}
                        </span>
                        <span className="results-flow-count">{steps.length}</span>
                      </div>
                      <div className="results-flow-steps">
                        {steps.map(([stepId, stepData]) => (
                          <StepResultCard
                            key={stepId}
                            stepId={stepId}
                            stepLabel={getStepLabel ? getStepLabel(stepId) : stepId}
                            stepData={stepData}
                            onGraphClick={onGraphClick}
                          />
                        ))}
                      </div>
                    </div>
                  );
                });
              })()}
            </>
          )}
        </div>
      </fieldset>
    </div>
  );
};

export default ResultsPanel;
