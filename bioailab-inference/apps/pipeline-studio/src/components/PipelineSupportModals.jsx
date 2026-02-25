import ResultsPanel from "./ResultsPanel";
import "./PipelineSupportModals.css";

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
      {resultsModalOpen && (
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
      )}

      {helpModal.open && helpModal.block && (
        <div className="helper-modal" onClick={closeHelpModal} role="dialog" aria-modal="true">
          <div className="helper-modal-inner helper-modal-inner--help" onClick={(e) => e.stopPropagation()}>
            <div className="help-modal-header">
              <div className="help-modal-title">
                <div className="help-modal-icon" style={activeHelpModel?.color ? { background: activeHelpModel.color } : undefined}>
                  {activeHelpModel?.icon || "B"}
                </div>
                <div>
                  <div className="help-modal-title-text">{activeHelpModel?.title || helpModal.block.name}</div>
                  <div className="help-modal-subtitle">{activeHelpModel?.subtitle || helpModal.block.name}</div>
                </div>
              </div>
              <div className="help-modal-actions">
                <button className="btn" type="button" onClick={() => addBlockToCanvas(helpModal.block)}>
                  {t("actions.add")}
                </button>
                <button className="helper-modal-close" type="button" onClick={closeHelpModal}>
                  {t("actions.close")}
                </button>
              </div>
            </div>

            <div className="help-tabs" role="tablist" aria-label={t("helper.title")}>
              {[
                { id: "overview", label: t("helper.tabs.overview") },
                { id: "io", label: t("helper.tabs.io") },
                { id: "config", label: t("helper.tabs.config") },
                { id: "examples", label: t("helper.tabs.examples") },
                { id: "troubleshooting", label: t("helper.tabs.troubleshooting") },
              ].map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  className={`help-tab${helpTab === tab.id ? " active" : ""}`}
                  role="tab"
                  aria-selected={helpTab === tab.id}
                  onClick={() => setHelpTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {helpTab === "overview" && (
              <div className="help-content">
                <div className="help-card">
                  <div className="help-card-title">{t("helper.sections.what")}</div>
                  <div className="help-card-text">{activeHelpModel?.what || helpModal.block.description}</div>
                </div>

                {activeHelpModel?.when?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.when")}</div>
                    <ul className="help-list">
                      {activeHelpModel.when.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {activeHelpModel?.how?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.how")}</div>
                    <ol className="help-list">
                      {activeHelpModel.how.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ol>
                  </div>
                ) : null}

                {activeHelpModel?.tips?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.tips")}</div>
                    <ul className="help-list">
                      {activeHelpModel.tips.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            )}

            {helpTab === "io" && (
              <div className="help-content">
                <div className="help-card">
                  <div className="help-card-title">{t("helper.sections.inputs")}</div>
                  <ul className="schema-list">
                    {helpModal.block.data_inputs?.length ? (
                      helpModal.block.data_inputs.map((key) => (
                        <li key={key}>
                          <span className="schema-key">{key}</span>
                          <small>{helpModal.block.input_schema?.[key]?.type || "any"}</small>
                        </li>
                      ))
                    ) : (
                      <li className="empty">{t("summary.noneInput")}</li>
                    )}
                  </ul>
                </div>

                <div className="help-card">
                  <div className="help-card-title">{t("helper.sections.outputs")}</div>
                  <ul className="schema-list">
                    {helpModal.block.data_outputs?.length ? (
                      helpModal.block.data_outputs.map((key) => (
                        <li key={key}>
                          <span className="schema-key">{key}</span>
                          <small>{helpModal.block.output_schema?.[key]?.type || "any"}</small>
                        </li>
                      ))
                    ) : (
                      <li className="empty">{t("summary.noneOutput")}</li>
                    )}
                  </ul>
                </div>
              </div>
            )}

            {helpTab === "config" && (
              <div className="help-content">
                <div className="help-card">
                  <div className="help-card-title">{t("helper.sections.configs")}</div>
                  <ul className="schema-list">
                    {helpModal.block.config_inputs?.length ? (
                      helpModal.block.config_inputs.map((key) => (
                        <li key={key}>
                          <span className="schema-key">{key}</span>
                          <small>
                            {helpModal.block.input_schema?.[key]?.description || helpModal.block.input_schema?.[key]?.type || ""}
                          </small>
                        </li>
                      ))
                    ) : (
                      <li className="empty">{t("summary.noneConfig")}</li>
                    )}
                  </ul>
                </div>
              </div>
            )}

            {helpTab === "examples" && (
              <div className="help-content">
                {activeHelpModel?.examples?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.examples")}</div>
                    {activeHelpModel.examples.map((ex) => (
                      <pre className="help-diagram" key={ex}>
                        {ex}
                      </pre>
                    ))}
                  </div>
                ) : null}

                {helpModal.block.name === "signal_filters" && library.filters?.length > 0 && (
                  <div className="help-card">
                    <div className="help-card-title">{t("summary.availableFilters")}</div>
                    <p className="component-hint">{t("summary.componentHint")}</p>
                    <ul className="component-list">
                      {library.filters.map((f) => (
                        <li
                          key={f.name}
                          role="button"
                          tabIndex={0}
                          onClick={() => addBlockToCanvas(helpModal.block, { filter_type: f.name })}
                          onKeyDown={(e) => e.key === "Enter" && addBlockToCanvas(helpModal.block, { filter_type: f.name })}
                        >
                          <span>{f.name}</span>
                          <small>{f.description}</small>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {helpModal.block.name === "curve_fitting" && library.curve_models?.length > 0 && (
                  <div className="help-card">
                    <div className="help-card-title">{t("summary.availableModels")}</div>
                    <p className="component-hint">{t("summary.componentHint")}</p>
                    <ul className="component-list">
                      {library.curve_models.map((m) => (
                        <li
                          key={m.name}
                          role="button"
                          tabIndex={0}
                          onClick={() => addBlockToCanvas(helpModal.block, { model_type: m.name })}
                          onKeyDown={(e) => e.key === "Enter" && addBlockToCanvas(helpModal.block, { model_type: m.name })}
                        >
                          <span>{m.name}</span>
                          <small>{m.description}</small>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {helpModal.block.name === "feature_extraction" && library.feature_extractors?.length > 0 && (
                  <div className="help-card">
                    <div className="help-card-title">{t("summary.availableExtractors")}</div>
                    <p className="component-hint">{t("summary.componentHint")}</p>
                    <ul className="component-list">
                      {library.feature_extractors.map((e) => (
                        <li
                          key={e.name}
                          role="button"
                          tabIndex={0}
                          onClick={() => addBlockToCanvas(helpModal.block, { extractor_type: e.name })}
                          onKeyDown={(ev) => ev.key === "Enter" && addBlockToCanvas(helpModal.block, { extractor_type: e.name })}
                        >
                          <span>{e.name}</span>
                          <small>{e.description}</small>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {helpTab === "troubleshooting" && (
              <div className="help-content">
                {activeHelpModel?.errors?.length ? (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.errors")}</div>
                    <ul className="help-list">
                      {activeHelpModel.errors.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <div className="help-card">
                    <div className="help-card-title">{t("helper.sections.errors")}</div>
                    <div className="help-card-text">{t("results.enableDebugHint")}</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}


    </>
  );
}
