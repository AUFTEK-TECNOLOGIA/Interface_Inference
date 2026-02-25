import { memo } from "react";

const HELP_TABS = [
  { id: "overview", labelKey: "helper.tabs.overview" },
  { id: "io", labelKey: "helper.tabs.io" },
  { id: "config", labelKey: "helper.tabs.config" },
  { id: "examples", labelKey: "helper.tabs.examples" },
  { id: "troubleshooting", labelKey: "helper.tabs.troubleshooting" },
];

function HelpSchemaCard({ title, items, emptyText, schema }) {
  return (
    <div className="help-card">
      <div className="help-card-title">{title}</div>
      <ul className="schema-list">
        {items?.length ? (
          items.map((key) => (
            <li key={key}>
              <span className="schema-key">{key}</span>
              <small>{schema(key)}</small>
            </li>
          ))
        ) : (
          <li className="empty">{emptyText}</li>
        )}
      </ul>
    </div>
  );
}

function HelpComponentOptionsCard({ title, hint, items, onSelect }) {
  if (!items?.length) return null;

  return (
    <div className="help-card">
      <div className="help-card-title">{title}</div>
      <p className="component-hint">{hint}</p>
      <ul className="component-list">
        {items.map((item) => (
          <li
            key={item.name}
            role="button"
            tabIndex={0}
            onClick={() => onSelect(item)}
            onKeyDown={(e) => e.key === "Enter" && onSelect(item)}
          >
            <span>{item.name}</span>
            <small>{item.description}</small>
          </li>
        ))}
      </ul>
    </div>
  );
}

function HelpHelperModal({
  t,
  helpModal,
  closeHelpModal,
  activeHelpModel,
  addBlockToCanvas,
  helpTab,
  setHelpTab,
  library,
}) {
  if (!helpModal.open || !helpModal.block) return null;

  const block = helpModal.block;

  return (
    <div className="helper-modal" onClick={closeHelpModal} role="dialog" aria-modal="true">
      <div className="helper-modal-inner helper-modal-inner--help" onClick={(e) => e.stopPropagation()}>
        <div className="help-modal-header">
          <div className="help-modal-title">
            <div className="help-modal-icon" style={activeHelpModel?.color ? { background: activeHelpModel.color } : undefined}>
              {activeHelpModel?.icon || "B"}
            </div>
            <div>
              <div className="help-modal-title-text">{activeHelpModel?.title || block.name}</div>
              <div className="help-modal-subtitle">{activeHelpModel?.subtitle || block.name}</div>
            </div>
          </div>
          <div className="help-modal-actions">
            <button className="btn" type="button" onClick={() => addBlockToCanvas(block)}>
              {t("actions.add")}
            </button>
            <button className="helper-modal-close" type="button" onClick={closeHelpModal}>
              {t("actions.close")}
            </button>
          </div>
        </div>

        <div className="help-tabs" role="tablist" aria-label={t("helper.title")}>
          {HELP_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`help-tab${helpTab === tab.id ? " active" : ""}`}
              role="tab"
              aria-selected={helpTab === tab.id}
              onClick={() => setHelpTab(tab.id)}
            >
              {t(tab.labelKey)}
            </button>
          ))}
        </div>

        {helpTab === "overview" && (
          <div className="help-content">
            <div className="help-card">
              <div className="help-card-title">{t("helper.sections.what")}</div>
              <div className="help-card-text">{activeHelpModel?.what || block.description}</div>
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
            <HelpSchemaCard
              title={t("helper.sections.inputs")}
              items={block.data_inputs}
              emptyText={t("summary.noneInput")}
              schema={(key) => block.input_schema?.[key]?.type || "any"}
            />
            <HelpSchemaCard
              title={t("helper.sections.outputs")}
              items={block.data_outputs}
              emptyText={t("summary.noneOutput")}
              schema={(key) => block.output_schema?.[key]?.type || "any"}
            />
          </div>
        )}

        {helpTab === "config" && (
          <div className="help-content">
            <HelpSchemaCard
              title={t("helper.sections.configs")}
              items={block.config_inputs}
              emptyText={t("summary.noneConfig")}
              schema={(key) => block.input_schema?.[key]?.description || block.input_schema?.[key]?.type || ""}
            />
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

            {block.name === "signal_filters" && (
              <HelpComponentOptionsCard
                title={t("summary.availableFilters")}
                hint={t("summary.componentHint")}
                items={library.filters}
                onSelect={(item) => addBlockToCanvas(block, { filter_type: item.name })}
              />
            )}

            {block.name === "curve_fitting" && (
              <HelpComponentOptionsCard
                title={t("summary.availableModels")}
                hint={t("summary.componentHint")}
                items={library.curve_models}
                onSelect={(item) => addBlockToCanvas(block, { model_type: item.name })}
              />
            )}

            {block.name === "feature_extraction" && (
              <HelpComponentOptionsCard
                title={t("summary.availableExtractors")}
                hint={t("summary.componentHint")}
                items={library.feature_extractors}
                onSelect={(item) => addBlockToCanvas(block, { extractor_type: item.name })}
              />
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
  );
}

export default memo(HelpHelperModal);
