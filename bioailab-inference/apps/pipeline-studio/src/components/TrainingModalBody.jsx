import { TRAINING_ALGORITHM_ITEMS, TRAINING_ALGO_PARAM_SCHEMA, REGRESSION_ITEMS, buildTrainingParamsForAlgorithm } from "../modulos/trainingModule";

export default function TrainingModalBody({
  t,
  trainModal,
  setTrainModal,
  parseExperimentIdsText,
  trainModelsDraft,
  setTrainModelsDraft,
  nodes,
  setDatasetSelectorOpen,
  setTrainBlockModal,
  setCandidatesModal,
  setNodes,
  runTraining,
}) {

    const step = Number.isFinite(Number(trainModal.step)) ? Number(trainModal.step) : 0;
    const protocolId = String(trainModal.protocolId || "").trim();
    const experimentIds = parseExperimentIdsText(trainModal.experimentIdsText);
    const enabledModels = Object.values(trainModelsDraft || {}).filter((s) => s?.enabled !== false).length;
    const totalModels = Object.keys(trainModelsDraft || {}).length;

    const stepItems = [
      { index: 0, title: t("training.steps.data"), desc: t("training.steps.dataDesc") },
      { index: 1, title: t("training.steps.models"), desc: t("training.steps.modelsDesc") },
      { index: 2, title: t("training.steps.review"), desc: t("training.steps.reviewDesc") },
    ];

    const canProceedFromData = () => Boolean(protocolId) && experimentIds.length > 0;
    const canProceedFromModels = () => enabledModels > 0;

    const goNext = () => {
      if (trainModal.running) return;
      if (step === 0 && !canProceedFromData()) {
        setTrainModal((prev) => ({ ...prev, error: t("training.validation.data") }));
        return;
      }
      if (step === 1 && !canProceedFromModels()) {
        setTrainModal((prev) => ({ ...prev, error: t("training.validation.models") }));
        return;
      }
      setTrainModal((prev) => ({ ...prev, step: Math.min(2, (Number(prev.step) || 0) + 1), error: "" }));
    };

    const goBack = () => {
      if (trainModal.running) return;
      setTrainModal((prev) => ({ ...prev, step: Math.max(0, (Number(prev.step) || 0) - 1), error: "" }));
    };

    const algorithmItems = [
      { key: "ridge", label: "Ridge" },
      { key: "rf", label: "Random Forest" },
      { key: "gbm", label: "Gradient Boosting" },
      { key: "svr", label: "SVR" },
      { key: "mlp", label: "MLP" },
    ];

    const algoParamSchema = {
      ridge: {
        defaults: { alpha: 1.0 },
        fields: [
          { key: "alpha", label: "alpha", type: "number", step: 0.1, min: 0 },
        ],
        gridDefaults: { alpha: [0.1, 1.0, 10.0] },
      },
      rf: {
        defaults: { n_estimators: 300, max_depth: null, n_jobs: -1 },
        fields: [
          { key: "n_estimators", label: "n_estimators", type: "number", step: 50, min: 10 },
          { key: "max_depth", label: "max_depth", type: "number", step: 1, min: 1, allowNull: true },
          { key: "n_jobs", label: "n_jobs", type: "number", step: 1, min: -1 },
        ],
        gridDefaults: { n_estimators: [200, 400], max_depth: [null, 6, 12] },
      },
      gbm: {
        defaults: { n_estimators: 400, learning_rate: 0.03, max_depth: 3 },
        fields: [
          { key: "n_estimators", label: "n_estimators", type: "number", step: 50, min: 10 },
          { key: "learning_rate", label: "learning_rate", type: "number", step: 0.01, min: 0.0001 },
          { key: "max_depth", label: "max_depth", type: "number", step: 1, min: 1 },
        ],
        gridDefaults: { n_estimators: [200, 400], learning_rate: [0.01, 0.03], max_depth: [2, 3, 4] },
      },
      svr: {
        defaults: { kernel: "rbf", C: 1.0, epsilon: 0.1 },
        fields: [
          { key: "kernel", label: "kernel", type: "select", options: ["rbf", "linear", "poly", "sigmoid"] },
          { key: "C", label: "C", type: "number", step: 0.5, min: 0.0 },
          { key: "epsilon", label: "epsilon", type: "number", step: 0.05, min: 0.0 },
        ],
        gridDefaults: { C: [0.5, 1.0, 2.0], epsilon: [0.05, 0.1, 0.2], kernel: ["rbf", "linear"] },
      },
      mlp: {
        defaults: { hidden_layer_sizes: "128,64", max_iter: 800, learning_rate_init: 0.001 },
        fields: [
          { key: "hidden_layer_sizes", label: "hidden_layer_sizes", type: "text", placeholder: "Ex: 128,64" },
          { key: "max_iter", label: "max_iter", type: "number", step: 50, min: 50 },
          { key: "learning_rate_init", label: "learning_rate_init", type: "number", step: 0.0005, min: 0.00001 },
        ],
        gridDefaults: { hidden_layer_sizes: ["128,64", "256,128"], max_iter: [500, 800], learning_rate_init: [0.001, 0.0005] },
      },
    };

    return (
      <div className="training-stepper">
        <div className="training-steps" aria-label={t("training.steps.label")}>
          {stepItems.map((s) => {
            const state = step === s.index ? "active" : step > s.index ? "done" : "upcoming";
            return (
              <button
                key={s.index}
                type="button"
                className={`training-step ${state}`}
                disabled={trainModal.running || (s.index > step)}
                onClick={() => {
                  if (trainModal.running) return;
                  if (s.index > step) return;
                  setTrainModal((prev) => ({ ...prev, step: s.index, error: "" }));
                }}
              >
                <span className="training-step-indicator" aria-hidden="true">{s.index + 1}</span>
                <span className="training-step-text">
                  <span className="training-step-title">{s.title}</span>
                  <span className="training-step-desc">{s.desc}</span>
                </span>
              </button>
            );
          })}
        </div>

        <div className="training-content">
          {step === 0 && (
            <div className="training-card">
              <div className="training-card-title">{t("training.steps.data")}</div>
              <div className="training-card-subtitle">{t("training.steps.dataDesc")}</div>

              <div className="training-form-grid">
                <div className="workspace-field">
                  <label>{t("training.protocolIdLabel") || "Protocol ID"}</label>
                  <input
                    value={trainModal.protocolId}
                    onChange={(e) => setTrainModal((prev) => ({ ...prev, protocolId: e.target.value }))}
                    placeholder={t("training.protocolIdPlaceholder") || "ID do protocolo"}
                    disabled={trainModal.running}
                    autoFocus
                  />
                </div>

                <div className="workspace-field">
                  <label>{t("training.yTransformLabel")}</label>
                  <div className="training-segmented">
                    <button
                      type="button"
                      className={`training-segment ${trainModal.yTransform === "log10p" ? "is-active" : ""}`}
                      disabled={trainModal.running}
                      onClick={() => setTrainModal((prev) => ({ ...prev, yTransform: "log10p" }))}
                    >
                      {t("training.yTransformLog10p")}
                    </button>
                    <button
                      type="button"
                      className={`training-segment ${trainModal.yTransform === "none" ? "is-active" : ""}`}
                      disabled={trainModal.running}
                      onClick={() => setTrainModal((prev) => ({ ...prev, yTransform: "none" }))}
                    >
                      {t("training.yTransformNone")}
                    </button>
                  </div>
                </div>
              </div>

              <div className="workspace-field">
                <label>{t("training.experimentIdsLabel")}</label>
                
                {/* Botão para abrir o seletor de datasets */}
                <div
                  className="training-dataset-trigger"
                  onClick={() => {
                    if (!trainModal.running) {
                      setDatasetSelectorOpen(true);
                    }
                  }}
                  style={{ cursor: trainModal.running ? "not-allowed" : "pointer", opacity: trainModal.running ? 0.6 : 1 }}
                >
                  <div className="training-dataset-info">
                    <div className="training-dataset-label">
                      {parseExperimentIdsText(trainModal.experimentIdsText).length > 0
                        ? t("datasets.selectedCount", { count: parseExperimentIdsText(trainModal.experimentIdsText).length })
                        : "Selecionar Dataset"}
                    </div>
                    <div className="training-dataset-count">
                      {trainModal.protocolId?.trim()
                        ? t("datasets.subtitle", { protocolId: trainModal.protocolId })
                        : "Escolha um dataset existente ou crie um novo"}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="training-dataset-btn"
                    disabled={trainModal.running}
                  >
                    Abrir
                  </button>
                </div>

                {/* Textarea alternativo para colar IDs manualmente */}
                <details className="training-manual-ids" style={{ marginTop: 8 }}>
                  <summary style={{ fontSize: 12, color: "var(--md-sys-color-outline)", cursor: "pointer" }}>
                    {t("training.manualIdsToggle") || "Ou cole os IDs manualmente"}
                  </summary>
                  <textarea
                    className="training-textarea"
                    value={trainModal.experimentIdsText}
                    onChange={(e) => setTrainModal((prev) => ({ ...prev, experimentIdsText: e.target.value }))}
                    placeholder={t("training.experimentIdsHint")}
                    disabled={trainModal.running}
                    style={{ marginTop: 8 }}
                  />
                </details>
                <small>{t("training.experimentIdsHint")}</small>
              </div>

              <div className="training-advanced-summary">
                <div className="training-advanced-summary-text">
                  <div className="training-advanced-summary-title">{t("training.advanced.title")}</div>
                  <div className="training-advanced-summary-subtitle">
                    {t("training.advanced.summary", {
                      metric: trainModal.selectionMetric || "rmse",
                      trials: trainModal.maxTrials ?? 60,
                      testSize: trainModal.testSize ?? 0.2,
                    })}
                  </div>
                </div>
              </div>

              <div className="training-advanced-card">
                <div className="training-advanced-card-title">{t("training.advanced.title")}</div>

                <div className="training-advanced-grid">
                  <div className="workspace-field" style={{ marginBottom: 0 }}>
                    <label>{t("training.advanced.selectionMetric")}</label>
                    <select
                      value={trainModal.selectionMetric}
                      disabled={trainModal.running}
                      onChange={(e) => setTrainModal((prev) => ({ ...prev, selectionMetric: e.target.value }))}
                    >
                      <option value="rmse">RMSE</option>
                      <option value="mae">MAE</option>
                      <option value="r2">R2</option>
                    </select>
                    <small>{t("training.advanced.selectionMetricHint")}</small>
                  </div>

                  <div className="workspace-field" style={{ marginBottom: 0 }}>
                    <label>{t("training.advanced.maxTrials")}</label>
                    <input
                      type="number"
                      min="1"
                      max="500"
                      step="1"
                      value={trainModal.maxTrials}
                      disabled={trainModal.running}
                      onChange={(e) => setTrainModal((prev) => ({ ...prev, maxTrials: e.target.value }))}
                    />
                    <small>{t("training.advanced.maxTrialsHint")}</small>
                  </div>

                  <div className="workspace-field" style={{ marginBottom: 0 }}>
                    <label>{t("training.advanced.testSize")}</label>
                    <input
                      type="number"
                      min="0"
                      max="0.8"
                      step="0.05"
                      value={trainModal.testSize}
                      disabled={trainModal.running}
                      onChange={(e) => setTrainModal((prev) => ({ ...prev, testSize: e.target.value }))}
                    />
                    <small>{t("training.advanced.testSizeHint")}</small>
                  </div>

                  <div className="workspace-field" style={{ marginBottom: 0 }}>
                    <label>{t("training.advanced.randomState")}</label>
                    <input
                      type="number"
                      step="1"
                      value={trainModal.randomState}
                      disabled={trainModal.running}
                      onChange={(e) => setTrainModal((prev) => ({ ...prev, randomState: e.target.value }))}
                    />
                    <small>{t("training.advanced.randomStateHint")}</small>
                  </div>
                </div>

                <div className="workspace-field" style={{ marginTop: 10, marginBottom: 0 }}>
                  <label className={`training-switch ${trainModal.permImportance ? "is-on" : ""}`}>
                    <input
                      type="checkbox"
                      checked={!!trainModal.permImportance}
                      disabled={trainModal.running}
                      onChange={(e) => setTrainModal((prev) => ({ ...prev, permImportance: e.target.checked }))}
                    />
                    <span className="training-switch-track" aria-hidden="true" />
                    <span className="training-switch-text">{t("training.advanced.permImportance")}</span>
                  </label>
                </div>

                {trainModal.permImportance && (
                  <div className="workspace-field" style={{ marginTop: 10, marginBottom: 0 }}>
                    <label>{t("training.advanced.permRepeats")}</label>
                    <input
                      type="number"
                      min="1"
                      max="50"
                      step="1"
                      value={trainModal.permRepeats}
                      disabled={trainModal.running}
                      onChange={(e) => setTrainModal((prev) => ({ ...prev, permRepeats: e.target.value }))}
                    />
                    <small>{t("training.advanced.permRepeatsHint")}</small>
                  </div>
                )}
              </div>

              {trainModal.error && <div className="training-banner training-banner--error">{typeof trainModal.error === 'string' ? trainModal.error : JSON.stringify(trainModal.error)}</div>}

              <div className="training-inline-metrics">
                <span className="training-metric">{t("training.metrics.experiments", { count: experimentIds.length })}</span>
                <span className="training-metric">{t("training.metrics.models", { enabled: enabledModels, total: totalModels })}</span>
              </div>
            </div>
          )}

          {step === 1 && (
            <div className="training-card">
              <div className="training-card-title">{t("training.steps.models")}</div>
              <div className="training-card-subtitle">{t("training.steps.modelsDesc")}</div>

              {trainModal.error && <div className="training-banner training-banner--error">{typeof trainModal.error === 'string' ? trainModal.error : JSON.stringify(trainModal.error)}</div>}

              <div className="training-models-list" role="list">
                {Object.keys(trainModelsDraft || {}).length === 0 ? (
                  <div className="training-empty">{t("training.emptyModels")}</div>
                ) : (
                  Object.entries(trainModelsDraft || {}).map(([stepId, spec]) => {
                    const node = (nodes || []).find((n) => n.id === stepId);
                    const label = node?.data?.label || stepId;
                    const blockName = node?.data?.blockName || "";
                    const enabled = spec?.enabled !== false;
                    const algorithms = Array.isArray(spec?.algorithms) && spec.algorithms.length ? spec.algorithms : ["ridge"];
                    const algoLabels = algorithms.map((k) => {
                      const key = String(k || "").trim().toLowerCase();
                      return TRAINING_ALGORITHM_ITEMS.find((a) => a.key === key)?.label || key || "ridge";
                    });

                    return (
                      <div className={`training-model-row ${enabled ? "" : "is-disabled"}`} key={stepId} role="listitem">
                        <div className="training-model-head">
                          <div className="training-model-icon" aria-hidden="true">ML</div>
                          <div className="training-model-title">
                            <div className="training-model-name" title={label}>{label}</div>
                            <div className="training-model-meta">
                              <span className="training-model-chip" title={blockName}>{blockName || "ml"}</span>
                              <span className="training-model-id" title={stepId}>{stepId}</span>
                              {algoLabels.slice(0, 3).map((name) => (
                                <span className="training-model-chip" key={name} title={t("training.algorithmsLabel")}>{name}</span>
                              ))}
                              {algoLabels.length > 3 && (
                                <span className="training-model-chip" title={t("training.algorithmsLabel")}>+{algoLabels.length - 3}</span>
                              )}
                            </div>
                          </div>
                          <label className="training-inline-toggle" title={t("training.blockEnabled")}>
                            <input
                              type="checkbox"
                              checked={enabled}
                              disabled={trainModal.running}
                              onChange={(e) =>
                                setTrainModelsDraft((prev) => ({
                                  ...prev,
                                  [stepId]: { ...(prev?.[stepId] || {}), enabled: e.target.checked },
                                }))
                              }
                            />
                            <span className="training-inline-toggle-label">{t("training.blockEnabled")}</span>
                          </label>
                        </div>

                        <div className="training-model-actions">
                          <button
                            type="button"
                            className="training-advanced"
                            disabled={trainModal.running || !enabled}
                            onClick={() => setTrainBlockModal({ open: true, stepId })}
                          >
                            {t("training.configureBlock")}
                          </button>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              <div className="training-inline-metrics">
                <span className="training-metric">{t("training.metrics.models", { enabled: enabledModels, total: totalModels })}</span>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="training-card">
              <div className="training-card-title">{t("training.steps.review")}</div>
              <div className="training-card-subtitle">{t("training.steps.reviewDesc")}</div>

              <div className="training-review-grid">
                <div className="training-review-item">
                  <div className="training-review-label">{t("training.review.protocolId") || "Protocol ID"}</div>
                  <div className="training-review-value">{protocolId || "-"}</div>
                </div>
                <div className="training-review-item">
                  <div className="training-review-label">{t("training.review.experiments")}</div>
                  <div className="training-review-value">{experimentIds.length}</div>
                </div>
                <div className="training-review-item">
                  <div className="training-review-label">{t("training.review.models")}</div>
                  <div className="training-review-value">{enabledModels}/{totalModels}</div>
                </div>
                <div className="training-review-item">
                  <div className="training-review-label">{t("training.review.transform")}</div>
                  <div className="training-review-value">{trainModal.yTransform === "log10p" ? t("training.yTransformLog10p") : t("training.yTransformNone")}</div>
                </div>
              </div>

              <div className="workspace-field" style={{ marginTop: 10 }}>
                <label>{t("training.applyToPipelineLabel")}</label>
                <label className={`training-switch ${trainModal.applyToPipeline ? "is-on" : ""}`}>
                  <input
                    type="checkbox"
                    checked={!!trainModal.applyToPipeline}
                    disabled={trainModal.running}
                    onChange={(e) => setTrainModal((prev) => ({ ...prev, applyToPipeline: e.target.checked }))}
                  />
                  <span className="training-switch-track" aria-hidden="true" />
                  <span className="training-switch-text">
                    {trainModal.applyToPipeline ? t("training.applyToPipelineOn") : t("training.applyToPipelineOff")}
                  </span>
                </label>
              </div>

              <div className="workspace-field" style={{ marginTop: 10 }}>
                <label>{t("training.gridSearchManualLabel")}</label>
                <label className={`training-switch ${trainModal.gridSearchManual ? "is-on" : ""}`}>
                  <input
                    type="checkbox"
                    checked={!!trainModal.gridSearchManual}
                    disabled={trainModal.running}
                    onChange={(e) => setTrainModal((prev) => ({ ...prev, gridSearchManual: e.target.checked }))}
                  />
                  <span className="training-switch-track" aria-hidden="true" />
                  <span className="training-switch-text">
                    {trainModal.gridSearchManual ? t("training.gridSearchManualOn") : t("training.gridSearchManualOff")}
                  </span>
                </label>
                <small className="training-hint">{t("training.gridSearchManualHint")}</small>
              </div>

              {trainModal.applyToPipeline && (
                <div className="workspace-field" style={{ marginTop: 10 }}>
                  <label>{t("training.changeReasonLabel")}</label>
                  <input
                    value={trainModal.changeReason}
                    onChange={(e) => setTrainModal((prev) => ({ ...prev, changeReason: e.target.value }))}
                    placeholder={t("training.changeReasonPlaceholder")}
                    disabled={trainModal.running}
                  />
                </div>
              )}

              {trainModal.error && <div className="training-banner training-banner--error">{typeof trainModal.error === 'string' ? trainModal.error : JSON.stringify(trainModal.error)}</div>}

              {trainModal.result && (
                <div className={`training-banner ${trainModal.result?.success ? "training-banner--success" : "training-banner--info"}`}>
                  <div className="training-banner-title">{t("training.resultTitle")}</div>
                  {trainModal.result?.version && (
                    <div className="training-banner-sub">{t("training.versionActivated", { version: trainModal.result.version })}</div>
                  )}
                  <div className="training-banner-chips">
                    <span className="training-chip training-chip--success">{t("training.trainedCount", { count: (trainModal.result.trained || []).length })}</span>
                    <span className="training-chip training-chip--warning">{t("training.skippedModelsCount", { count: (trainModal.result.skipped || []).length })}</span>
                    <span className="training-chip training-chip--error">{t("training.errorsCount", { count: (trainModal.result.errors || []).length })}</span>
                  </div>
                  
                  {/* Modelos treinados com sucesso - com botões para ver candidatos */}
                  {(trainModal.result.gridSearchResults || []).filter(r => r.status === "trained").length > 0 && (
                    <div className="training-banner-section">
                      <div className="training-banner-section-title" style={{ color: "var(--success-color, #4caf50)" }}>
                        ✓ {t("training.trainedModels")}
                      </div>
                      <div className="training-results-grid">
                        {(trainModal.result.gridSearchResults || []).filter(r => r.status === "trained").map((res, idx) => (
                          <div key={idx} className="training-result-card">
                            <div className="training-result-card-header">
                              <span className="training-result-card-title">{res.step_id}</span>
                              <span className="training-result-card-badge">{res.n_candidates} candidatos</span>
                            </div>
                            <div className="training-result-card-info">
                              <span>{res.n_samples} amostras</span>
                              {res.candidates?.[res.best_index] && (
                                <span>Melhor: {res.candidates[res.best_index].algorithm} (R²: {(res.candidates[res.best_index].metrics?.val_r2 ?? res.candidates[res.best_index].metrics?.train_r2 ?? 0).toFixed(3)})</span>
                              )}
                            </div>
                            <button
                              type="button"
                              className="training-result-card-btn"
                              onClick={() => {
                                setTrainModal((prev) => ({ ...prev, open: false }));
                                setCandidatesModal({
                                  open: true,
                                  sessionPath: res.session_path,
                                  stepId: res.step_id,
                                });
                              }}
                            >
                              📊 Ver candidatos e gráficos
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* ============================================= */}
                  {/* RESULTADOS DE REGRESSÃO MATEMÁTICA           */}
                  {/* ============================================= */}
                  {trainModal.result?.isRegression && (trainModal.result.regressionResults || []).filter(r => r.status === "trained").length > 0 && (
                    <div className="training-banner-section">
                      <div className="training-banner-section-title" style={{ color: "var(--success-color, #4caf50)" }}>
                        📈 Regressões Ajustadas
                      </div>
                      <div className="training-results-grid">
                        {(trainModal.result.regressionResults || []).filter(r => r.status === "trained").map((res, idx) => (
                          <div key={idx} className="training-result-card" style={{ background: "var(--bg-tertiary)" }}>
                            <div className="training-result-card-header">
                              <span className="training-result-card-title">{res.step_id}</span>
                              <span className="training-result-card-badge" style={{ background: "var(--primary-color)" }}>
                                {res.regression_type}
                              </span>
                            </div>
                            <div className="training-result-card-info" style={{ flexDirection: "column", alignItems: "flex-start", gap: 6 }}>
                              <code style={{ fontSize: 13, background: "var(--bg-secondary)", padding: "4px 8px", borderRadius: 4, width: "100%", overflow: "auto" }}>
                                {res.equation}
                              </code>
                              <div style={{ display: "flex", gap: 12, fontSize: 12 }}>
                                <span>R² = <strong>{(res.metrics?.r2 ?? 0).toFixed(4)}</strong></span>
                                <span>RMSE = {(res.metrics?.rmse ?? 0).toFixed(2)}</span>
                                <span>n = {res.n_samples}</span>
                              </div>
                            </div>
                            {res.comparison && res.comparison.length > 1 && (
                              <details style={{ marginTop: 8, fontSize: 12 }}>
                                <summary style={{ cursor: "pointer", color: "var(--text-secondary)" }}>
                                  Comparativo de regressões
                                </summary>
                                <table style={{ width: "100%", marginTop: 6, fontSize: 11, borderCollapse: "collapse" }}>
                                  <thead>
                                    <tr style={{ borderBottom: "1px solid var(--border-color)" }}>
                                      <th style={{ textAlign: "left", padding: 4 }}>Tipo</th>
                                      <th style={{ textAlign: "right", padding: 4 }}>R²</th>
                                      <th style={{ textAlign: "right", padding: 4 }}>RMSE</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {res.comparison.map((c, cidx) => (
                                      <tr key={cidx} style={{ background: c.selected ? "var(--primary-color-alpha)" : "transparent" }}>
                                        <td style={{ padding: 4 }}>{c.type} {c.selected ? "✓" : ""}</td>
                                        <td style={{ textAlign: "right", padding: 4 }}>{(c.r2 ?? 0).toFixed(4)}</td>
                                        <td style={{ textAlign: "right", padding: 4 }}>{(c.rmse ?? 0).toFixed(2)}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </details>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Modelos pulados por falta de dados */}
                  {(trainModal.result.skipped || []).length > 0 && (
                    <div className="training-banner-section">
                      <div className="training-banner-section-title" style={{ color: "var(--warning-color, #ff9800)" }}>
                        ⊘ {t("training.skippedModels")}
                      </div>
                      <ul className="training-banner-list training-banner-list--muted">
                        {(trainModal.result.skipped || []).map((s, idx) => (
                          <li key={idx}>{s.step_id}: {s.reason}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {/* Botão geral para ver todos os candidatos (modo manual desabilitado) */}
                  {trainModal.result?.session_path && !trainModal.result.gridSearchResults && (
                    <div className="training-banner-actions">
                      <button
                        type="button"
                        className="btn-secondary training-banner-btn"
                        onClick={() => {
                          setTrainModal((prev) => ({ ...prev, open: false }));
                          setCandidatesModal({
                            open: true,
                            sessionPath: trainModal.result.session_path,
                            stepId: trainModal.result.step_id || "",
                          });
                        }}
                      >
                        {t("training.viewCandidates")}
                      </button>
                    </div>
                  )}
                </div>
              )}

              {trainModal.result && (trainModal.result.errors || []).length > 0 && (
                <div className="training-errors">
                  <div className="training-errors-title">{t("training.errorsTitle")}</div>
                  <ul className="results-error-list">
                    {(trainModal.result.errors || []).map((e, idx) => (
                      <li key={idx}>{e}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {trainBlockModal?.open &&
            createPortal(
              <div className="training-block-overlay" role="dialog" aria-modal="true" onMouseDown={() => setTrainBlockModal({ open: false, stepId: "" })}>
                <div className="training-block-modal" onMouseDown={(e) => e.stopPropagation()}>
                  {(() => {
                  const stepId = String(trainBlockModal.stepId || "");
                  const spec = trainModelsDraft?.[stepId] || {};
                  const enabled = spec?.enabled !== false;
                  const node = (nodes || []).find((n) => n.id === stepId);
                  const label = node?.data?.label || stepId;
                  const blockName = String(node?.data?.blockName || "");
                  const isForecaster = blockName === "ml_forecaster_series";
                  const algorithms = Array.isArray(spec?.algorithms) && spec.algorithms.length ? spec.algorithms : ["ridge"];
                  const activeAlgorithmRaw = String(spec?.activeAlgorithm || algorithms[0] || "ridge").trim().toLowerCase();
                  const activeAlgorithm = algorithms.includes(activeAlgorithmRaw) ? activeAlgorithmRaw : String(algorithms[0] || "ridge").trim().toLowerCase();
                  const schema = TRAINING_ALGO_PARAM_SCHEMA[activeAlgorithm] || TRAINING_ALGO_PARAM_SCHEMA.ridge;

                  const updateSpec = (patch) => {
                    setTrainModelsDraft((prev) => ({
                      ...prev,
                      [stepId]: { ...(prev?.[stepId] || {}), ...patch },
                    }));
                  };

                  const toggleAlgorithm = (algoKey) => {
                    const key = String(algoKey || "").trim().toLowerCase();
                    if (!key) return;
                    const current = Array.isArray(spec?.algorithms) && spec.algorithms.length ? spec.algorithms : ["ridge"];
                    const has = current.includes(key);
                    if (has && current.length <= 1) return;

                    const next = has ? current.filter((a) => a !== key) : [...current, key];
                    const nextActive = has && activeAlgorithm === key ? next[0] : activeAlgorithm;
                    const paramsByAlgorithm = { ...(spec?.paramsByAlgorithm || {}) };
                    if (!paramsByAlgorithm[key]) {
                      paramsByAlgorithm[key] = buildTrainingParamsForAlgorithm(key);
                    }
                    updateSpec({ algorithms: next, activeAlgorithm: nextActive, paramsByAlgorithm });
                  };

                  const setActiveAlgo = (algoKey) => {
                    const key = String(algoKey || "").trim().toLowerCase();
                    if (!key) return;
                    updateSpec({ activeAlgorithm: key });
                  };

                  const setParamRow = (algoKey, paramKey, patch) => {
                    const algo = String(algoKey || "ridge").trim().toLowerCase() || "ridge";
                    const key = String(paramKey || "").trim();
                    if (!key) return;
                    setTrainModelsDraft((prev) => {
                      const prevSpec = prev?.[stepId] || {};
                      const paramsByAlgorithm = { ...(prevSpec?.paramsByAlgorithm || {}) };
                      const algoRows = { ...(paramsByAlgorithm[algo] || buildTrainingParamsForAlgorithm(algo)) };
                      algoRows[key] = { ...(algoRows[key] || {}), ...(patch || {}) };
                      paramsByAlgorithm[algo] = algoRows;
                      return { ...prev, [stepId]: { ...prevSpec, paramsByAlgorithm } };
                    });
                  };

                  const resetAlgorithmParams = (algoKey) => {
                    const algo = String(algoKey || "ridge").trim().toLowerCase() || "ridge";
                    setTrainModelsDraft((prev) => {
                      const prevSpec = prev?.[stepId] || {};
                      const paramsByAlgorithm = { ...(prevSpec?.paramsByAlgorithm || {}) };
                      paramsByAlgorithm[algo] = buildTrainingParamsForAlgorithm(algo);
                      return { ...prev, [stepId]: { ...prevSpec, paramsByAlgorithm } };
                    });
                  };

                  const rows = spec?.paramsByAlgorithm?.[activeAlgorithm] || buildTrainingParamsForAlgorithm(activeAlgorithm);

                  const disabled = trainModal.running || !enabled;
                  const updateForecasterConfig = (key, value) => {
                    setNodes((prev) =>
                      (prev || []).map((n) => {
                        if (n.id !== stepId) return n;
                        const prevConfig = n?.data?.config || {};
                        return { ...n, data: { ...n.data, config: { ...prevConfig, [key]: value } } };
                      })
                    );
                  };
                  const algoLabel = (algoKey) => {
                    const key = String(algoKey || "").trim().toLowerCase();
                    return TRAINING_ALGORITHM_ITEMS.find((a) => a.key === key)?.label || key || "ridge";
                  };

                  // UI atual: tabela de parâmetros (fixo/grade por parâmetro), com seleção de 1+ algoritmos
                  return (
                    <>
                      <div className="training-block-header">
                        <div>
                          <div className="training-block-title">{t("training.configureBlock")}</div>
                          <div className="training-block-sub">{label}</div>
                        </div>
                        <button className="workspace-home-close" type="button" onClick={() => setTrainBlockModal({ open: false, stepId: "" })}>
                          {t("actions.close")}
                        </button>
                      </div>

                      <div className="training-inline-card">
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
                          <div>
                            <div className="training-inline-card-title">
                              {isForecaster ? t("training.forecaster.blockTitle") : t("training.blockConfig.title")}
                            </div>
                            <div className="training-inline-card-subtitle">
                              {isForecaster ? t("training.forecaster.blockSubtitle") : t("training.blockConfig.subtitle")}
                            </div>
                          </div>
                          <label className="training-inline-toggle" title={t("training.blockEnabled")}>
                            <input type="checkbox" checked={enabled} disabled={trainModal.running} onChange={(e) => updateSpec({ enabled: e.target.checked })} />
                            <span className="training-inline-toggle-label">{t("training.blockEnabled")}</span>
                          </label>
                        </div>

                        {isForecaster && (
                          <div className="training-forecaster-grid" style={{ display: "grid", gap: 10, marginTop: 12 }}>
                            <div className="training-forecaster-card" style={{ border: "1px solid #e6e8f0", borderRadius: 14, padding: 12, background: "rgba(245,247,255,0.65)" }}>
                              <div style={{ fontWeight: 700, marginBottom: 6 }}>{t("training.forecaster.seriesConfigTitle")}</div>
                              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
                                <div className="workspace-field" style={{ marginBottom: 0 }}>
                                  <label>{t("training.forecaster.targetChannel")}</label>
                                  <input
                                    value={String(node?.data?.config?.target_channel || "")}
                                    disabled={trainModal.running}
                                    onChange={(e) => updateForecasterConfig("target_channel", e.target.value)}
                                    placeholder={t("training.forecaster.targetChannelPlaceholder")}
                                  />
                                  <small>{t("training.forecaster.targetChannelHint")}</small>
                                </div>
                                <div className="workspace-field" style={{ marginBottom: 0 }}>
                                  <label>{t("training.forecaster.window")}</label>
                                  <input
                                    type="number"
                                    value={String(node?.data?.config?.window ?? 30)}
                                    disabled={trainModal.running}
                                    min={1}
                                    max={2048}
                                    onChange={(e) => updateForecasterConfig("window", e.target.value === "" ? undefined : Number(e.target.value))}
                                  />
                                </div>
                                <div className="workspace-field" style={{ marginBottom: 0 }}>
                                  <label>{t("training.forecaster.horizon")}</label>
                                  <input
                                    type="number"
                                    value={String(node?.data?.config?.horizon ?? 1)}
                                    disabled={trainModal.running}
                                    min={1}
                                    max={2048}
                                    onChange={(e) => updateForecasterConfig("horizon", e.target.value === "" ? undefined : Number(e.target.value))}
                                  />
                                </div>
                                <div className="workspace-field" style={{ marginBottom: 0 }}>
                                  <label>{t("training.forecaster.inputChannels")}</label>
                                  <input
                                    value={Array.isArray(node?.data?.config?.input_channels) ? node.data.config.input_channels.join(", ") : String(node?.data?.config?.input_channels || "")}
                                    disabled={trainModal.running}
                                    onChange={(e) => updateForecasterConfig("input_channels", e.target.value)}
                                    placeholder={t("training.forecaster.inputChannelsPlaceholder")}
                                  />
                                  <small>{t("training.forecaster.inputChannelsHint")}</small>
                                </div>
                                <div className="workspace-field" style={{ marginBottom: 0 }}>
                                  <label>{t("training.forecaster.maxSamples")}</label>
                                  <input
                                    type="number"
                                    value={String(node?.data?.config?.max_samples ?? 2000)}
                                    disabled={trainModal.running}
                                    min={50}
                                    max={50000}
                                    onChange={(e) => updateForecasterConfig("max_samples", e.target.value === "" ? undefined : Number(e.target.value))}
                                  />
                                </div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* ============================================= */}
                        {/* SELEÇÃO: REGRESSÃO MATEMÁTICA vs ML          */}
                        {/* ============================================= */}
                        {!isForecaster && (
                          <div className="workspace-field" style={{ marginBottom: 14, marginTop: 12 }}>
                            <label>Tipo de Modelo</label>
                            <div className="training-chip-group">
                              <button
                                type="button"
                                className={`training-chip-select ${!trainModal.useRegression ? "is-selected" : ""}`}
                                disabled={trainModal.running}
                                onClick={() => setTrainModal((p) => ({ ...p, useRegression: false }))}
                              >
                                🤖 Machine Learning
                              </button>
                              <button
                                type="button"
                                className={`training-chip-select ${trainModal.useRegression ? "is-selected" : ""}`}
                                disabled={trainModal.running}
                                onClick={() => setTrainModal((p) => ({ ...p, useRegression: true }))}
                              >
                                📈 Regressão Matemática
                              </button>
                            </div>
                            <small>
                              {trainModal.useRegression
                                ? "Ajusta uma equação matemática simples (ex: y = ax + b). Mais interpretável e rápido."
                                : "Treina modelos de machine learning (Ridge, XGBoost, etc.). Mais preciso para dados complexos."}
                            </small>
                          </div>
                        )}

                        {/* ============================================= */}
                        {/* REGRESSÕES MATEMÁTICAS                       */}
                        {/* ============================================= */}
                        {trainModal.useRegression && !isForecaster && (
                          <div className="regression-section" style={{ marginBottom: 16, padding: 12, background: "var(--bg-tertiary)", borderRadius: 8 }}>
                            <div className="workspace-field" style={{ marginBottom: 10 }}>
                              <label>Tipo de Regressão</label>
                              <div className="training-chip-group">
                                {REGRESSION_ITEMS.map((item) => (
                                  <button
                                    key={item.key}
                                    type="button"
                                    className={`training-chip-select ${trainModal.regressionType === item.key ? "is-selected" : ""}`}
                                    disabled={trainModal.running || trainModal.regressionAutoSelect}
                                    onClick={() => setTrainModal((p) => ({ ...p, regressionType: item.key }))}
                                    title={`${item.equation} - ${item.description}`}
                                  >
                                    {item.label}
                                  </button>
                                ))}
                              </div>
                            </div>

                            <div className="workspace-field" style={{ marginBottom: 10 }}>
                              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                <input
                                  type="checkbox"
                                  checked={trainModal.regressionAutoSelect}
                                  disabled={trainModal.running}
                                  onChange={(e) => setTrainModal((p) => ({ ...p, regressionAutoSelect: e.target.checked }))}
                                />
                                Seleção automática (testa todos e escolhe o melhor R²)
                              </label>
                            </div>

                            {trainModal.regressionType === "polynomial" && !trainModal.regressionAutoSelect && (
                              <div className="workspace-field" style={{ marginBottom: 0 }}>
                                <label>Grau do Polinômio</label>
                                <input
                                  type="number"
                                  value={trainModal.polynomialDegree}
                                  min={2}
                                  max={10}
                                  disabled={trainModal.running}
                                  onChange={(e) => setTrainModal((p) => ({ ...p, polynomialDegree: Number(e.target.value) || 3 }))}
                                  style={{ width: 80 }}
                                />
                              </div>
                            )}

                            <div style={{ marginTop: 10, padding: 8, background: "var(--bg-secondary)", borderRadius: 6, fontSize: 13 }}>
                              <strong>Equação:</strong>{" "}
                              {REGRESSION_ITEMS.find((r) => r.key === trainModal.regressionType)?.equation || "y = ?"}
                              <br />
                              <small style={{ color: "var(--text-secondary)" }}>
                                {REGRESSION_ITEMS.find((r) => r.key === trainModal.regressionType)?.description}
                              </small>
                            </div>
                          </div>
                        )}

                        {/* ============================================= */}
                        {/* ALGORITMOS DE ML (quando não usar regressão) */}
                        {/* ============================================= */}
                        {!trainModal.useRegression && (
                        <>
                        <div className="workspace-field" style={{ marginBottom: 10, marginTop: 12 }}>
                          <label>{isForecaster ? t("training.forecaster.regressorLabel") : t("training.algorithmsLabel")}</label>
                          <div className="training-chip-group">
                            {TRAINING_ALGORITHM_ITEMS.map((item) => {
                              const selected = algorithms.includes(item.key);
                              return (
                                <button
                                  key={item.key}
                                  type="button"
                                  className={`training-chip-select ${selected ? "is-selected" : ""}`}
                                  disabled={disabled}
                                  onClick={() => toggleAlgorithm(item.key)}
                                >
                                  {item.label}
                                </button>
                              );
                            })}
                          </div>
                          <small>{t("training.blockConfig.algorithmsHint")}</small>
                        </div>

                        {algorithms.length > 1 && (
                          <div className="workspace-field" style={{ marginBottom: 10 }}>
                            <label>{t("training.blockConfig.activeAlgorithm")}</label>
                            <div className="training-chip-group">
                              {algorithms.map((k) => {
                                const key = String(k || "").trim().toLowerCase();
                                const selected = key === activeAlgorithm;
                                return (
                                  <button
                                    key={key}
                                    type="button"
                                    className={`training-chip-select ${selected ? "is-selected" : ""}`}
                                    disabled={disabled}
                                    onClick={() => setActiveAlgo(key)}
                                  >
                                    {algoLabel(key)}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        <div className="training-param-table-wrap" role="region" aria-label={t("training.paramsTable.aria")}>
                          <table className="training-param-table">
                            <thead>
                              <tr>
                                <th>{t("training.paramsTable.param")}</th>
                                <th>{t("training.paramsTable.mode")}</th>
                                <th>{t("training.paramsTable.value")}</th>
                                <th>{t("training.paramsTable.min")}</th>
                                <th>{t("training.paramsTable.max")}</th>
                                <th>{t("training.paramsTable.divisions")}</th>
                              </tr>
                            </thead>
                            <tbody>
                              {(schema.fields || []).map((field) => {
                                const row = rows?.[field.key] || {};
                                const mode = row?.mode === "grid" && field.grid ? "grid" : "fixed";
                                const isNull = !!row?.isNull && !!field.allowNull;
                                const effectiveValue =
                                  isNull ? "" : String(row?.value ?? (field.defaultValue === null ? "" : field.defaultValue ?? ""));

                                const valueInput = (() => {
                                  if (field.kind === "select") {
                                    return (
                                      <select
                                        value={effectiveValue}
                                        disabled={disabled || isNull}
                                        onChange={(e) => setParamRow(activeAlgorithm, field.key, { value: e.target.value, isNull: false })}
                                      >
                                        {(field.options || []).map((opt) => (
                                          <option key={opt} value={opt}>
                                            {opt}
                                          </option>
                                        ))}
                                      </select>
                                    );
                                  }

                                  if (field.kind === "text") {
                                    return (
                                      <input
                                        type="text"
                                        value={effectiveValue}
                                        placeholder={field.placeholder}
                                        disabled={disabled || isNull}
                                        onChange={(e) => setParamRow(activeAlgorithm, field.key, { value: e.target.value, isNull: false })}
                                      />
                                    );
                                  }

                                  return (
                                    <input
                                      type="number"
                                      value={effectiveValue}
                                      min={field.min}
                                      step={field.step || 1}
                                      disabled={disabled || isNull}
                                      onChange={(e) => setParamRow(activeAlgorithm, field.key, { value: e.target.value, isNull: false })}
                                    />
                                  );
                                })();

                                return (
                                  <tr key={field.key}>
                                    <td className="training-param-key">{field.label}</td>
                                    <td>
                                      {field.grid ? (
                                        <div className="training-mini-segment" role="group" aria-label={t("training.paramsTable.mode")}>
                                          <button
                                            type="button"
                                            className={`training-mini-segment-btn ${mode === "fixed" ? "is-active" : ""}`}
                                            disabled={disabled}
                                            onClick={() => setParamRow(activeAlgorithm, field.key, { mode: "fixed" })}
                                          >
                                            {t("training.paramsTable.fixed")}
                                          </button>
                                          <button
                                            type="button"
                                            className={`training-mini-segment-btn ${mode === "grid" ? "is-active" : ""}`}
                                            disabled={disabled}
                                            onClick={() => setParamRow(activeAlgorithm, field.key, { mode: "grid", isNull: false })}
                                          >
                                            {t("training.paramsTable.grid")}
                                          </button>
                                        </div>
                                      ) : (
                                        <span className="training-param-muted">{t("training.paramsTable.fixed")}</span>
                                      )}
                                    </td>
                                    <td>
                                      {mode === "fixed" ? (
                                        <div className="training-param-value">
                                          {field.allowNull && (
                                            <label className="training-null-toggle">
                                              <input
                                                type="checkbox"
                                                checked={isNull}
                                                disabled={disabled}
                                                onChange={(e) => setParamRow(activeAlgorithm, field.key, { isNull: e.target.checked })}
                                              />
                                              <span>{t("training.paramsTable.nullLabel")}</span>
                                            </label>
                                          )}
                                          {valueInput}
                                        </div>
                                      ) : field.kind === "select" ? (
                                        <div className="training-select-grid">
                                          <div className="training-select-grid-chips">
                                            {(field.options || []).map((opt) => {
                                              const option = String(opt);
                                              const selectedChoices = Array.isArray(row?.choices) ? row.choices.map((v) => String(v)) : [];
                                              const selected = selectedChoices.includes(option);
                                              return (
                                                <button
                                                  key={option}
                                                  type="button"
                                                  className={`training-chip-select ${selected ? "is-selected" : ""}`}
                                                  disabled={disabled}
                                                  onClick={() => {
                                                    const next = selected
                                                      ? selectedChoices.filter((v) => v !== option)
                                                      : [...selectedChoices, option];
                                                    setParamRow(activeAlgorithm, field.key, { choices: next });
                                                  }}
                                                >
                                                  {option}
                                                </button>
                                              );
                                            })}
                                          </div>
                                          <div className="training-select-grid-actions">
                                            <button
                                              type="button"
                                              className="workspace-tertiary training-select-grid-action"
                                              disabled={disabled}
                                              onClick={() => setParamRow(activeAlgorithm, field.key, { choices: (field.options || []).map((o) => String(o)) })}
                                            >
                                              {t("actions.selectAll")}
                                            </button>
                                            <button
                                              type="button"
                                              className="workspace-tertiary training-select-grid-action"
                                              disabled={disabled}
                                              onClick={() => setParamRow(activeAlgorithm, field.key, { choices: [] })}
                                            >
                                              {t("actions.clear")}
                                            </button>
                                          </div>
                                        </div>
                                      ) : (
                                        <span className="training-param-muted">-</span>
                                      )}
                                    </td>
                                    <td>
                                      {mode === "grid" && field.kind !== "select" ? (
                                        <input
                                          type="number"
                                          value={String(row?.min ?? "")}
                                          disabled={disabled}
                                          onChange={(e) => setParamRow(activeAlgorithm, field.key, { min: e.target.value })}
                                        />
                                      ) : (
                                        <span className="training-param-muted">-</span>
                                      )}
                                    </td>
                                    <td>
                                      {mode === "grid" && field.kind !== "select" ? (
                                        <input
                                          type="number"
                                          value={String(row?.max ?? "")}
                                          disabled={disabled}
                                          onChange={(e) => setParamRow(activeAlgorithm, field.key, { max: e.target.value })}
                                        />
                                      ) : (
                                        <span className="training-param-muted">-</span>
                                      )}
                                    </td>
                                    <td>
                                      {mode === "grid" && field.kind !== "select" ? (
                                        <input
                                          type="number"
                                          min="1"
                                          max="25"
                                          step="1"
                                          value={String(row?.divisions ?? "3")}
                                          disabled={disabled}
                                          onChange={(e) => setParamRow(activeAlgorithm, field.key, { divisions: e.target.value })}
                                        />
                                      ) : (
                                        <span className="training-param-muted">-</span>
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                        </>
                        )}
                        {/* FIM: Seção de algoritmos ML */}
                      </div>

                      <div className="training-inline-card">
                        <div className="training-inline-card-title">{t("training.blockConfig.searchTitle")}</div>
                        <div className="training-inline-card-subtitle">{t("training.blockConfig.searchSubtitle")}</div>
                        <div className="training-inline-form">
                          <div className="workspace-field" style={{ marginBottom: 0 }}>
                            <label>{t("training.advanced.selectionMetric")}</label>
                            <select
                              value={String(spec?.selectionMetric || trainModal.selectionMetric || "rmse")}
                              disabled={disabled}
                              onChange={(e) => updateSpec({ selectionMetric: e.target.value })}
                            >
                              <option value="rmse">RMSE</option>
                              <option value="mae">MAE</option>
                              <option value="r2">R2</option>
                            </select>
                          </div>
                          <div className="workspace-field" style={{ marginBottom: 0 }}>
                            <label>{t("training.advanced.maxTrials")}</label>
                            <input
                              type="number"
                              min="1"
                              max="500"
                              step="1"
                              value={spec?.maxTrials ?? trainModal.maxTrials ?? 60}
                              disabled={disabled}
                              onChange={(e) => updateSpec({ maxTrials: e.target.value })}
                            />
                          </div>
                        </div>
                      </div>

                      <div className="training-block-actions">
                        <button className="workspace-tertiary" type="button" disabled={disabled} onClick={() => resetAlgorithmParams(activeAlgorithm)}>
                          {t("training.presets.defaults")}
                        </button>
                        <button className="workspace-tertiary" type="button" onClick={() => setTrainBlockModal({ open: false, stepId: "" })}>
                          {t("actions.close")}
                        </button>
                      </div>
                    </>
                  );

                  /* legacy UI removida (mantida comentada apenas para refer┬¬ncia)
                  */
                  /* legacy config UI (removida)
                  return (
                    <>
                      <div className="training-block-header">
                        <div>
                          <div className="training-block-title">{t("training.configureBlock")}</div>
                          <div className="training-block-sub">{label}</div>
                        </div>
                        <button className="workspace-home-close" type="button" onClick={() => setTrainBlockModal({ open: false, stepId: "" })}>
                          {t("actions.close")}
                        </button>
                      </div>

                      <div className="training-inline-card">
                        <div className="training-inline-card-title">{t("training.blockConfig.title")}</div>
                        <div className="training-inline-card-subtitle">{t("training.blockConfig.subtitle")}</div>

                        <div className="workspace-field" style={{ marginBottom: 10 }}>
                          <label>{t("training.algorithmsLabel")}</label>
                          <div className="training-chip-group">
                            {TRAINING_ALGORITHM_ITEMS.map((item) => {
                              const selected = algorithms.includes(item.key);
                              return (
                                <button
                                  key={item.key}
                                  type="button"
                                  className={`training-chip-select ${selected ? "is-selected" : ""}`}
                                  disabled={disabled}
                                  onClick={() => toggleAlgorithm(item.key)}
                                >
                                  {item.label}
                                </button>
                              );
                            })}
                          </div>
                          <small>{t("training.blockConfig.algorithmsHint")}</small>
                        </div>

                        <div className="workspace-field" style={{ marginBottom: 10 }}>
                          <label>{t("training.blockConfig.activeAlgorithm")}</label>
                          <div className="training-chip-group">
                            {algorithms.map((k) => {
                              const key = String(k || "").trim().toLowerCase();
                              const selected = key === activeAlgorithm;
                              return (
                                <button
                                  key={key}
                                  type="button"
                                  className={`training-chip-select ${selected ? "is-selected" : ""}`}
                                  disabled={disabled}
                                  onClick={() => setActiveAlgo(key)}
                                >
                                  {algoLabel(key)}
                                </button>
                              );
                            })}
                          </div>
                        </div>

                        <div className="training-param-table-wrap" role="region" aria-label={t("training.paramsTable.aria")}>
                          <table className="training-param-table">
                            <thead>
                              <tr>
                                <th>{t("training.paramsTable.param")}</th>
                                <th>{t("training.paramsTable.mode")}</th>
                                <th>{t("training.paramsTable.value")}</th>
                                <th>{t("training.paramsTable.min")}</th>
                                <th>{t("training.paramsTable.max")}</th>
                                <th>{t("training.paramsTable.divisions")}</th>
                              </tr>
                            </thead>
                            <tbody>
                              {(schema.fields || []).map((field) => {
                                const row = rows?.[field.key] || {};
                                const mode = row?.mode === "grid" && field.grid ? "grid" : "fixed";
                                const isNull = !!row?.isNull && !!field.allowNull;
                                const effectiveValue =
                                  isNull ? "" : String(row?.value ?? (field.defaultValue === null ? "" : field.defaultValue ?? ""));

                                const valueInput = (() => {
                                  if (field.kind === "select") {
                                    return (
                                      <select
                                        value={effectiveValue}
                                        disabled={disabled || isNull}
                                        onChange={(e) => setParamRow(activeAlgorithm, field.key, { value: e.target.value, isNull: false })}
                                      >
                                        {(field.options || []).map((opt) => (
                                          <option key={opt} value={opt}>
                                            {opt}
                                          </option>
                                        ))}
                                      </select>
                                    );
                                  }

                                  if (field.kind === "text") {
                                    return (
                                      <input
                                        type="text"
                                        value={effectiveValue}
                                        placeholder={field.placeholder}
                                        disabled={disabled || isNull}
                                        onChange={(e) => setParamRow(activeAlgorithm, field.key, { value: e.target.value, isNull: false })}
                                      />
                                    );
                                  }

                                  return (
                                    <input
                                      type="number"
                                      value={effectiveValue}
                                      min={field.min}
                                      step={field.step || 1}
                                      disabled={disabled || isNull}
                                      onChange={(e) => setParamRow(activeAlgorithm, field.key, { value: e.target.value, isNull: false })}
                                    />
                                  );
                                })();

                                return (
                                  <tr key={field.key}>
                                    <td className="training-param-key">{field.label}</td>
                                    <td>
                                      {field.grid ? (
                                        <div className="training-mini-segment" role="group" aria-label={t("training.paramsTable.mode")}>
                                          <button
                                            type="button"
                                            className={`training-mini-segment-btn ${mode === "fixed" ? "is-active" : ""}`}
                                            disabled={disabled}
                                            onClick={() => setParamRow(activeAlgorithm, field.key, { mode: "fixed" })}
                                          >
                                            {t("training.paramsTable.fixed")}
                                          </button>
                                          <button
                                            type="button"
                                            className={`training-mini-segment-btn ${mode === "grid" ? "is-active" : ""}`}
                                            disabled={disabled}
                                            onClick={() => setParamRow(activeAlgorithm, field.key, { mode: "grid", isNull: false })}
                                          >
                                            {t("training.paramsTable.grid")}
                                          </button>
                                        </div>
                                      ) : (
                                        <span className="training-param-muted">—</span>
                                      )}
                                    </td>
                                    <td>
                                      {mode === "fixed" ? (
                                        <div className="training-param-value">
                                          {field.allowNull && (
                                            <label className="training-null-toggle">
                                              <input
                                                type="checkbox"
                                                checked={isNull}
                                                disabled={disabled}
                                                onChange={(e) => setParamRow(activeAlgorithm, field.key, { isNull: e.target.checked })}
                                              />
                                              <span>{t("training.paramsTable.nullLabel")}</span>
                                            </label>
                                          )}
                                          {valueInput}
                                        </div>
                                      ) : (
                                        <span className="training-param-muted">—</span>
                                      )}
                                    </td>
                                    <td>
                                      {mode === "grid" ? (
                                        <input
                                          type="number"
                                          value={String(row?.min ?? "")}
                                          disabled={disabled}
                                          onChange={(e) => setParamRow(activeAlgorithm, field.key, { min: e.target.value })}
                                        />
                                      ) : (
                                        <span className="training-param-muted">—</span>
                                      )}
                                    </td>
                                    <td>
                                      {mode === "grid" ? (
                                        <input
                                          type="number"
                                          value={String(row?.max ?? "")}
                                          disabled={disabled}
                                          onChange={(e) => setParamRow(activeAlgorithm, field.key, { max: e.target.value })}
                                        />
                                      ) : (
                                        <span className="training-param-muted">—</span>
                                      )}
                                    </td>
                                    <td>
                                      {mode === "grid" ? (
                                        <input
                                          type="number"
                                          min="1"
                                          max="25"
                                          step="1"
                                          value={String(row?.divisions ?? "3")}
                                          disabled={disabled}
                                          onChange={(e) => setParamRow(activeAlgorithm, field.key, { divisions: e.target.value })}
                                        />
                                      ) : (
                                        <span className="training-param-muted">—</span>
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>

                        <div className="training-inline-actions">
                          <button className="workspace-tertiary" type="button" disabled={disabled} onClick={() => resetAlgorithmParams(activeAlgorithm)}>
                            {t("training.blockConfig.resetDefaults")}
                          </button>
                        </div>
                      </div>

                      <div className="training-inline-card">
                        <div className="training-inline-card-title">{t("training.blockConfig.searchTitle")}</div>
                        <div className="training-inline-card-subtitle">{t("training.blockConfig.searchSubtitle")}</div>
                        <div className="training-inline-form">
                          <div className="workspace-field" style={{ marginBottom: 0 }}>
                            <label>{t("training.advanced.selectionMetric")}</label>
                            <select
                              value={spec?.selectionMetric || trainModal.selectionMetric || "rmse"}
                              disabled={disabled}
                              onChange={(e) => updateSpec({ selectionMetric: e.target.value })}
                            >
                              <option value="rmse">RMSE</option>
                              <option value="mae">MAE</option>
                              <option value="r2">R2</option>
                            </select>
                            <small>{t("training.advanced.selectionMetricHint")}</small>
                          </div>

                          <div className="workspace-field" style={{ marginBottom: 0 }}>
                            <label>{t("training.advanced.maxTrials")}</label>
                            <input
                              type="number"
                              min="1"
                              max="500"
                              step="1"
                              value={spec?.maxTrials ?? trainModal.maxTrials ?? 60}
                              disabled={disabled}
                              onChange={(e) => updateSpec({ maxTrials: e.target.value })}
                            />
                            <small>{t("training.advanced.maxTrialsHint")}</small>
                          </div>
                        </div>
                      </div>

                      <div className="training-block-actions">
                        <button className="workspace-tertiary" type="button" onClick={() => setTrainBlockModal({ open: false, stepId: "" })}>
                          {t("actions.close")}
                        </button>
                      </div>
                    </>
                  );

                  return (
                    <>
                      <div className="training-block-header">
                        <div>
                          <div className="training-block-title">{t("training.configureBlock")}</div>
                          <div className="training-block-sub">{label}</div>
                        </div>
                        <button className="workspace-home-close" type="button" onClick={() => setTrainBlockModal({ open: false, stepId: "" })}>
                          {t("actions.close")}
                        </button>
                      </div>

                      <div className="training-inline-card">
                        <div className="training-inline-card-title">{t("training.presets.title")}</div>
                        <div className="training-inline-card-subtitle">{t("training.presets.helper")}</div>

                        <div className="training-inline-form">
                          {(schema.fields || []).map((f) => {
                            const current = paramsObj?.[f.key];
                            if (f.type === "select") {
                              return (
                                <div className="workspace-field" key={f.key} style={{ marginBottom: 0 }}>
                                  <label>{f.label}</label>
                                  <select
                                    value={String(current ?? (schema.defaults?.[f.key] ?? ""))}
                                    disabled={trainModal.running || !enabled}
                                    onChange={(e) => setParamValue(f.key, e.target.value)}
                                  >
                                    {(f.options || []).map((opt) => (
                                      <option key={opt} value={opt}>
                                        {opt}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                              );
                            }

                            if (f.type === "number") {
                              const value = current == null ? "" : Number(current);
                              return (
                                <div className="workspace-field" key={f.key} style={{ marginBottom: 0 }}>
                                  <label>{f.label}</label>
                                  <div className="training-inline-number">
                                    <input
                                      type="number"
                                      value={value}
                                      min={f.min}
                                      step={f.step || 1}
                                      disabled={trainModal.running || !enabled}
                                      onChange={(e) => {
                                        const raw = e.target.value;
                                        if (raw === "" && f.allowNull) setParamValue(f.key, "__null__");
                                        else setParamValue(f.key, raw === "" ? "__delete__" : Number(raw));
                                      }}
                                    />
                                    {f.allowNull && (
                                      <button
                                        type="button"
                                        className="training-inline-chip"
                                        disabled={trainModal.running || !enabled}
                                        onClick={() => setParamValue(f.key, "__null__")}
                                      >
                                        null
                                      </button>
                                    )}
                                  </div>
                                </div>
                              );
                            }

                            return (
                              <div className="workspace-field" key={f.key} style={{ marginBottom: 0 }}>
                                <label>{f.label}</label>
                                <input
                                  type="text"
                                  value={String(current ?? schema.defaults?.[f.key] ?? "")}
                                  placeholder={f.placeholder}
                                  disabled={trainModal.running || !enabled}
                                  onChange={(e) => setParamValue(f.key, e.target.value)}
                                />
                              </div>
                            );
                          })}
                        </div>

                        <div className="training-inline-actions">
                          <button className="workspace-tertiary" type="button" disabled={trainModal.running || !enabled} onClick={applyDefaults}>
                            {t("training.presets.defaults")}
                          </button>
                          <button className="workspace-tertiary" type="button" disabled={trainModal.running || !enabled} onClick={clearParams}>
                            {t("training.presets.clear")}
                          </button>
                        </div>
                      </div>

                      <div className="workspace-field">
                        <label className={`training-switch ${spec?.gridSearch ? "is-on" : ""}`}>
                          <input
                            type="checkbox"
                            checked={!!spec?.gridSearch}
                            disabled={trainModal.running || !enabled}
                            onChange={(e) =>
                              setTrainModelsDraft((prev) => ({
                                ...prev,
                                [stepId]: { ...(prev?.[stepId] || {}), gridSearch: e.target.checked },
                              }))
                            }
                          />
                          <span className="training-switch-track" aria-hidden="true" />
                          <span className="training-switch-text">{t("training.grid.enable")}</span>
                        </label>
                      </div>

                      <div className="workspace-field">
                        <label>{t("training.algorithmLabel")}</label>
                        <div className="training-chip-group">
                          {algorithmItems.map((item) => (
                            <button
                              key={item.key}
                              type="button"
                              className={`training-chip-select ${algo === item.key ? "is-selected" : ""}`}
                              disabled={trainModal.running || !enabled}
                              onClick={() =>
                                setTrainModelsDraft((prev) => ({
                                  ...prev,
                                  [stepId]: {
                                    ...(prev?.[stepId] || {}),
                                    algorithm: item.key,
                                    algorithms: Array.isArray(prev?.[stepId]?.algorithms) && prev[stepId].algorithms.length ? prev[stepId].algorithms : [item.key],
                                  },
                                }))
                              }
                            >
                              {item.label}
                            </button>
                          ))}
                        </div>
                      </div>

                      {spec?.gridSearch && (
                        <>
                          <div className="training-inline-card">
                            <div className="training-inline-card-title">{t("training.presets.gridTitle")}</div>
                            <div className="training-inline-card-subtitle">{t("training.presets.gridHelper")}</div>
                            <div className="training-inline-actions">
                              <button className="workspace-tertiary" type="button" disabled={trainModal.running || !enabled} onClick={applyGridDefaults}>
                                {t("training.presets.applyGridDefaults")}
                              </button>
                            </div>
                          </div>

                          <div className="workspace-field">
                            <label>{t("training.grid.algorithms")}</label>
                            <div className="training-chip-group">
                              {algorithmItems.map((item) => {
                                const list = Array.isArray(spec?.algorithms) ? spec.algorithms : [];
                                const selected = list.includes(item.key);
                                return (
                                  <button
                                    key={item.key}
                                    type="button"
                                    className={`training-chip-select ${selected ? "is-selected" : ""}`}
                                    disabled={trainModal.running || !enabled}
                                    onClick={() => {
                                      const next = selected ? list.filter((x) => x !== item.key) : [...list, item.key];
                                      setTrainModelsDraft((prev) => ({
                                        ...prev,
                                        [stepId]: { ...(prev?.[stepId] || {}), algorithms: next.length ? next : [algo] },
                                      }));
                                    }}
                                  >
                                    {item.label}
                                  </button>
                                );
                              })}
                            </div>
                          </div>

                          <div className="training-advanced-grid">
                            <div className="workspace-field" style={{ marginBottom: 0 }}>
                              <label>{t("training.advanced.selectionMetric")}</label>
                              <select
                                value={spec?.selectionMetric || trainModal.selectionMetric || "rmse"}
                                disabled={trainModal.running || !enabled}
                                onChange={(e) =>
                                  setTrainModelsDraft((prev) => ({
                                    ...prev,
                                    [stepId]: { ...(prev?.[stepId] || {}), selectionMetric: e.target.value },
                                  }))
                                }
                              >
                                <option value="rmse">RMSE</option>
                                <option value="mae">MAE</option>
                                <option value="r2">R2</option>
                              </select>
                            </div>

                            <div className="workspace-field" style={{ marginBottom: 0 }}>
                              <label>{t("training.advanced.maxTrials")}</label>
                              <input
                                type="number"
                                min="1"
                                max="500"
                                step="1"
                                value={spec?.maxTrials ?? trainModal.maxTrials ?? 60}
                                disabled={trainModal.running || !enabled}
                                onChange={(e) =>
                                  setTrainModelsDraft((prev) => ({
                                    ...prev,
                                    [stepId]: { ...(prev?.[stepId] || {}), maxTrials: e.target.value },
                                  }))
                                }
                              />
                            </div>
                          </div>

                          <div className="workspace-field">
                            <label>{t("training.grid.gridLabel")}</label>
                            <textarea
                              className="training-params"
                              value={spec?.gridText || ""}
                              disabled={trainModal.running || !enabled}
                              placeholder={t("training.grid.gridHint")}
                              onChange={(e) =>
                                setTrainModelsDraft((prev) => ({
                                  ...prev,
                                  [stepId]: { ...(prev?.[stepId] || {}), gridText: e.target.value },
                                }))
                              }
                              rows={5}
                            />
                          </div>
                        </>
                      )}

                      <div className="workspace-field">
                        <label className={`training-switch ${trainBlockModal.advanced ? "is-on" : ""}`}>
                          <input
                            type="checkbox"
                            checked={!!trainBlockModal.advanced}
                            disabled={trainModal.running || !enabled}
                            onChange={(e) => setTrainBlockModal((prev) => ({ ...prev, advanced: e.target.checked }))}
                          />
                          <span className="training-switch-track" aria-hidden="true" />
                          <span className="training-switch-text">
                            {trainBlockModal.advanced ? t("training.hideAdvanced") : t("training.showAdvanced")}
                          </span>
                        </label>
                      </div>

                      {trainBlockModal.advanced && (
                        <div className="workspace-field">
                          <label>{t("training.paramsLabel")}</label>
                          <textarea
                            className="training-params"
                            value={spec?.paramsText || ""}
                            disabled={trainModal.running || !enabled}
                            placeholder={t("training.paramsPlaceholder")}
                            onChange={(e) =>
                              setTrainModelsDraft((prev) => ({
                                ...prev,
                                [stepId]: { ...(prev?.[stepId] || {}), paramsText: e.target.value },
                              }))
                            }
                            rows={4}
                          />
                        </div>
                      )}

                      <div className="training-block-actions">
                        <button className="workspace-tertiary" type="button" onClick={() => setTrainBlockModal({ open: false, stepId: "" })}>
                          {t("actions.close")}
                        </button>
                      </div>
                    </>
                  );
                  */

                  })()}
                </div>
              </div>,
              document.body
            )}
        </div>

        <div className="training-footer">
          <button className="workspace-tertiary" type="button" disabled={trainModal.running || step === 0} onClick={goBack}>
            {t("training.back")}
          </button>
          <button
            className="workspace-secondary"
            type="button"
            disabled={trainModal.running}
            onClick={() => setTrainModal((prev) => ({ ...prev, experimentIdsText: "", result: null, error: "" }))}
          >
            {t("actions.clear")}
          </button>
          {step < 2 ? (
            <button className="workspace-primary" type="button" disabled={trainModal.running} onClick={goNext}>
              {t("training.next")}
            </button>
          ) : (
            <button className="workspace-primary" type="button" disabled={trainModal.running} onClick={runTraining}>
              {trainModal.running ? t("training.running") : t("training.run")}
            </button>
          )}
        </div>
      </div>
    );
  
}
