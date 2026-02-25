export default function NodeConfigFormContent({
  selectedNode,
  t,
  renderConfigField,
  useDefaultExperiment,
  setUseDefaultExperiment,
  setNodes,
  edges,
  updateNodeConfigField,
  applyConfigToSelectedNode,
  setFeaturesMergeInputs,
  setSequentialInputsCount,
  setValueInListDraft,
  valueInListDraft,
  effectiveConfigInputs,
  availableChannels,
  availableFitChannels,
  simulation,
  trainModal,
  setTrainModal,
  trainModelsDraft,
  setTrainModelsDraft,
  nodes,
  setDatasetSelectorOpen,
  setTrainBlockModal,
  setCandidatesModal,
  trainBlockModal,
  runTraining,
}) {
  return (
                <div className="config-form">
                  {selectedNode && selectedNode.data && selectedNode.data.blockName === "experiment_fetch" ? (
                    <>
                                            {/* Grupo 1: Fonte de Dados */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.dataSource")}</legend>
                        {["experimentId", "tenant"].map((f) => renderConfigField(f))}
                        
                        {/* Botão para experimento padrão */}
                        <div className="config-field">
                          <label className="config-label">
                            <input
                              type="checkbox"
                              checked={useDefaultExperiment}
                              onChange={(e) => {
                                const newValue = e.target.checked;
                                setUseDefaultExperiment(newValue);
                                
                                // Atualizar todos os campos de uma vez para evitar problemas de estado
                                const currentConfig = selectedNode.data.config || {};
                                let nextConfig;
                                
                                if (newValue) {
                                  // Se ativar experimento padrão, preencher os campos obrigatórios
                                  nextConfig = {
                                    ...currentConfig,
                                    use_default_experiment: true,
                                    experimentId: "019b221a-bfa8-705a-9b40-8b30f144ef68",
                                    analysisId: "68cb3fb380ac865ce0647ea8",
                                    tenant: "corsan",
                                    generate_output_graphs: true,
                                    include_experiment_output: true,
                                    include_experiment_data_output: true,
                                    graph_config: {
                                      turbidimetry: ["f1","f2","f3","f4","f5","f6","f7","f8","clear","nir"],
                                      nephelometry: ["f1","f2","f3","f4","f5","f6","f7","f8","clear","nir"],
                                      fluorescence: ["f1","f2","f3","f4","f5","f6","f7","f8","clear","nir"],
                                    },
                                  };
                                } else {
                                  // Se desativar, limpar os campos
                                  nextConfig = {
                                    ...currentConfig,
                                    use_default_experiment: false,
                                    experimentId: "",
                                    analysisId: "",
                                    tenant: "",
                                    generate_output_graphs: false
                                  };
                                }
                                
                                applyConfigToSelectedNode(nextConfig);
                              }}
                              className="config-checkbox"
                            />
                            {t("configuration.useDemoExperiment")}
                          </label>
                          <div className="config-description">
                            Carrega dados de exemplo para testar o pipeline
                          </div>
                        </div>
                      </fieldset>

                      {/* Grupo 2: Debug */}
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_experiment_output)}
                              onChange={(e) => updateNodeConfigField("include_experiment_output", e.target.checked)}
                            />
                            <span>Mostrar metadados do experimento (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_experiment_data_output)}
                              onChange={(e) => updateNodeConfigField("include_experiment_data_output", e.target.checked)}
                            />
                            <span>Mostrar dados brutos (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.generate_output_graphs)}
                              onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)}
                            />
                            <span>Gerar gráficos de visualização</span>
                          </label>
                        </div>
                        {selectedNode.data.config?.generate_output_graphs && (
                          <div className="graph-options">
                            {renderConfigField("graph_config")}
                          </div>
                        )}
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName?.includes("_extraction") ? (
                    <>
                      {/* Blocos de extração de sensor - Layout com cards */}
                      
                      {/* Card 0: Modo de saída (apenas para sensores espectrais) */}
                      {['turbidimetry_extraction', 'nephelometry_extraction', 'fluorescence_extraction'].includes(selectedNode?.data?.blockName) && (
                        <fieldset className="config-group">
                          <legend>{t("configuration.outputMode")}</legend>
                          <div className="config-field">
                            <label className="config-label">Formato dos valores</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.output_mode || "raw"}
                              onChange={(e) => updateNodeConfigField("output_mode", e.target.value)}
                            >
                              <option value="raw">RAW (valores brutos)</option>
                              <option value="basic_counts">Basic Counts (RAW / gain × timeMs)</option>
                            </select>
                          </div>
                        </fieldset>
                      )}

                      {/* Card 1: Limpeza de Dados */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.dataCleaning")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={selectedNode.data.config?.remove_duplicates !== false}
                              onChange={(e) => updateNodeConfigField("remove_duplicates", e.target.checked)}
                            />
                            <span>Remover timestamps duplicados</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={selectedNode.data.config?.validate_data !== false}
                              onChange={(e) => updateNodeConfigField("validate_data", e.target.checked)}
                            />
                            <span>Validar dados extraídos</span>
                          </label>
                        </div>
                      </fieldset>

                      {/* Card 2: Debug/Saídas */}
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Incluir dados brutos (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.generate_output_graphs)}
                              onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)}
                            />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "time_slice" ? (
                    <>
                      {/* Bloco Time Slice - Layout com cards */}
                      
                      {/* Card 1: Configuração de Corte */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.timeCut")}</legend>
                        <div className="config-field">
                          <label className="config-label">Modo de corte</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.slice_mode || "time"}
                            onChange={(e) => updateNodeConfigField("slice_mode", e.target.value)}
                          >
                            <option value="time">Por tempo (minutos)</option>
                            <option value="index">Por índice</option>
                          </select>
                        </div>
                        
                        {(selectedNode.data.config?.slice_mode || "time") === "time" ? (
                          <>
                            <div className="config-field">
                              <label className="config-label">Tempo inicial (min)</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.start_time_min ?? 0}
                                onChange={(e) => updateNodeConfigField("start_time_min", parseFloat(e.target.value) || 0)}
                                step="0.5"
                                min="0"
                              />
                            </div>
                            <div className="config-field">
                              <label className="config-label">Tempo final (min)</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.end_time_min ?? ""}
                                onChange={(e) => updateNodeConfigField("end_time_min", e.target.value ? parseFloat(e.target.value) : null)}
                                placeholder="Até o fim"
                                step="0.5"
                                min="0"
                              />
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="config-field">
                              <label className="config-label">Índice inicial</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.start_index ?? 0}
                                onChange={(e) => updateNodeConfigField("start_index", parseInt(e.target.value) || 0)}
                                min="0"
                              />
                            </div>
                            <div className="config-field">
                              <label className="config-label">Índice final</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.end_index ?? ""}
                                onChange={(e) => updateNodeConfigField("end_index", e.target.value ? parseInt(e.target.value) : null)}
                                placeholder="Até o fim"
                                min="0"
                              />
                            </div>
                          </>
                        )}
                        
                      </fieldset>

                      {/* Card 2: Debug */}
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.generate_output_graphs)}
                              onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)}
                            />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "sensor_fusion" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.sensorFusion")}</legend>
                        <p className="config-hint">{t("configuration.sensorFusionHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.sensorFusionInputs")}</label>
                          {(() => {
                            const maxInputs = 6;
                            const currentKeys = (selectedNode?.data?.dataInputs || []).filter((k) => String(k).startsWith("sensor_data_"));
                            const configuredCount = Number(selectedNode?.data?.config?.inputs_count);
                            const count = Math.max(
                              1,
                              Math.min(maxInputs, Number.isFinite(configuredCount) ? configuredCount : (currentKeys.length || 2))
                            );
                            const keys = Array.from({ length: count }, (_, i) => `sensor_data_${i + 1}`);

                            const sources = (() => {
                              const val = selectedNode?.data?.config?.sources;
                              if (Array.isArray(val)) return val;
                              if (typeof val === "string" && val.trim().length) {
                                try {
                                  const parsed = JSON.parse(val);
                                  return Array.isArray(parsed) ? parsed : [];
                                } catch {
                                  return [];
                                }
                              }
                              return [];
                            })();

                            const updateSource = (inputKey, patch) => {
                              const next = Array.isArray(sources) ? [...sources] : [];
                              const idx = next.findIndex((s) => s && s.input === inputKey);
                              const base = idx >= 0 && next[idx] ? next[idx] : { input: inputKey };
                              const merged = { ...base, ...patch, input: inputKey };
                              if (idx >= 0) next[idx] = merged;
                              else next.push(merged);
                              updateNodeConfigField("sources", next);
                            };

                            return (
                              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                                  <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                                    <button
                                      type="button"
                                      className="btn btn-secondary"
                                      onClick={() =>
                                        setSequentialInputsCount(selectedNode.id, count - 1, {
                                          prefix: "sensor_data_",
                                          max: 6,
                                          configKey: "inputs_count",
                                        })
                                      }
                                      disabled={count <= 1}
                                    >
                                      -
                                    </button>
                                    <span style={{ fontSize: 12 }}>{t("configuration.inputsCount", { count })}</span>
                                    <button
                                      type="button"
                                      className="btn btn-secondary"
                                      onClick={() =>
                                        setSequentialInputsCount(selectedNode.id, count + 1, {
                                          prefix: "sensor_data_",
                                          max: 6,
                                          configKey: "inputs_count",
                                        })
                                      }
                                      disabled={count >= maxInputs}
                                    >
                                      +
                                    </button>
                                  </span>
                                </div>

                                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                  {keys.map((k) => {
                                    const s = sources.find((x) => x && x.input === k) || { input: k };
                                    const prefix = typeof s.prefix === "string" ? s.prefix : "";
                                    const channels = Array.isArray(s.channels) ? s.channels.join(", ") : s.channels || "";

                                    const edge = edges.find((e) => {
                                      if (e.target !== selectedNode.id) return false;
                                      const th = e.targetHandle || "";
                                      const inputName = th.includes("-in-") ? th.split("-in-")[1] : th;
                                      return inputName === k;
                                    });
                                    const sourceId = edge?.source;
                                    const sourceNode = sourceId ? nodes.find((n) => n.id === sourceId) : null;
                                    const stepData = sourceId ? simulation?.step_results?.[sourceId] : null;
                                    const stepSensorData = stepData?.data?.sensor_data || stepData?.sensor_data || null;
                                    const availableChannels = (() => {
                                      const fromData = stepSensorData?.available_channels;
                                      if (Array.isArray(fromData) && fromData.length) return fromData.map(String);
                                      const ch = stepSensorData?.channels;
                                      if (ch && typeof ch === "object" && !Array.isArray(ch)) return Object.keys(ch);

                                      const blockName = String(sourceNode?.data?.blockName || "");
                                      const spectralDefaults = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clear", "nir"];
                                      const conversionDefaults = {
                                        rgb_conversion: ["RGB_R", "RGB_G", "RGB_B"],
                                        hsv_conversion: ["HSV_H", "HSV_S", "HSV_V"],
                                        hsl_conversion: ["HSL_H", "HSL_S", "HSL_L"],
                                        hsb_conversion: ["HSB_H", "HSB_S", "HSB_B"],
                                        lab_conversion: ["LAB_L", "LAB_A", "LAB_B"],
                                        luv_conversion: ["LUV_L", "LUV_U", "LUV_V"],
                                        xyz_conversion: ["XYZ_X", "XYZ_Y", "XYZ_Z"],
                                        cmyk_conversion: ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"],
                                        xyy_conversion: ["xyY_x", "xyY_y", "xyY_Y"],
                                      };
                                      if (blockName in conversionDefaults) return conversionDefaults[blockName];
                                      if (blockName.endsWith("_extraction") && !blockName.includes("temperatures") && !blockName.includes("power_supply") && !blockName.includes("peltier") && !blockName.includes("nema") && !blockName.includes("control_state")) {
                                        return spectralDefaults;
                                      }
                                      return [];
                                    })();

                                    const suggestedPrefix = (() => {
                                      const sk = stepSensorData?.sensor_key;
                                      if (typeof sk === "string" && sk.trim()) return sk.trim().slice(0, 16);
                                      const sn = stepSensorData?.sensor_name;
                                      if (typeof sn === "string" && sn.trim()) return sn.trim().split("_")[0].slice(0, 16);
                                      return "";
                                    })();

                                    const selectedList = (() => {
                                      if (Array.isArray(s.channels)) return s.channels.map(String).filter(Boolean);
                                      if (typeof s.channels === "string") {
                                        return s.channels
                                          .split(",")
                                          .map((v) => v.trim())
                                          .filter(Boolean);
                                      }
                                      return [];
                                    })();

                                    const isAll = availableChannels.length > 0 && (selectedList.length === 0 || selectedList.length === availableChannels.length);

                                    return (
                                      <div
                                        key={k}
                                        style={{
                                          border: "1px solid var(--color-gray-200)",
                                          borderRadius: 12,
                                          padding: 10,
                                          background: "rgba(255,255,255,0.6)",
                                        }}
                                      >
                                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8, marginBottom: 8 }}>
                                          <div style={{ fontWeight: 600, fontSize: 12 }}>{k}</div>
                                          <div style={{ fontSize: 11, color: "var(--color-gray-500)", textAlign: "right" }}>
                                            {sourceNode?.data?.label ? (
                                              <span title={sourceNode?.data?.blockName || ""}>{sourceNode.data.label}</span>
                                            ) : (
                                              <span>{t("configuration.noConnection")}</span>
                                            )}
                                            {stepSensorData?.color_space ? <span>{` • ${stepSensorData.color_space}`}</span> : null}
                                          </div>
                                        </div>
                                        <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 8 }}>
                                          <div className="config-field" style={{ margin: 0 }}>
                                            <label className="config-label">{t("configuration.prefix")}</label>
                                            <input
                                              type="text"
                                              className="config-input"
                                              value={prefix}
                                              onChange={(e) => updateSource(k, { prefix: e.target.value })}
                                              placeholder={t("configuration.prefixPlaceholder")}
                                            />
                                          </div>
                                          <div className="config-field" style={{ margin: 0 }}>
                                            <label className="config-label">{t("configuration.channels")}</label>
                                            <input
                                              type="text"
                                              className="config-input"
                                              value={channels}
                                              onChange={(e) => {
                                                const parsed = e.target.value
                                                  .split(",")
                                                  .map((v) => v.trim())
                                                  .filter(Boolean);
                                                updateSource(k, { channels: parsed.length ? parsed : e.target.value });
                                              }}
                                              placeholder={t("configuration.sensorFusionChannelsPlaceholder")}
                                            />
                                            <small className="config-hint-small">{t("configuration.sensorFusionChannelsHint")}</small>
                                          </div>
                                        </div>

                                        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginTop: 8 }}>
                                          <button
                                            type="button"
                                            className="btn btn-secondary"
                                            onClick={() => {
                                              if (suggestedPrefix && !prefix) updateSource(k, { prefix: suggestedPrefix });
                                              updateSource(k, { channels: [] });
                                            }}
                                            disabled={!availableChannels.length && !suggestedPrefix}
                                            title={t("configuration.useSuggested")}
                                          >
                                            {t("configuration.useSuggested")}
                                          </button>
                                          {availableChannels.length > 0 && (
                                            <span style={{ fontSize: 11, color: "var(--color-gray-500)" }}>
                                              {t("configuration.availableChannels", { count: availableChannels.length })}
                                            </span>
                                          )}
                                        </div>

                                        {availableChannels.length > 0 && (
                                          <div className="chip-group" style={{ marginTop: 8 }}>
                                            <button
                                              type="button"
                                              className={`chip chip-all ${isAll ? "active" : ""}`}
                                              onClick={() => updateSource(k, { channels: [] })}
                                              title={t("configuration.allChannels")}
                                            >
                                              {t("common.all")}
                                            </button>
                                            {availableChannels.map((chName) => {
                                              const isActive = isAll ? false : selectedList.includes(chName);
                                              return (
                                                <button
                                                  key={chName}
                                                  type="button"
                                                  className={`chip ${isActive ? "active" : ""}`}
                                                  onClick={() => {
                                                    if (isAll) {
                                                      updateSource(k, { channels: [chName] });
                                                      return;
                                                    }
                                                    const nextSet = new Set(selectedList);
                                                    if (nextSet.has(chName)) nextSet.delete(chName);
                                                    else nextSet.add(chName);
                                                    updateSource(k, { channels: Array.from(nextSet) });
                                                  }}
                                                >
                                                  {chName}
                                                </button>
                                              );
                                            })}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            );
                          })()}
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.mergeMode")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.merge_mode || "intersection"}
                              onChange={(e) => updateNodeConfigField("merge_mode", e.target.value)}
                            >
                              <option value="intersection">{t("configuration.mergeModes.intersection")}</option>
                              <option value="union">{t("configuration.mergeModes.union")}</option>
                            </select>
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.resampleStep")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.resample_step ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("resample_step", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder={t("configuration.resampleStepPlaceholder")}
                              step="0.1"
                            />
                            <small className="config-hint-small">{t("configuration.resampleStepHint")}</small>
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "outlier_removal" ? (
                    <>
                      {/* Bloco Outlier Removal - Layout com cards */}
                      
                      {/* Card 1: Método de Detecção */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.outlierDetection")}</legend>
                        <div className="config-field">
                          <label className="config-label">Método</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.method || "zscore"}
                            onChange={(e) => updateNodeConfigField("method", e.target.value)}
                          >
                            <option value="zscore">Z-Score (desvio padrão)</option>
                            <option value="iqr">IQR (intervalo interquartil)</option>
                            <option value="mad">MAD (desvio mediano absoluto)</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-label">
                            Limiar ({selectedNode.data.config?.method === "iqr" ? "multiplicador IQR" : "desvios"})
                          </label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.threshold ?? (selectedNode.data.config?.method === "iqr" ? 1.5 : 3.0)}
                            onChange={(e) => updateNodeConfigField("threshold", parseFloat(e.target.value) || 3.0)}
                            step="0.1"
                            min="0.1"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Estratégia</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.replace_strategy || "remove"}
                            onChange={(e) => updateNodeConfigField("replace_strategy", e.target.value)}
                          >
                            <option value="remove">Remover linhas com outliers</option>
                            <option value="interpolate">Interpolar valores</option>
                          </select>
                        </div>
                      </fieldset>

                      {/* Card 2: Debug */}
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.generate_output_graphs)}
                              onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)}
                            />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "moving_average_filter" ? (
                    <>
                      {/* Bloco Moving Average Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.movingAverage")}</legend>
                        <div className="config-field">
                          <label className="config-label">Tamanho da janela</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.window_size ?? 5}
                            onChange={(e) => updateNodeConfigField("window_size", parseInt(e.target.value) || 5)}
                            min="1"
                            max="50"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Alinhamento</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.alignment || "center"}
                            onChange={(e) => updateNodeConfigField("alignment", e.target.value)}
                          >
                            <option value="center">Centralizado</option>
                            <option value="left">Esquerda (causal)</option>
                            <option value="right">Direita</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "savgol_filter" ? (
                    <>
                      {/* Bloco Savitzky-Golay Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.savgol")}</legend>
                        <div className="config-field">
                          <label className="config-label">Tamanho da janela (ímpar)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.window_size ?? 11}
                            onChange={(e) => updateNodeConfigField("window_size", parseInt(e.target.value) || 11)}
                            min="3"
                            max="51"
                            step="2"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Ordem do polinômio</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.poly_order ?? 3}
                            onChange={(e) => updateNodeConfigField("poly_order", parseInt(e.target.value) || 3)}
                            min="1"
                            max="10"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "median_filter" ? (
                    <>
                      {/* Bloco Median Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.medianFilter")}</legend>
                        <div className="config-field">
                          <label className="config-label">Tamanho do kernel (ímpar)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.kernel_size ?? 5}
                            onChange={(e) => updateNodeConfigField("kernel_size", parseInt(e.target.value) || 5)}
                            min="3"
                            max="31"
                            step="2"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "lowpass_filter" ? (
                    <>
                      {/* Bloco Lowpass Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.lowpass")}</legend>
                        <div className="config-field">
                          <label className="config-label">Frequência de corte (0-1)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.cutoff_freq ?? 0.1}
                            onChange={(e) => updateNodeConfigField("cutoff_freq", parseFloat(e.target.value) || 0.1)}
                            min="0.01"
                            max="0.99"
                            step="0.01"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Ordem do filtro</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.order ?? 4}
                            onChange={(e) => updateNodeConfigField("order", parseInt(e.target.value) || 4)}
                            min="1"
                            max="10"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "exponential_filter" ? (
                    <>
                      {/* Bloco Exponential Filter */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.ema")}</legend>
                        <div className="config-field">
                          <label className="config-label">Alpha (0-1, menor = mais suave)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.alpha ?? 0.3}
                            onChange={(e) => updateNodeConfigField("alpha", parseFloat(e.target.value) || 0.3)}
                            min="0.01"
                            max="1.0"
                            step="0.05"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "derivative" ? (
                    <>
                      {/* Bloco Derivative */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.derivative")}</legend>
                        <div className="config-field">
                          <label className="config-label">Método</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.method || "gradient"}
                            onChange={(e) => updateNodeConfigField("method", e.target.value)}
                          >
                            <option value="gradient">Gradiente (numpy)</option>
                            <option value="diff">Diferença simples</option>
                            <option value="savgol">Savgol (suavizado)</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Ordem da derivada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.order || 1}
                            onChange={(e) => updateNodeConfigField("order", parseInt(e.target.value))}
                          >
                            <option value={1}>1ª derivada (velocidade)</option>
                            <option value={2}>2ª derivada (aceleração)</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "integral" ? (
                    <>
                      {/* Bloco Integral */}
                      <fieldset className="config-group">
                        <legend>∫ Integral</legend>
                        <div className="config-field">
                          <label className="config-label">Método</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.method || "trapz"}
                            onChange={(e) => updateNodeConfigField("method", e.target.value)}
                          >
                            <option value="trapz">Trapezoidal (recomendado)</option>
                            <option value="cumsum">Soma cumulativa</option>
                            <option value="simpson">Simpson</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "normalize" ? (
                    <>
                      {/* Bloco Normalize */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.normalization")}</legend>
                        <div className="config-field">
                          <label className="config-label">Método</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.method || "minmax"}
                            onChange={(e) => updateNodeConfigField("method", e.target.value)}
                          >
                            <option value="minmax">Min-Max (0 a 1)</option>
                            <option value="zscore">Z-Score (média 0, std 1)</option>
                            <option value="robust">Robust (baseado em mediana)</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "curve_fit" ? (
                    <>
                      {/* Bloco Curve Fit */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.curveFit")}</legend>
                        <p className="config-hint">Ajusta modelo matemático aos dados de crescimento</p>
                        <div className="config-field">
                          <label className="config-label">Modelo matemático</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.model || "richards"}
                            onChange={(e) => updateNodeConfigField("model", e.target.value)}
                          >
                            <option value="richards">Richards (flexível)</option>
                            <option value="gompertz">Gompertz (clássico)</option>
                            <option value="logistic">Logístico</option>
                            <option value="baranyi">Baranyi (com lag)</option>
                            <option value="auto">Automático (melhor fit)</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Todos os canais</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="ex: f1, clear, nir"
                            />
                          )}
                          <small className="config-hint-small">
                            {availableChannels.length > 0 
                              ? "Selecione um canal específico ou todos" 
                              : "Conecte um bloco para ver os canais disponíveis"}
                          </small>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Tentativas de ajuste</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.max_attempts ?? 15}
                            onChange={(e) => updateNodeConfigField("max_attempts", parseInt(e.target.value) || 15)}
                            min="1"
                            max="50"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Tolerância de erro</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.tolerance ?? 0.001}
                            onChange={(e) => updateNodeConfigField("tolerance", parseFloat(e.target.value) || 0.001)}
                            min="0.0001"
                            max="0.1"
                            step="0.0001"
                          />
                        </div>
                      </fieldset>
                      <fieldset className="config-group">
                        <legend>Reamostragem (ML)</legend>
                        <p className="config-hint">Padroniza saída para uso em modelos de Machine Learning</p>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input 
                              type="checkbox" 
                              checked={Boolean(selectedNode.data.config?.resample_output)} 
                              onChange={(e) => updateNodeConfigField("resample_output", e.target.checked)} 
                            />
                            <span>Reamostrar em grid regular</span>
                          </label>
                        </div>
                        {selectedNode.data.config?.resample_output && (
                          <>
                            <div className="config-field">
                              <label className="config-label">Número de pontos</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.resample_points ?? 100}
                                onChange={(e) => updateNodeConfigField("resample_points", parseInt(e.target.value) || 100)}
                                min="10"
                                max="1000"
                              />
                              <small className="config-hint-small">Quantidade de pontos na curva de saída</small>
                            </div>
                            <div className="config-field">
                              <label className="config-label">X mínimo (min)</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.resample_xmin ?? 0}
                                onChange={(e) => updateNodeConfigField("resample_xmin", parseFloat(e.target.value) || 0)}
                                min="0"
                                step="1"
                              />
                            </div>
                            <div className="config-field">
                              <label className="config-label">X máximo (min)</label>
                              <input
                                type="number"
                                className="config-input"
                                value={selectedNode.data.config?.resample_xmax ?? 0}
                                onChange={(e) => updateNodeConfigField("resample_xmax", parseFloat(e.target.value) || 0)}
                                min="0"
                                step="1"
                                placeholder="0 = automático"
                              />
                              <small className="config-hint-small">0 = usa máximo do experimento. Ex: 1440 = 24h</small>
                            </div>
                          </>
                        )}
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "curve_fit_best" ? (
                    <>
                      {/* Bloco Curve Fit Best */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.bestFit")}</legend>
                        <p className="config-hint">Testa múltiplos modelos e retorna o melhor</p>
                        <div className="config-field">
                          <label className="config-label">Modelos a testar</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.models || "all"}
                            onChange={(e) => updateNodeConfigField("models", e.target.value)}
                            placeholder="all ou: richards, gompertz, logistic"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Primeiro canal disponível</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="ex: f1 (vazio = primeiro)"
                            />
                          )}
                          <small className="config-hint-small">
                            {availableChannels.length > 0 
                              ? "Selecione o canal para comparar modelos" 
                              : "Conecte um bloco para ver os canais disponíveis"}
                          </small>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "statistical_features" ? (
                    <>
                      {/* Bloco Statistical Features */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.statisticalFeatures")}</legend>
                        <p className="config-hint">Extrai estatísticas básicas dos dados</p>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Todos os canais</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="vazio = todos"
                            />
                          )}
                        </div>
                        <div className="config-field">
                          <label className="config-label">Features a extrair</label>
                          <div className="chip-group">
                            {["max", "min", "mean", "std", "range", "variance", "median", "sum"].map((feat) => {
                              const currentFeatures = selectedNode.data.config?.features || ["max", "min", "mean", "std", "range"];
                              const isSelected = currentFeatures.includes(feat);
                              return (
                                <button
                                  key={feat}
                                  type="button"
                                  className={`chip ${isSelected ? "active" : ""}`}
                                  onClick={() => {
                                    const newFeatures = isSelected
                                      ? currentFeatures.filter(f => f !== feat)
                                      : [...currentFeatures, feat];
                                    updateNodeConfigField("features", newFeatures);
                                  }}
                                >
                                  {feat}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "temporal_features" ? (
                    <>
                      {/* Bloco Temporal Features */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.temporalFeatures")}</legend>
                        <p className="config-hint">Extrai características relacionadas ao tempo</p>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Todos os canais</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="vazio = todos"
                            />
                          )}
                        </div>
                        <div className="config-field">
                          <label className="config-label">Threshold (%)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.threshold_percent ?? 50}
                            onChange={(e) => updateNodeConfigField("threshold_percent", parseFloat(e.target.value) || 50)}
                            min="0"
                            max="100"
                          />
                          <small className="config-hint-small">Percentual do máximo para time_to_threshold</small>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Features a extrair</label>
                          <div className="chip-group">
                            {["time_to_max", "time_to_min", "time_to_threshold", "time_at_max", "time_at_min", "duration"].map((feat) => {
                              const currentFeatures = selectedNode.data.config?.features || ["time_to_max", "time_to_min", "time_to_threshold"];
                              const isSelected = currentFeatures.includes(feat);
                              return (
                                <button
                                  key={feat}
                                  type="button"
                                  className={`chip ${isSelected ? "active" : ""}`}
                                  onClick={() => {
                                    const newFeatures = isSelected
                                      ? currentFeatures.filter(f => f !== feat)
                                      : [...currentFeatures, feat];
                                    updateNodeConfigField("features", newFeatures);
                                  }}
                                >
                                  {feat.replace(/_/g, " ")}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "shape_features" ? (
                    <>
                      {/* Bloco Shape Features */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.shapeFeatures")}</legend>
                        <p className="config-hint">Extrai características da geometria da curva</p>
                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          {availableChannels.length > 0 ? (
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            >
                              <option value="">Todos os canais</option>
                              {availableChannels.map((ch) => (
                                <option key={ch} value={ch}>{ch}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder="vazio = todos"
                            />
                          )}
                        </div>
                        <div className="config-field">
                          <label className="config-label">Janela de suavização</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.smoothing_window ?? 5}
                            onChange={(e) => updateNodeConfigField("smoothing_window", parseInt(e.target.value) || 5)}
                            min="1"
                            max="20"
                          />
                          <small className="config-hint-small">Suaviza derivadas para evitar ruído</small>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Features a extrair</label>
                          <div className="chip-group">
                            {["inflection_points", "peaks", "valleys", "zero_crossings", "slope_start", "slope_end", "auc", "max_derivative"].map((feat) => {
                              const currentFeatures = selectedNode.data.config?.features || ["inflection_points", "peaks", "auc"];
                              const isSelected = currentFeatures.includes(feat);
                              return (
                                <button
                                  key={feat}
                                  type="button"
                                  className={`chip ${isSelected ? "active" : ""}`}
                                  onClick={() => {
                                    const newFeatures = isSelected
                                      ? currentFeatures.filter(f => f !== feat)
                                      : [...currentFeatures, feat];
                                    updateNodeConfigField("features", newFeatures);
                                  }}
                                >
                                  {feat.replace(/_/g, " ")}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "growth_features" ? (
                    <>
                      {/* Bloco Growth Features */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.growthFeatures")}</legend>
                        <p className="config-hint">Extrai parâmetros microbiológicos (numéricos) das curvas</p>
                        
                        {/* Seleção de canais via chips */}
                        <div className="config-field">
                          <label className="config-label">Canais a processar</label>
                          {availableFitChannels.length > 0 ? (
                            <>
                              <div className="chip-group">
                                {(() => {
                                  const selectedChannels = selectedNode.data.config?.selected_channels || [];
                                  const normalizedSelected = Array.isArray(selectedChannels) ? selectedChannels : 
                                    (typeof selectedChannels === 'string' && selectedChannels.trim() 
                                      ? selectedChannels.split(',').map(s => s.trim()).filter(Boolean) 
                                      : []);
                                  const isAllSelected = normalizedSelected.length === 0 || normalizedSelected.length === availableFitChannels.length;
                                  
                                  return (
                                    <>
                                      <button
                                        type="button"
                                        className={`chip chip-all ${isAllSelected ? "active" : ""}`}
                                        onClick={() => updateNodeConfigField("selected_channels", [])}
                                        title="Processar todos os canais"
                                      >
                                        {t("common.all")}
                                      </button>
                                      {availableFitChannels.map((chName) => {
                                        const isActive = isAllSelected ? false : normalizedSelected.includes(chName);
                                        return (
                                          <button
                                            key={chName}
                                            type="button"
                                            className={`chip ${isActive ? "active" : ""}`}
                                            onClick={() => {
                                              if (isAllSelected) {
                                                // Se "Todos" está selecionado, clicar em um canal seleciona só ele
                                                updateNodeConfigField("selected_channels", [chName]);
                                                return;
                                              }
                                              const nextSet = new Set(normalizedSelected);
                                              if (nextSet.has(chName)) nextSet.delete(chName);
                                              else nextSet.add(chName);
                                              // Se ficou vazio ou todos selecionados, volta para []
                                              const arr = Array.from(nextSet);
                                              updateNodeConfigField("selected_channels", arr.length === availableFitChannels.length ? [] : arr);
                                            }}
                                          >
                                            {chName}
                                          </button>
                                        );
                                      })}
                                    </>
                                  );
                                })()}
                              </div>
                              <small className="config-hint-small">
                                Selecione os canais ou clique em "Todos"
                              </small>
                            </>
                          ) : (
                            <p className="config-hint" style={{ marginTop: 4 }}>
                              Conecte a um curve_fit ou load_data para ver os canais disponíveis
                            </p>
                          )}
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Features a extrair</label>
                          <div className="chip-group">
                            {["lag_time", "growth_rate", "asymptote", "doubling_time", "inflection_time", "y0", "r_squared", "rmse", "aic"].map((feat) => {
                              const currentFeatures = selectedNode.data.config?.features || ["lag_time", "growth_rate", "asymptote", "r_squared"];
                              const isSelected = currentFeatures.includes(feat);
                              return (
                                <button
                                  key={feat}
                                  type="button"
                                  className={`chip ${isSelected ? "active" : ""}`}
                                  onClick={() => {
                                    const newFeatures = isSelected
                                      ? currentFeatures.filter(f => f !== feat)
                                      : [...currentFeatures, feat];
                                    updateNodeConfigField("features", newFeatures);
                                  }}
                                >
                                  {feat === "growth_rate" ? "µmax" : feat === "lag_time" ? "λ (lag)" : feat.replace(/_/g, " ")}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "features_merge" ? (
                    <>
                      {/* Bloco Features Merge */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.combineFeatures")}</legend>
                        <p className="config-hint">Combina múltiplos blocos de features em um único output</p>
                        
                        <div className="config-info">
                          <pre className="gate-diagram">{`features_a  \\\\\nfeatures_b ---- MERGE ---- features\nfeatures_c  ///\nfeatures_d  \\\\\\n`}</pre>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.featuresInputsLabel")}</label>
                          {(() => {
                            const configured = selectedNode.data.config?.features_inputs;
                            const baseInputs = Array.isArray(configured) && configured.length
                              ? configured
                              : (selectedNode.data.dataInputs || []).filter((k) => String(k).startsWith("features_"));
                            const inputs = baseInputs.length ? baseInputs : ["features_a"];

                            return (
                              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                  {inputs.map((key) => (
                                    <span
                                      key={key}
                                      className="chip active"
                                      style={{ cursor: "default", display: "inline-flex", alignItems: "center", gap: 6 }}
                                    >
                                      <code>{key}</code>
                                      <button
                                        type="button"
                                        className="btn btn-tertiary"
                                        onClick={() => {
                                          if (inputs.length <= 1) return;
                                          setFeaturesMergeInputs(selectedNode.id, inputs.filter((k) => k !== key));
                                        }}
                                        title={t("actions.remove")}
                                        aria-label={t("actions.remove")}
                                        style={{
                                          padding: "0 6px",
                                          height: 20,
                                          lineHeight: "18px",
                                          fontSize: 12,
                                          borderRadius: 999,
                                        }}
                                      >
                                        ×
                                      </button>
                                    </span>
                                  ))}
                                </div>
                                <div>
                                  <button
                                    type="button"
                                    className="btn btn-secondary"
                                    onClick={() => {
                                      const used = new Set(inputs.map(String));
                                      let next = null;
                                      for (let i = 0; i < 26; i += 1) {
                                        const cand = `features_${String.fromCharCode(97 + i)}`;
                                        if (!used.has(cand)) {
                                          next = cand;
                                          break;
                                        }
                                      }
                                      if (!next) {
                                        next = `features_${inputs.length + 1}`;
                                      }
                                      setFeaturesMergeInputs(selectedNode.id, [...inputs, next]);
                                    }}
                                  >
                                    + {t("configuration.addInput")}
                                  </button>
                                  <div className="config-hint-small" style={{ marginTop: 6 }}>
                                    {t("configuration.featuresMergeInputsHint")}
                                  </div>
                                </div>
                              </div>
                            );
                          })()}
                        </div>
                         
                        <div className="config-field">
                          <label className="config-label">Modo de Merge</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.merge_mode || "flat"}
                            onChange={(e) => updateNodeConfigField("merge_mode", e.target.value)}
                          >
                            <option value="flat">Flat (combina por canal)</option>
                            <option value="grouped">Grouped (agrupa por fonte)</option>
                          </select>
                          <small className="config-hint-small">
                            Flat: combina todas features no mesmo canal<br/>
                            Grouped: mantém separação por bloco fonte
                          </small>
                        </div>
                        
                        <div className="config-hint" style={{marginTop: '12px'}}>
                          <strong>{t("configuration.connectFeaturesTitle")}</strong>
                          <ul style={{fontSize: '12px', margin: '4px 0 0 16px'}}>
                            <li><code>features_a</code> ← statistical_features</li>
                            <li><code>features_b</code> ← temporal_features</li>
                            <li><code>features_c</code> ← shape_features</li>
                            <li><code>features_d</code> ← growth_features</li>
                          </ul>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar features combinadas (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName?.endsWith("_conversion") ? (
                    <>
                      {/* Blocos de Conversão Espectral Individual */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.spectralConversion")}</legend>
                        <div className="config-info">
                          <strong>Espaço de cor:</strong> {selectedNode.data.blockName.replace("_conversion", "").toUpperCase()}
                        </div>

                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.calculate_matrix ?? true)} onChange={(e) => updateNodeConfigField("calculate_matrix", e.target.checked)} />
                            <span>Usar matriz de conversão (referência)</span>
                          </label>
                        </div>
                        
                        <div className="config-hint">
                          <small>
                            {t("configuration.defaultsInfoTitle")}:<br />
                            • Turbidimetria/Nefelometria: adaptação sim, luminosidade sim<br />
                            • Fluorescência: adaptação não, luminosidade não
                          </small>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_chromatic_adaptation ?? true)} onChange={(e) => updateNodeConfigField("apply_chromatic_adaptation", e.target.checked)} />
                            <span>Adaptação cromática (Bradford)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_luminosity_correction ?? true)} onChange={(e) => updateNodeConfigField("apply_luminosity_correction", e.target.checked)} />
                            <span>Correção de luminosidade</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_gamma)} onChange={(e) => updateNodeConfigField("apply_gamma", e.target.checked)} />
                            <span>Correção Gamma (sRGB)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.auto_exposure)} onChange={(e) => updateNodeConfigField("auto_exposure", e.target.checked)} />
                            <span>Auto Exposure</span>
                          </label>
                        </div>
                        {(selectedNode.data.blockName === "hsv_conversion" || selectedNode.data.blockName === "hsb_conversion") && (
                          <div className="config-field">
                            <label className="config-toggle">
                              <input type="checkbox" checked={Boolean(selectedNode.data.config?.return_hue_unwrapped ?? true)} onChange={(e) => updateNodeConfigField("return_hue_unwrapped", e.target.checked)} />
                              <span>Hue Unwrapped (contínuo)</span>
                            </label>
                          </div>
                        )}
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar Gráficos</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "spectral_conversion" ? (
                    <>
                      {/* Bloco Spectral Conversion */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.spectralConversion")}</legend>
                        <div className="config-field">
                          <label className="config-label">Espaço de Cor Alvo</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.target_color_space || "LAB"}
                            onChange={(e) => updateNodeConfigField("target_color_space", e.target.value)}
                          >
                            <option value="XYZ">XYZ</option>
                            <option value="RGB">RGB</option>
                            <option value="LAB">LAB (CIE L*a*b*)</option>
                            <option value="HSV">HSV</option>
                            <option value="HSB">HSB</option>
                            <option value="CMYK">CMYK</option>
                            <option value="xyY">xyY</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Tipo de Calibração</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.calibration_type || "standard"}
                            onChange={(e) => updateNodeConfigField("calibration_type", e.target.value)}
                          >
                            <option value="standard">Standard</option>
                            <option value="custom">Custom</option>
                          </select>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_chromatic_adaptation ?? true)} onChange={(e) => updateNodeConfigField("apply_chromatic_adaptation", e.target.checked)} />
                            <span>Aplicar adaptação cromática</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.apply_luminosity_correction ?? true)} onChange={(e) => updateNodeConfigField("apply_luminosity_correction", e.target.checked)} />
                            <span>Aplicar correção de luminosidade</span>
                          </label>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "amplitude_detector" ? (
                    <>
                      {/* Bloco Amplitude Detector */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.amplitudeDetector")}</legend>
                        <p className="config-hint">Detecta crescimento pela amplitude relativa (max-min) em todos os canais</p>
                        <div className="config-field">
                          <label className="config-label">Amplitude mínima (%)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.min_amplitude_percent ?? 5.0}
                            onChange={(e) => updateNodeConfigField("min_amplitude_percent", parseFloat(e.target.value) || 5.0)}
                            min="0.1"
                            max="100"
                            step="0.5"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Direção esperada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.expected_direction || "auto"}
                            onChange={(e) => updateNodeConfigField("expected_direction", e.target.value)}
                          >
                            <option value="auto">Automático</option>
                            <option value="increasing">Crescente</option>
                            <option value="decreasing">Decrescente</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "derivative_detector" ? (
                    <>
                      {/* Bloco Derivative Detector */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.derivativeDetector")}</legend>
                        <p className="config-hint">Detecta crescimento pela taxa de variação em todos os canais</p>
                        <div className="config-field">
                          <label className="config-label">Amplitude mínima (%)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.min_amplitude_percent ?? 5.0}
                            onChange={(e) => updateNodeConfigField("min_amplitude_percent", parseFloat(e.target.value) || 5.0)}
                            min="0.1"
                            max="100"
                            step="0.5"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Direção esperada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.expected_direction || "auto"}
                            onChange={(e) => updateNodeConfigField("expected_direction", e.target.value)}
                          >
                            <option value="auto">Automático</option>
                            <option value="increasing">Crescente</option>
                            <option value="decreasing">Decrescente</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ratio_detector" ? (
                    <>
                      {/* Bloco Ratio Detector */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.ratioDetector")}</legend>
                        <p className="config-hint">Detecta crescimento pela razão início/fim em todos os canais</p>
                        <div className="config-field">
                          <label className="config-label">Razão mínima (max/min)</label>
                          <input
                            type="number"
                            className="config-input"
                            value={selectedNode.data.config?.min_growth_ratio ?? 1.2}
                            onChange={(e) => updateNodeConfigField("min_growth_ratio", parseFloat(e.target.value) || 1.2)}
                            min="1.0"
                            max="10"
                            step="0.1"
                          />
                        </div>
                        <div className="config-field">
                          <label className="config-label">Direção esperada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.expected_direction || "auto"}
                            onChange={(e) => updateNodeConfigField("expected_direction", e.target.value)}
                          >
                            <option value="auto">Automático</option>
                            <option value="increasing">Crescente</option>
                            <option value="decreasing">Decrescente</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.generate_output_graphs)} onChange={(e) => updateNodeConfigField("generate_output_graphs", e.target.checked)} />
                            <span>Gerar gráfico de visualização</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "boolean_extractor" ? (
                    <>
                      {/* Bloco Boolean Extractor */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.booleanExtractor")}</legend>
                        <p className="config-hint">Extrai um booleano de <strong>source_data</strong> e passa <strong>sensor_data</strong> adiante</p>
                        <div className="config-field">
                          <label className="config-label">Caminho do campo (em source_data)</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.field_path || "has_any_growth"}
                            onChange={(e) => updateNodeConfigField("field_path", e.target.value)}
                            placeholder="ex: has_any_growth"
                          />
                          <small className="config-hint">Use ponto para campos aninhados: channel_results.R.has_growth</small>
                        </div>
                        <div className="config-field">
                          <label className="config-label">Valor padrão (se não encontrar)</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.default_value === true ? "true" : "false"}
                            onChange={(e) => updateNodeConfigField("default_value", e.target.value === "true")}
                          >
                            <option value="false">false</option>
                            <option value="true">true</option>
                          </select>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "value_in_list" ? (
                    <>
                      {/* Bloco Value In List */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.valueInList")}</legend>
                        <p className="config-hint">{t("configuration.valueInListHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.allowedValues")}</label>
                          <div className="chip-group" style={{ marginBottom: 8 }}>
                            {(() => {
                              const current = Array.isArray(selectedNode.data.config?.allowed_values)
                                ? selectedNode.data.config.allowed_values
                                : [];
                              if (!current.length) {
                                return <small className="config-hint-small">{t("configuration.noValuesConfigured")}</small>;
                              }
                              return current.map((val) => (
                                <span
                                  key={String(val)}
                                  className="chip active"
                                  style={{ cursor: "default", display: "inline-flex", alignItems: "center", gap: 6 }}
                                >
                                  <code>{String(val)}</code>
                                  <button
                                    type="button"
                                    className="btn btn-tertiary"
                                    onClick={() => {
                                      const next = current.filter((v) => String(v) !== String(val));
                                      updateNodeConfigField("allowed_values", next);
                                    }}
                                    title={t("actions.remove")}
                                    aria-label={t("actions.remove")}
                                    style={{
                                      padding: "0 6px",
                                      height: 20,
                                      lineHeight: "18px",
                                      fontSize: 12,
                                      borderRadius: 999,
                                    }}
                                  >
                                    ×
                                  </button>
                                </span>
                              ));
                            })()}
                          </div>

                          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                            <input
                              type="text"
                              className="config-input"
                              value={valueInListDraft}
                              onChange={(e) => setValueInListDraft(e.target.value)}
                              placeholder={t("configuration.valueToAdd")}
                            />
                            <button
                              type="button"
                              className="btn btn-secondary"
                              onClick={() => {
                                const nextValue = String(valueInListDraft || "").trim();
                                if (!nextValue) return;
                                const current = Array.isArray(selectedNode.data.config?.allowed_values)
                                  ? selectedNode.data.config.allowed_values.map(String)
                                  : [];
                                if (current.includes(nextValue)) {
                                  setValueInListDraft("");
                                  return;
                                }
                                updateNodeConfigField("allowed_values", [...current, nextValue]);
                                setValueInListDraft("");
                              }}
                            >
                              + {t("configuration.addInput")}
                            </button>
                          </div>

                          <div className="config-hint-small" style={{ marginTop: 6 }}>
                            {t("configuration.valueInListConnectHint")}
                          </div>
                        </div>

                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.invert)}
                              onChange={(e) => updateNodeConfigField("invert", e.target.checked)}
                            />
                            <span>{t("configuration.invertMatch")}</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.case_sensitive)}
                              onChange={(e) => updateNodeConfigField("case_sensitive", e.target.checked)}
                            />
                            <span>{t("configuration.caseSensitive")}</span>
                          </label>
                        </div>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={selectedNode.data.config?.trim !== false}
                              onChange={(e) => updateNodeConfigField("trim", e.target.checked)}
                            />
                            <span>{t("configuration.trimValues")}</span>
                          </label>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "numeric_compare" ? (
                    <>
                      {/* Bloco Numeric Compare */}
                      <fieldset className="config-group">
                        <legend>Comparação numérica</legend>
                        <p className="config-hint">Compara um valor numérico com um threshold. Útil para verificar se diluição != 1, por exemplo.</p>

                        <div className="config-field">
                          <label className="config-label">Operador</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.operator || "!="}
                            onChange={(e) => updateNodeConfigField("operator", e.target.value)}
                          >
                            <option value="==">Igual a (==)</option>
                            <option value="!=">Diferente de (!=)</option>
                            <option value=">">Maior que (&gt;)</option>
                            <option value=">=">Maior ou igual (&gt;=)</option>
                            <option value="<">Menor que (&lt;)</option>
                            <option value="<=">Menor ou igual (&lt;=)</option>
                          </select>
                        </div>

                        <div className="config-field">
                          <label className="config-label">Threshold (valor de comparação)</label>
                          <input
                            type="number"
                            className="config-input"
                            step="any"
                            value={selectedNode.data.config?.threshold ?? 1}
                            onChange={(e) => updateNodeConfigField("threshold", parseFloat(e.target.value) || 0)}
                          />
                          <small className="config-hint-small">
                            Exemplo: threshold=1 com operador "!=" detecta se diluição é diferente de 1
                          </small>
                        </div>

                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.invert)}
                              onChange={(e) => updateNodeConfigField("invert", e.target.checked)}
                            />
                            <span>Inverter resultado</span>
                          </label>
                        </div>

                        <div className="config-info" style={{marginTop: 8}}>
                          <pre className="gate-diagram">{`dilution_factor ---- CMP ---- condition\n                     ↑\n               threshold=${selectedNode.data.config?.threshold ?? 1}\n               operator="${selectedNode.data.config?.operator || "!="}"                     `}</pre>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "condition_gate" ? (
                    <>
                      {/* Bloco Condition Gate */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.conditionalGate")}</legend>
                        <p className="config-hint">Passa os dados somente se a condição bater com o valor esperado</p>
                        <div className="config-field">
                          <label className="config-label">Passar dados quando condição for:</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.pass_when === false ? "false" : "true"}
                            onChange={(e) => updateNodeConfigField("pass_when", e.target.value === "true")}
                          >
                            <option value="true">{t("options.trueCondition")}</option>
                            <option value="false">{t("options.falseCondition")}</option>
                          </select>
                        </div>
                        <div className="config-info">
                          <pre className="gate-diagram">{`data -----\\\\\n          ---- GATE ---- data (ou inativo)\ncondition --//\n`}</pre>
                          <p className="config-hint" style={{marginTop: '8px'}}>
                            <strong>Exemplo:</strong><br/>
                            • pass_when=TRUE + condição=true → passa dados<br/>
                            • pass_when=TRUE + condição=false → dados inativos<br/>
                            • pass_when=FALSE + condição=false → passa dados<br/>
                            • pass_when=FALSE + condição=true → dados inativos
                          </p>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "and_gate" ? (
                    <>
                      {/* Bloco AND Gate */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.andGate")}</legend>
                        <p className="config-hint">Retorna <strong>true</strong> se AMBAS condições de entrada forem true</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`condition_a --\\\\\n              ---- AND ---- result\ncondition_b --//\n`}</pre>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "or_gate" ? (
                    <>
                      {/* Bloco OR Gate */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.orGate")}</legend>
                        <p className="config-hint">Retorna <strong>true</strong> se PELO MENOS UMA condição de entrada for true</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`condition_a --\\\\\n              ---- OR ---- result\ncondition_b --//\n`}</pre>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "not_gate" ? (
                    <>
                      {/* Bloco NOT Gate */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.notGate")}</legend>
                        <p className="config-hint">Inverte o valor booleano: true→false, false→true</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`condition ---- NOT ---- result\n`}</pre>
                        </div>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "condition_branch" ? (
                    <>
                      {/* Bloco Condition Branch */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.conditionalBranch")}</legend>
                        <p className="config-hint">Direciona os dados para uma das duas saídas baseado na condição</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`              /-- data_if_true (se condition=true)\ndata + condition --|\n              \\\\-- data_if_false (se condition=false)\n`}</pre>
                        </div>
                        <p className="config-hint" style={{ marginTop: 8 }}>{t("configuration.inactiveOutputHint")}</p>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "merge" ? (
                    <>
                      {/* Bloco Merge */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.flowMerge")}</legend>
                        <p className="config-hint">Junta dois fluxos condicionais - passa adiante o que estiver ativo</p>
                        <div className="config-info">
                          <pre className="gate-diagram">{`data_a --\\\\\n         ---- MERGE ---- data\ndata_b --//\n`}</pre>
                        </div>
                        <p className="config-hint" style={{ marginTop: 8 }}>{t("configuration.flowMergeHint")}</p>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "label" ? (
                    <>
                      {/* Bloco Label */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.labelTag")}</legend>
                        <p className="config-hint">Adiciona uma tag/identificador aos dados para agrupamento</p>
                        <div className="config-info">
                          <pre className="gate-diagram" style={{fontSize: '11px'}}>{`experiment_data --\\\\\n                --[ label ]--+-- experiment_data (com tag)\nexperiment -------//          \\\\-- experiment (com tag)\n`}</pre>
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Identificador (Label)</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.label || ""}
                            onChange={(e) => updateNodeConfigField("label", e.target.value)}
                            placeholder="ex: ecoli, coliformes, salmonella"
                          />
                          <small className="config-hint-small">Esta tag será propagada através dos blocos seguintes</small>
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label" htmlFor={`${selectedNode.id}-label_color`}>
                            {t("configuration.labelColor")}
                          </label>
                          <div className="config-color-row">
                            <input
                              id={`${selectedNode.id}-label_color`}
                              type="color"
                              className="config-color-input"
                              value={
                                typeof selectedNode.data.config?.label_color === "string" &&
                                /^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(selectedNode.data.config.label_color.trim())
                                  ? selectedNode.data.config.label_color.trim()
                                  : "#6366f1"
                              }
                              onChange={(e) => updateNodeConfigField("label_color", e.target.value)}
                            />
                            <input
                              type="text"
                              className="config-input config-color-text"
                              value={typeof selectedNode.data.config?.label_color === "string" ? selectedNode.data.config.label_color : ""}
                              placeholder="#RRGGBB"
                              onChange={(e) => updateNodeConfigField("label_color", e.target.value)}
                            />
                            <button
                              type="button"
                              className="config-color-clear"
                              onClick={() => updateNodeConfigField("label_color", undefined)}
                            >
                              {t("actions.clear")}
                            </button>
                          </div>
                          <small className="config-hint-small">{t("configuration.labelColorHint")}</small>
                        </div>

                        <p className="config-hint" style={{marginTop: 8, background: '#d4edda', padding: 8, borderRadius: 4, border: '1px solid #28a745'}}>
                          <strong>{t("configuration.positionLabel")}:</strong> Entre experiment_fetch e xxx_extraction<br/>
                          <code style={{fontSize: 10}}>[experiment_fetch] → [label] → [fluorescence_extraction] → ...</code>
                        </p>
                        
                        <p className="config-hint" style={{marginTop: 8, background: '#e7f3ff', padding: 8, borderRadius: 4}}>
                          <strong>{t("configuration.connectionsLabel")}:</strong><br/>
                          • <code>experiment_data</code> do fetch → entrada <code>experiment_data</code> do label<br/>
                          • Saída <code>experiment_data</code> do label → xxx_extraction
                        </p>
                        
                        <p className="config-hint" style={{marginTop: 8}}>
                          <strong>Exemplo de saída do response_builder:</strong>
                          <code style={{display: 'block', marginTop: 4, fontSize: 11, background: '#f5f5f5', padding: 4}}>
                            {`{
  "presence_ecoli": true,
  "predict_nmp_ecoli": 2.99,
  "presence_coliformes": true,
  "predict_nmp_coliformes": 1.66
}`}
                          </code>
                        </p>
                      </fieldset>
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar dados processados (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_inference" ? (
                    <>
                      {/* Bloco ML Inference */}
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlInference")}</legend>
                        <p className="config-hint">Executa predição usando modelos ONNX treinados</p>
                        
                        <div className="config-field">
                          <label className="config-label">Arquivo do modelo (.onnx)</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                          <small className="config-hint-small">Caminho do arquivo do modelo no servidor</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">Arquivo do scaler (.joblib)</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">Obrigatório quando model_path estiver preenchido</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">Unidade de saída</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.output_unit || ""}
                            onChange={(e) => updateNodeConfigField("output_unit", e.target.value)}
                            placeholder="NMP/100mL ou UFC/mL"
                          />
                          <small className="config-hint-small">Será propagada para os resultados e para o response_builder</small>
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Feature de Entrada</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.input_feature || "growth_rate"}
                            onChange={(e) => updateNodeConfigField("input_feature", e.target.value)}
                          >
                            <optgroup label="Growth Features (do curve_fit)">
                              <option value="growth_rate">µmax (growth_rate)</option>
                              <option value="asymptote">Assíntota (A)</option>
                              <option value="lag_time">Lag Time (λ)</option>
                              <option value="inflection_time">Tempo de Inflexão (T)</option>
                              <option value="doubling_time">Tempo de Duplicação</option>
                            </optgroup>
                            <optgroup label="Statistical Features">
                              <option value="max">Máximo</option>
                              <option value="min">Mínimo</option>
                              <option value="mean">Média</option>
                              <option value="std">Desvio Padrão</option>
                              <option value="range">Range (max-min)</option>
                            </optgroup>
                            <optgroup label="Shape Features">
                              <option value="auc">Área sob a Curva (AUC)</option>
                              <option value="slope_start">Inclinação Inicial</option>
                              <option value="slope_end">Inclinação Final</option>
                              <option value="max_derivative">Derivada Máxima</option>
                            </optgroup>
                            <optgroup label="Temporal Features">
                              <option value="time_to_max">Tempo até Máximo</option>
                              <option value="time_to_threshold">Tempo até Threshold</option>
                              <option value="duration">Duração Total</option>
                            </optgroup>
                          </select>
                          <small className="config-hint-small">Valor que será enviado ao modelo</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">Canal</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.channel || ""}
                            onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            placeholder="vazio = primeiro disponível"
                          />
                          <small className="config-hint-small">Canal das features para usar (ex: R, f1, X)</small>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group">
                        <legend>{t("configuration.output")}</legend>
                        <div className="config-info">
                          <p><strong>Saída:</strong> <code>prediction</code></p>
                          <ul style={{fontSize: '12px', marginLeft: '16px', marginTop: '4px'}}>
                            <li><code>value</code> - Valor predito</li>
                            <li><code>unit</code> - Unidade (NMP/100mL ou UFC/mL)</li>
                            <li><code>input_feature</code> - Feature usada</li>
                            <li><code>input_value</code> - Valor da feature</li>
                          </ul>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar detalhes da inferência (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_inference_series" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlInferenceSeries")}</legend>
                        <p className="config-hint">{t("configuration.mlSeriesHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.modelPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.scalerPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">{t("configuration.scalerOptionalHint")}</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.channel")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.channel || ""}
                            onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                            placeholder={t("configuration.channelPlaceholder")}
                          />
                          <small className="config-hint-small">{t("configuration.channelHint")}</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.inputLayout")}</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.input_layout || "sequence"}
                            onChange={(e) => updateNodeConfigField("input_layout", e.target.value)}
                          >
                            <option value="sequence">{t("configuration.layouts.sequence")}</option>
                            <option value="flat">{t("configuration.layouts.flat")}</option>
                            <option value="channels_first">{t("configuration.layouts.channelsFirst")}</option>
                          </select>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.maxLength")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.max_length ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("max_length", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="ex: 2048"
                              min="1"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.align")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.align || "end"}
                              onChange={(e) => updateNodeConfigField("align", e.target.value)}
                            >
                              <option value="end">{t("configuration.alignOptions.end")}</option>
                              <option value="start">{t("configuration.alignOptions.start")}</option>
                            </select>
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group">
                        <legend>{t("configuration.output")}</legend>
                        <div className="config-field">
                          <label className="config-label">{t("configuration.outputUnit")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.output_unit || ""}
                            onChange={(e) => updateNodeConfigField("output_unit", e.target.value)}
                            placeholder="ex: NMP/100mL, UFC/mL"
                          />
                          <small className="config-hint-small">{t("configuration.outputUnitHint")}</small>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_inference_multichannel" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlInferenceMultichannel")}</legend>
                        <p className="config-hint">{t("configuration.mlMultichannelHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.modelPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.scalerPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">{t("configuration.scalerOptionalHint")}</small>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.channels")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={
                              Array.isArray(selectedNode.data.config?.channels)
                                ? selectedNode.data.config.channels.join(", ")
                                : selectedNode.data.config?.channels || ""
                            }
                            onChange={(e) => {
                              const raw = e.target.value;
                              const parsed = raw
                                .split(",")
                                .map((s) => s.trim())
                                .filter(Boolean);
                              updateNodeConfigField("channels", parsed.length ? parsed : raw);
                            }}
                            placeholder="ex: f1, f2, clear"
                          />
                          <small className="config-hint-small">{t("configuration.channelsHint")}</small>
                        </div>
                        {availableChannels.length > 0 && (() => {
                          const selectedList = Array.isArray(selectedNode.data.config?.channels)
                            ? selectedNode.data.config.channels.map(String).filter(Boolean)
                            : typeof selectedNode.data.config?.channels === "string"
                              ? selectedNode.data.config.channels
                                  .split(",")
                                  .map((v) => v.trim())
                                  .filter(Boolean)
                              : [];
                          const isAll =
                            selectedList.length === 0 || selectedList.length === availableChannels.length;
                          return (
                            <>
                              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
                                <span style={{ fontSize: 11, color: "var(--color-gray-500)" }}>
                                  {t("configuration.availableChannels", { count: availableChannels.length })}
                                </span>
                              </div>
                              <div className="chip-group" style={{ marginTop: 6 }}>
                                <button
                                  type="button"
                                  className={`chip chip-all ${isAll ? "active" : ""}`}
                                  onClick={() => updateNodeConfigField("channels", [])}
                                  title={t("configuration.allChannels")}
                                >
                                  {t("common.all")}
                                </button>
                                {availableChannels.map((chName) => {
                                  const isActive = isAll ? false : selectedList.includes(chName);
                                  return (
                                    <button
                                      key={chName}
                                      type="button"
                                      className={`chip ${isActive ? "active" : ""}`}
                                      onClick={() => {
                                        if (isAll) {
                                          updateNodeConfigField("channels", [chName]);
                                          return;
                                        }
                                        const nextSet = new Set(selectedList);
                                        if (nextSet.has(chName)) nextSet.delete(chName);
                                        else nextSet.add(chName);
                                        updateNodeConfigField("channels", Array.from(nextSet));
                                      }}
                                    >
                                      {chName}
                                    </button>
                                  );
                                })}
                              </div>
                            </>
                          );
                        })()}

                        <div className="config-field">
                          <label className="config-label">{t("configuration.inputLayout")}</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.input_layout || "time_channels"}
                            onChange={(e) => updateNodeConfigField("input_layout", e.target.value)}
                          >
                            <option value="time_channels">{t("configuration.layouts.timeChannels")}</option>
                            <option value="channels_time">{t("configuration.layouts.channelsTime")}</option>
                          </select>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.maxLength")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.max_length ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("max_length", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="ex: 2048"
                              min="1"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.align")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.align || "end"}
                              onChange={(e) => updateNodeConfigField("align", e.target.value)}
                            >
                              <option value="end">{t("configuration.alignOptions.end")}</option>
                              <option value="start">{t("configuration.alignOptions.start")}</option>
                            </select>
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group">
                        <legend>{t("configuration.output")}</legend>
                        <div className="config-field">
                          <label className="config-label">{t("configuration.outputUnit")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.output_unit || ""}
                            onChange={(e) => updateNodeConfigField("output_unit", e.target.value)}
                            placeholder="ex: NMP/100mL, UFC/mL"
                          />
                          <small className="config-hint-small">{t("configuration.outputUnitHint")}</small>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_transform_series" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlTransformSeries")}</legend>
                        <p className="config-hint">{t("configuration.mlTransformHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.modelPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.scalerPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">{t("configuration.scalerOptionalHint")}</small>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.channel")}</label>
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder={t("configuration.channelPlaceholder")}
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.outputChannel")}</label>
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.output_channel || "ml"}
                              onChange={(e) => updateNodeConfigField("output_channel", e.target.value)}
                              placeholder="ml"
                            />
                            <small className="config-hint-small">{t("configuration.outputChannelHint")}</small>
                          </div>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.inputLayout")}</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.input_layout || "sequence"}
                            onChange={(e) => updateNodeConfigField("input_layout", e.target.value)}
                          >
                            <option value="sequence">{t("configuration.layouts.sequence")}</option>
                            <option value="flat">{t("configuration.layouts.flat")}</option>
                            <option value="channels_first">{t("configuration.layouts.channelsFirst")}</option>
                          </select>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.maxLength")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.max_length ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("max_length", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="ex: 2048"
                              min="1"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.align")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.align || "end"}
                              onChange={(e) => updateNodeConfigField("align", e.target.value)}
                            >
                              <option value="end">{t("configuration.alignOptions.end")}</option>
                              <option value="start">{t("configuration.alignOptions.start")}</option>
                            </select>
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "ml_detector" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.mlDetector")}</legend>
                        <p className="config-hint">{t("configuration.mlDetectorHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.modelPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.model_path || ""}
                            onChange={(e) => updateNodeConfigField("model_path", e.target.value)}
                            placeholder="resources/model.onnx"
                          />
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.scalerPath")}</label>
                          <input
                            type="text"
                            className="config-input"
                            value={selectedNode.data.config?.scaler_path || ""}
                            onChange={(e) => updateNodeConfigField("scaler_path", e.target.value)}
                            placeholder="resources/scaler.joblib"
                          />
                          <small className="config-hint-small">{t("configuration.scalerOptionalHint")}</small>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.channel")}</label>
                            <input
                              type="text"
                              className="config-input"
                              value={selectedNode.data.config?.channel || ""}
                              onChange={(e) => updateNodeConfigField("channel", e.target.value)}
                              placeholder={t("configuration.channelPlaceholder")}
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.inputLayout")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.input_layout || "sequence"}
                              onChange={(e) => updateNodeConfigField("input_layout", e.target.value)}
                            >
                              <option value="sequence">{t("configuration.layouts.sequence")}</option>
                              <option value="flat">{t("configuration.layouts.flat")}</option>
                              <option value="channels_first">{t("configuration.layouts.channelsFirst")}</option>
                            </select>
                          </div>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.threshold")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.threshold ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("threshold", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0.5"
                              step="0.01"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.operator")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.operator || ">="}
                              onChange={(e) => updateNodeConfigField("operator", e.target.value)}
                            >
                              <option value=">=">≥</option>
                              <option value=">">&gt;</option>
                              <option value="<=">≤</option>
                              <option value="<">&lt;</option>
                            </select>
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.align")}</label>
                            <select
                              className="config-select"
                              value={selectedNode.data.config?.align || "end"}
                              onChange={(e) => updateNodeConfigField("align", e.target.value)}
                            >
                              <option value="end">{t("configuration.alignOptions.end")}</option>
                              <option value="start">{t("configuration.alignOptions.start")}</option>
                            </select>
                          </div>
                        </div>

                        <div className="config-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.maxLength")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.max_length ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("max_length", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="ex: 2048"
                              min="1"
                            />
                          </div>
                          <div className="config-field">
                            <label className="config-label">{t("configuration.padValue")}</label>
                            <input
                              type="number"
                              className="config-input"
                              value={selectedNode.data.config?.pad_value ?? ""}
                              onChange={(e) =>
                                updateNodeConfigField("pad_value", e.target.value === "" ? undefined : Number(e.target.value))
                              }
                              placeholder="0"
                            />
                          </div>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>{t("configuration.includeRawOutput")}</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : ["response_builder", "response_pack"].includes(selectedNode?.data?.blockName) ? (
                    <>
                      {/* Bloco Response Builder / Response Pack */}
                      <fieldset className="config-group">
                        <legend>
                          {selectedNode?.data?.blockName === "response_pack"
                            ? t("configuration.responsePack")
                            : t("configuration.responseBuilder")}
                        </legend>
                        <p className="config-hint">
                          {selectedNode?.data?.blockName === "response_pack"
                            ? t("configuration.responsePackHint")
                            : "Monta o JSON de resposta final conectando múltiplas entradas"}
                        </p>
                        
                        <div className="config-info" style={{marginBottom: '12px'}}>
                          <pre className="gate-diagram" style={{fontSize: '11px'}}>{`input_1 (condition) --\\\\\ninput_2 (predict_nmp) --+-- response\ninput_3 (predict_ufc) --//\n`}</pre>
                        </div>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.responseInputsLabel")}</label>
                          {(() => {
                            const maxInputs = 8;
                            const currentKeys = (selectedNode?.data?.dataInputs || []).filter((k) => String(k).startsWith("input_"));
                            const configuredCount = Number(selectedNode?.data?.config?.inputs_count);
                            const count = Math.max(
                              1,
                              Math.min(maxInputs, Number.isFinite(configuredCount) ? configuredCount : (currentKeys.length || 1))
                            );
                            const keys = Array.from({ length: count }, (_, i) => `input_${i + 1}`);
                            return (
                              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                  {keys.map((key, idx) => (
                                    <span
                                      key={key}
                                      className="chip active"
                                      style={{ cursor: "default", display: "inline-flex", alignItems: "center", gap: 6 }}
                                    >
                                      <code>{key}</code>
                                      <button
                                        type="button"
                                        className="btn btn-tertiary"
                                        onClick={() => {
                                          if (count <= 1) return;
                                          setSequentialInputsCount(selectedNode.id, count - 1, { prefix: "input_", max: 8, configKey: "inputs_count" });
                                        }}
                                        title={t("actions.remove")}
                                        aria-label={t("actions.remove")}
                                        style={{
                                          padding: "0 6px",
                                          height: 20,
                                          lineHeight: "18px",
                                          fontSize: 12,
                                          borderRadius: 999,
                                          opacity: idx === count - 1 ? 1 : 0.65,
                                        }}
                                        disabled={idx !== count - 1 || count <= 1}
                                      >
                                        -
                                      </button>
                                    </span>
                                  ))}
                                </div>
                                <div>
                                  <button
                                    type="button"
                                    className="btn btn-secondary"
                                    onClick={() => {
                                      if (count >= maxInputs) return;
                                      setSequentialInputsCount(selectedNode.id, count + 1, { prefix: "input_", max: 8, configKey: "inputs_count" });
                                    }}
                                    disabled={count >= maxInputs}
                                  >
                                    + {t("configuration.addInput")}
                                  </button>
                                  <div className="config-hint-small" style={{ marginTop: 6 }}>
                                    {t("configuration.responseInputsHint")}
                                  </div>
                                </div>
                              </div>
                            );
                          })()}
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Mapeamentos de Campos</label>
                          <small className="config-hint-small" style={{display: 'block', marginBottom: '8px'}}>
                            Define como extrair valores de cada entrada
                          </small>
                          
                          {/* Editor de mapeamentos */}
                          {(() => {
                            const inputCount = Math.max(
                              1,
                              Math.min(
                                8,
                                Number.isFinite(Number(selectedNode?.data?.config?.inputs_count))
                                  ? Number(selectedNode?.data?.config?.inputs_count)
                                  : ((selectedNode?.data?.dataInputs || []).filter((k) => String(k).startsWith("input_")).length || 1)
                              )
                            );
                            const mappings = (() => {
                              try {
                                const val = selectedNode.data.config?.field_mappings;
                                if (Array.isArray(val)) return val;
                                return JSON.parse(val || "[]");
                              } catch { return []; }
                            })();
                            
                            return (
                              <div className="mappings-editor">
                                {mappings.map((m, idx) => (
                                  <div key={idx} className="mapping-row" style={{
                                    display: 'grid',
                                    gridTemplateColumns: '60px 1fr 80px 30px',
                                    gap: '4px',
                                    marginBottom: '4px',
                                    alignItems: 'center'
                                  }}>
                                    <select
                                      className="config-select"
                                      style={{padding: '4px', fontSize: '11px'}}
                                      value={m.input || 1}
                                      onChange={(e) => {
                                        const newMappings = [...mappings];
                                        newMappings[idx] = {...m, input: parseInt(e.target.value)};
                                        updateNodeConfigField("field_mappings", newMappings);
                                      }}
                                    >
                                      {Array.from({ length: inputCount }, (_, i) => i + 1).map((i) => (
                                        <option key={i} value={i}>input_{i}</option>
                                      ))}
                                    </select>
                                    <input
                                      type="text"
                                      className="config-input"
                                      style={{padding: '4px', fontSize: '11px'}}
                                      value={m.field || ""}
                                      onChange={(e) => {
                                        const newMappings = [...mappings];
                                        newMappings[idx] = {...m, field: e.target.value};
                                        updateNodeConfigField("field_mappings", newMappings);
                                      }}
                                      placeholder="nome_campo"
                                    />
                                    <input
                                      type="text"
                                      className="config-input"
                                      style={{padding: '4px', fontSize: '11px'}}
                                      value={m.path || ""}
                                      onChange={(e) => {
                                        const newMappings = [...mappings];
                                        newMappings[idx] = {...m, path: e.target.value};
                                        updateNodeConfigField("field_mappings", newMappings);
                                      }}
                                      placeholder="path"
                                    />
                                    <button
                                      type="button"
                                      className="btn-icon"
                                      style={{padding: '2px 6px', fontSize: '14px'}}
                                      onClick={() => {
                                        const newMappings = mappings.filter((_, i) => i !== idx);
                                        updateNodeConfigField("field_mappings", newMappings);
                                      }}
                                    >
                                      ×
                                    </button>
                                  </div>
                                ))}
                                
                                <button
                                  type="button"
                                  className="btn-secondary"
                                  style={{marginTop: '4px', padding: '4px 8px', fontSize: '11px'}}
                                  onClick={() => {
                                    const newMappings = [...mappings, {input: 1, field: "", path: ""}];
                                    updateNodeConfigField("field_mappings", newMappings);
                                  }}
                                >
                                  + Adicionar Campo
                                </button>
                              </div>
                            );
                          })()}
                          
                          <small className="config-hint-small" style={{marginTop: '8px', display: 'block'}}>
                            <strong>Paths comuns:</strong><br/>
                            • <code>.</code> ou vazio → valor direto<br/>
                            • <code>value</code> → prediction.value<br/>
                            • <code>success</code> → condition boolean
                          </small>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group">
                        <legend>{t("configuration.staticFields")}</legend>
                        <div className="config-field">
                          <label className="config-label">Campos fixos (JSON)</label>
                          <textarea
                            className="config-textarea"
                            style={{minHeight: '60px', fontFamily: 'monospace', fontSize: '11px'}}
                            value={(() => {
                              try {
                                const val = selectedNode.data.config?.static_fields;
                                if (typeof val === 'object') return JSON.stringify(val, null, 2);
                                return val || '{\n  "analysis_mode": "prediction"\n}';
                              } catch { return '{}'; }
                            })()}
                            onChange={(e) => {
                              try {
                                const parsed = JSON.parse(e.target.value);
                                updateNodeConfigField("static_fields", parsed);
                              } catch {
                                updateNodeConfigField("static_fields", e.target.value);
                              }
                            }}
                            placeholder='{"analysis_mode": "prediction"}'
                          />
                          <small className="config-hint-small">Campos que serão adicionados à resposta</small>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group">
                        <legend>{t("configuration.groupByLabel")}</legend>
                        <p className="config-hint">Se os dados vierem de blocos com label, agrupa automaticamente</p>
                        
                        <div className="config-field">
                          <label className="config-toggle">
                            <input 
                              type="checkbox" 
                              checked={selectedNode.data.config?.group_by_label !== false} 
                              onChange={(e) => updateNodeConfigField("group_by_label", e.target.checked)} 
                            />
                            <span>Agrupar resultados por label</span>
                          </label>
                          <small className="config-hint-small">Detecta automaticamente dados vindos de blocos label</small>
                        </div>
                        
                        <div className="config-field">
                          <label className="config-label">Formato de Agrupamento</label>
                          <select
                            className="config-select"
                            value={selectedNode.data.config?.flatten_labels !== false ? "flat" : "nested"}
                            onChange={(e) => updateNodeConfigField("flatten_labels", e.target.value === "flat")}
                          >
                            <option value="flat">Prefixo (presence_ecoli, predict_nmp_ecoli)</option>
                            <option value="nested">Aninhado (ecoli: {"{"} presence, predict_nmp {"}"})</option>
                          </select>
                          <small className="config-hint-small">Como os campos com label aparecerão na resposta</small>
                        </div>
                        
                        <div className="config-info" style={{marginTop: '8px', background: '#f0f8ff', padding: '8px', borderRadius: '4px', fontSize: '11px'}}>
                          <strong>Exemplo (modo prefixo):</strong>
                          <code style={{display: 'block', marginTop: '4px', whiteSpace: 'pre', background: '#f5f5f5', padding: '4px'}}>
{`{
  "presence_ecoli": true,
  "predict_nmp_ecoli": 2.99,
  "presence_coliformes": true,
  "predict_nmp_coliformes": 1.66
}`}
                          </code>
                        </div>
                      </fieldset>
                      
                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input type="checkbox" checked={Boolean(selectedNode.data.config?.include_raw_output)} onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)} />
                            <span>Mostrar resposta montada (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : selectedNode?.data?.blockName === "response_merge" ? (
                    <>
                      <fieldset className="config-group">
                        <legend>{t("configuration.responseMerge")}</legend>
                        <p className="config-hint">{t("configuration.responseMergeHint")}</p>

                        <div className="config-field">
                          <label className="config-label">{t("configuration.responseInputsLabel")}</label>
                          {(() => {
                            const maxInputs = 8;
                            const currentKeys = (selectedNode?.data?.dataInputs || []).filter((k) => String(k).startsWith("input_"));
                            const configuredCount = Number(selectedNode?.data?.config?.inputs_count);
                            const count = Math.max(
                              1,
                              Math.min(maxInputs, Number.isFinite(configuredCount) ? configuredCount : (currentKeys.length || 1))
                            );
                            const keys = Array.from({ length: count }, (_, i) => `input_${i + 1}`);
                            return (
                              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                  {keys.map((key, idx) => (
                                    <span
                                      key={key}
                                      className="chip active"
                                      style={{ cursor: "default", display: "inline-flex", alignItems: "center", gap: 6 }}
                                    >
                                      <code>{key}</code>
                                      <button
                                        type="button"
                                        className="btn btn-tertiary"
                                        onClick={() => {
                                          if (count <= 1) return;
                                          setSequentialInputsCount(selectedNode.id, count - 1, { prefix: "input_", max: 8, configKey: "inputs_count" });
                                        }}
                                        title={t("actions.remove")}
                                        aria-label={t("actions.remove")}
                                        style={{
                                          padding: "0 6px",
                                          height: 20,
                                          lineHeight: "18px",
                                          fontSize: 12,
                                          borderRadius: 999,
                                          opacity: idx === count - 1 ? 1 : 0.65,
                                        }}
                                        disabled={idx !== count - 1 || count <= 1}
                                      >
                                        -
                                      </button>
                                    </span>
                                  ))}
                                </div>
                                <div>
                                  <button
                                    type="button"
                                    className="btn btn-secondary"
                                    onClick={() => {
                                      if (count >= maxInputs) return;
                                      setSequentialInputsCount(selectedNode.id, count + 1, { prefix: "input_", max: 8, configKey: "inputs_count" });
                                    }}
                                    disabled={count >= maxInputs}
                                  >
                                    + {t("configuration.addInput")}
                                  </button>
                                  <div className="config-hint-small" style={{ marginTop: 6 }}>
                                    {t("configuration.responseInputsHint")}
                                  </div>
                                </div>
                              </div>
                            );
                          })()}
                        </div>

                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.fail_if_multiple)}
                              onChange={(e) => updateNodeConfigField("fail_if_multiple", e.target.checked)}
                            />
                            <span>Falhar se mais de uma saída estiver preenchida</span>
                          </label>
                        </div>
                      </fieldset>

                      <fieldset className="config-group config-group-debug">
                        <legend>{t("configuration.debug")}</legend>
                        <div className="config-field">
                          <label className="config-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(selectedNode.data.config?.include_raw_output)}
                              onChange={(e) => updateNodeConfigField("include_raw_output", e.target.checked)}
                            />
                            <span>Mostrar detalhes do merge (JSON)</span>
                          </label>
                        </div>
                      </fieldset>
                    </>
                  ) : effectiveConfigInputs.length ? (
                    effectiveConfigInputs.map((field) => renderConfigField(field))
                  ) : (
                    <p className="no-config">Este bloco não possui configurações</p>
                  )}
                </div>


  );
}
