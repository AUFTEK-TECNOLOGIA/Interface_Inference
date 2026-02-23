import { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";
import { useI18n } from "../locale/i18n";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8001";

/**
 * Componente simples de gráfico de dispersão (sem dependência externa)
 */
function ScatterPlot({ data, title, xLabel, yLabel, showIdealLine = true, width = 320, height = 240 }) {
  if (!data?.length) {
    return <div className="mcp-chart-empty">Sem dados</div>;
  }

  const padding = { top: 30, right: 20, bottom: 40, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Calcular escalas
  const xVals = data.map(d => d.x);
  const yVals = data.map(d => d.y);
  const allVals = [...xVals, ...yVals];
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const range = maxVal - minVal || 1;
  const margin = range * 0.1;
  const domainMin = minVal - margin;
  const domainMax = maxVal + margin;

  const scaleX = (v) => padding.left + ((v - domainMin) / (domainMax - domainMin)) * chartWidth;
  const scaleY = (v) => padding.top + chartHeight - ((v - domainMin) / (domainMax - domainMin)) * chartHeight;

  // Gerar ticks
  const tickCount = 5;
  const tickStep = (domainMax - domainMin) / (tickCount - 1);
  const ticks = Array.from({ length: tickCount }, (_, i) => domainMin + i * tickStep);

  return (
    <div className="mcp-chart-container">
      <svg width={width} height={height} className="mcp-chart-svg">
        {/* Título */}
        <text x={width / 2} y={15} textAnchor="middle" className="mcp-chart-title">{title}</text>

        {/* Grid lines */}
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={padding.left} y1={scaleY(t)} x2={width - padding.right} y2={scaleY(t)} className="mcp-chart-grid" />
            <line x1={scaleX(t)} y1={padding.top} x2={scaleX(t)} y2={height - padding.bottom} className="mcp-chart-grid" />
          </g>
        ))}

        {/* Linha ideal (y = x) */}
        {showIdealLine && (
          <line
            x1={scaleX(domainMin)}
            y1={scaleY(domainMin)}
            x2={scaleX(domainMax)}
            y2={scaleY(domainMax)}
            className="mcp-chart-ideal-line"
          />
        )}

        {/* Pontos */}
        {data.map((d, i) => (
          <circle
            key={i}
            cx={scaleX(d.x)}
            cy={scaleY(d.y)}
            r={5}
            className={`mcp-chart-point ${d.isVal ? "mcp-chart-point--val" : "mcp-chart-point--train"}`}
          />
        ))}

        {/* Eixos */}
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} className="mcp-chart-axis" />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} className="mcp-chart-axis" />

        {/* Ticks e labels */}
        {ticks.map((t, i) => (
          <g key={i}>
            <text x={scaleX(t)} y={height - padding.bottom + 15} textAnchor="middle" className="mcp-chart-tick-label">
              {t.toFixed(1)}
            </text>
            <text x={padding.left - 8} y={scaleY(t) + 4} textAnchor="end" className="mcp-chart-tick-label">
              {t.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Labels dos eixos */}
        <text x={width / 2} y={height - 5} textAnchor="middle" className="mcp-chart-axis-label">{xLabel}</text>
        <text x={12} y={height / 2} textAnchor="middle" className="mcp-chart-axis-label" transform={`rotate(-90, 12, ${height / 2})`}>{yLabel}</text>
      </svg>
    </div>
  );
}

/**
 * Componente de gráfico de resíduos
 */
function ResidualsPlot({ data, title, width = 320, height = 240 }) {
  if (!data?.length) {
    return <div className="mcp-chart-empty">Sem dados</div>;
  }

  const padding = { top: 30, right: 20, bottom: 40, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const xVals = data.map(d => d.x);
  const yVals = data.map(d => d.y);
  const xMin = Math.min(...xVals);
  const xMax = Math.max(...xVals);
  const xRange = xMax - xMin || 1;
  const yAbsMax = Math.max(...yVals.map(Math.abs));
  const yRange = yAbsMax * 1.2 || 1;

  const scaleX = (v) => padding.left + ((v - xMin + xRange * 0.05) / (xRange * 1.1)) * chartWidth;
  const scaleY = (v) => padding.top + chartHeight / 2 - (v / yRange) * (chartHeight / 2);

  return (
    <div className="mcp-chart-container">
      <svg width={width} height={height} className="mcp-chart-svg">
        <text x={width / 2} y={15} textAnchor="middle" className="mcp-chart-title">{title}</text>

        {/* Linha zero */}
        <line
          x1={padding.left}
          y1={scaleY(0)}
          x2={width - padding.right}
          y2={scaleY(0)}
          className="mcp-chart-zero-line"
        />

        {/* Pontos */}
        {data.map((d, i) => (
          <circle
            key={i}
            cx={scaleX(d.x)}
            cy={scaleY(d.y)}
            r={5}
            className={`mcp-chart-point ${d.isVal ? "mcp-chart-point--val" : "mcp-chart-point--train"}`}
          />
        ))}

        {/* Eixos */}
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} className="mcp-chart-axis" />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} className="mcp-chart-axis" />

        <text x={width / 2} y={height - 5} textAnchor="middle" className="mcp-chart-axis-label">Predito</text>
        <text x={12} y={height / 2} textAnchor="middle" className="mcp-chart-axis-label" transform={`rotate(-90, 12, ${height / 2})`}>Resíduo</text>
      </svg>
    </div>
  );
}

/**
 * Painel para visualizar e selecionar candidatos do Grid Search.
 * Layout split-view similar ao DatasetSelector.
 */
export default function ModelCandidatesPanel({
  tenant,
  sessionPath,
  stepId,
  onSelect,
  onClose,
  onBack,
}) {
  const { t } = useI18n();
  const listRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [session, setSession] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [sortBy, setSortBy] = useState("rank");
  const [sortDir, setSortDir] = useState("asc");
  const [applying, setApplying] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  // Carregar candidatos
  useEffect(() => {
    if (!sessionPath) return;

    const loadCandidates = async () => {
      setLoading(true);
      setError("");
      try {
        const match = sessionPath.match(/candidates_(\d{8}_\d{6})/);
        const sessionId = match ? match[1] : "";

        if (!sessionId) {
          throw new Error("Session ID não encontrado no path");
        }

        const res = await axios.get(`${API_URL}/training/candidates/${tenant}/${sessionId}`);
        setSession(res.data);
        if (res.data?.best_index !== undefined) {
          setSelectedIndex(res.data.best_index);
        }
      } catch (err) {
        setError(err?.response?.data?.detail || err?.message || t("candidates.error"));
      } finally {
        setLoading(false);
      }
    };

    loadCandidates();
  }, [sessionPath, tenant]);

  // Carregar predições quando um candidato é selecionado
  useEffect(() => {
    if (selectedIndex === null || !sessionPath) return;

    const loadPredictions = async () => {
      setLoadingPredictions(true);
      setPredictions(null);
      try {
        const match = sessionPath.match(/candidates_(\d{8}_\d{6})/);
        const sessionId = match ? match[1] : "";
        if (!sessionId) return;

        const res = await axios.get(
          `${API_URL}/training/candidates/${tenant}/${sessionId}/predictions/${selectedIndex}`
        );
        setPredictions(res.data.predictions);
      } catch (err) {
        console.warn("Predições não disponíveis:", err?.message);
        setPredictions(null);
      } finally {
        setLoadingPredictions(false);
      }
    };

    loadPredictions();
  }, [selectedIndex, sessionPath, tenant]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!session?.candidates?.length) return;

      const candidates = sortedCandidates();
      const currentIdx = candidates.findIndex((c) => {
        const origIdx = session.candidates.findIndex(
          (oc) => oc.rank === c.rank && oc.algorithm === c.algorithm
        );
        return origIdx === selectedIndex;
      });

      if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        const nextIdx = Math.min(currentIdx + 1, candidates.length - 1);
        const next = candidates[nextIdx];
        const origIdx = session.candidates.findIndex(
          (oc) => oc.rank === next.rank && oc.algorithm === next.algorithm
        );
        setSelectedIndex(origIdx);
      } else if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        const prevIdx = Math.max(currentIdx - 1, 0);
        const prev = candidates[prevIdx];
        const origIdx = session.candidates.findIndex(
          (oc) => oc.rank === prev.rank && oc.algorithm === prev.algorithm
        );
        setSelectedIndex(origIdx);
      } else if (e.key === "Enter") {
        e.preventDefault();
        handleApply();
      } else if (e.key === "Escape") {
        e.preventDefault();
        onClose?.();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [session, selectedIndex, sortBy, sortDir]);

  // Ordenar candidatos
  const sortedCandidates = useCallback(() => {
    if (!session?.candidates) return [];
    let list = [...session.candidates];

    // Filtrar por termo de busca
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      list = list.filter((c) => 
        c.algorithm?.toLowerCase().includes(term) ||
        String(c.rank).includes(term)
      );
    }

    list.sort((a, b) => {
      let va, vb;
      if (sortBy === "rank") {
        va = a.rank;
        vb = b.rank;
      } else if (sortBy === "algorithm") {
        va = a.algorithm || "";
        vb = b.algorithm || "";
        return sortDir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
      } else if (sortBy === "score") {
        va = a.score ?? 0;
        vb = b.score ?? 0;
      } else {
        va = a.metrics?.[sortBy] ?? 0;
        vb = b.metrics?.[sortBy] ?? 0;
      }
      if (sortDir === "asc") return va - vb;
      return vb - va;
    });

    return list;
  }, [session, sortBy, sortDir, searchTerm]);

  // Aplicar modelo selecionado
  const handleApply = async () => {
    if (selectedIndex === null || applying) return;

    setApplying(true);
    setError("");

    try {
      const res = await axios.post(`${API_URL}/training/select-candidate`, null, {
        params: {
          tenant,
          session_path: sessionPath,
          candidate_index: selectedIndex,
          step_id: stepId,
          apply_to_pipeline: true,
          change_reason: `Modelo selecionado manualmente: ${session?.candidates?.[selectedIndex]?.algorithm || ""}`,
        },
      });

      if (onSelect) {
        onSelect(res.data);
      }
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || t("candidates.error"));
    } finally {
      setApplying(false);
    }
  };

  // Formatar métrica
  const formatMetric = (value) => {
    if (value === null || value === undefined) return "-";
    if (typeof value === "number") {
      if (Math.abs(value) < 0.001) return value.toExponential(3);
      return value.toFixed(4);
    }
    return String(value);
  };

  // Extrair métricas disponíveis
  const availableMetrics = useCallback(() => {
    if (!session?.candidates?.length) return [];
    const firstMetrics = session.candidates[0]?.metrics || {};
    return Object.keys(firstMetrics).filter((k) => 
      k.startsWith("val_") || k.startsWith("train_") || k === "r2"
    );
  }, [session]);

  // Get algorithm icon
  const getAlgoIcon = (algo) => {
    const icons = {
      xgb: "🌳",
      rf: "🌲",
      svr: "📐",
      mlp: "🧠",
      ridge: "📏",
      lasso: "🎯",
      elastic: "🔗",
      knn: "📍",
    };
    return icons[algo?.toLowerCase()] || "🤖";
  };

  if (loading) {
    return (
      <div className="mcp-container">
        <div className="mcp-loading">
          <div className="mcp-spinner" />
          <span>{t("candidates.loading")}</span>
        </div>
      </div>
    );
  }

  const candidates = sortedCandidates();
  const selectedCandidate = selectedIndex !== null ? session?.candidates?.[selectedIndex] : null;

  return (
    <div className="mcp-container">
      {/* Header */}
      <div className="mcp-header">
        <div className="mcp-header-left">
          <h2>🎯 {t("candidates.title")}</h2>
          <span className="mcp-header-subtitle">{session?.session_id || "-"}</span>
        </div>

        <div className="mcp-header-stats">
          <div className="mcp-stat mcp-stat-samples">
            <span className="mcp-stat-value">{session?.n_samples || 0}</span>
            <span className="mcp-stat-label">Amostras</span>
          </div>
          <div className="mcp-stat mcp-stat-candidates">
            <span className="mcp-stat-value">{session?.candidates?.length || 0}</span>
            <span className="mcp-stat-label">Candidatos</span>
          </div>
          <div className="mcp-stat mcp-stat-best">
            <span className="mcp-stat-value">#{(session?.best_index ?? 0) + 1}</span>
            <span className="mcp-stat-label">Melhor</span>
          </div>
        </div>

        <div className="mcp-shortcuts">
          <kbd>↑↓</kbd> navegar
          <kbd>Enter</kbd> aplicar
          <kbd>Esc</kbd> fechar
        </div>

        <button type="button" className="mcp-close" onClick={onClose}>
          ✕
        </button>
      </div>

      {/* Collection Info */}
      {session?.n_collected !== undefined && session?.n_total_experiments !== undefined && (
        <div className="mcp-collection-info">
          <div className="mcp-collection-stat">
            <span className="mcp-collection-label">Coletados:</span>
            <span className="mcp-collection-value">{session.n_collected} de {session.n_total_experiments}</span>
            {session.n_skipped > 0 && (
              <span className="mcp-collection-skipped">({session.n_skipped} ignorados)</span>
            )}
          </div>
          {session.skipped_reasons?.length > 0 && (
            <details className="mcp-collection-details">
              <summary className="mcp-collection-summary">Ver motivos de descarte</summary>
              <div className="mcp-collection-reasons">
                {session.skipped_reasons.slice(0, 15).map((reason, idx) => (
                  <div key={idx} className="mcp-reason-item">
                    <span className="mcp-reason-text">{reason}</span>
                  </div>
                ))}
                {session.skipped_reasons.length > 15 && (
                  <div className="mcp-reason-more">
                    +{session.skipped_reasons.length - 15} outros motivos...
                  </div>
                )}
              </div>
            </details>
          )}
        </div>
      )}

      {error && (
        <div className="mcp-error">
          <span>⚠️ {error}</span>
          <button onClick={() => setError("")}>✕</button>
        </div>
      )}

      {/* Main split layout */}
      <div className="mcp-main">
        {/* Lista de candidatos (painel esquerdo) */}
        <div className="mcp-list-panel">
          <div className="mcp-filters">
            <input
              type="text"
              className="mcp-search"
              placeholder="Buscar algoritmo..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <select
              className="mcp-sort-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="rank">{t("candidates.table.rank")}</option>
              <option value="algorithm">{t("candidates.table.algorithm")}</option>
              <option value="score">{t("candidates.table.score")}</option>
              {availableMetrics().map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <button
              type="button"
              className="mcp-sort-dir"
              onClick={() => setSortDir((d) => (d === "asc" ? "desc" : "asc"))}
              title={sortDir === "asc" ? "Ascendente" : "Descendente"}
            >
              {sortDir === "asc" ? "↑" : "↓"}
            </button>
          </div>

          <div className="mcp-list" ref={listRef}>
            {candidates.map((c) => {
              const originalIndex = session.candidates.findIndex(
                (oc) => oc.rank === c.rank && oc.algorithm === c.algorithm
              );
              const isSelected = selectedIndex === originalIndex;
              const isBest = originalIndex === session?.best_index;

              return (
                <div
                  key={`${c.rank}-${c.algorithm}`}
                  className={`mcp-item ${isSelected ? "is-active" : ""} ${isBest ? "is-best" : ""}`}
                  onClick={() => setSelectedIndex(originalIndex)}
                >
                  <div className="mcp-item-rank">
                    <span className="mcp-rank-badge">#{c.rank}</span>
                    {isBest && <span className="mcp-best-star" title="Melhor modelo">★</span>}
                  </div>
                  <div className="mcp-item-main">
                    <div className="mcp-item-header">
                      <span className="mcp-item-icon">{getAlgoIcon(c.algorithm)}</span>
                      <span className="mcp-item-algo">{c.algorithm}</span>
                    </div>
                    <div className="mcp-item-metrics">
                      <span className="mcp-item-score">
                        {session?.selection_metric || "score"}: <strong>{formatMetric(c.score)}</strong>
                      </span>
                      {c.metrics?.val_r2 !== undefined && (
                        <span className="mcp-item-r2">R²: {formatMetric(c.metrics.val_r2)}</span>
                      )}
                    </div>
                  </div>
                  <div className="mcp-item-select">
                    <input
                      type="radio"
                      name="candidate"
                      checked={isSelected}
                      onChange={() => setSelectedIndex(originalIndex)}
                    />
                  </div>
                </div>
              );
            })}

            {candidates.length === 0 && (
              <div className="mcp-empty">
                {searchTerm ? "Nenhum candidato encontrado" : "Sem candidatos"}
              </div>
            )}
          </div>
        </div>

        {/* Painel de detalhes (direita) */}
        <div className="mcp-preview-panel">
          {selectedCandidate ? (
            <>
              <div className="mcp-preview-header">
                <div className="mcp-preview-title">
                  <span className="mcp-preview-icon">{getAlgoIcon(selectedCandidate.algorithm)}</span>
                  <h3>{selectedCandidate.algorithm}</h3>
                  <span className="mcp-preview-rank">#{selectedCandidate.rank}</span>
                  {selectedIndex === session?.best_index && (
                    <span className="mcp-preview-best-badge">★ Melhor</span>
                  )}
                </div>
              </div>

              {/* Métricas principais */}
              <div className="mcp-metrics-cards">
                <div className="mcp-metric-card mcp-metric-primary">
                  <span className="mcp-metric-label">{session?.selection_metric || "Score"}</span>
                  <span className="mcp-metric-value">{formatMetric(selectedCandidate.score)}</span>
                </div>
                {selectedCandidate.metrics?.val_r2 !== undefined && (
                  <div className="mcp-metric-card">
                    <span className="mcp-metric-label">R² (val)</span>
                    <span className="mcp-metric-value">{formatMetric(selectedCandidate.metrics.val_r2)}</span>
                  </div>
                )}
                {selectedCandidate.metrics?.val_rmse !== undefined && (
                  <div className="mcp-metric-card">
                    <span className="mcp-metric-label">RMSE (val)</span>
                    <span className="mcp-metric-value">{formatMetric(selectedCandidate.metrics.val_rmse)}</span>
                  </div>
                )}
                {selectedCandidate.metrics?.val_mae !== undefined && (
                  <div className="mcp-metric-card">
                    <span className="mcp-metric-label">MAE (val)</span>
                    <span className="mcp-metric-value">{formatMetric(selectedCandidate.metrics.val_mae)}</span>
                  </div>
                )}
              </div>

              {/* Gráficos */}
              <div className="mcp-charts-section">
                <h4>📊 Desempenho</h4>
                {loadingPredictions ? (
                  <div className="mcp-charts-loading">
                    <div className="mcp-spinner-small" />
                    Carregando gráficos...
                  </div>
                ) : predictions ? (
                  <>
                    <div className="mcp-charts-grid">
                      <ScatterPlot
                        title="Predito vs Real"
                        xLabel="Real"
                        yLabel="Predito"
                        showIdealLine={true}
                        data={[
                          ...(predictions.train_actual || []).map((v, i) => ({
                            x: v,
                            y: predictions.train_predicted?.[i] ?? v,
                            isVal: false,
                          })),
                          ...(predictions.val_actual || []).map((v, i) => ({
                            x: v,
                            y: predictions.val_predicted?.[i] ?? v,
                            isVal: true,
                          })),
                        ]}
                      />
                      <ResidualsPlot
                        title="Resíduos"
                        data={[
                          ...(predictions.train_predicted || []).map((v, i) => ({
                            x: v,
                            y: predictions.train_residuals?.[i] ?? 0,
                            isVal: false,
                          })),
                          ...(predictions.val_predicted || []).map((v, i) => ({
                            x: v,
                            y: predictions.val_residuals?.[i] ?? 0,
                            isVal: true,
                          })),
                        ]}
                      />
                    </div>
                    <div className="mcp-charts-legend">
                      <span className="mcp-legend-item mcp-legend-train">● Treino ({(predictions.train_actual || []).length})</span>
                      <span
                        className="mcp-legend-item mcp-legend-val"
                        style={{ opacity: (predictions.val_actual || []).length ? 1 : 0.4 }}
                      >
                        ● Validação ({(predictions.val_actual || []).length})
                      </span>
                    </div>
                    {(!(predictions.val_actual || []).length) && (
                      <div className="mcp-charts-note">
                        Sem amostras de validação para este treino (dados insuficientes ou test_size=0).
                      </div>
                    )}
                  </>
                ) : (
                  <div className="mcp-charts-empty">
                    Dados de predição não disponíveis para este candidato.
                  </div>
                )}
              </div>

              {/* Todas as métricas */}
              <div className="mcp-all-metrics">
                <h4>📈 Todas as Métricas</h4>
                <div className="mcp-metrics-grid">
                  {Object.entries(selectedCandidate.metrics || {}).map(([k, v]) => (
                    <div key={k} className="mcp-metric-item">
                      <span className="mcp-metric-key">{k}</span>
                      <span className="mcp-metric-val">{formatMetric(v)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Parâmetros */}
              <div className="mcp-params-section">
                <h4>⚙️ Parâmetros</h4>
                <pre className="mcp-params-code">
                  {JSON.stringify(selectedCandidate.params || {}, null, 2)}
                </pre>
              </div>
            </>
          ) : (
            <div className="mcp-preview-empty">
              <span className="mcp-preview-empty-icon">👈</span>
              <p>Selecione um candidato para ver os detalhes</p>
            </div>
          )}
        </div>
      </div>

      {/* Footer com ações */}
      <div className="mcp-footer">
        {onBack && (
          <button type="button" className="mcp-btn-back" onClick={onBack}>
            ← {t("candidates.actions.back") || "Voltar"}
          </button>
        )}
        <div className="mcp-footer-spacer" />
        <button type="button" className="mcp-btn-cancel" onClick={onClose}>
          {t("candidates.actions.close")}
        </button>
        <button
          type="button"
          className="mcp-btn-apply"
          disabled={selectedIndex === null || applying}
          onClick={handleApply}
        >
          {applying ? t("candidates.actions.applying") : `✓ ${t("candidates.actions.apply")}`}
        </button>
      </div>
    </div>
  );
}
