import React, { useState, useMemo } from 'react';
import Chevron from "./Chevron";
import { useI18n } from "../locale/i18n";

const StepResultCard = ({ stepId, stepLabel, stepData, onGraphClick, defaultExpanded = false, hideHeader = false }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [showFullOutput, setShowFullOutput] = useState(false);
  const { t } = useI18n();
  
  const outputGraphs = stepData?.output_graphs || stepData?.data?.output_graphs || {};
  const hasGraphs = Object.keys(outputGraphs).length > 0;

  // Coletar todas as chaves disponíveis
  const availableKeys = useMemo(() => {
    if (Array.isArray(stepData)) return [];
    const topKeys = Object.keys(stepData || {});
    const dataKeys = Object.keys(stepData?.data || {});
    return [...new Set([...topKeys, ...dataKeys])];
  }, [stepData]);

  // Coletar TODAS as chaves que terminam com _json (genérico)
  const jsonOutputs = useMemo(() => {
    const results = [];
    const seen = new Set();

    // Helper para coletar JSONs de um objeto (até 2 níveis)
    const collectJsons = (obj, prefix = "", depth = 0) => {
      if (!obj || typeof obj !== "object") return;

      Object.entries(obj).forEach(([key, value]) => {
        const pathKey = prefix ? `${prefix}.${key}` : key;

        if (String(key).endsWith("_json") && value != null) {
          if (seen.has(pathKey)) return;
          seen.add(pathKey);

          const displayKey = pathKey.replace(/_json$/, "");
          const displayName = displayKey.replace(/_/g, " ");
          results.push({ key: pathKey, displayName, value });
          return;
        }

        // Alguns blocos guardam JSONs aninhados (ex.: dentro de `data`/`sensor_data`)
        if (depth < 1 && value && typeof value === "object" && !Array.isArray(value)) {
          collectJsons(value, pathKey, depth + 1);
        }
      });
    };

    // Verificar tanto no nível raiz quanto em .data
    collectJsons(stepData);
    collectJsons(stepData?.data, "data");

    return results;
  }, [stepData]);

  const hasJsonOutputs = jsonOutputs.length > 0;

  return (
    <div className="step-item">
      {!hideHeader && (
        <div className="step-item-header" onClick={() => setIsExpanded(!isExpanded)}>
          <div className="step-item-title">
            <span className="step-item-toggle" aria-hidden="true">
              <Chevron expanded={isExpanded} size={18} />
            </span>
            <span className="step-item-name">{t("results.step", { label: stepLabel || stepId })}</span>
          </div>
          <div className="step-item-badges">
            {hasGraphs && (
              <span className="results-badge results-badge--info">
                {t("results.graphs")}: {Object.keys(outputGraphs).length}
              </span>
            )}
            {hasJsonOutputs && (
              <span className="results-badge results-badge--debug">
                JSON: {jsonOutputs.length}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Conteúdo expandido */}
      {(hideHeader || isExpanded) && (
        <div className="step-item-content">
          {/* Saídas de debug */}
          <div className="step-section">
            <div className="step-section-title">{t("results.outputs")}</div>
            {availableKeys.length > 0 && (
              <div className="step-available-keys">
                <small>{t("results.availableKeys", { keys: availableKeys.join(", ") })}</small>
              </div>
            )}

            {/* Renderizar todos os JSONs encontrados */}
            {jsonOutputs.map(({ key, displayName, value }) => (
              <div key={key} className="step-subsection">
                <div className="step-subsection-title">
                  {displayName} {t("results.jsonSuffix")}
                </div>
                <pre className="step-code">{JSON.stringify(value, null, 2)}</pre>
              </div>
            ))}

            {/* Fallback quando não houver campos de debug */}
            {!hasJsonOutputs && (
              <div className="step-output-summary">
                <small>{t("results.enableDebugHint")}</small>
                {availableKeys.length > 0 && (
                  <div style={{marginTop:8}}>
                    <small>{t("results.availableKeys", { keys: availableKeys.join(", ") })}</small>
                  </div>
                )}
                <div style={{ marginTop: 10 }}>
                  <button
                    type="button"
                    className="results-show-full"
                    onClick={() => setShowFullOutput((prev) => !prev)}
                  >
                    {showFullOutput ? t("results.hideFullOutput") : t("results.showFullOutput")}
                  </button>
                  {showFullOutput && (
                    <pre className="step-code" style={{ marginTop: 10 }}>
                      {JSON.stringify(stepData, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Gráficos */}
          {hasGraphs && (
            <div className="step-section">
              <div className="step-section-title">{t("results.graphs")}</div>
              <div className="step-graphs">
                {Object.entries(outputGraphs).map(([sensorName, graphUri], index) => (
                  <div key={`${sensorName}-${index}`} className="step-graph-item">
                    <div className="step-graph-label">{sensorName}</div>
                    <img
                      src={graphUri}
                      alt={`${stepId} - ${sensorName}`}
                      onClick={() => onGraphClick(graphUri, `${stepId} - ${sensorName}`)}
                      className="step-graph-img"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default StepResultCard;
