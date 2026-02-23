/**
 * TrainingStudio - ML Training Interface
 * 
 * Layout:
 * - Painel esquerdo: Datasets salvos + Modelos
 * - Painel direito: Configuração de modelos OU Resultados
 * - Overlay fullscreen: DatasetSelector para criar/editar datasets
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import axios from "axios";
import "./TrainingStudio.css";
import DatasetSelector from "./DatasetSelector";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

// ============================================================================
// Constantes
// ============================================================================

const ALGORITHM_ITEMS = [
  { key: "ridge", label: "Ridge", color: "#6366f1" },
  { key: "rf", label: "Random Forest", color: "#22c55e" },
  { key: "gbm", label: "Gradient Boosting", color: "#f59e0b" },
  { key: "xgb", label: "XGBoost", color: "#ef4444" },
  { key: "lgbm", label: "LightGBM", color: "#10b981" },
  { key: "catboost", label: "CatBoost", color: "#eab308" },
  { key: "svr", label: "SVR", color: "#8b5cf6" },
  { key: "mlp", label: "MLP", color: "#ec4899" },
  { key: "cnn", label: "CNN", color: "#0ea5e9" },
  { key: "lstm", label: "LSTM", color: "#f97316" },
];

// Regressões Matemáticas
const REGRESSION_ITEMS = [
  { key: "linear", label: "Linear", equation: "y = ax + b", color: "#06b6d4" },
  { key: "quadratic", label: "Quadrática", equation: "y = ax² + bx + c", color: "#8b5cf6" },
  { key: "exponential", label: "Exponencial", equation: "y = a·eᵇˣ + c", color: "#f43f5e" },
  { key: "logarithmic", label: "Logarítmica", equation: "y = a·ln(x) + b", color: "#22c55e" },
  { key: "power", label: "Potência", equation: "y = a·xᵇ + c", color: "#f59e0b" },
  { key: "polynomial", label: "Polinomial", equation: "y = aₙxⁿ + ... + a₀", color: "#ec4899" },
];

// Métodos de remoção de outliers
const OUTLIER_METHODS = [
  { key: "none", label: "Nenhuma", desc: "Usar todos os dados sem filtrar outliers" },
  { key: "ransac", label: "RANSAC", desc: "Remove outliers usando consensus aleatório. Melhor para dados com outliers extremos" },
  { key: "iqr", label: "IQR", desc: "Remove pontos fora de 1.5×IQR. Bom para distribuições normais" },
  { key: "zscore", label: "Z-Score", desc: "Remove pontos com Z > 2.5. Simples e efetivo" },
];

// Métodos robustos para regressão
const ROBUST_METHODS = [
  { key: "ols", label: "OLS (Padrão)", desc: "Mínimos quadrados ordinários. Sensível a outliers" },
  { key: "theil_sen", label: "Theil-Sen", desc: "Mediana das inclinações. Muito robusto contra outliers (só linear)" },
  { key: "huber", label: "Huber", desc: "IRLS com pesos de Huber. Balanceia eficiência e robustez" },
  { key: "bisquare", label: "Bisquare (Tukey)", desc: "Outliers extremos recebem peso ZERO. Muito robusto" },
  { key: "welsch", label: "Welsch", desc: "Decaimento exponencial dos pesos. Suave mas eficaz" },
  { key: "cauchy", label: "Cauchy", desc: "Para dados com caudas pesadas (heavy-tailed)" },
  { key: "lad", label: "LAD (L1)", desc: "Minimiza erro absoluto (não quadrático). Mediana condicional (só linear)" },
  { key: "mm", label: "MM-Estimator", desc: "Alta robustez (~50% breakdown) + alta eficiência (~95%) (só linear)" },
  { key: "lts", label: "LTS", desc: "Least Trimmed Squares. Extremamente robusto, ignora 25% piores pontos" },
  { key: "ransac_fit", label: "RANSAC", desc: "Ajuste por consenso aleatório. Extremamente robusto (só linear)" },
];

// ============================================================================
// History Card Component
// ============================================================================


function HistoryCard({ entry, onClick, onDelete }) {
  const isRegression = entry.mode === "regression";
  const timestamp = entry.timestamp ? new Date(entry.timestamp) : null;
  const dateStr = timestamp ? timestamp.toLocaleDateString("pt-BR") : "";
  const timeStr = timestamp ? timestamp.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" }) : "";
  const label = entry.label || entry.stepLabel || "";
  const blockName = entry.block_name || "";
  const unit = entry.output_unit || "";
  const feature = entry.input_feature || "";
  const channel = entry.channel || "";

  const r2 = isRegression 
    ? entry.metrics?.r_squared 
    : entry.metrics?.r2_score;

  return (
    <div className={`ts-history-card ${entry.applied ? "is-applied" : ""} ${entry.is_active ? "is-active" : ""}`} onClick={onClick}>
      <div className="ts-history-card-header">
        <span className="ts-history-card-type">
          {isRegression ? "Regressao" : "ML"}
        </span>
        {entry.is_active && <span className="ts-history-card-badge is-active">Ativo</span>}
        {!entry.is_active && entry.applied && <span className="ts-history-card-badge">Aplicado</span>}
      </div>
      <div className="ts-history-card-body">
        {isRegression ? (
          <>
            <span className="ts-history-card-title">
              {entry.metrics?.regression_type || "Linear"}
            </span>
            <span className="ts-history-card-equation">
              {entry.metrics?.equation || "y = ax + b"}
            </span>
          </>
        ) : (
          <span className="ts-history-card-title">
            {entry.metrics?.best_model || "Modelo ML"}
          </span>
        )}
        {(label || unit || feature || channel || blockName) && (
          <div className="ts-history-card-tags">
            {label && <span className="ts-history-card-tag">{label}</span>}
            {blockName && <span className="ts-history-card-tag">{blockName}</span>}
            {unit && <span className="ts-history-card-tag">{unit}</span>}
            {feature && <span className="ts-history-card-tag">{feature}</span>}
            {channel && <span className="ts-history-card-tag">{channel}</span>}
          </div>
        )}
        {r2 !== undefined && r2 !== null && (
          <span className="ts-history-card-metric">
            R2 {Number(r2).toFixed(4)}
          </span>
        )}
      </div>
      <div className="ts-history-card-footer">
        <span className="ts-history-card-date">{dateStr}</span>
        <span className="ts-history-card-time">{timeStr}</span>
      </div>
      {onDelete && (
        <button 
          className="ts-history-card-delete" 
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          title="Excluir"
        >
          x
        </button>
      )}
    </div>
  );
}

function NewTrainingCard({ onClick }) {
  return (
    <div className="ts-history-card ts-history-card-new" onClick={onClick}>
      <div className="ts-history-card-plus">
        <svg viewBox="0 0 24 24" width="48" height="48">
          <path fill="currentColor" d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
        </svg>
      </div>
      <span className="ts-history-card-label">Novo Treinamento</span>
    </div>
  );
}

const ALGO_PARAM_SCHEMA = {
  ridge: {
    fields: [
      { key: "alpha", label: "alpha", kind: "float", step: 0.1, min: 0, defaultValue: 1.0, grid: true, gridHint: { min: 0.1, max: 10.0, divisions: 3 } },
    ],
  },
  rf: {
    fields: [
      { key: "n_estimators", label: "n_estimators", kind: "int", step: 50, min: 10, defaultValue: 300, grid: true, gridHint: { min: 200, max: 600, divisions: 3 } },
      { key: "max_depth", label: "max_depth", kind: "int", step: 1, min: 1, allowNull: true, defaultValue: null, grid: true, gridHint: { min: 4, max: 16, divisions: 4 } },
      { key: "min_samples_split", label: "min_samples_split", kind: "int", step: 1, min: 2, defaultValue: 2, grid: true, gridHint: { min: 2, max: 10, divisions: 5 } },
      { key: "min_samples_leaf", label: "min_samples_leaf", kind: "int", step: 1, min: 1, defaultValue: 1, grid: true, gridHint: { min: 1, max: 6, divisions: 6 } },
      { key: "max_features", label: "max_features", kind: "float", step: 0.05, min: 0.05, defaultValue: 1.0, grid: true, gridHint: { min: 0.3, max: 1.0, divisions: 4 } },
      { key: "bootstrap", label: "bootstrap", kind: "select", options: ["true", "false"], defaultValue: "true", grid: true },
    ],
  },
  xgb: {
    fields: [
      { key: "n_estimators", label: "n_estimators", kind: "int", step: 50, min: 10, defaultValue: 300, grid: true, gridHint: { min: 100, max: 600, divisions: 6 } },
      { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.01, min: 0.0001, defaultValue: 0.05, grid: true, gridHint: { min: 0.01, max: 0.2, divisions: 5 } },
      { key: "max_depth", label: "max_depth", kind: "int", step: 1, min: 1, defaultValue: 4, grid: true, gridHint: { min: 2, max: 10, divisions: 5 } },
      { key: "subsample", label: "subsample", kind: "float", step: 0.05, min: 0.05, defaultValue: 0.9, grid: true, gridHint: { min: 0.5, max: 1.0, divisions: 6 } },
      { key: "colsample_bytree", label: "colsample_bytree", kind: "float", step: 0.05, min: 0.05, defaultValue: 0.8, grid: true, gridHint: { min: 0.5, max: 1.0, divisions: 6 } },
      { key: "gamma", label: "gamma", kind: "float", step: 0.1, min: 0.0, defaultValue: 0.0, grid: true, gridHint: { min: 0.0, max: 2.0, divisions: 5 } },
      { key: "reg_alpha", label: "reg_alpha", kind: "float", step: 0.05, min: 0.0, defaultValue: 0.0, grid: true, gridHint: { min: 0.0, max: 1.0, divisions: 6 } },
      { key: "reg_lambda", label: "reg_lambda", kind: "float", step: 0.05, min: 0.0, defaultValue: 1.0, grid: true, gridHint: { min: 0.5, max: 2.0, divisions: 4 } },
    ],
  },
  lgbm: {
    fields: [
      { key: "n_estimators", label: "n_estimators", kind: "int", step: 50, min: 10, defaultValue: 400, grid: true, gridHint: { min: 200, max: 600, divisions: 3 } },
      { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.01, min: 0.0001, defaultValue: 0.03, grid: true, gridHint: { min: 0.01, max: 0.1, divisions: 3 } },
      { key: "max_depth", label: "max_depth", kind: "int", step: 1, min: -1, defaultValue: -1, grid: true, gridHint: { min: -1, max: 12, divisions: 4 } },
      { key: "subsample", label: "subsample", kind: "float", step: 0.05, min: 0.1, defaultValue: 1.0, grid: true, gridHint: { min: 0.6, max: 1.0, divisions: 5 } },
      { key: "colsample_bytree", label: "colsample_bytree", kind: "float", step: 0.05, min: 0.1, defaultValue: 1.0, grid: true, gridHint: { min: 0.6, max: 1.0, divisions: 5 } },
    ],
  },
  catboost: {
    fields: [
      { key: "n_estimators", label: "n_estimators", kind: "int", step: 50, min: 50, defaultValue: 400, grid: true, gridHint: { min: 200, max: 600, divisions: 3 } },
      { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.01, min: 0.0001, defaultValue: 0.1, grid: true, gridHint: { min: 0.03, max: 0.2, divisions: 4 } },
      { key: "max_depth", label: "max_depth", kind: "int", step: 1, min: 2, defaultValue: 6, grid: true, gridHint: { min: 4, max: 10, divisions: 4 } },
      { key: "l2_leaf_reg", label: "l2_leaf_reg", kind: "float", step: 0.5, min: 0, defaultValue: 3.0, grid: true, gridHint: { min: 1.0, max: 6.0, divisions: 3 } },
    ],
  },
  gbm: {
    fields: [
      { key: "n_estimators", label: "n_estimators", kind: "int", step: 50, min: 10, defaultValue: 400, grid: true, gridHint: { min: 200, max: 600, divisions: 3 } },
      { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.01, min: 0.0001, defaultValue: 0.03, grid: true, gridHint: { min: 0.01, max: 0.1, divisions: 3 } },
      { key: "max_depth", label: "max_depth", kind: "int", step: 1, min: 1, defaultValue: 3, grid: true, gridHint: { min: 2, max: 5, divisions: 4 } },
    ],
  },
  svr: {
    fields: [
      { key: "kernel", label: "kernel", kind: "select", options: ["rbf", "linear", "poly", "sigmoid"], defaultValue: "rbf", grid: true },
      { key: "C", label: "C", kind: "float", step: 0.5, min: 0.0, defaultValue: 1.0, grid: true, gridHint: { min: 0.5, max: 4.0, divisions: 4 } },
      { key: "epsilon", label: "epsilon", kind: "float", step: 0.05, min: 0.0, defaultValue: 0.1, grid: true, gridHint: { min: 0.05, max: 0.3, divisions: 4 } },
      { key: "gamma", label: "gamma", kind: "float", step: 0.05, min: 0.0, defaultValue: 0.1, grid: true, gridHint: { min: 0.01, max: 1.0, divisions: 6 } },
    ],
  },
  mlp: {
    fields: [
      { key: "hidden_layer_sizes", label: "hidden_layer_sizes", kind: "text", placeholder: "Optional, e.g. 128,64", defaultValue: "", grid: false },
      { key: "layers", label: "layers", kind: "int", step: 1, min: 1, defaultValue: 2, grid: true, gridHint: { min: 1, max: 5, divisions: 5 } },
      { key: "hidden", label: "hidden", kind: "int", step: 16, min: 16, defaultValue: 128, grid: true, gridHint: { min: 16, max: 512, divisions: 7 } },
      { key: "dropout", label: "dropout", kind: "float", step: 0.05, min: 0.0, defaultValue: 0.1, grid: true, gridHint: { min: 0.0, max: 0.5, divisions: 6 } },
      { key: "activation", label: "activation", kind: "select", options: ["relu", "tanh", "sigmoid"], defaultValue: "relu", grid: true },
      { key: "optimizer", label: "optimizer", kind: "select", options: ["adam", "sgd", "rmsprop"], defaultValue: "adam", grid: true },
      { key: "epochs", label: "epochs", kind: "int", step: 10, min: 1, defaultValue: 200, grid: true, gridHint: { min: 100, max: 500, divisions: 5 } },
      { key: "batch_size", label: "batch_size", kind: "int", step: 8, min: 1, defaultValue: 64, grid: true, gridHint: { min: 16, max: 128, divisions: 4 } },
      { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.0005, min: 0.00001, defaultValue: 0.001, grid: true, gridHint: { min: 0.0005, max: 0.005, divisions: 4 } },
    ],
  },
  cnn: {
    fields: [
      { key: "epochs", label: "epochs", kind: "int", step: 10, min: 1, defaultValue: 200, grid: true, gridHint: { min: 100, max: 500, divisions: 5 } },
      { key: "batch_size", label: "batch_size", kind: "int", step: 8, min: 1, defaultValue: 64, grid: true, gridHint: { min: 16, max: 128, divisions: 4 } },
      { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.0005, min: 0.00001, defaultValue: 0.001, grid: true, gridHint: { min: 0.0005, max: 0.005, divisions: 4 } },
    ],
  },
  lstm: {
    fields: [
      { key: "hidden_size", label: "hidden_size", kind: "int", step: 8, min: 4, defaultValue: 64, grid: true, gridHint: { min: 32, max: 128, divisions: 4 } },
      { key: "num_layers", label: "num_layers", kind: "int", step: 1, min: 1, defaultValue: 1, grid: true, gridHint: { min: 1, max: 3, divisions: 3 } },
      { key: "dropout", label: "dropout", kind: "float", step: 0.05, min: 0.0, defaultValue: 0.0, grid: true, gridHint: { min: 0.0, max: 0.5, divisions: 6 } },
      { key: "epochs", label: "epochs", kind: "int", step: 10, min: 1, defaultValue: 200, grid: true, gridHint: { min: 100, max: 500, divisions: 5 } },
      { key: "batch_size", label: "batch_size", kind: "int", step: 8, min: 1, defaultValue: 64, grid: true, gridHint: { min: 16, max: 128, divisions: 4 } },
      { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.0005, min: 0.00001, defaultValue: 0.001, grid: true, gridHint: { min: 0.0005, max: 0.005, divisions: 4 } },
    ],
  },
};

// ============================================================================
// Helpers
// ============================================================================

const buildDefaultParams = (algorithmKey) => {
  const algo = String(algorithmKey || "ridge").trim().toLowerCase() || "ridge";
  const schema = ALGO_PARAM_SCHEMA[algo] || ALGO_PARAM_SCHEMA.ridge;
  const out = {};
  (schema.fields || []).forEach((field) => {
    const hint = field.gridHint || {};
    const allowNull = !!field.allowNull;
    const defaultIsNull = allowNull && field.defaultValue === null;
    out[field.key] = {
      mode: "fixed",
      value: defaultIsNull ? "" : field.defaultValue ?? "",
      isNull: defaultIsNull,
      min: hint.min ?? "",
      max: hint.max ?? "",
      divisions: hint.divisions ?? 3,
      choices: [],
    };
  });
  return out;
};

const algoInfo = (key) => ALGORITHM_ITEMS.find((a) => a.key === key) || ALGORITHM_ITEMS[0];
const algoGpuTag = (key) => {
  switch (String(key || "").toLowerCase()) {
    case "xgb":
    case "catboost":
    case "mlp":
    case "cnn":
    case "lstm":
      return "gpu";
    case "lgbm":
      return "maybe";
    default:
      return "cpu";
  }
};
const algoGpuBadge = (key) => (algoGpuTag(key) === "gpu" ? "G" : algoGpuTag(key) === "maybe" ? "G*" : "C");
const algoGpuLabel = (key) => (algoGpuTag(key) === "gpu" ? "GPU" : algoGpuTag(key) === "maybe" ? "GPU (build)" : "CPU");
const formatErrorMessage = (value) => {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value.map(formatErrorMessage).filter(Boolean).join(" | ");
  }
  if (typeof value === "object") {
    if (value.msg) return String(value.msg);
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
};

// ============================================================================
// SVG Icons
// ============================================================================

const IconChevron = ({ open }) => (
  <svg className={`ts-icon-chevron ${open ? "is-open" : ""}`} viewBox="0 0 24 24" width="16" height="16">
    <path fill="currentColor" d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/>
  </svg>
);

const IconClose = () => (
  <svg viewBox="0 0 24 24" width="20" height="20">
    <path fill="currentColor" d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
  </svg>
);

const IconDatabase = () => (
  <svg viewBox="0 0 24 24" width="16" height="16">
    <path fill="currentColor" d="M12 3C7.58 3 4 4.79 4 7v10c0 2.21 3.59 4 8 4s8-1.79 8-4V7c0-2.21-3.58-4-8-4zm0 2c3.87 0 6 1.5 6 2s-2.13 2-6 2-6-1.5-6-2 2.13-2 6-2zm6 12c0 .5-2.13 2-6 2s-6-1.5-6-2v-2.23c1.61.78 3.72 1.23 6 1.23s4.39-.45 6-1.23V17zm0-5c0 .5-2.13 2-6 2s-6-1.5-6-2V9.77C7.61 10.55 9.72 11 12 11s4.39-.45 6-1.23V12z"/>
  </svg>
);

const IconModel = () => (
  <svg viewBox="0 0 24 24" width="16" height="16">
    <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
  </svg>
);

const IconSettings = () => (
  <svg viewBox="0 0 24 24" width="16" height="16">
    <path fill="currentColor" d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
  </svg>
);

const IconAdd = () => (
  <svg viewBox="0 0 24 24" width="16" height="16">
    <path fill="currentColor" d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
  </svg>
);

const IconEdit = () => (
  <svg viewBox="0 0 24 24" width="14" height="14">
    <path fill="currentColor" d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
  </svg>
);

// ============================================================================
// Sub-componentes
// ============================================================================

function Section({ title, badge, icon, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen);
  
  return (
    <div className={`ts-section ${open ? "is-open" : ""}`}>
      <button type="button" className="ts-section-header" onClick={() => setOpen(!open)}>
        {icon && <span className="ts-section-icon">{icon}</span>}
        <span className="ts-section-title">{title}</span>
        {badge !== undefined && <span className="ts-section-badge">{badge}</span>}
        <IconChevron open={open} />
      </button>
      {open && <div className="ts-section-content">{children}</div>}
    </div>
  );
}

function ModelCard({ model, isSelected, onClick, onToggle, disabled }) {
  const algorithms = model.algorithms || ["ridge"];
  const isRegression = model.modelType === "regression";
  const regressionLabels = {
    linear: "Linear",
    quadratic: "Quadrática", 
    exponential: "Exponencial",
    logarithmic: "Logarítmica",
    power: "Potência",
    polynomial: "Polinomial"
  };
  
  return (
    <div 
      className={`ts-model-card ${isSelected ? "is-selected" : ""} ${!model.enabled ? "is-disabled" : ""}`}
      onClick={() => onClick?.(model.stepId)}
    >
      <label className="ts-model-toggle" onClick={(e) => e.stopPropagation()}>
        <input
          type="checkbox"
          checked={model.enabled !== false}
          disabled={disabled}
          onChange={(e) => onToggle?.(model.stepId, e.target.checked)}
        />
        <span className="ts-toggle-track" />
      </label>
      
      <div className="ts-model-info">
        <span className="ts-model-name">{model.label || model.stepId}</span>
        <span className="ts-model-meta">
          {model.bacteria && <span className="ts-model-bacteria" title="Bacteria">{model.bacteria}</span>}
          {model.outputUnit && <span className="ts-model-unit" title="Unidade de saida">{model.outputUnit}</span>}
          {model.inputFeature && <span className="ts-model-feature" title="Feature de entrada">{model.inputFeature}</span>}
          {model.channel && <span className="ts-model-channel" title="Canal">{model.channel}</span>}
          {!model.bacteria && !model.outputUnit && !model.inputFeature && <span className="ts-model-block">{model.blockName}</span>}
        </span>
      </div>
      
      <div className="ts-model-algos">
        {isRegression ? (
          // Mostrar tipo de regressão
          <span
            className="ts-model-algo ts-regression-badge"
            style={{ background: "#9c27b0" }}
            title={`Regressao ${regressionLabels[model.regressionType] || "Linear"}${model.regressionAutoSelect ? " (Auto)" : ""}`}
          >
            {model.regressionAutoSelect ? "AUTO" : (model.regressionType || "linear").substring(0, 3).toUpperCase()}
          </span>
        ) : (
          // Mostrar algoritmos ML
          <>
            {algorithms.slice(0, 3).map((algo) => (
              <span
                key={algo}
                className="ts-model-algo"
                style={{ background: algoInfo(algo).color }}
                title={`${algoInfo(algo).label} - ${algoGpuLabel(algo)}`}
              >
                {algo.substring(0, 2).toUpperCase()}
                <span className={`ts-model-algo-gpu ts-gpu-${algoGpuTag(algo)}`}>{algoGpuBadge(algo)}</span>
              </span>
            ))}
            {algorithms.length > 3 && <span className="ts-model-algo-more">+{algorithms.length - 3}</span>}
          </>
        )}
      </div>
      
      <span className={`ts-model-status ts-status-${model.status || "pending"}`} />
    </div>
  );
}

/**
 * Lista de datasets salvos com opção de editar
 */
function DatasetList({ tenant, onSelect, onEdit, onCreate, selectedIds, disabled }) {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadDatasets = useCallback(async () => {
    if (!tenant) {
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const res = await axios.get(`${API_URL}/datasets/${tenant}`);
      const data = res.data;
      if (Array.isArray(data)) {
        setDatasets(data);
      } else if (data && Array.isArray(data.datasets)) {
        setDatasets(data.datasets);
      } else {
        setDatasets([]);
      }
    } catch {
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }, [tenant]);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Ao selecionar, busca detalhes completos
  const handleSelect = async (ds) => {
    try {
      const res = await axios.get(`${API_URL}/datasets/${tenant}/${ds.id}`);
      onSelect(res.data);
    } catch {
      onSelect(ds);
    }
  };

  // Expor reload para o parent
  useEffect(() => {
    DatasetList.reload = loadDatasets;
  }, [loadDatasets]);

  if (loading) {
    return <div className="ts-dataset-loading">Carregando...</div>;
  }

  return (
    <div className="ts-dataset-list">
      {(!Array.isArray(datasets) || datasets.length === 0) ? (
        <div className="ts-dataset-empty">
          <p>Nenhum dataset</p>
        </div>
      ) : (
        datasets.map((ds) => (
          <div
            key={ds.id}
            className={`ts-dataset-item ${(selectedIds || []).includes(ds.id) ? "is-selected" : ""}`}
          >
            <button
              type="button"
              className="ts-dataset-item-main"
              disabled={disabled}
              onClick={() => handleSelect(ds)}
            >
              <input
                type="checkbox"
                checked={(selectedIds || []).includes(ds.id)}
                readOnly
                aria-label={`Selecionar dataset ${ds.name}`}
              />
              <div className="ts-dataset-item-info">
                <span className="ts-dataset-item-name">{ds.name}</span>
                <span className="ts-dataset-item-meta">
                  {ds.experiment_count || 0} exp · {ds.viewed_count || 0} vistos
                </span>
              </div>
              <span className="ts-dataset-item-count">{ds.experiment_count || 0}</span>
            </button>
            <button
              type="button"
              className="ts-dataset-item-edit"
              disabled={disabled}
              onClick={() => onEdit(ds)}
              title="Editar dataset"
            >
              <IconEdit />
            </button>
          </div>
        ))
      )}
      <button type="button" className="ts-dataset-create" onClick={onCreate} disabled={disabled}>
        <IconAdd />
        <span>Criar Dataset</span>
      </button>
    </div>
  );
}

/**
 * Configuração de algoritmo
 */
function AlgorithmConfig({ algorithm, params, onParamChange, disabled }) {
  const info = algoInfo(algorithm);
  const schema = ALGO_PARAM_SCHEMA[algorithm] || ALGO_PARAM_SCHEMA.ridge;
  
  return (
    <div className="ts-algo-config">
      <div className="ts-algo-header" style={{ borderLeftColor: info.color }}>
        <span className="ts-algo-name">{info.label}</span>
      </div>
      
      <div className="ts-algo-params">
        {schema.fields.map((field) => {
          const row = params?.[field.key] || {};
          const mode = row?.mode === "grid" && field.grid ? "grid" : "fixed";
          const isNull = !!row?.isNull && !!field.allowNull;
          
          return (
            <div key={field.key} className="ts-param">
              <span className="ts-param-name">{field.label}</span>
              
              {field.grid && (
                <div className="ts-param-mode">
                  <button
                    type="button"
                    className={mode === "fixed" ? "is-active" : ""}
                    disabled={disabled}
                    onClick={() => onParamChange(field.key, { mode: "fixed" })}
                  >
                    Fixo
                  </button>
                  <button
                    type="button"
                    className={mode === "grid" ? "is-active" : ""}
                    disabled={disabled}
                    onClick={() => onParamChange(field.key, { mode: "grid" })}
                  >
                    Grid
                  </button>
                </div>
              )}
              
              <div className="ts-param-input">
                {mode === "fixed" ? (
                  <>
                    {field.kind === "select" ? (
                      <select
                        value={row?.value ?? field.defaultValue}
                        disabled={disabled || isNull}
                        onChange={(e) => onParamChange(field.key, { value: e.target.value })}
                      >
                        {(field.options || []).map((opt) => (
                          <option key={opt} value={opt}>{opt}</option>
                        ))}
                      </select>
                    ) : field.kind === "text" ? (
                      <input
                        type="text"
                        value={row?.value ?? field.defaultValue ?? ""}
                        placeholder={field.placeholder}
                        disabled={disabled || isNull}
                        onChange={(e) => onParamChange(field.key, { value: e.target.value })}
                      />
                    ) : (
                      <input
                        type="number"
                        value={row?.value ?? field.defaultValue ?? ""}
                        step={field.step || 1}
                        min={field.min}
                        disabled={disabled || isNull}
                        onChange={(e) => onParamChange(field.key, { value: e.target.value })}
                      />
                    )}
                    {field.allowNull && (
                      <label className="ts-param-null">
                        <input
                          type="checkbox"
                          checked={isNull}
                          disabled={disabled}
                          onChange={(e) => onParamChange(field.key, { isNull: e.target.checked })}
                        />
                        null
                      </label>
                    )}
                  </>
                ) : (
                  <div className="ts-param-grid">
                    {field.kind === "select" ? (
                      <div className="ts-param-chips">
                        {(field.options || []).map((opt) => {
                          const isSelected = (row?.choices || []).includes(opt);
                          return (
                            <button
                              key={opt}
                              type="button"
                              className={isSelected ? "is-selected" : ""}
                              disabled={disabled}
                              onClick={() => {
                                const choices = row?.choices || [];
                                const next = isSelected ? choices.filter((c) => c !== opt) : [...choices, opt];
                                onParamChange(field.key, { choices: next });
                              }}
                            >
                              {opt}
                            </button>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="ts-param-range">
                        <input
                          type="number"
                          placeholder="Min"
                          value={row?.min ?? ""}
                          disabled={disabled}
                          onChange={(e) => onParamChange(field.key, { min: e.target.value })}
                        />
                        <span>—</span>
                        <input
                          type="number"
                          placeholder="Max"
                          value={row?.max ?? ""}
                          disabled={disabled}
                          onChange={(e) => onParamChange(field.key, { max: e.target.value })}
                        />
                        <input
                          type="number"
                          placeholder="Steps"
                          value={row?.divisions ?? 3}
                          min={1}
                          max={25}
                          disabled={disabled}
                          onChange={(e) => onParamChange(field.key, { divisions: e.target.value })}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ProgressBar({ results, total, currentModel, experimentCount }) {
  const trained = results?.filter((r) => r.status === "trained")?.length || 0;
  const errors = results?.filter((r) => r.status === "error")?.length || 0;
  const skipped = results?.filter((r) => r.status === "skipped")?.length || 0;
  const done = trained + errors + skipped;
  const progress = total > 0 ? (done / total) * 100 : 0;
  
  return (
    <div className="ts-progress">
      <div className="ts-progress-header">
        {currentModel && (
          <div className="ts-progress-current">
            <span className="ts-progress-current-label">Treinando:</span>
            <span className="ts-progress-current-name">{currentModel.label}</span>
            <span className="ts-progress-current-info">
              ({currentModel.algorithms?.length || 1} algoritmo{(currentModel.algorithms?.length || 1) > 1 ? 's' : ''} · {experimentCount} experimentos)
            </span>
            <span className="ts-progress-spinner" />
          </div>
        )}
        <div className="ts-progress-counter">
          {done} / {total} modelos
        </div>
      </div>
      <div className="ts-progress-bar">
        <div className="ts-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="ts-progress-stats">
        <span className="ts-stat-success">{trained} treinados</span>
        <span className="ts-stat-skip">{skipped} pulados</span>
        <span className="ts-stat-error">{errors} erros</span>
      </div>
    </div>
  );
}

function ResultCard({ result, isSelected, onViewCandidates }) {
  // Suporta tanto resultados de ML (candidates) quanto regressão
  const isRegression = result.isRegression || result.regression_type;
  
  // Para ML: usar candidates/best_index
  const bestCandidate = result.candidates?.[result.best_index];
  const bestAlgo = bestCandidate?.algorithm || "?";
  const bestR2 = bestCandidate?.metrics?.val_r2 ?? bestCandidate?.metrics?.train_r2;
  
  // Para regressão: usar campos diretos
  const regressionR2 = result.metrics?.r2;
  const regressionType = result.regression_type;
  
  const regressionLabels = {
    linear: "Linear",
    quadratic: "Quadrática",
    exponential: "Exponencial",
    logarithmic: "Logarítmica",
    power: "Potência",
    polynomial: "Polinomial",
  };
  
  return (
    <button
      type="button"
      className={`ts-result-row ${isSelected ? "is-selected" : ""} ts-result-${result.status}`}
      onClick={() => onViewCandidates?.(result)}
      disabled={result.status !== "trained"}
    >
      <div className="ts-result-row-main">
        <span className="ts-result-row-name">{result.step_id}</span>
        {result.status === "trained" && !isRegression && (
          <span className="ts-result-row-meta">
            {result.n_candidates} candidatos · {result.n_samples} amostras
          </span>
        )}
        {result.status === "trained" && isRegression && (
          <span className="ts-result-row-meta">
            📈 Regressão · {result.metrics?.n_samples || result.n_collected} amostras
          </span>
        )}
        {result.status === "skipped" && (
          <span className="ts-result-row-meta ts-result-row-skip">{result.reason}</span>
        )}
        {result.status === "error" && (
          <span className="ts-result-row-meta ts-result-row-error">
            {formatErrorMessage(result.error)}
          </span>
        )}
      </div>
      {result.status === "trained" && !isRegression && (
        <div className="ts-result-row-stats">
          <span className="ts-result-row-algo" style={{ color: algoInfo(bestAlgo).color }}>
            {algoInfo(bestAlgo).label}
          </span>
          {bestR2 !== undefined && (
            <span className="ts-result-row-r2">R² {bestR2.toFixed(3)}</span>
          )}
        </div>
      )}
      {result.status === "trained" && isRegression && (
        <div className="ts-result-row-stats">
          <span className="ts-result-row-algo ts-regression-algo" style={{ color: "#9c27b0" }}>
            {regressionLabels[regressionType] || regressionType}
          </span>
          {regressionR2 !== undefined && (
            <span className="ts-result-row-r2">R² {regressionR2.toFixed(3)}</span>
          )}
        </div>
      )}
      {result.status === "trained" && <span className="ts-result-row-arrow">→</span>}
    </button>
  );
}

/**
 * Gráfico SVG simples para visualizar regressão
 */
function RegressionChart({ plotData, equation, regressionType }) {
  const width = 500;
  const height = 300;
  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
  // Dados inliers (usados no ajuste)
  const xData = plotData.data_points?.x || plotData.x || [];
  const yData = plotData.data_points?.y || plotData.y || [];
  
  // Dados originais (incluindo outliers)
  const xOriginal = plotData.original_data?.x || xData;
  const yOriginal = plotData.original_data?.y || yData;
  
  // Índices dos outliers
  const outlierIndices = new Set(plotData.outlier_indices || []);
  const hasOutliers = outlierIndices.size > 0;
  
  // Curva de regressão
  const xFit = plotData.curve?.x || plotData.x_fit || [];
  const yFit = plotData.curve?.y || plotData.y_fit || [];
  
  if (xData.length === 0 && xOriginal.length === 0) return null;
  
  // Eixos FIXOS conforme especificado
  const xMinPad = 0;
  const xMaxPad = 1440;
  const yMinPad = 0;
  const yMaxPad = 10;
  
  // Funções de escala
  const scaleX = (x) => padding.left + ((x - xMinPad) / (xMaxPad - xMinPad)) * chartWidth;
  const scaleY = (y) => padding.top + chartHeight - ((y - yMinPad) / (yMaxPad - yMinPad)) * chartHeight;
  
  // Gerar ticks
  const xTicks = [];
  const yTicks = [];
  const numTicks = 5;
  
  for (let i = 0; i <= numTicks; i++) {
    xTicks.push(xMinPad + (i / numTicks) * (xMaxPad - xMinPad));
    yTicks.push(yMinPad + (i / numTicks) * (yMaxPad - yMinPad));
  }
  
  // Path da curva de regressão (apenas dentro do range visível)
  let fitPath = "";
  if (xFit.length > 0 && yFit.length > 0) {
    const visiblePoints = xFit.map((x, i) => ({ x, y: yFit[i] }))
      .filter(p => p.x >= xMinPad && p.x <= xMaxPad);
    
    if (visiblePoints.length > 0) {
      fitPath = visiblePoints.map((p, i) => {
        const sx = scaleX(p.x);
        const sy = scaleY(p.y);
        return i === 0 ? `M ${sx} ${sy}` : `L ${sx} ${sy}`;
      }).join(" ");
    }
  }
  
  // Separar pontos em inliers e outliers
  const inlierPoints = [];
  const outlierPoints = [];
  
  xOriginal.forEach((x, i) => {
    const point = { x, y: yOriginal[i] };
    if (outlierIndices.has(i)) {
      outlierPoints.push(point);
    } else {
      inlierPoints.push(point);
    }
  });
  
  return (
    <div className="ts-regression-chart-container">
      <label>Gráfico de Ajuste</label>
      <svg width={width} height={height} className="ts-regression-chart">
        {/* Grid */}
        <g className="ts-chart-grid">
          {yTicks.map((y, i) => (
            <line
              key={`h-${i}`}
              x1={padding.left}
              y1={scaleY(y)}
              x2={width - padding.right}
              y2={scaleY(y)}
              stroke="#e5e7eb"
              strokeDasharray="2,2"
            />
          ))}
          {xTicks.map((x, i) => (
            <line
              key={`v-${i}`}
              x1={scaleX(x)}
              y1={padding.top}
              x2={scaleX(x)}
              y2={height - padding.bottom}
              stroke="#e5e7eb"
              strokeDasharray="2,2"
            />
          ))}
        </g>
        
        {/* Eixos */}
        <g className="ts-chart-axes">
          <line
            x1={padding.left}
            y1={height - padding.bottom}
            x2={width - padding.right}
            y2={height - padding.bottom}
            stroke="#9ca3af"
            strokeWidth="1"
          />
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={height - padding.bottom}
            stroke="#9ca3af"
            strokeWidth="1"
          />
        </g>
        
        {/* Labels dos eixos */}
        <g className="ts-chart-labels">
          {xTicks.map((x, i) => (
            <text
              key={`xl-${i}`}
              x={scaleX(x)}
              y={height - padding.bottom + 20}
              textAnchor="middle"
              fontSize="10"
              fill="#6b7280"
            >
              {x.toFixed(1)}
            </text>
          ))}
          {yTicks.map((y, i) => (
            <text
              key={`yl-${i}`}
              x={padding.left - 8}
              y={scaleY(y) + 3}
              textAnchor="end"
              fontSize="10"
              fill="#6b7280"
            >
              {y.toFixed(1)}
            </text>
          ))}
        </g>
        
        {/* Curva de regressão */}
        {fitPath && (
          <path
            d={fitPath}
            fill="none"
            stroke="#9c27b0"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
        
        {/* Pontos outliers (em vermelho, renderizados primeiro) */}
        {hasOutliers && (
          <g className="ts-chart-outliers">
            {outlierPoints.map((p, i) => {
              // Verificar se ponto está dentro do range visível
              const inRangeX = p.x >= xMinPad && p.x <= xMaxPad;
              const inRangeY = p.y >= yMinPad && p.y <= yMaxPad;
              
              if (inRangeX && inRangeY) {
                // Ponto dentro do range - mostrar normalmente
                return (
                  <circle
                    key={`outlier-${i}`}
                    cx={scaleX(p.x)}
                    cy={scaleY(p.y)}
                    r="5"
                    fill="#ef4444"
                    stroke="#fff"
                    strokeWidth="1.5"
                    opacity="0.7"
                  />
                );
              } else {
                // Ponto fora do range - mostrar seta na borda indicando direção
                const clampedX = Math.max(xMinPad, Math.min(xMaxPad, p.x));
                const clampedY = Math.max(yMinPad, Math.min(yMaxPad, p.y));
                const cx = scaleX(clampedX);
                const cy = scaleY(clampedY);
                
                // Determinar direção da seta
                const isLeft = p.x < xMinPad;
                const isRight = p.x > xMaxPad;
                const isUp = p.y > yMaxPad;
                const isDown = p.y < yMinPad;
                
                // Criar path de seta
                let arrowPath = "";
                if (isLeft) {
                  arrowPath = `M ${padding.left + 8} ${cy} L ${padding.left} ${cy - 5} L ${padding.left} ${cy + 5} Z`;
                } else if (isRight) {
                  arrowPath = `M ${width - padding.right - 8} ${cy} L ${width - padding.right} ${cy - 5} L ${width - padding.right} ${cy + 5} Z`;
                } else if (isUp) {
                  arrowPath = `M ${cx} ${padding.top + 8} L ${cx - 5} ${padding.top} L ${cx + 5} ${padding.top} Z`;
                } else if (isDown) {
                  arrowPath = `M ${cx} ${height - padding.bottom - 8} L ${cx - 5} ${height - padding.bottom} L ${cx + 5} ${height - padding.bottom} Z`;
                }
                
                return arrowPath ? (
                  <path
                    key={`outlier-arrow-${i}`}
                    d={arrowPath}
                    fill="#ef4444"
                    opacity="0.7"
                  />
                ) : null;
              }
            })}
          </g>
        )}
        
        {/* Pontos inliers (dados usados no ajuste) */}
        <g className="ts-chart-points">
          {inlierPoints.map((p, i) => (
            <circle
              key={`inlier-${i}`}
              cx={scaleX(p.x)}
              cy={scaleY(p.y)}
              r="5"
              fill="#6366f1"
              stroke="#fff"
              strokeWidth="1.5"
            />
          ))}
        </g>
        
        {/* Legenda */}
        <g className="ts-chart-legend" transform={`translate(${padding.left + 10}, ${padding.top + 10})`}>
          <rect x="0" y="0" width="130" height={hasOutliers ? 64 : 44} fill="rgba(255,255,255,0.9)" rx="4" />
          <circle cx="12" cy="12" r="4" fill="#6366f1" />
          <text x="24" y="15" fontSize="11" fill="#374151">Dados ({inlierPoints.length})</text>
          <line x1="6" y1="32" x2="18" y2="32" stroke="#9c27b0" strokeWidth="2.5" />
          <text x="24" y="35" fontSize="11" fill="#374151">Regressão</text>
          {hasOutliers && (
            <>
              <circle cx="12" cy="52" r="4" fill="#ef4444" opacity="0.7" />
              <text x="24" y="55" fontSize="11" fill="#374151">Outliers ({outlierPoints.length})</text>
            </>
          )}
        </g>
      </svg>
      
      {/* Equação abaixo do gráfico */}
      <div className="ts-regression-chart-equation">
        {equation}
      </div>
    </div>
  );
}

/**
 * Visualizador de resultados de regressão
 */
function RegressionResultView({ tenant, result, onBack, onApply }) {
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState(!!result.version);
  const [error, setError] = useState("");
  
  const regressionLabels = {
    linear: "Linear",
    quadratic: "Quadrática",
    exponential: "Exponencial",
    logarithmic: "Logarítmica",
    power: "Potência",
    polynomial: "Polinomial",
  };
  
  const metrics = result.metrics || {};
  const plotData = result.plot_data || {};
  const comparison = result.comparison || [];
  
  // Aplicar regressão ao pipeline
  const handleApply = async () => {
    if (applying || applied) return;
    setApplying(true);
    setError("");
    
    try {
      await axios.post(`${API_URL}/training/apply-regression`, null, {
        params: {
          tenant,
          step_id: result.step_id,
          regression_type: result.regression_type,
          coefficients: JSON.stringify(result.coefficients),
          equation: result.equation,
          r2_score: result.metrics?.r2,
          rmse: result.metrics?.rmse,
          mae: result.metrics?.mae,
          n_samples: result.metrics?.n_samples,
          y_transform: result.y_transform || "none",
        },
      });
      setApplied(true);
      onApply?.();
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Erro ao aplicar");
    } finally {
      setApplying(false);
    }
  };
  
  return (
    <div className="ts-regression-view">
      {/* Header */}
      <div className="ts-regression-view-header">
        <button type="button" className="ts-btn-back" onClick={onBack}>
          ← Voltar
        </button>
        <div className="ts-regression-view-title">
          <h3>Resultado: {result.step_id}</h3>
          <span className="ts-regression-view-type">
            📈 Regressão {regressionLabels[result.regression_type] || result.regression_type}
          </span>
        </div>
      </div>
      
      {/* Equação principal */}
      <div className="ts-regression-equation-box">
        <label>Equação Ajustada</label>
        <div className="ts-regression-equation-display">
          {result.equation || "y = ?"}
        </div>
        {result.y_transform && result.y_transform !== "none" && (
          <div className="ts-regression-transform-note">
            ⚠️ Y transformado: {result.y_transform === "log10p" ? "log₁₀(1+y)" : result.y_transform}
          </div>
        )}
      </div>
      
      {/* Métricas */}
      <div className="ts-regression-metrics">
        <div className="ts-regression-metric">
          <span className="ts-regression-metric-value">{metrics.r2?.toFixed(4) || "—"}</span>
          <span className="ts-regression-metric-label">R² Score</span>
        </div>
        <div className="ts-regression-metric">
          <span className="ts-regression-metric-value">{metrics.rmse?.toFixed(4) || "—"}</span>
          <span className="ts-regression-metric-label">RMSE</span>
        </div>
        <div className="ts-regression-metric">
          <span className="ts-regression-metric-value">{metrics.mae?.toFixed(4) || "—"}</span>
          <span className="ts-regression-metric-label">MAE</span>
        </div>
        <div className="ts-regression-metric">
          <span className="ts-regression-metric-value">{metrics.n_samples || "—"}</span>
          <span className="ts-regression-metric-label">Amostras</span>
        </div>
        {(metrics.n_outliers_removed > 0 || result.regression?.n_outliers_removed > 0) && (
          <div className="ts-regression-metric ts-outliers-metric">
            <span className="ts-regression-metric-value">
              {metrics.n_outliers_removed || result.regression?.n_outliers_removed || 0}
            </span>
            <span className="ts-regression-metric-label">Outliers Removidos</span>
          </div>
        )}
      </div>
      
      {/* Info de outliers */}
      {(result.outlier_method && result.outlier_method !== "none") && (
        <div className="ts-regression-outlier-info">
          <span className="ts-outlier-badge">
            🎯 {result.outlier_method.toUpperCase()}
          </span>
          <span className="ts-outlier-text">
            Remoção de outliers ativa - {metrics.n_outliers_removed || result.regression?.n_outliers_removed || 0} pontos removidos
          </span>
        </div>
      )}
      
      {/* Gráfico de regressão */}
      {plotData && (plotData.data_points?.x || plotData.x || plotData.curve?.x) && (
        <RegressionChart
          plotData={plotData}
          equation={result.equation}
          regressionType={result.regression_type}
        />
      )}
      
      {/* Comparativo se auto_select */}
      {comparison.length > 0 && (
        <div className="ts-regression-comparison">
          <label>Comparativo de Tipos</label>
          <div className="ts-regression-comparison-table">
            {comparison.map((item) => (
              <div
                key={item.type}
                className={`ts-regression-comparison-row ${item.selected ? "is-selected" : ""}`}
              >
                <span className="ts-regression-comparison-type">
                  {item.selected && "✓ "}{regressionLabels[item.type] || item.type}
                </span>
                <span className="ts-regression-comparison-eq">{item.equation}</span>
                <span className="ts-regression-comparison-r2">R² {item.r2?.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Coeficientes */}
      {result.coefficients && (
        <div className="ts-regression-coefficients">
          <label>Coeficientes</label>
          <div className="ts-regression-coefficients-list">
            {Object.entries(result.coefficients).map(([key, val]) => (
              <div key={key} className="ts-regression-coeff">
                <span className="ts-regression-coeff-name">{key}</span>
                <span className="ts-regression-coeff-value">{typeof val === "number" ? val.toFixed(6) : String(val)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Versão aplicada */}
      {applied && (
        <div className="ts-regression-applied">
          <span className="ts-regression-applied-icon">✓</span>
          <span>Aplicado ao pipeline{result.version ? ` - Versão ${result.version}` : ""}</span>
        </div>
      )}
      
      {/* Erro */}
      {error && (
        <div className="ts-regression-error">
          <span>⚠️ {error}</span>
        </div>
      )}
      
      {/* Botão Aplicar */}
      {!applied && (
        <div className="ts-regression-actions">
          <button
            type="button"
            className="ts-btn-primary"
            onClick={handleApply}
            disabled={applying}
          >
            {applying ? "Aplicando..." : "✓ Aplicar ao Pipeline"}
          </button>
        </div>
      )}
      
      {/* Erros e skipped */}
      {(result.errors?.length > 0 || result.skipped_reasons?.length > 0) && (
        <div className="ts-regression-warnings">
          {result.errors?.length > 0 && (
            <details>
              <summary>⚠️ {result.errors.length} erro(s)</summary>
              <ul>
                {result.errors.map((e, i) => <li key={i}>{e}</li>)}
              </ul>
            </details>
          )}
          {result.skipped_reasons?.length > 0 && (
            <details>
              <summary>ℹ️ {result.skipped_reasons.length} ignorado(s)</summary>
              <ul>
                {result.skipped_reasons.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Visualizador inline de candidatos - Design limpo
 */
function CandidatesView({ tenant, result, onBack, onApply }) {
  const [expandedIndex, setExpandedIndex] = useState(result?.best_index ?? 0);
  const [predictions, setPredictions] = useState(null);
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState("");
  const errorMessage = useMemo(() => formatErrorMessage(error), [error]);
  
  const candidates = result?.candidates || [];
  const sessionPath = result?.session_path || "";
  
  // Carregar predições quando candidato expande
  useEffect(() => {
    if (expandedIndex === null || !sessionPath) return;
    
    const loadPredictions = async () => {
      setLoadingPredictions(true);
      setPredictions(null);
      try {
        const match = sessionPath.match(/candidates_(\d{8}_\d{6})/);
        const sessionId = match ? match[1] : "";
        if (!sessionId) return;
        
        const res = await axios.get(
          `${API_URL}/training/candidates/${tenant}/${sessionId}/predictions/${expandedIndex}`
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
  }, [expandedIndex, sessionPath, tenant]);
  
  // Aplicar modelo
  const handleApply = async (index) => {
    if (applying) return;
    setApplying(true);
    setError("");
    
    try {
      await axios.post(`${API_URL}/training/select-candidate`, null, {
        params: {
          tenant,
          session_path: sessionPath,
          candidate_index: index,
          step_id: result.step_id,
          apply_to_pipeline: true,
          change_reason: `Modelo selecionado: ${candidates[index]?.algorithm || ""}`,
        },
      });
      onApply?.();
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Erro ao aplicar");
    } finally {
      setApplying(false);
    }
  };
  
  const formatMetric = (value) => {
    if (value === null || value === undefined) return "-";
    if (typeof value === "number") {
      if (Math.abs(value) < 0.001) return value.toExponential(2);
      return value.toFixed(4);
    }
    return String(value);
  };

  const toggleExpand = (idx) => {
    setExpandedIndex(expandedIndex === idx ? null : idx);
  };
  
  return (
    <div className="ts-candidates-clean">
      {/* Header simples */}
      <div className="ts-cand-header">
        <button type="button" className="ts-cand-back" onClick={onBack}>
          ← Voltar
        </button>
        <div className="ts-cand-title">
          <h3>{result.step_id}</h3>
          <span>{candidates.length} candidatos · {result.n_samples} amostras</span>
        </div>
      </div>
      
      {errorMessage && (
        <div className="ts-cand-error">
          {errorMessage}
          <button onClick={() => setError("")}>×</button>
        </div>
      )}
      
      {/* Tabela de candidatos */}
      <div className="ts-cand-table-wrap">
        <table className="ts-cand-table">
          <thead>
            <tr>
              <th style={{width: 50}}>#</th>
              <th>Algoritmo</th>
              <th>Parâmetros</th>
              <th>{result.selection_metric || "Score"}</th>
              <th>R² (val)</th>
              <th>RMSE</th>
              <th style={{width: 90}}>Ação</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((c, idx) => {
              const isBest = idx === result.best_index;
              const isExpanded = expandedIndex === idx;
              // Resumo dos parâmetros principais
              const paramSummary = Object.entries(c.params || {})
                .slice(0, 3)
                .map(([k, v]) => `${k}=${typeof v === 'number' ? (v < 0.001 ? v.toExponential(1) : v) : v}`)
                .join(', ');
              
              return (
                <React.Fragment key={idx}>
                  <tr 
                    className={`ts-cand-row ${isBest ? "is-best" : ""} ${isExpanded ? "is-expanded" : ""}`}
                    onClick={() => toggleExpand(idx)}
                  >
                    <td>
                      <span className="ts-cand-rank">
                        {c.rank}
                        {isBest && <span className="ts-cand-star">★</span>}
                      </span>
                    </td>
                    <td>
                      <span className="ts-cand-algo" style={{ color: algoInfo(c.algorithm).color }}>
                        {algoInfo(c.algorithm).label}
                      </span>
                    </td>
                    <td className="ts-cand-params-cell" title={Object.entries(c.params || {}).map(([k, v]) => `${k}=${v}`).join(', ')}>
                      {paramSummary || '-'}
                    </td>
                    <td className="ts-cand-metric">{formatMetric(c.score)}</td>
                    <td className="ts-cand-metric">{formatMetric(c.metrics?.val_r2)}</td>
                    <td className="ts-cand-metric">{formatMetric(c.metrics?.val_rmse)}</td>
                    <td>
                      <button
                        type="button"
                        className="ts-cand-apply-btn"
                        onClick={(e) => { e.stopPropagation(); handleApply(idx); }}
                        disabled={applying}
                      >
                        Aplicar
                      </button>
                    </td>
                  </tr>
                  
                  {/* Linha expandida com detalhes */}
                  {isExpanded && (
                    <tr className="ts-cand-expanded-row">
                      <td colSpan={7}>
                        <div className="ts-cand-details">
                          {/* Métricas */}
                          <div className="ts-cand-section">
                            <h4>Métricas</h4>
                            <div className="ts-cand-metrics-grid">
                              {Object.entries(c.metrics || {}).map(([k, v]) => (
                                <div key={k} className="ts-cand-metric-item">
                                  <span className="ts-cand-metric-key">{k}</span>
                                  <span className="ts-cand-metric-val">{formatMetric(v)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                          
                          {/* Gráficos */}
                          <div className="ts-cand-section">
                            <h4>Gráficos</h4>
                            {loadingPredictions ? (
                              <div className="ts-cand-loading">Carregando...</div>
                            ) : predictions ? (
                              <div className="ts-cand-charts">
                                <ScatterPlotInline
                                  title="Predito vs Real"
                                  xLabel="Real"
                                  yLabel="Predito"
                                  data={[
                                    ...(predictions.train_actual || []).map((v, i) => ({
                                      x: v, y: predictions.train_predicted?.[i] ?? v, isVal: false,
                                    })),
                                    ...(predictions.val_actual || []).map((v, i) => ({
                                      x: v, y: predictions.val_predicted?.[i] ?? v, isVal: true,
                                    })),
                                  ]}
                                />
                                <ResidualsPlotInline
                                  title="Resíduos"
                                  data={[
                                    ...(predictions.train_predicted || []).map((v, i) => ({
                                      x: v, y: predictions.train_residuals?.[i] ?? 0, isVal: false,
                                    })),
                                    ...(predictions.val_predicted || []).map((v, i) => ({
                                      x: v, y: predictions.val_residuals?.[i] ?? 0, isVal: true,
                                    })),
                                  ]}
                                />
                                <div className="ts-cand-legend">
                                  <span className="ts-legend-train">● Treino</span>
                                  <span className="ts-legend-val">● Validação</span>
                                </div>
                              </div>
                            ) : (
                              <div className="ts-cand-no-data">Dados não disponíveis</div>
                            )}
                          </div>
                          
                          {/* Parâmetros */}
                          <div className="ts-cand-section">
                            <h4>Parâmetros</h4>
                            <div className="ts-cand-params">
                              {Object.entries(c.params || {}).map(([k, v]) => (
                                <span key={k} className="ts-cand-param">
                                  {k}: <strong>{String(v)}</strong>
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * Gera ticks "bonitos" para um eixo
 */
function generateTicks(min, max, count = 5) {
  const range = max - min;
  if (range === 0) return [min];
  
  // Calcular step "bonito"
  const rawStep = range / (count - 1);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / magnitude;
  let niceStep;
  if (normalized <= 1) niceStep = magnitude;
  else if (normalized <= 2) niceStep = 2 * magnitude;
  else if (normalized <= 5) niceStep = 5 * magnitude;
  else niceStep = 10 * magnitude;
  
  const niceMin = Math.floor(min / niceStep) * niceStep;
  const niceMax = Math.ceil(max / niceStep) * niceStep;
  
  const ticks = [];
  for (let v = niceMin; v <= niceMax + niceStep * 0.5; v += niceStep) {
    ticks.push(v);
  }
  return ticks.slice(0, count + 2);
}

/**
 * Formata valor para exibição em eixo
 */
function formatAxisValue(v) {
  if (Math.abs(v) >= 10000 || (Math.abs(v) < 0.01 && v !== 0)) {
    return v.toExponential(1);
  }
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(Math.abs(v) < 1 ? 2 : 1);
}

/**
 * Gráfico de dispersão inline
 */
function ScatterPlotInline({ data, title, xLabel, yLabel, width = 280, height = 200 }) {
  if (!data?.length) return <div className="ts-chart-empty">Sem dados</div>;
  
  const padding = { top: 25, right: 15, bottom: 40, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
  const xVals = data.map(d => d.x);
  const yVals = data.map(d => d.y);
  const allVals = [...xVals, ...yVals];
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const range = maxVal - minVal || 1;
  const margin = range * 0.1;
  const domainMin = minVal - margin;
  const domainMax = maxVal + margin;
  
  const ticks = generateTicks(domainMin, domainMax, 5);
  
  const scaleX = (v) => padding.left + ((v - domainMin) / (domainMax - domainMin)) * chartWidth;
  const scaleY = (v) => padding.top + chartHeight - ((v - domainMin) / (domainMax - domainMin)) * chartHeight;
  
  return (
    <div className="ts-chart">
      <svg width={width} height={height}>
        <text x={width / 2} y={12} textAnchor="middle" className="ts-chart-title">{title}</text>
        
        {/* Grid lines */}
        {ticks.map((t, i) => (
          <g key={i}>
            <line
              x1={padding.left}
              y1={scaleY(t)}
              x2={width - padding.right}
              y2={scaleY(t)}
              stroke="#f3f4f6"
              strokeWidth={1}
            />
            <line
              x1={scaleX(t)}
              y1={padding.top}
              x2={scaleX(t)}
              y2={height - padding.bottom}
              stroke="#f3f4f6"
              strokeWidth={1}
            />
          </g>
        ))}
        
        {/* Linha ideal */}
        <line
          x1={scaleX(domainMin)} y1={scaleY(domainMin)}
          x2={scaleX(domainMax)} y2={scaleY(domainMax)}
          stroke="#d1d5db" strokeWidth={1.5} strokeDasharray="4,4"
        />
        
        {/* Pontos */}
        {data.map((d, i) => (
          <circle
            key={i}
            cx={scaleX(d.x)}
            cy={scaleY(d.y)}
            r={3}
            fill={d.isVal ? "#1d4ed8" : "#000000"}
            opacity={0.7}
          />
        ))}
        
        {/* Eixos */}
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#9ca3af" />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#9ca3af" />
        
        {/* Ticks e valores do eixo X */}
        {ticks.map((t, i) => {
          const x = scaleX(t);
          if (x < padding.left || x > width - padding.right) return null;
          return (
            <g key={`x-${i}`}>
              <line x1={x} y1={height - padding.bottom} x2={x} y2={height - padding.bottom + 4} stroke="#9ca3af" />
              <text x={x} y={height - padding.bottom + 14} textAnchor="middle" className="ts-axis-tick">
                {formatAxisValue(t)}
              </text>
            </g>
          );
        })}
        
        {/* Ticks e valores do eixo Y */}
        {ticks.map((t, i) => {
          const y = scaleY(t);
          if (y < padding.top || y > height - padding.bottom) return null;
          return (
            <g key={`y-${i}`}>
              <line x1={padding.left - 4} y1={y} x2={padding.left} y2={y} stroke="#9ca3af" />
              <text x={padding.left - 6} y={y + 3} textAnchor="end" className="ts-axis-tick">
                {formatAxisValue(t)}
              </text>
            </g>
          );
        })}
        
        <text x={width / 2} y={height - 2} textAnchor="middle" className="ts-chart-label">{xLabel}</text>
        <text x={12} y={height / 2} textAnchor="middle" className="ts-chart-label" transform={`rotate(-90, 12, ${height / 2})`}>{yLabel}</text>
      </svg>
    </div>
  );
}

/**
 * Gráfico de resíduos inline
 */
function ResidualsPlotInline({ data, title, width = 280, height = 200 }) {
  if (!data?.length) return <div className="ts-chart-empty">Sem dados</div>;
  
  const padding = { top: 25, right: 15, bottom: 40, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
  const xVals = data.map(d => d.x);
  const yVals = data.map(d => d.y);
  const xMin = Math.min(...xVals);
  const xMax = Math.max(...xVals);
  const xRange = xMax - xMin || 1;
  const xMargin = xRange * 0.05;
  const yAbsMax = Math.max(...yVals.map(Math.abs));
  const yRange = yAbsMax * 1.2 || 1;
  
  const scaleX = (v) => padding.left + ((v - xMin + xMargin) / (xRange + 2 * xMargin)) * chartWidth;
  const scaleY = (v) => padding.top + chartHeight / 2 - (v / yRange) * (chartHeight / 2);
  
  const xTicks = generateTicks(xMin, xMax, 5);
  const yTicks = generateTicks(-yAbsMax, yAbsMax, 5);
  
  return (
    <div className="ts-chart">
      <svg width={width} height={height}>
        <text x={width / 2} y={12} textAnchor="middle" className="ts-chart-title">{title}</text>
        
        {/* Grid lines horizontais */}
        {yTicks.map((t, i) => (
          <line
            key={`yg-${i}`}
            x1={padding.left}
            y1={scaleY(t)}
            x2={width - padding.right}
            y2={scaleY(t)}
            stroke="#f3f4f6"
            strokeWidth={1}
          />
        ))}
        
        {/* Linha zero */}
        <line
          x1={padding.left} y1={scaleY(0)}
          x2={width - padding.right} y2={scaleY(0)}
          stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4,4"
        />
        
        {/* Pontos */}
        {data.map((d, i) => (
          <circle
            key={i}
            cx={scaleX(d.x)}
            cy={scaleY(d.y)}
            r={3}
            fill={d.isVal ? "#1d4ed8" : "#000000"}
            opacity={0.7}
          />
        ))}
        
        {/* Eixos */}
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#9ca3af" />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#9ca3af" />
        
        {/* Ticks e valores do eixo X */}
        {xTicks.map((t, i) => {
          const x = scaleX(t);
          if (x < padding.left || x > width - padding.right) return null;
          return (
            <g key={`x-${i}`}>
              <line x1={x} y1={height - padding.bottom} x2={x} y2={height - padding.bottom + 4} stroke="#9ca3af" />
              <text x={x} y={height - padding.bottom + 14} textAnchor="middle" className="ts-axis-tick">
                {formatAxisValue(t)}
              </text>
            </g>
          );
        })}
        
        {/* Ticks e valores do eixo Y */}
        {yTicks.map((t, i) => {
          const y = scaleY(t);
          if (y < padding.top || y > height - padding.bottom) return null;
          return (
            <g key={`y-${i}`}>
              <line x1={padding.left - 4} y1={y} x2={padding.left} y2={y} stroke="#9ca3af" />
              <text x={padding.left - 6} y={y + 3} textAnchor="end" className="ts-axis-tick">
                {formatAxisValue(t)}
              </text>
            </g>
          );
        })}
        
        <text x={width / 2} y={height - 2} textAnchor="middle" className="ts-chart-label">Predito</text>
        <text x={12} y={height / 2} textAnchor="middle" className="ts-chart-label" transform={`rotate(-90, 12, ${height / 2})`}>Resíduo</text>
      </svg>
    </div>
  );
}

// ============================================================================
// Componente Principal
// ============================================================================

export default function TrainingStudio({
  tenant,
  pipeline,
  pipelineData,
  nodes,
  onClose,
  onOpenCandidates,
}) {
  // Dataset selecionado
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  
  // Controle do DatasetSelector (null = fechado, {protocolId, datasetId} = aberto)
  const [datasetSelectorConfig, setDatasetSelectorConfig] = useState(null);
  const [datasetListKey, setDatasetListKey] = useState(0); // Para forçar reload
  
  // Configurações de treinamento
  const [yTransform, setYTransform] = useState("log10p");
  const [testSize, setTestSize] = useState(0.2);
  const [selectionMetric, setSelectionMetric] = useState("rmse");
  const [maxTrials, setMaxTrials] = useState(60);
  const [invalidateCache, setInvalidateCache] = useState(false);  // Limpar cache antes de treinar
  
  // Modelos
  const [models, setModels] = useState(() => {
    const trainable = new Set(["ml_inference", "ml_inference_series", "ml_inference_multichannel", "ml_forecaster_series"]);
    const out = {};
    (nodes || []).filter((n) => trainable.has(n?.data?.blockName)).forEach((n) => {
      const config = n.data?.config || {};
      
      // Extrair bactéria do model_path: predict/coliformes/... -> Coliformes, predict/ecoli/... -> E. coli
      let bacteria = "";
      if (config.model_path) {
        const match = config.model_path.match(/predict[\\\/]([^\\\/]+)/i);
        if (match) {
          const raw = match[1].toLowerCase();
          // Formatar nome bonito
          if (raw === "ecoli" || raw === "e_coli") bacteria = "E. coli";
          else if (raw === "coliformes") bacteria = "Coliformes";
          else bacteria = raw.charAt(0).toUpperCase() + raw.slice(1);
        }
      }
      
      // Tag do resource (ex: turbidimetria_NMP)
      const tag = config.resource || "";
      
      out[n.id] = {
        stepId: n.id,
        label: n.data?.label || n.id,
        blockName: n.data?.blockName || "",
        // Info adicional para identificação
        outputUnit: config.output_unit || "",
        inputFeature: config.input_feature || "",
        channel: config.channel || "",
        tag: tag,       // Ex: "turbidimetria_NMP"
        bacteria: bacteria,  // Ex: "Coliformes" ou "E. coli"
        enabled: true,
        // Tipo de modelo: "ml" ou "regression"
        modelType: "ml",
        // ML Config
        algorithms: ["ridge"],
        paramsByAlgorithm: { ridge: buildDefaultParams("ridge") },
        // Regression Config
        regressionType: "linear",
        regressionAutoSelect: false,
        polynomialDegree: 3,
        outlierMethod: "none", // none, ransac, iqr, zscore
        robustMethod: "ols",   // ols, theil_sen, huber, ransac_fit
        // Status
        status: null,
      };
    });
    return out;
  });

  const stepLabelMap = useMemo(() => {
    const editor = pipelineData?.editor;
    const nodesList = Array.isArray(editor?.nodes) ? editor.nodes : [];
    const edgesList = Array.isArray(editor?.edges) ? editor.edges : [];
    const nodesById = new Map(nodesList.map((n) => [n.id, n]));
    const incoming = new Map();
    edgesList.forEach((edge) => {
      const target = edge?.target;
      const source = edge?.source;
      if (!target || !source) return;
      if (!incoming.has(target)) incoming.set(target, []);
      incoming.get(target).push(source);
    });

    const normalizeLabel = (rawValue) => {
      const raw = String(rawValue || "").trim();
      if (!raw) return "";
      const lowered = raw.toLowerCase();
      if (lowered === "e_coli" || lowered === "ecoli" || lowered === "e. coli") return "ecoli";
      if (lowered === "coliformes_totais" || lowered === "coliformes") return "coliformes";
      return raw.replace(/_/g, " ");
    };

    const findLabel = (stepId) => {
      const stack = [stepId];
      const visited = new Set();
      while (stack.length) {
        const current = stack.pop();
        if (!current || visited.has(current)) continue;
        visited.add(current);
        const node = nodesById.get(current);
        if (node?.data?.blockName === "label") {
          const cfg = node.data?.config || {};
          const raw = cfg.label || node.data?.label || "";
          return normalizeLabel(raw);
        }
        const srcs = incoming.get(current) || [];
        srcs.forEach((src) => stack.push(src));
      }
      return "";
    };

    const out = {};
    nodesList.forEach((n) => {
      const blockName = n?.data?.blockName;
      if (!blockName) return;
      if (!["ml_inference", "ml_inference_series", "ml_inference_multichannel", "ml_forecaster_series"].includes(blockName)) return;
      const label = findLabel(n.id);
      if (label) out[n.id] = label;
    });
    return out;
  }, [pipelineData]);
  
  // UI
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [activeTab, setActiveTab] = useState("config");
  const [running, setRunning] = useState(false);
  const [currentTrainingModel, setCurrentTrainingModel] = useState(null);
  const [results, setResults] = useState([]);
  const [viewingResult, setViewingResult] = useState(null); // Resultado sendo analisado (candidatos)
  const [error, setError] = useState("");
  const errorMessage = useMemo(() => formatErrorMessage(error), [error]);
  
  // History view states
  const [viewMode, setViewMode] = useState("history"); // "history" | "training"
  const [historyEntries, setHistoryEntries] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [viewingHistoryEntry, setViewingHistoryEntry] = useState(null);
  
  // AbortController para cancelamento de treinamento
  const abortControllerRef = useRef(null);
  
  // Computed
  const experimentIds = useMemo(() => {
    const merged = [];
    (selectedDatasets || []).forEach((ds) => {
      (ds?.experiment_ids || []).forEach((id) => {
        if (id && !merged.includes(id)) merged.push(id);
      });
    });
    return merged;
  }, [selectedDatasets]);
  const protocolId = useMemo(() => {
    const protocols = (selectedDatasets || [])
      .map((ds) => ds?.protocol_id)
      .filter((p) => p);
    const unique = Array.from(new Set(protocols));
    return unique.length === 1 ? unique[0] : "";
  }, [selectedDatasets]);
  const enabledModels = useMemo(() => Object.values(models).filter((m) => m.enabled !== false), [models]);
  const selectedModel = selectedModelId ? models[selectedModelId] : null;
  
  // Handlers
  const toggleModel = useCallback((stepId, enabled) => {
    setModels((prev) => ({ ...prev, [stepId]: { ...prev[stepId], enabled } }));
  }, []);
  
  const toggleAlgorithm = useCallback((stepId, algoKey) => {
    setModels((prev) => {
      const model = prev[stepId];
      if (!model) return prev;
      const current = model.algorithms || ["ridge"];
      const has = current.includes(algoKey);
      if (has && current.length <= 1) return prev;
      const next = has ? current.filter((a) => a !== algoKey) : [...current, algoKey];
      const paramsByAlgorithm = { ...model.paramsByAlgorithm };
      if (!paramsByAlgorithm[algoKey]) {
        paramsByAlgorithm[algoKey] = buildDefaultParams(algoKey);
      }
      return { ...prev, [stepId]: { ...model, algorithms: next, paramsByAlgorithm } };
    });
  }, []);
  
  // Handlers para tipo de modelo e regressão
  const setModelType = useCallback((stepId, modelType) => {
    setModels((prev) => ({ ...prev, [stepId]: { ...prev[stepId], modelType } }));
  }, []);
  
  const setRegressionType = useCallback((stepId, regressionType) => {
    setModels((prev) => ({ ...prev, [stepId]: { ...prev[stepId], regressionType } }));
  }, []);
  
  const setRegressionAutoSelect = useCallback((stepId, regressionAutoSelect) => {
    setModels((prev) => ({ ...prev, [stepId]: { ...prev[stepId], regressionAutoSelect } }));
  }, []);
  
  const setPolynomialDegree = useCallback((stepId, polynomialDegree) => {
    setModels((prev) => ({ ...prev, [stepId]: { ...prev[stepId], polynomialDegree } }));
  }, []);
  
  const setOutlierMethod = useCallback((stepId, outlierMethod) => {
    setModels((prev) => ({ ...prev, [stepId]: { ...prev[stepId], outlierMethod } }));
  }, []);
  
  const setRobustMethod = useCallback((stepId, robustMethod) => {
    setModels((prev) => ({ ...prev, [stepId]: { ...prev[stepId], robustMethod } }));
  }, []);
  
  const updateParam = useCallback((stepId, algoKey, paramKey, patch) => {
    setModels((prev) => {
      const model = prev[stepId];
      if (!model) return prev;
      const paramsByAlgorithm = { ...model.paramsByAlgorithm };
      const algoParams = { ...(paramsByAlgorithm[algoKey] || buildDefaultParams(algoKey)) };
      algoParams[paramKey] = { ...algoParams[paramKey], ...patch };
      paramsByAlgorithm[algoKey] = algoParams;
      return { ...prev, [stepId]: { ...model, paramsByAlgorithm } };
    });
  }, []);
  
  // ============================================================================
  // History Functions
  // ============================================================================
  
  // Load training history for all models
  const loadHistory = useCallback(async () => {
    if (!tenant) {
      setHistoryLoading(false);
      return;
    }
    
    setHistoryLoading(true);
    const allEntries = [];
    
    try {
      // Load history for each model step
      for (const model of Object.values(models)) {
        try {
          const res = await axios.get(`${API_URL}/training/history/${tenant}/${model.stepId}`);
          const entries = res.data?.entries || [];
          entries.forEach((e) => {
            const fallbackLabel = stepLabelMap[model.stepId] || model.label;
            allEntries.push({ ...e, stepId: model.stepId, stepLabel: fallbackLabel });
          });
        } catch {
          // Ignore errors for individual steps
        }
      }
      
      // Sort by timestamp descending
      allEntries.sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));
      setHistoryEntries(allEntries);
    } catch (err) {
      console.error("Failed to load training history:", err);
    } finally {
      setHistoryLoading(false);
    }
  }, [tenant, models]);
  
  // Save training result to history
  const saveToHistory = useCallback(async (stepId, mode, config, result) => {
    if (!tenant || !stepId) return;
    
    try {
      await axios.post(`${API_URL}/training/history/${tenant}/${stepId}`, {
        mode,
        config,
        result,
        applied: false,
      });
      // Reload history
      loadHistory();
    } catch (err) {
      console.error("Failed to save training history:", err);
    }
  }, [tenant, loadHistory]);
  
  // Delete history entry
  const deleteHistoryEntry = useCallback(async (stepId, historyId) => {
    if (!tenant || !stepId || !historyId) return;
    
    try {
      await axios.delete(`${API_URL}/training/history/${tenant}/${stepId}/${historyId}`);
      setHistoryEntries((prev) => prev.filter((e) => e.id !== historyId));
    } catch (err) {
      console.error("Failed to delete history entry:", err);
    }
  }, [tenant]);
  
  // Load full history entry details
  const loadHistoryEntryDetails = useCallback(async (stepId, historyId) => {
    if (!tenant || !stepId || !historyId) return null;
    
    try {
      const res = await axios.get(`${API_URL}/training/history/${tenant}/${stepId}/${historyId}`);
      return res.data;
    } catch (err) {
      console.error("Failed to load history entry:", err);
      return null;
    }
  }, [tenant]);
  
  // View a history entry
  const handleViewHistoryEntry = useCallback(async (entry) => {
    const details = await loadHistoryEntryDetails(entry.stepId, entry.id);
    if (details) {
      // Convert to viewable result format
      const viewableResult = {
        ...details.result,
        step_id: entry.stepId,
        isRegression: details.mode === "regression",
      };
      setViewingHistoryEntry(viewableResult);
    }
  }, [loadHistoryEntryDetails]);
  
  // Load history on mount
  useEffect(() => {
    loadHistory();
  }, [loadHistory]);
  
  // Treinar
  const runTraining = useCallback(async () => {
    if (!tenant || !pipeline) return;
    if (!selectedDatasets || selectedDatasets.length === 0) {
      setError("Selecione ao menos um dataset");
      return;
    }
    if (enabledModels.length === 0) {
      setError("Habilite ao menos um modelo");
      return;
    }
    
    // Criar AbortController para esta sessão de treinamento
    const controller = new AbortController();
    abortControllerRef.current = controller;
    
    setRunning(true);
    setError("");
    setResults([]);
    setActiveTab("results");
    setDatasetSelectorConfig(null);
    setCurrentTrainingModel(null);
    
    const newResults = [];
    let cancelled = false;
    
    for (const model of enabledModels) {
      // Verificar se foi cancelado
      if (controller.signal.aborted) {
        cancelled = true;
        break;
      }
      
      setCurrentTrainingModel(model);
      setModels((prev) => ({ ...prev, [model.stepId]: { ...prev[model.stepId], status: "training" } }));
      
      try {
        // =====================================================================
        // REGRESSÃO MATEMÁTICA
        // =====================================================================
        if (model.modelType === "regression") {
          const params = new URLSearchParams();
          params.append("tenant", tenant);
          params.append("step_id", model.stepId);
          if (protocolId) params.append("protocolId", protocolId);
          experimentIds.forEach((id) => params.append("experimentIds", id));
          params.append("regression_type", model.regressionType || "linear");
          params.append("polynomial_degree", String(model.polynomialDegree || 3));
          params.append("y_transform", yTransform);  // Usar mesma transformação Y do ML
          params.append("outlier_method", model.outlierMethod || "none"); // Remoção de outliers
          params.append("robust_method", model.robustMethod || "ols");    // Método robusto
          // Não aplicar automaticamente - deixar usuário revisar e aplicar
          params.append("apply_to_pipeline", "false");
          if (invalidateCache) {
            params.append("invalidate_cache", "true");
          }
          
          const res = await axios.post(
            `${API_URL}/training/regression?${params.toString()}`,
            pipelineData,
            { signal: controller.signal }
          );
          
          const result = { 
            ...res.data, 
            step_id: model.stepId,
            isRegression: true,
          };
          newResults.push(result);
          setResults([...newResults]);
          setModels((prev) => ({ ...prev, [model.stepId]: { ...prev[model.stepId], status: result.status } }));
          
          // Save to history if training was successful
          if (result.status === "trained") {
            saveToHistory(model.stepId, "regression", {
              regressionType: model.regressionType,
              outlierMethod: model.outlierMethod,
              robustMethod: model.robustMethod,
              yTransform,
            }, result);
          }
          continue;
        }
        // =====================================================================
        // MACHINE LEARNING
        // =====================================================================
        const paramGridByAlgorithm = {};
        for (const algo of model.algorithms) {
          const algoParams = model.paramsByAlgorithm?.[algo] || {};
          const grid = {};
          const schema = ALGO_PARAM_SCHEMA[algo] || {};
          
          for (const field of schema.fields || []) {
            const row = algoParams[field.key] || {};
            if (row.mode === "grid" && field.grid) {
              if (field.kind === "select") {
                const choices = row.choices || [];
                if (choices.length > 0) grid[field.key] = choices;
              } else {
                const min = Number(row.min);
                const max = Number(row.max);
                const divisions = Math.max(1, Math.min(25, parseInt(row.divisions) || 3));
                if (Number.isFinite(min) && Number.isFinite(max) && max >= min) {
                  const step = divisions > 1 ? (max - min) / (divisions - 1) : 0;
                  const values = [];
                  for (let i = 0; i < divisions; i++) {
                    const v = min + step * i;
                    values.push(field.kind === "int" ? Math.round(v) : v);
                  }
                  grid[field.key] = values;
                }
              }
            }
          }
          
          if (Object.keys(grid).length > 0) {
            paramGridByAlgorithm[algo] = grid;
          }
        }
        
        const params = new URLSearchParams();
        params.append("tenant", tenant);
        params.append("step_id", model.stepId);
        if (protocolId) {
          params.append("protocolId", protocolId);
        }
        experimentIds.forEach((id) => params.append("experimentIds", id));
        params.append("algorithm", model.algorithms[0] || "ridge");
        model.algorithms.forEach((a) => params.append("algorithms", a));
        params.append("y_transform", yTransform);
        params.append("selection_metric", selectionMetric);
        params.append("max_trials", String(maxTrials));
        params.append("test_size", String(testSize));
        if (invalidateCache) {
          params.append("invalidate_cache", "true");
        }
        
        if (Object.keys(paramGridByAlgorithm).length > 0) {
          params.append("param_grid", JSON.stringify(paramGridByAlgorithm));
        }
        
        const res = await axios.post(
          `${API_URL}/training/grid-search?${params.toString()}`,
          pipelineData,  // Enviar pipeline data (objeto completo) no body para cálculo correto do cache
          { signal: controller.signal }
        );
        const result = { ...res.data, step_id: model.stepId };
        newResults.push(result);
        setResults([...newResults]);
        setModels((prev) => ({ ...prev, [model.stepId]: { ...prev[model.stepId], status: result.status } }));
        
        // Save to history if training was successful
        if (result.status === "trained") {
          saveToHistory(model.stepId, "ml", {
            algorithms: model.algorithms,
            yTransform,
            selectionMetric,
            maxTrials,
            testSize,
          }, result);
        }
      } catch (err) {
        // Verificar se foi cancelamento
        if (axios.isCancel(err) || err.name === "CanceledError" || controller.signal.aborted) {
          cancelled = true;
          setModels((prev) => ({ ...prev, [model.stepId]: { ...prev[model.stepId], status: "cancelled" } }));
          break;
        }
        const result = { step_id: model.stepId, status: "error", error: err?.response?.data?.detail || err.message };
        newResults.push(result);
        setResults([...newResults]);
        setModels((prev) => ({ ...prev, [model.stepId]: { ...prev[model.stepId], status: "error" } }));
      }
    }
    
    if (cancelled) {
      setError("Treinamento cancelado pelo usuário");
    }
    
    abortControllerRef.current = null;
    setCurrentTrainingModel(null);
    setRunning(false);
  }, [tenant, pipelineData, selectedDatasets, enabledModels, protocolId, experimentIds, yTransform, selectionMetric, maxTrials, testSize, invalidateCache, saveToHistory]);
  
  // Cancelar treinamento
  const cancelTraining = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);
  
  // Handler para quando salvar/fechar dataset selector
  const handleDatasetSelectorClose = useCallback(() => {
    setDatasetSelectorConfig(null);
    setDatasetListKey((k) => k + 1); // Força reload da lista
  }, []);

  // Handler para criar novo dataset
  const handleCreateDataset = useCallback(() => {
    setDatasetSelectorConfig({ protocolId: "", datasetId: null });
  }, []);

  // Handler para editar dataset existente
  const handleEditDataset = useCallback(async (ds) => {
    // Busca o protocolo do dataset para abrir o selector já no protocolo certo
    try {
      const res = await axios.get(`${API_URL}/datasets/${tenant}/${ds.id}`);
      setDatasetSelectorConfig({ 
        protocolId: res.data.protocol_id || "", 
        datasetId: ds.id,
        initialSelection: res.data.experiment_ids || []
      });
    } catch {
      setDatasetSelectorConfig({ protocolId: ds.protocol_id || "", datasetId: ds.id });
    }
  }, [tenant]);

  // Handler para quando seleção do dataset selector mudar
  const handleDatasetSelectionChange = useCallback((expIds) => {
    // Não faz nada aqui - a seleção é salva dentro do DatasetSelector
  }, []);

  return (
    <div className="ts-container">
      {/* DatasetSelector Overlay */}
      {datasetSelectorConfig && (
        <div className="ts-dataset-selector-overlay">
          <DatasetSelector
            tenant={tenant}
            protocolId={datasetSelectorConfig.protocolId}
            selectedExperimentIds={datasetSelectorConfig.initialSelection || []}
            onSelectionChange={handleDatasetSelectionChange}
            onClose={handleDatasetSelectorClose}
            disabled={running}
          />
        </div>
      )}
      {/* Header */}
      <div className="ts-header">
        <div className="ts-header-title">
          <h2>Training Studio</h2>
          <span className="ts-header-pipeline">{pipeline}</span>
        </div>
        <div className="ts-header-stats">
          <div className="ts-header-stat">
            <span className="ts-header-stat-value">{experimentIds.length}</span>
            <span className="ts-header-stat-label">experimentos</span>
          </div>
          <div className="ts-header-stat">
            <span className="ts-header-stat-value">{enabledModels.length}</span>
            <span className="ts-header-stat-label">modelos</span>
          </div>
        </div>
        <button type="button" className="ts-close" onClick={onClose} aria-label="Fechar">
          <IconClose />
        </button>
      </div>
      
      {/* Progress */}
      {running && (
        <ProgressBar 
          results={results} 
          total={enabledModels.length} 
          currentModel={currentTrainingModel}
          experimentCount={experimentIds.length}
        />
      )}
      
      {/* Main */}
      <div className="ts-main">
        {/* Sidebar */}
        <div className="ts-sidebar">
          <Section title="Dataset" badge={experimentIds.length} icon={<IconDatabase />} defaultOpen={true}>
            <DatasetList
              key={datasetListKey}
              tenant={tenant}
              selectedIds={(selectedDatasets || []).map((d) => d.id)}
              onSelect={(ds) => {
                setSelectedDatasets((prev) => {
                  const list = Array.isArray(prev) ? prev : [];
                  if (list.find((d) => d.id === ds.id)) {
                    return list.filter((d) => d.id !== ds.id);
                  }
                  return [...list, ds];
                });
              }}
              onEdit={handleEditDataset}
              onCreate={handleCreateDataset}
              disabled={running}
            />
            {selectedDatasets.length > 0 && (
              <div className="ts-dataset-selected">
                <span className="ts-dataset-selected-name">
                  {selectedDatasets.map((ds) => ds.name).join(", ")}
                </span>
                <span className="ts-dataset-selected-protocol">
                  Protocolo: {protocolId || "múltiplos"}
                </span>
                <span className="ts-dataset-selected-count">{experimentIds.length} experimentos</span>
              </div>
            )}
          </Section>
          
          <Section title="Modelos" badge={enabledModels.length} icon={<IconModel />} defaultOpen={true}>
            <div className="ts-models">
              {Object.values(models).map((model) => {
                const bacteriaLabel = stepLabelMap[model.stepId] || model.bacteria || "";
                const modelView = bacteriaLabel ? { ...model, bacteria: bacteriaLabel } : model;
                return (
                <ModelCard
                  key={model.stepId}
                  model={modelView}
                  isSelected={selectedModelId === model.stepId}
                  onClick={(id) => { setSelectedModelId(id); setActiveTab("config"); }}
                  onToggle={toggleModel}
                  disabled={running}
                />
                );
              })}
            </div>
          </Section>
          
          <Section title="Configurações" icon={<IconSettings />} defaultOpen={false}>
            <div className="ts-settings">
              <div className="ts-field">
                <label>Transformação Y</label>
                <div className="ts-toggle-group">
                  <button className={yTransform === "log10p" ? "is-active" : ""} onClick={() => setYTransform("log10p")} disabled={running}>
                    log10(1+y)
                  </button>
                  <button className={yTransform === "none" ? "is-active" : ""} onClick={() => setYTransform("none")} disabled={running}>
                    Nenhuma
                  </button>
                </div>
              </div>
              <div className="ts-field-row">
                <div className="ts-field">
                  <label>Métrica</label>
                  <select value={selectionMetric} onChange={(e) => setSelectionMetric(e.target.value)} disabled={running}>
                    <option value="rmse">RMSE</option>
                    <option value="mae">MAE</option>
                    <option value="r2">R²</option>
                  </select>
                </div>
                <div className="ts-field">
                  <label>Test size</label>
                  <input type="number" value={testSize} min={0} max={0.8} step={0.05} onChange={(e) => setTestSize(e.target.value)} disabled={running} />
                </div>
                <div className="ts-field">
                  <label>Max trials</label>
                  <input type="number" value={maxTrials} min={1} max={500} onChange={(e) => setMaxTrials(e.target.value)} disabled={running} />
                </div>
              </div>
            </div>
          </Section>
        </div>
        
        {/* Content - configuração ou resultados */}
        <div className="ts-content">
          <div className="ts-tabs">
            <button className={activeTab === "config" ? "is-active" : ""} onClick={() => setActiveTab("config")}>
              Configuração
            </button>
            <button 
              className={activeTab === "results" ? "is-active" : ""} 
              onClick={() => setActiveTab("results")}
              disabled={results.length === 0 && !running}
            >
              Resultados {results.length > 0 && `(${results.length})`}
            </button>
            <button 
              className={activeTab === "history" ? "is-active" : ""} 
              onClick={() => setActiveTab("history")}
            >
              Histórico {historyEntries.length > 0 && `(${historyEntries.length})`}
            </button>
          </div>
          
          <div className="ts-tab-content">
            {activeTab === "config" && (
              selectedModel ? (
                <div className="ts-model-config">
                  <div className="ts-model-config-header">
                    <h3>{selectedModel.label}</h3>
                    <span>{selectedModel.blockName}</span>
                  </div>
                  
                  {/* Seletor de tipo de modelo */}
                  <div className="ts-model-type-selector">
                    <label>Tipo de Modelo</label>
                    <div className="ts-model-type-chips">
                      <button
                        type="button"
                        className={`ts-model-type-chip ${selectedModel.modelType !== "regression" ? "is-selected" : ""}`}
                        disabled={running}
                        onClick={() => setModelType(selectedModel.stepId, "ml")}
                      >
                        🤖 Machine Learning
                      </button>
                      <button
                        type="button"
                        className={`ts-model-type-chip ${selectedModel.modelType === "regression" ? "is-selected" : ""}`}
                        disabled={running}
                        onClick={() => setModelType(selectedModel.stepId, "regression")}
                      >
                        📈 Regressão Matemática
                      </button>
                    </div>
                    <small className="ts-model-type-hint">
                      {selectedModel.modelType === "regression" 
                        ? "Ajusta uma equação matemática simples (ex: y = ax + b). Mais interpretável e rápido."
                        : "Treina modelos de ML (Ridge, XGBoost, etc.). Mais preciso para dados complexos."}
                    </small>
                  </div>
                  
                  {/* REGRESSÃO MATEMÁTICA */}
                  {selectedModel.modelType === "regression" && (
                    <div className="ts-regression-config">
                      <div className="ts-regression-type-selector">
                        <label>Tipo de Regressão</label>
                        <div className="ts-algo-chips">
                          {REGRESSION_ITEMS.map((item) => (
                            <button
                              key={item.key}
                              type="button"
                              className={`ts-algo-chip ${selectedModel.regressionType === item.key ? "is-selected" : ""}`}
                              style={{ "--algo-color": item.color }}
                              disabled={running || selectedModel.regressionAutoSelect}
                              onClick={() => setRegressionType(selectedModel.stepId, item.key)}
                              title={`${item.equation}`}
                            >
                              {item.label}
                            </button>
                          ))}
                        </div>
                      </div>
                      
                      <div className="ts-regression-options">
                        <label className="ts-checkbox-label">
                          <input
                            type="checkbox"
                            checked={selectedModel.regressionAutoSelect}
                            disabled={running}
                            onChange={(e) => setRegressionAutoSelect(selectedModel.stepId, e.target.checked)}
                          />
                          Seleção automática (testa todos e escolhe o melhor R²)
                        </label>
                      </div>
                      
                      {selectedModel.regressionType === "polynomial" && !selectedModel.regressionAutoSelect && (
                        <div className="ts-regression-degree">
                          <label>Grau do Polinômio</label>
                          <input
                            type="number"
                            value={selectedModel.polynomialDegree || 3}
                            min={2}
                            max={10}
                            disabled={running}
                            onChange={(e) => setPolynomialDegree(selectedModel.stepId, Number(e.target.value) || 3)}
                          />
                        </div>
                      )}
                      
                      <div className="ts-regression-outliers">
                        <label>Remoção de Outliers</label>
                        <select
                          value={selectedModel.outlierMethod || "none"}
                          disabled={running}
                          onChange={(e) => setOutlierMethod(selectedModel.stepId, e.target.value)}
                        >
                          {OUTLIER_METHODS.map((m) => (
                            <option key={m.key} value={m.key}>{m.label}</option>
                          ))}
                        </select>
                        <small className="ts-outlier-hint">
                          {OUTLIER_METHODS.find((m) => m.key === (selectedModel.outlierMethod || "none"))?.desc || ""}
                        </small>
                      </div>
                      
                      {/* Método robusto - para todas as regressões */}
                      <div className="ts-regression-outliers">
                        <label>Método Robusto</label>
                        <select
                          value={selectedModel.robustMethod || "ols"}
                          disabled={running}
                          onChange={(e) => setRobustMethod(selectedModel.stepId, e.target.value)}
                        >
                          {ROBUST_METHODS.map((m) => (
                            <option key={m.key} value={m.key}>{m.label}</option>
                          ))}
                        </select>
                        <small className="ts-outlier-hint">
                          {ROBUST_METHODS.find((m) => m.key === (selectedModel.robustMethod || "ols"))?.desc || ""}
                        </small>
                      </div>
                      
                      <div className="ts-regression-preview">
                        <div className="ts-regression-equation">
                          <strong>Equação:</strong> {REGRESSION_ITEMS.find((r) => r.key === selectedModel.regressionType)?.equation || "y = ?"}
                        </div>
                        <small>{REGRESSION_ITEMS.find((r) => r.key === selectedModel.regressionType)?.label || ""}</small>
                      </div>
                    </div>
                  )}
                  
                  {/* MACHINE LEARNING */}
                  {selectedModel.modelType !== "regression" && (
                  <>
                  <div className="ts-algo-selector">
                    <label>Algoritmos</label>
                    <div className="ts-algo-chips">
                      {ALGORITHM_ITEMS.map((item) => (
                        <button
                          key={item.key}
                          type="button"
                          className={`ts-algo-chip ${(selectedModel.algorithms || []).includes(item.key) ? "is-selected" : ""}`}
                          style={{ "--algo-color": item.color }}
                              disabled={running}
                              onClick={() => toggleAlgorithm(selectedModel.stepId, item.key)}
                            >
                              {item.label}
                              <span className={`ts-algo-chip-gpu ts-gpu-${algoGpuTag(item.key)}`} title={algoGpuLabel(item.key)}>
                                {algoGpuLabel(item.key)}
                              </span>
                            </button>
                          ))}
                        </div>
                      </div>
                      
                      <div className="ts-algo-configs">
                        {(selectedModel.algorithms || ["ridge"]).map((algo) => (
                          <AlgorithmConfig
                            key={algo}
                            algorithm={algo}
                            params={selectedModel.paramsByAlgorithm?.[algo] || {}}
                            onParamChange={(paramKey, patch) => updateParam(selectedModel.stepId, algo, paramKey, patch)}
                            disabled={running}
                          />
                        ))}
                      </div>
                    </>
                    )}
                    </div>
                  ) : (
                    <div className="ts-empty">
                      <p>Selecione um modelo para configurar</p>
                    </div>
                  )
                )}
                
                {activeTab === "results" && (
                  viewingResult ? (
                    // Mostrar view de regressão ou de candidatos ML dependendo do tipo
                    viewingResult.isRegression || viewingResult.regression_type ? (
                      <RegressionResultView
                        tenant={tenant}
                        result={viewingResult}
                        onBack={() => setViewingResult(null)}
                        onApply={() => setViewingResult(null)}
                      />
                    ) : (
                      <CandidatesView
                        tenant={tenant}
                        result={viewingResult}
                        onBack={() => setViewingResult(null)}
                        onApply={() => setViewingResult(null)}
                      />
                    )
                  ) : (
                    <div className="ts-results">
                      {results.length === 0 && !running ? (
                        <div className="ts-empty">
                          <p>Execute o treinamento para ver resultados</p>
                        </div>
                      ) : (
                        <div className="ts-results-grid">
                          {results.map((result) => (
                            <ResultCard
                              key={result.step_id}
                              result={result}
                              isSelected={viewingResult?.step_id === result.step_id}
                              onViewCandidates={(r) => setViewingResult(r)}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  )
                )}
                
                {activeTab === "history" && (
                  viewingHistoryEntry ? (
                    // Mostrar view de resultado do histórico
                    viewingHistoryEntry.isRegression || viewingHistoryEntry.regression_type ? (
                      <RegressionResultView
                        tenant={tenant}
                        result={viewingHistoryEntry}
                        onBack={() => setViewingHistoryEntry(null)}
                        onApply={() => {
                          setViewingHistoryEntry(null);
                          loadHistory();
                        }}
                      />
                    ) : (
                      <CandidatesView
                        tenant={tenant}
                        result={viewingHistoryEntry}
                        onBack={() => setViewingHistoryEntry(null)}
                        onApply={() => {
                          setViewingHistoryEntry(null);
                          loadHistory();
                        }}
                      />
                    )
                  ) : (
                    <div className="ts-history">
                      {historyLoading ? (
                        <div className="ts-history-loading">
                          <span className="ts-progress-spinner" />
                          <span>Carregando histórico...</span>
                        </div>
                      ) : historyEntries.length === 0 ? (
                        <div className="ts-history-empty">
                          <div className="ts-history-empty-icon">📋</div>
                          <h3>Nenhum treinamento anterior</h3>
                          <p>Execute um treinamento para salvar no histórico.</p>
                          <button 
                            type="button" 
                            className="ts-btn-primary"
                            onClick={() => setActiveTab("config")}
                          >
                            Configurar Treinamento
                          </button>
                        </div>
                      ) : (
                        <div className="ts-history-grid">
                          {historyEntries.map((entry) => (
                            <HistoryCard
                              key={`${entry.stepId}-${entry.id}`}
                              entry={entry}
                              onClick={() => handleViewHistoryEntry(entry)}
                              onDelete={() => deleteHistoryEntry(entry.stepId, entry.id)}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  )
                )}
              </div>
        </div>
      </div>
      
      {/* Error */}
      {errorMessage && (
        <div className="ts-error">
          <span>{errorMessage}</span>
          <button onClick={() => setError("")}>×</button>
        </div>
      )}
      
      {/* Footer */}
      <div className="ts-footer">
        <label className="ts-cache-toggle" title="Limpar cache e reprocessar todos os experimentos do zero">
          <input 
            type="checkbox" 
            checked={invalidateCache} 
            onChange={(e) => setInvalidateCache(e.target.checked)}
            disabled={running}
          />
          <span className="ts-cache-toggle-icon">🔄</span>
          <span className="ts-cache-toggle-label">Limpar cache</span>
        </label>
        <div className="ts-footer-buttons">
          <button type="button" className="ts-btn-secondary" onClick={onClose} disabled={running}>
            Fechar
          </button>
          {running ? (
            <button
              type="button"
              className="ts-btn-danger"
              onClick={cancelTraining}
            >
              ✕ Cancelar
            </button>
          ) : (
            <button
              type="button"
              className="ts-btn-primary"
              disabled={!selectedDatasets || selectedDatasets.length === 0 || enabledModels.length === 0}
              onClick={runTraining}
            >
              Treinar {enabledModels.length} modelo{enabledModels.length !== 1 ? "s" : ""}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
