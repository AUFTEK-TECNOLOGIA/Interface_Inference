export const TRAINING_ALGORITHM_ITEMS = [
  { key: "ridge", label: "Ridge" },
  { key: "rf", label: "Random Forest" },
  { key: "gbm", label: "Gradient Boosting" },
  { key: "xgb", label: "XGBoost" },
  { key: "lgbm", label: "LightGBM" },
  { key: "catboost", label: "CatBoost" },
  { key: "svr", label: "SVR" },
  { key: "mlp", label: "MLP" },
  { key: "cnn", label: "CNN" },
  { key: "lstm", label: "LSTM" },
];

export const REGRESSION_ITEMS = [
  { key: "linear", label: "Linear", equation: "y = ax + b", description: "Regressão linear simples" },
  { key: "quadratic", label: "Quadrática", equation: "y = ax² + bx + c", description: "Polinômio de grau 2" },
  { key: "exponential", label: "Exponencial", equation: "y = a·eᵇˣ + c", description: "Crescimento/decaimento exponencial" },
  { key: "logarithmic", label: "Logarítmica", equation: "y = a·ln(x) + b", description: "Crescimento logarítmico" },
  { key: "power", label: "Potência", equation: "y = a·xᵇ + c", description: "Lei de potência" },
  { key: "polynomial", label: "Polinomial", equation: "y = aₙxⁿ + ... + a₀", description: "Polinômio de grau N" },
];

export const TRAINING_ALGO_PARAM_SCHEMA = {
  ridge: { fields: [{ key: "alpha", label: "alpha", kind: "float", step: 0.1, min: 0, defaultValue: 1.0, grid: true, gridHint: { min: 0.1, max: 10.0, divisions: 3 } }] },
  rf: { fields: [
    { key: "n_estimators", label: "n_estimators", kind: "int", step: 50, min: 10, defaultValue: 300, grid: true, gridHint: { min: 200, max: 600, divisions: 3 } },
    { key: "max_depth", label: "max_depth", kind: "int", step: 1, min: 1, allowNull: true, defaultValue: null, grid: true, gridHint: { min: 4, max: 16, divisions: 4 } },
    { key: "min_samples_split", label: "min_samples_split", kind: "int", step: 1, min: 2, defaultValue: 2, grid: true, gridHint: { min: 2, max: 10, divisions: 5 } },
    { key: "min_samples_leaf", label: "min_samples_leaf", kind: "int", step: 1, min: 1, defaultValue: 1, grid: true, gridHint: { min: 1, max: 6, divisions: 6 } },
    { key: "max_features", label: "max_features", kind: "float", step: 0.05, min: 0.05, defaultValue: 1.0, grid: true, gridHint: { min: 0.3, max: 1.0, divisions: 4 } },
    { key: "bootstrap", label: "bootstrap", kind: "select", options: ["true", "false"], defaultValue: "true", grid: true },
    { key: "n_jobs", label: "n_jobs", kind: "int", step: 1, min: -1, defaultValue: -1, grid: false },
  ] },
  gbm: { fields: [
    { key: "n_estimators", label: "n_estimators", kind: "int", step: 50, min: 10, defaultValue: 400, grid: true, gridHint: { min: 200, max: 600, divisions: 3 } },
    { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.01, min: 0.0001, defaultValue: 0.03, grid: true, gridHint: { min: 0.01, max: 0.1, divisions: 3 } },
    { key: "max_depth", label: "max_depth", kind: "int", step: 1, min: 1, defaultValue: 3, grid: true, gridHint: { min: 2, max: 5, divisions: 4 } },
    { key: "subsample", label: "subsample", kind: "float", step: 0.05, min: 0.05, defaultValue: 1.0, grid: true, gridHint: { min: 0.6, max: 1.0, divisions: 5 } },
    { key: "min_samples_split", label: "min_samples_split", kind: "int", step: 1, min: 2, defaultValue: 2, grid: true, gridHint: { min: 2, max: 10, divisions: 5 } },
    { key: "min_samples_leaf", label: "min_samples_leaf", kind: "int", step: 1, min: 1, defaultValue: 1, grid: true, gridHint: { min: 1, max: 6, divisions: 6 } },
    { key: "max_features", label: "max_features", kind: "float", step: 0.05, min: 0.05, allowNull: true, defaultValue: null, grid: true, gridHint: { min: 0.3, max: 1.0, divisions: 4 } },
  ] },
  xgb: { fields: [
    { key: "n_estimators", label: "n_estimators", kind: "int", step: 50, min: 10, defaultValue: 300, grid: true, gridHint: { min: 100, max: 600, divisions: 6 } },
    { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.01, min: 0.0001, defaultValue: 0.05, grid: true, gridHint: { min: 0.01, max: 0.2, divisions: 5 } },
    { key: "max_depth", label: "max_depth", kind: "int", step: 1, min: 1, defaultValue: 4, grid: true, gridHint: { min: 2, max: 10, divisions: 9 } },
    { key: "subsample", label: "subsample", kind: "float", step: 0.05, min: 0.05, defaultValue: 0.9, grid: true, gridHint: { min: 0.5, max: 1.0, divisions: 6 } },
    { key: "colsample_bytree", label: "colsample_bytree", kind: "float", step: 0.05, min: 0.05, defaultValue: 0.8, grid: true, gridHint: { min: 0.5, max: 1.0, divisions: 6 } },
    { key: "gamma", label: "gamma", kind: "float", step: 0.1, min: 0.0, defaultValue: 0.0, grid: true, gridHint: { min: 0.0, max: 2.0, divisions: 5 } },
  ] },
  lgbm: { fields: [
    { key: "n_estimators", label: "n_estimators", kind: "int", step: 50, min: 10, defaultValue: 400, grid: true, gridHint: { min: 100, max: 600, divisions: 6 } },
    { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.01, min: 0.0001, defaultValue: 0.05, grid: true, gridHint: { min: 0.01, max: 0.2, divisions: 5 } },
    { key: "num_leaves", label: "num_leaves", kind: "int", step: 4, min: 2, defaultValue: 31, grid: true, gridHint: { min: 15, max: 63, divisions: 5 } },
    { key: "max_depth", label: "max_depth", kind: "int", step: 1, min: -1, defaultValue: -1, grid: true, gridHint: { min: -1, max: 15, divisions: 5 } },
    { key: "subsample", label: "subsample", kind: "float", step: 0.05, min: 0.05, defaultValue: 1.0, grid: true, gridHint: { min: 0.6, max: 1.0, divisions: 5 } },
    { key: "colsample_bytree", label: "colsample_bytree", kind: "float", step: 0.05, min: 0.05, defaultValue: 1.0, grid: true, gridHint: { min: 0.6, max: 1.0, divisions: 5 } },
  ] },
  catboost: { fields: [
    { key: "iterations", label: "iterations", kind: "int", step: 50, min: 10, defaultValue: 500, grid: true, gridHint: { min: 200, max: 800, divisions: 4 } },
    { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.01, min: 0.0001, defaultValue: 0.03, grid: true, gridHint: { min: 0.01, max: 0.1, divisions: 3 } },
    { key: "depth", label: "depth", kind: "int", step: 1, min: 1, defaultValue: 6, grid: true, gridHint: { min: 4, max: 10, divisions: 4 } },
    { key: "l2_leaf_reg", label: "l2_leaf_reg", kind: "float", step: 0.5, min: 0.0, defaultValue: 3.0, grid: true, gridHint: { min: 1.0, max: 6.0, divisions: 4 } },
  ] },
  svr: { fields: [
    { key: "kernel", label: "kernel", kind: "select", options: ["rbf", "linear", "poly", "sigmoid"], defaultValue: "rbf", grid: true },
    { key: "C", label: "C", kind: "float", step: 0.5, min: 0.0, defaultValue: 1.0, grid: true, gridHint: { min: 0.5, max: 4.0, divisions: 4 } },
    { key: "epsilon", label: "epsilon", kind: "float", step: 0.05, min: 0.0, defaultValue: 0.1, grid: true, gridHint: { min: 0.05, max: 0.3, divisions: 4 } },
    { key: "gamma", label: "gamma", kind: "float", step: 0.05, min: 0.0, defaultValue: 0.1, grid: true, gridHint: { min: 0.01, max: 1.0, divisions: 6 } },
  ] },
  mlp: { fields: [
    { key: "hidden_layer_sizes", label: "hidden_layer_sizes", kind: "text", placeholder: "Optional, e.g. 128,64", defaultValue: "", grid: false },
    { key: "layers", label: "layers", kind: "int", step: 1, min: 1, defaultValue: 2, grid: true, gridHint: { min: 1, max: 5, divisions: 5 } },
    { key: "hidden", label: "hidden", kind: "int", step: 16, min: 16, defaultValue: 128, grid: true, gridHint: { min: 16, max: 512, divisions: 7 } },
    { key: "dropout", label: "dropout", kind: "float", step: 0.05, min: 0.0, defaultValue: 0.1, grid: true, gridHint: { min: 0.0, max: 0.5, divisions: 6 } },
    { key: "activation", label: "activation", kind: "select", options: ["relu", "tanh", "sigmoid"], defaultValue: "relu", grid: true },
    { key: "optimizer", label: "optimizer", kind: "select", options: ["adam", "sgd", "rmsprop"], defaultValue: "adam", grid: true },
    { key: "epochs", label: "epochs", kind: "int", step: 10, min: 1, defaultValue: 200, grid: true, gridHint: { min: 100, max: 500, divisions: 5 } },
    { key: "batch_size", label: "batch_size", kind: "int", step: 8, min: 1, defaultValue: 64, grid: true, gridHint: { min: 16, max: 128, divisions: 4 } },
    { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.0005, min: 0.00001, defaultValue: 0.001, grid: true, gridHint: { min: 0.0005, max: 0.005, divisions: 4 } },
  ] },
  cnn: { fields: [
    { key: "epochs", label: "epochs", kind: "int", step: 10, min: 1, defaultValue: 200, grid: true, gridHint: { min: 100, max: 500, divisions: 5 } },
    { key: "batch_size", label: "batch_size", kind: "int", step: 8, min: 1, defaultValue: 64, grid: true, gridHint: { min: 16, max: 128, divisions: 4 } },
    { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.0005, min: 0.00001, defaultValue: 0.001, grid: true, gridHint: { min: 0.0005, max: 0.005, divisions: 4 } },
  ] },
  lstm: { fields: [
    { key: "hidden_size", label: "hidden_size", kind: "int", step: 8, min: 4, defaultValue: 64, grid: true, gridHint: { min: 32, max: 128, divisions: 4 } },
    { key: "num_layers", label: "num_layers", kind: "int", step: 1, min: 1, defaultValue: 1, grid: true, gridHint: { min: 1, max: 3, divisions: 3 } },
    { key: "dropout", label: "dropout", kind: "float", step: 0.05, min: 0.0, defaultValue: 0.0, grid: true, gridHint: { min: 0.0, max: 0.5, divisions: 6 } },
    { key: "epochs", label: "epochs", kind: "int", step: 10, min: 1, defaultValue: 200, grid: true, gridHint: { min: 100, max: 500, divisions: 5 } },
    { key: "batch_size", label: "batch_size", kind: "int", step: 8, min: 1, defaultValue: 64, grid: true, gridHint: { min: 16, max: 128, divisions: 4 } },
    { key: "learning_rate", label: "learning_rate", kind: "float", step: 0.0005, min: 0.00001, defaultValue: 0.001, grid: true, gridHint: { min: 0.0005, max: 0.005, divisions: 4 } },
  ] },
};

export const parseExperimentIdsText = (value) => {
  const raw = String(value || "").trim();
  if (!raw) return [];
  return Array.from(new Set(raw.split(/[\s,;]+/g).map((t) => t.trim()).filter(Boolean)));
};

export const buildTrainingParamsForAlgorithm = (algorithmKey) => {
  const algo = String(algorithmKey || "ridge").trim().toLowerCase() || "ridge";
  const schema = TRAINING_ALGO_PARAM_SCHEMA[algo] || TRAINING_ALGO_PARAM_SCHEMA.ridge;
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
