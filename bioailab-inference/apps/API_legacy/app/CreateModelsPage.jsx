// src/pages/MachineLearning/CreateForecasterPage.jsx
import React, {
  useState, useEffect, useRef, useCallback, useMemo, memo,
} from 'react';

import {
  Box, Button, Card, CardActionArea, CardContent, Checkbox, Chip, Container, Dialog,
  DialogActions, DialogContent, DialogTitle, Divider, FormControl, FormControlLabel,
  Grid, InputLabel, MenuItem, Paper, Select, Stack, Stepper, Step, StepLabel, Switch,
  Tabs, Tab, TextField, Typography, TableContainer, Table, TableHead, TableRow,
  TableCell, TableBody, Autocomplete, CircularProgress, Backdrop, Alert, FormGroup
} from '@mui/material';

import { useNavigate } from 'react-router-dom';

// Icons
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents';

// Chart.js
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip as ChartTooltip, Legend, BarElement
} from 'chart.js';

// API Calls
import { getProtocolos } from '../../infra/api/Protocols';
import { getExperimentos, getExperimentosDados } from '../../infra/api/ExperimentoApi';

// ──────────────────────────────────────────────────────────────────────────────
// Constantes e Configurações
// ──────────────────────────────────────────────────────────────────────────────
const API_BASE = 'http://127.0.0.1:8000';

const palette = ['#1976d2', '#d32f2f', '#388e3c', '#f9a825', '#5e35b1', '#00838f', '#6a1b9a', '#afb42b'];
ChartJS.register(BarElement, CategoryScale, LinearScale, PointElement, LineElement, Title, ChartTooltip, Legend);

const SENSOR_INFO = {
  VIS1: { key: 'VIS1', display: 'Nefelometria' },
  VIS2: { key: 'VIS2', display: 'Turbidimetria' },
  UV: { key: 'UV', display: 'Fluorescência' },
};

const SENSOR_BLOCK_MAP = { UV: 'spectral_uv', VIS1: 'spectral_vis_1', VIS2: 'spectral_vis_2' };

// === RAW bands e detectores ===
const RAW_BANDS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'CLR', 'NIR'];
const isRawBand = (c) => RAW_BANDS.includes(String(c).toUpperCase());
const isDerived = (c) => ['R', 'G', 'B', 'C', 'M', 'Y', 'K', 'H', 'S', 'V'].includes(String(c).toUpperCase());

// === Matriz VIS (mesma do backend) para gerar cores ===
const XYZ_VIS = [
  [0.39814, 1.29540, 0.36956, 0.10902, 0.71942, 1.78180, 1.10110, -0.03991, -0.27597, -0.02347],
  [0.01396, 0.16748, 0.23538, 1.42750, 1.88670, 1.14200, 0.46497, -0.02702, -0.24468, -0.01993],
  [1.95010, 6.45490, 2.78010, 0.18501, 0.15325, 0.09539, 0.10563, 0.08866, -0.61140, -0.00938],
];

const CHANNEL_GROUPS = [
  { title: 'RGB', items: ['R', 'G', 'B'] },
  { title: 'CMYK', items: ['C', 'M', 'Y', 'K'] },
  { title: 'HSV', items: ['H', 'S', 'V'] },
  { title: 'ABS', items: [...Array.from({ length: 8 }, (_, i) => `F${i + 1}`), 'CLR', 'NIR'] },
];

// === Normaliza bloco bruto do sensor -> basic counts (mesma fórmula do back) ===
function normalizeBlockByCfg(rawBlock = {}, cfg) {
  const g = as7341Gain(cfg?.gain);           // 0.5…512
  const tUs = as7341TintUs(cfg);             // microssegundos
  const f = (g > 0 && tUs > 0) ? (g * tUs) : 1.0;
  const out = {};
  RAW_BANDS.forEach(b => {
    const arr = rawBlock?.[b] || [];
    out[b] = Array.isArray(arr) ? arr.map(v => Number(v) / f) : [];
  });
  return out;
}

// === Helpers de cor (espelha o backend) ===
const dot = (a, b) => a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0);
function spectrumToXYZ(spec, M = XYZ_VIS) {
  const x = dot(M[0], spec), y = dot(M[1], spec), z = dot(M[2], spec);
  const tot = x + y + z;
  if (tot > 0) return [x / tot, y / tot, z / tot];
  return [0, 0, 0];
}
function xyzToRGB([x, y, z]) {
  const r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
  const g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
  const b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
  return [r, g, b];
}
function rgbToCMYK({ R, G, B }) {
  const max = Math.max(R, G, B);
  const K = 1 - max;
  if (K < 1) return { C: (1 - R - K) / (1 - K), M: (1 - G - K) / (1 - K), Y: (1 - B - K) / (1 - K), K };
  return { C: 0, M: 0, Y: 0, K: 1 };
}
function rgbToHSV({ R, G, B }) {
  const max = Math.max(R, G, B), min = Math.min(R, G, B), d = max - min;
  let h = 0;
  if (d) { if (max === R) h = ((G - B) / d) % 6; else if (max === G) h = (B - R) / d + 2; else h = (R - G) / d + 4; h *= 60; }
  const s = max === 0 ? 0 : d / max;
  return { H: h, S: s, V: max };
}

// === Gera apenas os canais derivados pedidos a partir dos basic counts ===
function buildDerivedFromNormalized(normBlock, wanted = []) {
  const need = new Set((wanted || []).map(c => c.toUpperCase()).filter(isDerived));
  if (!need.size) return {};
  const T = (normBlock?.F1 || []).length;

  const needRGB = ['R', 'G', 'B', 'C', 'M', 'Y', 'K', 'H', 'S', 'V'].some(x => need.has(x));
  const out = {};
  let R = null, G = null, B = null;

  if (needRGB) {
    R = new Array(T); G = new Array(T); B = new Array(T);
    for (let i = 0; i < T; i++) {
      const spec = RAW_BANDS.map(b => Number(normBlock?.[b]?.[i] ?? 0));
      const xyz = spectrumToXYZ(spec, XYZ_VIS);
      const [r, g, b] = xyzToRGB(xyz);
      R[i] = r; G[i] = g; B[i] = b;
    }
    if (need.has('R')) out.R = R;
    if (need.has('G')) out.G = G;
    if (need.has('B')) out.B = B;
  }
  if (['C', 'M', 'Y', 'K'].some(x => need.has(x))) {
    const C = new Array(T), M = new Array(T), Y = new Array(T), K = new Array(T);
    for (let i = 0; i < T; i++) {
      const cmyk = rgbToCMYK({ R: R[i], G: G[i], B: B[i] });
      C[i] = cmyk.C; M[i] = cmyk.M; Y[i] = cmyk.Y; K[i] = cmyk.K;
    }
    if (need.has('C')) out.C = C;
    if (need.has('M')) out.M = M;
    if (need.has('Y')) out.Y = Y;
    if (need.has('K')) out.K = K;
  }
  if (['H', 'S', 'V'].some(x => need.has(x))) {
    const H = new Array(T), S = new Array(T), V = new Array(T);
    for (let i = 0; i < T; i++) {
      const hsv = rgbToHSV({ R: R[i], G: G[i], B: B[i] });
      H[i] = hsv.H; S[i] = hsv.S; V[i] = hsv.V;
    }
    if (need.has('H')) out.H = H;
    if (need.has('S')) out.S = S;
    if (need.has('V')) out.V = V;
  }
  return out;
}

const FCT_SPECS = [
  ['fct_hidden', 'Hidden units', 16, 2048, 16],
  ['fct_layers', 'LSTM layers', 1, 10, 1],
  ['fct_dropout', 'Dropout', 0, 0.50, 0.05],
  ['fct_window', 'Window', 1, 1440, 180],
  ['fct_horizon', 'Horizon', 1, 1440, 180],
  ['target_hours', 'Duração (h)', 1, 24, 1],
  ['epochs_fc', 'Epochs', 5, 30, 5],
  ['batch_fc', 'Batch size', 8, 512, 16],
  ['lr', 'Learning rate', 0.00001, 0.01, 0.0005],
];

const defaultFct = Object.fromEntries(FCT_SPECS.map(([k, , min]) => [k, String(min)]));
defaultFct.fct_layers = '2';
defaultFct.fct_dropout = '0.2';
defaultFct.fct_window = '180';
defaultFct.fct_horizon = '60';
defaultFct.target_hours = '24';
defaultFct.epochs_fc = '5';
defaultFct.batch_fc = '128';
defaultFct.lr = '0.001';

// Nome usado no backend
export const REG_NAME_MAP = { random_forest: 'rf', svr: 'svr', xgb: 'xgb', mlp: 'mlp' };
export const REG_MODELS = ['random_forest', 'svr', 'xgb', 'mlp'];
export const REG_LABEL = {
  random_forest: 'Random Forest',
  svr: 'SVR',
  xgb: 'XGBoost',
  mlp: 'MLP',
};


// Hiperparâmetros completos restaurados
export const REG_SPECS = {
  random_forest: [
    ['n_estimators', 'Árvores', 10, 500, 10],
    ['max_depth', 'Profundidade máx.', 1, 50, 1],
    ['min_samples_split', 'Min split', 2, 20, 1],
    ['min_samples_leaf', 'Min leaf', 1, 20, 1],
    ['max_features', 'Máx. features', 0.1, 1.0, 0.05],
    ['bootstrap', 'Bootstrap', 0, 1, 1], // 0 = False, 1 = True
  ],
  svr: [
    ['C', 'C (reg.)', 0.01, 100.0, 0.1],
    ['epsilon', 'Épsilon', 0.001, 1.0, 0.001],
    ['gamma', 'Gamma', 0.0001, 1.0, 0.0001],
    ['kernel', 'Kernel (codificado)', 0, 3, 1], // 0=linear, 1=poly, 2=rbf, 3=sigmoid
  ],
  xgb: [
    ['n_estimators', 'Árvores', 10, 500, 10],
    ['learning_rate', 'LR', 0.001, 1.0, 0.01],
    ['max_depth', 'Profundidade', 1, 50, 1],
    ['subsample', 'Subsample', 0.1, 1.0, 0.05],
    ['colsample_bytree', 'Cols/Árvore', 0.1, 1.0, 0.05],
    ['gamma', 'Gamma', 0, 10, 0.1],
    ['reg_alpha', 'Reg. L1 (alpha)', 0, 1.0, 0.05],
    ['reg_lambda', 'Reg. L2 (lambda)', 0, 1.0, 0.05],
  ],
  mlp: [
    ['layers', 'Camadas', 1, 5, 1],
    ['hidden', 'Unidades', 16, 512, 16],
    ['dropout', 'Dropout', 0, 0.5, 0.05],
    ['learning_rate', 'LR', 0.0001, 1.0, 0.0001],
    ['batch_size', 'Batch Size', 8, 512, 8],
    ['epochs', 'Epochs', 10, 300, 10],
    ['activation', 'Ativação', 0, 2, 1],
    ['optimizer', 'Otimizador', 0, 2, 1],
  ],
};

const SVR_KERNELS = ['linear', 'poly', 'rbf', 'sigmoid'];

const MLP_ACTIVATIONS = ['relu', 'tanh', 'sigmoid'];
const MLP_OPTIMIZERS = ['adam', 'sgd', 'rmsprop'];

export const REG_DEFAULTS = {
  random_forest: {
    n_estimators: 200, max_depth: 12, min_samples_split: 2, min_samples_leaf: 1,
    max_features: 1.0, bootstrap: 1,
  },
  svr: { C: 2.0, epsilon: 0.1, gamma: 0.1, kernel: 'rbf' },
  xgb: {
    n_estimators: 300, learning_rate: 0.05, max_depth: 4, subsample: 0.9,
    colsample_bytree: 0.8, gamma: 0.1, reg_alpha: 0.0, reg_lambda: 1.0,
  },
  mlp: {
    layers: 2, hidden: 128, dropout: 0.1, learning_rate: 0.001, batch_size: 64,
    epochs: 100, activation: 'relu', optimizer: 'adam',
  },
};

// Modelos disponíveis (frontend)
export const MODEL_OPTIONS = ['mlp', 'cnn', 'lstm', 'gbm', 'random_forest', 'svr', 'xgb'];
export const MODEL_LABEL_PT = {
  mlp: 'Perceptron Multicamadas (MLP)',
  cnn: 'Rede Convolucional (CNN)',
  lstm: 'Long Short-Term Memory (LSTM)',
  gbm: 'Boosting de Gradiente (GBM)',
  random_forest: 'Floresta Aleatória (RF)',
  svr: 'Vetores de Suporte (SVR)',
  xgb: 'XGBoost',
};
export const modelLabel = (key) => MODEL_LABEL_PT[key] || String(key).toUpperCase();

// Backend atual
export const SUPPORTED_BY_BACKEND = new Set(['random_forest', 'svr', 'xgb', 'mlp']);
// Usado nos steps antigos
export const HYPER_SPECS = REG_SPECS;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers puros (reutilizados em vários passos)
// ──────────────────────────────────────────────────────────────────────────────
const chipSX = (lbl, sel) => {
  const pal = {
    R: '#c62828', G: '#2e7d32', B: '#1565c0', C: '#00838f', M: '#ad1457',
    Y: '#f9a825', K: '#212121', H: '#5e35b1', S: '#6a1b9a', V: '#558b2f'
  };
  const fg = pal[lbl] || '#424242';
  const bg = sel ? fg : `${fg}20`;
  return { mx: .5, my: .5, bgcolor: bg, color: sel ? '#fff' : fg, fontWeight: 600 };
};

const clamp = (x, min, max) => {
  const n = Number(x);
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, n));
};

function rangeFrom(start, end, step) {
  const arr = [];
  for (let v = start; v <= end; v += step) arr.push(Number(v.toFixed(6)));
  return arr;
}

function pivotBlock(readings, blk) {
  const out = {}; if (!readings?.length) return out;
  const keys = Object.keys(readings[0]?.[blk] || {});
  keys.forEach(k => { out[k.toUpperCase()] = []; });
  for (const r of readings) {
    const b = r?.[blk]; if (!b) continue;
    for (const k of keys) out[k.toUpperCase()].push(Number(b[k] ?? 0));
  }
  return out;
}

function buildEnsaioFromCRM(meta, frames) {
  return {
    experiment_UUID: meta.id,
    timestamps: frames.map(f => f.timestamp ?? 0),
    light_sensor_config: meta.light_sensor_config ?? {},
    spectral_uv: pivotBlock(frames, 'spectral_uv'),
    spectral_vis_1: pivotBlock(frames, 'spectral_vis_1'),
    spectral_vis_2: pivotBlock(frames, 'spectral_vis_2'),
  };
}
const SENSOR_CFG_KEY = { UV: 'uv', VIS1: 'visible_1', VIS2: 'visible_2' };

// AS7341: enum de ganho e integração em microssegundos (2.78 µs por passo)
const AS7341_GAIN = { 0: 0.5, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 256, 10: 512 };
const as7341Gain = (g) => AS7341_GAIN[Number(g)] ?? 1.0;
const as7341TintUs = (cfg) => {
  if (!cfg) return 1.0;
  const atime = Number(cfg.atime ?? cfg.ATIME ?? 0);
  const astep = Number(cfg.astep ?? cfg.ASTEP ?? 0);
  // (ATIME+1)*(ASTEP+1)*2.78  →  microssegundos
  return (atime + 1) * (astep + 1) * 2.78;
};

/**
 * Aplica o ganho e tempo de integração inversos à previsão para retornar
 * a uma escala similar aos "counts" brutos do sensor.
 * Isso é útil para visualização, já que a escala dos "basic counts" (pós-normalização)
 * é muito pequena (ex: 1e-5).
 */
function applyInverseGainToForecast(yForecast, ensaio, channelsMap) {
  if (!yForecast || !ensaio?.light_sensor_config || !channelsMap) return yForecast;

  // pré-calcula fator por sensor
  const factorBySensor = {};
  for (const sensor of Object.keys(channelsMap)) {
    const cfgKey = SENSOR_CFG_KEY[sensor] ?? sensor.toLowerCase();
    const cfg = ensaio.light_sensor_config?.[cfgKey];
    const g = as7341Gain(cfg?.gain);
    const tUs = as7341TintUs(cfg);
    let f = (g > 0 && tUs > 0) ? (g * tUs) : 1.0;
    if (!Number.isFinite(f) || f <= 0) f = 1.0;
    factorBySensor[sensor] = f;

    console.log(`[applyInverseGain] ${sensor} | ${cfgKey} | gain=${cfg?.gain} → ${g} | atime=${cfg?.atime} | astep=${cfg?.astep} | tUs=${tUs} | factor=${f.toExponential(4)}`);
  }

  // aplica só em bandas RAW; derivados ficam sem escala adicional
  const out = [];
  let idx = 0;
  for (const [sensor, chans] of Object.entries(channelsMap)) {
    const f = factorBySensor[sensor] ?? 1.0;
    for (const ch of (chans || [])) {
      const U = String(ch).toUpperCase();
      const serie = yForecast[idx] ?? [];
      const scale = isRawBand(U) ? f : 1.0;
      out.push(Array.isArray(serie) ? serie.map(v => (typeof v === 'number' ? v * scale : v)) : serie);
      idx += 1;
    }
  }
  return out.length ? out : yForecast;
}

// Recorta o ensaio até o índice "sliceStart" (exclusivo)
function sliceEnsaio(ensaio, sliceStart) {
  const n = Math.max(0, Math.min(Number(sliceStart) || 0, (ensaio?.timestamps?.length || 0)));
  const out = {
    ...ensaio,
    timestamps: (ensaio.timestamps || []).slice(0, n),
    spectral_uv: {}, spectral_vis_1: {}, spectral_vis_2: {},
  };
  ['spectral_uv', 'spectral_vis_1', 'spectral_vis_2'].forEach(blk => {
    const src = ensaio?.[blk] || {};
    Object.keys(src).forEach(k => { out[blk][k] = (src[k] || []).slice(0, n); });
  });
  return out;
}

function parseCheckpointName(filename = '') {
  const params = { raw: filename };
  try {
    const match = filename.match(/h(?<h>\d+)_l(?<l>\d+)_d(?<d>[\d.p]+)_w(?<w>\d+)_hor(?<hor>\d+)/);
    if (!match?.groups) return params;
    const g = match.groups;
    return { ...params, h: +g.h, l: +g.l, d: +g.d.replace('p', '.'), w: +g.w, hor: +g.hor };
  } catch {
    return params;
  }
}

function getObservedDataWithDerived(ensaio, channelsMap = {}) {
  if (!ensaio || !Object.keys(channelsMap).length) {
    return { labels: [], series: [], timestamps: [] };
  }
  const labels = [];
  const series = [];
  const timestamps = ensaio.timestamps || [];

  for (const [sensor, chans] of Object.entries(channelsMap)) {
    const blockKey = SENSOR_BLOCK_MAP[sensor];
    const rawBlock = ensaio?.[blockKey] || {};                      // <- counts do sensor (bruto)
    const cfgKey = SENSOR_CFG_KEY[sensor] ?? sensor.toLowerCase();
    const cfg = ensaio?.light_sensor_config?.[cfgKey];

    // basic counts (apenas para gerar derivados corretamente)
    const normBlock = normalizeBlockByCfg(rawBlock, cfg);
    const derived = buildDerivedFromNormalized(normBlock, chans);

    for (const ch of (chans || [])) {
      const U = String(ch).toUpperCase();
      const arr = isRawBand(U)
        ? (rawBlock[U] || []).map(Number)   // <-- AQUI: RAW volta a ser counts brutos
        : isDerived(U)
          ? (derived[U] || [])
          : [];
      labels.push(`${sensor}:${U}`);
      series.push(arr);
    }
  }
  return { labels, series, timestamps };
}

function extractProtoId(exp) {
  if (!exp || typeof exp !== 'object') return undefined;
  let pid =
    exp.protocolId ?? exp.protocoloId ?? exp.protocol_id ?? exp.protocolo_id ??
    exp.protocol ?? exp.protocolo;
  if (pid === undefined && exp.general_info && typeof exp.general_info === 'object') {
    pid =
      exp.general_info.protocolId ?? exp.general_info.protocoloId ??
      exp.general_info.protocol_id ?? exp.general_info.protocolo_id ??
      exp.general_info.protocol ?? exp.general_info.protocolo;
  }
  if (typeof pid === 'object' && pid !== null) pid = pid.id ?? pid.uuid ?? pid._id ?? pid.value;
  return pid;
}

// monta ModelSpec p/ API /train_regressor (mantém comportamento)
function buildRegressorPayload(featureFile, regForm) {

  const NUM = (x) => {
    const n = Number(String(x).replace(',', '.'));
    return Number.isFinite(n) ? n : NaN;
  };

  const toMlpCat = (key, v) => {
    const table = key === 'activation' ? MLP_ACTIVATIONS : MLP_OPTIMIZERS;
    if (typeof v === 'number' && Number.isFinite(v)) {
      const idx = Math.max(0, Math.min(table.length - 1, Math.round(v)));
      return table[idx];
    }
    const s = String(v).toLowerCase();
    return table.includes(s) ? s : table[0];
  };

  const onlySupported = (regForm.models || []).filter(m =>
    SUPPORTED_BY_BACKEND.has(typeof m === 'string' ? m : m.name)
  );

  const modelSpecs = onlySupported.map(m => {
    const uiName = typeof m === 'string' ? m : m.name;
    const name = REG_NAME_MAP[uiName] || uiName;

    // GRID (tem prioridade se existir)
    const rawGrid = (typeof m === 'object' && m.grid) ? m.grid : regForm.grids?.[uiName];
    let grid = null;
    if (rawGrid) {
      grid = {};
      for (const [k, vals] of Object.entries(rawGrid)) {
        if (k === '__ui') continue;

        // SVR.kernel é categórico: manter strings
        if (uiName === 'svr' && k === 'kernel') {
          const arr = (vals || []).map(String).filter(v => SVR_KERNELS.includes(v));
          if (arr.length) grid[k] = arr;
          continue;
        }

        // MLP: activation/optimizer categóricos
        if (uiName === 'mlp' && (k === 'activation' || k === 'optimizer')) {
          const arr = (vals || []).map(v => toMlpCat(k, v)).filter(Boolean);
          if (arr.length) grid[k] = Array.from(new Set(arr));
          continue;
        }

        const arr = (vals || []).map(NUM).filter(Number.isFinite);
        if (arr.length) grid[k] = arr;
      }
      if (!Object.keys(grid).length) grid = null;
    }

    // PARAMS (se não houver grid)
    let params = null;
    if (!grid) {
      const srcParams = (typeof m === 'object' && m.params) ? m.params : (regForm.params?.[uiName] || {});
      const defParams = REG_DEFAULTS[uiName] || {};
      const merged = { ...defParams, ...srcParams };
      params = {};
      for (const [k, v] of Object.entries(merged)) {

        if (uiName === 'mlp' && (k === 'activation' || k === 'optimizer')) {
          params[k] = toMlpCat(k, v);
          continue;
        }


        const n = NUM(v);
        params[k] = Number.isFinite(n) ? n : merged[k];
      }
    }
    return grid ? { name, grid } : { name, params };
  });

  return {
    featureFile,
    models: modelSpecs,          // <- SUA API espera objetos {name, params|grid}
    testSize: Number(regForm.testSize ?? 0.2),
    permImp: !!regForm.permImp,
  };
}

// Map de chaves antigas → novas (payload forecaster)
const keyMap = {
  fct_hidden: 'hiddenUnits', fct_layers: 'layers', fct_dropout: 'dropout',
  fct_window: 'window', fct_horizon: 'horizon', epochs_fc: 'epochs',
  batch_fc: 'batchSize', lr: 'learningRate', fct_bidirectional: 'bidirectional',
  target_hours: 'targetLen',
};

function buildNewPayload(form) {
  return {
    experiment: {
      sensors: Object.entries(form.channelMap).map(([sensor, chans]) => `${sensor}:${chans.join(',')}`),
      units: form.units,
    },
    data: {
      jsonFile: form.localFile ? `data/${form.localFile.name}` : null,
      directory: 'data',
      slice: { start: Number(form.slice_start), end: Number(form.slice_end) },
    },
    model: {
      architecture: 'lstm',
      hiddenUnits: Number(form.fct.fct_hidden),
      layers: Number(form.fct.fct_layers),
      dropout: Number(form.fct.fct_dropout),
      bidirectional: form.fct.fct_bidirectional,
      window: Number(form.fct.fct_window),
      horizon: Number(form.fct.fct_horizon),
      targetLen: Number(form.fct.target_hours) * 60,
    },
    training: {
      epochs: Number(form.fct.epochs_fc),
      batchSize: Number(form.fct.batch_fc),
      learningRate: Number(form.fct.lr),
      testSize: Number(form.test_size) / 100,
    },
    paramSearch: {
      enabled: !!Object.keys(form.fct_grid).length,
      grid: Object.fromEntries(
        Object.entries(form.fct_grid).map(([oldKey, values]) => [
          keyMap[oldKey] || oldKey,
          values
        ])
      )
    }
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Componentes de UI
// ──────────────────────────────────────────────────────────────────────────────
const NumInput = memo(({ label, value, onChange, onBlur }) => (
  <TextField
    fullWidth size="small" label={label} value={value}
    inputMode="decimal"
    onChange={onChange}
    onBlur={onBlur}
    onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); }}
  />
));

const ChannelPicker = memo(({ open, sensor, init, onClose, onSave }) => {
  const [sel, setSel] = useState(init);
  const [q, setQ] = useState('');
  useEffect(() => { if (open) { setSel(init); setQ(''); } }, [open, init]);

  const match = txt => o => o.toLowerCase().includes(txt);
  const groups = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return CHANNEL_GROUPS;
    return CHANNEL_GROUPS
      .map(g => ({ ...g, items: g.items.filter(match(s)) }))
      .filter(g => g.items.length);
  }, [q]);

  const toggle = c => setSel(s => s.includes(c) ? s.filter(x => x !== c) : [...s, c]);
  const toggleGroup = items => {
    const all = items.every(c => sel.includes(c));
    setSel(s => {
      const set = new Set(s);
      items.forEach(c => all ? set.delete(c) : set.add(c));
      return Array.from(set);
    });
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm" scroll="paper">
      <DialogTitle>Canais — {SENSOR_INFO[sensor]?.display}</DialogTitle>
      <DialogContent dividers sx={{ pt: 2 }}>
        <TextField fullWidth size="small" placeholder="Filtrar…" value={q}
          onChange={e => setQ(e.target.value)} sx={{ mb: 2 }} />
        <Box sx={{ mb: 2, p: 1, border: '1px solid', borderColor: 'divider', borderRadius: 1, bgcolor: 'background.default' }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="subtitle2">Selecionados ({sel.length})</Typography>
            <Stack direction="row" spacing={1}>
              <Button size="small" onClick={() => setSel(CHANNEL_GROUPS.flatMap(g => g.items))}>Todos</Button>
              <Button size="small" onClick={() => setSel([])}>Limpar</Button>
            </Stack>
          </Stack>
          <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
            {sel.map(c => <Chip key={c} label={c} onDelete={() => toggle(c)} sx={chipSX(c, true)} />)}
            {!sel.length && <Typography variant="body2" color="text.secondary">Nenhum canal.</Typography>}
          </Box>
        </Box>
        <Stack spacing={2}>
          {groups.map(g => {
            const vis = g.items;
            const all = vis.every(c => sel.includes(c));
            const some = vis.some(c => sel.includes(c));
            return (
              <Paper key={g.title} variant="outlined" sx={{ p: 1.5 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="subtitle2">{g.title}</Typography>
                    <Chip size="small" label={all ? 'Todos' : (some ? 'Parcial' : 'Nenhum')}
                      variant={some ? 'filled' : 'outlined'}
                      color={all ? 'primary' : (some ? 'warning' : 'default')} />
                  </Stack>
                  <Button size="small" onClick={() => toggleGroup(vis)}>
                    {all ? 'Limpar grupo' : 'Marcar grupo'}
                  </Button>
                </Stack>
                <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
                  {vis.map(c => (
                    <Chip key={c} label={c} clickable onClick={() => toggle(c)}
                      sx={chipSX(c, sel.includes(c))} />
                  ))}
                </Box>
              </Paper>
            );
          })}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancelar</Button>
        <Button variant="contained" onClick={() => onSave(sel)}>Aplicar</Button>
      </DialogActions>
    </Dialog>
  );
});

// ──────────────────────────────────────────────────────────────────────────────
// Passo 1: Protocolos
// ──────────────────────────────────────────────────────────────────────────────
const StepProtocols = memo(({ protocols, form, onFormChange, onNext }) => {
  const unitOpts = useMemo(() => {
    const p = protocols.find(pp => pp.id === form.protocolId);
    return p ? [...new Set(p.calibrationSets?.map(c => c.unit) || [])] : [];
  }, [protocols, form.protocolId]);

  return (
    <Paper variant="outlined" sx={{ p: 3, mb: 2 }}>
      <Typography variant="h6" mb={2}>Protocolo</Typography>
      <Grid container spacing={2}>
        {protocols.map(p => (
          <Grid item xs={12} sm={6} md={4} key={p.id}>
            <Card sx={{ border: form.protocolId === p.id ? '2px solid #1976d2' : '' }}>
              <CardActionArea onClick={() => onFormChange('protocolId', p.id)}>
                <CardContent>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Checkbox checked={form.protocolId === p.id} readOnly />
                    <Box>
                      <Typography fontWeight={600}>{p.name}</Typography>
                      <Typography variant="body2" color="text.secondary">{p.description || 'Resumo do protocolo…'}</Typography>
                    </Box>
                  </Stack>
                </CardContent>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>
      {form.protocolId && (
        <FormControl fullWidth sx={{ mt: 2 }}>
          <InputLabel>Unidade</InputLabel>
          <Select
            value={form.units[0] || ''}
            label="Unidade"
            onChange={e => onFormChange('units', [e.target.value])}
          >
            {unitOpts.map(u => <MenuItem key={u} value={u}>{u}</MenuItem>)}
          </Select>
        </FormControl>
      )}
      <Stack direction="row" justifyContent="flex-end" sx={{ mt: 3 }}>
        <Button variant="contained" disabled={!form.protocolId || !form.units.length} onClick={onNext}>
          Próximo
        </Button>
      </Stack>
    </Paper>
  );
});

// ──────────────────────────────────────────────────────────────────────────────
// Passo 2: Dataset
// ──────────────────────────────────────────────────────────────────────────────
const StepDataset = memo(({ form, onFormChange, fileInputRef, onNext, onBack }) => (
  <Paper variant="outlined" sx={{ p: 3, mb: 2 }}>
    <Typography variant="h6" mb={2}>Dataset</Typography>
    <Stack direction="row" spacing={2} alignItems="center" mb={2}>
      <Button variant="contained" startIcon={<FolderOpenIcon />} onClick={() => fileInputRef.current.click()}>
        Selecionar JSON
      </Button>
      <input
        ref={fileInputRef} type="file" hidden accept=".json"
        onChange={e => onFormChange('localFile', e.target.files[0])}
      />
      {form.localFile && <Typography>{form.localFile.name}</Typography>}
    </Stack>
    <Grid container spacing={2}>
      {['test_size', 'slice_start', 'slice_end'].map((f, i) => (
        <Grid item xs={12} sm={4} key={f}>
          <TextField
            fullWidth inputMode="decimal"
            label={['Test %', 'Slice start', 'Slice end'][i]}
            value={form[f]}
            onChange={e => onFormChange(f, e.target.value)}
          />
        </Grid>
      ))}
    </Grid>
    <Stack direction="row" justifyContent="space-between" sx={{ mt: 3 }}>
      <Button startIcon={<ArrowBackIcon />} onClick={onBack}>Voltar</Button>
      <Button variant="contained" disabled={!form.localFile} onClick={onNext}>Próximo</Button>
    </Stack>
  </Paper>
));

// ──────────────────────────────────────────────────────────────────────────────
//— Passo 3: Sensores
// ──────────────────────────────────────────────────────────────────────────────
const StepSensors = memo(({ form, onFormChange, onNext, onBack }) => {
  const [pickerState, setPickerState] = useState({ open: false, sensor: null });
  const openPicker = s => setPickerState({ open: true, sensor: s });
  const closePicker = () => setPickerState({ open: false, sensor: null });

  const handleSavePicker = (channels) => {
    const newChannelMap = { ...form.channelMap, [pickerState.sensor]: channels };
    onFormChange('channelMap', newChannelMap);
    closePicker();
  };

  const handleSensorToggle = (key, checked) => {
    const nextSensors = checked ? [...form.sensors, key] : form.sensors.filter(x => x !== key);
    onFormChange('sensors', nextSensors);
    if (!checked) {
      const { [key]: _, ...rest } = form.channelMap;
      onFormChange('channelMap', rest);
    }
  };

  return (
    <Paper variant="outlined" sx={{ p: 3, mb: 2 }}>
      <Typography variant="h6" mb={2}>Sensores</Typography>
      <Box sx={{ mb: 2 }}>
        {Object.values(SENSOR_INFO).map(s => (
          <FormControlLabel
            key={s.key}
            label={s.display}
            control={
              <Checkbox
                checked={form.sensors.includes(s.key)}
                onChange={e => handleSensorToggle(s.key, e.target.checked)}
              />
            }
          />
        ))}
      </Box>
      {form.sensors.map(k => {
        const chans = form.channelMap[k] || [];
        return (
          <Stack key={k} direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
            <Typography sx={{ minWidth: 170 }}>{SENSOR_INFO[k].display} — Canais</Typography>
            <Box sx={{ flex: 1, display: 'flex', flexWrap: 'wrap', p: 1, border: 1, borderColor: 'divider', borderRadius: 1 }}>
              {chans.length ? chans.map(c => (
                <Chip
                  key={c} label={c}
                  onDelete={() => onFormChange('channelMap', { ...form.channelMap, [k]: chans.filter(x => x !== c) })}
                  sx={chipSX(c, true)}
                />
              )) : <Typography variant="body2" color="text.secondary">—</Typography>}
            </Box>
            <Button variant="outlined" onClick={() => openPicker(k)}>Editar…</Button>
          </Stack>
        );
      })}
      <Stack direction="row" justifyContent="space-between" sx={{ mt: 3 }}>
        <Button startIcon={<ArrowBackIcon />} onClick={onBack}>Voltar</Button>
        <Button
          variant="contained"
          disabled={!form.sensors.length || form.sensors.some(s => (form.channelMap[s] || []).length === 0)}
          onClick={onNext}
        >
          Próximo
        </Button>
      </Stack>
      <ChannelPicker
        open={pickerState.open}
        sensor={pickerState.sensor}
        init={form.channelMap[pickerState.sensor] || []}
        onClose={closePicker}
        onSave={handleSavePicker}
      />
    </Paper>
  );
});

// ──────────────────────────────────────────────────────────────────────────────
// Passo 4: Forecaster (hiperparâmetros)
// ──────────────────────────────────────────────────────────────────────────────
const StepForecaster = memo(({ form, onFormChange, loading, onExecute, onBack }) => {
  const [inputValues, setInputValues] = useState({});

  useEffect(() => {
    const newValues = {};
    FCT_SPECS.forEach(([key, , min, max, step]) => {
      const isGrid = key in form.fct_grid;
      if (isGrid) {
        const ui = form.fct_grid_ui?.[key] || { start: min, end: Math.min(min + 2 * step, max), step };
        newValues[`${key}-start`] = ui.start;
        newValues[`${key}-end`] = ui.end;
        newValues[`${key}-step`] = ui.step;
      } else {
        newValues[`${key}-single`] = form.fct[key];
      }
    });
    setInputValues(newValues);
  }, [form.fct, form.fct_grid, form.fct_grid_ui]);

  const handleInputChange = (id, value) => setInputValues(prev => ({ ...prev, [id]: value }));

  const handleInputBlur = (id, key, part) => {
    const rawValue = inputValues[id];
    onFormChange(null, prevForm => {
      const [, , min, max] = FCT_SPECS.find(s => s[0] === key);
      if (part === 'single') {
        const clampedValue = String(clamp(rawValue, min, max));
        const newFct = { ...prevForm.fct, [key]: clampedValue };
        return { ...prevForm, fct: newFct };
      } else {
        const ui = prevForm.fct_grid_ui[key] || {};
        let clampedValue;
        if (part === 'start') clampedValue = clamp(rawValue, min, ui.end);
        if (part === 'end') clampedValue = clamp(rawValue, ui.start, max);
        if (part === 'step') clampedValue = clamp(rawValue, Number.EPSILON, max);

        const nextUi = { ...ui, [part]: clampedValue };
        const nextGridValues = rangeFrom(nextUi.start, nextUi.end, nextUi.step);

        const newGrid = { ...prevForm.fct_grid, [key]: nextGridValues };
        const newUiState = { ...prevForm.fct_grid_ui, [key]: nextUi };
        return { ...prevForm, fct_grid: newGrid, fct_grid_ui: newUiState };
      }
    });
  };

  const toggleGrid = (key, use) => {
    onFormChange(null, o => {
      const g = { ...o.fct_grid };
      const ui = { ...(o.fct_grid_ui || {}) };
      if (use) {
        const [, , min, max, defStep] = FCT_SPECS.find(s => s[0] === key);
        const start = o.fct[key] ? Number(o.fct[key]) : min;
        const end = Math.min(start + 2 * defStep, max);
        const step = defStep;
        ui[key] = { start, end, step };
        g[key] = rangeFrom(start, end, step);
      } else {
        delete g[key];
        delete ui[key];
      }
      return { ...o, fct_grid: g, fct_grid_ui: ui };
    });
  };

  return (
    <>
      <Backdrop open={loading} sx={{ color: '#fff', zIndex: (t) => t.zIndex.drawer + 1, bgcolor: 'rgba(0, 0, 0, 0.8)' }}>
        <Stack alignItems="center" spacing={2}>
          <CircularProgress color="inherit" />
          <Typography>Treinando modelo… Por favor, aguarde.</Typography>
        </Stack>
      </Backdrop>

      <Paper variant="outlined" sx={{ p: 3, mb: 2 }}>
        <Typography variant="h6" mb={2}>Forecaster LSTM</Typography>
        {FCT_SPECS.map(([key, label, min, max, step]) => {
          const useGrid = key in form.fct_grid;
          return (
            <Box key={key}>
              <Grid container spacing={1} alignItems="center" sx={{ py: 1 }}>
                <Grid item xs={12} md={3}>
                  <Typography fontWeight={600}>{label}</Typography>
                  <Typography variant="caption" color="text.secondary">{useGrid ? 'Faixa + passo' : 'Valor único'}</Typography>
                </Grid>
                <Grid item xs={12} md={1} sx={{ display: 'flex', justifyContent: 'center' }}>
                  <Switch size="small" checked={useGrid} onChange={e => toggleGrid(key, e.target.checked)} />
                </Grid>
                {!useGrid ? (
                  <Grid item xs={12} md={4}>
                    <NumInput
                      label="Valor"
                      value={inputValues[`${key}-single`] ?? ''}
                      onChange={e => handleInputChange(`${key}-single`, e.target.value)}
                      onBlur={() => handleInputBlur(`${key}-single`, key, 'single')}
                    />
                  </Grid>
                ) : (
                  <>
                    <Grid item xs={12} md={2.5}>
                      <NumInput
                        label="Início"
                        value={inputValues[`${key}-start`] ?? ''}
                        onChange={e => handleInputChange(`${key}-start`, e.target.value)}
                        onBlur={() => handleInputBlur(`${key}-start`, key, 'start')}
                      />
                    </Grid>
                    <Grid item xs={12} md={2.5}>
                      <NumInput
                        label="Fim"
                        value={inputValues[`${key}-end`] ?? ''}
                        onChange={e => handleInputChange(`${key}-end`, e.target.value)}
                        onBlur={() => handleInputBlur(`${key}-end`, key, 'end')}
                      />
                    </Grid>
                    <Grid item xs={12} md={2.5}>
                      <NumInput
                        label="Passo"
                        value={inputValues[`${key}-step`] ?? ''}
                        onChange={e => handleInputChange(`${key}-step`, e.target.value)}
                        onBlur={() => handleInputBlur(`${key}-step`, key, 'step')}
                      />
                    </Grid>
                  </>
                )}
              </Grid>
              <Divider />
            </Box>
          );
        })}
        <FormControlLabel sx={{ mt: 2 }}
          control={<Switch checked={form.fct.fct_bidirectional} onChange={e => onFormChange('fct', { ...form.fct, fct_bidirectional: e.target.checked })} />}
          label="Bidirectional LSTM"
        />
        <Stack direction="row" justifyContent="space-between" sx={{ mt: 3 }}>
          <Button startIcon={<ArrowBackIcon />} onClick={onBack}>Voltar</Button>
          <Button variant="contained" disabled={loading} onClick={onExecute}>Executar</Button>
        </Stack>
      </Paper>
    </>
  );
});

// ──────────────────────────────────────────────────────────────────────────────
// Card de checkpoint (forecaster)
// ──────────────────────────────────────────────────────────────────────────────
const CheckpointCard = memo(({ trial, isBest, isSelected, onClick }) => {
  const params = useMemo(() => parseCheckpointName(trial.modelFile), [trial.modelFile]);
  return (
    <Card
      variant="outlined"
      sx={{
        mb: 1.5,
        border: isSelected ? '2px solid' : '1px solid',
        borderColor: isSelected ? 'primary.main' : 'divider',
        bgcolor: isSelected ? 'action.selected' : 'transparent',
        transition: 'all 0.2s ease-in-out',
        '&:hover': { borderColor: 'primary.light' }
      }}
    >
      <CardActionArea onClick={onClick} sx={{ p: 1.5 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="body2" fontWeight={600} title={trial.modelFile}>
            Trial #{trial.trialId}
          </Typography>
          {isBest && <Chip size="small" label="🏆 Melhor" color="warning" />}
        </Stack>
        <Grid container spacing={0.5}>
          {params.w && <Grid item><Chip variant="outlined" size="small" label={`w: ${params.w}`} /></Grid>}
          {params.hor && <Grid item><Chip variant="outlined" size="small" label={`h: ${params.hor}`} /></Grid>}
          {params.l && <Grid item><Chip variant="outlined" size="small" label={`l: ${params.l}`} /></Grid>}
          {params.d && <Grid item><Chip variant="outlined" size="small" label={`d: ${params.d}`} /></Grid>}
        </Grid>
        <Typography variant="caption" display="block" mt={1}>
          Train MSE: {trial.trainMse.toFixed(5)}
        </Typography>
      </CardActionArea>
    </Card>
  );
});

// ──────────────────────────────────────────────────────────────────────────────
// Passo 5: Resultados do Forecaster  (VOLTA/AVANÇAR NO RODAPÉ)
// ──────────────────────────────────────────────────────────────────────────────
const StepResults = memo(({ form, fctRes, onBack }) => {
  const [tab, setTab] = useState(0); // 0: Visão-geral, 1: Teste Rápido
  const [expSel, setExpSel] = useState(null);
  const [testJson, setTestJson] = useState('');
  const [sliceStart, setSliceStart] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const [expList, setExpList] = useState([]);
  const [fullObservedData, setFullObservedData] = useState(null);
  const [forecastResult, setForecastResult] = useState(null);

  const trials = useMemo(
    () => [...(fctRes?.trials || [])].map((t, i) => ({ ...t, trialId: i })).sort((a, b) => a.trainMse - b.trainMse),
    [fctRes]
  );
  const bestTrial = trials[0] || null;
  const [selectedTrial, setSelectedTrial] = useState(bestTrial || null);

  useEffect(() => {
    if (!form.protocolId) { setExpList([]); return; }
    (async () => {
      try {
        const all = await getExperimentos();
        const filtered = (all); // filtrar por protocolo se desejar
        if (!filtered.length) {
          const distinct = [...new Set((all || []).map(e => String(extractProtoId(e))))];
          if (distinct.length === 1 && (distinct[0] === 'undefined' || distinct[0] === 'null')) {
            setExpList(all || []);
          } else {
            setExpList(filtered);
          }
        } else {
          setExpList(filtered);
        }
      } catch (err) {
        console.error('Falha ao carregar experimentos:', err);
        setExpList([]);
      }
    })();
  }, [form.protocolId]);

  // Carrega dados ao selecionar experimento
  useEffect(() => {
    if (!expSel) {
      setFullObservedData(null);
      setForecastResult(null);
      return;
    }
    setIsLoading(true);
    setForecastResult(null);
    setFullObservedData(null);

    getExperimentosDados(expSel.id)
      .then(frames => {
        const sortedFrames = (frames || []).sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
        const ensaio = buildEnsaioFromCRM(expSel, sortedFrames);
        const dataReal = getObservedDataWithDerived(ensaio, fctRes?.channels);
        setFullObservedData(dataReal);
        const minLen = Math.min(...(dataReal.series || []).map(s => s?.length || 0));
        setSliceStart(Number.isFinite(minLen) ? Math.floor(minLen * 0.8) : 0);
      })
      .catch((err) => {
        console.error("Falha ao carregar dados do experimento:", err);
        setFullObservedData(null);
      })
      .finally(() => setIsLoading(false));
  }, [expSel, fctRes?.channels]);

  const handleRun = useCallback(async () => {
    if (!selectedTrial) return alert('Selecione um checkpoint');

    setIsLoading(true);
    let ensaioCompleto;

    try {
      if (testJson.trim()) {
        ensaioCompleto = JSON.parse(testJson);
      } else if (expSel) {
        const frames = await getExperimentosDados(expSel.id);
        const sortedFrames = (frames || []).sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
        ensaioCompleto = buildEnsaioFromCRM(expSel, sortedFrames);
      } else {
        throw new Error('Nenhum experimento selecionado ou JSON fornecido.');
      }

      setFullObservedData(getObservedDataWithDerived(ensaioCompleto, fctRes?.channels));

      const body = {
        ensaio: sliceEnsaio(ensaioCompleto, Number(sliceStart)),
        sensors: Object.entries(fctRes.channels).map(([s, chs]) => `${s}:${chs.join(',')}`),
        units: form.units,
        fctFile: selectedTrial.modelFile,
      };

      const res = await fetch(`${API_BASE}/infer_forecaster`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const result = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(result));

      const yForecastAdjusted = applyInverseGainToForecast(result?.yForecast, ensaioCompleto, fctRes?.channels);
      setForecastResult({ ...result, yForecast: yForecastAdjusted });
    } catch (e) {
      alert(`Erro ao executar a previsão: ${e.message || e}`);
      setForecastResult(null);
    } finally {
      setIsLoading(false);
    }
  }, [selectedTrial, testJson, expSel, sliceStart, form.units, fctRes?.channels]);

  const chartData = useMemo(() => {
    if (!fullObservedData || !Array.isArray(fullObservedData.series)) return { datasets: [] };

    const { labels: channelLabels = [], series: fullSeries = [] } = fullObservedData;
    const cut = Math.max(0, Number(sliceStart) || 0);
    const yForecast = forecastResult?.yForecast || [];
    const forecastLen = yForecast[0]?.length || 0;

    const toXY = (arr, offset = 0) => (arr || []).map((y, i) => ({ x: offset + i, y }));
    const datasets = [];

    channelLabels.forEach((label, i) => {
      const color = palette[i % palette.length];

      datasets.push({
        label: `Real ${label}`,
        data: toXY(fullSeries[i], 0),
        borderColor: `${color}80`,
        borderWidth: 1,
        pointRadius: 0,
        showLine: true,
      });

      if (forecastLen && yForecast[i]) {
        datasets.push({
          label: `Previsão ${label}`,
          data: toXY(yForecast[i], 0),
          borderColor: color,
          borderWidth: 2,
          pointRadius: 0,
          borderDash: [6, 3],
          showLine: true,
        });
      }
    });

    const allY = datasets.flatMap(d => d.data?.map(p => p.y)).filter(v => v != null && isFinite(v));
    if (allY.length) {
      const ymin = Math.min(...allY);
      const ymax = Math.max(...allY);
      datasets.push({
        label: 'Corte',
        data: [{ x: cut, y: ymin }, { x: cut, y: ymax }],
        borderColor: '#000',
        borderWidth: 2,
        borderDash: [4, 4],
        pointRadius: 0,
        showLine: true,
      });
    }

    return { datasets };
  }, [fullObservedData, forecastResult, sliceStart]);

  return (
    <>
      <Backdrop open={isLoading} sx={{ color: '#fff', zIndex: (t) => t.zIndex.drawer + 1, bgcolor: 'rgba(0,0,0,0.7)' }}>
        <CircularProgress color="inherit" />
      </Backdrop>

      <Paper variant="outlined" sx={{ p: { xs: 1.5, md: 3 } }}>
        <Typography variant="h6" mb={2}>Resultados do Forecaster</Typography>

        <Grid container spacing={3}>
          {/* Coluna Esquerda: Checkpoints */}
          <Grid item xs={12} md={4} lg={3}>
            <Typography variant="subtitle1" gutterBottom>Checkpoints</Typography>
            <Paper variant="outlined" sx={{ p: 1, maxHeight: 'calc(100vh - 320px)', overflowY: 'auto' }}>
              {trials.map(t => (
                <CheckpointCard
                  key={t.modelFile}
                  trial={t}
                  isBest={t.modelFile === bestTrial?.modelFile}
                  isSelected={t.modelFile === selectedTrial?.modelFile}
                  onClick={() => setSelectedTrial(t)}
                />
              ))}
            </Paper>
          </Grid>

          {/* Coluna Direita: Análise */}
          <Grid item xs={12} md={8} lg={9}>
            <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tab label="Visão Geral" value={0} />
              <Tab label="Teste Rápido" value={1} />
            </Tabs>

            <Box sx={{ pt: 2 }}>
              {tab === 1 && (
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} sm={6}>
                          <Autocomplete
                            options={expList}
                            value={expSel}
                            onChange={(_, v) => { setExpSel(v); setTestJson(''); }}
                            getOptionLabel={(o) => {
                              const id = o?.id ?? o?.uuid ?? o?._id ?? '—';
                              const ns = o?.numeroSerie ?? o?.serial ?? '';
                              const p = extractProtoId(o) ?? 's/ proto';
                              return `${String(id).slice(0, 8)}… ${ns} · proto:${p}`;
                            }}
                            isOptionEqualToValue={(a, b) =>
                              (a?.id ?? a?.uuid ?? a?._id) === (b?.id ?? b?.uuid ?? b?._id)
                            }
                            renderInput={(params) => (
                              <TextField {...params} label="Escolha experimento por protocolo" />
                            )}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            fullWidth
                            type="number"
                            label="2. Ponto de corte (minutos)"
                            value={sliceStart}
                            onChange={e => setSliceStart(e.target.value)}
                          />
                        </Grid>
                        <Grid item xs={12} sm={12}>
                          <Button
                            fullWidth
                            size="large"
                            variant="contained"
                            disabled={isLoading || !selectedTrial}
                            onClick={handleRun}
                          >
                            3. Rodar Previsão
                          </Button>
                        </Grid>
                      </Grid>
                    </Paper>
                  </Grid>

                  {/* Gráfico */}
                  {chartData && chartData.datasets?.length > 0 && (
                    <Grid item xs={12}>
                      <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
                        <Box sx={{ height: 350, mt: 2 }}>
                          <Line
                            data={chartData}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              interaction: { mode: 'index', intersect: false },
                              plugins: {
                                legend: { position: 'top' },
                                tooltip: { mode: 'index', intersect: false },
                              },
                              parsing: false,
                              scales: {
                                x: {
                                  type: 'linear',
                                  title: { display: true, text: 'Tempo (minutos)' },
                                },
                                y: {
                                  type: 'linear',
                                  title: { display: true, text: 'Intensidade (basic counts)' },
                                },
                              },
                            }}
                          />
                        </Box>
                      </Paper>
                    </Grid>
                  )}
                </Grid>
              )}

              {tab === 0 && (
                <TableContainer component={Paper}>
                  <Table stickyHeader size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>ID</TableCell>
                        <TableCell>Window</TableCell>
                        <TableCell>Horizon</TableCell>
                        <TableCell>Layers</TableCell>
                        <TableCell>Dropout</TableCell>
                        <TableCell align="right">Train MSE</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {trials.map(t => {
                        const params = parseCheckpointName(t.modelFile);
                        return (
                          <TableRow key={t.modelFile} hover selected={t.modelFile === selectedTrial?.modelFile}>
                            <TableCell>#{t.trialId}</TableCell>
                            <TableCell>{params.w}</TableCell>
                            <TableCell>{params.hor}</TableCell>
                            <TableCell>{params.l}</TableCell>
                            <TableCell>{params.d}</TableCell>
                            <TableCell align="right">{t.trainMse.toFixed(5)}</TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          </Grid>
        </Grid>

        {/* RODAPÉ PADRÃO: Voltar (esq) + Avançar (dir) */}
        <Stack direction="row" justifyContent="space-between" sx={{ mt: 3 }}>
          <Button startIcon={<ArrowBackIcon />} onClick={onBack}>Voltar</Button>
          <Button
            variant="contained"
            color="secondary"
            disabled={!selectedTrial || !selectedTrial?.featureFile}
            onClick={() => {
              if (!selectedTrial?.featureFile) {
                alert('Este trial não possui featureFile (.npz).');
                return;
              }
              // mesmo fluxo que antes (vai para o step de Regressor)
              window.dispatchEvent(new CustomEvent('goRegressor', { detail: { trial: selectedTrial } }));
            }}
          >
            Avançar
          </Button>
        </Stack>
      </Paper>
    </>
  );
});

// ──────────────────────────────────────────────────────────────────────────────
// Step 5 — Regressor (configuração)
// ──────────────────────────────────────────────────────────────────────────────
const StepRegressor = memo(({ selectedTrial, regForm, setRegForm, onTrain, onBack, loadingRegressor }) => {
  const [activeTab, setActiveTab] = useState(() => {
    const firstSupported = (regForm.models || []).find(m => SUPPORTED_BY_BACKEND.has(m));
    return firstSupported || 'random_forest';
  });

  useEffect(() => {
    const supportedSel = (regForm.models || []).filter(m => SUPPORTED_BY_BACKEND.has(m));
    setActiveTab(prev => supportedSel.includes(prev) ? prev : supportedSel[0] || 'random_forest');
  }, [regForm.models]);

  const selectAll = () => setRegForm(prev => ({ ...prev, models: [...MODEL_OPTIONS] }));
  const clearAll = () => setRegForm(prev => ({ ...prev, models: [], params: {}, grids: {} }));

  const toggleModel = (name) => {
    setRegForm(prev => {
      const models = prev.models.includes(name) ? prev.models.filter(m => m !== name) : [...prev.models, name];
      const params = { ...(prev.params || {}) };
      const grids = { ...(prev.grids || {}) };
      Object.keys(params).forEach(k => { if (!models.includes(k)) delete params[k]; });
      Object.keys(grids).forEach(k => { if (!models.includes(k)) delete grids[k]; });
      return { ...prev, models, params, grids };
    });
  };

  const setParam = (model, key, raw) => {
    setRegForm(prev => ({
      ...prev,
      params: { ...prev.params, [model]: { ...(prev.params[model] || {}), [key]: raw } }
    }));
  };

  const toggleGrid = (model, key, use) => {
    setRegForm(prev => {
      const grids = { ...(prev.grids || {}) };
      const spec = (REG_SPECS[model] || []).find(s => s[0] === key) || [null, null, 0, 1, 0.1];
      const [, , min, max, defStep] = spec;

      if (use) {
        if (model === 'svr' && key === 'kernel') {
          // grid categórico: começa com todas as opções
          grids[model] = {
            ...(grids[model] || {}),
            kernel: [...SVR_KERNELS],
            __ui: { ...(grids[model]?.__ui || {}) } // nada especial pra UI aqui
          };
          return { ...prev, grids };
        }

        // MLP: activation/optimizer são categóricos → começa com TODAS as opções
        if (model === 'mlp' && (key === 'activation' || key === 'optimizer')) {
          const opts = key === 'activation' ? MLP_ACTIVATIONS : MLP_OPTIMIZERS;
          grids[model] = {
            ...(grids[model] || {}),
            [key]: [...opts],
            __ui: { ...(grids[model]?.__ui || {}) }
          };
          return { ...prev, grids };
        }

        const raw = prev.params?.[model]?.[key];
        const start = Number(raw) || min;
        const end = Math.min(max, start + 2 * defStep);
        const values = rangeFrom(start, end, defStep);
        grids[model] = {
          ...(grids[model] || {}),
          [key]: values,
          __ui: { ...(grids[model]?.__ui || {}), [key]: { start, end, step: defStep } }
        };
      } else {
        const g = { ...(grids[model] || {}) };
        const ui = { ...(g.__ui || {}) };
        delete g[key];
        delete ui[key];
        g.__ui = ui;
        grids[model] = g;
      }
      return { ...prev, grids };
    });
  };

  const toggleKernelInGrid = (model, kernel, checked) => {
    setRegForm(prev => {
      const grids = { ...(prev.grids || {}) };
      const g = { ...(grids[model] || {}) };
      const arr = Array.isArray(g.kernel) ? [...g.kernel] : [];
      const idx = arr.indexOf(kernel);
      if (checked && idx === -1) arr.push(kernel);
      if (!checked && idx !== -1) arr.splice(idx, 1);
      g.kernel = arr;
      grids[model] = g;
      return { ...prev, grids };
    });
  };
  const toggleMlpCatInGrid = (model, key, option, checked) => {
    setRegForm(prev => {
      const grids = { ...(prev.grids || {}) };
      const g = { ...(grids[model] || {}) };
      const arr = Array.isArray(g[key]) ? [...g[key]] : [];
      const idx = arr.indexOf(option);
      if (checked && idx === -1) arr.push(option);
      if (!checked && idx !== -1) arr.splice(idx, 1);
      g[key] = arr;
      grids[model] = g;
      return { ...prev, grids };
    });
  };
  const TxtNum = ({ label, value, disabled, onChange }) => {
    const [local, setLocal] = useState(value ?? '');
    useEffect(() => { setLocal(value ?? ''); }, [value]);
    const NUM_RE = /^-?\d*(?:[.,]\d*)?$/; // aceita -, dígitos, vírgula/ponto
    return (
      <TextField
        size="small"
        label={label}
        value={local}
        inputMode="decimal"
        type="text"
        disabled={disabled}
        onChange={(e) => {
          const v = e.target.value;
          if (v === '' || NUM_RE.test(v)) setLocal(v);
        }}
        onBlur={() => onChange(local)}   // commita ao sair
        onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); e.currentTarget.blur(); } }}
        sx={{ width: '100%' }}
      />
    );
  };

  const setGridPart = (model, key, part, raw, spec) => {
    const [, , min, , defStep] = spec;
    setRegForm(prev => {
      const grids = { ...(prev.grids || {}) };
      const gModel = grids[model] || {};
      const ui = { ...(gModel.__ui || {}) };
      const cur = { ...(ui[key] || { start: min, end: min + 2 * defStep, step: defStep }) };

      cur[part] = raw;
      const parsed = { start: parseFloat(cur.start), end: parseFloat(cur.end), step: parseFloat(cur.step) };
      const values = rangeFrom(parsed.start, parsed.end, parsed.step);

      const nextModel = { ...gModel, [key]: values, __ui: { ...ui, [key]: cur } };
      grids[model] = nextModel;
      return { ...prev, grids };
    });
  };

  const ParamRow = ({ model, spec }) => {
    const [key, label, min, max] = spec;
    const p = regForm.params[model] || {};
    const g = regForm.grids[model] || {};
    const ui = g.__ui?.[key] || {};
    const usingGrid = !!g[key];

    const isSVRKernel = model === 'svr' && key === 'kernel';
    const isMLPActivation = model === 'mlp' && key === 'activation';
    const isMLPOptimizer = model === 'mlp' && key === 'optimizer';
    const isCat = isSVRKernel || isMLPActivation || isMLPOptimizer;
    const single = isSVRKernel
      ? (p[key] ?? REG_DEFAULTS.svr.kernel ?? 'rbf')
      : isMLPActivation
        ? (p[key] ?? REG_DEFAULTS.mlp.activation ?? 'relu')
        : isMLPOptimizer
          ? (p[key] ?? REG_DEFAULTS.mlp.optimizer ?? 'adam')
          : (p[key] ?? REG_DEFAULTS[model]?.[key]?.toString() ?? min.toString());
              const opts = isSVRKernel
              ? SVR_KERNELS
              : (isMLPActivation ? MLP_ACTIVATIONS : MLP_OPTIMIZERS);


    const templateSm = usingGrid
      ? 'minmax(190px,1.8fr) 90px repeat(3, minmax(120px,1fr))'
      : 'minmax(190px,1.8fr) 90px minmax(180px,1fr)';

    return (
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: templateSm }, gap: 1, alignItems: 'center', py: 1 }}>
        <Box>
          <Typography sx={{ fontWeight: 600, lineHeight: 1 }}>{label}</Typography>
          {!isCat && (
            <Typography variant="caption" color="text.secondary">
              [{min}…{max}] • {usingGrid ? 'Faixa + passo' : 'Valor único'}
            </Typography>
          )}



        </Box>
        <Box>
          <Button size="small" variant={usingGrid ? 'contained' : 'outlined'} onClick={() => toggleGrid(model, key, !usingGrid)}>
            Grid
          </Button>
        </Box>
        {!usingGrid ? (
          isCat ? (
            
            <FormControl size="small" fullWidth>
              <InputLabel>
                {isSVRKernel ? 'Kernel' : (isMLPActivation ? 'Ativação' : 'Otimizador')}
              </InputLabel>


              <Select
                label={isSVRKernel ? 'Kernel' : (isMLPActivation ? 'Ativação' : 'Otimizador')}
                value={single}
                onChange={(e) => setParam(model, key, e.target.value)}
              >
                {opts.map(opt => (
                  <MenuItem key={opt} value={opt}>{opt.toUpperCase()}</MenuItem>
                ))}
              </Select>

            </FormControl>
          ) : (
            <TxtNum label="Valor" value={single} onChange={(v) => setParam(model, key, v)} />
          )
        ) : (
          isCat ? (
            <FormGroup row sx={{ gridColumn: { sm: '3 / -1' } }}>
              {(isSVRKernel ? SVR_KERNELS : (isMLPActivation ? MLP_ACTIVATIONS : MLP_OPTIMIZERS)).map(opt => (
                <FormControlLabel
                  key={opt}
                  control={
                    <Checkbox
                      size="small"
                      checked={Array.isArray(g[key]) && g[key].includes(opt)}
                      onChange={(e) =>
                        isSVRKernel
                          ? toggleKernelInGrid('svr', opt, e.target.checked)
                          : toggleMlpCatInGrid('mlp', key, opt, e.target.checked)
                      }
                    />
                  }
                  label={opt.toUpperCase()}
                  sx={{ mr: 1 }}
                />
              ))}
            </FormGroup>
          ) : (
            <>
              <TxtNum label="Início" value={ui.start ?? ''} onChange={(v) => setGridPart(model, key, 'start', v, spec)} />
              <TxtNum label="Fim" value={ui.end ?? ''} onChange={(v) => setGridPart(model, key, 'end', v, spec)} />
              <TxtNum label="Passo" value={ui.step ?? ''} onChange={(v) => setGridPart(model, key, 'step', v, spec)} />
            </>
          )
        )}
        <Divider sx={{ gridColumn: '1 / -1', mt: 1 }} />
      </Box>
    );
  };

  const supportedSelected = (regForm.models || []).filter(m => SUPPORTED_BY_BACKEND.has(m));

  return (
    <Paper variant="outlined" sx={{ p: 3 }}>
      {loadingRegressor && (
        <Backdrop open sx={{ color: '#fff', zIndex: (t) => t.zIndex.drawer + 1, bgcolor: 'rgba(0, 0, 0, 0.8)' }}>
          <Stack alignItems="center" spacing={2}>
            <CircularProgress color="inherit" />
            <Typography>Treinando Regressor… Por favor, aguarde.</Typography>
          </Stack>
        </Backdrop>
      )}

      <Typography variant="h6" gutterBottom>Regressor</Typography>

      {!selectedTrial && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Selecione um checkpoint na etapa anterior para liberar as configurações.
        </Alert>
      )}



      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              gap: 1.5,
              maxHeight: 'calc(100vh - 320px)', // mesmo padrão das outras listas
              overflowY: 'auto',
            }}
          >
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="subtitle2">Modelos</Typography>
              <Stack direction="row" spacing={1}>
                <Button size="small" onClick={selectAll}>Todos</Button>
                <Button size="small" onClick={clearAll}>Limpar</Button>
              </Stack>
            </Stack>

            <Divider />

            {/* Lista vertical, alinhada e consistente */}
            <FormGroup
              sx={{
                '& .MuiFormControlLabel-root': { m: 0, py: 0.25 },
                '& .MuiCheckbox-root': { p: 0.5 }, // deixa mais “denso” como nas outras telas
                opacity: 1,
              }}
            >
              {MODEL_OPTIONS.map((m) => {
                const disabled = !SUPPORTED_BY_BACKEND.has(m);
                return (
                  <FormControlLabel
                    key={m}
                    control={
                      <Checkbox
                        size="small"
                        checked={regForm.models.includes(m)}
                        onChange={() => toggleModel(m)}
                        disabled={disabled}
                      />
                    }
                    label={MODEL_LABEL_PT[m] + (disabled ? '' : '')}
                    sx={{ opacity: disabled ? 0.55 : 1 }}
                  />
                );
              })}
            </FormGroup>
          </Paper>
        </Grid>


        <Grid item xs={12} md={8}>
          {!supportedSelected.length ? (
            <Alert severity="info">Selecione ao menos um modelo suportado.</Alert>
          ) : (
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Tabs
                value={activeTab}
                onChange={(_, v) => setActiveTab(v)}
                variant="scrollable"
                scrollButtons="auto"
                sx={{ mb: 2 }}
              >
                {supportedSelected.map(m => (<Tab key={m} value={m} label={REG_LABEL[m]} />))}
              </Tabs>
              {(REG_SPECS[activeTab] || []).map(spec => (
                <ParamRow key={`${activeTab}-${spec[0]}`} model={activeTab} spec={spec} />
              ))}
            </Paper>
          )}



        </Grid>



      </Grid>

      {/* Rodapé padrão: alinhado com os outros steps */}
      <Divider sx={{ mt: 3, mb: 2 }} />
      <Stack direction="row" justifyContent="space-between">
        <Button startIcon={<ArrowBackIcon />} onClick={onBack}>Voltar</Button>
        <Button
          variant="contained"
          onClick={onTrain}
          disabled={!selectedTrial || !supportedSelected.length}
        >
          Treinar Regressor
        </Button>
      </Stack>
    </Paper>
  );
});

// ──────────────────────────────────────────────────────────────────────────────
// Resultados do Regressor (rodapé padronizado)
// ──────────────────────────────────────────────────────────────────────────────
const getImpRaw = (m) =>
  m?.feature_importances ??
  m?.feature_importances_ ??
  m?.featureImportances ??
  m?.importances ??
  [];

const RegCheckpointCard = memo(({ item, isBest, isSelected, onClick }) => {
  const primaryParams = useMemo(() => {
    const RESERVED = new Set([
      'name', 'model', 'modelFile', 'model_file', 'metrics',
      'mse', 'rmse', 'mae', 'r2',
      'feature_importances_', 'feature_importances', 'importances',
      'y_true', 'yTrue', 'pred', 'preds', 'y_pred', 'featureFile'
    ]);
    const obj = {};
    Object.entries(item || {}).forEach(([k, v]) => {
      if (!RESERVED.has(k) && typeof v !== 'object') obj[k] = v;
    });
    return Object.entries(obj).slice(0, 3);
  }, [item]);

  return (
    <Card
      variant="outlined"
      sx={{
        mb: 1.5,
        border: isSelected ? '2px solid' : '1px solid',
        borderColor: isSelected ? 'primary.main' : 'divider',
        bgcolor: isSelected ? 'action.selected' : 'transparent',
        transition: 'all 0.2s ease-in-out',
        '&:hover': { borderColor: 'primary.light' }
      }}
    >
      <CardActionArea onClick={onClick} sx={{ p: 1.5 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" mb={0.5}>
          <Typography variant="body2" fontWeight={600} title={item.model_file}>
            {String(item.model || '—').toUpperCase()}
          </Typography>
          {isBest && <Chip size="small" label="🏆 Melhor" color="warning" />}
        </Stack>

        <Grid container spacing={0.5} sx={{ mb: .5 }}>
          {primaryParams.map(([k, v]) => (
            <Grid item key={k}><Chip variant="outlined" size="small" label={`${k}: ${v}`} /></Grid>
          ))}
        </Grid>

        <Typography variant="caption" display="block">
          RMSE: {Number(item.rmse).toFixed(4)} • MAE: {Number(item.mae).toFixed(4)} • R²: {Number(item.r2).toFixed(3)}
        </Typography>

        {item.featureFile && (
          <Typography variant="caption" display="block" color="text.secondary" title={item.featureFile}>
            features: {item.featureFile}
          </Typography>
        )}
      </CardActionArea>
    </Card>
  );
});

const StepRegressorResults = memo(({ form, selectedTrial, regRes, onInferFinal, regInfer, onBack }) => {
  const COLOR_SEL = '#1976d2';
  const COLOR_BEST = '#ed6c02';

  const [tab, setTab] = useState(0);
  const [expSel, setExpSel] = useState(null);
  const [sliceStart, setSliceStart] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [expList, setExpList] = useState([]);

  const [topK, setTopK] = useState(20);
  const unit = form?.units?.[0] || '';

  const [quickObserved, setQuickObserved] = useState(null);
  const [quickResult, setQuickResult] = useState(null);

  const models = useMemo(() => {
    const src = Array.isArray(regRes?.models) ? regRes.models : [];
    return src.map((m, i) => {
      const rmse = m.metrics?.rmse ?? m.rmse;
      const mae = m.metrics?.mae ?? m.mae;
      const r2 = m.metrics?.r2 ?? m.r2;
      return {
        ...m,
        idx: i,
        model: m.name ?? m.model ?? '—',
        model_file: m.modelFile ?? m.model_file ?? '—',
        featureFile: m.featureFile ?? undefined,
        rmse: Number(rmse),
        mae: Number(mae),
        r2: Number(r2),
        _y_true_raw: m[`y_true_${unit}`] ?? m.y_true ?? m.yTrue ?? [],
        _preds_raw: m[`pred_${unit}`] ?? m.preds ?? m.pred ?? m.y_pred ?? [],
        _imps_raw: getImpRaw(m),
      };
    });
  }, [regRes, unit]);

  const regsByRmse = useMemo(() => [...models].sort((a, b) => a.rmse - b.rmse), [models]);
  const best = regsByRmse[0] || null;
  const [selected, setSelected] = useState(best || null);
  useEffect(() => { if (!selected && best) setSelected(best); }, [best, selected]);

  useEffect(() => {
    if (!form.protocolId) { setExpList([]); return; }
    (async () => {
      try {
        const all = await getExperimentos();
        const filtered = (all);
        if (!filtered.length) {
          const distinct = [...new Set((all || []).map(e => String(extractProtoId(e))))];
          if (distinct.length === 1 && (distinct[0] === 'undefined' || distinct[0] === 'null')) {
            setExpList(all || []);
          } else {
            setExpList(filtered);
          }
        } else {
          setExpList(filtered);
        }
      } catch {
        setExpList([]);
      }
    })();
  }, [form.protocolId]);

  const chartDataQuick = useMemo(() => {
    const channelLabels = quickObserved?.labels || [];
    const fullSeries = quickObserved?.series || [];
    const cut = Math.max(0, Number(sliceStart) || 0);
    const yForecast = quickResult?.yForecast || [];
    const forecastLen = yForecast[0]?.length || 0;

    const toXY = (arr, offset = 0) => (arr || []).map((y, i) => ({ x: offset + i, y }));
    const datasets = [];

    channelLabels.forEach((label, i) => {
      const color = palette[i % palette.length];

      datasets.push({
        label: `Real ${label}`,
        data: toXY(fullSeries[i], 0),
        borderColor: `${color}80`,
        borderWidth: 1,
        pointRadius: 0,
        showLine: true,
      });

      if (forecastLen && yForecast[i]) {
        datasets.push({
          label: `Previsão ${label}`,
          data: toXY(yForecast[i], 0),
          borderColor: color,
          borderWidth: 2,
          pointRadius: 0,
          borderDash: [6, 3],
          showLine: true,
        });
      }
    });

    const allY = datasets.flatMap(d => d.data?.map(p => p.y)).filter(v => v != null && isFinite(v));
    if (allY.length) {
      const ymin = Math.min(...allY);
      const ymax = Math.max(...allY);
      datasets.push({
        label: 'Corte',
        data: [{ x: cut, y: ymin }, { x: cut, y: ymax }],
        borderColor: '#000',
        borderWidth: 2,
        borderDash: [4, 4],
        pointRadius: 0,
        showLine: true,
      });
    }
    return { datasets };
  }, [quickObserved, quickResult, sliceStart]);

  const fmt3 = (v) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(3));
  const toArr = (v) => {
    if (Array.isArray(v)) return v;
    if (typeof v === 'string') { try { return JSON.parse(v); } catch { } }
    return [];
  };

  const yT_sel = toArr(selected?._y_true_raw);

  const yP_sel = toArr(selected?._preds_raw);
  const yT_best = toArr(best?._y_true_raw);
  const yP_best = toArr(best?._preds_raw);

  const scatterSel = yT_sel.map((v, i) => ({ x: v, y: yP_sel[i] }));
  const scatterBest = yT_best.map((v, i) => ({ x: v, y: yP_best[i] }));
  const allVals = [...yT_sel, ...yP_sel, ...yT_best, ...yP_best].filter(Number.isFinite);
  const minXY = allVals.length ? Math.min(...allVals) : 0;
  const maxXY = allVals.length ? Math.max(...allVals) : 1;
  const diag = [{ x: minXY, y: minXY }, { x: maxXY, y: maxXY }];

  const residsSel = yP_sel.map((p, i) => ({ x: p, y: p - (yT_sel[i] ?? 0) }));
  const residsBest = yP_best.map((p, i) => ({ x: p, y: p - (yT_best[i] ?? 0) }));
  const allPreds = [...yP_sel, ...yP_best].filter(Number.isFinite);
  const zero = allPreds.length ? [{ x: Math.min(...allPreds), y: 0 }, { x: Math.max(...allPreds), y: 0 }] : [];
  const showBoth = !!(best && selected && best.model_file !== selected.model_file);

  const impSel = (selected?._imps_raw || []).map((v, i) => ({ feat: `F${i}`, val: Number(v) || 0 }));
  const impBest = (best?._imps_raw || []).map((v, i) => ({ feat: `F${i}`, val: Number(v) || 0 }));
  const impSelSorted = [...impSel].sort((a, b) => Math.abs(b.val) - Math.abs(a.val)).slice(0, 20);
  const labelsImp = impSelSorted.map(d => d.feat);
  const lookupBest = new Map(impBest.map(d => [d.feat, d.val]));
  const dataImpSel = impSelSorted.map(d => Math.abs(d.val));
  const dataImpBest = impSelSorted.map(d => Math.abs(lookupBest.get(d.feat) ?? 0));

  function shortName(file) {
    if (!file) return '';
    let base = file.replace(/^reg_/, '').replace(/\.joblib$/, '');
    const parts = base.split('_');
    const tail = parts.slice(-2).join('_'); // ex.: 001_d5e49903
    const arch = parts.find(p => ['rf', 'svr', 'xgb', 'mlp', 'cnn', 'lstm', 'gbm'].includes(p)) || '';
    return `${arch.toUpperCase()} (${tail})`;
  }

  // ao selecionar experimento: carrega séries observadas e define corte padrão
  useEffect(() => {
    let alive = true;
    (async () => {
      if (!expSel) { if (alive) setQuickObserved(null); return; }
      try {
        const frames = await getExperimentosDados(expSel.id);
        const sortedFrames = (frames || []).sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
        const ensaio = buildEnsaioFromCRM(expSel, sortedFrames);
        if (!alive) return;
        setQuickObserved(getObservedDataWithDerived(ensaio, form?.channelMap || {}));
        const params = parseCheckpointName(selectedTrial?.modelFile || '');
        if (Number.isFinite(params.w)) setSliceStart(params.w);
      } catch {
        if (alive) setQuickObserved(null);
      }
    })();
    return () => { alive = false; };
  }, [expSel, selectedTrial, form?.channelMap]);

  const handleRunAll = useCallback(async () => {
    if (!selectedTrial) return alert('Selecione um checkpoint do forecaster');
    if (!selected) return alert('Selecione um modelo do regressor');
    try {
      setIsLoading(true);
      let target = expSel;
      if (!target) {
        const all = await getExperimentos();
        target = all?.[0];
        if (!target) throw new Error('Nenhum experimento disponível.');
      }
      const frames = await getExperimentosDados(target.id);
      const sortedFrames = (frames || []).sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
      const ensaio = buildEnsaioFromCRM(target, sortedFrames);

      setQuickObserved(getObservedDataWithDerived(ensaio, form?.channelMap || {}));

      // FORECASTER
      const bodyFct = {
        ensaio: sliceEnsaio(ensaio, Number(sliceStart) || 0),
        sensors: Object.entries(form?.channelMap || {}).map(([s, chs]) => `${s}:${(chs || []).join(',')}`),
        units: form.units,
        fctFile: (selectedTrial?.modelFile || ''),
      };
      const r1 = await fetch(`${API_BASE}/infer_forecaster`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bodyFct),
      });
      const js1 = await r1.json();
      if (!r1.ok) {
        const msg = js1?.detail || js1?.error || JSON.stringify(js1);
        throw new Error(`Forecaster: ${msg}`);
      }
      const yAdj = applyInverseGainToForecast(js1?.yForecast, ensaio, form?.channelMap);
      setQuickResult({ yForecast: yAdj || null });

      // REGRESSOR
      const bodyReg = {
        ensaio: sliceEnsaio(ensaio, Number(sliceStart) || 0),
        sensors: Object.entries(form?.channelMap || {}).map(([s, chs]) => `${s}:${(chs || []).join(',')}`),
        units: form.units,
        forecasterFile: (selectedTrial?.modelFile || ''),
        regressorFile: (selected.model_file || selected.modelFile || ''),
        sliceStart: Number(sliceStart) || 0,
      };
      const r2 = await fetch(`${API_BASE}/infer_regressor`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bodyReg),
      });
      const js2 = await r2.json();
      if (!r2.ok) {
        const msg = js2?.detail || js2?.error || JSON.stringify(js2);
        throw new Error(`Regressor: ${msg}`);
      }
      setQuickResult(prev => ({ ...(prev || {}), yPredicted: js2?.yPredicted }));
    } catch (e) {
      alert(`Erro na previsão completa: ${e.message || e}`);
      setQuickResult(null);
    } finally {
      setIsLoading(false);
    }
  }, [selectedTrial, selected, expSel, sliceStart, form?.channelMap, form?.units]);

  return (
    <>
      <Backdrop open={isLoading} sx={{ color: '#fff', zIndex: (t) => t.zIndex.drawer + 1, bgcolor: 'rgba(0,0,0,0.7)' }}>
        <CircularProgress color="inherit" />
      </Backdrop>

      <Paper variant="outlined" sx={{ p: { xs: 1.5, md: 3 } }}>
        <Typography variant="h6" mb={2}>Resultados do Regressor</Typography>

        <Grid container spacing={3}>
          {/* Esquerda — Modelos */}
          <Grid item xs={12} md={4} lg={3}>
            <Typography variant="subtitle1" gutterBottom>Modelos</Typography>
            <Paper variant="outlined" sx={{ p: 1, maxHeight: 'calc(100vh - 320px)', overflowY: 'auto' }}>
              {regsByRmse.map(m => (
                <RegCheckpointCard
                  key={m.model_file}
                  item={m}
                  isBest={best && m.model_file === best.model_file}
                  isSelected={selected && m.model_file === selected.model_file}
                  onClick={() => setSelected(m)}
                />
              ))}
            </Paper>
          </Grid>

          {/* Direita — Abas */}
          <Grid item xs={12} md={8} lg={9}>
            <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tab label="Visão Geral" value={0} />
              <Tab label="Teste Rápido" value={1} />
            </Tabs>

            <Box sx={{ pt: 2 }}>
              {/* Visão Geral */}
              {tab === 0 && (
                <Box>
                  <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1, flexWrap: 'wrap' }}>
                      <Typography variant="subtitle2">Comparação:</Typography>
                      <Chip size="small" variant="outlined" label={`Selecionado: ${(() => {
                        const f = selected?.modelFile || selected?.model_file; return f ? shortName(f) : '—';
                      })()}`} />
                      {best && (
                        <Chip size="small" color="warning" icon={<EmojiEventsIcon />} label={`BEST: ${(() => {
                          const f = best?.modelFile || best?.model_file; return f ? shortName(f) : '—';
                        })()}`} />
                      )}
                    </Stack>

                    <Grid container spacing={2} sx={{ mb: 2 }}>
                      {['rmse', 'mae', 'r2'].map((k) => (
                        <Grid item xs={12} md={4} key={k}>
                          <Paper variant="outlined" sx={{ p: 1.5 }}>
                            <Typography variant="caption" color="text.secondary">{k.toUpperCase()}</Typography>
                            <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: .5 }}>
                              <Box>
                                <Typography variant="body2" color="text.secondary">Sel.</Typography>
                                <Typography variant="h6" sx={{ m: 0 }}>{fmt3(selected?.[k])}</Typography>
                              </Box>
                              <Divider flexItem orientation="vertical" />
                              <Box>
                                <Typography variant="body2" color="text.secondary">Best</Typography>
                                <Typography variant="h6" sx={{ m: 0 }}>{fmt3(best?.[k])}</Typography>
                              </Box>
                            </Stack>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>

                    {(() => {
                      const isSame = selected?.model_file === best?.model_file;
                      const getHyper = (r) =>
                        (r?.params) || (r?.best_params) || (r?.hyperparameters) || (r?.hyperparams) || {};
                      const pSel = getHyper(selected);
                      const pBest = getHyper(best);
                      const keys = Array.from(new Set([...Object.keys(pSel), ...Object.keys(pBest)])).sort();
                      const fmtVal = (v) => {
                        if (v == null) return '—';
                        if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') return String(v);
                        try { return JSON.stringify(v); } catch { return String(v); }
                      };
                      return (
                        <Paper variant="outlined" sx={{ p: 1.5 }}>
                          <Typography variant="subtitle2" sx={{ mb: 1 }}>Parâmetros</Typography>
                          {keys.length ? (
                            <Table size="small">
                              <TableHead>
                                <TableRow>
                                  <TableCell>Parâmetro</TableCell>
                                  <TableCell>Selecionado</TableCell>
                                  <TableCell>Best</TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {keys.map((k) => {
                                  const a = pSel[k]; const b = pBest[k];
                                  const diff = !isSame && JSON.stringify(a) !== JSON.stringify(b);
                                  return (
                                    <TableRow key={k}>
                                      <TableCell sx={{ whiteSpace: 'nowrap' }}>{k}</TableCell>
                                      <TableCell sx={{ fontWeight: diff ? 600 : 400 }}>{fmtVal(a)}</TableCell>
                                      <TableCell sx={{ fontWeight: diff ? 600 : 400 }}>{fmtVal(b)}</TableCell>
                                    </TableRow>
                                  );
                                })}
                              </TableBody>
                            </Table>
                          ) : (
                            <Typography variant="caption" color="text.secondary">
                              (Sem hiperparâmetros disponíveis para estes modelos.)
                            </Typography>
                          )}
                        </Paper>
                      );
                    })()}
                  </Paper>

                  {/* Gráficos de diagnóstico */}
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Box sx={{ height: 360 }}>
                        <Line
                          data={{
                            datasets: [
                              { label: 'Selecionado — Predito vs Real', data: scatterSel, showLine: false, pointRadius: 3, borderColor: COLOR_SEL, backgroundColor: COLOR_SEL },
                              ...(showBoth ? [{ label: 'BEST — Predito vs Real', data: scatterBest, showLine: false, pointRadius: 3, borderColor: COLOR_BEST, backgroundColor: COLOR_BEST }] : []),
                              { label: '1:1', data: diag, showLine: true, pointRadius: 0, borderWidth: 3, borderColor: '#9e9e9e', backgroundColor: '#9e9e9e' }
                            ]
                          }}
                          options={{
                            responsive: true, maintainAspectRatio: false, parsing: false,
                            plugins: { legend: { position: 'bottom' } },
                            scales: { x: { type: 'linear', title: { display: true, text: 'Real' } }, y: { type: 'linear', title: { display: true, text: 'Predito' } } }
                          }}
                        />
                      </Box>
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <Box sx={{ height: 360 }}>
                        <Line
                          data={{
                            datasets: [
                              { label: 'Selecionado — Resíduo', data: residsSel, showLine: false, pointRadius: 3, borderColor: COLOR_SEL, backgroundColor: COLOR_SEL },
                              ...(showBoth ? [{ label: 'BEST — Resíduo', data: residsBest, showLine: false, pointRadius: 3, borderColor: COLOR_BEST, backgroundColor: COLOR_BEST }] : []),
                              { label: 'Zero', data: zero, showLine: true, pointRadius: 0, borderWidth: 3, borderColor: '#9e9e9e', backgroundColor: '#9e9e9e' }
                            ]
                          }}
                          options={{
                            responsive: true, maintainAspectRatio: false, parsing: false,
                            plugins: { legend: { position: 'bottom' } },
                            scales: { x: { type: 'linear', title: { display: true, text: 'Predito' } }, y: { type: 'linear', title: { display: true, text: 'Resíduo' } } }
                          }}
                        />
                      </Box>
                    </Grid>
                  </Grid>

                  {!(yT_sel.length || yT_best.length) && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      Este resultado não inclui séries <code>y_true/preds</code>. Ative o salvamento no backend para ver os diagramas.
                    </Alert>
                  )}

                  {/* Importâncias */}
                  <Box sx={{ mt: 2 }}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1, gap: 1, flexWrap: 'wrap' }}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Chip size="small" variant="outlined" label={`Selecionado:  ${shortName(selected?.modelFile || selected?.model_file)}`} />
                        {showBoth && (
                          <Chip size="small" color="warning" icon={<EmojiEventsIcon />} label={`BEST: ${shortName(best?.modelFile || best?.model_file)}`} />
                        )}
                      </Stack>
                      <TextField
                        label="Top-K" type="number" size="small" value={topK}
                        onChange={(e) => {
                          const v = parseInt(e.target.value, 10);
                          setTopK(Number.isFinite(v) ? Math.max(3, Math.min(50, v)) : 20);
                        }}
                        sx={{ width: 100 }}
                      />
                    </Stack>

                    {labelsImp.length ? (
                      <Box sx={{ height: 360 }}>
                        <Bar
                          data={{
                            labels: labelsImp,
                            datasets: [
                              { label: 'Selecionado |Imp|', data: dataImpSel, borderColor: COLOR_SEL, backgroundColor: COLOR_SEL },
                              ...(showBoth ? [{ label: 'BEST |Imp|', data: dataImpBest, borderColor: COLOR_BEST, backgroundColor: COLOR_BEST }] : []),
                            ]
                          }}
                          options={{
                            responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } },
                            scales: { x: { title: { display: true, text: 'Features (ordenadas por |Imp| do selecionado)' } }, y: { title: { display: true, text: '|Importância|' }, beginAtZero: true } }
                          }}
                        />
                      </Box>
                    ) : (
                      <Alert severity="info">Este resultado não inclui importâncias de features.</Alert>
                    )}
                  </Box>
                </Box>
              )}

              {/* Teste Rápido */}
              {tab === 1 && (
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} sm={6}>
                          <Autocomplete
                            options={expList}
                            value={expSel}
                            onChange={(_, v) => setExpSel(v)}
                            getOptionLabel={(o) => {
                              const id = o?.id ?? o?.uuid ?? o?._id ?? '—';
                              const ns = o?.numeroSerie ?? o?.serial ?? '';
                              const p = extractProtoId(o) ?? 's/ proto';
                              return `${String(id).slice(0, 8)}… ${ns} · proto:${p}`;
                            }}
                            isOptionEqualToValue={(a, b) =>
                              (a?.id ?? a?.uuid ?? a?._id) === (b?.id ?? b?.uuid ?? b?._id)
                            }
                            renderInput={(params) => (<TextField {...params} label="Escolha experimento por protocolo" />)}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            fullWidth type="number" label="2. Ponto de corte (minutos)"
                            value={sliceStart} onChange={e => setSliceStart(e.target.value)}
                          />
                        </Grid>
                        <Grid item xs={12} sm={12}>
                          <Button
                            fullWidth size="large" variant="contained"
                            disabled={isLoading || !selectedTrial || !selected}
                            onClick={handleRunAll}
                          >
                            Rodar previsão completa
                          </Button>
                        </Grid>
                      </Grid>
                    </Paper>
                  </Grid>

                  {/* Resultado calibrado: ŷ */}
                  {quickResult?.yPredicted != null && (
                    <Grid item xs={12}>
                      <Paper variant="outlined" sx={{ p: 2 }}>
                        <Typography variant="h6">Predição (calibração)</Typography>
                        <Typography sx={{ mt: 1 }}>
                          ŷ = <strong>{Number(quickResult.yPredicted).toFixed(3)}</strong> {unit}
                        </Typography>
                      </Paper>
                    </Grid>
                  )}

                  {/* Gráfico do forecaster (real vs previsão por canal) */}
                  {chartDataQuick?.datasets?.length > 0 && (
                    <Grid item xs={12}>
                      <Paper variant="outlined" sx={{ p: 2 }}>
                        <Typography variant="subtitle1">Séries previstas pelo forecaster</Typography>
                        <Box sx={{ height: 350, mt: 2 }}>
                          <Line
                            data={chartDataQuick}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              interaction: { mode: 'index', intersect: false },
                              plugins: { legend: { position: 'top' }, tooltip: { mode: 'index', intersect: false } },
                              parsing: false,
                              scales: {
                                x: { type: 'linear', title: { display: true, text: 'Tempo (minutos)' } },
                                y: { type: 'linear', title: { display: true, text: 'Intensidade (basic counts)' } },
                              },
                            }}
                          />
                        </Box>
                      </Paper>
                    </Grid>
                  )}
                </Grid>
              )}
            </Box>
          </Grid>
        </Grid>

        {/* RODAPÉ PADRÃO: só Voltar */}
        <Stack direction="row" justifyContent="flex-start" sx={{ mt: 3 }}>
          <Button startIcon={<ArrowBackIcon />} onClick={onBack}>Voltar</Button>
        </Stack>
      </Paper>
    </>
  );
});

// ──────────────────────────────────────────────────────────────────────────────
// Componente Principal (Container)
// ──────────────────────────────────────────────────────────────────────────────
export default function CreateForecasterPage() {
  const nav = useNavigate();
  const fileInput = useRef();

  const [step, setStep] = useState(0);
  const [protocols, setProtocols] = useState([]);
  const [loading, setLoading] = useState({ protocols: true, train: false });
  const [fctRes, setFctRes] = useState(null);

  const [form, setForm] = useState({
    protocolId: '',
    units: [],
    localFile: null,
    test_size: '20',
    slice_start: '0',
    slice_end: '0',
    sensors: [],
    channelMap: {},
    fct: { ...defaultFct, fct_bidirectional: false },
    fct_grid: {},
    fct_grid_ui: {},
  });

  const handleFormChange = useCallback((key, value) => {
    setForm(prev => (key === null && typeof value === 'function') ? value(prev) : { ...prev, [key]: value });
  }, []);

  useEffect(() => {
    getProtocolos()
      .then(data => setProtocols(data || []))
      .catch(() => setProtocols([]))
      .finally(() => setLoading(p => ({ ...p, protocols: false })));
  }, []);

  const handleExecuteTraining = useCallback(async () => {
    setLoading(prev => ({ ...prev, train: true }));
    try {
      const payload = buildNewPayload(form);
      const res = await fetch(`${API_BASE}/train_forecaster`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setFctRes(data);
      setStep(4);
    } catch (e) {
      alert(`Falha no treinamento: ${e.message}`);
    } finally {
      setLoading(prev => ({ ...prev, train: false }));
    }
  }, [form]);

  const handleStep = useCallback((newStep) => setStep(newStep), []);
  const stepsTxt = ['Protocolo', 'Dataset', 'Sensores', 'Forecaster', 'Resultados', 'Regressor', 'Predição Final'];

  // Estado do Regressor
  const [selectedTrialForReg, setSelectedTrialForReg] = useState(null);
  const [regForm, setRegForm] = useState({
    models: ['random_forest'],
    params: { ...REG_DEFAULTS },
    grids: {},
    testSize: 0.2,
    permImp: false,
  });
  const [regRes, setRegRes] = useState(null);
  const [regInfer, setRegInfer] = useState(null);
  const [loadingRegressor, setLoadingRegressor] = useState(false);

  useEffect(() => {
    const h = (e) => { setSelectedTrialForReg(e.detail.trial); setStep(5); };
    window.addEventListener('goRegressor', h);
    return () => window.removeEventListener('goRegressor', h);
  }, []);

  const handleTrainRegressor = useCallback(async () => {
    if (!selectedTrialForReg?.featureFile) {
      alert('Trial inválido (sem featureFile).');
      return;
    }
    try {
      setLoadingRegressor(true);
      // usa o test_size da etapa 2 (percentual) convertendo para fração [0.05..0.95]
      const ts = (() => {
        const n = Number(String(form.test_size).replace(',', '.'));
        if (!Number.isFinite(n)) return 0.2;
        return Math.max(0.05, Math.min(0.95, n / 100));
      })();
      const payload = buildRegressorPayload(
        selectedTrialForReg.featureFile,
        { ...regForm, testSize: ts }
      ); const r = await fetch(`${API_BASE}/train_regressor`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const text = await r.text();
      let js; try { js = JSON.parse(text); } catch { /* texto puro */ }
      if (!r.ok) {
        const msg = js?.detail || js?.error || text;
        throw new Error(msg);
      }
      setRegRes(js);
      setStep(6);
    } catch (e) {
      console.error('Erro completo do backend:', e);
      alert(`Erro treinando regressor:\n${e?.message ?? String(e)}`);
    } finally {
      setLoadingRegressor(false);
    }
  }, [selectedTrialForReg, regForm, form.test_size]);

  const handleInferFinal = useCallback(async (regressorFile) => {
    try {
      const all = await getExperimentos();
      const any = all?.[0];
      if (!any) { alert('Nenhum experimento disponível para inferir.'); return; }
      const frames = await getExperimentosDados(any.id);
      const sorted = (frames || []).sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
      const ensaio = buildEnsaioFromCRM(any, sorted);

      const body = {
        ensaio,
        sensors: Object.entries(fctRes?.channels || {}).map(([s, chs]) => `${s}:${chs.join(',')}`),
        units: form.units,
        forecasterFile: (selectedTrialForReg?.modelFile || ''),
        regressorFile,
        sliceStart: 0,
      };

      const r = await fetch(`${API_BASE}/infer_regressor`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      const js = await r.json();
      if (!r.ok) {
        const msg = js?.detail || js?.error || JSON.stringify(js);
        throw new Error(msg);
      }
      setRegInfer(js?.yPredicted);
    } catch (e) {
      alert(`Erro na inferência final: ${e.message || e}`);
    }
  }, [fctRes?.channels, form.units, selectedTrialForReg]);

  const currentStepComponent = () => {
    switch (step) {
      case 0: return <StepProtocols protocols={protocols} form={form} onFormChange={handleFormChange} onNext={() => handleStep(1)} />;
      case 1: return <StepDataset form={form} onFormChange={handleFormChange} fileInputRef={fileInput} onNext={() => handleStep(2)} onBack={() => handleStep(0)} />;
      case 2: return <StepSensors form={form} onFormChange={handleFormChange} onNext={() => handleStep(3)} onBack={() => handleStep(1)} />;
      case 3: return <StepForecaster form={form} onFormChange={handleFormChange} loading={loading.train} onExecute={handleExecuteTraining} onBack={() => handleStep(2)} />;
      case 4: return <StepResults form={form} fctRes={fctRes} onBack={() => handleStep(3)} />;
      case 5: return (
        <StepRegressor
          selectedTrial={selectedTrialForReg}
          regForm={regForm}
          setRegForm={setRegForm}
          onTrain={handleTrainRegressor}
          loadingRegressor={loadingRegressor}
          onBack={() => setStep(4)}
          datasetTestSize={(() => {
            const n = Number(String(form.test_size).replace(',', '.'));
            return Number.isFinite(n) ? n / 100 : 0.2;
          })()
          }
        />
      );
      case 6: return (
        <StepRegressorResults
          form={form}
          selectedTrial={selectedTrialForReg}
          regRes={regRes}
          regInfer={regInfer}
          onInferFinal={handleInferFinal}
          onBack={() => setStep(5)}
        />
      );
      default: return null;
    }
  };

  return (
    <Container sx={{ py: 4 }}>
      <Stack direction="row" spacing={1} alignItems="center" mb={2}>
        <Button size="small" variant="outlined" onClick={() => nav(-1)}>Voltar</Button>
        <Typography variant="h4" color="primary">BioAiLab • Forecaster</Typography>
      </Stack>
      <Divider sx={{ mb: 3 }} />
      <Stepper activeStep={step} sx={{ mb: 3 }}>
        {stepsTxt.map(t => <Step key={t}><StepLabel>{t}</StepLabel></Step>)}
      </Stepper>

      {loading.protocols ? <CircularProgress /> : currentStepComponent()}
    </Container>
  );
}
