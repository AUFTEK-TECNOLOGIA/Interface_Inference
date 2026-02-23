import { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";
import { useI18n } from "../locale/i18n";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8001";

// Nomes amigáveis dos sensores
const SENSOR_NAMES = {
  turbidimetry: "Turbidimetria",
  nephelometry: "Nefelometria",
  fluorescence: "Fluorescência",
  temperature: "Temperatura",
};

// Ordem preferida dos sensores
const SENSOR_ORDER = ["turbidimetry", "nephelometry", "fluorescence", "temperature"];

/**
 * Componente para criar e selecionar datasets de treinamento.
 * 
 * Layout: Split-view com lista à esquerda e preview à direita
 * 
 * Features:
 * - Navegação por teclado (←/→ para navegar, G=Bom, B=Ruim)
 * - Auto-avançar após classificar
 * - Preview de múltiplos sensores
 * - Gráficos grandes para análise
 */
export default function DatasetSelector({
  tenant,
  protocolId: initialProtocolId,
  selectedExperimentIds,
  onSelectionChange,
  onClose,
  disabled,
}) {
  const i18n = useI18n();
  const t = typeof i18n?.t === "function" ? i18n.t : (key) => key;
  const containerRef = useRef(null);
  
  // Modo: "home" (tela inicial), "select" (seleção), "load" (lista salvos)
  const [mode, setMode] = useState("home");
  const [error, setError] = useState("");
  
  // Protocol ID editável (para criar novo)
  const [protocolId, setProtocolId] = useState(initialProtocolId || "");
  
  // Lista de experimentos
  const [experiments, setExperiments] = useState([]);
  const [experimentsLoading, setExperimentsLoading] = useState(false);
  
  // Seleção e classificação
  const [selected, setSelected] = useState(new Set(selectedExperimentIds || []));
  const [ratings, setRatings] = useState({}); // "good" | "bad" | null
  const [viewed, setViewed] = useState(new Set()); // IDs já visualizados
  
  // Dataset carregado (para edição)
  const [loadedDataset, setLoadedDataset] = useState(null); // {id, name, ...}
  
  // Datasets salvos
  const [datasets, setDatasets] = useState([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  
  // Filtros
  const [filterLabOnly, setFilterLabOnly] = useState(true);
  const [filterRating, setFilterRating] = useState("all");
  const [filterViewed, setFilterViewed] = useState("all"); // "all" | "viewed" | "not-viewed"
  const [searchQuery, setSearchQuery] = useState("");
  
  // Criar dataset
  const [newDatasetName, setNewDatasetName] = useState("");
  const [newDatasetDesc, setNewDatasetDesc] = useState("");
  const [saving, setSaving] = useState(false);
  
  // Preview com múltiplos sensores (agora sempre visível no painel direito)
  const [previewExpId, setPreviewExpId] = useState(null);
  const [previewIndex, setPreviewIndex] = useState(-1);
  const [previewGraphs, setPreviewGraphs] = useState({});
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewInfo, setPreviewInfo] = useState(null);
  const [previewSensor, setPreviewSensor] = useState(null);
  const [autoAdvance, setAutoAdvance] = useState(true);

  // Edição de lab results (mock)
  const [editingLabResults, setEditingLabResults] = useState(false);
  const [labResultsData, setLabResultsData] = useState(null); // calibration raw data
  const [bacteriaOptions, setBacteriaOptions] = useState([]);
  const [labResultsSaving, setLabResultsSaving] = useState(false);

  // Cache para pré-carregamento de experimentos
  const previewCacheRef = useRef(new Map());
  const prefetchingRef = useRef(new Set());
  const [cacheSize, setCacheSize] = useState(0);

  // Limpar cache quando protocolId mudar
  useEffect(() => {
    previewCacheRef.current.clear();
    prefetchingRef.current.clear();
    setCacheSize(0);
  }, [protocolId]);

  // Atualizar contador de cache periodicamente
  useEffect(() => {
    const interval = setInterval(() => {
      setCacheSize(previewCacheRef.current.size);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Carregar experimentos (só quando sair do modo home)
  useEffect(() => {
    if (!tenant || !protocolId || mode === "home") return;
    
    const loadExperiments = async () => {
      setExperimentsLoading(true);
      setError("");
      try {
        const res = await axios.get(`${API_URL}/datasets/experiments/${tenant}/${protocolId}`);
        setExperiments(res.data.experiments || []);
      } catch (err) {
        setError(extractErrorMessage(err));
      } finally {
        setExperimentsLoading(false);
      }
    };
    
    loadExperiments();
  }, [tenant, protocolId, mode]);

  // Carregar datasets salvos
  useEffect(() => {
    if (!tenant) return;
    
    const loadDatasets = async () => {
      setDatasetsLoading(true);
      try {
        const res = await axios.get(`${API_URL}/datasets/${tenant}`);
        setDatasets(res.data.datasets || []);
      } catch (err) {
        console.error("Erro ao carregar datasets:", err);
      } finally {
        setDatasetsLoading(false);
      }
    };
    
    loadDatasets();
  }, [tenant]);

  // Sincronizar seleção externa
  useEffect(() => {
    setSelected(new Set(selectedExperimentIds || []));
  }, [selectedExperimentIds]);

  // Filtrar experimentos
  const filteredExperiments = useCallback(() => {
    let list = experiments;
    
    if (filterLabOnly) {
      list = list.filter((e) => e.has_lab_results);
    }
    
    if (filterRating !== "all") {
      list = list.filter((e) => {
        const rating = ratings[e.experiment_id];
        if (filterRating === "good") return rating === "good";
        if (filterRating === "bad") return rating === "bad";
        if (filterRating === "unrated") return !rating;
        return true;
      });
    }
    
    if (filterViewed !== "all") {
      list = list.filter((e) => {
        const isViewed = viewed.has(e.experiment_id);
        if (filterViewed === "viewed") return isViewed;
        if (filterViewed === "not-viewed") return !isViewed;
        return true;
      });
    }
    
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter((e) =>
        e.experiment_id?.toLowerCase().includes(q) ||
        e.name?.toLowerCase().includes(q) ||
        e.labels?.some((l) => l.toLowerCase().includes(q))
      );
    }
    
    return list;
  }, [experiments, filterLabOnly, filterRating, filterViewed, ratings, viewed, searchQuery]);

  const filtered = filteredExperiments();

  // Toggle seleção
  const toggleExperiment = (expId) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(expId)) {
        next.delete(expId);
      } else {
        next.add(expId);
      }
      return next;
    });
  };

  // Classificar experimento
  const rateExperiment = useCallback((expId, rating) => {
    setRatings((prev) => {
      const current = prev[expId];
      return {
        ...prev,
        [expId]: current === rating ? null : rating,
      };
    });
    
    // Auto-avançar para o próximo após classificar
    if (autoAdvance && previewExpId === expId) {
      setTimeout(() => {
        navigatePreview(1);
      }, 300);
    }
  }, [autoAdvance, previewExpId]);

  // Selecionar todos bons
  const selectAllGood = () => {
    const goodIds = Object.entries(ratings)
      .filter(([_, r]) => r === "good")
      .map(([id]) => id);
    setSelected(new Set(goodIds));
  };

  // Selecionar todos visíveis
  const selectAll = () => {
    setSelected((prev) => {
      const next = new Set(prev);
      for (const e of filtered) {
        next.add(e.experiment_id);
      }
      return next;
    });
  };

  // Limpar seleção
  const clearSelection = () => {
    setSelected(new Set());
  };

  // Pré-carregar experimento em background (sem atualizar UI)
  const prefetchPreview = useCallback(async (expId) => {
    // Já está no cache ou sendo carregado
    if (previewCacheRef.current.has(expId) || prefetchingRef.current.has(expId)) {
      return;
    }
    
    prefetchingRef.current.add(expId);
    
    try {
      const res = await axios.get(`${API_URL}/datasets/preview/${tenant}/${expId}`);
      const graphs = res.data.graphs || {};
      const info = {
        dataPoints: res.data.data_points,
        durationMinutes: res.data.duration_minutes,
        sensors: res.data.sensors || [],
        experiment: res.data.experiment || {},
        labResults: res.data.lab_results || [],
      };
      
      // Salvar no cache
      previewCacheRef.current.set(expId, { graphs, info });
      setCacheSize(previewCacheRef.current.size);
    } catch (err) {
      // Silencioso - prefetch falhou, vai carregar normal quando necessário
      console.debug("Prefetch falhou para", expId);
    } finally {
      prefetchingRef.current.delete(expId);
    }
  }, [tenant]);

  // Pré-carregar experimentos adjacentes
  const prefetchAdjacent = useCallback((currentIndex, count = 3) => {
    // Pré-carregar os próximos N experimentos
    for (let i = 1; i <= count; i++) {
      const nextIdx = currentIndex + i;
      if (nextIdx < filtered.length) {
        prefetchPreview(filtered[nextIdx].experiment_id);
      }
    }
    // Pré-carregar o anterior também
    if (currentIndex > 0) {
      prefetchPreview(filtered[currentIndex - 1].experiment_id);
    }
  }, [filtered, prefetchPreview]);

  // Carregar preview (agora abre no painel lateral)
  const loadPreview = async (expId, index) => {
    setPreviewExpId(expId);
    setPreviewIndex(index);
    
    // Marcar como visto
    setViewed((prev) => new Set([...prev, expId]));
    
    // Verificar se já temos no cache
    const cached = previewCacheRef.current.get(expId);
    if (cached) {
      setPreviewGraphs(cached.graphs);
      setPreviewInfo(cached.info);
      
      // Selecionar primeiro sensor na ordem preferida
      const availableSensors = Object.keys(cached.graphs);
      const firstSensor = SENSOR_ORDER.find((s) => availableSensors.includes(s)) || availableSensors[0];
      setPreviewSensor(firstSensor);
      setPreviewLoading(false);
      
      // Pré-carregar adjacentes em background
      prefetchAdjacent(index);
      return;
    }
    
    // Não está no cache, carregar normalmente
    setPreviewGraphs({});
    setPreviewInfo(null);
    setPreviewSensor(null);
    setPreviewLoading(true);
    
    try {
      const res = await axios.get(`${API_URL}/datasets/preview/${tenant}/${expId}`);
      const graphs = res.data.graphs || {};
      const info = {
        dataPoints: res.data.data_points,
        durationMinutes: res.data.duration_minutes,
        sensors: res.data.sensors || [],
        experiment: res.data.experiment || {},
        labResults: res.data.lab_results || [],
      };
      
      setPreviewGraphs(graphs);
      setPreviewInfo(info);
      
      // Salvar no cache
      previewCacheRef.current.set(expId, { graphs, info });
      setCacheSize(previewCacheRef.current.size);
      
      // Selecionar primeiro sensor na ordem preferida
      const availableSensors = Object.keys(graphs);
      const firstSensor = SENSOR_ORDER.find((s) => availableSensors.includes(s)) || availableSensors[0];
      setPreviewSensor(firstSensor);
      
      // Pré-carregar adjacentes em background
      prefetchAdjacent(index);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setPreviewLoading(false);
    }
  };

  // Navegação entre experimentos
  const navigatePreview = useCallback((direction) => {
    const newIndex = previewIndex + direction;
    if (newIndex >= 0 && newIndex < filtered.length) {
      const exp = filtered[newIndex];
      loadPreview(exp.experiment_id, newIndex);
    }
  }, [previewIndex, filtered]);

  // Selecionar primeiro experimento automaticamente
  useEffect(() => {
    if (filtered.length > 0 && !previewExpId && !experimentsLoading) {
      loadPreview(filtered[0].experiment_id, 0);
    }
  }, [filtered, experimentsLoading]);

  // Atalhos de teclado
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignorar se estiver em um input
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      
      if (mode === "select" && previewExpId) {
        switch (e.key) {
          case "ArrowLeft":
            e.preventDefault();
            navigatePreview(-1);
            break;
          case "ArrowRight":
            e.preventDefault();
            navigatePreview(1);
            break;
          case "ArrowUp":
            e.preventDefault();
            navigatePreview(-1);
            break;
          case "ArrowDown":
            e.preventDefault();
            navigatePreview(1);
            break;
          case "g":
          case "G":
            e.preventDefault();
            rateExperiment(previewExpId, "good");
            break;
          case "b":
          case "B":
            e.preventDefault();
            rateExperiment(previewExpId, "bad");
            break;
          case "Escape":
            e.preventDefault();
            onClose?.();
            break;
          case " ":
            e.preventDefault();
            toggleExperiment(previewExpId);
            break;
          case "1":
          case "2":
          case "3":
          case "4":
            e.preventDefault();
            const sensorIdx = parseInt(e.key) - 1;
            if (sortedSensors[sensorIdx]) {
              setPreviewSensor(sortedSensors[sensorIdx]);
            }
            break;
        }
      }
    };
    
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [mode, previewExpId, navigatePreview, rateExperiment, onClose]);

  // Aplicar seleção
  const applySelection = () => {
    if (onSelectionChange) {
      onSelectionChange(Array.from(selected));
    }
    if (onClose) {
      onClose();
    }
  };

  // Helper para extrair mensagem de erro
  const extractErrorMessage = (err) => {
    const detail = err?.response?.data?.detail;
    if (!detail) return err?.message || "Erro desconhecido";
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // Erros de validação Pydantic
      return detail.map((d) => d?.msg || JSON.stringify(d)).join("; ");
    }
    if (typeof detail === "object") {
      return detail?.msg || JSON.stringify(detail);
    }
    return String(detail);
  };

  // ==========================================================================
  // Lab Results Editing (Mock Data)
  // ==========================================================================

  // Check if current experiment is from mock source
  const isMockExperiment = previewExpId?.startsWith("mock:");

  // Load lab results for editing
  const loadLabResultsForEdit = useCallback(async () => {
    if (!previewExpId || !isMockExperiment) return;
    
    try {
      const [labRes, bacteriaRes] = await Promise.all([
        axios.get(`${API_URL}/mock/experiments/${tenant}/${previewExpId}/lab-results`),
        axios.get(`${API_URL}/mock/bacteria-options`),
      ]);
      
      setLabResultsData(labRes.data.calibration || {});
      setBacteriaOptions(bacteriaRes.data.bacteria_options || []);
      setEditingLabResults(true);
    } catch (err) {
      setError(extractErrorMessage(err));
    }
  }, [previewExpId, tenant, isMockExperiment]);

  // Save lab results changes
  const saveLabResults = async () => {
    if (!previewExpId || !labResultsData) return;
    
    setLabResultsSaving(true);
    try {
      await axios.put(`${API_URL}/mock/experiments/${tenant}/${previewExpId}/lab-results`, {
        calibration: labResultsData,
      });
      
      // Invalidate cache for this experiment
      previewCacheRef.current.delete(previewExpId);
      setCacheSize(previewCacheRef.current.size);
      
      // Reload preview to show updated data
      setEditingLabResults(false);
      
      // Reload preview data
      const res = await axios.get(`${API_URL}/datasets/preview/${tenant}/${previewExpId}`);
      const graphs = res.data.graphs || {};
      const info = {
        dataPoints: res.data.data_points,
        durationMinutes: res.data.duration_minutes,
        sensors: res.data.sensors || [],
        experiment: res.data.experiment || {},
        labResults: res.data.lab_results || [],
      };
      setPreviewGraphs(graphs);
      setPreviewInfo(info);
      previewCacheRef.current.set(previewExpId, { graphs, info });
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLabResultsSaving(false);
    }
  };

  // Update a specific lab result field
  const updateLabResultField = (calibrationId, field, value) => {
    setLabResultsData((prev) => {
      const updated = { ...prev };
      if (!updated[calibrationId]) {
        updated[calibrationId] = {};
      }
      if (field === "count") {
        updated[calibrationId].count = value === "" ? null : parseFloat(value);
      } else {
        updated[calibrationId][field] = value;
      }
      return updated;
    });
  };

  // Add new lab result
  const addNewLabResult = () => {
    // Generate a new UUID for the calibration
    const newId = `new-${Date.now()}`;
    setLabResultsData((prev) => ({
      ...prev,
      [newId]: { count: 0, unit: "NMP/100mL" },
    }));
  };

  // Delete lab result
  const deleteLabResult = (calibrationId) => {
    setLabResultsData((prev) => {
      const updated = { ...prev };
      delete updated[calibrationId];
      return updated;
    });
  };

  // Cancel editing
  const cancelLabResultsEdit = () => {
    setEditingLabResults(false);
    setLabResultsData(null);
  };

  // ==========================================================================
  // End Lab Results Editing
  // ==========================================================================

  // Salvar dataset
  const saveDataset = async () => {
    const nameToSave = loadedDataset?.name || newDatasetName.trim();
    if (!nameToSave || selected.size === 0) return;
    
    setSaving(true);
    setError("");
    
    try {
      const params = new URLSearchParams({
        name: nameToSave,
        protocol_id: protocolId,
        description: loadedDataset?.description || newDatasetDesc.trim(),
        ratings: JSON.stringify(ratings),
      });
      
      // Se estamos editando um dataset existente, passar o ID
      if (loadedDataset?.id) {
        params.append("dataset_id", loadedDataset.id);
      }
      
      for (const expId of selected) {
        params.append("experiment_ids", expId);
      }
      
      for (const expId of viewed) {
        params.append("viewed_ids", expId);
      }
      
      const saveRes = await axios.post(`${API_URL}/datasets/${tenant}?${params.toString()}`);
      
      // Atualizar loadedDataset se estamos editando
      if (loadedDataset?.id) {
        setLoadedDataset((prev) => ({
          ...prev,
          experiment_count: selected.size,
          viewed_count: viewed.size,
        }));
      } else {
        // Novo dataset salvo
        setLoadedDataset({
          id: saveRes.data.dataset_id,
          name: nameToSave,
          description: newDatasetDesc.trim(),
          protocol_id: protocolId,
          experiment_count: selected.size,
          viewed_count: viewed.size,
        });
        setNewDatasetName("");
        setNewDatasetDesc("");
      }
      
      // Recarregar lista de datasets
      const res = await axios.get(`${API_URL}/datasets/${tenant}`);
      setDatasets(res.data.datasets || []);
      
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  // Carregar dataset para edição
  const loadDataset = async (datasetId) => {
    try {
      const res = await axios.get(`${API_URL}/datasets/${tenant}/${datasetId}`);
      const data = res.data;
      
      // Carregar seleção
      setSelected(new Set(data.experiment_ids || []));
      
      // Carregar visualizados
      setViewed(new Set(data.viewed_ids || []));
      
      // Carregar ratings
      setRatings(data.ratings || {});
      
      // Definir protocolId do dataset
      setProtocolId(data.protocol_id || "");
      
      // Guardar referência ao dataset carregado
      setLoadedDataset({
        id: data.id,
        name: data.name,
        description: data.description || "",
        protocol_id: data.protocol_id,
        experiment_count: (data.experiment_ids || []).length,
        viewed_count: (data.viewed_ids || []).length,
      });
      
      // Ir para modo seleção
      setMode("select");
      
    } catch (err) {
      setError(extractErrorMessage(err));
    }
  };
  
  // Fechar dataset e voltar para home
  const closeDataset = () => {
    setLoadedDataset(null);
    setSelected(new Set());
    setViewed(new Set());
    setRatings({});
    setExperiments([]);
    setNewDatasetName("");
    setNewDatasetDesc("");
    setProtocolId("");
    setMode("home");
  };

  // Deletar dataset
  const deleteDataset = async (datasetId) => {
    if (!window.confirm("Tem certeza que deseja excluir este dataset?")) return;
    
    try {
      await axios.delete(`${API_URL}/datasets/${tenant}/${datasetId}`);
      setDatasets((prev) => prev.filter((d) => d.id !== datasetId));
    } catch (err) {
      setError(extractErrorMessage(err));
    }
  };

  // Estatísticas
  const totalWithLab = experiments.filter((e) => e.has_lab_results).length;
  const goodCount = Object.values(ratings).filter((r) => r === "good").length;
  const badCount = Object.values(ratings).filter((r) => r === "bad").length;
  const viewedCount = viewed.size;
  const notViewedCount = experiments.length - viewedCount;

  // Ordenar sensores para exibição
  const sortedSensors = Object.keys(previewGraphs).sort((a, b) => {
    const idxA = SENSOR_ORDER.indexOf(a);
    const idxB = SENSOR_ORDER.indexOf(b);
    return (idxA === -1 ? 999 : idxA) - (idxB === -1 ? 999 : idxB);
  });

  // Experimento atual sendo visualizado
  const currentExp = filtered.find(e => e.experiment_id === previewExpId);

  return (
    <div className="ds2-container" ref={containerRef} tabIndex={-1}>
      {/* Header */}
      <div className="ds2-header">
        <div className="ds2-header-left">
          {mode === "home" ? (
            <>
              <h2>Datasets</h2>
              <span className="ds2-header-subtitle">{tenant}</span>
            </>
          ) : loadedDataset ? (
            <>
              <div className="ds2-loaded-badge">
                <span className="ds2-loaded-icon">📝</span>
                <span className="ds2-loaded-name">{loadedDataset.name}</span>
                <button 
                  type="button" 
                  className="ds2-loaded-close"
                  onClick={closeDataset}
                  title="Voltar para início"
                >
                  ×
                </button>
              </div>
              <span className="ds2-header-subtitle">{protocolId?.slice(0, 16)}...</span>
            </>
          ) : (
            <>
              <button 
                type="button" 
                className="ds2-back-btn"
                onClick={() => setMode("home")}
                title="Voltar"
              >
                ←
              </button>
              <h2>Novo Dataset</h2>
              <span className="ds2-header-subtitle">{protocolId?.slice(0, 16)}...</span>
            </>
          )}
        </div>
        
        {mode !== "home" && (
          <div className="ds2-header-stats">
            <div className="ds2-stat">
              <span className="ds2-stat-value">{filtered.length}</span>
              <span className="ds2-stat-label">experimentos</span>
            </div>
            <div className="ds2-stat ds2-stat-viewed" title="Já visualizados">
              <span className="ds2-stat-value">{viewedCount}</span>
              <span className="ds2-stat-label">vistos</span>
            </div>
            <div className="ds2-stat ds2-stat-good">
              <span className="ds2-stat-value">{goodCount}</span>
              <span className="ds2-stat-label">bons</span>
            </div>
            <div className="ds2-stat ds2-stat-bad">
              <span className="ds2-stat-value">{badCount}</span>
              <span className="ds2-stat-label">ruins</span>
            </div>
            <div className="ds2-stat ds2-stat-selected">
              <span className="ds2-stat-value">{selected.size}</span>
              <span className="ds2-stat-label">selecionados</span>
            </div>
            {cacheSize > 0 && (
              <div className="ds2-stat ds2-stat-cache" title="Experimentos pré-carregados em cache">
                <span className="ds2-stat-value">{cacheSize}</span>
                <span className="ds2-stat-label">em cache</span>
              </div>
            )}
          </div>
        )}

        <button type="button" className="ds2-close" onClick={onClose}>
          ✕
        </button>
      </div>

      {/* Tabs - só aparece fora do modo home */}
      {mode !== "home" && (
        <div className="ds2-tabs">
          <div className="ds2-tabs-right">
            <div className="ds2-shortcuts">
              <kbd>↑</kbd><kbd>↓</kbd> nav
              <kbd>G</kbd> bom
              <kbd>B</kbd> ruim
              <kbd>1-4</kbd> sensor
            </div>
            <label className="ds2-auto-advance">
              <input
                type="checkbox"
                checked={autoAdvance}
                onChange={(e) => setAutoAdvance(e.target.checked)}
              />
              Auto-avançar
            </label>
          </div>
        </div>
      )}

      {error && (
        <div className="ds2-error">
          <span>{error}</span>
          <button type="button" onClick={() => setError("")}>×</button>
        </div>
      )}

      {/* Tela inicial - Home */}
      {mode === "home" && (
        <div className="ds2-home">
          <div className="ds2-home-content">
            {/* Criar novo */}
            <div className="ds2-home-section">
              <h3>Criar Novo Dataset</h3>
              <div className="ds2-home-new">
                <input
                  type="text"
                  className="ds2-home-input"
                  placeholder="Cole o Protocol ID..."
                  value={protocolId}
                  onChange={(e) => setProtocolId(e.target.value)}
                />
                <button
                  type="button"
                  className="ds2-home-btn-new"
                  disabled={!protocolId.trim()}
                  onClick={() => {
                    if (protocolId.trim()) {
                      setMode("select");
                    }
                  }}
                >
                  Criar Dataset
                </button>
              </div>
            </div>

            {/* Datasets existentes */}
            <div className="ds2-home-section">
              <h3>Seus Datasets</h3>
              {datasetsLoading ? (
                <div className="ds2-loading">
                  <div className="ds2-spinner" />
                  <span>Carregando...</span>
                </div>
              ) : datasets.length === 0 ? (
                <div className="ds2-home-empty">
                  <p>Nenhum dataset criado ainda</p>
                  <p className="ds2-home-hint">Cole um Protocol ID acima para começar</p>
                </div>
              ) : (
                <div className="ds2-home-list">
                  {datasets.map((ds) => (
                    <div
                      key={ds.id}
                      className="ds2-home-item"
                      onClick={() => loadDataset(ds.id)}
                    >
                      <div className="ds2-home-item-icon">📊</div>
                      <div className="ds2-home-item-info">
                        <div className="ds2-home-item-name">{ds.name}</div>
                        <div className="ds2-home-item-meta">
                          {ds.experiment_count || 0} experimentos
                          {ds.created_at && ` · ${ds.created_at}`}
                        </div>
                      </div>
                      <div className="ds2-home-item-arrow">→</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Main content - split view (modo select) */}
      {mode === "select" && (
        <div className="ds2-main">
          <>
            {/* Lista de experimentos (esquerda) */}
            <div className="ds2-list-panel">
              {/* Filtros */}
              <div className="ds2-filters">
                <input
                  type="text"
                  className="ds2-search"
                  placeholder="Buscar..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <label className="ds2-filter-check">
                  <input
                    type="checkbox"
                    checked={filterLabOnly}
                    onChange={(e) => setFilterLabOnly(e.target.checked)}
                  />
                  LAB
                </label>
                <select
                  className="ds2-filter-select"
                  value={filterRating}
                  onChange={(e) => setFilterRating(e.target.value)}
                >
                  <option value="all">Nota</option>
                  <option value="good">✓ Bons</option>
                  <option value="bad">✗ Ruins</option>
                  <option value="unrated">Sem nota</option>
                </select>
                <select
                  className="ds2-filter-select"
                  value={filterViewed}
                  onChange={(e) => setFilterViewed(e.target.value)}
                >
                  <option value="all">Todos</option>
                  <option value="not-viewed">Novos ({notViewedCount})</option>
                  <option value="viewed">Vistos ({viewedCount})</option>
                </select>
              </div>

              {/* Ações em lote */}
              <div className="ds2-batch">
                <button type="button" onClick={selectAll}>
                  Todos ({filtered.length})
                </button>
                <button type="button" className="ds2-batch-good" onClick={selectAllGood} disabled={goodCount === 0}>
                  ✓ Bons ({goodCount})
                </button>
                <button type="button" onClick={clearSelection}>
                  Limpar
                </button>
              </div>

              {/* Lista */}
              <div className="ds2-list">
                {experimentsLoading ? (
                  <div className="ds2-loading">
                    <div className="ds2-spinner" />
                    <span>Carregando...</span>
                  </div>
                ) : filtered.length === 0 ? (
                  <div className="ds2-empty">
                    Nenhum experimento encontrado
                  </div>
                ) : (
                  filtered.map((exp, index) => (
                    <div
                      key={exp.experiment_id}
                      className={`ds2-item ${selected.has(exp.experiment_id) ? "is-selected" : ""} ${ratings[exp.experiment_id] === "good" ? "is-good" : ""} ${ratings[exp.experiment_id] === "bad" ? "is-bad" : ""} ${previewExpId === exp.experiment_id ? "is-active" : ""} ${viewed.has(exp.experiment_id) ? "is-viewed" : "is-new"}`}
                      onClick={() => loadPreview(exp.experiment_id, index)}
                    >
                      <input
                        type="checkbox"
                        checked={selected.has(exp.experiment_id)}
                        onChange={(e) => {
                          e.stopPropagation();
                          toggleExperiment(exp.experiment_id);
                        }}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <span className="ds2-item-num">{index + 1}</span>
                      <div className="ds2-item-info">
                        <span className="ds2-item-id">
                          {exp.experiment_id.slice(0, 12)}...
                          {!viewed.has(exp.experiment_id) && <span className="ds2-new-dot" title="Novo">●</span>}
                        </span>
                        <div className="ds2-item-badges">
                          {exp.has_lab_results && <span className="ds2-badge-lab">LAB</span>}
                          {ratings[exp.experiment_id] === "good" && <span className="ds2-badge-good">✓</span>}
                          {ratings[exp.experiment_id] === "bad" && <span className="ds2-badge-bad">✗</span>}
                        </div>
                      </div>
                      <span className="ds2-item-pts">{exp.data_points} pts</span>
                    </div>
                  ))
                )}
              </div>

              {/* Painel de salvar dataset (inline) */}
              {(selected.size > 0 || loadedDataset) && (
                <div className="ds2-save-inline">
                  <div className="ds2-save-header">
                    {loadedDataset ? (
                      <span>Atualizar "{loadedDataset.name}" ({selected.size} exp, {viewedCount} vistos)</span>
                    ) : (
                      <span>Salvar {selected.size} experimentos</span>
                    )}
                  </div>
                  <div className="ds2-save-form">
                    {!loadedDataset && (
                      <input
                        type="text"
                        className="ds2-save-name"
                        value={newDatasetName}
                        onChange={(e) => setNewDatasetName(e.target.value)}
                        placeholder="Nome do dataset..."
                        disabled={saving}
                      />
                    )}
                    <button
                      type="button"
                      className="ds2-save-btn"
                      onClick={saveDataset}
                      disabled={saving || (!loadedDataset && !newDatasetName.trim()) || selected.size === 0}
                    >
                      {saving ? "..." : loadedDataset ? "Atualizar" : "Salvar"}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Preview (direita) */}
            <div className="ds2-preview-panel">
              {previewLoading ? (
                <div className="ds2-preview-loading">
                  <div className="ds2-spinner" />
                  <span>Carregando gráficos...</span>
                </div>
              ) : previewExpId && sortedSensors.length > 0 ? (
                <>
                  {/* Header do preview */}
                  <div className="ds2-preview-header">
                    <div className="ds2-preview-nav">
                      <button
                        type="button"
                        onClick={() => navigatePreview(-1)}
                        disabled={previewIndex <= 0}
                      >
                        Anterior
                      </button>
                      <span className="ds2-preview-counter">
                        {previewIndex + 1} / {filtered.length}
                      </span>
                      <button
                        type="button"
                        onClick={() => navigatePreview(1)}
                        disabled={previewIndex >= filtered.length - 1}
                      >
                        Próximo
                      </button>
                    </div>
                  </div>

                  {/* Resumo do experimento */}
                  <div className="ds2-experiment-summary">
                    <div className="ds2-summary-row">
                      <div className="ds2-summary-item">
                        <span className="ds2-summary-label">ID</span>
                        <span className="ds2-summary-value ds2-summary-id">{previewExpId?.slice(0, 24)}</span>
                      </div>
                      {previewInfo?.experiment?.created_at && (
                        <div className="ds2-summary-item">
                          <span className="ds2-summary-label">Data</span>
                          <span className="ds2-summary-value">{previewInfo.experiment.created_at}</span>
                        </div>
                      )}
                      <div className="ds2-summary-item">
                        <span className="ds2-summary-label">Duração</span>
                        <span className="ds2-summary-value">{previewInfo?.durationMinutes || 0} min</span>
                      </div>
                      <div className="ds2-summary-item">
                        <span className="ds2-summary-label">Pontos</span>
                        <span className="ds2-summary-value">{previewInfo?.dataPoints || 0}</span>
                      </div>
                      {previewInfo?.experiment?.diluicao_display && (
                        <div className="ds2-summary-item ds2-summary-diluicao">
                          <span className="ds2-summary-label">Diluição</span>
                          <span className="ds2-summary-value">{previewInfo.experiment.diluicao_display}</span>
                        </div>
                      )}
                      {previewInfo?.experiment?.device && (
                        <div className="ds2-summary-item">
                          <span className="ds2-summary-label">Dispositivo</span>
                          <span className="ds2-summary-value ds2-summary-id">{previewInfo.experiment.device}</span>
                        </div>
                      )}
                    </div>

                    {/* Resultados do laboratório */}
                    {previewInfo?.labResults?.length > 0 && !editingLabResults && (
                      <div className="ds2-lab-results">
                        <div className="ds2-lab-header">
                          <span className="ds2-lab-title">Resultados de Laboratório:</span>
                          {isMockExperiment && (
                            <button
                              type="button"
                              className="ds2-lab-edit-btn"
                              onClick={loadLabResultsForEdit}
                              title="Editar resultados"
                            >
                              ✏️ Editar
                            </button>
                          )}
                        </div>
                        <div className="ds2-lab-list">
                          {previewInfo.labResults.map((lr, idx) => (
                            <div key={idx} className="ds2-lab-item">
                              <span className="ds2-lab-bacteria">{lr.bacteria || "Amostra"}</span>
                              {lr.count != null && (
                                <span className="ds2-lab-count">
                                  {typeof lr.count === 'number' ? lr.count.toLocaleString() : lr.count} {lr.unit || "UFC/100mL"}
                                </span>
                              )}
                              {lr.presence != null && (
                                <span className={`ds2-lab-presence ${lr.presence ? "positive" : "negative"}`}>
                                  {lr.presence ? "Positivo" : "Negativo"}
                                </span>
                              )}
                              {lr.date && <span className="ds2-lab-date">{lr.date}</span>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Lab Results sem dados - mostrar botão para adicionar se mock */}
                    {(!previewInfo?.labResults || previewInfo.labResults.length === 0) && !editingLabResults && isMockExperiment && (
                      <div className="ds2-lab-results ds2-lab-empty">
                        <div className="ds2-lab-header">
                          <span className="ds2-lab-title">Sem resultados de laboratório</span>
                          <button
                            type="button"
                            className="ds2-lab-edit-btn"
                            onClick={loadLabResultsForEdit}
                            title="Adicionar resultados"
                          >
                            ➕ Adicionar
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Editor de Lab Results */}
                    {editingLabResults && labResultsData && (
                      <div className="ds2-lab-editor">
                        <div className="ds2-lab-editor-header">
                          <span className="ds2-lab-title">Editar Resultados de Laboratório</span>
                          <div className="ds2-lab-editor-actions">
                            <button
                              type="button"
                              className="ds2-lab-add-btn"
                              onClick={addNewLabResult}
                            >
                              ➕ Adicionar
                            </button>
                          </div>
                        </div>
                        <div className="ds2-lab-editor-list">
                          {Object.entries(labResultsData).map(([calibrationId, item]) => {
                            const bacteriaName = bacteriaOptions.find((b) => b.id === calibrationId)?.name || "";
                            return (
                              <div key={calibrationId} className="ds2-lab-editor-item">
                                <div className="ds2-lab-editor-row">
                                  <div className="ds2-lab-editor-field">
                                    <label>Bactéria</label>
                                    <select
                                      value={calibrationId}
                                      onChange={(e) => {
                                        // Change calibration ID
                                        const newId = e.target.value;
                                        if (newId && newId !== calibrationId) {
                                          setLabResultsData((prev) => {
                                            const updated = { ...prev };
                                            updated[newId] = updated[calibrationId];
                                            delete updated[calibrationId];
                                            return updated;
                                          });
                                        }
                                      }}
                                    >
                                      <option value={calibrationId}>
                                        {bacteriaName || calibrationId.slice(0, 8)}
                                      </option>
                                      {bacteriaOptions
                                        .filter((b) => b.id !== calibrationId && !labResultsData[b.id])
                                        .map((b) => (
                                          <option key={b.id} value={b.id}>
                                            {b.name}
                                          </option>
                                        ))}
                                    </select>
                                  </div>
                                  <div className="ds2-lab-editor-field">
                                    <label>Contagem</label>
                                    <input
                                      type="number"
                                      value={item.count ?? ""}
                                      onChange={(e) => updateLabResultField(calibrationId, "count", e.target.value)}
                                      placeholder="0"
                                    />
                                  </div>
                                  <div className="ds2-lab-editor-field">
                                    <label>Unidade</label>
                                    <select
                                      value={item.unit || "NMP/100mL"}
                                      onChange={(e) => updateLabResultField(calibrationId, "unit", e.target.value)}
                                    >
                                      <option value="NMP/100mL">NMP/100mL</option>
                                      <option value="NPM/mL">NPM/mL</option>
                                      <option value="UFC/100mL">UFC/100mL</option>
                                      <option value="UFC/mL">UFC/mL</option>
                                    </select>
                                  </div>
                                  <button
                                    type="button"
                                    className="ds2-lab-delete-btn"
                                    onClick={() => deleteLabResult(calibrationId)}
                                    title="Remover"
                                  >
                                    🗑️
                                  </button>
                                </div>
                              </div>
                            );
                          })}
                          {Object.keys(labResultsData).length === 0 && (
                            <div className="ds2-lab-editor-empty">
                              Nenhum resultado. Clique em "Adicionar" para criar.
                            </div>
                          )}
                        </div>
                        <div className="ds2-lab-editor-footer">
                          <button
                            type="button"
                            className="ds2-lab-cancel-btn"
                            onClick={cancelLabResultsEdit}
                            disabled={labResultsSaving}
                          >
                            Cancelar
                          </button>
                          <button
                            type="button"
                            className="ds2-lab-save-btn"
                            onClick={saveLabResults}
                            disabled={labResultsSaving}
                          >
                            {labResultsSaving ? "Salvando..." : "Salvar"}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Tabs de sensores */}
                  <div className="ds2-sensor-tabs">
                    {sortedSensors.map((sensor, idx) => (
                      <button
                        key={sensor}
                        type="button"
                        className={`ds2-sensor-tab ${previewSensor === sensor ? "is-active" : ""}`}
                        onClick={() => setPreviewSensor(sensor)}
                      >
                        <span className="ds2-sensor-num">{idx + 1}</span>
                        {SENSOR_NAMES[sensor] || sensor}
                      </button>
                    ))}
                  </div>

                  {/* Gráfico */}
                  <div className="ds2-graph-container">
                    {previewSensor && previewGraphs[previewSensor] && (
                      <img 
                        src={previewGraphs[previewSensor]} 
                        alt={`${previewSensor}`}
                        className="ds2-graph"
                      />
                    )}
                  </div>

                  {/* Ações de classificação */}
                  <div className="ds2-preview-actions">
                    <button
                      type="button"
                      className={`ds2-action-btn ds2-action-good ${ratings[previewExpId] === "good" ? "is-active" : ""}`}
                      onClick={() => rateExperiment(previewExpId, "good")}
                    >
                      Bom <kbd>G</kbd>
                    </button>
                    <button
                      type="button"
                      className={`ds2-action-btn ds2-action-bad ${ratings[previewExpId] === "bad" ? "is-active" : ""}`}
                      onClick={() => rateExperiment(previewExpId, "bad")}
                    >
                      Ruim <kbd>B</kbd>
                    </button>
                    <button
                      type="button"
                      className={`ds2-action-btn ds2-action-select ${selected.has(previewExpId) ? "is-active" : ""}`}
                      onClick={() => toggleExperiment(previewExpId)}
                    >
                      {selected.has(previewExpId) ? "Selecionado" : "Selecionar"} <kbd>Espaço</kbd>
                    </button>
                  </div>
                </>
              ) : (
                <div className="ds2-preview-empty">
                  <p>Selecione um experimento para visualizar os gráficos</p>
                </div>
              )}
            </div>
          </>
        </div>
      )}

      {/* Footer com ações principais */}
      {mode !== "home" && (
        <div className="ds2-footer">
          <button type="button" className="ds2-btn-cancel" onClick={onClose}>
            Cancelar
          </button>
          <button
            type="button"
            className="ds2-btn-apply"
            onClick={applySelection}
            disabled={selected.size === 0 || disabled}
          >
            Usar {selected.size} experimentos
          </button>
        </div>
      )}
    </div>
  );
}
