import BlocksSidebar from "./BlocksSidebar";

export default function PipelineBlocksPanel({
  t,
  width,
  blocksQuery,
  setBlocksQuery,
  favoriteBlocks,
  recentBlocks,
  library,
  blockMatchesQuery,
  renderBlockCardMini,
  renderBlockCard,
  onStartResize,
}) {
  return (
    <BlocksSidebar
      t={t}
      width={width}
      blocksQuery={blocksQuery}
      setBlocksQuery={setBlocksQuery}
      favoriteBlocks={favoriteBlocks}
      recentBlocks={recentBlocks}
      library={library}
      blockMatchesQuery={blockMatchesQuery}
      renderBlockCardMini={renderBlockCardMini}
      onStartResize={onStartResize}
    >
      {/* 1) Entrada / Fonte */}
      <details className="stage-accordion" open>
        <summary className="stage-summary">
          <div className="stage-header">
            <span className="stage-number">1</span>
            <span className="stage-title">{t("blocksPanel.stages.acquisition")}</span>
          </div>
        </summary>
        <div className="blocks-grid">
          {(library.blocks || [])
            .filter((b) => ["experiment_fetch"].includes(b.name))
            .filter((b) => blockMatchesQuery(b, blocksQuery))
            .map(renderBlockCard)}
        </div>
      </details>

      {/* 2) Roteamento */}
      <details className="stage-accordion" open>
        <summary className="stage-summary">
          <div className="stage-header">
            <span className="stage-number">2</span>
            <span className="stage-title">{t("blocksPanel.stages.routing")}</span>
          </div>
        </summary>
        <div className="blocks-grid">
          {(library.blocks || [])
            .filter((b) =>
              [
                "label",
                "value_in_list",
                "numeric_compare",
                "condition_branch",
                "condition_gate",
                "and_gate",
                "or_gate",
                "not_gate",
                "boolean_extractor",
                "merge",
              ].includes(b.name)
            )
            .filter((b) => blockMatchesQuery(b, blocksQuery))
            .map(renderBlockCard)}
        </div>
      </details>

      {/* 3) Aquisição / Extração de sensores */}
      <details className="stage-accordion" open>
        <summary className="stage-summary">
          <div className="stage-header">
            <span className="stage-number">3</span>
            <span className="stage-title">{t("blocksPanel.stages.sensorExtraction")}</span>
          </div>
        </summary>

        <div className="sensor-category">
          <span className="sensor-category-label">Sensores principais</span>
          <div className="blocks-grid">
            {(library.blocks || [])
              .filter((b) =>
                [
                  "turbidimetry_extraction",
                  "nephelometry_extraction",
                  "fluorescence_extraction",
                  "resonant_frequencies_extraction",
                ].includes(b.name)
              )
              .filter((b) => blockMatchesQuery(b, blocksQuery))
              .map(renderBlockCard)}
          </div>
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Temperaturas</span>
          <div className="blocks-grid">
            {(library.blocks || [])
              .filter((b) => ["temperatures_extraction"].includes(b.name))
              .filter((b) => blockMatchesQuery(b, blocksQuery))
              .map(renderBlockCard)}
          </div>
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Fonte de alimentacao</span>
          <div className="blocks-grid">
            {(library.blocks || [])
              .filter((b) => ["power_supply_extraction"].includes(b.name))
              .filter((b) => blockMatchesQuery(b, blocksQuery))
              .map(renderBlockCard)}
          </div>
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Pastilha Peltier</span>
          <div className="blocks-grid">
            {(library.blocks || [])
              .filter((b) => ["peltier_currents_extraction"].includes(b.name))
              .filter((b) => blockMatchesQuery(b, blocksQuery))
              .map(renderBlockCard)}
          </div>
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Agitador magnetico</span>
          <div className="blocks-grid">
            {(library.blocks || [])
              .filter((b) => ["nema_currents_extraction"].includes(b.name))
              .filter((b) => blockMatchesQuery(b, blocksQuery))
              .map(renderBlockCard)}
          </div>
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Estados de controle</span>
          <div className="blocks-grid">
            {(library.blocks || [])
              .filter((b) => ["control_state_extraction"].includes(b.name))
              .filter((b) => blockMatchesQuery(b, blocksQuery))
              .map(renderBlockCard)}
          </div>
        </div>
      </details>

      {/* 4) Preparação (limpeza) */}
      <details className="stage-accordion" open>
        <summary className="stage-summary">
          <div className="stage-header">
            <span className="stage-number">4</span>
            <span className="stage-title">{t("blocksPanel.stages.preparation")}</span>
          </div>
        </summary>
        <div className="blocks-grid">
          {(library.blocks || [])
            .filter((b) =>
              [
                "sensor_fusion",
                "time_slice",
                "outlier_removal",
                "moving_average_filter",
                "savgol_filter",
                "median_filter",
                "lowpass_filter",
                "exponential_filter",
              ].includes(b.name)
            )
            .filter((b) => blockMatchesQuery(b, blocksQuery))
            .map(renderBlockCard)}
        </div>
      </details>

      {/* 5) Transformações */}
      <details className="stage-accordion" open>
        <summary className="stage-summary">
          <div className="stage-header">
            <span className="stage-number">5</span>
            <span className="stage-title">{t("blocksPanel.stages.transformations")}</span>
          </div>
        </summary>
        <div className="blocks-grid">
          {(library.blocks || [])
            .filter((b) =>
              [
                "derivative",
                "integral",
                "normalize",
                "xyz_conversion",
                "rgb_conversion",
                "lab_conversion",
                "luv_conversion",
                "hsv_conversion",
                "hsl_conversion",
                "hsb_conversion",
                "cmyk_conversion",
                "xyy_conversion",
              ].includes(b.name)
            )
            .filter((b) => blockMatchesQuery(b, blocksQuery))
            .map(renderBlockCard)}
        </div>
      </details>

      {/* 6) Detecção */}
      <details className="stage-accordion" open>
        <summary className="stage-summary">
          <div className="stage-header">
            <span className="stage-number">6</span>
            <span className="stage-title">{t("blocksPanel.stages.detection")}</span>
          </div>
        </summary>
        <div className="blocks-grid">
          {(library.blocks || [])
            .filter((b) => ["amplitude_detector", "derivative_detector", "ratio_detector"].includes(b.name))
            .filter((b) => blockMatchesQuery(b, blocksQuery))
            .map(renderBlockCard)}
        </div>
      </details>

      {/* 7) Features */}
      <details className="stage-accordion" open>
        <summary className="stage-summary">
          <div className="stage-header">
            <span className="stage-number">7</span>
            <span className="stage-title">{t("blocksPanel.stages.features")}</span>
          </div>
        </summary>
        <div className="blocks-grid">
          {(library.blocks || [])
            .filter((b) =>
              [
                "statistical_features",
                "temporal_features",
                "shape_features",
                "growth_features",
                "feature_extraction",
                "features_merge",
              ].includes(b.name)
            )
            .filter((b) => blockMatchesQuery(b, blocksQuery))
            .map(renderBlockCard)}
        </div>
      </details>

      {/* 8) Modelo / Predição */}
      <details className="stage-accordion" open>
        <summary className="stage-summary">
          <div className="stage-header">
            <span className="stage-number">8</span>
            <span className="stage-title">{t("blocksPanel.stages.modelling")}</span>
          </div>
        </summary>
        <div className="blocks-grid">
          {(library.blocks || [])
            .filter((b) =>
              [
                "curve_fit",
                "curve_fit_best",
                "ml_inference",
                "ml_inference_series",
                "ml_inference_multichannel",
                "ml_forecaster_series",
                "ml_transform_series",
                "ml_detector",
              ].includes(b.name)
            )
            .filter((b) => blockMatchesQuery(b, blocksQuery))
            .map(renderBlockCard)}
        </div>
      </details>

      {/* 9) Resposta / Saídas */}
      <details className="stage-accordion" open>
        <summary className="stage-summary">
          <div className="stage-header">
            <span className="stage-number">9</span>
            <span className="stage-title">{t("blocksPanel.stages.response")}</span>
          </div>
        </summary>
        <div className="blocks-grid">
          {(library.blocks || [])
            .filter((b) => ["response_pack", "response_merge", "response_builder"].includes(b.name))
            .filter((b) => blockMatchesQuery(b, blocksQuery))
            .map(renderBlockCard)}
        </div>
      </details>
    </BlocksSidebar>
  );
}
