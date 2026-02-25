import BlocksSidebar from "./BlocksSidebar";
import BlockStagePanel from "./BlockStagePanel";

const renderStageGrid = (library, names, blockMatchesQuery, blocksQuery, renderBlockCard) => (
  <div className="blocks-grid">
    {(library.blocks || [])
      .filter((b) => names.includes(b.name))
      .filter((b) => blockMatchesQuery(b, blocksQuery))
      .map(renderBlockCard)}
  </div>
);

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
      <BlockStagePanel number={1} title={t("blocksPanel.stages.acquisition")}>
        {renderStageGrid(library, ["experiment_fetch"], blockMatchesQuery, blocksQuery, renderBlockCard)}
      </BlockStagePanel>

      <BlockStagePanel number={2} title={t("blocksPanel.stages.routing")}>
        {renderStageGrid(
          library,
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
          ],
          blockMatchesQuery,
          blocksQuery,
          renderBlockCard
        )}
      </BlockStagePanel>

      <BlockStagePanel number={3} title={t("blocksPanel.stages.sensorExtraction")}>
        <div className="sensor-category">
          <span className="sensor-category-label">Sensores principais</span>
          {renderStageGrid(
            library,
            [
              "turbidimetry_extraction",
              "nephelometry_extraction",
              "fluorescence_extraction",
              "resonant_frequencies_extraction",
            ],
            blockMatchesQuery,
            blocksQuery,
            renderBlockCard
          )}
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Temperaturas</span>
          {renderStageGrid(library, ["temperatures_extraction"], blockMatchesQuery, blocksQuery, renderBlockCard)}
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Fonte de alimentacao</span>
          {renderStageGrid(library, ["power_supply_extraction"], blockMatchesQuery, blocksQuery, renderBlockCard)}
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Pastilha Peltier</span>
          {renderStageGrid(library, ["peltier_currents_extraction"], blockMatchesQuery, blocksQuery, renderBlockCard)}
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Agitador magnetico</span>
          {renderStageGrid(library, ["nema_currents_extraction"], blockMatchesQuery, blocksQuery, renderBlockCard)}
        </div>

        <div className="sensor-category">
          <span className="sensor-category-label">Estados de controle</span>
          {renderStageGrid(library, ["control_state_extraction"], blockMatchesQuery, blocksQuery, renderBlockCard)}
        </div>
      </BlockStagePanel>

      <BlockStagePanel number={4} title={t("blocksPanel.stages.preparation")}>
        {renderStageGrid(
          library,
          [
            "sensor_fusion",
            "time_slice",
            "outlier_removal",
            "moving_average_filter",
            "savgol_filter",
            "median_filter",
            "lowpass_filter",
            "exponential_filter",
          ],
          blockMatchesQuery,
          blocksQuery,
          renderBlockCard
        )}
      </BlockStagePanel>

      <BlockStagePanel number={5} title={t("blocksPanel.stages.transformations")}>
        {renderStageGrid(
          library,
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
          ],
          blockMatchesQuery,
          blocksQuery,
          renderBlockCard
        )}
      </BlockStagePanel>

      <BlockStagePanel number={6} title={t("blocksPanel.stages.detection")}>
        {renderStageGrid(
          library,
          ["amplitude_detector", "derivative_detector", "ratio_detector"],
          blockMatchesQuery,
          blocksQuery,
          renderBlockCard
        )}
      </BlockStagePanel>

      <BlockStagePanel number={7} title={t("blocksPanel.stages.features")}>
        {renderStageGrid(
          library,
          [
            "statistical_features",
            "temporal_features",
            "shape_features",
            "growth_features",
            "feature_extraction",
            "features_merge",
          ],
          blockMatchesQuery,
          blocksQuery,
          renderBlockCard
        )}
      </BlockStagePanel>

      <BlockStagePanel number={8} title={t("blocksPanel.stages.modelling")}>
        {renderStageGrid(
          library,
          [
            "curve_fit",
            "curve_fit_best",
            "ml_inference",
            "ml_inference_series",
            "ml_inference_multichannel",
            "ml_forecaster_series",
            "ml_transform_series",
            "ml_detector",
          ],
          blockMatchesQuery,
          blocksQuery,
          renderBlockCard
        )}
      </BlockStagePanel>

      <BlockStagePanel number={9} title={t("blocksPanel.stages.response")}>
        {renderStageGrid(
          library,
          ["response_pack", "response_merge", "response_builder"],
          blockMatchesQuery,
          blocksQuery,
          renderBlockCard
        )}
      </BlockStagePanel>
    </BlocksSidebar>
  );
}
