export default function BlocksSidebar({
  t,
  width,
  blocksQuery,
  setBlocksQuery,
  favoriteBlocks,
  recentBlocks,
  library,
  blockMatchesQuery,
  renderBlockCardMini,
  children,
  onStartResize,
}) {
  return (
    <aside className="panel-blocks" style={{ width }}>
      <div className="panel-header">
        <h2>{t("panels.blocks")}</h2>
      </div>
      <div className="panel-content">
        <div className="blocks-panel-top">
          <input
            className="blocks-panel-search"
            value={blocksQuery}
            onChange={(e) => setBlocksQuery(e.target.value)}
            placeholder={t("blocksPanel.searchPlaceholder")}
          />
          <p className="panel-hint">{t("blocksPanel.hint")}</p>

          {favoriteBlocks.length > 0 && (
            <details className="stage-accordion" open>
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">F</span>
                  <span className="stage-title">{t("blocksPanel.favorites")}</span>
                </div>
              </summary>
              <div className="blocks-panel-mini-row">
                {favoriteBlocks
                  .map((name) => (library.blocks || []).find((b) => b.name === name))
                  .filter(Boolean)
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCardMini)}
              </div>
            </details>
          )}

          {recentBlocks.length > 0 && (
            <details className="stage-accordion">
              <summary className="stage-summary">
                <div className="stage-header">
                  <span className="stage-number">R</span>
                  <span className="stage-title">{t("blocksPanel.recents")}</span>
                </div>
              </summary>
              <div className="blocks-panel-mini-row">
                {recentBlocks
                  .map((name) => (library.blocks || []).find((b) => b.name === name))
                  .filter(Boolean)
                  .filter((b) => blockMatchesQuery(b, blocksQuery))
                  .map(renderBlockCardMini)}
              </div>
            </details>
          )}
        </div>

        {children}
      </div>
      <div className="resize-handle resize-handle-right" onMouseDown={(e) => onStartResize(e, 1)} />
    </aside>
  );
}
