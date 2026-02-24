export default function AppHeader({
  t,
  workspace,
  pipelineName,
  setPipelineName,
  onOpenWorkspace,
  onSave,
  onDownload,
  onAutoLayout,
  onOpenResults,
  hasNodes,
  canAutoLayout,
  hasSimulation,
  isDarkTheme,
  onToggleTheme,
}) {
  return (
    <header>
      <div>
        <h1>{t("app.title")}</h1>
        <p>{t("app.subtitle")}</p>
        {workspace?.tenant && (
          <p className="workspace-badge">
            {t("workspace.current", { tenant: workspace.tenant, pipeline: workspace.pipeline })}
            {workspace?.version ? ` • ${t("workspace.currentVersion", { version: workspace.version })}` : ""}
          </p>
        )}
      </div>
      <div className="header-actions">
        <input
          value={pipelineName}
          onChange={(e) => setPipelineName(e.target.value)}
          placeholder={t("app.pipelineNamePlaceholder")}
        />
        <button onClick={onOpenWorkspace} className="btn-load" title={t("workspace.openTitle")} type="button">
          {t("workspace.open")}
        </button>
        <button onClick={onSave} disabled={!hasNodes} className="btn-save" title={t("app.saveTitle")}>
          {t("actions.save")}
        </button>
        <button onClick={onDownload} disabled={!hasNodes} className="btn-load" title={t("workspace.downloadTitle")} type="button">
          {t("workspace.download")}
        </button>
        <button onClick={onAutoLayout} disabled={!canAutoLayout} className="btn-load" title={t("actions.autoLayoutTitle")} type="button">
          {t("actions.autoLayout")}
        </button>
        <button onClick={onOpenResults} disabled={!hasSimulation} className="btn-load" title={t("actions.viewResults")} type="button">
          {t("actions.viewResults")}
        </button>
        <label className="header-theme-switch" title="Alternar tema escuro">
          <input type="checkbox" checked={!!isDarkTheme} onChange={(e) => onToggleTheme(e.target.checked)} />
          <span className="header-theme-slider" aria-hidden="true" />
        </label>
      </div>
    </header>
  );
}
