export default function WorkspaceHomeModal({
  t,
  workspaceHomeOpen,
  workspace,
  setWorkspaceHomeOpen,
  workspaceHomeMode,
  setWorkspaceHomeMode,
  workspaceListLoading,
  workspaceList,
  workspaceActionLoading,
  workspaceError,
  newTenantName,
  setNewTenantName,
  handleCreateWorkspace,
  handleLoadWorkspace,
  fetchWorkspaces,
  selectedWorkspaceKey,
  setSelectedWorkspaceKey,
  workspaceCardMenuKey,
  setWorkspaceCardMenuKey,
  openVersionsModal,
  setDuplicateModal,
  duplicateLogoFileInputRef,
  setDeleteModal,
  setWorkspaceMetaDraft,
  setEditModal,
  triggerLoadPipeline,
  resolveWorkspaceLogoSrc,
  workspaceInitials,
}) {
  if (!workspaceHomeOpen) return null;

  return (
        <div className="workspace-home">
          <div className="workspace-home-bg" aria-hidden="true" />
          <div className="workspace-home-card" role="dialog" aria-modal="true">
            <div className="workspace-home-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.kicker")}</div>
                <h2 className="workspace-home-title">{t("workspace.title")}</h2>
                <div className="workspace-home-subtitle">{t("workspace.subtitle")}</div>
              </div>
              {workspace?.tenant && (
                <button className="workspace-home-close" type="button" onClick={() => setWorkspaceHomeOpen(false)}>
                  {t("actions.close")}
                </button>
              )}
            </div>

            <div className="workspace-home-grid">
              <div className="workspace-panel">
                <div className="workspace-panel-title">{workspaceHomeMode === "create" ? t("workspace.createTitle") : t("workspace.loadTitle")}</div>
                <small className="workspace-muted">{workspaceHomeMode === "create" ? t("workspace.createSubtitle") : t("workspace.availableSubtitle")}</small>

                {workspaceHomeMode === "create" ? (
                  <>
                    <div className="workspace-field workspace-create-field">
                      <label>{t("workspace.tenantLabel")}</label>
                      <input
                        value={newTenantName}
                        onChange={(e) => setNewTenantName(e.target.value)}
                        placeholder={t("workspace.tenantPlaceholder")}
                        disabled={workspaceActionLoading}
                      />
                    </div>
                    <button
                      className="workspace-primary"
                      type="button"
                      disabled={workspaceActionLoading || !String(newTenantName || "").trim()}
                      onClick={handleCreateWorkspace}
                    >
                      {workspaceActionLoading ? t("workspace.creating") : t("workspace.create")}
                    </button>
                    <div
                      className="workspace-mode-link workspace-back-action"
                      role="button"
                      tabIndex={0}
                      onClick={() => setWorkspaceHomeMode("available")}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") setWorkspaceHomeMode("available");
                      }}
                    >
                      {t("workspace.backToAvailable")}
                    </div>
                  </>
                ) : (
                  <>
                    <small className="workspace-muted">{t("workspace.pickHint")}</small>

                    <div className="workspace-pipeline-grid" role="list">
                      {workspaceListLoading ? (
                        <div className="workspace-empty-state">{t("workspace.loadingList")}</div>
                      ) : !workspaceList.length ? (
                        <div className="workspace-empty-state">{t("workspace.pickHint")}</div>
                      ) : (
                        workspaceList.map((w) => {
                          const key = `${w.tenant}/${w.pipeline}`;
                          const tenantId = String(w.tenant || "").trim();
                          const pipelineId = String(w.pipeline || "").trim();
                          const subtitle =
                            pipelineId && tenantId && pipelineId !== tenantId
                              ? `${t("workspace.tenantLabel")}: ${tenantId} / ${t("workspace.pipelineIdLabel")}: ${pipelineId}`
                              : `${t("workspace.tenantLabel")}: ${tenantId || pipelineId}`;
                          return (
                            <button
                              key={key}
                              type="button"
                              role="listitem"
                              className={`workspace-pipeline-card ${selectedWorkspaceKey === key ? "selected" : ""} ${workspaceCardMenuKey === key ? "menu-open" : ""}`}
                              onClick={() => {
                                setSelectedWorkspaceKey(key);
                                setWorkspaceCardMenuKey("");
                              }}
                              title={`${w.tenant} / ${w.pipeline}`}
                            >
                              <div className="workspace-pipeline-card-header">
                                <div className="workspace-pipeline-identity">
                                  <div className="workspace-pipeline-logo">
                                    {w.logo ? (
                                      <img src={resolveWorkspaceLogoSrc(w.logo, w.tenant, w.pipeline)} alt={w.title || w.pipeline} />
                                    ) : (
                                      <span className="workspace-pipeline-logo-fallback">{workspaceInitials(w)}</span>
                                    )}
                                  </div>
                                  <div className="workspace-pipeline-main">
                                    <div className="workspace-pipeline-title">{w.title || w.pipeline}</div>
                                    <div className="workspace-pipeline-sub">{String(subtitle || "").replace(/[^\u0020-\u007E]+/g, " | ")}</div>
                                  </div>
                                </div>
                                <div className="workspace-pipeline-actions" onMouseDown={(e) => e.stopPropagation()}>
                                  <button
                                    type="button"
                                    className="workspace-pipeline-action-btn"
                                    title={t("workspace.editAction")}
                                    onClick={(e) => {
                                      e.preventDefault();
                                      e.stopPropagation();
                                      setWorkspaceCardMenuKey("");
                                      setSelectedWorkspaceKey(key);
                                      setWorkspaceMetaDraft((prev) => ({
                                        ...prev,
                                        title: w.title || w.pipeline || "",
                                        logo: w.logo || "",
                                        accent_color: w.accent_color || prev.accent_color,
                                      }));
                                      setEditModal({
                                        open: true,
                                        target: { tenant: w.tenant, pipeline: w.pipeline, title: w.title || w.pipeline },
                                      });
                                    }}
                                  >
                                    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                                      <path
                                        fill="currentColor"
                                        d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25Zm2.92 2.33H5v-.92l8.06-8.06.92.92L5.92 19.58ZM20.71 7.04a1.003 1.003 0 0 0 0-1.42L18.37 3.29a1.003 1.003 0 0 0-1.42 0L15.13 5.1l3.75 3.75 1.83-1.81Z"
                                      />
                                    </svg>
                                  </button>
                                  <button
                                    type="button"
                                    className="workspace-pipeline-action-btn"
                                    aria-haspopup="menu"
                                    aria-expanded={workspaceCardMenuKey === key}
                                    onClick={(e) => {
                                      e.preventDefault();
                                      e.stopPropagation();
                                      setSelectedWorkspaceKey(key);
                                      setWorkspaceCardMenuKey((prev) => (prev === key ? "" : key));
                                    }}
                                  >
                                    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                                      <path fill="currentColor" d="M6 10a2 2 0 1 0 0 4a2 2 0 0 0 0-4Zm6 0a2 2 0 1 0 0 4a2 2 0 0 0 0-4Zm6 0a2 2 0 1 0 0 4a2 2 0 0 0 0-4Z" />
                                    </svg>
                                  </button>

                                  {workspaceCardMenuKey === key && (
                                    <div className="workspace-pipeline-menu" role="menu" aria-label={t("workspace.cardMenuLabel")}>
                                      <button
                                        type="button"
                                        className="workspace-pipeline-menu-item"
                                        role="menuitem"
                                        onClick={(e) => {
                                          e.preventDefault();
                                          e.stopPropagation();
                                          setWorkspaceCardMenuKey("");
                                          openVersionsModal({ tenant: w.tenant, pipeline: w.pipeline, title: w.title || w.pipeline });
                                        }}
                                      >
                                        {t("workspace.versionsAction")}
                                      </button>
                                      <button
                                        type="button"
                                        className="workspace-pipeline-menu-item"
                                        role="menuitem"
                                        onClick={(e) => {
                                          e.preventDefault();
                                          e.stopPropagation();
                                          setWorkspaceCardMenuKey("");
                                          setDuplicateModal({
                                            open: true,
                                            source: { tenant: w.tenant, pipeline: w.pipeline, title: w.title || w.pipeline },
                                            tenant: "",
                                            logoFile: null,
                                          });
                                          if (duplicateLogoFileInputRef.current) duplicateLogoFileInputRef.current.value = "";
                                        }}
                                      >
                                        {t("workspace.duplicateAction")}
                                      </button>
                                      <button
                                        type="button"
                                        className="workspace-pipeline-menu-item danger"
                                        role="menuitem"
                                        onClick={(e) => {
                                          e.preventDefault();
                                          e.stopPropagation();
                                          setWorkspaceCardMenuKey("");
                                          setDeleteModal({
                                            open: true,
                                            target: { tenant: w.tenant, pipeline: w.pipeline, title: w.title || w.pipeline },
                                          });
                                        }}
                                      >
                                        {t("workspace.deleteAction")}
                                      </button>
                                    </div>
                                  )}
                                </div>
                              </div>
                              <div className="workspace-pipeline-meta">
                                <div className="workspace-pipeline-tags">
                                  <span className="workspace-pill">{w.active_version || "v1"}</span>
                                  <span className="workspace-pill muted">{typeof w.versions_count === "number" ? `${w.versions_count} ${t("workspace.versionsLabel")}` : t("workspace.versionsLabel")}</span>
                                </div>
                              </div>
                            </button>
                          );
                        })
                      )}
                    </div>

                    <div className="workspace-actions-row">
                      <button
                        className="workspace-secondary"
                        type="button"
                        disabled={workspaceActionLoading || !selectedWorkspaceKey}
                        onClick={handleLoadWorkspace}
                      >
                        {workspaceActionLoading ? t("workspace.loading") : t("workspace.load")}
                      </button>
                      <button
                        className="workspace-tertiary"
                        type="button"
                        disabled={workspaceListLoading || workspaceActionLoading}
                        onClick={fetchWorkspaces}
                      >
                        {t("workspace.refresh")}
                      </button>
                    </div>

                    <div className="workspace-divider" />

                    <button className="workspace-tertiary" type="button" onClick={triggerLoadPipeline} disabled={workspaceActionLoading}>
                      {t("workspace.loadFromFile")}
                    </button>
                    <div
                      className="workspace-mode-link workspace-create-switch"
                      role="button"
                      tabIndex={0}
                      onClick={() => setWorkspaceHomeMode("create")}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") setWorkspaceHomeMode("create");
                      }}
                    >
                      {t("workspace.createClientCta")}
                    </div>
                  </>
                )}
              </div>
            </div>

            {workspaceError && <div className="workspace-error">{workspaceError}</div>}
          </div>
        </div>
  );
}
