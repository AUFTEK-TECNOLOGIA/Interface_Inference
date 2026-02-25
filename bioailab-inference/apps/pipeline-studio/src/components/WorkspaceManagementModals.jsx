export default function WorkspaceManagementModals({
  t,
  versionsModal,
  setVersionsModal,
  versionActionsModal,
  setVersionActionsModal,
  versionLogsModal,
  setVersionLogsModal,
  renameVersionModal,
  setRenameVersionModal,
  deleteVersionModal,
  setDeleteVersionModal,
  deleteModal,
  setDeleteModal,
  workspaceActionLoading,
  versionsPageStart,
  versionsPageEnd,
  versionsSorted,
  versionsPageItems,
  versionsCurrentPage,
  versionsTotalPages,
  formatDateTime,
  handleOpenVersionInEditor,
  handleCreateNewVersionClean,
  handleCreateNewVersionCopy,
  handleActivateVersion,
  handleRenameVersion,
  handleDeleteVersion,
  handleDeleteWorkspace,
}) {
  const closeVersionsModal = () => {
    setVersionActionsModal({ open: false, version: null });
    setVersionLogsModal({ open: false, version: null, query: "" });
    setVersionsModal({ open: false, target: null, active: "", versions: [], loading: false, reasonDraft: "", page: 0, query: "" });
  };

  return (
    <>
      {versionsModal?.open && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={closeVersionsModal}>
          <div className="workspace-modal workspace-modal-wide" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.versionsTitle")}</div>
                <div className="workspace-modal-title">{versionsModal?.target?.title || ""}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={closeVersionsModal}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-muted">{t("workspace.versionsHint")}</div>

            <div className="workspace-versions-toolbar">
              <div className="workspace-field">
                <label>{t("workspace.changeReasonLabel")}</label>
                <input
                  value={versionsModal?.reasonDraft || ""}
                  onChange={(e) => setVersionsModal((prev) => ({ ...prev, reasonDraft: e.target.value }))}
                  placeholder={t("workspace.changeReasonPlaceholder")}
                  disabled={workspaceActionLoading}
                />
              </div>
              <div className="workspace-field">
                <label>{t("workspace.searchVersionsLabel")}</label>
                <input
                  value={versionsModal?.query || ""}
                  onChange={(e) => setVersionsModal((prev) => ({ ...prev, query: e.target.value, page: 0 }))}
                  placeholder={t("workspace.searchVersionsPlaceholder")}
                  disabled={workspaceActionLoading}
                />
              </div>
            </div>

            {!versionsModal.loading && (
              <div className="workspace-version-summary">
                {t("workspace.versionsShowing", { start: versionsPageStart, end: versionsPageEnd, total: versionsSorted.length })}
                {versionsModal?.active ? ` • ${t("workspace.activeVersionShort", { id: versionsModal.active })}` : ""}
              </div>
            )}

            {versionsModal.loading ? (
              <div className="workspace-muted">{t("workspace.loadingVersions")}</div>
            ) : (
              <div className="workspace-version-list" role="list">
                {versionsPageItems.map((v) => {
                  const history = Array.isArray(v.history) ? v.history : [];
                  const lastChange = history.length ? history[history.length - 1] : null;

                  return (
                    <div key={v.id} className={`workspace-version-row ${v.is_active ? "active" : ""}`} role="listitem">
                      <div className="workspace-version-main">
                        <div className="workspace-version-top">
                          <div className="workspace-version-name">{String(v.name || v.id)}</div>
                          <div className="workspace-version-id">{v.id}</div>
                        </div>
                        <div className="workspace-version-meta">
                          {v.is_active ? t("workspace.activeVersionLabel") : t("workspace.inactiveVersionLabel")}
                          {v.based_on ? ` • ${t("workspace.basedOnLabel")} ${v.based_on}` : ""}
                          {v.created_at ? ` • ${t("workspace.versionCreatedAt", { date: formatDateTime(v.created_at) })}` : ""}
                          {v.updated_at ? ` • ${t("workspace.versionUpdatedAt", { date: formatDateTime(v.updated_at) })}` : ""}
                        </div>
                        {lastChange?.at && (
                          <div className="workspace-version-lastchange">
                            {t("workspace.versionLastChangeLabel", { date: formatDateTime(lastChange.at) })}
                            {lastChange?.reason ? ` — ${String(lastChange.reason).trim()}` : ""}
                          </div>
                        )}
                      </div>

                      <div className="workspace-version-actions">
                        <button
                          className="workspace-secondary"
                          type="button"
                          disabled={workspaceActionLoading || versionsModal.loading}
                          onClick={() => handleOpenVersionInEditor(v.id)}
                        >
                          {t("workspace.openVersionAction")}
                        </button>
                        <button
                          className="workspace-tertiary"
                          type="button"
                          disabled={workspaceActionLoading}
                          onClick={() => setVersionActionsModal({ open: true, version: v })}
                        >
                          {t("actions.more")}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {!versionsModal.loading && versionsSorted.length > 0 && (
              <div className="workspace-version-pagination" aria-label={t("workspace.paginationLabel")}>
                <button
                  className="workspace-tertiary"
                  type="button"
                  disabled={workspaceActionLoading || versionsCurrentPage <= 0}
                  onClick={() => setVersionsModal((prev) => ({ ...prev, page: Math.max(0, (prev.page || 0) - 1) }))}
                >
                  {t("workspace.paginationPrev")}
                </button>
                <div className="workspace-version-pagination-meta">
                  {t("workspace.paginationMeta", { page: versionsCurrentPage + 1, total: versionsTotalPages })}
                </div>
                <button
                  className="workspace-tertiary"
                  type="button"
                  disabled={workspaceActionLoading || versionsCurrentPage >= versionsTotalPages - 1}
                  onClick={() =>
                    setVersionsModal((prev) => ({
                      ...prev,
                      page: Math.min(versionsTotalPages - 1, (prev.page || 0) + 1),
                    }))
                  }
                >
                  {t("workspace.paginationNext")}
                </button>
              </div>
            )}

            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={closeVersionsModal}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary" type="button" disabled={workspaceActionLoading || versionsModal.loading} onClick={handleCreateNewVersionClean}>
                {workspaceActionLoading ? t("workspace.creatingVersion") : t("workspace.createVersionCleanAction")}
              </button>
            </div>
          </div>
        </div>
      )}

      {versionActionsModal?.open && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={() => setVersionActionsModal({ open: false, version: null })}>
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.versionActionsTitle")}</div>
                <div className="workspace-modal-title">{String(versionActionsModal?.version?.name || versionActionsModal?.version?.id || "")}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setVersionActionsModal({ open: false, version: null })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-version-actions-groups">
              <div className="workspace-version-actions-group">
                <div className="workspace-version-actions-group-title">{t("workspace.versionActionsViewTitle")}</div>
                <div className="workspace-version-actions-group-buttons">
                  <button
                    className="workspace-secondary"
                    type="button"
                    disabled={workspaceActionLoading || versionsModal.loading}
                    onClick={() => handleOpenVersionInEditor(versionActionsModal.version.id)}
                  >
                    {t("workspace.openVersionAction")}
                  </button>
                  <button
                    className="workspace-tertiary"
                    type="button"
                    disabled={workspaceActionLoading}
                    onClick={() => setVersionLogsModal({ open: true, version: versionActionsModal.version, query: "" })}
                  >
                    {t("workspace.openVersionLogsAction")}
                  </button>
                </div>
              </div>

              <div className="workspace-version-actions-group">
                <div className="workspace-version-actions-group-title">{t("workspace.versionActionsEditTitle")}</div>
                <div className="workspace-version-actions-group-buttons">
                  <button
                    className="workspace-tertiary"
                    type="button"
                    disabled={workspaceActionLoading || versionsModal.loading}
                    onClick={() => handleCreateNewVersionCopy(versionActionsModal.version.id, { activate: false })}
                  >
                    {t("workspace.copyVersionAction")}
                  </button>
                  <button
                    className="workspace-tertiary"
                    type="button"
                    disabled={workspaceActionLoading}
                    onClick={() =>
                      setRenameVersionModal({
                        open: true,
                        version: versionActionsModal.version.id,
                        name: String(versionActionsModal.version.name || ""),
                        reason: String(versionsModal?.reasonDraft || "").trim(),
                      })
                    }
                  >
                    {t("workspace.renameVersionAction")}
                  </button>
                </div>
              </div>

              <div className="workspace-version-actions-group">
                <div className="workspace-version-actions-group-title">{t("workspace.versionActionsActivationTitle")}</div>
                <div className="workspace-version-actions-group-buttons">
                  <button
                    className="workspace-tertiary"
                    type="button"
                    disabled={workspaceActionLoading || Boolean(versionActionsModal?.version?.is_active)}
                    onClick={() => handleActivateVersion(versionActionsModal.version.id)}
                  >
                    {t("workspace.activateVersionAction")}
                  </button>
                </div>
              </div>

              <div className="workspace-version-actions-group danger">
                <div className="workspace-version-actions-group-title">{t("workspace.versionActionsDangerTitle")}</div>
                <div className="workspace-version-actions-group-buttons">
                  <button
                    className="workspace-tertiary danger"
                    type="button"
                    disabled={workspaceActionLoading || Boolean(versionActionsModal?.version?.is_active && (versionsSorted || []).length <= 1)}
                    onClick={() => setDeleteVersionModal({ open: true, version: versionActionsModal.version.id })}
                  >
                    {t("workspace.deleteVersionAction")}
                  </button>
                </div>
              </div>
            </div>

            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => setVersionActionsModal({ open: false, version: null })}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary" type="button" disabled={workspaceActionLoading} onClick={() => setVersionActionsModal({ open: false, version: null })}>
                {t("actions.close")}
              </button>
            </div>
          </div>
        </div>
      )}

      {versionLogsModal?.open && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={() => setVersionLogsModal({ open: false, version: null, query: "" })}>
          <div className="workspace-modal workspace-modal-wide" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.versionLogsTitle")}</div>
                <div className="workspace-modal-title">{String(versionLogsModal?.version?.name || versionLogsModal?.version?.id || "")}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setVersionLogsModal({ open: false, version: null, query: "" })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-versions-toolbar">
              <div className="workspace-field">
                <label>{t("workspace.searchLogsLabel")}</label>
                <input
                  value={versionLogsModal?.query || ""}
                  onChange={(e) => setVersionLogsModal((prev) => ({ ...prev, query: e.target.value }))}
                  placeholder={t("workspace.searchLogsPlaceholder")}
                />
              </div>
            </div>

            <div className="workspace-version-history workspace-version-history-full" aria-label={t("workspace.versionHistoryTitle")}>
              <div className="workspace-version-history-list">
                {(() => {
                  const history = Array.isArray(versionLogsModal?.version?.history) ? versionLogsModal.version.history : [];
                  const q = String(versionLogsModal?.query || "").trim().toLowerCase();
                  const filtered = q
                    ? history.filter((h) => String(h?.reason || "").toLowerCase().includes(q) || String(h?.action || "").toLowerCase().includes(q))
                    : history;
                  const items = [...filtered].reverse();
                  if (!items.length) return <div className="workspace-muted">{t("workspace.noLogs")}</div>;

                  return items.map((h, idx) => (
                    <div key={`vh-${idx}`} className="workspace-version-history-item">
                      <span className="workspace-version-history-when">{formatDateTime(h.at)}</span>
                      <span className="workspace-version-history-reason">{String(h.reason || "").trim() || "-"}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>

            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" onClick={() => setVersionLogsModal({ open: false, version: null, query: "" })}>
                {t("actions.close")}
              </button>
            </div>
          </div>
        </div>
      )}

      {renameVersionModal?.open && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={() => setRenameVersionModal({ open: false, version: "", name: "", reason: "" })}>
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.renameVersionTitle")}</div>
                <div className="workspace-modal-title">{renameVersionModal.version}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setRenameVersionModal({ open: false, version: "", name: "", reason: "" })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-field">
              <label>{t("workspace.versionNameLabel")}</label>
              <input
                value={renameVersionModal.name}
                onChange={(e) => setRenameVersionModal((prev) => ({ ...prev, name: e.target.value }))}
                placeholder={t("workspace.versionNamePlaceholder")}
                disabled={workspaceActionLoading}
                autoFocus
              />
            </div>

            <div className="workspace-field">
              <label>{t("workspace.changeReasonLabel")}</label>
              <input
                value={renameVersionModal.reason}
                onChange={(e) => setRenameVersionModal((prev) => ({ ...prev, reason: e.target.value }))}
                placeholder={t("workspace.changeReasonPlaceholder")}
                disabled={workspaceActionLoading}
              />
            </div>

            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => setRenameVersionModal({ open: false, version: "", name: "", reason: "" })}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary" type="button" disabled={workspaceActionLoading || !String(renameVersionModal?.name || "").trim()} onClick={handleRenameVersion}>
                {workspaceActionLoading ? t("workspace.savingVersionName") : t("workspace.saveVersionNameAction")}
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteVersionModal?.open && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={() => setDeleteVersionModal({ open: false, version: "" })}>
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.deleteVersionTitle")}</div>
                <div className="workspace-modal-title">{deleteVersionModal.version}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setDeleteVersionModal({ open: false, version: "" })}>
                {t("actions.close")}
              </button>
            </div>
            <div className="workspace-muted">{t("workspace.deleteVersionConfirm")}</div>
            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => setDeleteVersionModal({ open: false, version: "" })}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary danger" type="button" disabled={workspaceActionLoading} onClick={handleDeleteVersion}>
                {workspaceActionLoading ? t("workspace.deletingVersion") : t("workspace.deleteVersionAction")}
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteModal?.open && (
        <div className="workspace-modal-overlay" role="dialog" aria-modal="true" onMouseDown={() => setDeleteModal({ open: false, target: null })}>
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.deleteTitle")}</div>
                <div className="workspace-modal-title">{deleteModal?.target?.title || ""}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setDeleteModal({ open: false, target: null })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-muted">{t("workspace.deleteConfirm")}</div>

            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => setDeleteModal({ open: false, target: null })}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary danger" type="button" disabled={workspaceActionLoading} onClick={handleDeleteWorkspace}>
                {workspaceActionLoading ? t("workspace.deleting") : t("workspace.deleteAction")}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
