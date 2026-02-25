export default function WorkspaceQuickModals({
  t,
  duplicateModal,
  setDuplicateModal,
  workspaceActionLoading,
  duplicateLogoFileInputRef,
  handleDuplicateWorkspace,
  editModal,
  setEditModal,
  workspaceMetaDraft,
  setWorkspaceMetaDraft,
  workspaceLogoFileInputRef,
  handleSaveWorkspaceAppearanceFromModal,
}) {
  return (
    <>
      {duplicateModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setDuplicateModal({ open: false, source: null, tenant: "", logoFile: null })}
        >
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.duplicateTitle")}</div>
                <div className="workspace-modal-title">{duplicateModal?.source?.title || ""}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setDuplicateModal({ open: false, source: null, tenant: "", logoFile: null })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-field">
              <label>{t("workspace.duplicateTenantLabel")}</label>
              <input
                value={duplicateModal.tenant}
                onChange={(e) => setDuplicateModal((prev) => ({ ...prev, tenant: e.target.value }))}
                placeholder={t("workspace.duplicateTenantPlaceholder")}
                disabled={workspaceActionLoading}
                autoFocus
              />
            </div>

            <div className="workspace-field">
              <label>{t("workspace.logoLocalLabel")}</label>
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => duplicateLogoFileInputRef.current?.click()}>
                {t("workspace.chooseLogo")}
              </button>
              {duplicateModal?.logoFile?.name && <small className="workspace-muted">{duplicateModal.logoFile.name}</small>}
            </div>

            <div className="workspace-modal-actions three">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => setDuplicateModal({ open: false, source: null, tenant: "", logoFile: null })}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary" type="button" disabled={workspaceActionLoading || !String(duplicateModal?.tenant || "").trim()} onClick={handleDuplicateWorkspace}>
                {workspaceActionLoading ? t("workspace.duplicating") : t("workspace.duplicateAction")}
              </button>
            </div>
          </div>
        </div>
      )}

      {editModal?.open && (
        <div
          className="workspace-modal-overlay"
          role="dialog"
          aria-modal="true"
          onMouseDown={() => setEditModal({ open: false, target: null })}
        >
          <div className="workspace-modal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="workspace-modal-header">
              <div>
                <div className="workspace-home-kicker">{t("workspace.appearanceTitle")}</div>
                <div className="workspace-modal-title">{editModal?.target?.title || ""}</div>
              </div>
              <button className="workspace-home-close" type="button" onClick={() => setEditModal({ open: false, target: null })}>
                {t("actions.close")}
              </button>
            </div>

            <div className="workspace-field">
              <label>{t("workspace.titleLabel")}</label>
              <input
                value={workspaceMetaDraft.title}
                onChange={(e) => setWorkspaceMetaDraft((prev) => ({ ...prev, title: e.target.value }))}
                disabled={workspaceActionLoading}
                autoFocus
              />
            </div>

            <div className="workspace-field">
              <label>{t("workspace.logoLocalLabel")}</label>
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => workspaceLogoFileInputRef.current?.click()}>
                {workspaceActionLoading ? t("workspace.uploadingLogo") : t("workspace.chooseLogo")}
              </button>
              {workspaceMetaDraft.logo && <small className="workspace-muted">{t("workspace.logoSelectedHint")}</small>}
            </div>

            <div className="workspace-modal-actions">
              <button className="workspace-tertiary" type="button" disabled={workspaceActionLoading} onClick={() => setEditModal({ open: false, target: null })}>
                {t("actions.cancel")}
              </button>
              <button className="workspace-secondary" type="button" disabled={workspaceActionLoading} onClick={handleSaveWorkspaceAppearanceFromModal}>
                {workspaceActionLoading ? t("workspace.savingAppearance") : t("workspace.saveAppearance")}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
