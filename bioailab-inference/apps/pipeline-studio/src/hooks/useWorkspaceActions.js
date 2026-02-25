import { useCallback } from "react";
import axios from "axios";

export default function useWorkspaceActions({
  apiUrl,
  sanitizeColor,
  selectedWorkspace,
  workspaceMetaDraft,
  setWorkspaceActionLoading,
  setWorkspaceError,
  fetchWorkspaces,
  setEditModal,
  setWorkspaceMetaDraft,
  newTenantName,
  loadPipelineFromJson,
  setNewTenantName,
  setSelectedWorkspaceKey,
  setWorkspaceHomeMode,
  selectedWorkspaceKey,
  duplicateModal,
  setDuplicateModal,
  duplicateLogoFileInputRef,
  deleteModal,
  workspace,
  setWorkspace,
  setDeleteModal,
  setVersionsModal,
  versionsModal,
  deleteVersionModal,
  setDeleteVersionModal,
  renameVersionModal,
  setRenameVersionModal,
  setVersionActionsModal,
  setVersionLogsModal,
}) {
const handleSaveWorkspaceAppearance = useCallback(async () => {
  if (!selectedWorkspace?.tenant || !selectedWorkspace?.pipeline) return;
  const tenant = selectedWorkspace.tenant;
  const pipeline = selectedWorkspace.pipeline;
  setWorkspaceActionLoading(true);
  setWorkspaceError("");
  try {
    const res = await axios.get(`${apiUrl}/pipelines/workspaces/${tenant}/${pipeline}`);
    const data = res.data || {};
    data.workspace = {
      ...(typeof data.workspace === "object" && data.workspace ? data.workspace : {}),
      title: String(workspaceMetaDraft.title || "").trim() || pipeline,
      logo: String(workspaceMetaDraft.logo || "").trim(),
      accent_color: sanitizeColor(workspaceMetaDraft.accent_color),
    };
    await axios.post(`${apiUrl}/pipelines/workspaces/save`, { tenant, pipeline, data, change_reason: "Atualização de personalização" });
    await fetchWorkspaces();
    return true;
  } catch (err) {
    setWorkspaceError(err.response?.data?.detail || err.message);
    return false;
  } finally {
    setWorkspaceActionLoading(false);
  }
}, [selectedWorkspace, workspaceMetaDraft, fetchWorkspaces]);

const handleSaveWorkspaceAppearanceFromModal = useCallback(async () => {
  const ok = await handleSaveWorkspaceAppearance();
  if (ok) setEditModal({ open: false, target: null });
}, [handleSaveWorkspaceAppearance]);

const handleUploadWorkspaceLogo = useCallback(
  async (file) => {
    if (!selectedWorkspace?.tenant || !selectedWorkspace?.pipeline || !file) return;
    const tenant = selectedWorkspace.tenant;
    const pipeline = selectedWorkspace.pipeline;
    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post(`${apiUrl}/pipelines/workspaces/${tenant}/${pipeline}/logo-upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const nextLogo = res.data?.asset_path || res.data?.logo || "";
      const nextAccent = sanitizeColor(res.data?.accent_color);
      setWorkspaceMetaDraft((prev) => ({ ...prev, logo: nextLogo, accent_color: nextAccent }));

      // Persistir automaticamente no JSON do pipeline
      const pipelineJson = await axios.get(`${apiUrl}/pipelines/workspaces/${tenant}/${pipeline}`);
      const data = pipelineJson.data || {};
      data.workspace = {
        ...(typeof data.workspace === "object" && data.workspace ? data.workspace : {}),
        title: String(workspaceMetaDraft.title || "").trim() || pipeline,
        logo: nextLogo,
        accent_color: nextAccent,
      };
      await axios.post(`${apiUrl}/pipelines/workspaces/save`, { tenant, pipeline, data });
      await fetchWorkspaces();
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  },
  [selectedWorkspace, fetchWorkspaces, sanitizeColor, workspaceMetaDraft.title]
);

const handleCreateWorkspace = useCallback(async () => {
  const tenant = String(newTenantName || "").trim();
  if (!tenant) return;
  setWorkspaceActionLoading(true);
  setWorkspaceError("");
  try {
    const res = await axios.post(`${apiUrl}/pipelines/workspaces/create`, { tenant });
    loadPipelineFromJson(res.data, { tenant, pipeline: tenant });
    setNewTenantName("");
    setSelectedWorkspaceKey("");
    setWorkspaceHomeMode("available");
  } catch (err) {
    setWorkspaceError(err.response?.data?.detail || err.message);
  } finally {
    setWorkspaceActionLoading(false);
  }
}, [newTenantName, loadPipelineFromJson]);

const handleLoadWorkspace = useCallback(async () => {
  const key = String(selectedWorkspaceKey || "");
  if (!key.includes("/")) return;
  const [tenant, pipeline] = key.split("/");
  setWorkspaceActionLoading(true);
  setWorkspaceError("");
  try {
    const res = await axios.get(`${apiUrl}/pipelines/workspaces/${tenant}/${pipeline}`);
    loadPipelineFromJson(res.data, { tenant, pipeline });
  } catch (err) {
    setWorkspaceError(err.response?.data?.detail || err.message);
  } finally {
    setWorkspaceActionLoading(false);
  }
}, [selectedWorkspaceKey, loadPipelineFromJson]);

const handleDuplicateWorkspace = useCallback(async () => {
  const source = duplicateModal?.source;
  if (!source?.tenant || !source?.pipeline) return;
  const targetTenant = String(duplicateModal?.tenant || "").trim();
  if (!targetTenant) return;

  const sourceTenant = source.tenant;
  const sourcePipeline = source.pipeline;
  const targetPipeline = targetTenant;

  setWorkspaceActionLoading(true);
  setWorkspaceError("");
  try {
    const res = await axios.post(`${apiUrl}/pipelines/workspaces/duplicate`, {
      source_tenant: sourceTenant,
      source_pipeline: sourcePipeline,
      target_tenant: targetTenant,
      target_pipeline: targetPipeline,
      target_title: targetTenant,
    });

    let payload = res.data || {};

    if (duplicateModal?.logoFile) {
      const form = new FormData();
      form.append("file", duplicateModal.logoFile);
      const up = await axios.post(`${apiUrl}/pipelines/workspaces/${targetTenant}/${targetPipeline}/logo-upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const nextLogo = up.data?.asset_path || up.data?.logo || "";
      const nextAccent = sanitizeColor(up.data?.accent_color);
      payload.workspace = {
        ...(typeof payload.workspace === "object" && payload.workspace ? payload.workspace : {}),
        title: targetTenant,
        logo: nextLogo,
        accent_color: nextAccent,
      };
      await axios.post(`${apiUrl}/pipelines/workspaces/save`, { tenant: targetTenant, pipeline: targetPipeline, data: payload });
    }

    await fetchWorkspaces();
    setDuplicateModal({ open: false, source: null, tenant: "", logoFile: null });
    if (duplicateLogoFileInputRef.current) duplicateLogoFileInputRef.current.value = "";

    loadPipelineFromJson(payload, { tenant: targetTenant, pipeline: targetPipeline });
  } catch (err) {
    setWorkspaceError(err.response?.data?.detail || err.message);
  } finally {
    setWorkspaceActionLoading(false);
  }
}, [duplicateModal, fetchWorkspaces, loadPipelineFromJson, sanitizeColor]);

const handleDeleteWorkspace = useCallback(async () => {
  const target = deleteModal?.target;
  if (!target?.tenant || !target?.pipeline) return;

  setWorkspaceActionLoading(true);
  setWorkspaceError("");
  try {
    await axios.delete(`${apiUrl}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}`);
    await fetchWorkspaces();

    const deletedKey = `${target.tenant}/${target.pipeline}`;
    if (selectedWorkspaceKey === deletedKey) setSelectedWorkspaceKey("");
    if (workspace?.tenant === target.tenant && workspace?.pipeline === target.pipeline) setWorkspace(null);

    setDeleteModal({ open: false, target: null });
  } catch (err) {
    setWorkspaceError(err.response?.data?.detail || err.message);
  } finally {
    setWorkspaceActionLoading(false);
  }
}, [deleteModal, fetchWorkspaces, selectedWorkspaceKey, workspace]);

const openVersionsModal = useCallback(
  async (target) => {
    if (!target?.tenant || !target?.pipeline) return;
    setVersionActionsModal({ open: false, version: null });
    setVersionLogsModal({ open: false, version: null, query: "" });
    setVersionsModal({ open: true, target, active: "", versions: [], loading: true, reasonDraft: "", page: 0, query: "" });
    setWorkspaceError("");
    try {
      const res = await axios.get(
        `${apiUrl}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions`
      );
      setVersionsModal((prev) => ({
        ...prev,
        active: res.data?.active || "",
        versions: Array.isArray(res.data?.versions) ? res.data.versions : [],
        loading: false,
        page: 0,
        query: "",
      }));
    } catch (err) {
      setVersionsModal((prev) => ({ ...prev, loading: false }));
      setWorkspaceError(err.response?.data?.detail || err.message);
    }
  },
  [setVersionsModal]
);

const handleCreateNewVersionCopy = useCallback(async (fromVersion, options = {}) => {
  const target = versionsModal?.target;
  if (!target?.tenant || !target?.pipeline) return;
  setWorkspaceActionLoading(true);
  setWorkspaceError("");
  try {
    const activate = Boolean(options?.activate);
    await axios.post(
      `${apiUrl}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/create-copy`,
      {
        reason: String(versionsModal?.reasonDraft || "").trim(),
        from_version: fromVersion ? String(fromVersion).trim() : "",
        activate,
      }
    );
    await fetchWorkspaces();
    await openVersionsModal(target);
  } catch (err) {
    setWorkspaceError(err.response?.data?.detail || err.message);
  } finally {
    setWorkspaceActionLoading(false);
  }
}, [versionsModal, fetchWorkspaces, openVersionsModal]);

const handleCreateNewVersionClean = useCallback(async () => {
  const target = versionsModal?.target;
  if (!target?.tenant || !target?.pipeline) return;
  setWorkspaceActionLoading(true);
  setWorkspaceError("");
  try {
    await axios.post(
      `${apiUrl}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/create-clean`,
      { reason: String(versionsModal?.reasonDraft || "").trim() }
    );
    await fetchWorkspaces();
    await openVersionsModal(target);
  } catch (err) {
    setWorkspaceError(err.response?.data?.detail || err.message);
  } finally {
    setWorkspaceActionLoading(false);
  }
}, [versionsModal, fetchWorkspaces, openVersionsModal]);

const handleActivateVersion = useCallback(
  async (versionId) => {
    const target = versionsModal?.target;
    if (!target?.tenant || !target?.pipeline || !versionId) return;
    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      await axios.post(
        `${apiUrl}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/${encodeURIComponent(versionId)}/activate`,
        { reason: String(versionsModal?.reasonDraft || "").trim() }
      );
      await fetchWorkspaces();
      await openVersionsModal(target);
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  },
  [versionsModal, fetchWorkspaces]
);

const handleDeleteVersion = useCallback(async () => {
  const target = versionsModal?.target;
  const version = String(deleteVersionModal?.version || "").trim();
  if (!target?.tenant || !target?.pipeline || !version) return;

  setWorkspaceActionLoading(true);
  setWorkspaceError("");
  try {
    await axios.delete(
      `${apiUrl}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/${encodeURIComponent(version)}`
    );
    await fetchWorkspaces();
    setDeleteVersionModal({ open: false, version: "" });
    await openVersionsModal(target);
  } catch (err) {
    setWorkspaceError(err.response?.data?.detail || err.message);
  } finally {
    setWorkspaceActionLoading(false);
  }
}, [versionsModal, deleteVersionModal, fetchWorkspaces, openVersionsModal]);

const handleRenameVersion = useCallback(async () => {
  const target = versionsModal?.target;
  const version = String(renameVersionModal?.version || "").trim();
  const name = String(renameVersionModal?.name || "").trim();
  if (!target?.tenant || !target?.pipeline || !version || !name) return;

  setWorkspaceActionLoading(true);
  setWorkspaceError("");
  try {
    await axios.post(
      `${apiUrl}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/${encodeURIComponent(version)}/rename`,
      { name, reason: String(renameVersionModal?.reason || "").trim() }
    );
    setRenameVersionModal({ open: false, version: "", name: "", reason: "" });
    await openVersionsModal(target);
  } catch (err) {
    setWorkspaceError(err.response?.data?.detail || err.message);
  } finally {
    setWorkspaceActionLoading(false);
  }
}, [versionsModal, renameVersionModal, openVersionsModal]);

const handleOpenVersionInEditor = useCallback(
  async (versionId) => {
    const target = versionsModal?.target;
    if (!target?.tenant || !target?.pipeline || !versionId) return;

    setWorkspaceActionLoading(true);
    setWorkspaceError("");
    try {
      const res = await axios.get(
        `${apiUrl}/pipelines/workspaces/${encodeURIComponent(target.tenant)}/${encodeURIComponent(target.pipeline)}/versions/${encodeURIComponent(versionId)}/load`
      );
      loadPipelineFromJson(res.data, { tenant: target.tenant, pipeline: target.pipeline, version: versionId });
      setVersionActionsModal({ open: false, version: null });
      setVersionLogsModal({ open: false, version: null, query: "" });
      setVersionsModal({ open: false, target: null, active: "", versions: [], loading: false, reasonDraft: "", page: 0, query: "" });
    } catch (err) {
      setWorkspaceError(err.response?.data?.detail || err.message);
    } finally {
      setWorkspaceActionLoading(false);
    }
  },
  [versionsModal, loadPipelineFromJson]
);

  return {
    handleSaveWorkspaceAppearance,
    handleSaveWorkspaceAppearanceFromModal,
    handleUploadWorkspaceLogo,
    handleCreateWorkspace,
    handleLoadWorkspace,
    handleDuplicateWorkspace,
    handleDeleteWorkspace,
    openVersionsModal,
    handleCreateNewVersionCopy,
    handleCreateNewVersionClean,
    handleActivateVersion,
    handleDeleteVersion,
    handleRenameVersion,
    handleOpenVersionInEditor,
  };
}
