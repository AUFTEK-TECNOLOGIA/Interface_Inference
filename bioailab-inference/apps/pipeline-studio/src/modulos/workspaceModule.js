export const sanitizeColor = (value) => {
  const v = String(value || "").trim();
  if (/^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(v)) return v;
  return "#1e90ff";
};

export const resolveWorkspaceLogoSrc = ({ logo, tenant, pipeline, apiUrl }) => {
  const value = String(logo || "").trim();
  if (!value) return "";
  if (value.startsWith("data:")) return value;
  if (value.startsWith("http://") || value.startsWith("https://")) return value;
  if (value.startsWith("/")) return `${apiUrl}${value}`;

  const safeTenant = String(tenant || "").trim();
  const safePipeline = String(pipeline || "").trim();
  if (safeTenant && safePipeline) {
    return `${apiUrl}/pipelines/workspaces/assets/${encodeURIComponent(safeTenant)}/${encodeURIComponent(safePipeline)}/${value.replace(/^\/+/, "")}`;
  }
  return value;
};

export const workspaceInitials = (workspaceInfo) => {
  const base = (workspaceInfo?.title || workspaceInfo?.pipeline || workspaceInfo?.tenant || "W").trim();
  const parts = base.split(/\s+/).filter(Boolean);
  const letters = parts.length >= 2 ? `${parts[0][0] || ""}${parts[1][0] || ""}` : `${base[0] || ""}${base[1] || ""}`;
  return letters.toUpperCase();
};
