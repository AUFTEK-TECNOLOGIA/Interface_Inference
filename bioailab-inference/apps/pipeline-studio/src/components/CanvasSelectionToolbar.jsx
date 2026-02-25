export default function CanvasSelectionToolbar({
  t,
  selectedNodes,
  clipboard,
  copySelectedNodes,
  pasteNodes,
  duplicateSelectedNodes,
  autoLayoutNodes,
  alignNodesLeft,
  alignNodesCenterH,
  alignNodesRight,
  alignNodesTop,
  alignNodesCenterV,
  alignNodesBottom,
  distributeNodesH,
  distributeNodesV,
}) {
  if (!selectedNodes.length) return null;

  const count = selectedNodes.length || 1;

  return (
    <div className="alignment-toolbar">
      <div className="alignment-toolbar-label">{count > 1 ? t("canvas.selectionPlural", { count }) : t("canvas.selection", { count })}</div>
      <div className="alignment-toolbar-buttons">
        <div className="alignment-group">
          <span className="alignment-group-label">Editar</span>
          <button onClick={copySelectedNodes} title="Copiar (Ctrl+C)"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg></button>
          <button onClick={pasteNodes} title="Colar (Ctrl+V)" disabled={clipboard.nodes.length === 0}><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M19 2h-4.18C14.4.84 13.3 0 12 0c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm7 18H5V4h2v3h10V4h2v16z"/></svg></button>
          <button onClick={duplicateSelectedNodes} title="Duplicar (Ctrl+D)"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M11 17H4a2 2 0 0 1-2-2V3a2 2 0 0 1 2-2h12v2H4v12h7v2m11-2V7a2 2 0 0 0-2-2H8v2h12v10a2 2 0 0 0 2 2h-1v-2M9 7v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H11a2 2 0 0 0-2 2m2 0h10v10H11V7z"/></svg></button>
        </div>

        {selectedNodes.length >= 2 && (
          <>
            <div className="alignment-separator" />
            <div className="alignment-group">
              <span className="alignment-group-label">{t("actions.autoLayout")}</span>
              <button onClick={autoLayoutNodes} title={t("actions.autoLayoutTitle")}><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M4 4h6v4H4V4zm10 0h6v4h-6V4zM4 10h6v4H4v-4zm10 0h6v4h-6v-4zM4 16h6v4H4v-4zm10 0h6v4h-6v-4z" /></svg></button>
            </div>
            <div className="alignment-group">
              <span className="alignment-group-label">Alinhar</span>
              <button onClick={alignNodesLeft} title="Alinhar à esquerda"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M4 22H2V2h2v20zM22 7H6v3h16V7zm-6 7H6v3h10v-3z"/></svg></button>
              <button onClick={alignNodesCenterH} title="Centralizar horizontalmente"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M11 2h2v5h8v3H13v4h6v3h-6v5h-2v-5H5v-3h6V10H3V7h8V2z"/></svg></button>
              <button onClick={alignNodesRight} title="Alinhar à direita"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M20 2h2v20h-2V2zM2 7h16v3H2V7zm6 7h10v3H8v-3z"/></svg></button>
              <div className="alignment-separator" />
              <button onClick={alignNodesTop} title="Alinhar ao topo"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M22 2v2H2V2h20zM7 22V6h3v16H7zm7-6V6h3v10h-3z"/></svg></button>
              <button onClick={alignNodesCenterV} title="Centralizar verticalmente"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M2 11v2h5v8h3V13h4v6h3v-6h5v-2h-5V5h-3v6h-4V3H7v8H2z"/></svg></button>
              <button onClick={alignNodesBottom} title="Alinhar abaixo"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M22 22v-2H2v2h20zM7 2v16h3V2H7zm7 6v10h3V8h-3z"/></svg></button>
            </div>
          </>
        )}

        {selectedNodes.length >= 3 && (
          <>
            <div className="alignment-separator" />
            <div className="alignment-group">
              <span className="alignment-group-label">Distribuir</span>
              <button onClick={distributeNodesH} title="Distribuir horizontalmente"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M4 5v14H2V5h2zm4 2v10h3V7H8zm5 2v6h3V9h-3zm5-2v10h3V7h-3zm4-2v14h2V5h-2z"/></svg></button>
              <button onClick={distributeNodesV} title="Distribuir verticalmente"><svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M5 4h14V2H5v2zm2 4h10v3H7V8zm2 5h6v3H9v-3zm-2 5h10v3H7v-3zm-2 4h14v2H5v-2z"/></svg></button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
