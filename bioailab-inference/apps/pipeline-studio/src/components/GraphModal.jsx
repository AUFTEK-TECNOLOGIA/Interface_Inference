export default function GraphModal({ open, onClose, graphList, graphIndex, graphModalTitle, graphModalSrc, onNavigate }) {
  if (!open) return null;

  return (
    <div className="graph-modal" onClick={onClose}>
      {graphList.length > 1 && (
        <button className="graph-modal-nav graph-modal-prev" onClick={(e) => { e.stopPropagation(); onNavigate(-1); }} title="Anterior (←)">
          ‹
        </button>
      )}

      <div className="graph-modal-inner" onClick={(e) => e.stopPropagation()}>
        <button className="graph-modal-close" onClick={onClose}>×</button>
        <h4>{graphModalTitle}</h4>
        <img src={graphModalSrc} alt={graphModalTitle} />
        {graphList.length > 1 && <div className="graph-modal-counter">{graphIndex + 1} / {graphList.length}</div>}
      </div>

      {graphList.length > 1 && (
        <button className="graph-modal-nav graph-modal-next" onClick={(e) => { e.stopPropagation(); onNavigate(1); }} title="Próximo (→)">
          ›
        </button>
      )}
    </div>
  );
}
