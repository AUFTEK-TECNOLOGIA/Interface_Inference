export default function ConfirmDeleteModal({ open, confirmDelete, onCancel, onConfirm }) {
  if (!open) return null;

  return (
    <div className="confirm-modal" onClick={onCancel}>
      <div className="confirm-modal-inner" onClick={(e) => e.stopPropagation()}>
        <h4>Confirmar exclusão</h4>
        {confirmDelete.nodeIds.length <= 1 ? (
          <p>Tem certeza que deseja excluir o bloco <strong>{confirmDelete.nodeId}</strong> ? Esta ação removerá o bloco e suas conexões.</p>
        ) : (
          <p>Tem certeza que deseja excluir <strong>{confirmDelete.nodeIds.length}</strong> blocos? Esta ação removerá os blocos e suas conexões.</p>
        )}
        <div className="confirm-modal-actions">
          <button className="btn" onClick={onCancel}>Cancelar</button>
          <button className="btn btn-danger" onClick={onConfirm}>Excluir</button>
        </div>
      </div>
    </div>
  );
}
