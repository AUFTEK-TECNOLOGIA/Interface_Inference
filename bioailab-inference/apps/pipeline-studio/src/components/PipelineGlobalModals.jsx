import BlockResultsModal from "./BlockResultsModal";
import GraphModal from "./GraphModal";
import ConfirmDeleteModal from "./ConfirmDeleteModal";

export default function PipelineGlobalModals({
  t,
  blockResultsModal,
  closeBlockResultsModal,
  getStepLabel,
  simulation,
  openGraphModal,
  graphModalOpen,
  closeGraphModal,
  graphList,
  graphIndex,
  graphModalTitle,
  graphModalSrc,
  handleNavigate,
  confirmDelete,
  handleCancelDelete,
  handleConfirmDelete,
}) {
  return (
    <>
      <BlockResultsModal
        open={blockResultsModal.open}
        t={t}
        onClose={closeBlockResultsModal}
        stepId={blockResultsModal.stepId}
        getStepLabel={getStepLabel}
        simulation={simulation}
        onGraphClick={openGraphModal}
      />

      <GraphModal
        open={graphModalOpen}
        onClose={closeGraphModal}
        graphList={graphList}
        graphIndex={graphIndex}
        graphModalTitle={graphModalTitle}
        graphModalSrc={graphModalSrc}
        onNavigate={handleNavigate}
      />

      <ConfirmDeleteModal
        open={confirmDelete.open}
        confirmDelete={confirmDelete}
        onCancel={handleCancelDelete}
        onConfirm={handleConfirmDelete}
      />
    </>
  );
}
