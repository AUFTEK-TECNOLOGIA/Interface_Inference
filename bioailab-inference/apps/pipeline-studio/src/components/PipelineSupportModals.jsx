import "./PipelineSupportModals.css";
import ResultsHelperModal from "./ResultsHelperModal";
import HelpHelperModal from "./HelpHelperModal";

export default function PipelineSupportModals(props) {
  return (
    <>
      <ResultsHelperModal {...props} />
      <HelpHelperModal {...props} />
    </>
  );
}
