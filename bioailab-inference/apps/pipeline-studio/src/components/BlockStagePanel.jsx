export default function BlockStagePanel({ number, title, children, defaultOpen = true }) {
  return (
    <details className="stage-accordion" open={defaultOpen}>
      <summary className="stage-summary">
        <div className="stage-header">
          <span className="stage-number">{number}</span>
          <span className="stage-title">{title}</span>
        </div>
      </summary>
      {children}
    </details>
  );
}
