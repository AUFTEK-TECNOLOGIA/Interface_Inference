export default function BlockCard({
  block,
  displayName,
  visual,
  isPinned,
  isActive = false,
  isMini = false,
  tags = [],
  onInspect,
  onAdd,
  onHelp,
  onTogglePin,
  onDragStart,
  t,
}) {
  if (!block) return null;

  return (
    <div
      key={block.name}
      className={`block-card${isMini ? " mini" : ""}${isActive ? " active" : ""}`}
      style={{ "--block-accent": visual.color }}
      draggable
      role="button"
      tabIndex={0}
      title={`${displayName} (${block.name})`}
      onClick={onInspect}
      onDoubleClick={onAdd}
      onDragStart={onDragStart}
      onKeyDown={(e) => {
        if (e.key === "Enter") return onAdd?.(e);
        if (!isMini && (e.key === "?" || e.key === "F1")) return onHelp?.(e);
        return undefined;
      }}
    >
      <div className="block-card-icon" aria-hidden="true">{visual.icon}</div>
      <div className="block-card-content">
        <strong>{displayName}</strong>
        {!isMini && block.description ? <small>{block.description}</small> : null}
        {!isMini && tags.length ? (
          <div className="block-card-tags">
            {tags.map((tag) => <span key={tag} className="block-card-tag is-accent">{tag}</span>)}
          </div>
        ) : null}
      </div>
      <button className={`block-card-pin${isPinned ? " is-active" : ""}`} type="button" onClick={onTogglePin} aria-label={isPinned ? t("actions.unpin") : t("actions.pin")} title={isPinned ? t("actions.unpin") : t("actions.pin")}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3h-4v2l1 1v6l-3 3v2h8v-2l-3-3V6l1-1V3Z" /></svg>
      </button>
      {!isMini && (
        <button className="block-card-help" type="button" onClick={onHelp} aria-label={t("helper.openLabel")} title={t("helper.openLabel")}>?</button>
      )}
    </div>
  );
}
