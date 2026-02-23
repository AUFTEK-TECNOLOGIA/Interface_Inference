import React from "react";
import Chevron from "./Chevron";

/**
 * Card Base - Componente padronizado para todos os grupos de informação
 * 
 * Props:
 * - title: Título do card (string)
 * - icon: Ícone opcional (ReactNode)
 * - actions: Elementos React para área de ações no header
 * - children: Conteúdo do card
 * - variant: 'default' | 'success' | 'error' | 'warning' | 'info'
 * - collapsible: Se true, card pode ser expandido/recolhido
 * - defaultExpanded: Estado inicial se collapsible
 * - noPadding: Remove padding do conteúdo
 * - className: Classes CSS adicionais
 */
const Card = ({
  title,
  icon,
  actions,
  children,
  variant = "default",
  collapsible = false,
  defaultExpanded = true,
  noPadding = false,
  className = "",
}) => {
  const [isExpanded, setIsExpanded] = React.useState(defaultExpanded);

  const variantClasses = {
    default: "",
    success: "card--success",
    error: "card--error",
    warning: "card--warning",
    info: "card--info",
  };

  return (
    <div className={`card ${variantClasses[variant]} ${className}`}>
      {title && (
        <div 
          className={`card__header ${collapsible ? "card__header--clickable" : ""}`}
          onClick={collapsible ? () => setIsExpanded(!isExpanded) : undefined}
        >
          <div className="card__title">
            {icon && <span className="card__icon">{icon}</span>}
            <span>{title}</span>
          </div>
          <div className="card__actions">
            {actions}
            {collapsible && (
              <span className="card__toggle">
                <Chevron expanded={isExpanded} size={18} />
              </span>
            )}
          </div>
        </div>
      )}
      {(!collapsible || isExpanded) && (
        <div className={`card__content ${noPadding ? "card__content--no-padding" : ""}`}>
          {children}
        </div>
      )}
    </div>
  );
};

/**
 * Card.Section - Seção interna do card com título opcional
 */
Card.Section = ({ title, icon, children, className = "" }) => (
  <div className={`card__section ${className}`}>
    {title && (
      <div className="card__section-header">
        {icon && <span className="card__section-icon">{icon}</span>}
        <span>{title}</span>
      </div>
    )}
    <div className="card__section-content">
      {children}
    </div>
  </div>
);

/**
 * Card.Badge - Badge para status ou contadores
 */
Card.Badge = ({ children, variant = "default" }) => (
  <span className={`card__badge card__badge--${variant}`}>
    {children}
  </span>
);

/**
 * Card.EmptyState - Estado vazio padronizado
 */
Card.EmptyState = ({ icon, message }) => (
  <div className="card__empty">
    {icon && <span className="card__empty-icon">{icon}</span>}
    <p>{message}</p>
  </div>
);

/**
 * Card.Code - Bloco de código/JSON
 */
Card.Code = ({ children, maxHeight = 300 }) => (
  <pre className="card__code" style={{ maxHeight: `${maxHeight}px` }}>
    {children}
  </pre>
);

export default Card;
