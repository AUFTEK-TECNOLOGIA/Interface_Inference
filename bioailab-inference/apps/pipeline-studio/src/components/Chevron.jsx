import React from "react";

export default function Chevron({ expanded, size = 18, className = "" }) {
  return (
    <svg
      className={`icon-chevron ${expanded ? "icon-chevron--expanded" : ""} ${className}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M9 6l6 6-6 6"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

