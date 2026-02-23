import React from 'react';

const JsonEditor = ({ value, onChange, title, placeholder, height = 200 }) => {
  return (
    <fieldset className="results-group">
      <legend>{title}</legend>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{ height: `${height}px` }}
        className="json-editor-textarea"
      />
    </fieldset>
  );
};

export default JsonEditor;