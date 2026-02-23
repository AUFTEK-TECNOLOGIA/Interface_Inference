import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import { LocaleProvider } from "./locale/i18n";

const container = document.getElementById("root");
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <LocaleProvider>
      <App />
    </LocaleProvider>
  </React.StrictMode>
);
