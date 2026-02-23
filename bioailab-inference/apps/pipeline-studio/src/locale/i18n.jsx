import React, { createContext, useCallback, useContext, useMemo, useState } from "react";
import { ptBR } from "./pt-BR";

const STORAGE_KEY = "pipelineStudio.locale";

const LOCALES = {
  "pt-BR": ptBR,
};

const LocaleContext = createContext({
  locale: "pt-BR",
  setLocale: () => {},
  t: (key) => key,
});

const getByPath = (obj, path) =>
  path.split(".").reduce((acc, part) => (acc && acc[part] != null ? acc[part] : undefined), obj);

const format = (template, vars) =>
  String(template).replace(/\{(\w+)\}/g, (_, name) =>
    Object.prototype.hasOwnProperty.call(vars || {}, name) ? String(vars[name]) : `{${name}}`
  );

export function LocaleProvider({ children, defaultLocale = "pt-BR" }) {
  const [locale, setLocaleState] = useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored && LOCALES[stored] ? stored : defaultLocale;
  });

  const setLocale = useCallback((nextLocale) => {
    if (!LOCALES[nextLocale]) return;
    localStorage.setItem(STORAGE_KEY, nextLocale);
    setLocaleState(nextLocale);
    document.documentElement.lang = nextLocale;
  }, []);

  const messages = LOCALES[locale] || LOCALES[defaultLocale];

  const t = useCallback(
    (key, vars) => {
      const value = getByPath(messages, key);
      if (value == null) return key;
      if (typeof value === "string") return format(value, vars);
      return value;
    },
    [messages]
  );

  const contextValue = useMemo(() => ({ locale, setLocale, t }), [locale, setLocale, t]);

  return <LocaleContext.Provider value={contextValue}>{children}</LocaleContext.Provider>;
}

export function useI18n() {
  return useContext(LocaleContext);
}

