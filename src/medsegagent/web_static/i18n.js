(() => {
  "use strict";

  const storageKey = "medseg-language";
  const normalize = (locale) => (locale === "en" ? "en" : "zh-CN");
  let language = "zh-CN";
  try {
    language = normalize(window.localStorage.getItem(storageKey));
  } catch {
    // Language selection still works when browser storage is unavailable.
  }

  const bindings = new WeakMap();
  const boundElements = new Set();
  const subscribers = new Set();
  let cleanupScheduled = false;
  const t = (zh, en) => (language === "en" ? en : zh);

  function pruneDetachedBindings() {
    for (const element of boundElements) {
      if (!element.isConnected) {
        boundElements.delete(element);
        bindings.delete(element);
      }
    }
  }

  function scheduleCleanup() {
    if (cleanupScheduled) return;
    cleanupScheduled = true;
    queueMicrotask(() => {
      cleanupScheduled = false;
      pruneDetachedBindings();
    });
  }

  function renderBinding(element, attribute, render) {
    try {
      const value = String(render());
      if (attribute === null) {
        if (element.textContent !== value) element.textContent = value;
      } else if (element.getAttribute(attribute) !== value) {
        element.setAttribute(attribute, value);
      }
    } catch {
      // One unavailable display value must not interrupt the workspace.
    }
  }

  function bind(element, attribute, render) {
    if (!element || typeof render !== "function") return;
    let entry = bindings.get(element);
    if (!entry) {
      entry = new Map();
      bindings.set(element, entry);
    }
    // Rebinding replaces the old renderer, including an initial static label.
    entry.set(attribute, render);
    boundElements.add(element);
    renderBinding(element, attribute, render);
    // List items are often bound before insertion. Wait until the synchronous
    // DOM update finishes, then release items replaced during polling.
    scheduleCleanup();
  }

  function refresh() {
    document.documentElement.lang = language;
    pruneDetachedBindings();
    for (const element of boundElements) {
      for (const [attribute, render] of bindings.get(element)) {
        renderBinding(element, attribute, render);
      }
    }
    document
      .getElementById("language-zh")
      ?.setAttribute("aria-pressed", String(language === "zh-CN"));
    document
      .getElementById("language-en")
      ?.setAttribute("aria-pressed", String(language === "en"));
  }

  function setLanguage(locale) {
    language = normalize(locale);
    try {
      window.localStorage.setItem(storageKey, language);
    } catch {
      // Keep the current selection for this page even if it cannot be saved.
    }
    refresh();
    for (const callback of subscribers) {
      try {
        callback(language);
      } catch {
        // Independent displays can continue updating if one callback fails.
      }
    }
  }

  window.MedSegI18n = {
    get language() {
      return language;
    },
    t,
    setLanguage,
    bindText: (element, render) => bind(element, null, render),
    bindAttr: (element, attribute, render) => bind(element, attribute, render),
    onChange(callback) {
      if (typeof callback !== "function") return () => {};
      subscribers.add(callback);
      return () => subscribers.delete(callback);
    },
  };

  // Only explicitly marked interface text is translated. Uploaded filenames,
  // user instructions, model responses and other content are never scanned.
  for (const element of document.querySelectorAll("[data-i18n-en]")) {
    const zh = element.textContent.trim();
    const en = element.getAttribute("data-i18n-en");
    bind(element, null, () => t(zh, en));
  }
  for (const attribute of ["aria-label", "title", "placeholder"]) {
    const marker = `data-i18n-${attribute}-en`;
    for (const element of document.querySelectorAll(`[${marker}]`)) {
      const zh = element.getAttribute(attribute) || "";
      const en = element.getAttribute(marker);
      bind(element, attribute, () => t(zh, en));
    }
  }

  document.getElementById("language-zh")?.addEventListener("click", () => {
    setLanguage("zh-CN");
  });
  document.getElementById("language-en")?.addEventListener("click", () => {
    setLanguage("en");
  });
  refresh();
})();
