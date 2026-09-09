"use strict";

// Run with: node tests/browser_state.cjs
// Executes the unmodified browser client with an in-memory DOM, HTTP transport,
// and viewer. This tests application state, not WebGL rendering or NIfTI decoding;
// those still require the real-browser acceptance described in docs/validation.md.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const staticRoot = path.resolve(__dirname, "../src/medsegagent/web_static");
const source = fs.readFileSync(path.join(staticRoot, "app.js"), "utf8");
const html = fs.readFileSync(path.join(staticRoot, "index.html"), "utf8");
const sliceTypes = {
  AXIAL: 0,
  CORONAL: 1,
  SAGITTAL: 2,
  MULTIPLANAR: 3,
  RENDER: 4,
};

function multiOutputTask(id = "A", status = "completed") {
  return {
    ...task(id),
    status,
    result: {
      summary: "已整理双肺和结节结果。",
      outputs: [
        {
          id: "lungs",
          name: "双肺",
          labels: [{ id: 1, name: "lungs", voxels: 50, volume_ml: 5 }],
          files: [
            {
              name: "lungs.nii.gz",
              url: `/api/tasks/${id}/files/lungs.nii.gz`,
            },
          ],
        },
        {
          id: "nodules",
          name: "肺结节",
          labels: [{ id: 1, name: "lung_nodules", voxels: 2, volume_ml: 0.2 }],
          files: [
            {
              name: "nodules.nii.gz",
              url: `/api/tasks/${id}/files/nodules.nii.gz`,
            },
          ],
        },
      ],
    },
  };
}

class Element {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.listeners = new Map();
    this.hidden = false;
    this.checked = false;
    this.value = "";
    this.text = "";
    const classes = new Set();
    this.classList = {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
    };
  }

  set textContent(value) {
    this.text = String(value);
    this.children = [];
  }

  get textContent() {
    return this.text + this.children.map((child) => child.textContent).join("");
  }

  set href(value) {
    this.attributes.href = String(value);
  }

  get href() {
    return this.attributes.href || "";
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.text = "";
    this.children = children;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  removeAttribute(name) {
    delete this.attributes[name];
  }

  addEventListener(name, callback) {
    const listeners = this.listeners.get(name) || [];
    listeners.push(callback);
    this.listeners.set(name, listeners);
  }

  async dispatch(name, properties = {}) {
    const event = { target: this, preventDefault() {}, ...properties };
    for (const callback of this.listeners.get(name) || [])
      await callback(event);
  }

  querySelectorAll(selector) {
    assert.equal(selector, "input", "extend the DOM fake for a new selector");
    return this.children.flatMap((child) => [
      ...(child.tagName === "INPUT" ? [child] : []),
      ...child.querySelectorAll(selector),
    ]);
  }

  reset() {}
  focus() {}
  scrollIntoView() {}
  close() {
    this.open = false;
  }
  showModal() {
    this.open = true;
  }
}

function storageFake(denied, entries) {
  const values = new Map(entries);
  function checkAccess() {
    if (denied) throw new DOMException("Storage disabled", "SecurityError");
  }
  const methods = {
    key(index) {
      checkAccess();
      return [...values.keys()][index] ?? null;
    },
    getItem(key) {
      checkAccess();
      return values.get(key) ?? null;
    },
    setItem(key, value) {
      checkAccess();
      values.set(key, String(value));
    },
    removeItem(key) {
      checkAccess();
      values.delete(key);
    },
  };
  return new Proxy(methods, {
    get(target, key) {
      if (key === "length") {
        checkAccess();
        return values.size;
      }
      return target[key];
    },
    ownKeys() {
      checkAccess();
      return [...values.keys()];
    },
    getOwnPropertyDescriptor() {
      return { enumerable: true, configurable: true };
    },
  });
}

class Viewer {
  constructor(options) {
    this.opts = { ...options };
    this.gl = {};
    this.volumes = [];
    this.mediaUrlMap = new Map();
    this.scene = {
      crosshairPos: [0.5, 0.5, 0.5],
      pan2Dxyzmm: [0, 0, 0, 1],
      renderAzimuth: 110,
      renderElevation: 10,
      volScaleMultiplier: 1,
    };
  }

  async attachToCanvas(canvas) {
    this.canvas = canvas;
  }
  setMouseEventConfig(config) {
    this.mouseConfig = config;
  }
  setSliceType(value) {
    this.sliceType = value;
  }
  clearCustomLayout() {
    this.layout = null;
  }
  setCustomLayout(layout) {
    this.layout = layout;
  }
  addVolume(volume) {
    this.volumes.push(volume);
  }
  removeVolume(volume) {
    this.volumes = this.volumes.filter((item) => item !== volume);
  }
  setOpacity(index, value) {
    this.volumes[index].opacity = value;
    this.updateGLVolume();
  }
  setPan2Dxyzmm(value) {
    this.scene.pan2Dxyzmm = value;
  }
  setScale(value) {
    this.scene.volScaleMultiplier = value;
  }
  setRenderAzimuthElevation(azimuth, elevation) {
    this.scene.renderAzimuth = azimuth;
    this.scene.renderElevation = elevation;
  }
  createOnLocationChange() {
    this.onLocationChange?.({ mm: [0, 0, 0] });
  }
  drawScene() {}
  updateGLVolume() {}
  resizeListener() {}
}

function task(id) {
  return {
    id,
    text: `Synthetic task ${id}`,
    upload_id: id,
    upload_name: `${id}.nii`,
    status: "completed",
    files: [
      {
        kind: "overlay",
        name: "segmentation.nii.gz",
        url: `/api/tasks/${id}/files/segmentation.nii.gz`,
      },
      ...[
        [5, "liver"],
        [6, "spleen"],
      ].map(([label_id, label_name]) => ({
        kind: "label",
        label_id,
        label_name,
        name: `${label_id}_${label_name}.nii.gz`,
        url: `/api/tasks/${id}/files/${label_id}_${label_name}.nii.gz`,
      })),
    ],
    result: {
      labels: [
        { id: 5, name: "liver", voxels: 20 },
        { id: 6, name: "spleen", voxels: 10 },
      ],
    },
  };
}

async function eventually(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.fail(message);
}

function browser(t, options = {}) {
  const elements = new Map();
  for (const match of html.matchAll(/<(\w+)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const element = new Element(match[1]);
    element.id = match[2];
    element.hidden = /\bhidden\b/.test(match[0]);
    element.checked = /\bchecked\b/.test(match[0]);
    element.disabled = /\bdisabled\b/.test(match[0]);
    element.value = /\bvalue="([^"]*)"/.exec(match[0])?.[1] || "";
    const href = /\bhref="([^"]*)"/.exec(match[0])?.[1];
    if (href) element.href = href;
    elements.set(element.id, element);
  }
  const viewButtons = [
    ...html.matchAll(/<button\b[^>]*\bdata-view="([^"]+)"[^>]*>/g),
  ].map((match) => {
    const element = new Element("button");
    element.dataset.view = match[1];
    return element;
  });
  function element(id) {
    assert.ok(elements.has(id), `Unknown DOM id: ${id}`);
    return elements.get(id);
  }

  const tasks = options.tasks || [task("A"), task("B")];
  const requests = [];
  const uploads = [];
  const decodes = [];
  const slow = new Map((options.slow || []).map((url) => [url, null]));
  const failures = new Set(options.failures || []);
  const timers = new Set();
  const fakeTimers = new Map();
  const clock = options.clock ? { now: 0, wall: Date.now() } : null;
  let viewer;
  const location = { href: options.url || "http://example.test/?task=A" };
  let session = structuredClone(
    options.session || {
      authenticated: true,
      github_enabled: true,
      identity: {
        kind: "github",
        display_name: "Researcher",
        login: "researcher",
      },
    },
  );
  const sessionStorage = storageFake(
    options.storageDenied,
    options.storageEntries,
  );
  const windowEvents = new Element("window");
  const documentEvents = new Element("document");
  const context = {
    console,
    performance: clock ? { now: () => clock.now } : performance,
    Date: clock
      ? class extends Date {
          constructor(...args) {
            super(...(args.length ? args : [clock.wall]));
          }
          static now() {
            return clock.wall;
          }
        }
      : Date,
    AbortController,
    DOMException,
    crypto: require("node:crypto").webcrypto,
    XMLHttpRequest: class {
      constructor() {
        this.upload = {};
        this.headers = {};
        uploads.push(this);
      }
      open(method, url) {
        this.method = method;
        this.url = url;
      }
      setRequestHeader(name, value) {
        this.headers[name] = value;
      }
      send(file) {
        this.file = file;
      }
      abort() {
        this.aborted = true;
        this.onabort?.();
        this.onloadend?.();
      }
      complete(data, status = 201) {
        if (this.aborted) return;
        this.status = status;
        this.responseText = JSON.stringify(data);
        this.onload?.();
        this.onloadend?.();
      }
      networkError() {
        this.onerror?.();
        this.onloadend?.();
      }
    },
    URL,
    location,
    sessionStorage,
    history: {
      replaceState(_state, _title, url) {
        location.href = String(url);
      },
    },
    setTimeout(callback, delay) {
      if (clock) {
        const timer = { callback, due: clock.now + delay };
        fakeTimers.set(timer, timer);
        return timer;
      }
      const timer = setTimeout(
        () => {
          timers.delete(timer);
          callback();
        },
        options.fastRetries && delay <= 1000 ? 0 : delay,
      );
      timers.add(timer);
      return timer;
    },
    clearTimeout(timer) {
      if (clock) {
        fakeTimers.delete(timer);
        return;
      }
      clearTimeout(timer);
      timers.delete(timer);
    },
    document: {
      hidden: false,
      getElementById: element,
      createElement: (tag) => new Element(tag),
      addEventListener: documentEvents.addEventListener.bind(documentEvents),
      querySelectorAll(selector) {
        assert.ok(["[data-view]", "button[data-view]"].includes(selector));
        return viewButtons;
      },
    },
    window: {
      addEventListener: windowEvents.addEventListener.bind(windowEvents),
      niivue: {
        SLICE_TYPE: sliceTypes,
        SHOW_RENDER: { ALWAYS: 1 },
        DRAG_MODE: { crosshair: 8, windowing: 9, pan: 3 },
        Niivue: class extends Viewer {
          constructor(config) {
            super(config);
            assert.equal(
              element("workspace").hidden,
              false,
              "NiiVue initializes only after its canvas workspace is visible",
            );
            viewer = this;
          }
        },
        NVImage: {
          async loadFromUrl({ url, ...imageOptions }) {
            decodes.push(imageOptions.name);
            if (options.beforeDecode) await options.beforeDecode(imageOptions);
            assert.ok(
              url instanceof ArrayBuffer,
              "client decodes an already fetched buffer",
            );
            return {
              ...imageOptions,
              // Deliberately permuted native/RAS dimensions catch slider-axis mistakes.
              dimsRAS: [3, 30, 10, 20],
              hdr: { dims: [3, 10, 20, 30] },
              img: new Uint8Array(0),
              robust_min: 0,
              robust_max: 100,
              cal_min: imageOptions.cal_min ?? 0,
              cal_max: imageOptions.cal_max ?? 100,
              setColormapLabel(map) {
                this.labelMap = map;
              },
            };
          },
        },
      },
    },
    async fetch(url, request = {}) {
      requests.push({
        url,
        signal: request.signal,
        method: request.method,
        body: request.body,
        credentials: request.credentials,
        headers: request.headers,
      });
      const json = (value) => ({
        ok: true,
        json: async () => structuredClone(value),
      });
      if (slow.has(url)) {
        await new Promise((resolve, reject) => {
          const abort = () =>
            reject(new DOMException("Request aborted", "AbortError"));
          if (request.signal.aborted) return abort();
          if (!options.ignoreAbort)
            request.signal.addEventListener("abort", abort, { once: true });
          slow.set(url, resolve);
        });
      }
      if (options.handleFetch) {
        const result = await options.handleFetch(url, request, json);
        if (result) return result;
      }
      if (url === "/api/session") {
        if (request.method === "DELETE")
          session = {
            authenticated: false,
            github_enabled: session.github_enabled,
          };
        return json(session);
      }
      if (url === "/api/auth/guest") {
        assert.equal(request.method, "POST");
        session = {
          authenticated: true,
          github_enabled: session.github_enabled,
          identity: { kind: "guest", display_name: "游客" },
        };
        return json(session);
      }
      if (url === "/api/config")
        return json({ max_upload_bytes: 1024, ...options.config });
      if (url === "/api/tasks")
        return json(
          session.identity?.kind === "guest" && options.guestTasks !== undefined
            ? options.guestTasks
            : tasks,
        );
      const selected = /^\/api\/tasks\/([^/]+)$/.exec(url)?.[1];
      if (selected) return json(tasks.find((item) => item.id === selected));
      if (failures.delete(url))
        return {
          ok: false,
          status: 404,
          json: async () => ({ detail: { code: "FILE_NOT_FOUND" } }),
        };
      const example = /^\/api\/examples\/([^/]+)$/.exec(url)?.[1];
      if (example) {
        assert.equal(request.method, "POST");
        return json({
          id: `example-${example}`,
          example_id: example,
          name: `${example}.nii.gz`,
          size: 128,
          shape: [3, 4, 5],
          spacing: [1, 1, 1],
        });
      }
      const upload = /^\/api\/uploads\/([^/]+)$/.exec(url)?.[1];
      if (upload)
        return json({ id: upload, name: `${upload}.nii`, available: true });
      assert.match(
        url,
        /^\/api\/(uploads\/[^/]+\/file|tasks\/[^/]+\/files\/[a-z0-9_.-]+\.nii(?:\.gz)?)$/i,
      );
      return { ok: true, arrayBuffer: async () => new ArrayBuffer(0) };
    },
  };

  vm.runInNewContext(source, context, { filename: "app.js" });
  t.after(() => {
    for (const release of slow.values()) release?.();
    for (const timer of timers) clearTimeout(timer);
  });
  return {
    element,
    requests,
    uploads,
    decodes,
    tasks,
    slow,
    sessionStorage,
    location,
    windowEvent: (name) => windowEvents.dispatch(name),
    async visibility(hidden) {
      context.document.hidden = hidden;
      await documentEvents.dispatch("visibilitychange");
    },
    async advance(ms) {
      assert.ok(clock, "use clock:true for timing tests");
      clock.now += ms;
      clock.wall += ms;
      const ready = [...fakeTimers.values()].filter(
        (timer) => timer.due <= clock.now,
      );
      for (const timer of ready) {
        if (!fakeTimers.delete(timer)) continue;
        await timer.callback();
      }
    },
    jumpWall(ms) {
      clock.wall += ms;
    },
    get pendingTimers() {
      return fakeTimers.size;
    },
    get viewer() {
      return viewer;
    },
    async loaded(id = "A") {
      await eventually(
        () =>
          element("canvas-shell").dataset.loaded === "true" &&
          element("canvas-shell").dataset.task === id &&
          element("viewer-indicator").hidden,
        `Task ${id} did not become visible: ${element("viewer-error").textContent}`,
      );
    },
    async select(id) {
      const buttons = element("task-list").children.map(
        (item) => item.children[0],
      );
      const button = buttons.find((item) =>
        item.textContent.includes(`Synthetic task ${id}`),
      );
      assert.ok(button, `Task ${id} exists in the rendered history`);
      await button.dispatch("click");
    },
    async view(mode) {
      await viewButtons
        .find((button) => button.dataset.view === mode)
        .dispatch("click");
    },
    async example(id, compact = false) {
      const container = element(
        compact ? "example-switch-list" : "example-cards",
      );
      const button = container.children
        .map((card) => card.children[0])
        .find((item) => item.dataset.example === id);
      assert.ok(button, `Example ${id} exists`);
      await button.dispatch("click");
    },
  };
}

test("the initial workspace is visible without a separate login screen", async (t) => {
  const b = browser(t, {
    session: { authenticated: false, github_enabled: true },
    slow: ["/api/session"],
  });
  assert.equal(b.element("workspace").hidden, false);
  for (const id of ["choose-image", "file", "refresh-tasks", "instruction", "new-task"])
    assert.equal(b.element(id).disabled, true, `${id} waits for a usable session`);
  assert.doesNotMatch(
    html,
    /login-dialog|guest-button|login-error|access-token|login-form|访问令牌|管理员|凭据/,
  );
  assert.doesNotMatch(source, /login-dialog|guest-button|access-token|login-submit|login-form/);
  assert.equal(b.requests.some((row) => row.url === "/api/auth/guest"), false);
  const release = b.slow.get("/api/session");
  b.slow.delete("/api/session");
  release();
  await b.loaded();
});

test("first entry automatically creates a guest cookie before loading the workspace data", async (t) => {
  const b = browser(t, {
    session: { authenticated: false, github_enabled: true },
  });
  await b.loaded();
  assert.equal(b.element("workspace").hidden, false);
  assert.equal(b.element("account-name").textContent, "未登录");
  assert.equal(b.element("logout").hidden, true);
  assert.equal(b.element("github-login").hidden, false);
  assert.equal(b.element("github-login").href, "/api/auth/github/start");
  assert.deepEqual(
    b.requests.slice(0, 4).map((request) => request.url),
    ["/api/session", "/api/auth/guest", "/api/config", "/api/tasks"],
  );
  const request = b.requests.find((row) => row.url === "/api/auth/guest");
  assert.equal(request.method, "POST");
  assert.equal(request.credentials, "same-origin");
  assert.equal(request.body, undefined);
  assert.equal(request.headers.Authorization, undefined);
  assert.equal(
    b.requests.some((row) => row.url === "/api/session" && row.method === "POST"),
    false,
  );
  assert.equal(b.viewer.volumes.length, 2);
});

for (const kind of ["guest", "github"]) {
  test(`an existing ${kind} cookie is reused without creating another guest`, async (t) => {
    const b = browser(t, {
      session: {
        authenticated: true,
        github_enabled: true,
        identity: { kind, display_name: "Existing Researcher", login: "researcher" },
      },
    });
    await b.loaded();
    assert.equal(b.requests.filter((row) => row.url === "/api/session").length, 1);
    assert.equal(b.requests.some((row) => row.url === "/api/auth/guest"), false);
    assert.equal(b.element("account-name").textContent,
      kind === "guest" ? "未登录" : "Existing Researcher");
    assert.equal(b.element("logout").hidden, kind === "guest");
    assert.equal(b.element("github-login").hidden, kind === "github");
  });
}

test("unconfigured GitHub access stays visible and disabled for guests", async (t) => {
  const b = browser(t, {
    session: { authenticated: false, github_enabled: false },
  });
  await b.loaded();
  assert.equal(b.element("account-name").textContent, "未登录");
  assert.equal(b.element("github-login").hidden, false);
  assert.equal(b.element("github-login").href, "");
  assert.equal(b.element("github-login").attributes["aria-disabled"], "true");
});

test("a failed guest request leaves the workspace visible and can be retried inline", async (t) => {
  let attempts = 0;
  const b = browser(t, {
    session: { authenticated: false, github_enabled: true },
    handleFetch(url) {
      if (url === "/api/auth/guest" && ++attempts === 1)
        return {
          ok: false,
          status: 503,
          json: async () => ({ detail: "暂时无法建立游客会话" }),
        };
    },
  });
  await eventually(
    () => !b.element("session-error").hidden,
    "Guest connection error did not appear",
  );
  assert.equal(b.element("workspace").hidden, false);
  assert.equal(b.element("session-error").textContent, "暂时无法建立游客会话");
  assert.equal(b.element("session-retry").hidden, false);
  assert.equal(b.element("session-retry").disabled, false);
  assert.match(html, /id="session-retry"[^>]*>\s*重新连接\s*</);
  assert.equal(b.element("file").disabled, true);
  assert.equal(b.requests.some((row) => row.url === "/api/config" || row.url === "/api/tasks"), false);
  await b.element("session-retry").dispatch("click");
  await b.loaded();
  assert.equal(b.element("session-error").hidden, true);
  assert.equal(attempts, 2);
  assert.equal(b.requests.filter((row) => row.url === "/api/session").length, 2);
});

test("retry after configuration fails reuses the guest cookie already issued", async (t) => {
  let configAttempts = 0;
  const b = browser(t, {
    session: { authenticated: false, github_enabled: true },
    handleFetch(url) {
      if (url === "/api/config" && ++configAttempts === 1)
        return {
          ok: false,
          status: 503,
          json: async () => ({ detail: "配置暂时不可用" }),
        };
    },
  });
  await eventually(() => !b.element("session-error").hidden, "Configuration error did not appear");
  assert.equal(b.element("workspace").hidden, false);
  assert.equal(b.element("file").disabled, true);
  await b.element("session-retry").dispatch("click");
  await b.loaded();
  assert.equal(configAttempts, 2);
  assert.equal(b.requests.filter((row) => row.url === "/api/auth/guest").length, 1);
  assert.equal(b.requests.filter((row) => row.url === "/api/session").length, 2);
});

for (const pending of ["/api/session", "/api/auth/guest"]) {
  test(`pending ${pending} prevents duplicate connection and protected actions`, async (t) => {
    const b = browser(t, {
      session: { authenticated: false, github_enabled: true },
      slow: [pending],
    });
    await eventually(() => typeof b.slow.get(pending) === "function", "Connection did not wait");
    assert.equal(b.element("workspace").hidden, false);
    assert.equal(b.element("session-retry").disabled, true);
    for (const id of ["choose-image", "file", "refresh-tasks", "instruction", "new-task"])
      assert.equal(b.element(id).disabled, true);
    await b.element("session-retry").dispatch("click");
    await b.element("refresh-tasks").dispatch("click");
    b.element("file").files = [{ name: "too-early.nii", size: 128 }];
    await b.element("file").dispatch("change");
    assert.equal(b.uploads.length, 0);
    assert.equal(b.requests.some((row) => row.url === "/api/config" || row.url === "/api/tasks"), false);
    assert.equal(b.requests.filter((row) => row.url === pending).length, 1);
    const release = b.slow.get(pending);
    b.slow.delete(pending);
    release();
    await b.loaded();
    assert.equal(b.requests.filter((row) => row.url === "/api/auth/guest").length, 1);
  });
}

test("OAuth failure remains explained after guest entry and is removed from the return URL", async (t) => {
  const b = browser(t, {
    url: "http://example.test/?task=A&error=oauth",
    session: { authenticated: false, github_enabled: true },
  });
  await b.loaded();
  assert.equal(b.element("session-error").hidden, false);
  assert.equal(b.element("session-error").textContent, "GitHub 登录未完成，请重试。");
  const url = new URL(b.location.href);
  assert.equal(url.searchParams.has("error"), false);
  assert.equal(url.searchParams.get("task"), "A");
  assert.equal(b.element("github-login").hidden, false);
  assert.equal(b.element("account-name").textContent, "未登录");
});

test("an initial image returning 401 leaves a clean workspace awaiting explicit reconnect", async (t) => {
  let expired = true;
  const b = browser(t, {
    session: {
      authenticated: true,
      github_enabled: true,
      identity: { kind: "guest", display_name: "游客" },
    },
    clock: true,
    handleFetch(url) {
      if (expired && url === "/api/uploads/A/file")
        return { ok: false, status: 401 };
    },
  });
  await eventually(
    () => !b.element("session-error").hidden && !b.element("session-retry").disabled,
    "Expired session did not offer reconnect",
  );
  assert.equal(b.element("workspace").hidden, false);
  assert.equal(b.element("session-error").textContent, "会话已过期，请重新连接。");
  assert.equal(b.element("session-retry").hidden, false);
  assert.equal(b.element("task-list").children.length, 0);
  assert.equal(b.viewer.volumes.length, 0);
  for (const id of ["choose-image", "file", "refresh-tasks", "instruction", "new-task"])
    assert.equal(b.element(id).disabled, true);
  const requestCount = b.requests.length;
  await b.advance(60000);
  assert.equal(b.requests.length, requestCount, "Expired bootstrap cannot restart polling or create a guest loop");
  expired = false;
  await b.element("session-retry").dispatch("click");
  await b.loaded();
  assert.equal(b.element("session-error").hidden, true);
  assert.equal(b.requests.filter((row) => row.url === "/api/session").length, 2);
  assert.equal(b.requests.some((row) => row.url === "/api/auth/guest"), false);
});

test("GitHub identity is shown as text and logout clears account-specific view state", async (t) => {
  const name = "Alice <repo> & 医学";
  const b = browser(t, {
    session: {
      authenticated: true,
      github_enabled: true,
      identity: { kind: "github", display_name: name, login: "alice" },
    },
    guestTasks: [],
    storageEntries: [
      ["medseg-output:A", "native"],
      ["unrelated-setting", "keep"],
    ],
  });
  await b.loaded();
  assert.equal(b.element("account-name").textContent, name);
  assert.equal(b.element("account-name").children.length, 0);
  assert.equal(b.element("logout").hidden, false);
  assert.equal(b.element("github-login").hidden, true);
  await b.element("logout").dispatch("click");
  assert.equal(b.element("workspace").hidden, false);
  assert.equal(b.element("account-name").textContent, "未登录");
  assert.equal(b.element("logout").hidden, true);
  assert.equal(b.element("github-login").hidden, false);
  assert.equal(b.element("task-list").children.length, 0);
  assert.equal(b.sessionStorage.getItem("medseg-output:A"), null);
  assert.equal(b.sessionStorage.getItem("medseg-view:A"), null);
  assert.equal(b.sessionStorage.getItem("unrelated-setting"), "keep");
});

const exampleConfig = {
  capabilities: {
    summary: "支持 CT 和 MR 解剖结构分割。",
    limits: ["仅支持三维影像。", "暂不支持任意目标。"],
  },
  examples: ["ct", "mr", "chest"].map((id) => ({
    id,
    title: `${id.toUpperCase()} 示例`,
    modality: id === "mr" ? "MR" : "CT",
    description: "匿名研究影像",
    size_bytes: 128,
    preview_url: `/api/examples/${id}/preview`,
    prompts: [
      { label: "肝脏", text: `请分割 ${id === "mr" ? "MR" : "CT"} 中的肝脏。` },
      {
        label: "左右肾",
        text: `请分割 ${id === "mr" ? "MR" : "CT"} 中的左右肾。`,
      },
    ],
    attribution: {
      label: "研究数据",
      url: "https://example.org/data",
      license: "CC BY 4.0",
      notice_url: `/api/examples/${id}/license`,
    },
  })),
};

test("examples load source images and suggestion buttons only fill the request", async (t) => {
  const b = browser(t, { url: "http://example.test/", config: exampleConfig });
  await eventually(
    () => b.element("file").disabled === false,
    "Workspace did not become ready",
  );
  assert.equal(b.element("example-cards").children.length, 3);
  assert.equal(
    b.element("example-cards").children[0].children[0].children[0].src,
    "/api/examples/ct/preview",
  );
  assert.equal(b.element("example-switch").hidden, true);
  const credits = b.element("example-cards").children[0].children[1].children;
  assert.equal(credits[0].href, "https://example.org/data");
  assert.equal(credits[2].href, "/api/examples/ct/license");
  assert.equal(credits[2].textContent, "许可说明");
  assert.equal(credits[2].attributes.download, "");
  await b.example("ct");
  await b.loaded("");
  assert.equal(b.element("canvas-shell").dataset.source, "example-ct");
  assert.equal(b.viewer.volumes.length, 1);
  assert.equal(b.element("file-meta").tagName, "DL");
  assert.equal(b.element("file-meta").hidden, false);
  assert.equal(b.element("file-size").textContent, "0.0 MiB");
  assert.equal(b.element("file-shape").textContent, "3 × 4 × 5");
  assert.equal(b.element("file-spacing").textContent, "1.00 × 1.00 × 1.00 mm");
  assert.equal(b.element("instruction").value, "");
  assert.equal(b.element("example-prompts").children.length, 2);
  assert.equal(b.element("example-switch").hidden, false);
  assert.equal(b.element("submit").disabled, true);
  await b.element("example-prompts").children[0].dispatch("click");
  assert.equal(b.element("instruction").value, "请分割 CT 中的肝脏。");
  assert.equal(b.element("submit").disabled, false);
  assert.equal(
    b.requests.some(
      (request) => request.url === "/api/tasks" && request.method === "POST",
    ),
    false,
  );
  await b.example("mr", true);
  await b.loaded("");
  assert.equal(b.element("canvas-shell").dataset.source, "example-mr");
  assert.equal(b.element("instruction").value, "");
  assert.equal(b.element("example-switch").open, false);
});

test("example records restore without exposing draft prompts", async (t) => {
  const a = task("A");
  a.input = { id: "A", name: "A.nii", example_id: "ct" };
  const b = browser(t, { tasks: [a], config: exampleConfig });
  await b.loaded();
  assert.equal(b.element("record-request").hidden, false);
  assert.equal(b.element("example-prompts").hidden, true);
  await b.element("new-task").dispatch("click");
  assert.equal(b.element("record-request").hidden, true);
});

test("gallery license notices require the example's authenticated same-origin route", async (t) => {
  for (const notice of [
    "https://other.test/license",
    "/api/examples/mr/license",
    "javascript:alert(1)",
  ]) {
    const config = structuredClone(exampleConfig);
    config.examples[0].attribution.notice_url = notice;
    const b = browser(t, { url: "http://example.test/", config });
    await eventually(
      () => b.element("file").disabled === false,
      "Workspace did not become ready",
    );
    const credits = b.element("example-cards").children[0].children[1].children;
    assert.equal(credits.length, 1);
    assert.equal(credits[0].href, "https://example.org/data");
  }
});

for (const next of ["new", "upload", "record", "logout", "example"]) {
  test(`a late example response cannot overwrite ${next}`, async (t) => {
    const b = browser(t, {
      url: "http://example.test/",
      config: exampleConfig,
      ignoreAbort: true,
    });
    await eventually(
      () => b.element("file").disabled === false,
      "Workspace did not become ready",
    );
    b.slow.set("/api/examples/ct", null);
    const pending = b.example("ct");
    await eventually(
      () => typeof b.slow.get("/api/examples/ct") === "function",
      "Example did not wait",
    );
    if (next === "new") await b.element("new-task").dispatch("click");
    if (next === "upload") {
      b.element("file").files = [{ name: "mine.nii", size: 128 }];
      await b.element("file").dispatch("change");
      b.uploads[0].complete({ id: "mine", name: "mine.nii", size: 128 });
      await b.loaded("");
    }
    if (next === "record") {
      await b.select("A");
      await b.loaded();
    }
    if (next === "logout") await b.element("logout").dispatch("click");
    if (next === "example") {
      await b.example("mr");
      await b.loaded("");
    }
    b.slow.get("/api/examples/ct")();
    await pending;
    assert.equal(
      b.requests.some(
        (request) => request.url === "/api/uploads/example-ct/file",
      ),
      false,
    );
    if (next === "new") {
      assert.equal(b.element("viewer-heading").textContent, "新建分割");
      assert.equal(b.element("viewer-indicator").hidden, true);
    }
    if (next === "upload") {
      assert.equal(b.element("canvas-shell").dataset.source, "mine");
      assert.equal(b.element("example-prompts").hidden, true);
    }
    if (next === "record")
      assert.equal(b.element("canvas-shell").dataset.task, "A");
    if (next === "logout") assert.equal(b.element("workspace").hidden, false);
    if (next === "example")
      assert.equal(b.element("canvas-shell").dataset.source, "example-mr");
  });
}

test("example errors are recoverable and previews cannot point outside the authenticated origin", async (t) => {
  const config = structuredClone(exampleConfig);
  config.examples.unshift({
    id: "remote",
    preview_url: "https://other.test/preview.png",
  });
  const b = browser(t, {
    url: "http://example.test/",
    config,
    failures: ["/api/examples/ct"],
  });
  await eventually(
    () => b.element("file").disabled === false,
    "Workspace did not become ready",
  );
  assert.equal(b.element("example-cards").children.length, 3);
  await b.example("ct");
  assert.equal(b.element("form-error").hidden, false);
  assert.equal(b.element("submit").disabled, true);
  assert.equal(b.element("viewer-indicator").hidden, true);
  await b.example("ct");
  await b.loaded("");
  assert.equal(b.element("form-error").hidden, true);
});

test("request failures stay readable and record metadata hides implementation names", async (t) => {
  for (const code of ["UNSUPPORTED_REQUEST", "MODALITY_CONFLICT"]) {
    const a = task("A");
    a.status = "failed";
    a.error = {
      code,
      message: "internal TotalSegmentator total_mr deepseek-v4-flash",
    };
    a.result.task = "total_mr";
    const b = browser(t, { tasks: [a] });
    await b.loaded();
    assert.doesNotMatch(
      b.element("task-error").textContent,
      /TotalSegmentator|total_mr|deepseek/,
    );
    assert.doesNotMatch(b.element("request-meta").textContent, /total_mr/);
    if (code === "UNSUPPORTED_REQUEST")
      assert.equal(
        b.element("task-error").textContent,
        "当前不支持这项分割需求，未开始分割。",
      );
    else assert.match(b.element("task-error").textContent, /不一致/);
  }
});

for (const pending of ["/api/config", "/api/tasks"]) {
  test(`workspace waits for ${pending} before accepting the first upload`, async (t) => {
    const b = browser(t, { url: "http://example.test/", slow: [pending] });
    await eventually(
      () => typeof b.slow.get(pending) === "function",
      "Initial request did not wait",
    );
    assert.equal(b.element("workspace").hidden, false);
    for (const id of ["choose-image", "file", "refresh-tasks", "instruction", "new-task"])
      assert.equal(b.element(id).disabled, true);
    const file = { name: "first.nii.gz", size: 128 };
    b.element("file").files = [file];
    await b.element("file").dispatch("change");
    assert.equal(b.uploads.length, 0, "unready controls cannot start an upload");
    assert.equal(b.element("form-error").hidden, true);

    const release = b.slow.get(pending);
    b.slow.delete(pending);
    release();
    await eventually(
      () => b.element("file").disabled === false,
      "Workspace controls stayed disabled",
    );
    assert.equal(b.element("viewer-heading").textContent, "新建分割");
    assert.equal(
      b.viewer,
      undefined,
      "Home opens without restoring an old image",
    );
    await b.element("file").dispatch("change");
    assert.equal(b.uploads.length, 1);
    assert.equal(b.uploads[0].file, file);
    assert.equal(b.element("form-error").hidden, true);
    b.uploads[0].complete({ id: "first", name: file.name, size: file.size });
    await b.loaded("");
    assert.equal(b.element("canvas-shell").dataset.source, "first");
    assert.equal(b.element("file-name").textContent, file.name);
    assert.equal(b.element("instruction").value, "");
    assert.equal(
      b.element("submit").disabled,
      true,
      "Viewing needs no segmentation request",
    );
    assert.equal(
      b.requests.some(
        (request) => request.url === "/api/tasks" && request.method === "POST",
      ),
      false,
    );
  });
}

test("A → slow B → A keeps the selected task and source aligned", async (t) => {
  const b = browser(t);
  await b.loaded();
  b.slow.set("/api/uploads/B/file", null);
  await b.select("B");
  await eventually(
    () => b.requests.some((item) => item.url === "/api/uploads/B/file"),
    "B download did not start",
  );
  await b.select("A");
  await b.loaded("A");
  assert.equal(b.element("canvas-shell").dataset.source, "A");
  assert.equal(b.viewer.volumes[0].name, "A.nii");
  assert.match(
    b.element("download-labels").children[0].href,
    /\/tasks\/A\/files\//,
  );
  assert.ok(
    b.requests.find((item) => item.url === "/api/uploads/B/file").signal
      .aborted,
  );
});

test("logout aborts an in-flight download and removes volumes", async (t) => {
  const b = browser(t);
  await b.loaded();
  b.slow.set("/api/uploads/B/file", null);
  await b.select("B");
  await eventually(
    () => b.requests.some((item) => item.url === "/api/uploads/B/file"),
    "B download did not start",
  );
  await b.element("logout").dispatch("click");
  assert.equal(b.element("workspace").hidden, false);
  assert.equal(b.element("account-name").textContent, "未登录");
  assert.equal(b.viewer.volumes.length, 0);
  assert.ok(
    b.requests.find((item) => item.url === "/api/uploads/B/file").signal
      .aborted,
  );
});

test("disabled browser storage does not prevent login, overlay display or logout", async (t) => {
  const b = browser(t, { storageDenied: true });
  await b.loaded();
  assert.equal(b.viewer.volumes[1].opacity, 0.55);
  await b.element("logout").dispatch("click");
  assert.equal(b.element("workspace").hidden, false);
  assert.equal(b.element("account-name").textContent, "未登录");
});

test("corrupt view preferences fall back to a visible segmentation", async (t) => {
  const b = browser(t, { storageEntries: [["medseg-view:A", "invalid JSON"]] });
  await b.loaded();
  assert.equal(b.viewer.volumes[1].opacity, 0.55);
  assert.equal(b.viewer.sliceType, sliceTypes.MULTIPLANAR);
  assert.equal(b.viewer.layout.length, 4);
  assert.equal(b.element("viewer-error").hidden, true);
});

test("a failed image download can be retried without changing tasks", async (t) => {
  const b = browser(t, { failures: ["/api/uploads/A/file"] });
  await eventually(
    () => !b.element("retry-viewer").hidden,
    "Retry control did not appear",
  );
  assert.equal(b.element("viewer-error").hidden, false);
  assert.equal(b.viewer.volumes.length, 0);
  await b.element("retry-viewer").dispatch("click");
  await b.loaded();
  assert.equal(b.element("viewer-error").hidden, true);
  assert.equal(b.element("retry-viewer").hidden, true);
  assert.match(
    b.element("download-labels").children[0].href,
    /\/tasks\/A\/files\//,
  );
});

test("each task restores its view, window, labels, opacity and 3D camera", async (t) => {
  const b = browser(t);
  await b.loaded();
  await b.view("sagittal");
  b.viewer.scene.crosshairPos = [0.15, 0.25, 0.75];
  b.viewer.setPan2Dxyzmm([1, 2, 3, 1.5]);
  b.viewer.setScale(1.2);
  b.viewer.setRenderAzimuthElevation(45, 30);
  await b.element("niivue-canvas").dispatch("pointerup");
  b.element("opacity").value = "70";
  await b.element("opacity").dispatch("input");
  b.element("crosshair-toggle").checked = false;
  await b.element("crosshair-toggle").dispatch("change");
  b.element("window-preset").value = "soft";
  await b.element("window-preset").dispatch("change");
  const liver = b
    .element("labels")
    .querySelectorAll("input")
    .find((input) => input.dataset.label === "5");
  liver.checked = false;
  await liver.dispatch("change");
  await b.select("B");
  await b.loaded("B");
  assert.equal(b.viewer.volumes[1].opacity, 0.55);
  await b.select("A");
  await b.loaded("A");
  assert.deepEqual(Array.from(b.viewer.scene.crosshairPos), [0.15, 0.25, 0.75]);
  assert.deepEqual(Array.from(b.viewer.scene.pan2Dxyzmm), [1, 2, 3, 1.5]);
  assert.equal(b.viewer.scene.renderAzimuth, 45);
  assert.equal(b.viewer.scene.renderElevation, 30);
  assert.equal(b.viewer.scene.volScaleMultiplier, 1.2);
  assert.equal(b.viewer.sliceType, sliceTypes.SAGITTAL);
  assert.equal(b.viewer.volumes[1].opacity, 0.7);
  assert.equal(b.viewer.opts.crosshairWidth, 0);
  assert.equal(b.viewer.volumes[0].cal_min, -160);
  assert.equal(b.viewer.volumes[0].cal_max, 240);
  assert.equal(b.element("window-width").value, "400");
  assert.equal(b.element("window-level").value, "40");
  assert.deepEqual(
    b
      .element("labels")
      .querySelectorAll("input")
      .map((input) => input.checked),
    [false, true],
  );
});

test("slice sliders use RAS dimensions and place the crosshair at voxel centers", async (t) => {
  const b = browser(t);
  await b.loaded();
  for (const [plane, axis, count] of [
    ["axial", 2, 20],
    ["sagittal", 0, 30],
    ["coronal", 1, 10],
  ]) {
    const input = b.element(`slice-${plane}`);
    assert.equal(input.max, count - 1);
    input.value = String(count - 1);
    await input.dispatch("input");
    assert.equal(b.viewer.scene.crosshairPos[axis], (count - 0.5) / count);
    assert.equal(
      b.element(`slice-${plane}-value`).value,
      `${count} / ${count}`,
    );
  }
});

test("leaving the page saves a new slice before the debounce timer runs", async (t) => {
  const b = browser(t);
  await b.loaded();
  const storedView = () =>
    JSON.parse(b.sessionStorage.getItem("medseg-view:A"));
  const previousSlice = storedView().frac[2];
  b.element("slice-axial").value = "3";
  await b.element("slice-axial").dispatch("input");
  assert.equal(b.viewer.scene.crosshairPos[2], 3.5 / 20);
  assert.equal(
    storedView().frac[2],
    previousSlice,
    "the save is still debounced",
  );
  await b.windowEvent("pagehide");
  assert.equal(storedView().frac[2], 3.5 / 20);
});

test("manual windowing shows a custom preset and reset restores automatic windowing", async (t) => {
  const b = browser(t);
  await b.loaded();
  const volume = b.viewer.volumes[0];
  assert.equal(b.element("window-preset").value, "auto");
  assert.equal(b.element("window-width").value, "100");
  assert.equal(b.element("window-level").value, "50");

  // Continuous NiiVue windowing changes the volume without an intensity callback.
  volume.cal_min = -40;
  volume.cal_max = 80;
  await b.element("niivue-canvas").dispatch("pointerup");
  assert.equal(b.element("window-preset").value, "custom");
  assert.equal(volume.cal_min, -40);
  assert.equal(volume.cal_max, 80);
  assert.equal(b.element("window-width").value, "120");
  assert.equal(b.element("window-level").value, "20");

  await b.element("reset-view").dispatch("click");
  assert.equal(b.element("window-preset").value, "auto");
  assert.equal(volume.cal_min, volume.robust_min);
  assert.equal(volume.cal_max, volume.robust_max);
  assert.equal(b.element("window-width").value, "100");
  assert.equal(b.element("window-level").value, "50");
});

test("numeric windowing applies on Enter and blur, and restores per record", async (t) => {
  const b = browser(t);
  await b.loaded();
  b.element("window-width").value = "300.5";
  b.element("window-level").value = "-20.25";
  await b.element("window-width").dispatch("keydown", { key: "Enter" });
  assert.equal(b.viewer.volumes[0].cal_min, -170.5);
  assert.equal(b.viewer.volumes[0].cal_max, 130);
  assert.equal(b.element("window-preset").value, "custom");
  assert.equal(b.element("window-error").hidden, true);
  b.element("window-level").value = "10.5";
  await b.element("window-level").dispatch("blur");
  assert.equal(b.viewer.volumes[0].cal_min, -139.75);
  assert.equal(b.viewer.volumes[0].cal_max, 160.75);
  await b.select("B");
  await b.loaded("B");
  assert.equal(b.element("window-width").value, "100");
  await b.select("A");
  await b.loaded("A");
  assert.equal(b.element("window-width").value, "300.5");
  assert.equal(b.element("window-level").value, "10.5");
  assert.equal(b.viewer.volumes[0].cal_min, -139.75);
  assert.equal(b.viewer.volumes[0].cal_max, 160.75);
});

test("invalid numeric windows never reach viewer or saved preferences", async (t) => {
  const b = browser(t);
  await b.loaded();
  for (const [width, level] of [
    ["", "0"],
    ["0", "0"],
    ["-2", "0"],
    ["NaN", "0"],
    ["Infinity", "0"],
    ["100", ""],
    ["100", "NaN"],
    ["1e308", "1e308"],
    ["0.01", "1e308"],
  ]) {
    b.element("window-width").value = width;
    b.element("window-level").value = level;
    await b.element("window-width").dispatch("keydown", { key: "Enter" });
    assert.equal(
      b.element("window-error").hidden,
      false,
      `${width} / ${level}`,
    );
    assert.equal(b.viewer.volumes[0].cal_min, 0);
    assert.equal(b.viewer.volumes[0].cal_max, 100);
    await b.windowEvent("pagehide");
    assert.deepEqual(
      JSON.parse(b.sessionStorage.getItem("medseg-view:A")).window,
      ["auto", 0, 100],
    );
  }
  b.element("window-preset").value = "soft";
  await b.element("window-preset").dispatch("change");
  assert.equal(b.element("window-error").hidden, true);
  assert.equal(b.element("window-width").value, "400");
  assert.equal(b.element("window-level").value, "40");
});

test("intensity callbacks synchronize numeric fields and clearing the image disables them", async (t) => {
  const b = browser(t);
  await b.loaded();
  b.viewer.volumes[0].cal_min = -125;
  b.viewer.volumes[0].cal_max = 75;
  b.viewer.onIntensityChange();
  assert.equal(b.element("window-preset").value, "custom");
  assert.equal(b.element("window-width").value, "200");
  assert.equal(b.element("window-level").value, "-25");
  await b.element("new-task").dispatch("click");
  assert.equal(b.element("window-width").disabled, true);
  assert.equal(b.element("window-level").disabled, true);
  assert.equal(b.element("window-width").value, "");
  assert.equal(b.element("window-level").value, "");
});

test("label color metadata drives legend and overlay; only volume is displayed", async (t) => {
  const a = task("A");
  a.result.total_seconds = 5.5;
  a.result.duration_seconds = 4;
  a.result.volume_measurement = { unit_assumption: "assumed_mm" };
  a.result.labels = [
    {
      id: 1,
      source_id: 5,
      name: "liver",
      color: "#ff0000",
      voxels: 12000,
      volume_ml: 24,
      component_count: 3,
      largest_component_voxels: 11000,
      largest_component_volume_ml: 22,
    },
    {
      id: 2,
      source_id: 1,
      name: "spleen",
      color: "#12AbCd",
      voxels: 0,
      volume_ml: 0,
      component_count: 0,
      largest_component_voxels: 0,
      largest_component_volume_ml: 0,
    },
  ];
  const b = browser(t, { tasks: [a] });
  await b.loaded();
  const rows = b.element("labels").children;
  assert.equal(rows[0].children[1].style.backgroundColor, "rgb(255,0,0)");
  assert.equal(rows[1].children[1].style.backgroundColor, "rgb(18,171,205)");
  assert.match(rows[0].textContent, /24 mL/);
  assert.match(rows[0].title, /12,000 体素/);
  assert.doesNotMatch(rows[0].title + rows[0].textContent, /连通/);
  assert.match(rows[1].textContent, /0 mL/);
  const map = b.viewer.volumes[1].labelMap;
  assert.deepEqual(Array.from(map.I), [0, 1, 2]);
  assert.deepEqual(Array.from(map.R), [0, 255, 18]);
  assert.deepEqual(Array.from(map.G), [0, 0, 171]);
  assert.deepEqual(Array.from(map.B), [0, 0, 205]);
  assert.equal(
    b.requests.some((request) => request.url.endsWith("result.json")),
    false,
  );
  assert.doesNotMatch(html, /download-result|结构化结果/);
  assert.match(b.element("result-summary").textContent, /处理用时 5\.5 秒/);
  assert.match(rows[0].children[2].children[1].title, /体积按 mm 估算/);
  assert.doesNotMatch(
    source,
    /component_count|largest_component|连通区|请核查分割边界与标签|文件保留|retentionHours/,
  );
  assert.doesNotMatch(
    html,
    /file-retention|label-help|连通区|文件保留|关于分割记录/,
  );
  assert.equal(b.element("status-detail").hidden, true);
});

test("older labels preserve IDs, use a red first class and mark missing volume", async (t) => {
  const b = browser(t);
  await b.loaded();
  const row = b.element("labels").children[0];
  assert.equal(row.children[1].style.backgroundColor, "rgb(255,0,0)");
  assert.match(row.textContent, /体积未计算/);
  assert.deepEqual(Array.from(b.viewer.volumes[1].labelMap.I), [0, 5, 6]);
  assert.equal(b.viewer.volumes[1].labelMap.R[1], 255);
});

test("downloads show numeric label order and exclude the combined overlay or external paths", async (t) => {
  const a = task("A");
  const file = (label_id, label_name) => ({
    kind: "label",
    label_id,
    label_name,
    name: `${label_id}_${label_name}.nii.gz`,
    url: `/api/tasks/A/files/${label_id}_${label_name}.nii.gz`,
  });
  a.files = [
    a.files[0],
    file(10, "spleen"),
    {
      ...file(2, "liver"),
      url: "http://example.test/api/tasks/A/files/2_liver.nii.gz",
    },
    { ...file(3, "kidney_left"), url: "https://example.org/private.nii.gz" },
    {
      ...file(4, "kidney_right"),
      url: "/api/tasks/B/files/4_kidney_right.nii.gz",
    },
    { ...file(7, "pancreas"), name: "../7_pancreas.nii.gz" },
  ];
  const b = browser(t, { tasks: [a] });
  await b.loaded();
  const links = b.element("download-labels").children;
  assert.equal(links.length, 2);
  assert.equal(links[0].href, "/api/tasks/A/files/2_liver.nii.gz");
  assert.equal(links[1].href, "/api/tasks/A/files/10_spleen.nii.gz");
  assert.equal(links[0].attributes.download, "2_liver.nii.gz");
  assert.match(links[0].textContent, /2\. 肝脏/);
  assert.doesNotMatch(html, /download-mask/);
  assert.equal(
    b.viewer.volumes.length,
    2,
    "combined overlay still drives the viewer",
  );
  await b.element("new-task").dispatch("click");
  assert.equal(b.element("download-labels").hidden, true);
  assert.equal(b.element("download-labels").children.length, 0);
});

test("a record with an expired source still exposes available result files without requesting the source", async (t) => {
  const a = {
    ...task("A"),
    input_available: false,
    result_available: true,
  };
  const b = browser(t, { tasks: [a] });
  await eventually(
    () => !b.element("result-panel").hidden,
    "Available result metadata was not displayed",
  );
  assert.equal(b.element("download-source").hidden, true);
  assert.equal(b.element("download-source").href, "");
  assert.equal(b.element("reuse-image").hidden, true);
  assert.equal(b.element("download-labels").hidden, false);
  assert.match(
    b.element("download-labels").children[0].href,
    /\/tasks\/A\/files\//,
  );
  assert.equal(b.element("labels").querySelectorAll("input").length, 2);
  assert.match(
    b.element("viewer-error").textContent,
    /原始影像.*到期|原始影像.*不可用/,
  );
  assert.doesNotMatch(b.element("status-detail").textContent, /可查看叠加/);
  assert.equal(b.element("canvas-shell").dataset.loaded, "false");
  assert.equal(
    b.requests.some((item) => item.url.includes("/file")),
    false,
  );
});

test("a fully expired record retains its request but never offers dead file links or inference reuse", async (t) => {
  const b = browser(t, {
    tasks: [{ ...task("A"), input_available: false, result_available: false }],
  });
  await eventually(
    () => !b.element("record-request").hidden,
    "Expired record did not open",
  );
  assert.equal(b.element("request-text").textContent, "Synthetic task A");
  assert.match(b.element("task-list").textContent, /文件已清理/);
  for (const id of ["download-source"]) {
    assert.equal(b.element(id).hidden, true, `${id} is unavailable`);
    assert.equal(
      b.element(id).href,
      "",
      `${id} must not retain a previous URL`,
    );
  }
  assert.equal(b.element("download-labels").hidden, true);
  assert.equal(b.element("download-labels").children.length, 0);
  assert.equal(b.element("reuse-image").hidden, true);
  assert.equal(b.element("result-panel").hidden, true);
  assert.equal(b.element("result-empty").hidden, false);
  assert.equal(b.element("downloads").hidden, true);
  assert.match(b.element("result-empty").textContent, /到期|清理/);
  assert.equal(
    b.requests.some((item) => item.url.includes("/file")),
    false,
  );
});

test("result expiry while the same source-expired record is open clears previously visible labels", async (t) => {
  const a = { ...task("A"), input_available: false, result_available: true };
  const b = browser(t, { tasks: [a] });
  await eventually(
    () => !b.element("result-panel").hidden,
    "Initial available result was not displayed",
  );
  a.result_available = false;
  a.result = null;
  await b.select("A");
  assert.equal(b.element("result-panel").hidden, true);
  assert.equal(b.element("result-empty").hidden, false);
  assert.equal(b.element("download-labels").hidden, true);
  assert.equal(b.element("download-labels").children.length, 0);
  assert.equal(b.element("canvas-shell").dataset.loaded, "false");
});

test("reuse validates the source and record selection clears the submittable draft", async (t) => {
  const b = browser(t);
  await b.loaded();
  assert.equal(b.element("request").hidden, true);
  assert.equal(b.element("submit").disabled, true);
  await b.element("reuse-image").dispatch("click");
  await eventually(
    () =>
      b.element("canvas-shell").dataset.task === "" &&
      b.element("canvas-shell").dataset.loaded === "true",
    "Validated source did not open as a fresh draft",
  );
  assert.ok(b.requests.some((item) => item.url === "/api/uploads/A"));
  assert.equal(b.element("instruction").value, "Synthetic task A");
  assert.equal(b.element("request").hidden, false);
  assert.equal(b.element("record-request").hidden, true);
  assert.equal(b.element("submit").disabled, false);
  assert.equal(b.element("download-labels").hidden, true);
  assert.equal(b.element("download-labels").children.length, 0);
  await b.select("B");
  await b.loaded("B");
  assert.equal(b.element("submit").disabled, true);
  assert.equal(b.element("request").hidden, true);
  assert.equal(b.element("request-text").textContent, "Synthetic task B");
  assert.match(
    b.element("download-labels").children[0].href,
    /\/tasks\/B\/files\//,
  );
  await b.element("new-task").dispatch("click");
  assert.equal(b.element("instruction").value, "");
  assert.equal(b.element("submit").disabled, true);
  assert.equal(b.viewer.volumes.length, 0);
  assert.equal(b.element("downloads").hidden, true);
});

test("a source that expires before reuse cannot become a submittable draft", async (t) => {
  const b = browser(t, { failures: ["/api/uploads/A"] });
  await b.loaded();
  await b.element("reuse-image").dispatch("click");
  assert.equal(b.element("request").hidden, true);
  assert.equal(b.element("submit").disabled, true);
  assert.equal(b.element("reuse-image").disabled, false);
  assert.match(b.element("task-error").textContent, /过期/);
  assert.equal(b.element("canvas-shell").dataset.task, "A");
});

test("a delayed reuse validation cannot overwrite a subsequently selected record", async (t) => {
  const b = browser(t);
  await b.loaded();
  b.slow.set("/api/uploads/A", null);
  const reuse = b.element("reuse-image").dispatch("click");
  await eventually(
    () => typeof b.slow.get("/api/uploads/A") === "function",
    "Reuse validation did not start",
  );
  await b.select("B");
  await b.loaded("B");
  b.slow.get("/api/uploads/A")();
  await reuse;
  assert.equal(b.element("canvas-shell").dataset.task, "B");
  assert.equal(b.element("request-text").textContent, "Synthetic task B");
  assert.equal(b.element("request").hidden, true);
  assert.equal(b.element("submit").disabled, true);
});

test("record search matches requests and image names without changing the open result", async (t) => {
  const b = browser(t);
  await b.loaded();
  const search = b.element("history-search");
  search.value = "  b.NII  ";
  await search.dispatch("input");
  assert.equal(b.element("task-list").children.length, 1);
  assert.match(b.element("task-list").textContent, /Synthetic task B/);
  assert.equal(b.element("canvas-shell").dataset.task, "A");
  assert.match(
    b.element("download-labels").children[0].href,
    /\/tasks\/A\/files\//,
  );
  search.value = "Synthetic task A";
  await search.dispatch("input");
  assert.equal(b.element("task-list").children.length, 1);
  assert.equal(
    b.element("task-list").children[0].children[0].attributes["aria-current"],
    "true",
  );
  search.value = "unmatched query";
  await search.dispatch("input");
  assert.equal(b.element("task-list").children.length, 0);
  assert.equal(b.element("history-empty").hidden, false);
  assert.match(b.element("history-empty").textContent, /没有匹配/);
  search.value = "";
  await search.dispatch("input");
  assert.equal(b.element("task-list").children.length, 2);
  assert.equal(b.element("history-empty").hidden, true);
});

test("request and result tabs respond to keyboard navigation", async (t) => {
  const b = browser(t);
  await b.loaded();
  assert.equal(b.element("tab-results").attributes["aria-selected"], "true");
  await b.element("tab-results").dispatch("keydown", { key: "Home" });
  assert.equal(b.element("panel-request").hidden, false);
  assert.equal(b.element("panel-results").hidden, true);
  assert.equal(b.element("tab-request").attributes.tabindex, "0");
  await b.element("tab-request").dispatch("keydown", { key: "ArrowRight" });
  assert.equal(b.element("panel-request").hidden, true);
  assert.equal(b.element("panel-results").hidden, false);
  assert.equal(b.element("tab-results").attributes.tabindex, "0");
});

test("finishing an earlier upload cannot replace a record selected during upload", async (t) => {
  const b = browser(t);
  await b.loaded();
  await b.element("new-task").dispatch("click");
  b.element("file").files = [{ name: "draft.nii", size: 128 }];
  await b.element("file").dispatch("change");
  assert.equal(b.uploads.length, 1);
  assert.equal(b.element("submit").disabled, true);
  await b.select("B");
  await b.loaded("B");
  b.uploads[0].complete({ id: "D", name: "draft.nii", size: 128 });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(b.element("record-request").hidden, false);
  assert.equal(b.element("request-text").textContent, "Synthetic task B");
  assert.equal(b.element("canvas-shell").dataset.task, "B");
  assert.equal(b.element("request").hidden, true);
  assert.equal(b.element("submit").disabled, true);
});

const chunkConfig = {
  max_upload_bytes: 1024,
  single_upload_bytes: 4,
  upload_chunk_bytes: 4,
};
const chunkFile = {
  name: "分块影像.nii.gz",
  size: 10,
  slice: (start, end) => ({ start, end, size: end - start }),
};
const uploadSession = (offset = 0) => ({
  id: "U",
  offset,
  total_bytes: 10,
  chunk_bytes: 4,
});

async function beginChunkUpload(b) {
  await b.loaded();
  await b.element("new-task").dispatch("click");
  b.element("file").files = [chunkFile];
  await b.element("file").dispatch("change");
}

test("chunk uploads retry lost acknowledgements with identical identity and bytes, then only open the source", async (t) => {
  let creates = 0,
    completions = 0;
  const b = browser(t, {
    config: chunkConfig,
    fastRetries: true,
    handleFetch(url, request, json) {
      if (url === "/api/upload-sessions") {
        if (++creates === 1) throw new TypeError("Lost create acknowledgement");
        return json(uploadSession());
      }
      if (url === "/api/upload-sessions/U/complete") {
        if (++completions === 1)
          throw new TypeError("Lost completion acknowledgement");
        return json({
          id: "uploaded-U",
          name: chunkFile.name,
          size: 10,
          shape: [2, 3, 4],
          spacing: [0.74, 0.74, 2.5],
        });
      }
    },
  });
  await beginChunkUpload(b);
  await eventually(() => b.uploads.length === 1, "First chunk was not sent");
  assert.equal(b.uploads[0].method, "PUT");
  b.uploads[0].networkError();
  await eventually(
    () => b.uploads.length === 2,
    "Lost chunk acknowledgement was not retried",
  );
  assert.deepEqual(b.uploads[1].file, b.uploads[0].file);
  assert.equal(b.uploads[1].headers["Upload-Offset"], "0");
  b.uploads[1].complete(uploadSession(4));
  await eventually(() => b.uploads.length === 3, "Second chunk was not sent");
  assert.deepEqual(b.uploads[2].file, { start: 4, end: 8, size: 4 });
  b.uploads[2].upload.onprogress({
    loaded: 2,
    total: 4,
    lengthComputable: true,
  });
  assert.equal(b.element("upload-bar").value, 60);
  b.uploads[2].complete(uploadSession(8));
  await eventually(() => b.uploads.length === 4, "Final chunk was not sent");
  assert.deepEqual(b.uploads[3].file, { start: 8, end: 10, size: 2 });
  b.uploads[3].complete(uploadSession(10));
  await b.loaded("");
  assert.equal(b.element("canvas-shell").dataset.source, "uploaded-U");
  assert.equal(b.viewer.volumes.length, 1);
  assert.equal(b.element("file-name").textContent, chunkFile.name);
  assert.equal(b.element("file-shape").textContent, "2 × 3 × 4");
  assert.equal(b.element("file-spacing").textContent, "0.74 × 0.74 × 2.50 mm");
  assert.equal(b.element("submit").disabled, true);
  const starts = b.requests.filter((r) => r.url === "/api/upload-sessions");
  assert.equal(starts.length, 2);
  assert.equal(starts[0].body, starts[1].body);
  assert.ok(JSON.parse(starts[0].body).message_id);
  assert.equal(completions, 2);
  assert.equal(
    b.requests.some(
      (r) =>
        r.method === "DELETE" ||
        (r.url === "/api/tasks" && r.method === "POST"),
    ),
    false,
  );
});

test("chunk network retries are bounded and abandoned sessions are cleaned up", async (t) => {
  const b = browser(t, {
    config: chunkConfig,
    fastRetries: true,
    handleFetch(url, request, json) {
      if (url === "/api/upload-sessions") return json(uploadSession());
      if (request.method === "DELETE") return json({});
    },
  });
  await beginChunkUpload(b);
  for (let attempt = 0; attempt < 3; attempt++) {
    await eventually(
      () => b.uploads.length === attempt + 1,
      "Expected retry did not start",
    );
    b.uploads[attempt].networkError();
  }
  await eventually(
    () => !b.element("form-error").hidden,
    "Exhausted retries did not surface an error",
  );
  assert.equal(b.uploads.length, 3);
  assert.equal(b.element("upload-progress").hidden, true);
  assert.equal(
    b.requests.filter(
      (r) => r.url === "/api/upload-sessions/U" && r.method === "DELETE",
    ).length,
    1,
  );
  assert.equal(
    b.requests.some((r) => r.url.endsWith("/complete")),
    false,
  );
});

for (const action of ["new", "record", "logout"]) {
  test(`canceling a chunk upload by ${action} aborts transport and prevents stale completion`, async (t) => {
    const b = browser(t, {
      config: chunkConfig,
      handleFetch(url, request, json) {
        if (url === "/api/upload-sessions") return json(uploadSession());
        if (request.method === "DELETE") return json({});
      },
    });
    await beginChunkUpload(b);
    await eventually(() => b.uploads.length === 1, "Chunk did not start");
    const xhr = b.uploads[0];
    if (action === "record") await b.select("B");
    else
      await b
        .element(action === "new" ? "new-task" : "logout")
        .dispatch("click");
    assert.equal(xhr.aborted, true);
    // Model a transport whose completion callback was already queued when aborted.
    xhr.status = 200;
    xhr.responseText = JSON.stringify(uploadSession(4));
    xhr.onload();
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(b.uploads.length, 1);
    assert.equal(
      b.requests.filter(
        (r) => r.method === "DELETE" && r.url === "/api/upload-sessions/U",
      ).length,
      1,
    );
    assert.equal(
      b.requests.some((r) => r.url.endsWith("/complete")),
      false,
    );
    if (action === "record") await b.loaded("B");
    else assert.equal(b.element("viewer-heading").textContent, "新建分割");
  });
}

test("a late upload-session creation is abandoned after switching to a new draft", async (t) => {
  let release;
  const b = browser(t, {
    config: chunkConfig,
    handleFetch: async (url, request, json) => {
      if (url === "/api/upload-sessions") {
        await new Promise((resolve) => {
          release = resolve;
        });
        return json(uploadSession());
      }
      if (request.method === "DELETE") return json({});
    },
  });
  await beginChunkUpload(b);
  await eventually(() => !!release, "Session creation did not start");
  await b.element("new-task").dispatch("click");
  release();
  await eventually(
    () => b.requests.some((r) => r.method === "DELETE"),
    "Late session was not cleaned up",
  );
  assert.equal(b.uploads.length, 0);
  assert.equal(b.element("viewer-heading").textContent, "新建分割");
  assert.equal(b.element("form-error").hidden, true);
});

test("an invalid chunk acknowledgement cannot skip bytes or finish the upload", async (t) => {
  const b = browser(t, {
    config: chunkConfig,
    handleFetch(url, request, json) {
      if (url === "/api/upload-sessions") return json(uploadSession());
      if (request.method === "DELETE") return json({});
    },
  });
  await beginChunkUpload(b);
  await eventually(() => b.uploads.length === 1, "Chunk did not start");
  b.uploads[0].complete(uploadSession(8));
  await eventually(
    () => !b.element("form-error").hidden,
    "Invalid acknowledgement was accepted",
  );
  assert.match(b.element("form-error").textContent, /进度校验失败/);
  assert.equal(
    b.requests.some((r) => r.url.endsWith("/complete")),
    false,
  );
  assert.equal(b.uploads.length, 1);
});

test("logout automatically enters an empty guest draft without the previous identity's record", async (t) => {
  const b = browser(t, { guestTasks: [] });
  await b.loaded();
  await b.element("logout").dispatch("click");
  assert.equal(b.element("workspace").hidden, false);
  assert.equal(b.element("account-name").textContent, "未登录");
  assert.equal(b.element("record-request").hidden, true);
  assert.equal(b.element("request-text").textContent, "");
  assert.equal(b.element("request").hidden, false);
  assert.equal(b.element("instruction").value, "");
  assert.equal(b.element("submit").disabled, true);
  assert.equal(b.element("task-list").children.length, 0);
  assert.equal(b.element("viewer-heading").textContent, "新建分割");
  assert.equal(b.element("downloads").hidden, true);
  for (const id of ["download-source"]) assert.equal(b.element(id).href, "");
  assert.equal(b.element("download-labels").children.length, 0);
});

test("multiple outputs switch their overlay, labels, downloads and saved visibility independently", async (t) => {
  const a = multiOutputTask();
  const b = browser(t, { tasks: [a] });
  await b.loaded();
  const original = b.viewer.volumes[0];
  assert.equal(b.element("result-output-choice").hidden, false);
  assert.equal(b.element("result-output").children.length, 2);
  assert.equal(b.element("result-output").value, "lungs");
  assert.equal(b.viewer.volumes[1].name, "lungs.nii.gz");
  assert.equal(
    b.element("download-labels").children[0].href,
    "/api/tasks/A/files/lungs.nii.gz",
  );
  assert.match(b.element("labels").textContent, /双肺/);
  const lungCheckbox = b.element("labels").querySelectorAll("input")[0];
  lungCheckbox.checked = false;
  await lungCheckbox.dispatch("change");
  b.element("result-output").value = "nodules";
  await b.element("result-output").dispatch("change");
  await b.loaded();
  assert.equal(
    b.element("download-labels").children[0].href,
    "/api/tasks/A/files/nodules.nii.gz",
  );
  assert.equal(b.element("labels").querySelectorAll("input")[0].checked, true);
  assert.equal(b.sessionStorage.getItem("medseg-output:A"), "nodules");
  assert.equal(b.viewer.volumes[1].name, "nodules.nii.gz");
  assert.ok(
    b.requests.some(
      (request) => request.url === "/api/tasks/A/files/nodules.nii.gz",
    ),
  );
  b.element("result-output").value = "lungs";
  await b.element("result-output").dispatch("change");
  await b.loaded();
  assert.equal(b.element("labels").querySelectorAll("input")[0].checked, false);
  assert.equal(b.viewer.volumes.length, 2);
  assert.equal(b.viewer.volumes[0], original);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    1,
  );
  assert.equal(b.decodes.filter((name) => name === "A.nii").length, 1);
});

test("the selected output survives a page reload and stale IDs fall back to an available output", async (t) => {
  for (const [saved, expected] of [
    ["nodules", "nodules"],
    ["removed", "lungs"],
  ]) {
    const b = browser(t, {
      tasks: [multiOutputTask()],
      storageEntries: [["medseg-output:A", saved]],
    });
    await b.loaded();
    assert.equal(b.element("result-output").value, expected);
    assert.equal(
      b.element("download-labels").children[0].href,
      `/api/tasks/A/files/${expected}.nii.gz`,
    );
  }
});

test("failed tasks retain available outputs while clearly marking the overall task incomplete", async (t) => {
  const a = multiOutputTask("A", "failed");
  a.error = { message: "第二个模型没有完成。" };
  a.result.completion = { unresolved: ["缺少请求的派生结果"] };
  const b = browser(t, { tasks: [a] });
  await b.loaded();
  assert.equal(b.element("result-panel").hidden, false);
  assert.equal(b.element("result-output").children.length, 2);
  assert.match(b.element("status-title").textContent, /失败/);
  assert.match(b.element("result-summary").textContent, /任务未完成/);
  assert.match(
    b.element("result-explanation").textContent,
    /已整理双肺和结节结果/,
  );
  assert.match(
    b.element("result-explanation").textContent,
    /尚未完成：缺少请求的派生结果/,
  );
  assert.match(b.element("task-error").textContent, /第二个模型/);
  assert.equal(b.element("download-labels").hidden, false);
});

test("new output file URLs cannot leave the current task", async (t) => {
  const a = multiOutputTask();
  a.result.outputs[0].files[0].url = "https://other.test/lungs.nii.gz";
  const b = browser(t, { tasks: [a] });
  await b.loaded();
  assert.equal(b.element("result-output").value, "nodules");
  assert.equal(b.element("result-output-choice").hidden, true);
  assert.equal(
    b.requests.some((request) => request.url.includes("other.test")),
    false,
  );
});

function runningTask(id = "A", elapsed = 20) {
  return {
    ...task(id),
    status: "running",
    result: null,
    files: [],
    result_available: false,
    elapsed_seconds: elapsed,
    progress: "Running local segmentation",
    agent_progress: { phase: "tool", tool: "segment", model_requests: 1 },
  };
}

test("estimated task progress ticks from server time and does not jump with wall clock", async (t) => {
  const b = browser(t, { tasks: [runningTask()], clock: true });
  await b.loaded();
  assert.equal(b.element("status-title").textContent, "正在分割");
  assert.equal(b.element("task-elapsed").textContent, "已用 20 秒");
  assert.equal(b.element("task-estimate").textContent, "预计约 60 秒");
  assert.ok(b.element("task-progress").value > 30);
  assert.equal(b.element("task-progress").attributes["aria-label"], "预估进度");
  b.jumpWall(3600000);
  await b.advance(1000);
  assert.equal(b.element("task-elapsed").textContent, "已用 21 秒");
  b.jumpWall(-7200000);
  await b.advance(1000); // Includes a server poll with a stale elapsed sample.
  assert.equal(b.element("task-elapsed").textContent, "已用 22 秒");
});

test("estimate never completes a running task and resumes after background time", async (t) => {
  const b = browser(t, { tasks: [runningTask("A", 59)], clock: true });
  await b.loaded();
  await b.advance(1000);
  assert.equal(b.element("task-elapsed").textContent, "已用 1 分钟");
  assert.equal(b.element("task-estimate").textContent, "仍在处理中");
  assert.equal(b.element("task-progress").value, 95);
  await b.visibility(true);
  await b.advance(90000);
  await b.visibility(false);
  assert.equal(b.element("task-elapsed").textContent, "已用 2 分 30 秒");
  assert.equal(b.element("task-progress").value, 95);
  assert.equal(b.element("status-title").textContent, "正在分割");
  assert.equal(b.element("cancel-task").hidden, false);
});

test("terminal elapsed is fixed in request metadata and old unknown time is omitted", async (t) => {
  const done = { ...task("A"), elapsed_seconds: 75.8 };
  const b = browser(t, { tasks: [done], clock: true });
  await b.loaded();
  assert.match(b.element("request-meta").textContent, /耗时 1 分 15 秒/);
  assert.equal(b.element("task-progress").hidden, true);
  assert.equal(b.element("task-timing").hidden, true);
  await b.advance(120000);
  assert.match(b.element("request-meta").textContent, /耗时 1 分 15 秒/);
  done.elapsed_seconds = null;
  await b.select("A");
  assert.doesNotMatch(b.element("request-meta").textContent, /耗时|NaN/);
});

test("task progress follows selected identity and stops for draft and logout", async (t) => {
  const b = browser(t, {
    tasks: [runningTask("A", 20), runningTask("B", 5)],
    guestTasks: [],
    clock: true,
  });
  await b.loaded();
  await b.advance(1000);
  await b.select("B");
  await b.loaded("B");
  assert.equal(b.element("task-elapsed").textContent, "已用 5 秒");
  await b.advance(1000);
  assert.equal(b.element("task-elapsed").textContent, "已用 6 秒");
  await b.element("new-task").dispatch("click");
  const previous = b.element("task-elapsed").textContent;
  await b.advance(10000);
  assert.equal(b.element("task-status").hidden, true);
  assert.equal(b.element("task-elapsed").textContent, previous);
  await b.select("A");
  await b.element("logout").dispatch("click");
  const loggedOut = b.element("task-elapsed").textContent;
  const detailRequests = b.requests.filter((row) => /^\/api\/tasks\/[AB]$/.test(row.url)).length;
  await b.advance(10000);
  assert.equal(b.element("task-elapsed").textContent, loggedOut);
  assert.equal(b.requests.filter((row) => /^\/api\/tasks\/[AB]$/.test(row.url)).length, detailRequests);
  assert.equal(b.element("task-list").children.length, 0);
});

test("tool phases have readable progress without internal tool names", async (t) => {
  const a = runningTask();
  const b = browser(t, { tasks: [a], clock: true });
  await b.loaded();
  for (const [phase, tool, title] of [
    ["reasoning", null, "正在理解请求"],
    ["tool", "get_capabilities", "正在匹配分割工具"],
    ["tool", "detect_modality", "正在识别影像类型"],
    ["tool", "inspect_artifact", "正在检查分割结果"],
    ["tool", "compose_masks", "正在合并分割结果"],
    ["publishing", null, "正在整理结果"],
  ]) {
    a.agent_progress = { phase, tool, model_requests: 1 };
    await b.select("A");
    assert.equal(b.element("status-title").textContent, title);
  }
});

test("failed and canceled tasks freeze duration even with partial results", async (t) => {
  const a = { ...multiOutputTask("A", "failed"), elapsed_seconds: 42.5 };
  const b = browser(t, { tasks: [a], clock: true });
  await b.loaded();
  assert.match(b.element("request-meta").textContent, /耗时 42 秒/);
  assert.equal(b.element("task-timing").hidden, true);
  assert.match(b.element("status-detail").textContent, /部分结果/);
  a.status = "canceled";
  await b.select("A");
  await b.advance(10000);
  assert.equal(b.element("status-title").textContent, "任务已取消");
  assert.match(b.element("request-meta").textContent, /耗时 42 秒/);
});

test("invalid elapsed does not produce NaN or fake completion", async (t) => {
  const a = runningTask("A", null);
  const b = browser(t, { tasks: [a], clock: true });
  await b.loaded();
  assert.equal(b.element("task-elapsed").textContent, "正在计时");
  assert.equal(b.element("task-progress").attributes.value, undefined);
  assert.doesNotMatch(b.element("task-estimate").textContent, /NaN/);
});

test("input-required tasks preserve results and show the actual question with low-frequency polling", async (t) => {
  const a = {
    ...multiOutputTask("A", "input_required"),
    elapsed_seconds: 42,
    error: {
      code: "MODALITY_REQUIRED",
      message: "请确认这是增强 CT 还是 MR？",
    },
    agent_progress: { phase: "tool", tool: "segment" },
  };
  const b = browser(t, { tasks: [a], clock: true });
  await b.loaded();
  const original = b.viewer.volumes[0];
  const overlay = b.viewer.volumes[1];
  assert.equal(b.element("status-title").textContent, "等待补充信息");
  assert.match(b.element("task-list").textContent, /等待补充信息/);
  assert.match(
    b.element("status-detail").textContent,
    /请确认这是增强 CT 还是 MR？.*通过 A2A 继续/,
  );
  assert.equal(b.element("task-error").hidden, true);
  assert.equal(b.element("task-timing").hidden, true);
  assert.equal(b.element("task-progress").hidden, true);
  assert.equal(b.element("cancel-task").hidden, false);
  assert.equal(b.element("cancel-task").disabled, false);
  assert.equal(b.element("request").hidden, true);
  assert.equal(b.element("result-panel").hidden, false);
  assert.equal(b.element("result-output").children.length, 2);
  assert.equal(b.element("download-labels").hidden, false);
  for (let i = 0; i < 14; i++) await b.advance(2000);
  assert.equal(b.requests.filter((r) => r.url === "/api/tasks/A").length, 1);
  await b.advance(2000);
  assert.equal(b.requests.filter((r) => r.url === "/api/tasks/A").length, 2);
  assert.equal(b.viewer.volumes[0], original);
  assert.equal(b.viewer.volumes[1], overlay);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    1,
  );
  assert.equal(b.decodes.filter((name) => name === "A.nii").length, 1);
});

test("input-required clarification accepts readable progress and omits structured internals", async (t) => {
  const a = { ...runningTask(), status: { state: "input_required" } };
  const b = browser(t, { tasks: [a], clock: true });
  await b.loaded();
  for (const [progress, expected] of [
    ["请选择需要保留的分割结果。", "请选择需要保留的分割结果。"],
    [{ message: "请说明目标器官。" }, "请说明目标器官。"],
    [{ text: "请补充影像模态。" }, "请补充影像模态。"],
    [{ percent: 40, phase: "tool" }, "请补充所需信息。"],
  ]) {
    a.progress = progress;
    await b.select("A");
    assert.equal(b.element("status-title").textContent, "等待补充信息");
    assert.equal(
      b.element("status-detail").textContent,
      `${expected} 通过 A2A 继续。`,
    );
    assert.doesNotMatch(
      b.element("status-detail").textContent,
      /\[object Object\]|percent|phase/,
    );
  }
});

test("work clock pauses for clarification and resumes through low-frequency polling without reloading the source", async (t) => {
  const a = runningTask();
  const b = browser(t, { tasks: [a], clock: true });
  await b.loaded();
  const original = b.viewer.volumes[0];
  Object.assign(a, {
    status: "input_required",
    elapsed_seconds: 22,
    error: { message: "请确认分割目标。" },
  });
  await b.advance(2000);
  assert.equal(b.element("status-title").textContent, "等待补充信息");
  assert.equal(b.element("task-timing").hidden, true);
  assert.equal(b.element("task-progress").hidden, true);
  const pausedText = b.element("task-elapsed").textContent;
  const requestCount = b.requests.filter(
    (r) => r.url === "/api/tasks/A",
  ).length;
  for (let i = 0; i < 14; i++) await b.advance(2000);
  assert.equal(b.element("task-elapsed").textContent, pausedText);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/tasks/A").length,
    requestCount,
  );
  Object.assign(a, { status: "running", elapsed_seconds: 23, error: null });
  await b.advance(2000);
  assert.equal(b.element("status-title").textContent, "正在分割");
  assert.equal(b.element("task-timing").hidden, false);
  assert.equal(b.element("task-progress").hidden, false);
  assert.equal(b.element("task-elapsed").textContent, "已用 23 秒");
  assert.doesNotMatch(b.element("status-detail").textContent, /A2A|请确认/);
  await b.advance(2000);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/tasks/A").length,
    requestCount + 2,
  );
  assert.equal(b.element("task-elapsed").textContent, "已用 25 秒");
  assert.equal(b.viewer.volumes[0], original);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    1,
  );
  assert.equal(b.decodes.filter((name) => name === "A.nii").length, 1);
});

test("input-required tasks without an image remain cancelable without a Web resume form", async (t) => {
  const a = {
    id: "A",
    text: "Synthetic task A",
    status: "input_required",
    input_available: false,
    result_available: false,
    error: { code: "INPUT_REQUIRED", message: "请通过 A2A 提供影像文件。" },
  };
  const b = browser(t, {
    tasks: [a],
    clock: true,
    handleFetch(url, request, json) {
      if (url === "/api/tasks/A/cancel") {
        assert.equal(request.method, "POST");
        Object.assign(a, { status: "canceled", error: null });
        return json(a);
      }
    },
  });
  await eventually(
    () => b.element("status-title").textContent === "等待补充信息",
    "waiting task was not displayed",
  );
  assert.match(
    b.element("status-detail").textContent,
    /请通过 A2A 提供影像文件.*通过 A2A 继续/,
  );
  assert.equal(b.element("viewer-error").hidden, true);
  assert.equal(b.element("empty-title").textContent, "尚未提供影像");
  assert.match(b.element("task-list").textContent, /等待补充信息/);
  assert.doesNotMatch(b.element("task-list").textContent, /文件已清理/);
  assert.equal(b.element("request").hidden, true);
  assert.equal(b.element("cancel-task").hidden, false);
  await b.element("cancel-task").dispatch("click");
  assert.equal(b.element("status-title").textContent, "任务已取消");
  assert.equal(b.element("cancel-task").hidden, true);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/tasks/A/cancel").length,
    1,
  );
  assert.equal(
    b.requests.some((r) => /continue|resume|message:send/.test(r.url)),
    false,
  );
});

test("other input-required tasks do not accelerate history refresh and manual refresh discovers expiry", async (t) => {
  const waiting = {
    ...runningTask("B"),
    status: "input_required",
    error: { message: "请补充目标。" },
  };
  const b = browser(t, { tasks: [task("A"), waiting], clock: true });
  await b.loaded();
  for (let i = 0; i < 4; i++) await b.advance(6000);
  assert.equal(b.requests.filter((r) => r.url === "/api/tasks").length, 1);
  await b.select("B");
  assert.equal(b.element("status-title").textContent, "等待补充信息");
  Object.assign(waiting, {
    status: "failed",
    error: { code: "INPUT_EXPIRED", message: "影像已过期。" },
  });
  await b.element("refresh-tasks").dispatch("click");
  assert.equal(b.element("status-title").textContent, "任务失败");
  assert.match(b.element("task-error").textContent, /影像已过期/);
  assert.equal(b.element("cancel-task").hidden, true);
});

test("a running source survives completion without another download or decode", async (t) => {
  const a = runningTask();
  const b = browser(t, { tasks: [a] });
  await b.loaded();
  const original = b.viewer.volumes[0];
  Object.assign(a, multiOutputTask(), { result_available: true });
  await b.select("A");
  await b.loaded();
  assert.equal(b.viewer.volumes[0], original);
  assert.equal(b.viewer.volumes[1].name, "lungs.nii.gz");
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    1,
  );
  assert.equal(b.decodes.filter((name) => name === "A.nii").length, 1);
});

test("completion during the source download reuses the same in-flight image", async (t) => {
  const a = runningTask();
  const b = browser(t, { tasks: [a], slow: ["/api/uploads/A/file"] });
  await eventually(
    () => typeof b.slow.get("/api/uploads/A/file") === "function",
    "source did not start",
  );
  Object.assign(a, multiOutputTask(), { result_available: true });
  await b.select("A");
  const release = b.slow.get("/api/uploads/A/file");
  b.slow.delete("/api/uploads/A/file");
  release();
  await b.loaded();
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    1,
  );
  assert.equal(b.decodes.filter((name) => name === "A.nii").length, 1);
  assert.equal(b.viewer.volumes[1].name, "lungs.nii.gz");
});

test("a ready source is visible and interactive while its selected overlay loads", async (t) => {
  const b = browser(t, {
    tasks: [multiOutputTask()],
    slow: ["/api/tasks/A/files/lungs.nii.gz"],
  });
  await eventually(
    () => typeof b.slow.get("/api/tasks/A/files/lungs.nii.gz") === "function",
    "mask did not start",
  );
  assert.equal(b.viewer.volumes.length, 1);
  assert.equal(b.element("niivue-canvas").style.visibility, "visible");
  assert.equal(b.element("canvas-shell").dataset.loaded, "true");
  assert.equal(b.element("window-width").disabled, false);
  assert.equal(b.element("viewer-indicator").dataset.stage, "overlay");
  assert.equal(
    b.element("viewer-loading-text").textContent,
    "正在载入分割结果…",
  );
  assert.equal(b.element("result-panel").hidden, false);
  assert.equal(
    b.requests.some((r) => r.url.includes("nodules.nii.gz")),
    false,
  );
  const release = b.slow.get("/api/tasks/A/files/lungs.nii.gz");
  b.slow.delete("/api/tasks/A/files/lungs.nii.gz");
  release();
  await b.loaded();
});

test("a stale slow overlay cannot block or replace a newer output selection", async (t) => {
  const url = "/api/tasks/A/files/nodules.nii.gz";
  const b = browser(t, { tasks: [multiOutputTask()], ignoreAbort: true });
  await b.loaded();
  const original = b.viewer.volumes[0];
  b.slow.set(url, null);
  b.element("result-output").value = "nodules";
  await b.element("result-output").dispatch("change");
  await eventually(
    () => typeof b.slow.get(url) === "function",
    "second mask did not start",
  );
  assert.equal(
    b.viewer.volumes.length,
    1,
    "old mask is removed before new labels are shown",
  );
  b.element("result-output").value = "lungs";
  await b.element("result-output").dispatch("change");
  await b.loaded();
  assert.equal(b.viewer.volumes[1].name, "lungs.nii.gz");
  const release = b.slow.get(url);
  b.slow.delete(url);
  release();
  await new Promise(setImmediate);
  assert.equal(b.viewer.volumes[1].name, "lungs.nii.gz");
  assert.equal(b.viewer.volumes[0], original);
  assert.equal(b.requests.find((r) => r.url === url).signal.aborted, true);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    1,
  );
});

test("overlay failure keeps the source visible and retries only the mask", async (t) => {
  const url = "/api/tasks/A/files/lungs.nii.gz";
  const b = browser(t, { tasks: [multiOutputTask()], failures: [url] });
  await eventually(
    () => !b.element("retry-viewer").hidden,
    "overlay retry absent",
  );
  const original = b.viewer.volumes[0];
  assert.ok(original);
  assert.equal(b.viewer.volumes.length, 1);
  assert.equal(b.element("niivue-canvas").style.visibility, "visible");
  assert.match(b.element("viewer-error").textContent, /分割结果显示失败/);
  await b.element("retry-viewer").dispatch("click");
  await b.loaded();
  assert.equal(b.viewer.volumes[0], original);
  assert.equal(b.requests.filter((r) => r.url === url).length, 2);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    1,
  );
});

test("logout clears a decoded source even when the next account has the same upload ID", async (t) => {
  const b = browser(t, { tasks: [multiOutputTask()] });
  await b.loaded();
  const original = b.viewer.volumes[0];
  await b.element("logout").dispatch("click");
  await b.select("A");
  await b.loaded();
  assert.notEqual(b.viewer.volumes[0], original);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    2,
  );
});

test("a source decode finishing after logout cannot populate the next account cache", async (t) => {
  let release;
  const b = browser(t, {
    tasks: [multiOutputTask()],
    beforeDecode: async (opts) => {
      if (opts.name === "A.nii" && !release)
        await new Promise((resolve) => {
          release = resolve;
        });
    },
  });
  await eventually(
    () => typeof release === "function",
    "source decode not pending",
  );
  await b.element("logout").dispatch("click");
  await b.select("A");
  await b.loaded();
  const original = b.viewer.volumes[0];
  release();
  await new Promise(setImmediate);
  assert.equal(b.viewer.volumes[0], original);
  assert.equal(b.viewer.volumes.length, 2);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    2,
  );
});

test("known source expiry discards the cached image before it can be reused", async (t) => {
  const a = multiOutputTask();
  const b = browser(t, { tasks: [a] });
  await b.loaded();
  const original = b.viewer.volumes[0];
  a.input_available = false;
  await b.select("A");
  assert.equal(b.viewer.volumes.length, 0);
  assert.equal(b.element("niivue-canvas").style.visibility, "hidden");
  a.input_available = true;
  await b.select("A");
  await b.loaded();
  assert.notEqual(b.viewer.volumes[0], original);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    2,
  );
});

test("switching to another upload releases the previous decoded image", async (t) => {
  const b = browser(t);
  await b.loaded();
  const original = b.viewer.volumes[0];
  await b.select("B");
  await b.loaded("B");
  await b.select("A");
  await b.loaded("A");
  assert.notEqual(b.viewer.volumes[0], original);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    2,
  );
});

test("different tasks on the same current upload reuse one decoded source", async (t) => {
  const a = task("A"),
    other = { ...task("B"), upload_id: "A", upload_name: "A.nii" };
  const b = browser(t, { tasks: [a, other] });
  await b.loaded();
  const original = b.viewer.volumes[0];
  await b.select("B");
  await b.loaded("B");
  assert.equal(b.viewer.volumes[0], original);
  assert.equal(b.element("canvas-shell").dataset.source, "A");
  assert.equal(
    b.requests.filter((r) => r.url === "/api/uploads/A/file").length,
    1,
  );
});

test("a canceled old overlay returning 401 cannot log out the replacement account", async (t) => {
  const url = "/api/tasks/A/files/nodules.nii.gz";
  let delayed = false;
  const b = browser(t, {
    tasks: [multiOutputTask()],
    ignoreAbort: true,
    handleFetch: async (path) =>
      delayed && path === url ? { ok: false, status: 401 } : null,
  });
  await b.loaded();
  b.slow.set(url, null);
  b.element("result-output").value = "nodules";
  await b.element("result-output").dispatch("change");
  await eventually(
    () => typeof b.slow.get(url) === "function",
    "mask was not delayed",
  );
  await b.element("logout").dispatch("click");
  await b.select("A");
  await b.loaded();
  delayed = true;
  const release = b.slow.get(url);
  b.slow.delete(url);
  release();
  await new Promise(setImmediate);
  assert.equal(b.element("workspace").hidden, false);
  assert.equal(b.element("account-name").textContent, "未登录");
  assert.equal(b.viewer.volumes.length, 2);
});

test("selected task polling no longer downloads full history on every tick", async (t) => {
  const b = browser(t, { tasks: [runningTask()], clock: true });
  await b.loaded();
  for (let i = 0; i < 5; i++) await b.advance(2000);
  assert.equal(b.requests.filter((r) => r.url === "/api/tasks/A").length, 6);
  assert.equal(b.requests.filter((r) => r.url === "/api/tasks").length, 1);
});

test("slow history refresh does not delay selected task polling or terminal state", async (t) => {
  const a = runningTask();
  const b = browser(t, { tasks: [a], clock: true });
  await b.loaded();
  b.slow.set("/api/tasks", null);
  a.status = "normalizing";
  await b.advance(2000);
  await eventually(
    () => typeof b.slow.get("/api/tasks") === "function",
    "status change did not refresh history",
  );
  const detailCount = b.requests.filter((r) => r.url === "/api/tasks/A").length;
  Object.assign(a, multiOutputTask(), {
    elapsed_seconds: 41.2,
    result_available: true,
  });
  await b.advance(2000);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/tasks/A").length,
    detailCount + 1,
  );
  assert.equal(b.element("status-title").textContent, "分割完成");
  assert.match(b.element("task-list").textContent, /分割完成/);
  assert.equal(
    b.requests.filter((r) => r.url === "/api/tasks").length,
    2,
    "pending history refresh is deduplicated",
  );
  const release = b.slow.get("/api/tasks");
  b.slow.delete("/api/tasks");
  release();
  await b.loaded();
});

test("independent history refresh discovers new tasks without a selected detail dependency", async (t) => {
  const b = browser(t, { tasks: [runningTask()], guestTasks: [], clock: true });
  await b.loaded();
  b.tasks.push(task("B"));
  await b.advance(30000);
  assert.match(b.element("task-list").textContent, /Synthetic task B/);
  assert.equal(b.requests.filter((r) => r.url === "/api/tasks").length, 2);
  await b.element("logout").dispatch("click");
  const detailRequests = b.requests.filter((row) => row.url === "/api/tasks/A").length;
  await b.advance(30000);
  assert.equal(b.requests.filter((row) => row.url === "/api/tasks/A").length, detailRequests);
  assert.equal(b.element("task-list").children.length, 0);
});
