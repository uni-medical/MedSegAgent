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

  async dispatch(name) {
    const event = { target: this, preventDefault() {} };
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
    element.value = /\bvalue="([^"]*)"/.exec(match[0])?.[1] || "";
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

  const tasks = [task("A"), task("B")];
  const requests = [];
  const slow = new Map();
  const failures = new Set(options.failures || []);
  const timers = new Set();
  let viewer;
  const location = { href: "http://example.test/?task=A" };
  const sessionStorage = storageFake(
    options.storageDenied,
    options.storageEntries,
  );
  const windowEvents = new Element("window");
  const context = {
    console,
    AbortController,
    URL,
    location,
    sessionStorage,
    history: {
      replaceState(_state, _title, url) {
        location.href = String(url);
      },
    },
    setTimeout(callback, delay) {
      const timer = setTimeout(() => {
        timers.delete(timer);
        callback();
      }, delay);
      timers.add(timer);
      return timer;
    },
    clearTimeout(timer) {
      clearTimeout(timer);
      timers.delete(timer);
    },
    document: {
      hidden: false,
      getElementById: element,
      createElement: (tag) => new Element(tag),
      addEventListener() {},
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
            viewer = this;
          }
        },
        NVImage: {
          async loadFromUrl({ url, ...imageOptions }) {
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
      requests.push({ url, signal: request.signal });
      const json = (value) => ({
        ok: true,
        json: async () => structuredClone(value),
      });
      if (url === "/api/session")
        return json({ authenticated: request.method !== "DELETE" });
      if (url === "/api/config") return json({ max_upload_bytes: 1024 });
      if (url === "/api/tasks") return json(tasks);
      const selected = /^\/api\/tasks\/([^/]+)$/.exec(url)?.[1];
      if (selected) return json(tasks.find((item) => item.id === selected));
      assert.match(
        url,
        /^\/api\/(uploads\/[^/]+\/file|tasks\/[^/]+\/files\/segmentation\.nii\.gz)$/,
      );
      if (slow.has(url)) {
        await new Promise((resolve, reject) => {
          const abort = () =>
            reject(new DOMException("Request aborted", "AbortError"));
          if (request.signal.aborted) return abort();
          request.signal.addEventListener("abort", abort, { once: true });
          slow.set(url, resolve);
        });
      }
      if (failures.delete(url)) return { ok: false, status: 404 };
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
    slow,
    sessionStorage,
    windowEvent: (name) => windowEvents.dispatch(name),
    get viewer() {
      return viewer;
    },
    async loaded(id = "A") {
      await eventually(
        () =>
          element("canvas-shell").dataset.loaded === "true" &&
          element("canvas-shell").dataset.task === id,
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
  };
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
  assert.match(b.element("download-mask").href, /\/tasks\/A\/files\//);
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
  assert.equal(b.element("workspace").hidden, true);
  assert.equal(b.element("login-dialog").open, true);
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
  assert.equal(b.element("workspace").hidden, true);
  assert.equal(b.element("login-dialog").open, true);
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
  assert.match(b.element("download-mask").href, /\/tasks\/A\/files\//);
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
