/* Optional nnInteractive workspace. Loaded only by an enabled Web service. */
export function voxelToWorld(affine, voxel) {
  return affine.slice(0, 3).map((row) =>
    row[3] + row[0] * voxel[0] + row[1] * voxel[1] + row[2] * voxel[2]);
}

export function worldToVoxel(affine, world) {
  const [[a, b, c, x], [d, e, f, y], [g, h, i, z]] = affine;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) throw new Error("影像坐标无效");
  const p = world.map((v, k) => v - [x, y, z][k]);
  return [
    [e * i - f * h, c * h - b * i, b * f - c * e],
    [f * g - d * i, a * i - c * g, c * d - a * f],
    [d * h - e * g, b * g - a * h, a * e - b * d],
  ].map((row) => row.reduce((sum, v, k) => sum + v * p[k], 0) / det);
}

export function canvasPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) * canvas.width / rect.width,
    (event.clientY - rect.top) * canvas.height / rect.height];
}

export function prepareSubmission(body, pending, newID) {
  // A lost ACK may already have advanced the server head. It must not change
  // the identity or body of a retry for the same user intent.
  const {expected_revision: _head, ...intent} = body;
  const fingerprint = JSON.stringify(intent);
  return pending?.fingerprint === fingerprint ? pending
    : {fingerprint, body: {...body, message_id: newID()}};
}

// All four corners must form an axis-aligned rectangle on one native XYZ slice.
// Checking only a diagonal could silently turn an oblique rectangle into a 3D box.
export function boxPrompt(worldCorners, geometry, positive = true) {
  if (worldCorners.length !== 4) throw new Error("框选需要四个角点");
  const voxels = worldCorners.map((p) => worldToVoxel(geometry.affine, p));
  const fixed = [0, 1, 2].filter((axis) =>
    Math.max(...voxels.map((p) => p[axis])) - Math.min(...voxels.map((p) => p[axis])) < 1e-3);
  if (fixed.length !== 1) throw new Error("此视图无法映射为单层框，请改用点选");
  const axis = fixed[0];
  const start = voxels[0].map(Math.round), end = voxels[2].map(Math.round);
  if ([0, 1, 2].some((k) => k !== axis && start[k] === end[k]))
    throw new Error("框太小，请扩大框选范围");
  for (const edge of [[0, 1], [1, 2], [2, 3], [3, 0]]) {
    const changed = [0, 1, 2].filter((k) => Math.abs(voxels[edge[0]][k] - voxels[edge[1]][k]) > 1e-3);
    if (changed.length !== 1) throw new Error("此视图的框不与原始网格对齐，请改用点选");
  }
  if ([start, end].some((p) => p.some((v, k) => v < 0 || v >= geometry.shape[k])))
    throw new Error("请在影像内部框选");
  start[axis] = end[axis] = Math.round(voxels[0][axis]);
  return {kind: "box", positive, world_start: voxelToWorld(geometry.affine, start),
    world_end: voxelToWorld(geometry.affine, end)};
}

export function mount(bridge) {
  const api = (path, options = {}) => bridge.api(path, {
    ...options, ...(options.body ? {body: JSON.stringify(options.body)} : {}),
  });
  const $ = (id) => document.getElementById(id);
  const canvas = $("niivue-canvas"), shell = $("canvas-shell");
  const style = document.createElement("link");
  style.rel = "stylesheet"; style.href = "/static/interactive.css"; document.head.append(style);
  const panel = document.createElement("details");
  panel.className = "interactive-panel"; panel.id = "interactive-panel";
  // Static markup only. Model output and region names are always textContent.
  panel.innerHTML = `
    <summary>区域交互 <span>nnInteractive · 实验</span></summary>
    <div id="interactive-recent" hidden>
      <label for="interactive-workspaces">继续编辑</label>
      <select id="interactive-workspaces"><option value="">选择区域工作区…</option></select>
    </div>
    <p id="interactive-hint">打开影像后，用点或框指定一个区域。</p>
    <button type="button" id="interactive-open" class="secondary">编辑当前影像</button>
    <div id="interactive-editor" hidden>
      <label for="interactive-name">区域名称</label>
      <input id="interactive-name" maxlength="80" value="区域 1" />
      <div class="interactive-tools" role="group" aria-label="区域标记方式">
        <button type="button" data-mark="navigate" aria-pressed="true">浏览</button>
        <button type="button" data-mark="positive" aria-pressed="false">＋ 前景点</button>
        <button type="button" data-mark="negative" aria-pressed="false">− 背景点</button>
        <button type="button" data-mark="box" aria-pressed="false">框选</button>
      </div>
      <div class="interactive-mark-actions">
        <span id="interactive-count">0 个待应用标记</span>
        <button type="button" id="interactive-undo" class="quiet">撤销标记</button>
        <button type="button" id="interactive-clear" class="quiet">清空标记</button>
      </div>
      <button type="button" id="interactive-apply" class="primary">应用标记</button>
      <label for="interactive-instruction">交给 Agent</label>
      <textarea id="interactive-instruction" rows="2" maxlength="4000"
        placeholder="例如：按我的标记修正区域，并比较体积变化。"></textarea>
      <button type="button" id="interactive-agent" class="secondary">执行区域任务</button>
      <button type="button" id="interactive-cancel" class="quiet" hidden>取消任务</button>
      <div id="interactive-history" hidden>
        <label for="interactive-revision">分割版本</label>
        <select id="interactive-revision"></select>
        <p id="interactive-measurement"></p>
        <a id="interactive-download">下载当前掩膜</a>
      </div>
    </div>
    <p id="interactive-status" role="status"></p>
    <p id="interactive-error" class="error" role="alert" hidden></p>`;
  document.querySelector(".context-content").prepend(panel);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("interactive-marks"); svg.setAttribute("aria-hidden", "true"); shell.append(svg);
  let workspace = null, marks = [], selected = null, mode = "navigate";
  let binding = null, busy = false, job = null, polling = null, drag = null;
  let overlay = null, overlayViewer = null, pendingSubmit = null, generation = 0;
  let overlayController = null, lastContext = "";
  const current = () => {
    const c = bridge.context();
    return !!binding && c.ready && c.epoch === binding.epoch && c.source === binding.source;
  };
  const error = (value = "") => {
    $("interactive-error").textContent = value;
    $("interactive-error").hidden = !value;
  };
  const status = (text) => { $("interactive-status").textContent = text; };
  const endpoint = (suffix = "") => `/api/omni/workspaces/${workspace.id}${suffix}`;
  function removeOverlay() {
    if (overlay && overlayViewer?.volumes.includes(overlay)) overlayViewer.removeVolume(overlay);
    overlay = overlayViewer = null;
  }
  function setMode(next) {
    mode = next; drag = null;
    for (const button of panel.querySelectorAll("[data-mark]"))
      button.setAttribute("aria-pressed", String(button.dataset.mark === mode));
    canvas.style.cursor = mode === "navigate" ? "" : "crosshair";
  }
  function update() {
    const loaded = !!bridge.context().source && bridge.context().ready;
    $("interactive-open").disabled = !loaded || busy;
    $("interactive-editor").hidden = !workspace || !current();
    const enabled = !!workspace && current() && !busy;
    for (const element of $("interactive-editor").querySelectorAll("input, textarea, select, [data-mark]")) element.disabled = !enabled;
    $("interactive-workspaces").disabled = !bridge.context().ready || busy;
    $("interactive-count").textContent = `${marks.length} 个待应用标记`;
    $("interactive-undo").disabled = $("interactive-clear").disabled = !enabled || !marks.length;
    $("interactive-apply").disabled = !enabled || !marks.length;
    $("interactive-agent").disabled = !enabled || !$("interactive-instruction").value.trim();
    $("interactive-cancel").hidden = !job;
  }
  function reset() {
    generation += 1; clearTimeout(polling); overlayController?.abort(); removeOverlay();
    workspace = null; marks = []; selected = null; job = null; binding = null;
    busy = false; pendingSubmit = null; setMode("navigate"); svg.replaceChildren();
    status(""); error(); update();
  }
  async function loadWorkspaces() {
    const epoch = bridge.context().epoch;
    if (!bridge.context().ready) return;
    try {
      const recent = await api("/api/omni/workspaces");
      if (epoch !== bridge.context().epoch) return;
      const select = $("interactive-workspaces"); select.replaceChildren();
      const placeholder = document.createElement("option");
      placeholder.value = ""; placeholder.textContent = "选择区域工作区…"; select.append(placeholder);
      for (const item of recent) {
        const option = document.createElement("option"); option.value = item.id;
        option.textContent = `${item.latest_name} · ${item.revision_count} 版 · ${new Date(item.created_at * 1000).toLocaleString("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"})}`;
        select.append(option);
      }
      select.value = workspace?.id || "";
      $("interactive-recent").hidden = !recent.length;
    } catch (e) { if (epoch === bridge.context().epoch) error(e.message); }
  }
  $("interactive-workspaces").addEventListener("change", async () => {
    const id = $("interactive-workspaces").value;
    if (!id || busy) return;
    reset(); busy = true; update(); status("正在恢复区域工作区…");
    const token = generation, epoch = bridge.context().epoch;
    try {
      const saved = await api(`/api/omni/workspaces/${id}`);
      if (token !== generation || epoch !== bridge.context().epoch) return;
      if (!await bridge.showWorkspace(saved)) throw new Error("影像未能恢复，请重试");
      if (token !== generation || epoch !== bridge.context().epoch) return;
      binding = {epoch, source: saved.upload_id}; workspace = saved;
      busy = !!saved.active_job; renderHistory(saved.latest_revision); status("已恢复区域工作区。");
      if (saved.active_job) { job = {id: saved.active_job}; await poll(); }
    } catch (e) { if (token === generation) { busy = false; error(e.message); } }
    update();
  });
  function sameContext(token) { return token === generation && current(); }
  function selectRevision(id) {
    selected = id || null;
    const revision = workspace.revisions.find((r) => r.id === selected);
    $("interactive-revision").value = selected || "";
    if (!revision) { removeOverlay(); return; }
    $("interactive-name").value = revision.name;
    const parent = workspace.revisions.find((r) => r.id === revision.parent_revision);
    const amount = revision.volume_ml == null ? `${revision.voxel_count.toLocaleString()} 体素（空间单位未知）`
      : `${revision.volume_ml.toFixed(2)} mL`;
    const difference = parent?.volume_ml != null && revision.volume_ml != null
      ? ` · 较父版本 ${revision.volume_ml - parent.volume_ml >= 0 ? "+" : ""}${(revision.volume_ml - parent.volume_ml).toFixed(2)} mL` : "";
    $("interactive-measurement").textContent = amount + difference;
    $("interactive-download").href = endpoint(`/revisions/${selected}/file`);
    showOverlay(selected).catch((e) => { if (current() && e.name !== "AbortError") error(e.message); });
  }
  function renderHistory(preferred = selected) {
    const list = $("interactive-revision"); list.replaceChildren();
    workspace.revisions.forEach((r, i) => {
      const option = document.createElement("option"); option.value = r.id;
      option.textContent = `版本 ${i + 1} · ${r.name}`; list.append(option);
    });
    $("interactive-history").hidden = !workspace.revisions.length;
    selectRevision(preferred || workspace.latest_revision);
    $("interactive-hint").textContent = workspace.capabilities?.nninteractive?.configured
      ? "前景点包含目标，背景点排除误分；框选后点击应用。Esc 返回浏览。"
      : "标记功能已就绪；服务器尚未配置 nnInteractive 权重，暂不能推理。";
  }
  async function showOverlay(id) {
    const token = generation, viewer = bridge.context().viewer;
    overlayController?.abort();
    const controller = new AbortController(); overlayController = controller;
    const response = await fetch(endpoint(`/revisions/${id}/file`), {
      credentials: "same-origin", signal: controller.signal,
    });
    if (!response.ok) throw new Error("掩膜读取失败，请重试");
    const buffer = await response.arrayBuffer();
    if (!sameContext(token) || selected !== id || controller.signal.aborted) return;
    const image = await window.niivue.NVImage.loadFromUrl({url: buffer, name: "interactive-mask.nii.gz",
      colormap: "red", opacity: 0.55, cal_min: 0, cal_max: 1, colorbarVisible: false});
    if (!sameContext(token) || selected !== id || controller.signal.aborted || bridge.context().viewer !== viewer) return;
    removeOverlay(); overlay = image; overlayViewer = viewer; viewer.addVolume(image); viewer.drawScene();
  }
  async function refresh() {
    const token = generation, fresh = await api(endpoint());
    if (!sameContext(token)) return false;
    workspace = fresh; return true;
  }
  async function poll() {
    if (!job || !current()) return;
    const token = generation;
    try {
      const next = await api(`/api/omni/jobs/${job.id}`);
      if (!sameContext(token)) return;
      job = next;
      if (["completed", "failed", "canceled", "input_required"].includes(job.status)) {
        const finished = job, previousHead = workspace.latest_revision;
        if (!await refresh()) return;
        const producedRevision = workspace.latest_revision !== previousHead;
        busy = false; job = null; pendingSubmit = null;
        if (finished.status === "completed") {
          if (workspace.latest_revision !== selected) marks = [];
          renderHistory(workspace.latest_revision);
          status(finished.result?.summary || "区域任务完成，结果已保存为新版本。");
        } else {
          if (producedRevision) marks = [];
          renderHistory(producedRevision ? workspace.latest_revision : selected);
          status(producedRevision ? "后续步骤未完成；已生成的新版本已保留，标记未重复应用。"
            : finished.status === "input_required" ? finished.result?.summary || "请补充标记或说明。"
            : finished.status === "canceled" ? "任务已取消，标记仍保留。" : "任务失败，标记仍保留。");
          if (finished.error) error(finished.error.message);
        }
        update(); loadWorkspaces(); return;
      }
      status(job.status === "queued" ? "等待可用计算资源…" : "正在处理区域…");
    } catch (e) {
      if (!sameContext(token)) return;
      if ([400, 401, 403, 404, 410].includes(e.status)) {
        reset(); error("区域工作区已不可用，请重新打开当前影像。"); return;
      }
      error(`状态连接中断，正在重试：${e.message}`);
    }
    polling = setTimeout(poll, 1500);
  }
  async function submit(operation) {
    if (!current() || busy) return;
    const token = generation;
    const body = {operation, prompts: marks.map((m) => m.prompt), name: $("interactive-name").value.trim(),
      base_revision: selected, expected_revision: workspace.latest_revision};
    if (operation === "agent") body.instruction = $("interactive-instruction").value.trim();
    pendingSubmit = prepareSubmission(body, pendingSubmit, () => crypto.randomUUID());
    busy = true; setMode("navigate"); error(); update(); status("正在提交区域任务…");
    try {
      const submitted = await api(endpoint("/jobs"), {method: "POST", body: pendingSubmit.body});
      if (!sameContext(token)) return;
      job = submitted; update(); await poll();
    } catch (e) {
      if (!sameContext(token)) return;
      busy = false; error(e.message); status("提交未确认；重试会复用本次请求编号。");
      if (!e.status || e.status >= 500) { update(); return; }
      const refreshed = await refresh().catch(() => false);
      if (!sameContext(token)) return;
      if (e.status && e.status < 500) {
        pendingSubmit = null;
        if (e.status === 409 && refreshed) {
          selected = workspace.latest_revision;
          status("区域版本已更新，请核对当前结果和保留的标记后重新提交。");
        }
      }
      if (refreshed) renderHistory(); update();
    }
  }
  $("interactive-open").addEventListener("click", async () => {
    if (busy || !bridge.context().source) return;
    reset(); const c = bridge.context(); binding = {epoch: c.epoch, source: c.source};
    const token = generation; busy = true; update(); status("正在建立区域工作区…");
    try {
      const opened = await api("/api/omni/workspaces", {method: "POST", body: {upload_id: c.source}});
      if (!sameContext(token)) return;
      workspace = opened; busy = !!opened.active_job; renderHistory(); error();
      status("区域工作区已打开。"); panel.open = true;
      loadWorkspaces();
      if (opened.active_job) { job = {id: opened.active_job}; await poll(); }
    } catch (e) { if (sameContext(token)) { error(e.message); busy = false; status(""); } }
    update();
  });
  for (const button of panel.querySelectorAll("[data-mark]"))
    button.addEventListener("click", () => { setMode(button.dataset.mark); error(); });
  $("interactive-undo").addEventListener("click", () => { marks.pop(); pendingSubmit = null; update(); });
  $("interactive-clear").addEventListener("click", () => { marks = []; pendingSubmit = null; update(); });
  $("interactive-instruction").addEventListener("input", update);
  $("interactive-apply").addEventListener("click", () => submit("refine"));
  $("interactive-agent").addEventListener("click", () => submit("agent"));
  $("interactive-revision").addEventListener("change", () => {
    marks = []; pendingSubmit = null; status(""); error();
    selectRevision($("interactive-revision").value); update();
  });
  $("interactive-cancel").addEventListener("click", async () => {
    const token = generation;
    try {
      await api(`/api/omni/jobs/${job.id}/cancel`, {method: "POST", body: {}});
      if (sameContext(token)) { clearTimeout(polling); await poll(); }
    } catch (e) { if (sameContext(token)) error(e.message); }
  });
  function pick(point) {
    const viewer = bridge.context().viewer;
    const tile = viewer.tileIndex(...point), type = viewer.screenSlices[tile]?.axCorSag;
    if (![0, 1, 2].includes(type)) throw new Error("请在二维切片上标记");
    const frac = Array.from(viewer.canvasPos2frac(point)).slice(0, 3);
    if (frac.length !== 3 || frac.some((v) => !Number.isFinite(v) || v < 0 || v > 1))
      throw new Error("请在影像内部标记");
    const world = Array.from(viewer.frac2mm(frac, 0, true)).slice(0, 3);
    const voxel = worldToVoxel(workspace.geometry.affine, world);
    if (voxel.some((v, k) => !Number.isFinite(v) || Math.round(v) < 0 || Math.round(v) >= workspace.geometry.shape[k]))
      throw new Error("请在影像内部标记");
    return {frac, world, type, tile, point};
  }
  function intercept(event) {
    if (!current() || !workspace || busy || mode === "navigate" || event.button !== 0) return false;
    event.preventDefault(); event.stopImmediatePropagation(); return true;
  }
  canvas.addEventListener("pointerdown", (event) => {
    if (!intercept(event)) return;
    try {
      const start = pick(canvasPoint(event, canvas));
      if (marks.length >= 128) throw new Error("最多保留 128 个标记");
      drag = {start, end: start, mode}; canvas.setPointerCapture(event.pointerId); error();
    } catch (e) { drag = null; error(e.message); }
  }, true);
  canvas.addEventListener("pointermove", (event) => {
    if (!drag || !current()) return;
    event.preventDefault(); event.stopImmediatePropagation();
    try { drag.end = pick(canvasPoint(event, canvas)); } catch { drag.end = null; }
  }, true);
  canvas.addEventListener("pointerup", (event) => {
    if (!intercept(event)) return;
    if (!drag) return;
    try {
      const end = pick(canvasPoint(event, canvas)), start = drag.start;
      if (end.tile !== start.tile) throw new Error("请在同一个切片视图内完成标记");
      if (drag.mode === "box") {
        const a = start.point, b = end.point;
        const corners = [a, [b[0], a[1]], b, [a[0], b[1]]].map(pick);
        if (corners.some((p) => p.tile !== start.tile)) throw new Error("框选不能跨视图");
        const prompt = boxPrompt(corners.map((p) => p.world), workspace.geometry);
        marks.push({prompt, start, end});
      } else {
        marks.push({prompt: {kind: "point", world: end.world, positive: drag.mode === "positive"}, start: end, end});
      }
      pendingSubmit = null; error();
    } catch (e) { error(e.message); }
    drag = null; update();
  }, true);
  canvas.addEventListener("pointercancel", () => { drag = null; }, true);
  // NiiVue also listens for mouse events. Prevent the compatibility event stream
  // from moving its crosshair while this extension owns a left-button gesture.
  for (const type of ["mousedown", "mouseup", "click", "dblclick"])
    canvas.addEventListener(type, intercept, true);
  canvas.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { setMode("navigate"); event.preventDefault(); }
  }, true);
  panel.addEventListener("toggle", () => {
    if (!panel.open) setMode("navigate");
    else loadWorkspaces();
  });
  function draw() {
    if (binding && !current()) reset();
    const c = bridge.context(), signature = `${c.epoch}:${c.source}:${c.ready}`;
    if (signature !== lastContext) {
      if (!c.ready) {
        $("interactive-workspaces").replaceChildren(); $("interactive-recent").hidden = true;
      }
      lastContext = signature; update();
      if (panel.open && c.ready) loadWorkspaces();
    }
    svg.replaceChildren();
    const viewer = bridge.context().viewer;
    if (current() && viewer && panel.open) {
      const rect = canvas.getBoundingClientRect();
      svg.setAttribute("viewBox", `0 0 ${rect.width} ${rect.height}`);
      for (const mark of [...marks, ...(drag?.end ? [{start: drag.start, end: drag.end,
        prompt: {kind: drag.mode === "box" ? "box" : "point", positive: drag.mode !== "negative"}}] : [])]) {
        const axis = [2, 1, 0][mark.start.type];
        const tolerance = 0.51 / viewer.volumes[0].dimsRAS[axis + 1];
        if (Math.abs(mark.start.frac[axis] - viewer.scene.crosshairPos[axis]) > tolerance) continue;
        const a = viewer.frac2canvasPosWithTile(mark.start.frac, mark.start.type);
        const b = viewer.frac2canvasPosWithTile(mark.end.frac, mark.end.type);
        if (!a || !b || a.tileIndex !== b.tileIndex) continue;
        const x = a.pos[0] * rect.width / canvas.width, y = a.pos[1] * rect.height / canvas.height;
        const x2 = b.pos[0] * rect.width / canvas.width, y2 = b.pos[1] * rect.height / canvas.height;
        const element = document.createElementNS("http://www.w3.org/2000/svg", mark.prompt.kind === "box" ? "rect" : "circle");
        const attributes = mark.prompt.kind === "box"
          ? {x: Math.min(x, x2), y: Math.min(y, y2), width: Math.abs(x2 - x), height: Math.abs(y2 - y)}
          : {cx: x, cy: y, r: 5};
        for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
        element.setAttribute("stroke", mark.prompt.positive ? "#74f0a5" : "#ff7777");
        svg.append(element);
      }
    }
    requestAnimationFrame(draw);
  }
  update(); requestAnimationFrame(draw);
  return {reset};
}
