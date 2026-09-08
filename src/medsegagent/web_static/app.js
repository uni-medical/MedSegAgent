/* MedSegAgent browser client. No credentials or image contents are persisted here. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const terminal = new Set([
    "completed",
    "succeeded",
    "success",
    "failed",
    "canceled",
    "cancelled",
    "rejected",
  ]);
  const success = new Set(["completed", "succeeded", "success"]);
  const statusNames = {
    submitted: "等待处理",
    queued: "排队中",
    pending: "排队中",
    working: "正在分割",
    running: "正在分割",
    routing: "选择工具",
    validating: "校验影像",
    completed: "分割完成",
    succeeded: "分割完成",
    success: "分割完成",
    failed: "任务失败",
    canceled: "任务已取消",
    cancelled: "任务已取消",
    rejected: "请求被拒绝",
    canceling: "正在取消",
    cancelling: "正在取消",
  };
  const palette = [
    [238, 135, 77],
    [87, 182, 237],
    [194, 137, 239],
    [93, 199, 162],
    [236, 109, 145],
    [235, 197, 92],
    [113, 157, 239],
    [214, 178, 147],
  ];
  const state = {
    authenticated: false,
    epoch: 0,
    upload: null,
    tasks: [],
    selected: null,
    labels: [],
    visibleLabels: new Set(),
    maxUpload: 0,
    uploading: false,
    submitting: false,
    pendingRequest: null,
    poll: null,
    viewer: null,
    viewerInitializing: null,
    viewerKey: null,
    resultKey: null,
    viewerWanted: null,
    viewerQueue: Promise.resolve(),
    uploadXHR: null,
    activeRequests: new Set(),
    viewerController: null,
    viewMode: "multiplanar",
    restoring: false,
    viewerSave: null,
    defaultScene: null,
  };
  const statusOf = (task) =>
    typeof task.status === "string"
      ? task.status
      : task.status?.state || "queued";
  const size = (bytes) =>
    bytes >= 1073741824
      ? `${(bytes / 1073741824).toFixed(1)} GiB`
      : `${(bytes / 1048576).toFixed(1)} MiB`;
  const errorNames = {
    CANCELED: "任务已取消，可使用此影像重新提交。",
    SERVER_RESTART: "服务重启中断了此任务，可使用此影像重新提交。",
    TASK_TIMEOUT: "任务处理超时，请稍后重试或使用更小的影像。",
    INFERENCE_FAILED: "分割模型运行失败，请检查影像与分割目标后重试。",
    CAPACITY_EXCEEDED: "当前任务较多，请等待已有任务完成后再提交。",
    FILE_NOT_FOUND: "影像或结果已过期，请重新上传。",
    INVALID_FILE: "影像不是完整、有效的 3D NIfTI，请检查文件后重试。",
  };
  const errorMessage = (value) =>
    value?.code && errorNames[value.code]
      ? errorNames[value.code]
      : typeof value === "string"
        ? value
        : Array.isArray(value)
          ? value.map((x) => x.msg || "请求参数不正确").join("；")
          : value?.message ||
            value?.detail ||
            value?.code ||
            "请求未完成，请重试。";
  const showError = (id, value) => {
    $(id).textContent = value ? errorMessage(value) : "";
    $(id).hidden = !value;
  };
  const taskURL = (id) => `/api/tasks/${encodeURIComponent(id)}`;
  const uploadURL = (id) => `/api/uploads/${encodeURIComponent(id)}/file`;
  const updateSubmit = () => {
    $("submit").disabled =
      !state.upload ||
      !$("instruction").value.trim() ||
      state.uploading ||
      state.submitting ||
      !state.authenticated;
    $("submit").textContent = state.submitting ? "正在提交…" : "开始分割";
  };

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    state.activeRequests.add(controller);
    try {
      const response = await fetch(path, {
        credentials: "same-origin",
        cache: "no-store",
        ...options,
        signal: controller.signal,
        headers: {
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...options.headers,
        },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (response.status === 401 && path !== "/api/session") lockWorkspace();
        const err = new Error(
          errorMessage(
            data.detail ||
              data.error ||
              data.message ||
              `请求失败 (${response.status})`,
          ),
        );
        err.status = response.status;
        throw err;
      }
      return data;
    } catch (err) {
      if (err.name === "AbortError")
        throw new Error("连接超时。任务可能仍在运行，请刷新任务记录确认。");
      if (err instanceof TypeError)
        throw new Error("无法连接服务器，连接恢复后会继续读取任务。");
      throw err;
    } finally {
      clearTimeout(timeout);
      state.activeRequests.delete(controller);
    }
  }

  function removeVolumes() {
    const viewer = state.viewer;
    if (!viewer?.gl) return;
    for (const volume of [...viewer.volumes].reverse())
      viewer.removeVolume(volume);
    viewer.mediaUrlMap?.clear();
    viewer.drawScene();
  }

  function clearViewer() {
    removeVolumes();
    state.viewerKey = null;
    state.resultKey = null;
    $("canvas-shell").dataset.loaded = "false";
    $("viewer-empty").hidden = false;
    $("result-panel").hidden = true;
    $("result-empty").hidden = false;
    $("window-preset").disabled = true;
    $("niivue-canvas").style.visibility = "hidden";
    $("viewer-indicator").hidden = true;
    $("retry-viewer").hidden = true;
    state.labels = [];
    state.visibleLabels.clear();
  }

  function invalidateViewer() {
    saveView();
    state.viewerController?.abort();
    state.viewerWanted = null;
    $("niivue-canvas").style.visibility = "hidden";
    $("canvas-shell").dataset.loaded = "false";
    $("result-panel").hidden = true;
    $("result-empty").hidden = false;
    $("viewer-indicator").hidden = false;
    $("retry-viewer").hidden = true;
    showError("viewer-error", "");
  }

  function lockWorkspace() {
    state.viewerController?.abort();
    clearTimeout(state.viewerSave);
    try {
      for (let i = sessionStorage.length - 1; i >= 0; i--) {
        const key = sessionStorage.key(i);
        if (key.startsWith("medseg-view:")) sessionStorage.removeItem(key);
      }
    } catch {
      /* Storage may be disabled. */
    }
    state.authenticated = false;
    state.epoch += 1;
    clearTimeout(state.poll);
    for (const controller of state.activeRequests) controller.abort();
    state.uploadXHR?.abort();
    state.upload = null;
    state.selected = null;
    state.tasks = [];
    state.labels = [];
    state.visibleLabels.clear();
    state.viewerWanted = null;
    state.uploading = false;
    state.submitting = false;
    state.pendingRequest = null;
    $("workspace").hidden = true;
    $("logout").hidden = true;
    $("task-list").replaceChildren();
    $("labels").replaceChildren();
    $("request").reset();
    $("file-name").textContent = "选择或拖入 NIfTI";
    $("file-meta").hidden = true;
    $("upload-progress").hidden = true;
    $("task-status").hidden = true;
    $("viewer-name").textContent = "上传影像开始分割";
    $("reuse-image").hidden = true;
    $("drop-zone").classList.remove("has-file");
    clearViewer();
    updateSubmit();
    if (!$("login-dialog").open) $("login-dialog").showModal();
  }

  async function openWorkspace() {
    state.authenticated = true;
    state.epoch += 1;
    $("login-dialog").close();
    $("workspace").hidden = false;
    $("logout").hidden = false;
    showError("login-error", "");
    const config = await api("/api/config");
    state.maxUpload = Number(config.max_upload_bytes) || 0;
    if (!state.maxUpload)
      throw new Error("服务器未提供上传大小限制，请检查服务配置。");
    $("file-help").textContent =
      `.nii / .nii.gz · 最大 ${size(state.maxUpload)}`;
    await refreshTasks();
    const requested = new URL(location.href).searchParams.get("task");
    if (requested)
      await selectTask(requested).catch((err) =>
        showError("connection-note", err.message),
      );
    else if (state.tasks.length) await selectTask(state.tasks[0].id);
    schedulePoll();
  }

  function uploadFile(file) {
    if (!file || state.uploading || state.submitting) return;
    state.upload = null;
    state.pendingRequest = null;
    $("drop-zone").classList.remove("has-file");
    updateSubmit();
    $("file-name").textContent = file.name;
    $("file-meta").hidden = true;
    showError("form-error", "");
    if (!/\.nii(?:\.gz)?$/i.test(file.name)) {
      showError("form-error", "请选择 .nii 或 .nii.gz 格式的 3D 影像。");
      return;
    }
    if (!file.size) {
      showError("form-error", "文件为空，请选择完整的 NIfTI 影像。");
      return;
    }
    if (!state.maxUpload || file.size > state.maxUpload) {
      showError(
        "form-error",
        `文件超过上传限制（${size(state.maxUpload)}）。请使用更小的影像或通过本地 CLI 处理。`,
      );
      return;
    }
    const epoch = state.epoch;
    state.upload = null;
    state.uploading = true;
    state.pendingRequest = null;
    $("file-name").textContent = file.name;
    $("file-meta").hidden = true;
    $("upload-progress").hidden = false;
    $("upload-bar").value = 0;
    $("upload-status").textContent = "正在上传…";
    updateSubmit();
    const xhr = new XMLHttpRequest();
    state.uploadXHR = xhr;
    xhr.open("POST", "/api/uploads");
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    // ASCII-safe header; the server decodes percent-encoding before validating the name.
    xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    xhr.timeout = 900000;
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const n = Math.round((e.loaded / e.total) * 100);
        $("upload-bar").value = n;
        $("upload-status").textContent =
          n === 100 ? "上传完成，正在校验影像…" : `正在上传 ${n}%`;
      }
    };
    xhr.onload = () => {
      if (epoch !== state.epoch) return;
      let data;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        data = {};
      }
      if (xhr.status === 401) {
        lockWorkspace();
        return;
      }
      if (xhr.status < 200 || xhr.status >= 300 || !data.id) {
        showError(
          "form-error",
          data.detail || data.error || `上传失败 (${xhr.status})`,
        );
        $("upload-progress").hidden = true;
        return;
      }
      state.upload = data;
      $("drop-zone").classList.add("has-file");
      $("upload-status").textContent = "影像已校验";
      $("upload-bar").value = 100;
      $("file-meta").textContent = [
        size(data.size || file.size),
        data.shape?.join(" × "),
        data.spacing
          ? `${data.spacing.map((x) => Number(x).toFixed(2)).join(" × ")} mm`
          : "",
      ]
        .filter(Boolean)
        .join(" / ");
      $("file-meta").hidden = false;
      state.selected = null;
      $("task-status").hidden = true;
      $("reuse-image").hidden = true;
      renderHistory();
      const url = new URL(location.href);
      url.searchParams.delete("task");
      history.replaceState({}, "", url);
      $("viewer-name").textContent = data.name || file.name;
      queueViewer({
        key: `upload:${data.id}`,
        uploadID: data.id,
        name: data.name || file.name,
      });
    };
    xhr.onerror = () => {
      if (epoch === state.epoch) {
        showError("form-error", "上传连接中断，请重新选择影像。");
        $("upload-progress").hidden = true;
      }
    };
    xhr.ontimeout = () => {
      if (epoch === state.epoch) {
        showError("form-error", "上传超时，请检查网络后重试。");
        $("upload-progress").hidden = true;
      }
    };
    xhr.onloadend = () => {
      if (epoch === state.epoch) {
        state.uploading = false;
        state.uploadXHR = null;
        updateSubmit();
      }
    };
    xhr.send(file);
  }

  async function refreshTasks() {
    const epoch = state.epoch;
    const data = await api("/api/tasks");
    if (epoch !== state.epoch) return;
    state.tasks = Array.isArray(data) ? data : data.tasks || data.items || [];
    renderHistory();
  }

  function renderHistory() {
    $("task-list").replaceChildren();
    $("history-empty").hidden = state.tasks.length > 0;
    $("task-count").textContent = state.tasks.length || "";
    for (const task of state.tasks) {
      const li = document.createElement("li"),
        button = document.createElement("button");
      button.type = "button";
      button.className = "task-item";
      button.setAttribute(
        "aria-current",
        String(state.selected?.id === task.id),
      );
      const title = document.createElement("span");
      title.className = "task-item-title";
      title.textContent = task.text || task.input?.text || task.name || task.id;
      const meta = document.createElement("span");
      meta.className = "task-item-meta";
      const status = document.createElement("span");
      status.textContent =
        task.error?.code === "MODALITY_REQUIRED"
          ? "待补充说明"
          : statusNames[statusOf(task)] || statusOf(task);
      if (statusOf(task) === "failed") status.className = "failed";
      const date = document.createElement("span"),
        time = task.created_at || task.createdAt;
      if (time) {
        const d = new Date(
          typeof time === "number" && time < 1e12 ? time * 1000 : time,
        );
        date.textContent = Number.isNaN(d.valueOf())
          ? ""
          : d.toLocaleString("zh-CN", {
              month: "2-digit",
              day: "2-digit",
              hour: "2-digit",
              minute: "2-digit",
            });
      }
      meta.append(status, date);
      button.append(title, meta);
      button.addEventListener("click", () =>
        selectTask(task.id).catch((err) =>
          showError("connection-note", err.message),
        ),
      );
      li.append(button);
      $("task-list").append(li);
    }
  }

  async function selectTask(id) {
    const epoch = state.epoch;
    // Mark the requested ID before fetching so a slower previous request cannot replace it.
    if (state.selected?.id !== id) invalidateViewer();
    state.selected = { id };
    let task;
    try {
      task = await api(taskURL(id));
    } catch (err) {
      if (epoch === state.epoch && state.selected?.id === id) {
        $("viewer-indicator").hidden = true;
        showError("viewer-error", "无法读取任务，请刷新任务记录后重试。");
      }
      throw err;
    }
    if (epoch !== state.epoch || state.selected?.id !== id) return;
    state.selected = task;
    const url = new URL(location.href);
    url.searchParams.set("task", id);
    history.replaceState({}, "", url);
    renderTask(task);
    renderHistory();
  }

  function renderTask(task) {
    const status = statusOf(task),
      complete = success.has(status);
    $("task-status").hidden = false;
    $("task-status").dataset.status = status;
    $("status-title").textContent =
      task.error?.code === "MODALITY_REQUIRED"
        ? "需要补充说明"
        : statusNames[status] || status;
    const progress = task.progress;
    $("status-detail").textContent = complete
      ? [task.modality, "可查看叠加结果"].filter(Boolean).join(" · ")
      : {
          queued: "等待推理资源",
          routing: "正在理解分割请求",
          validating: "正在检查影像",
          working: "正在生成分割掩膜",
          failed: "可补充请求后重新提交",
          canceled: "可使用同一影像新建任务",
        }[status] || "";
    const percent = typeof progress === "number" ? progress : progress?.percent;
    $("task-progress").hidden = terminal.has(status);
    if (Number.isFinite(percent))
      $("task-progress").value = Math.max(0, Math.min(100, percent));
    else $("task-progress").removeAttribute("value");
    $("cancel-task").hidden = terminal.has(status);
    $("cancel-task").disabled = ["canceling", "cancelling"].includes(status);
    showError(
      "task-error",
      task.error ||
        (status === "failed"
          ? "任务未完成，请查看错误信息后新建任务重试。"
          : ""),
    );
    $("viewer-name").textContent =
      task.upload_name || task.input?.name || task.text || `任务 ${task.id}`;
    const uploadID = task.upload_id || task.input?.upload_id;
    $("reuse-image").hidden = !uploadID || task.files_expired;
    if (uploadID)
      queueViewer({
        key: `${task.id}:${complete ? "result" : "source"}`,
        taskID: task.id,
        uploadID,
        name: task.upload_name || "source.nii",
        result: complete ? task.result : null,
        completed: complete,
      });
    else if (state.viewerWanted?.taskID !== task.id) {
      state.viewerWanted = null;
      clearViewer();
      showError("viewer-error", "此任务没有可通过 Web 读取的源影像。");
    }
  }

  function schedulePoll() {
    clearTimeout(state.poll);
    if (!state.authenticated) return;
    state.poll = setTimeout(
      async () => {
        const epoch = state.epoch;
        try {
          const selectedID = state.selected?.id;
          if (selectedID) {
            const task = await api(taskURL(selectedID));
            if (epoch === state.epoch && state.selected?.id === selectedID) {
              state.selected = task;
              renderTask(task);
            }
          }
          if (epoch === state.epoch) await refreshTasks();
          if (epoch === state.epoch) showError("connection-note", "");
        } catch (err) {
          if (state.authenticated) showError("connection-note", err.message);
        } finally {
          if (epoch === state.epoch) schedulePoll();
        }
      },
      document.hidden
        ? 10000
        : state.selected && !terminal.has(statusOf(state.selected))
          ? 2000
          : 6000,
    );
  }

  async function getViewer() {
    if (state.viewer) return state.viewer;
    if (state.viewerInitializing) return state.viewerInitializing;
    state.viewerInitializing = initializeViewer();
    try {
      return await state.viewerInitializing;
    } finally {
      state.viewerInitializing = null;
    }
  }

  async function initializeViewer() {
    if (!window.niivue?.Niivue)
      throw new Error("查看器脚本未能加载，请刷新页面。");
    const viewer = new window.niivue.Niivue({
      backColor: [0, 0, 0, 1],
      crosshairColor: [0.55, 0.75, 0.85, 0.6],
      crosshairWidth: 0.6,
      isColorbar: false,
      isOrientCube: true,
      show3Dcrosshair: false,
      isRadiologicalConvention: true,
      dragAndDropEnabled: false,
      isNearestInterpolation: true,
      fontMinPx: 10,
      fontSizeScaling: 0.25,
      multiplanarShowRender: window.niivue.SHOW_RENDER.ALWAYS,
      logging: false,
    });
    await viewer.attachToCanvas($("niivue-canvas"));
    if (!viewer.gl)
      throw new Error("当前浏览器无法初始化 WebGL2，请启用硬件加速。");
    const D = window.niivue.DRAG_MODE;
    viewer.setMouseEventConfig({
      leftButton: { primary: D.crosshair, withCtrl: D.crosshair },
      rightButton: D.windowing,
      centerButton: D.pan,
    });
    state.viewer = viewer;
    state.defaultScene = {
      azimuth: viewer.scene.renderAzimuth,
      elevation: viewer.scene.renderElevation,
    };
    applyViewMode(state.viewMode);
    viewer.onIntensityChange = () => {
      if (!state.restoring) {
        $("window-preset").value = "custom";
        scheduleSaveView();
      }
    };
    viewer.onLocationChange = (location) => {
      if (location.mm)
        $("location").textContent = `位置 ${Array.from(location.mm)
          .slice(0, 3)
          .map((x) => Number(x).toFixed(1))
          .join(", ")} mm`;
      updateSlices();
      scheduleSaveView();
    };
    return viewer;
  }

  function applyViewMode(mode) {
    const viewer = state.viewer,
      S = window.niivue?.SLICE_TYPE;
    if (!viewer || !S) return;
    const types = {
      multiplanar: "MULTIPLANAR",
      axial: "AXIAL",
      coronal: "CORONAL",
      sagittal: "SAGITTAL",
      render: "RENDER",
    };
    state.viewMode = Object.hasOwn(types, mode) ? mode : "multiplanar";
    viewer.clearCustomLayout();
    viewer.setSliceType(S[types[state.viewMode]]);
    if (state.viewMode === "multiplanar")
      viewer.setCustomLayout([
        { sliceType: S.AXIAL, position: [0, 0, 0.5, 0.5] },
        { sliceType: S.SAGITTAL, position: [0.5, 0, 0.5, 0.5] },
        { sliceType: S.CORONAL, position: [0, 0.5, 0.5, 0.5] },
        { sliceType: S.RENDER, position: [0.5, 0.5, 0.5, 0.5] },
      ]);
    $("canvas-shell").dataset.view = state.viewMode;
    document.querySelectorAll("[data-view]").forEach((button) => {
      if (button.tagName === "BUTTON")
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.view === state.viewMode),
        );
    });
    scheduleSaveView();
  }

  function updateSlices() {
    const viewer = state.viewer,
      dims = viewer?.volumes[0]?.dimsRAS;
    if (!dims) return;
    for (const [name, axis] of [
      ["axial", 2],
      ["sagittal", 0],
      ["coronal", 1],
    ]) {
      const n = dims[axis + 1],
        k = Math.max(
          0,
          Math.min(
            n - 1,
            Math.round(viewer.scene.crosshairPos[axis] * n - 0.5),
          ),
        );
      $("slice-" + name).max = n - 1;
      $("slice-" + name).value = k;
      $("slice-" + name + "-value").value = `${k + 1} / ${n}`;
    }
  }

  const viewStorageKey = (key) =>
    "medseg-view:" + key?.replace(/:(source|result)$/, "");
  function scheduleSaveView() {
    clearTimeout(state.viewerSave);
    if (!state.restoring) state.viewerSave = setTimeout(saveView, 150);
  }
  function saveView() {
    const v = state.viewer;
    if (!state.viewerKey || !v?.volumes[0] || state.restoring) return;
    try {
      sessionStorage.setItem(
        viewStorageKey(state.viewerKey),
        JSON.stringify({
          mode: state.viewMode,
          frac: Array.from(v.scene.crosshairPos),
          pan: Array.from(v.scene.pan2Dxyzmm),
          az: v.scene.renderAzimuth,
          el: v.scene.renderElevation,
          scale: v.scene.volScaleMultiplier,
          window: [
            $("window-preset").value,
            v.volumes[0].cal_min,
            v.volumes[0].cal_max,
          ],
          crosshair: $("crosshair-toggle").checked,
          opacity: Number($("opacity").value),
          labels:
            state.resultKey === state.viewerKey
              ? [...state.visibleLabels]
              : null,
        }),
      );
      const keys = Object.keys(sessionStorage).filter((k) =>
        k.startsWith("medseg-view:"),
      );
      for (const key of keys.slice(0, Math.max(0, keys.length - 24)))
        sessionStorage.removeItem(key);
    } catch {
      /* Viewer preferences are optional in restricted browser storage. */
    }
  }
  function restoreView(key) {
    const v = state.viewer;
    state.restoring = true;
    let saved = null;
    try {
      saved = JSON.parse(sessionStorage.getItem(viewStorageKey(key)) || "null");
    } catch {
      /* Always apply visible defaults without storage. */
    }
    try {
      v.scene.crosshairPos = [0.5, 0.5, 0.5];
      v.setPan2Dxyzmm([0, 0, 0, 1]);
      v.setScale(1);
      v.setRenderAzimuthElevation(
        state.defaultScene.azimuth,
        state.defaultScene.elevation,
      );
      $("window-preset").value = "auto";
      $("opacity").value = "55";
      $("opacity-value").value = "55%";
      $("crosshair-toggle").checked = true;
      v.opts.crosshairWidth = 0.6;
      applyViewMode(saved?.mode || "multiplanar");
      focusForeground(v);
      if (saved) {
        if (
          Array.isArray(saved.frac) &&
          saved.frac.length === 3 &&
          saved.frac.every((n) => Number.isFinite(n) && n >= 0 && n <= 1)
        )
          v.scene.crosshairPos = saved.frac;
        if (
          Array.isArray(saved.pan) &&
          saved.pan.length === 4 &&
          saved.pan.every(Number.isFinite) &&
          saved.pan[3] > 0
        )
          v.setPan2Dxyzmm(saved.pan);
        if (Number.isFinite(saved.az) && Number.isFinite(saved.el))
          v.setRenderAzimuthElevation(saved.az, saved.el);
        if (Number.isFinite(saved.scale) && saved.scale > 0)
          v.setScale(saved.scale);
        if (
          Array.isArray(saved.window) &&
          saved.window.length === 3 &&
          saved.window.slice(1).every(Number.isFinite) &&
          saved.window[2] > saved.window[1]
        ) {
          v.volumes[0].cal_min = saved.window[1];
          v.volumes[0].cal_max = saved.window[2];
          $("window-preset").value = saved.window[0];
        }
        $("crosshair-toggle").checked = saved.crosshair !== false;
        v.opts.crosshairWidth = saved.crosshair === false ? 0 : 0.6;
        if (Number.isFinite(saved.opacity))
          $("opacity").value = Math.max(0, Math.min(100, saved.opacity));
        $("opacity-value").value = $("opacity").value + "%";
        if (Array.isArray(saved.labels))
          state.visibleLabels = new Set(
            saved.labels.filter((id) =>
              state.labels.some((label) => label.id === id),
            ),
          );
      }
      $("labels")
        .querySelectorAll("input")
        .forEach((input) => {
          input.checked = state.visibleLabels.has(Number(input.dataset.label));
        });
      updateOverlay();
      if (!v.volumes[1]) v.updateGLVolume();
      updateSlices();
      v.createOnLocationChange();
    } catch {
      /* A stale preference must not block a valid image. */
    } finally {
      state.restoring = false;
    }
  }

  async function imageBuffer(url, controller) {
    const response = await fetch(url, {
      credentials: "same-origin",
      cache: "no-store",
      signal: controller.signal,
    });
    if (response.status === 401) {
      lockWorkspace();
      throw new Error("登录已过期，请重新登录。");
    }
    if (!response.ok)
      throw new Error(
        response.status === 404 || response.status === 410
          ? "影像文件已过期或不可访问，请重新上传。"
          : `影像读取失败 (${response.status})`,
      );
    return response.arrayBuffer();
  }

  function queueViewer(request, force = false) {
    if (!force && state.viewerWanted?.key === request.key) return;
    saveView();
    state.viewerController?.abort();
    state.viewerWanted = request;
    const epoch = state.epoch;
    $("result-panel").hidden = true;
    $("result-empty").hidden = false;
    $("viewer-indicator").hidden = false;
    $("retry-viewer").hidden = true;
    showError("viewer-error", "");
    $("niivue-canvas").style.visibility = "hidden";
    $("canvas-shell").dataset.loaded = "false";
    state.viewerQueue = state.viewerQueue
      .catch(() => {})
      .then(async () => {
        if (epoch !== state.epoch || state.viewerWanted !== request) return;
        const controller = new AbortController();
        state.viewerController = controller;
        const timeout = setTimeout(() => controller.abort("timeout"), 180000);
        const current = () =>
          epoch === state.epoch && state.viewerWanted === request;
        try {
          state.viewerKey = null;
          state.resultKey = null;
          state.labels = [];
          state.visibleLabels.clear();
          removeVolumes();
          if (request.completed) {
            const result =
              request.result ||
              (await api(`${taskURL(request.taskID)}/files/result.json`));
            if (!current()) return;
            showResult(result, request.taskID, false);
            state.resultKey = request.key;
          }
          const viewer = await getViewer();
          if (!current()) return;
          const source = await imageBuffer(
            uploadURL(request.uploadID),
            controller,
          );
          const background = await window.niivue.NVImage.loadFromUrl({
            url: source,
            name: request.name || "source.nii",
            colormap: "gray",
            opacity: 1,
            colorbarVisible: false,
          });
          if (!current()) return;
          viewer.addVolume(background);
          if (request.completed) {
            const mask = await imageBuffer(
              `${taskURL(request.taskID)}/files/segmentation.nii.gz`,
              controller,
            );
            const overlay = await window.niivue.NVImage.loadFromUrl({
              url: mask,
              name: "segmentation.nii.gz",
              colormap: "gray",
              opacity: 0,
              cal_min: 0,
              cal_max: Math.max(1, ...state.labels.map((l) => l.id)),
              colorbarVisible: false,
            });
            if (!current()) return;
            viewer.addVolume(overlay);
          }
          if (!current()) return;
          restoreView(request.key);
          state.viewerKey = request.key;
          $("viewer-empty").hidden = true;
          $("canvas-shell").dataset.loaded = "true";
          $("niivue-canvas").style.visibility = "visible";
          $("window-preset").disabled = false;
          $("canvas-shell").dataset.source = request.uploadID;
          $("canvas-shell").dataset.task = request.taskID || "";
          viewer.resizeListener();
          viewer.drawScene();
          saveView();
        } catch (err) {
          if (current()) {
            removeVolumes();
            state.viewerKey = null;
            showError(
              "viewer-error",
              controller.signal.reason === "timeout"
                ? "载入超时，请检查网络后重试。"
                : `影像显示失败：${err.message}`,
            );
            $("retry-viewer").hidden = false;
          }
        } finally {
          clearTimeout(timeout);
          if (current()) {
            $("viewer-indicator").hidden = true;
            state.viewerController = null;
          }
        }
      });
  }

  function showResult(result, taskID, apply = true) {
    state.labels = (result.labels || [])
      .filter(
        (label) =>
          Number.isInteger(Number(label.id)) &&
          Number(label.id) > 0 &&
          Number(label.id) <= 65535,
      )
      .map((label) => ({ ...label, id: Number(label.id) }));
    state.visibleLabels = new Set(state.labels.map((label) => label.id));
    $("labels").replaceChildren();
    $("label-count").textContent = `(${state.labels.length})`;
    for (const [index, label] of state.labels.entries()) {
      const wrap = document.createElement("label");
      wrap.className = "label-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = true;
      checkbox.dataset.label = String(label.id);
      checkbox.addEventListener("change", () => {
        checkbox.checked
          ? state.visibleLabels.add(label.id)
          : state.visibleLabels.delete(label.id);
        updateOverlay();
        scheduleSaveView();
      });
      const swatch = document.createElement("span");
      swatch.className = "label-swatch";
      swatch.style.backgroundColor = `rgb(${palette[index % palette.length].join(",")})`;
      const name = document.createElement("span");
      name.className = "label-name";
      name.textContent =
        {
          liver: "肝脏",
          kidney_left: "左肾",
          kidney_right: "右肾",
          spleen: "脾脏",
          pancreas: "胰腺",
          lung_nodules: "肺结节",
          liver_lesions: "肝病灶",
          aorta: "主动脉",
          gallbladder: "胆囊",
          stomach: "胃",
        }[label.name] ||
        label.name ||
        String(label.id);
      name.title = label.name || "";
      const count = document.createElement("span");
      count.className = "label-voxels";
      count.textContent = Number.isFinite(label.voxels)
        ? `${label.voxels.toLocaleString()} 体素`
        : "";
      wrap.append(checkbox, swatch, name, count);
      $("labels").append(wrap);
    }
    $("download-mask").href = `${taskURL(taskID)}/files/segmentation.nii.gz`;
    $("download-result").href = `${taskURL(taskID)}/files/result.json`;
    const elapsed =
      result.duration_seconds ??
      result.elapsed_seconds ??
      result.runtime_seconds;
    $("result-summary").textContent = [
      Number.isFinite(elapsed) ? `推理用时 ${elapsed.toFixed(1)} 秒` : "",
      result.detection_status === "no_target_detected"
        ? "未检出目标，不能据此排除病变"
        : "请核查分割边界与标签",
    ]
      .filter(Boolean)
      .join(" / ");
    $("result-panel").hidden = false;
    $("result-empty").hidden = true;
    if (apply) updateOverlay();
  }

  function updateOverlay() {
    const viewer = state.viewer,
      overlay = viewer?.volumes[1];
    if (!overlay) return;
    const map = {
      I: [0],
      R: [0],
      G: [0],
      B: [0],
      A: [0],
      labels: ["background"],
    };
    for (const [index, label] of state.labels.entries()) {
      const color = palette[index % palette.length];
      map.I.push(label.id);
      map.R.push(color[0]);
      map.G.push(color[1]);
      map.B.push(color[2]);
      map.A.push(state.visibleLabels.has(label.id) ? 255 : 0);
      map.labels.push(label.name || String(label.id));
    }
    // Discrete label lookup preserves IDs; visibility never modifies voxel data.
    overlay.setColormapLabel(map);
    viewer.setOpacity(1, Number($("opacity").value) / 100);
  }

  function focusForeground(viewer) {
    const overlay = viewer.volumes[1],
      image = overlay?.img,
      affine = overlay?.hdr?.affine;
    if (!image || !affine?.[0] || !state.labels.length) return;
    const label = [...state.labels].sort(
      (a, b) => (b.voxels || 0) - (a.voxels || 0),
    )[0].id;
    const nx = overlay.hdr.dims[1],
      ny = overlay.hdr.dims[2],
      stride = Math.max(1, Math.ceil(image.length / 1000000));
    let count = 0,
      x = 0,
      y = 0,
      z = 0;
    for (let i = 0; i < image.length; i += stride)
      if (image[i] === label) {
        count++;
        x += i % nx;
        y += Math.floor(i / nx) % ny;
        z += Math.floor(i / (nx * ny));
      }
    if (!count) return;
    const p = [x / count, y / count, z / count, 1];
    const mm = affine
      .slice(0, 3)
      .map((row) => row.reduce((sum, value, i) => sum + value * p[i], 0));
    viewer.scene.crosshairPos = viewer.mm2frac(mm);
    viewer.drawScene();
  }

  $("login-dialog").addEventListener("cancel", (event) =>
    event.preventDefault(),
  );
  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("login-submit").disabled = true;
    showError("login-error", "");
    try {
      await api("/api/session", {
        method: "POST",
        body: JSON.stringify({ token: $("access-token").value.trim() }),
      });
      $("access-token").value = "";
      await openWorkspace();
    } catch (err) {
      lockWorkspace();
      showError("login-error", err.message);
    } finally {
      $("login-submit").disabled = false;
    }
  });
  $("logout").addEventListener("click", async () => {
    try {
      await api("/api/session", { method: "DELETE" });
      const url = new URL(location.href);
      url.searchParams.delete("task");
      history.replaceState({}, "", url);
      lockWorkspace();
    } catch (err) {
      showError("connection-note", err.message);
    }
  });
  $("file").addEventListener("change", (event) =>
    uploadFile(event.target.files[0]),
  );
  for (const name of ["dragenter", "dragover"])
    $("drop-zone").addEventListener(name, (event) => {
      event.preventDefault();
      $("drop-zone").classList.add("dragging");
    });
  for (const name of ["dragleave", "drop"])
    $("drop-zone").addEventListener(name, (event) => {
      event.preventDefault();
      $("drop-zone").classList.remove("dragging");
    });
  $("drop-zone").addEventListener("drop", (event) => {
    if (event.dataTransfer.files.length !== 1)
      showError("form-error", "请一次上传一张 3D 影像。");
    else uploadFile(event.dataTransfer.files[0]);
  });
  $("request").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.upload || state.uploading || state.submitting) return;
    const text = $("instruction").value.trim();
    if (!text) {
      showError("form-error", "请填写需要分割的目标。");
      return;
    }
    const signature = JSON.stringify([state.upload.id, text]);
    if (state.pendingRequest?.signature !== signature)
      state.pendingRequest = { signature, id: crypto.randomUUID() };
    state.submitting = true;
    updateSubmit();
    showError("form-error", "");
    try {
      const task = await api("/api/tasks", {
        method: "POST",
        body: JSON.stringify({
          upload_id: state.upload.id,
          text,
          message_id: state.pendingRequest.id,
        }),
      });
      state.pendingRequest = null;
      await refreshTasks();
      await selectTask(task.id);
      schedulePoll();
    } catch (err) {
      showError("form-error", err.message);
    } finally {
      state.submitting = false;
      updateSubmit();
    }
  });
  $("refresh-tasks").addEventListener("click", async () => {
    try {
      await refreshTasks();
      if (state.selected?.id) await selectTask(state.selected.id);
      showError("connection-note", "");
    } catch (err) {
      showError("connection-note", err.message);
    }
  });
  $("cancel-task").addEventListener("click", async () => {
    const id = state.selected?.id;
    if (!id) return;
    $("cancel-task").disabled = true;
    try {
      await api(`${taskURL(id)}/cancel`, { method: "POST" });
      if (state.selected?.id === id) await selectTask(id);
      await refreshTasks();
    } catch (err) {
      showError("task-error", err.message);
    } finally {
      $("cancel-task").disabled = false;
    }
  });
  document.querySelectorAll("button[data-view]").forEach((button) =>
    button.addEventListener("click", async () => {
      try {
        await getViewer();
        applyViewMode(button.dataset.view);
      } catch (err) {
        showError("viewer-error", err.message);
      }
    }),
  );
  $("reset-view").addEventListener("click", () => {
    const v = state.viewer;
    if (!v?.volumes.length || !state.viewerKey) return;
    v.scene.crosshairPos = [0.5, 0.5, 0.5];
    v.setPan2Dxyzmm([0, 0, 0, 1]);
    v.setScale(1);
    v.setRenderAzimuthElevation(
      state.defaultScene.azimuth,
      state.defaultScene.elevation,
    );
    v.volumes[0].cal_min = v.volumes[0].robust_min;
    v.volumes[0].cal_max = v.volumes[0].robust_max;
    $("window-preset").value = "auto";
    focusForeground(v);
    v.updateGLVolume();
    v.createOnLocationChange();
    updateSlices();
    saveView();
  });
  for (const [id, factor] of [
    ["zoom-in", 1.2],
    ["zoom-out", 1 / 1.2],
  ])
    $(id).addEventListener("click", () => {
      const v = state.viewer;
      if (!v?.volumes.length || !state.viewerKey) return;
      const p = Array.from(v.scene.pan2Dxyzmm);
      p[3] = Math.max(0.25, Math.min(8, p[3] * factor));
      v.setPan2Dxyzmm(p);
      v.setScale(
        Math.max(0.25, Math.min(8, v.scene.volScaleMultiplier * factor)),
      );
      v.drawScene();
      saveView();
    });
  $("window-preset").addEventListener("change", () => {
    const v = state.viewer,
      src = v?.volumes[0];
    if (!src) return;
    const values = {
      soft: [-160, 240],
      lung: [-1350, 150],
      bone: [-500, 1500],
    };
    [src.cal_min, src.cal_max] = values[$("window-preset").value] || [
      src.robust_min,
      src.robust_max,
    ];
    v.updateGLVolume();
    saveView();
  });
  $("crosshair-toggle").addEventListener("change", () => {
    if (state.viewer) {
      state.viewer.opts.crosshairWidth = $("crosshair-toggle").checked
        ? 0.6
        : 0;
      state.viewer.drawScene();
      saveView();
    }
  });
  $("opacity").addEventListener("input", () => {
    $("opacity-value").value = `${$("opacity").value}%`;
    if (state.viewer?.volumes[1])
      state.viewer.setOpacity(1, Number($("opacity").value) / 100);
    saveView();
  });
  for (const [id, visible] of [
    ["labels-all", true],
    ["labels-none", false],
  ])
    $(id).addEventListener("click", () => {
      state.visibleLabels = new Set(
        visible ? state.labels.map((label) => label.id) : [],
      );
      $("labels")
        .querySelectorAll("input")
        .forEach((input) => {
          input.checked = visible;
        });
      updateOverlay();
      saveView();
    });
  for (const [name, axis] of [
    ["axial", 2],
    ["sagittal", 0],
    ["coronal", 1],
  ])
    $("slice-" + name).addEventListener("input", () => {
      const v = state.viewer,
        n = v?.volumes[0]?.dimsRAS?.[axis + 1];
      if (!n || !state.viewerKey) return;
      v.scene.crosshairPos[axis] = (Number($("slice-" + name).value) + 0.5) / n;
      v.createOnLocationChange();
      v.drawScene();
      updateSlices();
      scheduleSaveView();
    });
  $("retry-viewer").addEventListener("click", () => {
    if (state.viewerWanted) queueViewer({ ...state.viewerWanted }, true);
  });
  $("instruction").addEventListener("input", () => {
    state.pendingRequest = null;
    updateSubmit();
  });
  function useSelectedImage() {
    const task = state.selected;
    if (!task?.upload_id) return;
    state.upload = {
      id: task.upload_id,
      name: task.upload_name || "当前任务影像",
    };
    state.pendingRequest = null;
    $("file-name").textContent = state.upload.name;
    $("drop-zone").classList.add("has-file");
    $("file-meta").hidden = true;
    $("upload-progress").hidden = true;
    $("instruction").value = task.text || "";
    showError("form-error", "");
    updateSubmit();
    $("instruction").focus();
    $("request").scrollIntoView({ block: "nearest" });
  }
  $("reuse-image").addEventListener("click", useSelectedImage);
  $("new-task").addEventListener("click", () => {
    if (state.uploading || state.submitting) return;
    invalidateViewer();
    clearViewer();
    state.upload = null;
    state.selected = null;
    state.pendingRequest = null;
    $("request").reset();
    $("file-name").textContent = "选择或拖入 NIfTI";
    $("drop-zone").classList.remove("has-file");
    $("file-meta").hidden = true;
    $("upload-progress").hidden = true;
    $("task-status").hidden = true;
    $("reuse-image").hidden = true;
    $("viewer-name").textContent = "上传影像开始分割";
    const url = new URL(location.href);
    url.searchParams.delete("task");
    history.replaceState({}, "", url);
    showError("form-error", "");
    renderHistory();
    updateSubmit();
  });
  $("niivue-canvas").addEventListener("pointerup", scheduleSaveView);
  $("niivue-canvas").addEventListener("wheel", scheduleSaveView, {
    passive: true,
  });
  $("niivue-canvas").addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    showError(
      "viewer-error",
      "浏览器显存不足或图形上下文已丢失。任务仍保存在服务器，请刷新页面或使用更小的影像。",
    );
  });
  window.addEventListener("pagehide", saveView);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) schedulePoll();
    else saveView();
  });
  window.addEventListener("online", () => {
    if (state.authenticated) schedulePoll();
  });
  api("/api/session")
    .then((session) => {
      if (session.authenticated === false) lockWorkspace();
      else return openWorkspace();
    })
    .catch((err) => {
      lockWorkspace();
      if (err.status !== 401) showError("login-error", err.message);
    });
})();
