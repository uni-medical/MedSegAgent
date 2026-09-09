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
    input_required: "等待补充信息",
    working: "正在分割",
    running: "正在分割",
    routing: "理解需求",
    validating: "校验影像",
    normalizing: "整理分割掩膜",
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
    [255, 0, 0],
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
    ready: false,
    identity: null,
    epoch: 0,
    upload: null,
    tasks: [],
    selected: null,
    outputSelection: new Map(),
    labels: [],
    visibleLabels: new Set(),
    maxUpload: 0,
    singleUpload: 0,
    uploadChunkBytes: 0,
    uploading: false,
    submitting: false,
    pendingRequest: null,
    poll: null,
    historyPoll: null,
    historyRequest: null,
    taskTick: null,
    taskClock: null,
    viewer: null,
    viewerInitializing: null,
    viewerKey: null,
    sourceImage: null,
    resultKey: null,
    viewerWanted: null,
    viewerQueue: Promise.resolve(),
    uploadXHR: null,
    chunkUpload: null,
    activeRequests: new Set(),
    viewerController: null,
    viewMode: "multiplanar",
    restoring: false,
    viewerSave: null,
    defaultScene: null,
    renderedRecord: null,
    windowInvalid: false,
    examples: [],
    exampleButtons: [],
    exampleRequest: null,
  };
  const statusOf = (task) =>
    typeof task.status === "string"
      ? task.status
      : task.status?.state || "queued";
  const isWorkingTask = (task) =>
    !terminal.has(statusOf(task)) && statusOf(task) !== "input_required";
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
    UNSUPPORTED_REQUEST: "当前不支持这项分割需求，未开始分割。",
    MODALITY_REQUIRED: "请在请求中说明影像是 CT 还是 MR。",
    MODALITY_CONFLICT: "请求中的 CT / MR 与所选影像不一致，请修改后重试。",
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
      !!state.exampleRequest ||
      !state.ready;
    $("submit").textContent = state.submitting ? "正在提交…" : "开始分割";
    for (const id of [
      "choose-image",
      "file",
      "refresh-tasks",
      "instruction",
      "new-task",
    ])
      $(id).disabled = !state.ready;
    updateExampleButtons();
  };

  function setContext(name) {
    for (const key of ["request", "results"]) {
      const active = key === name;
      $("tab-" + key).setAttribute("aria-selected", String(active));
      $("tab-" + key).setAttribute("tabindex", active ? "0" : "-1");
      $("panel-" + key).hidden = !active;
    }
  }

  function formatDate(value) {
    if (!value) return "";
    const date = new Date(
      typeof value === "number" && value < 1e12 ? value * 1000 : value,
    );
    return Number.isNaN(date.valueOf())
      ? ""
      : date.toLocaleString("zh-CN", {
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
        });
  }

  function renderDownloads(input, task = null) {
    const inputOK =
      !!input?.id &&
      input.available !== false &&
      task?.input_available !== false;
    const output = task ? selectedOutput(task) : null;
    const availableFiles = output?.files || [];
    const hasClassFiles = availableFiles.some((file) => file?.kind === "label");
    const files = availableFiles
      .filter(
        (file) =>
          file &&
          (!hasClassFiles || file.kind === "label") &&
          /^[a-z0-9_.-]+\.nii(?:\.gz)?$/i.test(file.name) &&
          (!output.legacy ||
            (file.kind === "label" &&
              Number.isInteger(file.label_id) &&
              file.label_id > 0 &&
              file.name.startsWith(`${file.label_id}_`))) &&
          sameOriginFile(
            file.url,
            `${taskURL(task.id)}/files/${encodeURIComponent(file.name)}`,
          ),
      )
      .sort((a, b) => (a.label_id || 0) - (b.label_id || 0));
    $("downloads").hidden = !inputOK && !files.length;
    $("download-source").hidden = !inputOK;
    $("download-source").removeAttribute("href");
    if (inputOK) $("download-source").href = uploadURL(input.id);
    $("download-labels").replaceChildren();
    $("download-labels").hidden = !files.length;
    for (const file of files) {
      const link = document.createElement("a");
      link.className = "file-download";
      link.href = `${taskURL(task.id)}/files/${encodeURIComponent(file.name)}`;
      link.setAttribute("download", file.name);
      link.title = file.name;
      const name = document.createElement("span");
      name.textContent = file.label_id
        ? `${file.label_id}. ${labelDisplayName(file.label_name, file.label_id)}`
        : file.display_name || output.name;
      const format = document.createElement("span");
      format.textContent = "NIfTI";
      link.append(name, format);
      $("download-labels").append(link);
    }
  }

  function sameOriginFile(value, expectedPath) {
    try {
      const url = new URL(value, location.href);
      return (
        url.origin === new URL(location.href).origin &&
        url.pathname === expectedPath &&
        !url.search &&
        !url.hash &&
        !url.username &&
        !url.password
      );
    } catch {
      return false;
    }
  }

  function resultOutputs(task) {
    if (!task?.result || task.result_available === false) return [];
    if (!Array.isArray(task.result.outputs))
      return success.has(statusOf(task)) && Array.isArray(task.result.labels)
        ? [
            {
              ...task.result,
              id: "legacy",
              legacy: true,
              name: "分割结果",
              files: task.files || [],
              maskURL: `${taskURL(task.id)}/files/segmentation.nii.gz`,
            },
          ]
        : [];
    return task.result.outputs.flatMap((output, index) => {
      if (!output || !Array.isArray(output.labels)) return [];
      const files = (Array.isArray(output.files) ? output.files : []).filter(
        (file) =>
          file &&
          /^[a-z0-9_.-]+\.nii(?:\.gz)?$/i.test(file.name) &&
          sameOriginFile(
            file.url,
            `${taskURL(task.id)}/files/${encodeURIComponent(file.name)}`,
          ),
      );
      const file = files.find((item) => item.kind !== "label") || files[0];
      if (!file) return [];
      const targets = Array.isArray(output.targets)
        ? output.targets
        : output.labels.map((label) => label.name);
      const name =
        output.name && !/\.nii(?:\.gz)?$/i.test(output.name)
          ? labelDisplayName(output.name)
          : targets.length
            ? targets
                .slice(0, 2)
                .map((target) => labelDisplayName(target))
                .join("、") +
              (targets.length > 2 ? `等 ${targets.length} 项` : "")
            : `结果 ${index + 1}`;
      return [
        {
          ...output,
          files,
          id: String(output.id || `output-${index}`),
          name,
          maskURL: file.url,
        },
      ];
    });
  }

  function selectedOutput(task) {
    const outputs = resultOutputs(task);
    let id = state.outputSelection.get(task.id);
    if (!id) {
      try {
        id = sessionStorage.getItem(`medseg-output:${task.id}`);
      } catch {
        /* Optional preference. */
      }
    }
    const output = outputs.find((item) => item.id === id) || outputs[0] || null;
    if (output) state.outputSelection.set(task.id, output.id);
    return output;
  }

  function renderOutputSelector(task, selected) {
    const outputs = resultOutputs(task);
    $("result-output-choice").hidden = outputs.length < 2;
    $("result-output").replaceChildren();
    for (const output of outputs) {
      const option = document.createElement("option");
      option.value = output.id;
      option.textContent = output.name;
      $("result-output").append(option);
    }
    $("result-output").value = selected?.id || "";
  }

  function labelDisplayName(name, id) {
    return (
      {
        liver: "肝脏",
        kidney_left: "左肾",
        kidney_right: "右肾",
        spleen: "脾脏",
        pancreas: "胰腺",
        lungs: "双肺",
        lung_left: "左肺",
        lung_right: "右肺",
        lung_upper_lobe_left: "左肺上叶",
        lung_lower_lobe_left: "左肺下叶",
        lung_upper_lobe_right: "右肺上叶",
        lung_middle_lobe_right: "右肺中叶",
        lung_lower_lobe_right: "右肺下叶",
        lung_nodules: "肺结节",
        liver_lesions: "肝病灶",
        aorta: "主动脉",
        gallbladder: "胆囊",
        stomach: "胃",
      }[name] ||
      name ||
      String(id)
    );
  }

  function renderFileMetadata(upload) {
    $("file-size").textContent = Number.isFinite(upload?.size)
      ? size(upload.size)
      : "—";
    $("file-shape").textContent = upload?.shape?.join(" × ") || "—";
    $("file-spacing").textContent = upload?.spacing
      ? `${upload.spacing.map((x) => Number(x).toFixed(2)).join(" × ")} mm`
      : "—";
    $("file-meta").hidden = !upload;
  }

  function setDraft(upload = null, text = "") {
    state.upload = upload;
    state.selected = null;
    stopTaskClock();
    state.renderedRecord = null;
    state.pendingRequest = null;
    $("request").hidden = false;
    $("record-request").hidden = true;
    $("task-status").hidden = true;
    $("reuse-image").hidden = true;
    $("instruction").value = text;
    $("file-name").textContent = upload?.name || "选择或拖入 NIfTI";
    $("viewer-heading").textContent = upload?.name || "新建分割";
    $("viewer-name").textContent = upload?.shape
      ? `${upload.shape.join(" × ")} · ${size(upload.size || 0)}`
      : "";
    $("viewer-name").hidden = !$("viewer-name").textContent;
    $("empty-title").textContent = "从一张影像开始";
    renderFileMetadata(upload);
    $("upload-progress").hidden = true;
    upload
      ? $("drop-zone").classList.add("has-file")
      : $("drop-zone").classList.remove("has-file");
    const url = new URL(location.href);
    url.searchParams.delete("task");
    history.replaceState({}, "", url);
    renderDownloads(upload);
    renderHistory();
    setContext("request");
    renderExampleContext();
    updateSubmit();
  }

  function cancelExample() {
    const pending = state.exampleRequest;
    state.exampleRequest = null;
    pending?.controller.abort();
    if (pending) $("viewer-indicator").hidden = true;
    updateExampleButtons();
  }

  function updateExampleButtons() {
    for (const button of state.exampleButtons) {
      button.disabled = !state.ready || state.uploading || state.submitting;
      button.setAttribute(
        "aria-busy",
        String(state.exampleRequest?.id === button.dataset.example),
      );
    }
  }

  function attributionLink(attribution) {
    if (!attribution?.label || !attribution?.url) return null;
    let url;
    try {
      url = new URL(attribution.url);
    } catch {
      return null;
    }
    if (!["https:", "http:"].includes(url.protocol)) return null;
    const link = document.createElement("a");
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = attribution.label;
    link.title = attribution.license || "";
    return link;
  }

  function renderAttribution(container, example) {
    const links = [];
    const source = attributionLink(example?.attribution);
    if (source) links.push(source);
    if (typeof example?.attribution?.notice_url === "string") {
      try {
        const notice = new URL(example.attribution.notice_url, location.href);
        if (
          notice.origin === new URL(location.href).origin &&
          notice.pathname ===
            `/api/examples/${encodeURIComponent(example.id)}/license` &&
          !notice.search &&
          !notice.hash &&
          !notice.username &&
          !notice.password
        ) {
          const link = document.createElement("a");
          link.href = notice.pathname;
          link.setAttribute("download", "");
          link.textContent = "许可说明";
          links.push(link);
        }
      } catch {
        /* An unavailable notice must not create an unsafe link. */
      }
    }
    container.replaceChildren();
    for (const [index, link] of links.entries()) {
      if (index) {
        const separator = document.createElement("span");
        separator.textContent = " · ";
        container.append(separator);
      }
      container.append(link);
    }
    container.hidden = !links.length;
    return links.length > 0;
  }

  function renderExamples(config) {
    state.examples = (Array.isArray(config.examples) ? config.examples : [])
      .filter((example) => {
        if (
          typeof example.id !== "string" ||
          !/^[A-Za-z0-9_-]+$/.test(example.id)
        )
          return false;
        try {
          const preview = new URL(example.preview_url, location.href);
          return (
            preview.origin === new URL(location.href).origin &&
            preview.pathname.startsWith("/api/")
          );
        } catch {
          return false;
        }
      })
      .slice(0, 3);
    state.exampleButtons = [];
    $("example-cards").replaceChildren();
    $("example-switch-list").replaceChildren();
    $("example-gallery").hidden = !state.examples.length;
    for (const example of state.examples) {
      for (const [container, compact] of [
        ["example-cards", false],
        ["example-switch-list", true],
      ]) {
        const card = document.createElement("div");
        card.className = "example-card";
        const button = document.createElement("button");
        button.className = "example-button";
        button.type = "button";
        button.dataset.example = example.id;
        button.title = example.description || "";
        const preview = document.createElement("img");
        preview.className = "example-preview";
        preview.src = example.preview_url;
        preview.alt = "";
        preview.loading = "lazy";
        preview.addEventListener("error", () => {
          preview.style.visibility = "hidden";
        });
        const caption = document.createElement("span");
        caption.className = "example-caption";
        const title = document.createElement("span");
        title.className = "example-title";
        title.textContent = example.title || example.id;
        const details = document.createElement("span");
        details.className = "example-size";
        details.textContent = [
          example.modality,
          Number.isFinite(example.size_bytes) ? size(example.size_bytes) : "",
        ]
          .filter(Boolean)
          .join(" · ");
        caption.append(title, details);
        button.append(preview, caption);
        button.addEventListener("click", () => loadExample(example));
        state.exampleButtons.push(button);
        card.append(button);
        if (!compact) {
          const credit = document.createElement("span");
          credit.className = "example-credit";
          if (renderAttribution(credit, example)) card.append(credit);
        }
        $(container).append(card);
      }
    }
  }

  function renderExampleContext() {
    const example = state.examples.find(
      (item) => item.id === state.upload?.example_id,
    );
    $("example-prompts").replaceChildren();
    const prompts = (Array.isArray(example?.prompts) ? example.prompts : [])
      .filter((prompt) => typeof prompt.text === "string" && prompt.text.trim())
      .slice(0, 3);
    $("example-prompts").hidden = !prompts.length;
    for (const prompt of prompts) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-prompt";
      button.textContent = prompt.label || prompt.text;
      button.addEventListener("click", () => {
        if (state.upload?.example_id !== example.id || state.submitting) return;
        $("instruction").value = prompt.text;
        state.pendingRequest = null;
        $("instruction").focus();
        updateSubmit();
      });
      $("example-prompts").append(button);
    }
    $("example-switch").hidden =
      !state.examples.length || (!state.upload && !state.selected);
    updateExampleButtons();
  }

  function acceptUploadedImage(data, fallback = {}) {
    setDraft(
      {
        ...data,
        name: data.name || fallback.name,
        size: data.size ?? fallback.size,
      },
      $("instruction").value,
    );
    $("drop-zone").classList.add("has-file");
    queueViewer({
      key: `upload:${data.id}`,
      uploadID: data.id,
      name: data.name || fallback.name,
    });
  }

  async function loadExample(example) {
    if (
      !state.ready ||
      $("workspace").hidden ||
      state.uploading ||
      state.submitting
    )
      return;
    cancelExample();
    const request = { id: example.id, controller: new AbortController() },
      epoch = state.epoch;
    state.exampleRequest = request;
    invalidateViewer();
    clearViewer();
    setDraft();
    $("viewer-indicator").hidden = false;
    showError("form-error", "");
    try {
      const data = await api(
        `/api/examples/${encodeURIComponent(example.id)}`,
        { method: "POST", signal: request.controller.signal },
      );
      if (epoch !== state.epoch || state.exampleRequest !== request) return;
      if (!data.id) throw new Error("示例影像不可用，请重试。");
      state.exampleRequest = null;
      acceptUploadedImage(data);
      $("example-switch").open = false;
    } catch (err) {
      if (epoch === state.epoch && state.exampleRequest === request) {
        showError("form-error", err.message);
        $("viewer-indicator").hidden = true;
      }
    } finally {
      if (state.exampleRequest === request) state.exampleRequest = null;
      updateExampleButtons();
      updateSubmit();
    }
  }

  async function api(path, options = {}) {
    const epoch = state.epoch;
    const { timeoutMs = 30000, ...requestOptions } = options;
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (options.signal?.aborted) abort();
    options.signal?.addEventListener("abort", abort, { once: true });
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    state.activeRequests.add(controller);
    try {
      const response = await fetch(path, {
        credentials: "same-origin",
        cache: "no-store",
        ...requestOptions,
        signal: controller.signal,
        headers: {
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...options.headers,
        },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (
          response.status === 401 &&
          path !== "/api/session" &&
          epoch === state.epoch &&
          !controller.signal.aborted
        )
          lockWorkspace();
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
      options.signal?.removeEventListener("abort", abort);
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
    clearSourceImage();
    removeVolumes();
    state.viewerKey = null;
    state.resultKey = null;
    $("canvas-shell").dataset.loaded = "false";
    $("viewer-empty").hidden = false;
    $("result-panel").hidden = true;
    $("result-empty").hidden = false;
    $("result-empty").textContent = "尚无分割结果";
    enableWindowControls(false);
    $("location").textContent = "点击定位 · 滚轮切片 · 右键调窗";
    $("niivue-canvas").style.visibility = "hidden";
    $("viewer-indicator").hidden = true;
    $("retry-viewer").hidden = true;
    state.labels = [];
    state.visibleLabels.clear();
    $("result-output-choice").hidden = true;
  }

  function invalidateViewer() {
    saveView();
    state.viewerController?.abort();
    state.viewerWanted = null;
    enableWindowControls(false);
    $("niivue-canvas").style.visibility = "hidden";
    $("canvas-shell").dataset.loaded = "false";
    $("result-panel").hidden = true;
    $("result-empty").hidden = false;
    $("viewer-indicator").hidden = false;
    $("retry-viewer").hidden = true;
    showError("viewer-error", "");
  }

  function lockWorkspace() {
    cancelExample();
    cancelUpload();
    state.viewerController?.abort();
    clearTimeout(state.viewerSave);
    try {
      for (let i = sessionStorage.length - 1; i >= 0; i--) {
        const key = sessionStorage.key(i);
        if (key.startsWith("medseg-view:") || key.startsWith("medseg-output:"))
          sessionStorage.removeItem(key);
      }
    } catch {
      /* Storage may be disabled. */
    }
    state.authenticated = false;
    state.ready = false;
    state.identity = null;
    state.epoch += 1;
    clearTimeout(state.poll);
    clearTimeout(state.historyPoll);
    state.historyRequest = null;
    for (const controller of state.activeRequests) controller.abort();
    state.upload = null;
    state.selected = null;
    stopTaskClock();
    state.tasks = [];
    state.outputSelection.clear();
    state.labels = [];
    state.visibleLabels.clear();
    state.viewerWanted = null;
    state.uploading = false;
    state.submitting = false;
    state.pendingRequest = null;
    state.renderedRecord = null;
    state.examples = [];
    state.exampleButtons = [];
    $("example-cards").replaceChildren();
    $("example-switch-list").replaceChildren();
    renderExampleContext();
    $("history-search").value = "";
    $("logout").hidden = true;
    $("github-login").hidden = false;
    $("account-name").textContent = "未登录";
    $("account-name").removeAttribute("title");
    $("task-list").replaceChildren();
    renderHistory();
    $("labels").replaceChildren();
    $("request").reset();
    $("request").hidden = false;
    $("instruction").value = "";
    $("record-request").hidden = true;
    $("request-text").textContent = "";
    $("request-meta").textContent = "";
    $("viewer-heading").textContent = "新建分割";
    $("downloads").hidden = true;
    $("download-source").removeAttribute("href");
    $("download-source").hidden = true;
    $("download-labels").replaceChildren();
    $("download-labels").hidden = true;
    for (const id of [
      "form-error",
      "task-error",
      "viewer-error",
      "connection-note",
    ])
      showError(id, "");
    setContext("request");
    $("file-name").textContent = "选择或拖入 NIfTI";
    $("file-meta").hidden = true;
    $("upload-progress").hidden = true;
    $("task-status").hidden = true;
    $("viewer-name").textContent = "";
    $("viewer-name").hidden = true;
    $("reuse-image").hidden = true;
    $("drop-zone").classList.remove("has-file");
    clearViewer();
    updateSubmit();
    showError("session-error", "会话已过期，请重新连接。");
    $("session-retry").hidden = false;
  }

  function configureLogin(session) {
    const signedIn = session.identity?.kind === "github";
    $("github-login").hidden = signedIn;
    $("github-login").setAttribute(
      "aria-disabled",
      String(session.github_enabled !== true),
    );
    if (session.github_enabled === true) {
      $("github-login").href = "/api/auth/github/start";
      $("github-login").title = "登录后进入个人账号，游客记录不会转入";
    } else {
      $("github-login").removeAttribute("href");
      $("github-login").title = "GitHub 登录暂未开放";
    }
    $("logout").hidden = !signedIn;
    const name = signedIn
      ? session.identity.display_name || session.identity.login || "已登录"
      : "未登录";
    $("account-name").textContent = name;
    $("account-name").setAttribute("title", name);
  }

  async function openWorkspace(session) {
    clearSourceImage();
    state.authenticated = true;
    state.ready = false;
    state.identity = session.identity || null;
    const epoch = ++state.epoch;
    updateSubmit();
    const config = await api("/api/config");
    if (epoch !== state.epoch) return;
    state.maxUpload = Number(config.max_upload_bytes) || 0;
    state.singleUpload = Number(config.single_upload_bytes) || state.maxUpload;
    state.uploadChunkBytes = Number(config.upload_chunk_bytes) || 0;
    if (!state.maxUpload)
      throw new Error("服务器未提供上传大小限制，请检查服务配置。");
    $("file-help").textContent =
      `.nii / .nii.gz · 最大 ${size(state.maxUpload)}`;
    renderExamples(config);
    await refreshTasks();
    if (epoch !== state.epoch) return;
    // Keep the workspace visible while session, upload limits and history load.
    state.ready = true;
    updateSubmit();
    const requested = new URL(location.href).searchParams.get("task");
    if (requested)
      await selectTask(requested).catch((err) => {
        if (epoch === state.epoch) showError("connection-note", err.message);
      });
    else setDraft();
    if (epoch !== state.epoch) return;
    schedulePoll();
    scheduleHistoryPoll();
  }

  function uploadFile(file) {
    if (
      !file ||
      !state.ready ||
      $("workspace").hidden ||
      state.uploading ||
      state.submitting
    )
      return;
    cancelExample();
    state.upload = null;
    renderExampleContext();
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
    if (file.size > state.singleUpload) {
      uploadInChunks(file);
      return;
    }
    const xhr = new XMLHttpRequest();
    state.uploadXHR = xhr;
    xhr.open("POST", "/api/uploads");
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    // ASCII-safe header; the server decodes percent-encoding before validating the name.
    xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    xhr.timeout = 900000;
    xhr.upload.onprogress = (e) => {
      if (
        epoch === state.epoch &&
        state.uploadXHR === xhr &&
        e.lengthComputable
      ) {
        const n = Math.round((e.loaded / e.total) * 100);
        $("upload-bar").value = n;
        $("upload-status").textContent =
          n === 100 ? "上传完成，正在校验影像…" : `正在上传 ${n}%`;
      }
    };
    xhr.onload = () => {
      if (epoch !== state.epoch || state.uploadXHR !== xhr) return;
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
      acceptUploadedImage(data, file);
      $("upload-status").textContent = "影像已校验";
      $("upload-bar").value = 100;
    };
    xhr.onerror = () => {
      if (epoch === state.epoch && state.uploadXHR === xhr) {
        showError("form-error", "上传连接中断，请重新选择影像。");
        $("upload-progress").hidden = true;
      }
    };
    xhr.ontimeout = () => {
      if (epoch === state.epoch && state.uploadXHR === xhr) {
        showError("form-error", "上传超时，请检查网络后重试。");
        $("upload-progress").hidden = true;
      }
    };
    xhr.onloadend = () => {
      if (epoch === state.epoch && state.uploadXHR === xhr) {
        state.uploading = false;
        state.uploadXHR = null;
        updateSubmit();
      }
    };
    xhr.send(file);
  }

  const uploadSessionURL = (id) =>
    `/api/upload-sessions/${encodeURIComponent(id)}`;

  function abandonUploadSession(request) {
    if (!request?.id || request.completed || request.cleanupSent) return;
    request.cleanupSent = true;
    fetch(uploadSessionURL(request.id), {
      method: "DELETE",
      credentials: "same-origin",
      keepalive: true,
    }).catch(() => {});
  }

  function cancelUpload() {
    const request = state.chunkUpload;
    state.chunkUpload = null;
    const xhr = state.uploadXHR;
    state.uploadXHR = null;
    state.uploading = false;
    $("upload-progress").hidden = true;
    request?.controller.abort();
    xhr?.abort();
    abandonUploadSession(request);
  }

  function uploadIsCurrent(request) {
    return (
      state.chunkUpload === request &&
      request.epoch === state.epoch &&
      !request.controller.signal.aborted
    );
  }

  async function retryUpload(request, operation) {
    for (let attempt = 0; attempt < 3; attempt++) {
      if (!uploadIsCurrent(request))
        throw new DOMException("Upload canceled", "AbortError");
      try {
        return await operation();
      } catch (err) {
        if (
          !uploadIsCurrent(request) ||
          attempt === 2 ||
          (err.status && err.status < 500 && ![408, 429].includes(err.status))
        )
          throw err;
        $("upload-status").textContent = "连接中断，正在重试上传…";
        await new Promise((resolve) => {
          const finish = () => {
            clearTimeout(timer);
            request.controller.signal.removeEventListener("abort", finish);
            resolve();
          };
          const timer = setTimeout(finish, 500 * (attempt + 1));
          request.controller.signal.addEventListener("abort", finish, {
            once: true,
          });
        });
      }
    }
  }

  function sendUploadChunk(request, file, offset, end) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      state.uploadXHR = xhr;
      xhr.open("PUT", uploadSessionURL(request.id));
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.setRequestHeader("Upload-Offset", String(offset));
      xhr.timeout = 120000;
      xhr.upload.onprogress = (event) => {
        if (!uploadIsCurrent(request) || !event.lengthComputable) return;
        const progress = Math.min(
          100,
          Math.round(((offset + event.loaded) / file.size) * 100),
        );
        $("upload-bar").value = progress;
        $("upload-status").textContent = `正在上传 ${progress}%`;
      };
      xhr.onload = () => {
        let data;
        try {
          data = JSON.parse(xhr.responseText);
        } catch {
          data = {};
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else {
          const error = new Error(
            errorMessage(
              data.detail || data.error || `上传失败 (${xhr.status})`,
            ),
          );
          error.status = xhr.status;
          reject(error);
          if (xhr.status === 401 && uploadIsCurrent(request)) lockWorkspace();
        }
      };
      xhr.onerror = () =>
        reject(new Error("上传连接中断，请检查网络后重新选择影像。"));
      xhr.ontimeout = () =>
        reject(new Error("上传超时，请检查网络后重新选择影像。"));
      xhr.onabort = () =>
        reject(new DOMException("Upload canceled", "AbortError"));
      xhr.onloadend = () => {
        if (state.uploadXHR === xhr) state.uploadXHR = null;
      };
      xhr.send(file.slice(offset, end));
    });
  }

  async function uploadInChunks(file) {
    const request = {
      controller: new AbortController(),
      epoch: state.epoch,
      id: null,
    };
    state.chunkUpload = request;
    try {
      if (
        !Number.isSafeInteger(state.uploadChunkBytes) ||
        state.uploadChunkBytes <= 0
      )
        throw new Error("服务器未提供分块上传配置，请刷新后重试。");
      const body = JSON.stringify({
        name: file.name,
        size: file.size,
        message_id: crypto.randomUUID(),
      });
      const session = await retryUpload(request, () =>
        api("/api/upload-sessions", {
          method: "POST",
          body,
          signal: request.controller.signal,
        }),
      );
      request.id = session.id;
      if (!uploadIsCurrent(request)) return;
      if (
        typeof session.id !== "string" ||
        !session.id ||
        session.total_bytes !== file.size ||
        !Number.isSafeInteger(session.offset) ||
        session.offset < 0 ||
        session.offset > file.size ||
        !Number.isSafeInteger(session.chunk_bytes) ||
        session.chunk_bytes <= 0
      )
        throw new Error("上传会话返回无效，请重新选择影像。");
      let offset = session.offset;
      const chunkSize = Math.min(state.uploadChunkBytes, session.chunk_bytes);
      while (offset < file.size) {
        const end = Math.min(offset + chunkSize, file.size);
        // Reuse identical bytes and offset after an acknowledgement is lost.
        const next = await retryUpload(request, () =>
          sendUploadChunk(request, file, offset, end),
        );
        if (!uploadIsCurrent(request)) return;
        if (
          next.id !== request.id ||
          next.total_bytes !== file.size ||
          !Number.isSafeInteger(next.offset) ||
          next.offset !== end
        )
          throw new Error("上传进度校验失败，请重新选择影像。");
        offset = next.offset;
        $("upload-bar").value = Math.round((offset / file.size) * 100);
      }
      if (!uploadIsCurrent(request)) return;
      $("upload-status").textContent = "上传完成，正在校验影像…";
      const data =
        session.upload ||
        (await retryUpload(request, () =>
          api(`${uploadSessionURL(request.id)}/complete`, {
            method: "POST",
            signal: request.controller.signal,
            timeoutMs: 900000,
          }),
        ));
      if (!uploadIsCurrent(request)) return;
      if (!data?.id) throw new Error("影像校验未完成，请重新选择影像。");
      request.completed = true;
      acceptUploadedImage(data, file);
    } catch (err) {
      if (uploadIsCurrent(request)) {
        showError("form-error", err.message);
        $("upload-progress").hidden = true;
      }
    } finally {
      abandonUploadSession(request);
      if (state.chunkUpload === request) {
        state.chunkUpload = null;
        state.uploading = false;
        updateSubmit();
      }
    }
  }

  async function refreshTasks() {
    const epoch = state.epoch;
    if (state.historyRequest?.epoch === epoch)
      return state.historyRequest.promise;
    const request = { epoch };
    state.historyRequest = request;
    request.promise = (async () => {
      try {
        const data = await api("/api/tasks");
        if (epoch !== state.epoch || state.historyRequest !== request) return;
        state.tasks = Array.isArray(data)
          ? data
          : data.tasks || data.items || [];
        // A full-list response may have started before the selected detail poll.
        // Keep that newer local detail authoritative while refreshing other rows.
        if (state.selected?.status) updateHistoryTask(state.selected);
        renderHistory();
      } finally {
        if (state.historyRequest === request) state.historyRequest = null;
      }
    })();
    return request.promise;
  }

  function updateHistoryTask(task) {
    const index = state.tasks.findIndex((row) => row.id === task.id);
    const previous = state.tasks[index];
    const summary = (row) =>
      row &&
      JSON.stringify([
        statusOf(row),
        row.text,
        row.upload_name,
        row.input_available,
        row.result_available,
        row.created_at,
        row.error?.code,
      ]);
    const changed = summary(previous) !== summary(task);
    if (index < 0) state.tasks.unshift(task);
    else state.tasks[index] = task;
    return changed;
  }

  function refreshHistoryInBackground() {
    const epoch = state.epoch;
    refreshTasks().catch((err) => {
      if (state.authenticated && epoch === state.epoch)
        showError("connection-note", err.message);
    });
  }

  function scheduleHistoryPoll() {
    clearTimeout(state.historyPoll);
    if (!state.authenticated) return;
    const otherRunning = state.tasks.some(
      (task) => task.id !== state.selected?.id && isWorkingTask(task),
    );
    state.historyPoll = setTimeout(
      async () => {
        const epoch = state.epoch;
        try {
          await refreshTasks();
        } catch (err) {
          if (state.authenticated && epoch === state.epoch)
            showError("connection-note", err.message);
        } finally {
          if (epoch === state.epoch) scheduleHistoryPoll();
        }
      },
      document.hidden ? 30000 : otherRunning ? 6000 : 30000,
    );
  }

  function renderHistory() {
    $("task-list").replaceChildren();
    $("history-empty").hidden = state.tasks.length > 0;
    $("task-count").textContent = state.tasks.length || "";
    const query = $("history-search").value.trim().toLocaleLowerCase();
    const tasks = state.tasks.filter((task) =>
      [task.text, task.upload_name, task.input?.name]
        .filter(Boolean)
        .join(" ")
        .toLocaleLowerCase()
        .includes(query),
    );
    $("history-empty").hidden = tasks.length > 0;
    $("history-empty").textContent = query
      ? "没有匹配的分割记录。"
      : "暂无分割记录";
    for (const task of tasks) {
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
        statusOf(task) === "input_required"
          ? statusNames.input_required
          : task.error?.code === "MODALITY_REQUIRED"
            ? "待补充说明"
            : statusNames[statusOf(task)] || statusOf(task);
      if (
        statusOf(task) !== "input_required" &&
        task.input_available === false &&
        task.result_available === false
      )
        status.textContent = "文件已清理";
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
      const source = document.createElement("span");
      source.className = "task-item-source";
      source.textContent = task.upload_name || task.input?.name || "";
      button.append(title, source, meta);
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
    cancelExample();
    const epoch = state.epoch;
    cancelUpload();
    // Mark the requested ID before fetching so a slower previous request cannot replace it.
    if (state.selected?.id !== id) invalidateViewer();
    stopTaskClock();
    state.selected = { id };
    state.upload = null;
    $("request").hidden = true;
    $("record-request").hidden = true;
    $("task-status").hidden = true;
    $("reuse-image").hidden = true;
    $("downloads").hidden = true;
    updateSubmit();
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
    updateHistoryTask(task);
    const url = new URL(location.href);
    url.searchParams.set("task", id);
    history.replaceState({}, "", url);
    renderTask(task);
    renderHistory();
    schedulePoll();
    scheduleHistoryPoll();
  }

  const DEFAULT_ESTIMATE_SECONDS = 60;
  const validSeconds = (value) => Number.isFinite(value) && value >= 0;
  function formatDuration(seconds) {
    const whole = Math.max(0, Math.floor(seconds));
    if (whole < 60) return `${whole} 秒`;
    const minutes = Math.floor(whole / 60),
      rest = whole % 60;
    return rest ? `${minutes} 分 ${rest} 秒` : `${minutes} 分钟`;
  }

  function taskStage(task) {
    const status = statusOf(task);
    if (!isWorkingTask(task) || ["canceling", "cancelling"].includes(status))
      return statusNames[status] || status;
    const progress = task.agent_progress;
    if (progress?.phase === "publishing") return "正在整理结果";
    if (progress?.phase === "reasoning")
      return progress.model_requests > 1 ? "正在分析与规划" : "正在理解请求";
    if (progress?.phase === "tool") {
      return (
        {
          get_capabilities: "正在匹配分割工具",
          detect_modality: "正在识别影像类型",
          segment:
            task.progress === "Running local segmentation"
              ? "正在分割"
              : "正在准备分割",
          inspect_artifact: "正在检查分割结果",
          compose_masks: "正在合并分割结果",
        }[progress.tool] || "正在处理"
      );
    }
    if (progress?.phase === "observed") return "正在分析结果";
    return statusNames[status] || "正在处理";
  }

  function stopTaskClock() {
    clearTimeout(state.taskTick);
    state.taskTick = null;
    state.taskClock = null;
  }

  function clockElapsed(clock) {
    if (!validSeconds(clock?.elapsed)) return null;
    return (
      clock.elapsed +
      (isWorkingTask(clock.task)
        ? Math.max(0, performance.now() - clock.observedAt) / 1000
        : 0)
    );
  }

  function drawTaskProgress() {
    clearTimeout(state.taskTick);
    state.taskTick = null;
    const clock = state.taskClock;
    if (
      !clock ||
      !state.authenticated ||
      clock.epoch !== state.epoch ||
      state.selected?.id !== clock.task.id
    )
      return;
    const task = clock.task,
      working = isWorkingTask(task);
    $("task-progress").hidden = !working;
    $("task-timing").hidden = !working;
    if (!working) return;
    const elapsed = clockElapsed(clock);
    const estimate =
      Number.isFinite(task.estimated_duration_seconds) &&
      task.estimated_duration_seconds > 0
        ? task.estimated_duration_seconds
        : DEFAULT_ESTIMATE_SECONDS;
    const overdue = validSeconds(elapsed) && elapsed >= estimate;
    $("task-elapsed").textContent = validSeconds(elapsed)
      ? `已用 ${formatDuration(elapsed)}`
      : "正在计时";
    $("task-estimate").textContent = overdue
      ? "仍在处理中"
      : `预计约 ${estimate} 秒`;
    const percent =
      typeof task.progress === "number"
        ? task.progress
        : task.progress?.percent;
    const measured = Number.isFinite(percent);
    const label = measured ? "任务进度" : "预估进度";
    $("task-progress").setAttribute("aria-label", label);
    $("task-progress").setAttribute(
      "aria-valuetext",
      `${label}，${$("task-elapsed").textContent}，${$("task-estimate").textContent}`,
    );
    if (measured) $("task-progress").value = Math.max(0, Math.min(99, percent));
    else if (validSeconds(elapsed))
      $("task-progress").value = Math.min(
        95,
        Math.max(2, (elapsed / estimate) * 95),
      );
    else $("task-progress").removeAttribute("value");
    if (!document.hidden) state.taskTick = setTimeout(drawTaskProgress, 1000);
  }

  function syncTaskClock(task) {
    const previous = state.taskClock;
    if (previous?.task !== task) {
      let elapsed = validSeconds(task.elapsed_seconds)
        ? task.elapsed_seconds
        : null;
      // A new response anchors the local monotonic clock. Keep running time from
      // stepping backward because of response latency; terminal time is authoritative.
      if (
        validSeconds(elapsed) &&
        previous?.task.id === task.id &&
        isWorkingTask(task) &&
        isWorkingTask(previous.task)
      )
        elapsed = Math.max(elapsed, clockElapsed(previous) ?? 0);
      state.taskClock = {
        task,
        elapsed,
        observedAt: performance.now(),
        epoch: state.epoch,
      };
    }
    drawTaskProgress();
  }

  function renderTask(task) {
    const status = statusOf(task),
      waiting = status === "input_required",
      complete = success.has(status);
    const input = task.input || { id: task.upload_id, name: task.upload_name };
    const uploadID = task.upload_id || input.id || input.upload_id;
    const inputOK = !!uploadID && task.input_available !== false;
    const output = selectedOutput(task);
    const resultOK = !!output;
    const partial = resultOK && terminal.has(status) && !complete;
    const outputChanged = state.renderedRecord?.outputID !== output?.id;
    const availabilityChanged =
      state.renderedRecord?.inputOK !== inputOK ||
      state.renderedRecord?.resultOK !== resultOK;
    const changed = state.renderedRecord?.id !== task.id;
    const justCompleted =
      complete && !success.has(state.renderedRecord?.status);
    if (changed || justCompleted) setContext(resultOK ? "results" : "request");
    state.renderedRecord = {
      id: task.id,
      status,
      inputOK,
      resultOK,
      outputID: output?.id,
    };
    renderOutputSelector(task, output);
    $("request").hidden = true;
    $("record-request").hidden = false;
    $("request-text").textContent = task.text || "未提供文本请求";
    $("request-meta").textContent = [
      formatDate(task.created_at),
      task.modality,
      terminal.has(status) && validSeconds(task.elapsed_seconds)
        ? `耗时 ${formatDuration(task.elapsed_seconds)}`
        : "",
    ]
      .filter(Boolean)
      .join(" · ");
    $("task-status").hidden = false;
    $("task-status").dataset.status = status;
    const title =
      !waiting && task.error?.code === "MODALITY_REQUIRED"
        ? "需要补充说明"
        : taskStage(task);
    if ($("status-title").textContent !== title)
      $("status-title").textContent = title;
    const clarification = waiting
      ? [
          task.error?.message,
          typeof task.progress === "string" ? task.progress : null,
          task.progress?.message,
          task.progress?.text,
        ].find((value) => typeof value === "string" && value.trim())
      : null;
    $("status-detail").textContent = waiting
      ? `${clarification || "请补充所需信息。"} 通过 A2A 继续。`
      : partial
        ? "已有部分结果，任务尚未完整完成。"
        : complete
          ? inputOK && resultOK
            ? ""
            : resultOK
              ? "原图不可用，仍可下载分割结果"
              : "分割文件已到期或已清理"
          : {
              queued: "等待推理资源",
              routing: "正在理解分割请求",
              validating: "正在检查影像",
              working: "正在生成分割掩膜",
            }[status] || "";
    $("status-detail").hidden = !$("status-detail").textContent;
    syncTaskClock(task);
    $("cancel-task").hidden = terminal.has(status);
    $("cancel-task").disabled = ["canceling", "cancelling"].includes(status);
    showError(
      "task-error",
      waiting
        ? ""
        : task.error ||
            (status === "failed"
              ? "任务未完成，请查看错误信息后新建任务重试。"
              : ""),
    );
    $("viewer-heading").textContent =
      task.upload_name || input.name || "分割记录";
    $("viewer-name").textContent = [task.modality, input.shape?.join(" × ")]
      .filter(Boolean)
      .join(" · ");
    $("viewer-name").hidden = !$("viewer-name").textContent;
    $("reuse-image").hidden = !inputOK;
    renderExampleContext();
    renderDownloads({ ...input, id: uploadID }, task);
    if (inputOK)
      queueViewer({
        key:
          output && !output.legacy
            ? `${task.id}:output:${output.id}:result`
            : `${task.id}:${resultOK ? "result" : "source"}`,
        taskID: task.id,
        uploadID,
        name: task.upload_name || input.name || "source.nii",
        result: output,
        maskURL: output?.maskURL,
        aggregate: task.result,
        partial,
        completed: resultOK,
      });
    else if (
      changed ||
      outputChanged ||
      availabilityChanged ||
      state.viewerWanted?.taskID === task.id
    ) {
      state.viewerController?.abort();
      state.viewerWanted = null;
      clearViewer();
      $("empty-title").textContent =
        waiting && !uploadID ? "尚未提供影像" : "原始影像不可用";
      showError(
        "viewer-error",
        waiting && !uploadID ? "" : "原始影像不可用，请重新上传。",
      );
      if (resultOK) {
        renderOutputSelector(task, output);
        showResult(output, false, task.result, partial);
      }
    }
    if (complete && !resultOK)
      $("result-empty").textContent = "分割文件已到期或已清理。";
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
              const changed = updateHistoryTask(task);
              renderTask(task);
              if (changed) {
                renderHistory();
                refreshHistoryInBackground();
                scheduleHistoryPoll();
              }
            }
          }
          if (epoch === state.epoch) showError("connection-note", "");
        } catch (err) {
          if (state.authenticated) showError("connection-note", err.message);
        } finally {
          if (epoch === state.epoch) schedulePoll();
        }
      },
      state.selected && statusOf(state.selected) === "input_required"
        ? 30000
        : document.hidden
          ? 10000
          : state.selected && isWorkingTask(state.selected)
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
        syncWindowNumbers(true);
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
  function enableWindowControls(enabled) {
    for (const id of ["window-preset", "window-width", "window-level"])
      $(id).disabled = !enabled;
    if (!enabled) {
      $("window-width").value = "";
      $("window-level").value = "";
      state.windowInvalid = false;
      for (const id of ["window-width", "window-level"])
        $(id).removeAttribute("aria-invalid");
      showError("window-error", "");
    }
  }
  function syncWindowNumbers(force = false) {
    const src = state.viewer?.volumes[0];
    if (!src || !Number.isFinite(src.cal_min) || !Number.isFinite(src.cal_max))
      return;
    if (force) {
      state.windowInvalid = false;
      showError("window-error", "");
      for (const id of ["window-width", "window-level"])
        $(id).removeAttribute("aria-invalid");
    }
    if (state.windowInvalid) return;
    for (const [id, value] of [
      ["window-width", src.cal_max - src.cal_min],
      ["window-level", src.cal_min / 2 + src.cal_max / 2],
    ]) {
      if (force || document.activeElement !== $(id))
        $(id).value = String(Number(value.toPrecision(12)));
    }
  }
  function applyWindowNumbers() {
    const viewer = state.viewer,
      src = viewer?.volumes[0];
    if (!src || !state.viewerKey || $("window-width").disabled) return;
    const widthText = $("window-width").value.trim(),
      levelText = $("window-level").value.trim(),
      width = Number(widthText),
      level = Number(levelText),
      minimum = level - width / 2,
      maximum = level + width / 2;
    const widthOK = widthText !== "" && Number.isFinite(width) && width > 0,
      levelOK = levelText !== "" && Number.isFinite(level),
      // NiiVue passes window bounds to WebGL as 32-bit floating-point uniforms.
      rangeOK =
        Number.isFinite(Math.fround(minimum)) &&
        Number.isFinite(Math.fround(maximum)) &&
        Math.fround(maximum) > Math.fround(minimum);
    if (!widthOK || !levelOK || !rangeOK) {
      state.windowInvalid = true;
      $("window-width").setAttribute(
        "aria-invalid",
        String(!widthOK || !rangeOK),
      );
      $("window-level").setAttribute(
        "aria-invalid",
        String(!levelOK || !rangeOK),
      );
      showError(
        "window-error",
        "请输入有效数字，窗宽必须大于 0；当前显示未改变。",
      );
      return;
    }
    src.cal_min = minimum;
    src.cal_max = maximum;
    $("window-preset").value = "custom";
    syncWindowNumbers(true);
    viewer.updateGLVolume();
    saveView();
  }
  function syncWindowPreset() {
    const src = state.viewer?.volumes[0];
    if (!src || state.restoring) return;
    const presets = {
      auto: [src.robust_min, src.robust_max],
      soft: [-160, 240],
      lung: [-1350, 150],
      bone: [-500, 1500],
    };
    const expected = presets[$("window-preset").value];
    if (
      expected &&
      expected.every(Number.isFinite) &&
      (Math.abs(src.cal_min - expected[0]) > 0.001 ||
        Math.abs(src.cal_max - expected[1]) > 0.001)
    )
      $("window-preset").value = "custom";
    syncWindowNumbers();
  }

  function saveView() {
    const v = state.viewer;
    if (!state.viewerKey || !v?.volumes[0] || state.restoring) return;
    syncWindowPreset();
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
      if (state.sourceImage?.volume === v.volumes[0]) {
        [v.volumes[0].cal_min, v.volumes[0].cal_max] = state.sourceImage.window;
      }
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
      syncWindowNumbers(true);
    }
    return Boolean(saved);
  }

  async function imageBuffer(url, controller) {
    const epoch = state.epoch;
    const response = await fetch(url, {
      credentials: "same-origin",
      cache: "no-store",
      signal: controller.signal,
    });
    if (epoch !== state.epoch || controller.signal.aborted)
      throw new DOMException("Image request was canceled", "AbortError");
    if (response.status === 401) {
      lockWorkspace();
      throw new Error("会话已过期，请重新连接。");
    }
    if (!response.ok)
      throw new Error(
        response.status === 404 || response.status === 410
          ? "影像文件已过期或不可访问，请重新上传。"
          : `影像读取失败 (${response.status})`,
      );
    return response.arrayBuffer();
  }

  function clearSourceImage() {
    state.sourceImage?.controller.abort();
    state.sourceImage = null;
  }

  function sourceImage(request) {
    if (
      state.sourceImage?.epoch === state.epoch &&
      state.sourceImage.uploadID === request.uploadID
    )
      return state.sourceImage.promise;
    clearSourceImage();
    const cached = {
      epoch: state.epoch,
      uploadID: request.uploadID,
      controller: new AbortController(),
      volume: null,
    };
    state.sourceImage = cached;
    // The source belongs to this image, not to an individual overlay request.
    // Switching outputs can cancel their masks without canceling this decode.
    cached.promise = (async () => {
      const timeout = setTimeout(
        () => cached.controller.abort("timeout"),
        180000,
      );
      try {
        const bytes = await imageBuffer(
          uploadURL(request.uploadID),
          cached.controller,
        );
        if (state.sourceImage !== cached || cached.controller.signal.aborted)
          throw new DOMException("Image request was canceled", "AbortError");
        const volume = await window.niivue.NVImage.loadFromUrl({
          url: bytes,
          name: request.name || "source.nii",
          colormap: "gray",
          opacity: 1,
          colorbarVisible: false,
        });
        if (state.sourceImage !== cached || cached.controller.signal.aborted)
          throw new DOMException("Image request was canceled", "AbortError");
        cached.volume = volume;
        cached.window = [volume.cal_min, volume.cal_max];
        return volume;
      } catch (err) {
        if (state.sourceImage === cached) clearSourceImage();
        if (cached.controller.signal.reason === "timeout")
          throw new Error("载入超时，请检查网络后重试。");
        throw err;
      } finally {
        clearTimeout(timeout);
      }
    })();
    return cached.promise;
  }

  function removeOverlays() {
    const viewer = state.viewer;
    if (!viewer?.gl) return;
    for (const volume of viewer.volumes.slice(1).reverse())
      viewer.removeVolume(volume);
    viewer.mediaUrlMap?.clear();
    viewer.drawScene();
  }

  function viewPosition() {
    const scene = state.viewer.scene;
    return JSON.stringify([
      state.viewMode,
      scene.crosshairPos,
      scene.pan2Dxyzmm,
      scene.renderAzimuth,
      scene.renderElevation,
      scene.volScaleMultiplier,
    ]);
  }

  function queueViewer(request, force = false) {
    if (!force && state.viewerWanted?.key === request.key) {
      if (request.result && state.resultKey === request.key)
        renderResultSummary(request.result, request.aggregate, request.partial);
      return;
    }
    saveView();
    state.viewerController?.abort();
    state.viewerWanted = request;
    const reusable =
      state.sourceImage?.epoch === state.epoch &&
      state.sourceImage.uploadID === request.uploadID;
    if (!reusable) {
      clearSourceImage();
      removeVolumes();
    } else removeOverlays();
    state.viewerKey = null;
    state.resultKey = null;
    enableWindowControls(false);
    const epoch = state.epoch;
    $("result-panel").hidden = true;
    $("result-empty").hidden = false;
    $("viewer-indicator").hidden = false;
    $("viewer-indicator").dataset.stage = "source";
    $("viewer-loading-text").textContent = "正在载入影像…";
    $("retry-viewer").hidden = true;
    showError("viewer-error", "");
    $("niivue-canvas").style.visibility = "hidden";
    $("canvas-shell").dataset.loaded = "false";
    // Stale fetch/decode work never blocks a newer selection. All viewer writes
    // below are guarded by this request and the authenticated identity epoch.
    state.viewerQueue = Promise.resolve().then(async () => {
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
        if (request.completed) {
          const result = request.result;
          if (!result || !Array.isArray(result.labels))
            throw new Error("分割标签信息不可用，请刷新任务后重试。");
          if (!current()) return;
          showResult(result, false, request.aggregate, request.partial);
          state.resultKey = request.key;
        }
        const viewer = await getViewer();
        if (!current()) return;
        const background = await sourceImage(request);
        if (!current()) return;
        if (viewer.volumes[0] !== background) {
          removeVolumes();
          viewer.addVolume(background);
        }
        const restored = restoreView(request.key);
        state.viewerKey = request.key;
        $("viewer-empty").hidden = true;
        $("canvas-shell").dataset.loaded = "true";
        $("niivue-canvas").style.visibility = "visible";
        enableWindowControls(true);
        $("canvas-shell").dataset.source = request.uploadID;
        $("canvas-shell").dataset.task = request.taskID || "";
        viewer.resizeListener();
        viewer.drawScene();
        if (request.completed) {
          $("viewer-indicator").dataset.stage = "overlay";
          $("viewer-loading-text").textContent = "正在载入分割结果…";
          const position = viewPosition();
          const mask = await imageBuffer(
            request.maskURL ||
              `${taskURL(request.taskID)}/files/segmentation.nii.gz`,
            controller,
          );
          if (!current()) return;
          const overlay = await window.niivue.NVImage.loadFromUrl({
            url: mask,
            name: request.maskURL?.split("/").pop() || "segmentation.nii.gz",
            colormap: "gray",
            opacity: 0,
            cal_min: 0,
            cal_max: Math.max(1, ...state.labels.map((l) => l.id)),
            colorbarVisible: false,
          });
          if (!current()) return;
          if (controller.signal.aborted)
            throw new DOMException("Image request was canceled", "AbortError");
          viewer.addVolume(overlay);
          updateOverlay();
          if (!restored && position === viewPosition()) focusForeground(viewer);
          updateSlices();
          viewer.createOnLocationChange();
        }
        if (!current()) return;
        viewer.drawScene();
        saveView();
      } catch (err) {
        if (current()) {
          // A failed or expired overlay must not hide a valid, decoded source.
          const hasSource = Boolean(
            state.sourceImage?.volume &&
            state.sourceImage.volume === state.viewer?.volumes[0],
          );
          if (hasSource) removeOverlays();
          else {
            removeVolumes();
            state.viewerKey = null;
          }
          showError(
            "viewer-error",
            controller.signal.reason === "timeout"
              ? "载入超时，请检查网络后重试。"
              : `${hasSource ? "分割结果" : "影像"}显示失败：${err.message}`,
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

  function showResult(result, apply = true, aggregate = null, partial = false) {
    state.labels = (result.labels || [])
      .filter(
        (label) =>
          Number.isInteger(Number(label.id)) &&
          Number(label.id) > 0 &&
          Number(label.id) <= 65535,
      )
      .map((label, index) => ({
        ...label,
        id: Number(label.id),
        rgb: labelColor(label.color, index),
      }));
    state.visibleLabels = new Set(state.labels.map((label) => label.id));
    $("labels").replaceChildren();
    $("label-search").value = "";
    $("label-search").hidden = state.labels.length < 8;
    $("label-count").textContent = `(${state.labels.length})`;
    for (const label of state.labels) {
      const wrap = document.createElement("label");
      wrap.className = "label-option";
      wrap.dataset.name = label.name || "";
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
      swatch.style.backgroundColor = `rgb(${label.rgb.join(",")})`;
      const name = document.createElement("span");
      name.className = "label-name";
      name.textContent = labelDisplayName(label.name, label.id);
      name.title = label.name || "";
      const detail = document.createElement("span");
      detail.className = "label-details";
      const count = document.createElement("span");
      count.className = "label-size";
      const volumeML =
        label.volume_ml ??
        (Number.isFinite(label.volume_mm3) ? label.volume_mm3 / 1000 : null);
      count.textContent = Number.isFinite(volumeML)
        ? formatVolume(volumeML)
        : "体积未计算";
      count.title =
        result.volume_measurement?.unit_assumption === "assumed_mm"
          ? "影像未注明空间单位，体积按 mm 估算。"
          : "";
      wrap.title = [
        `${name.textContent} (${label.name}) · 标签 ${label.id}`,
        Number.isFinite(label.voxels)
          ? `${label.voxels.toLocaleString()} 体素`
          : "",
      ]
        .filter(Boolean)
        .join("\n");
      detail.append(name, count);
      wrap.append(checkbox, swatch, detail);
      $("labels").append(wrap);
    }
    renderResultSummary(result, aggregate, partial);
    $("result-panel").hidden = false;
    $("result-empty").hidden = true;
    if (apply) updateOverlay();
  }

  function renderResultSummary(result, aggregate, partial) {
    const elapsed =
      aggregate?.total_seconds ??
      result.total_seconds ??
      result.duration_seconds ??
      result.elapsed_seconds ??
      result.runtime_seconds;
    $("result-summary").textContent = [
      partial ? "部分结果 · 任务未完成" : "",
      Array.isArray(aggregate?.outputs) && aggregate.outputs.length > 1
        ? `共 ${aggregate.outputs.length} 项结果`
        : "",
      Number.isFinite(elapsed) ? `处理用时 ${elapsed.toFixed(1)} 秒` : "",
      result.detection_status === "no_target_detected" ? "未检出目标" : "",
    ]
      .filter(Boolean)
      .join(" / ");
    $("result-summary").hidden = !$("result-summary").textContent;
    const unresolved = Array.isArray(aggregate?.completion?.unresolved)
      ? aggregate.completion.unresolved.filter(
          (item) => typeof item === "string",
        )
      : [];
    $("result-explanation").textContent = [
      typeof aggregate?.summary === "string" ? aggregate.summary : "",
      unresolved.length ? `尚未完成：${unresolved.join("；")}` : "",
    ]
      .filter(Boolean)
      .join("\n");
    $("result-explanation").hidden = !$("result-explanation").textContent;
  }

  function labelColor(color, index) {
    if (typeof color === "string" && /^#[0-9a-f]{6}$/i.test(color))
      return [1, 3, 5].map((offset) =>
        parseInt(color.slice(offset, offset + 2), 16),
      );
    return palette[index % palette.length];
  }

  function formatVolume(value) {
    if (value > 0 && value < 0.01) return "< 0.01 mL";
    return `${value.toLocaleString("zh-CN", { maximumFractionDigits: 2 })} mL`;
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
    for (const label of state.labels) {
      const color = label.rgb;
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

  async function connectWorkspace() {
    if ($("session-retry").disabled) return;
    $("session-retry").disabled = true;
    $("session-retry").hidden = true;
    showError("session-error", "");
    try {
      let session = await api("/api/session");
      configureLogin(session);
      if (session.authenticated !== true)
        session = await api("/api/auth/guest", { method: "POST" });
      if (session.authenticated !== true)
        throw new Error("暂时无法建立游客会话，请重新连接。");
      configureLogin(session);
      await openWorkspace(session);
    } catch (err) {
      lockWorkspace();
      showError("session-error", err.message);
    } finally {
      $("session-retry").disabled = false;
    }
  }
  $("session-retry").addEventListener("click", connectWorkspace);
  $("logout").addEventListener("click", async () => {
    if ($("logout").disabled || $("session-retry").disabled) return;
    $("logout").disabled = true;
    // Start cleanup while the authentication cookie still exists.
    cancelUpload();
    updateSubmit();
    try {
      await api("/api/session", { method: "DELETE" });
      const url = new URL(location.href);
      url.searchParams.delete("task");
      history.replaceState({}, "", url);
      lockWorkspace();
      await connectWorkspace();
    } catch (err) {
      showError("connection-note", err.message);
    } finally {
      $("logout").disabled = false;
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
    if (!state.ready || !state.upload || state.uploading || state.submitting)
      return;
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
      updateHistoryTask(task);
      renderHistory();
      await selectTask(task.id);
      refreshHistoryInBackground();
      schedulePoll();
    } catch (err) {
      showError("form-error", err.message);
    } finally {
      state.submitting = false;
      updateSubmit();
    }
  });
  $("refresh-tasks").addEventListener("click", async () => {
    if (!state.ready) return;
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
      refreshHistoryInBackground();
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
    syncWindowNumbers(true);
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
    if (!src || !state.viewerKey) return;
    const values = {
      soft: [-160, 240],
      lung: [-1350, 150],
      bone: [-500, 1500],
    };
    [src.cal_min, src.cal_max] = values[$("window-preset").value] || [
      src.robust_min,
      src.robust_max,
    ];
    syncWindowNumbers(true);
    v.updateGLVolume();
    saveView();
  });
  for (const id of ["window-width", "window-level"]) {
    $(id).addEventListener("blur", applyWindowNumbers);
    $(id).addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        applyWindowNumbers();
      }
    });
  }
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
  async function useSelectedImage() {
    const task = state.selected;
    if (!task?.upload_id || task.input_available === false) return;
    const epoch = state.epoch;
    $("reuse-image").disabled = true;
    try {
      const input = await api(
        `/api/uploads/${encodeURIComponent(task.upload_id)}`,
      );
      if (epoch !== state.epoch || state.selected?.id !== task.id) return;
      invalidateViewer();
      setDraft(input, task.text || "");
      queueViewer({
        key: `upload:${input.id}`,
        uploadID: input.id,
        name: input.name,
      });
      showError("form-error", "");
      $("instruction").focus();
    } catch (err) {
      if (epoch === state.epoch && state.selected?.id === task.id) {
        if (err.status === 404 || err.status === 410) {
          state.selected = { ...state.selected, input_available: false };
          updateHistoryTask(state.selected);
          renderTask(state.selected);
          renderHistory();
        }
        showError("task-error", err.message);
      }
    } finally {
      $("reuse-image").disabled = false;
    }
  }
  $("reuse-image").addEventListener("click", useSelectedImage);
  function newTask() {
    if (!state.ready || state.submitting) return;
    cancelExample();
    cancelUpload();
    invalidateViewer();
    clearViewer();
    $("request").reset();
    setDraft();
    showError("form-error", "");
    showError("viewer-error", "");
    showError("connection-note", "");
  }
  $("new-task").addEventListener("click", newTask);
  $("choose-image").addEventListener("click", () => {
    if (!state.ready) return;
    newTask();
    $("file").click();
  });
  for (const name of ["request", "results"]) {
    $("tab-" + name).addEventListener("click", () => setContext(name));
    $("tab-" + name).addEventListener("keydown", (event) => {
      if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const next =
          event.key === "Home"
            ? "request"
            : event.key === "End"
              ? "results"
              : name === "request"
                ? "results"
                : "request";
        setContext(next);
        $("tab-" + next).focus();
      }
    });
  }
  $("history-search").addEventListener("input", renderHistory);
  $("result-output").addEventListener("change", () => {
    const task = state.selected;
    if (!task) return;
    const id = $("result-output").value;
    if (!resultOutputs(task).some((output) => output.id === id)) return;
    saveView();
    state.outputSelection.set(task.id, id);
    try {
      sessionStorage.setItem(`medseg-output:${task.id}`, id);
      const keys = Object.keys(sessionStorage).filter((key) =>
        key.startsWith("medseg-output:"),
      );
      for (const key of keys.slice(0, Math.max(0, keys.length - 24)))
        sessionStorage.removeItem(key);
    } catch {
      /* Optional preference; image data is never stored. */
    }
    renderTask(task);
  });
  $("label-search").addEventListener("input", () => {
    const query = $("label-search").value.trim().toLocaleLowerCase();
    for (const row of $("labels").children)
      row.hidden = !(row.textContent + row.dataset.name)
        .toLocaleLowerCase()
        .includes(query);
  });
  $("niivue-canvas").addEventListener("pointerup", () => {
    syncWindowPreset();
    syncWindowNumbers(true);
    scheduleSaveView();
  });
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
  window.addEventListener("pagehide", () => {
    clearTimeout(state.taskTick);
    state.taskTick = null;
    saveView();
    cancelUpload();
  });
  document.addEventListener("visibilitychange", () => {
    drawTaskProgress();
    if (!document.hidden) schedulePoll();
    else saveView();
  });
  window.addEventListener("pageshow", () => {
    drawTaskProgress();
    if (state.authenticated) schedulePoll();
  });
  window.addEventListener("online", () => {
    if (state.authenticated) schedulePoll();
  });
  const loginUrl = new URL(location.href);
  const oauthFailed = loginUrl.searchParams.get("error") === "oauth";
  const oauthError = "GitHub 登录未完成，请重试。";
  if (oauthFailed) {
    loginUrl.searchParams.delete("error");
    history.replaceState({}, "", loginUrl);
  }
  updateSubmit();
  connectWorkspace().then(() => {
    if (oauthFailed && state.ready) showError("session-error", oauthError);
  });
})();
