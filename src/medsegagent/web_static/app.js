/* MedSegAgent browser client. No credentials or image contents are persisted here. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const terminal = new Set(['completed', 'succeeded', 'success', 'failed', 'canceled', 'cancelled', 'rejected']);
  const success = new Set(['completed', 'succeeded', 'success']);
  const statusNames = {submitted:'等待处理', queued:'排队中', pending:'排队中', working:'正在分割', running:'正在分割', routing:'选择工具', validating:'校验影像', completed:'分割完成', succeeded:'分割完成', success:'分割完成', failed:'任务失败', canceled:'任务已取消', cancelled:'任务已取消', rejected:'请求被拒绝', canceling:'正在取消', cancelling:'正在取消'};
  const palette = [[238,135,77],[87,182,237],[194,137,239],[93,199,162],[236,109,145],[235,197,92],[113,157,239],[214,178,147]];
  const state = {authenticated:false, epoch:0, upload:null, tasks:[], selected:null, labels:[], visibleLabels:new Set(), maxUpload:0, uploading:false, submitting:false, pendingRequest:null, poll:null, viewer:null, viewerInitializing:null, viewerKey:null, resultKey:null, viewerWanted:null, viewerQueue:Promise.resolve(), uploadXHR:null, activeRequests:new Set()};
  const statusOf = task => typeof task.status === 'string' ? task.status : task.status?.state || 'queued';
  const size = bytes => bytes >= 1073741824 ? `${(bytes / 1073741824).toFixed(1)} GiB` : `${(bytes / 1048576).toFixed(1)} MiB`;
  const errorMessage = value => typeof value === 'string' ? value : Array.isArray(value) ? value.map(x => x.msg || '请求参数不正确').join('；') : value?.message || value?.detail || value?.code || '请求未完成，请重试。';
  const showError = (id, value) => { $(id).textContent = value ? errorMessage(value) : ''; $(id).hidden = !value; };
  const taskURL = id => `/api/tasks/${encodeURIComponent(id)}`;
  const uploadURL = id => `/api/uploads/${encodeURIComponent(id)}/file`;
  const updateSubmit = () => { $('submit').disabled = !state.upload || state.uploading || state.submitting || !state.authenticated; };

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    state.activeRequests.add(controller);
    try {
      const response = await fetch(path, {credentials:'same-origin', cache:'no-store', ...options, signal:controller.signal, headers:{...(options.body ? {'Content-Type':'application/json'} : {}), ...options.headers}});
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (response.status === 401 && path !== '/api/session') lockWorkspace();
        const err = new Error(errorMessage(data.detail || data.error || data.message || `请求失败 (${response.status})`));
        err.status = response.status;
        throw err;
      }
      return data;
    } catch (err) {
      if (err.name === 'AbortError') throw new Error('连接超时。任务可能仍在运行，请刷新任务记录确认。');
      if (err instanceof TypeError) throw new Error('无法连接服务器，连接恢复后会继续读取任务。');
      throw err;
    } finally { clearTimeout(timeout); state.activeRequests.delete(controller); }
  }

  function clearViewer() {
    if (state.viewer?.gl) {
      for (const volume of [...state.viewer.volumes]) state.viewer.removeVolume(volume);
      state.viewer.drawScene();
    }
    state.viewerKey = null; state.resultKey = null;
    $('viewer-empty').hidden = false;
    $('result-panel').hidden = true;
  }

  function lockWorkspace() {
    state.authenticated = false;
    state.epoch += 1;
    clearTimeout(state.poll);
    for (const controller of state.activeRequests) controller.abort();
    state.uploadXHR?.abort();
    state.upload = null; state.selected = null; state.tasks = []; state.labels = []; state.visibleLabels.clear();
    state.viewerWanted = null; state.uploading = false; state.submitting = false; state.pendingRequest = null;
    $('workspace').hidden = true; $('logout').hidden = true;
    $('task-list').replaceChildren(); $('labels').replaceChildren();
    $('request').reset(); $('file-name').textContent = '选择或拖入 NIfTI'; $('file-meta').hidden = true;
    $('upload-progress').hidden = true; $('task-status').hidden = true;
    $('viewer-name').textContent = '原图与分割叠加';
    clearViewer(); updateSubmit();
    if (!$('login-dialog').open) $('login-dialog').showModal();
  }

  async function openWorkspace() {
    state.authenticated = true;
    state.epoch += 1;
    $('login-dialog').close(); $('workspace').hidden = false; $('logout').hidden = false;
    showError('login-error', '');
    const config = await api('/api/config');
    state.maxUpload = Number(config.max_upload_bytes) || 0;
    if (!state.maxUpload) throw new Error('服务器未提供上传大小限制，请检查服务配置。');
    $('file-help').textContent = `.nii / .nii.gz · 最大 ${size(state.maxUpload)}`;
    await refreshTasks();
    const requested = new URL(location.href).searchParams.get('task');
    if (requested) await selectTask(requested).catch(err => showError('connection-note', err.message));
    else if (state.tasks.length) await selectTask(state.tasks[0].id);
    schedulePoll();
  }

  function uploadFile(file) {
    if (!file || state.uploading || state.submitting) return;
    state.upload = null; state.pendingRequest = null; updateSubmit();
    $('file-name').textContent = file.name; $('file-meta').hidden = true;
    showError('form-error', '');
    if (!/\.nii(?:\.gz)?$/i.test(file.name)) { showError('form-error', '请选择 .nii 或 .nii.gz 格式的 3D 影像。'); return; }
    if (!file.size) { showError('form-error', '文件为空，请选择完整的 NIfTI 影像。'); return; }
    if (!state.maxUpload || file.size > state.maxUpload) { showError('form-error', `文件超过上传限制（${size(state.maxUpload)}）。请使用更小的影像或通过本地 CLI 处理。`); return; }
    const epoch = state.epoch;
    state.upload = null; state.uploading = true; state.pendingRequest = null;
    $('file-name').textContent = file.name; $('file-meta').hidden = true;
    $('upload-progress').hidden = false; $('upload-bar').value = 0;
    $('upload-status').textContent = '正在上传…'; updateSubmit();
    const xhr = new XMLHttpRequest(); state.uploadXHR = xhr;
    xhr.open('POST', '/api/uploads'); xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    // ASCII-safe header; the server decodes percent-encoding before validating the name.
    xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name));
    xhr.timeout = 900000;
    xhr.upload.onprogress = e => { if (e.lengthComputable) { const n = Math.round(e.loaded / e.total * 100); $('upload-bar').value = n; $('upload-status').textContent = n === 100 ? '上传完成，正在校验影像…' : `正在上传 ${n}%`; } };
    xhr.onload = () => {
      if (epoch !== state.epoch) return;
      let data; try { data = JSON.parse(xhr.responseText); } catch { data = {}; }
      if (xhr.status === 401) { lockWorkspace(); return; }
      if (xhr.status < 200 || xhr.status >= 300 || !data.id) { showError('form-error', data.detail || data.error || `上传失败 (${xhr.status})`); $('upload-progress').hidden = true; return; }
      state.upload = data;
      $('upload-status').textContent = '影像已校验'; $('upload-bar').value = 100;
      $('file-meta').textContent = [size(data.size || file.size), data.shape?.join(' × '), data.spacing ? `${data.spacing.map(x => Number(x).toFixed(2)).join(' × ')} mm` : ''].filter(Boolean).join(' / ');
      $('file-meta').hidden = false;
      state.selected = null; $('task-status').hidden = true; renderHistory();
      const url = new URL(location.href); url.searchParams.delete('task'); history.replaceState({}, '', url);
      $('viewer-name').textContent = data.name || file.name;
      queueViewer({key:`upload:${data.id}`, uploadID:data.id, name:data.name || file.name});
    };
    xhr.onerror = () => { if (epoch === state.epoch) { showError('form-error', '上传连接中断，请重新选择影像。'); $('upload-progress').hidden = true; } };
    xhr.ontimeout = () => { if (epoch === state.epoch) { showError('form-error', '上传超时，请检查网络后重试。'); $('upload-progress').hidden = true; } };
    xhr.onloadend = () => { if (epoch === state.epoch) { state.uploading = false; state.uploadXHR = null; updateSubmit(); } };
    xhr.send(file);
  }

  async function refreshTasks() {
    const epoch = state.epoch;
    const data = await api('/api/tasks');
    if (epoch !== state.epoch) return;
    state.tasks = Array.isArray(data) ? data : data.tasks || data.items || [];
    renderHistory();
  }

  function renderHistory() {
    $('task-list').replaceChildren();
    $('history-empty').hidden = state.tasks.length > 0;
    for (const task of state.tasks) {
      const li = document.createElement('li'), button = document.createElement('button');
      button.type = 'button'; button.className = 'task-item'; button.setAttribute('aria-current', String(state.selected?.id === task.id));
      const title = document.createElement('span'); title.className = 'task-item-title'; title.textContent = task.text || task.input?.text || task.name || task.id;
      const meta = document.createElement('span'); meta.className = 'task-item-meta';
      const status = document.createElement('span'); status.textContent = statusNames[statusOf(task)] || statusOf(task); if (statusOf(task) === 'failed') status.className = 'failed';
      const date = document.createElement('span'), time = task.created_at || task.createdAt;
      if (time) { const d = new Date(typeof time === 'number' && time < 1e12 ? time * 1000 : time); date.textContent = Number.isNaN(d.valueOf()) ? '' : d.toLocaleString('zh-CN', {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}); }
      meta.append(status, date); button.append(title, meta); button.addEventListener('click', () => selectTask(task.id).catch(err => showError('connection-note', err.message)));
      li.append(button); $('task-list').append(li);
    }
  }

  async function selectTask(id) {
    const epoch = state.epoch;
    // Mark the requested ID before fetching so a slower previous request cannot replace it.
    state.selected = {id};
    const task = await api(taskURL(id));
    if (epoch !== state.epoch || state.selected?.id !== id) return;
    state.selected = task;
    const url = new URL(location.href); url.searchParams.set('task', id); history.replaceState({}, '', url);
    renderTask(task); renderHistory();
  }

  function renderTask(task) {
    const status = statusOf(task), complete = success.has(status);
    $('task-status').hidden = false; $('status-title').textContent = statusNames[status] || status;
    const progress = task.progress;
    $('status-detail').textContent = typeof progress === 'string' ? progress : progress?.message || task.stage || task.text || '';
    const percent = typeof progress === 'number' ? progress : progress?.percent;
    $('task-progress').hidden = terminal.has(status);
    if (Number.isFinite(percent)) $('task-progress').value = Math.max(0, Math.min(100, percent)); else $('task-progress').removeAttribute('value');
    $('cancel-task').hidden = terminal.has(status); $('cancel-task').disabled = ['canceling','cancelling'].includes(status);
    showError('task-error', task.error || (status === 'failed' ? '任务未完成，请查看错误信息后新建任务重试。' : ''));
    $('viewer-name').textContent = task.upload_name || task.input?.name || task.text || `任务 ${task.id}`;
    const uploadID = task.upload_id || task.input?.upload_id;
    if (uploadID) queueViewer({key:`${task.id}:${complete ? 'result' : 'source'}`, taskID:task.id, uploadID, name:task.upload_name || 'source.nii', result:complete ? task.result : null, completed:complete});
    else if (state.viewerWanted?.taskID !== task.id) { state.viewerWanted = null; clearViewer(); showError('viewer-error', '此任务没有可通过 Web 读取的源影像。'); }
  }

  function schedulePoll() {
    clearTimeout(state.poll);
    if (!state.authenticated) return;
    state.poll = setTimeout(async () => {
      const epoch = state.epoch;
      try {
        const selectedID = state.selected?.id;
        if (selectedID) {
          const task = await api(taskURL(selectedID));
          if (epoch === state.epoch && state.selected?.id === selectedID) { state.selected = task; renderTask(task); }
        }
        if (epoch === state.epoch) await refreshTasks();
        if (epoch === state.epoch) showError('connection-note', '');
      } catch (err) { if (state.authenticated) showError('connection-note', err.message); }
      finally { if (epoch === state.epoch) schedulePoll(); }
    }, document.hidden ? 10000 : state.selected && !terminal.has(statusOf(state.selected)) ? 2000 : 6000);
  }

  async function getViewer() {
    if (state.viewer) return state.viewer;
    if (state.viewerInitializing) return state.viewerInitializing;
    state.viewerInitializing = initializeViewer();
    try { return await state.viewerInitializing; }
    finally { state.viewerInitializing = null; }
  }

  async function initializeViewer() {
    if (!window.niivue?.Niivue) throw new Error('查看器脚本未能加载，请刷新页面。');
    const viewer = new window.niivue.Niivue({backColor:[0,0,0,1], crosshairColor:[0.55,0.7,1,0.75], crosshairWidth:0.7, isColorbar:false, isOrientCube:true, show3Dcrosshair:true, isRadiologicalConvention:true, dragAndDropEnabled:false, isNearestInterpolation:true, multiplanarShowRender:window.niivue.SHOW_RENDER.ALWAYS, logging:false});
    await viewer.attachToCanvas($('niivue-canvas'));
    if (!viewer.gl) throw new Error('当前浏览器无法初始化 WebGL2。请启用硬件加速，或使用支持 WebGL2 的浏览器。');
    viewer.setSliceType(window.niivue.SLICE_TYPE.MULTIPLANAR);
    viewer.onLocationChange = location => { if (Array.isArray(location.mm) || ArrayBuffer.isView(location.mm)) $('location').textContent = `位置 ${Array.from(location.mm).slice(0,3).map(x => Number(x).toFixed(1)).join(', ')} mm`; };
    state.viewer = viewer;
    return viewer;
  }

  function queueViewer(request) {
    if (state.viewerWanted?.key === request.key || state.viewerKey === request.key) return;
    state.viewerWanted = request;
    const epoch = state.epoch;
    $('result-panel').hidden = true; $('viewer-indicator').hidden = false; showError('viewer-error', '');
    $('niivue-canvas').style.visibility = 'hidden';
    // NiiVue loadVolumes mutates its scene; serialize changes and discard obsolete requests.
    state.viewerQueue = state.viewerQueue.catch(() => {}).then(async () => {
      if (epoch !== state.epoch || state.viewerWanted !== request) return;
      try {
        if (request.completed && state.resultKey !== request.key) {
          const result = request.result || await api(`${taskURL(request.taskID)}/files/result.json`);
          if (epoch !== state.epoch || state.viewerWanted !== request) return;
          // Downloads remain available even if the browser cannot initialize WebGL2.
          showResult(result, request.taskID, false); state.resultKey = request.key;
        } else if (request.completed) $('result-panel').hidden = false;
        const viewer = await getViewer();
        const volumes = [{url:uploadURL(request.uploadID),name:request.name || 'source.nii',colormap:'gray',opacity:1,colorbarVisible:false}];
        if (request.completed) volumes.push({url:`${taskURL(request.taskID)}/files/segmentation.nii.gz`,name:'segmentation.nii.gz',colormap:'gray',opacity:0,cal_min:0,cal_max:Math.max(1,...state.labels.map(label=>label.id)),colorbarVisible:false});
        await viewer.loadVolumes(volumes);
        if (epoch !== state.epoch || state.viewerWanted !== request) { clearViewer(); return; }
        if (viewer.volumes.length !== volumes.length) throw new Error('影像未完整载入，请检查任务文件是否已过期后重试。');
        $('viewer-empty').hidden = true;
        if (request.completed) {
          updateOverlay();
          focusForeground(viewer);
        }
        viewer.drawScene(); state.viewerKey = request.key; $('niivue-canvas').style.visibility = 'visible';
      } catch (err) {
        if (epoch === state.epoch && state.viewerWanted === request) { state.viewerWanted = null; showError('viewer-error', `影像显示失败：${err.message}。任务文件仍可在完成后下载。`); }
      } finally { if (epoch === state.epoch && (!state.viewerWanted || state.viewerWanted === request)) $('viewer-indicator').hidden = true; }
    });
  }

  function showResult(result, taskID, apply = true) {
    state.labels = (result.labels || []).filter(label => Number.isInteger(Number(label.id)) && Number(label.id) > 0 && Number(label.id) <= 65535).map(label => ({...label,id:Number(label.id)}));
    state.visibleLabels = new Set(state.labels.map(label => label.id));
    $('labels').replaceChildren(); $('label-count').textContent = `(${state.labels.length})`;
    for (const [index, label] of state.labels.entries()) {
      const wrap = document.createElement('label'); wrap.className = 'label-option';
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = true; checkbox.dataset.label = String(label.id);
      checkbox.addEventListener('change', () => { checkbox.checked ? state.visibleLabels.add(label.id) : state.visibleLabels.delete(label.id); updateOverlay(); });
      const swatch = document.createElement('span'); swatch.className = 'label-swatch'; swatch.style.backgroundColor = `rgb(${palette[index % palette.length].join(',')})`;
      const name = document.createElement('span'); name.className = 'label-name'; name.textContent = label.name || String(label.id);
      const count = document.createElement('span'); count.className = 'label-voxels'; count.textContent = Number.isFinite(label.voxels) ? `${label.voxels.toLocaleString()} vox` : '';
      wrap.append(checkbox, swatch, name, count); $('labels').append(wrap);
    }
    $('download-mask').href = `${taskURL(taskID)}/files/segmentation.nii.gz`;
    $('download-result').href = `${taskURL(taskID)}/files/result.json`;
    const elapsed = result.duration_seconds ?? result.elapsed_seconds ?? result.runtime_seconds;
    $('result-summary').textContent = [result.tool || result.task || '', Number.isFinite(elapsed) ? `推理用时 ${elapsed.toFixed(1)} 秒` : '', result.detection_status === 'no_target_detected' ? '未检出目标，不能据此排除病变' : '请核查分割边界与标签'].filter(Boolean).join(' / ');
    $('result-panel').hidden = false;
    if (apply) updateOverlay();
  }

  function updateOverlay() {
    const viewer = state.viewer, overlay = viewer?.volumes[1];
    if (!overlay) return;
    const map = {I:[0],R:[0],G:[0],B:[0],A:[0],labels:['background']};
    for (const [index, label] of state.labels.entries()) {
      const color = palette[index % palette.length];
      map.I.push(label.id); map.R.push(color[0]); map.G.push(color[1]); map.B.push(color[2]); map.A.push(state.visibleLabels.has(label.id) ? 255 : 0); map.labels.push(label.name || String(label.id));
    }
    // Discrete label lookup preserves IDs; visibility never modifies voxel data.
    overlay.setColormapLabel(map); viewer.setOpacity(1,Number($('opacity').value)/100); viewer.updateGLVolume();
  }

  function focusForeground(viewer) {
    const overlay = viewer.volumes[1], image = overlay?.img, affine = overlay?.hdr?.affine;
    if (!image || !affine?.[0] || !state.labels.length) return;
    const label = [...state.labels].sort((a,b) => (b.voxels || 0) - (a.voxels || 0))[0].id;
    const nx = overlay.hdr.dims[1], ny = overlay.hdr.dims[2], stride = Math.max(1,Math.ceil(image.length/1000000));
    let count=0,x=0,y=0,z=0;
    for (let i=0;i<image.length;i+=stride) if (image[i] === label) { count++; x+=i%nx; y+=Math.floor(i/nx)%ny; z+=Math.floor(i/(nx*ny)); }
    if (!count) return;
    const p=[x/count,y/count,z/count,1];
    const mm=affine.slice(0,3).map(row => row.reduce((sum,value,i) => sum+value*p[i],0));
    viewer.scene.crosshairPos = viewer.mm2frac(mm); viewer.drawScene();
  }

  $('login-dialog').addEventListener('cancel', event => event.preventDefault());
  $('login-form').addEventListener('submit', async event => {
    event.preventDefault(); $('login-submit').disabled = true; showError('login-error','');
    try { await api('/api/session',{method:'POST',body:JSON.stringify({token:$('access-token').value.trim()})}); $('access-token').value=''; await openWorkspace(); }
    catch (err) { lockWorkspace(); showError('login-error',err.message); }
    finally { $('login-submit').disabled = false; }
  });
  $('logout').addEventListener('click', async () => { try { await api('/api/session',{method:'DELETE'}); const url = new URL(location.href); url.searchParams.delete('task'); history.replaceState({},'',url); lockWorkspace(); } catch(err) { showError('connection-note',err.message); } });
  $('file').addEventListener('change', event => uploadFile(event.target.files[0]));
  for (const name of ['dragenter','dragover']) $('drop-zone').addEventListener(name,event => {event.preventDefault();$('drop-zone').classList.add('dragging');});
  for (const name of ['dragleave','drop']) $('drop-zone').addEventListener(name,event => {event.preventDefault();$('drop-zone').classList.remove('dragging');});
  $('drop-zone').addEventListener('drop',event => {if(event.dataTransfer.files.length!==1) showError('form-error','请一次上传一张 3D 影像。'); else uploadFile(event.dataTransfer.files[0]);});
  $('request').addEventListener('submit', async event => {
    event.preventDefault(); if (!state.upload || state.uploading || state.submitting) return;
    const text=$('instruction').value.trim(), modality=document.querySelector('input[name=modality]:checked').value;
    if (!text) {showError('form-error','请填写需要分割的目标。');return;}
    const signature=JSON.stringify([state.upload.id,text,modality]);
    if (state.pendingRequest?.signature !== signature) state.pendingRequest={signature,id:crypto.randomUUID()};
    state.submitting=true; updateSubmit(); showError('form-error','');
    try {
      const task=await api('/api/tasks',{method:'POST',body:JSON.stringify({upload_id:state.upload.id,text,modality,message_id:state.pendingRequest.id})});
      state.pendingRequest=null; await refreshTasks(); await selectTask(task.id); schedulePoll();
    } catch(err) {showError('form-error',err.message);}
    finally {state.submitting=false;updateSubmit();}
  });
  $('refresh-tasks').addEventListener('click',async () => { try {await refreshTasks();if(state.selected?.id)await selectTask(state.selected.id);showError('connection-note','');}catch(err){showError('connection-note',err.message);} });
  $('cancel-task').addEventListener('click',async () => {
    const id=state.selected?.id;if(!id)return;$('cancel-task').disabled=true;
    try {await api(`${taskURL(id)}/cancel`,{method:'POST'});if(state.selected?.id===id)await selectTask(id);await refreshTasks();}
    catch(err){showError('task-error',err.message);}
    finally{$('cancel-task').disabled=false;}
  });
  document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click',async () => {
    try {const viewer=await getViewer();const types={multiplanar:'MULTIPLANAR',axial:'AXIAL',coronal:'CORONAL',sagittal:'SAGITTAL',render:'RENDER'};viewer.setSliceType(window.niivue.SLICE_TYPE[types[button.dataset.view]]);document.querySelectorAll('[data-view]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));}
    catch(err){showError('viewer-error',err.message);}
  }));
  $('reset-view').addEventListener('click',()=>{const viewer=state.viewer;if(viewer?.volumes.length){viewer.scene.crosshairPos=[0.5,0.5,0.5];viewer.scene.pan2Dxyzmm=[0,0,0,1];viewer.resetBriCon();focusForeground(viewer);viewer.drawScene();}});
  $('opacity').addEventListener('input',()=>{$('opacity-value').value=`${$('opacity').value}%`;if(state.viewer?.volumes[1])state.viewer.setOpacity(1,Number($('opacity').value)/100);});
  for(const [id,visible] of [['labels-all',true],['labels-none',false]]) $(id).addEventListener('click',()=>{state.visibleLabels=new Set(visible?state.labels.map(label=>label.id):[]);$('labels').querySelectorAll('input').forEach(input=>{input.checked=visible;});updateOverlay();});
  $('niivue-canvas').addEventListener('webglcontextlost',event=>{event.preventDefault();showError('viewer-error','浏览器显存不足或图形上下文已丢失。任务仍保存在服务器，请刷新页面或使用更小的影像。');});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)schedulePoll();});
  window.addEventListener('online',()=>{if(state.authenticated)schedulePoll();});
  api('/api/session').then(session=>{if(session.authenticated===false)lockWorkspace();else return openWorkspace();}).catch(err=>{lockWorkspace();if(err.status!==401)showError('login-error',err.message);});
})();
