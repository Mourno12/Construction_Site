/**
 * app.js
 * Dashboard logic for SiteGuard. Handles:
 *  - Auth gate: register / log in / log out (JWT stored in localStorage)
 *  - Mode switching (webcam / upload video / upload image)
 *  - Capturing webcam frames and streaming them over WebSocket
 *  - Uploading video/image files and kicking off server-side processing
 *  - Rendering incoming detection results: annotated frame, stats,
 *    compliance meter, alert feed (with evidence thumbnails), worker table
 *  - Session History & Reports: past sessions, violation evidence gallery,
 *    and CSV/JSON export
 */

(() => {
  'use strict';

  const AUTH_STORAGE_KEY = 'siteguard_auth'; // { token, username, role }

  const els = {
    authGate: document.getElementById('authGate'),
    appShell: document.getElementById('appShell'),
    authTabs: Array.from(document.querySelectorAll('.auth-tab')),
    authForms: Array.from(document.querySelectorAll('.auth-form')),
    loginForm: document.getElementById('loginForm'),
    loginError: document.getElementById('loginError'),
    registerForm: document.getElementById('registerForm'),
    registerError: document.getElementById('registerError'),

    usernameText: document.getElementById('usernameText'),
    roleBadge: document.getElementById('roleBadge'),
    logoutBtn: document.getElementById('logoutBtn'),

    engineDot: document.getElementById('engineDot'),
    engineStatusText: document.getElementById('engineStatusText'),
    clockChip: document.getElementById('clockChip'),

    modeBtns: Array.from(document.querySelectorAll('.mode-btn')),
    sourcePanes: Array.from(document.querySelectorAll('.source-pane')),

    viewfinder: document.getElementById('viewfinder'),
    feedImage: document.getElementById('feedImage'),
    feedVideoRaw: document.getElementById('feedVideoRaw'),
    vfEmpty: document.getElementById('vfEmpty'),
    vfBadgeRow: document.getElementById('vfBadgeRow'),
    fpsBadge: document.getElementById('fpsBadge'),
    demoBadge: document.getElementById('demoBadge'),

    webcamStartBtn: document.getElementById('webcamStartBtn'),
    webcamStartBtnLabel: document.getElementById('webcamStartBtnLabel'),
    webcamStopBtn: document.getElementById('webcamStopBtn'),

    videoDropzone: document.getElementById('videoDropzone'),
    videoDropzoneLabel: document.getElementById('videoDropzoneLabel'),
    videoInput: document.getElementById('videoInput'),
    videoStartBtn: document.getElementById('videoStartBtn'),
    videoStopBtn: document.getElementById('videoStopBtn'),

    imageDropzone: document.getElementById('imageDropzone'),
    imageDropzoneLabel: document.getElementById('imageDropzoneLabel'),
    imageInput: document.getElementById('imageInput'),
    imageStartBtn: document.getElementById('imageStartBtn'),

    statTotal: document.getElementById('statTotal'),
    statSafe: document.getElementById('statSafe'),
    statUnsafe: document.getElementById('statUnsafe'),
    statFps: document.getElementById('statFps'),
    complianceValue: document.getElementById('complianceValue'),
    complianceFill: document.getElementById('complianceFill'),

    alertCount: document.getElementById('alertCount'),
    alertFeed: document.getElementById('alertFeed'),

    workerTableBody: document.getElementById('workerTableBody'),
    trackedCountHint: document.getElementById('trackedCountHint'),
    exportWorkersCsvBtn: document.getElementById('exportWorkersCsvBtn'),
    exportWorkersJsonBtn: document.getElementById('exportWorkersJsonBtn'),

    sessionHistoryBody: document.getElementById('sessionHistoryBody'),
    historyScopeSwitch: document.getElementById('historyScopeSwitch'),
    violationGallery: document.getElementById('violationGallery'),
    exportAllViolationsCsvBtn: document.getElementById('exportAllViolationsCsvBtn'),
    exportAllViolationsJsonBtn: document.getElementById('exportAllViolationsJsonBtn'),
    refreshHistoryBtn: document.getElementById('refreshHistoryBtn'),

    toastStack: document.getElementById('toastStack'),
    captureCanvas: document.getElementById('captureCanvas'),
  };

  const state = {
    auth: null, // { token, username, email, role }
    mode: 'webcam',
    sessionId: null,
    ws: null,
    webcamStream: null,
    webcamCaptureTimer: null,
    videoFile: null,
    videoServerPath: null,
    imageFile: null,
    workers: new Map(), // trackId -> row data
    alertElementsByTrackId: new Map(), // trackId -> <li> (for thumbnail injection)
    alertCount: 0,
    demoMode: null,
    engineReady: false,
    lastToastMessage: null,
    lastToastAt: 0,
    historyScope: 'all', // 'all' | 'mine' - only meaningful for admins; operators are always scoped server-side
  };

  const WEBCAM_CAPTURE_INTERVAL_MS = 130; // ~7-8 fps upload cadence; server processes each frame
  const MAX_TRACKED_WORKERS = 300; // safety cap - a flaky fallback tracker can mint many spurious IDs

  // ==================================================================
  // Auth: storage helpers
  // ==================================================================
  function loadAuth() {
    try {
      const raw = localStorage.getItem(AUTH_STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  function saveAuth(auth) {
    state.auth = auth;
    localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(auth));
  }

  function clearAuth() {
    state.auth = null;
    localStorage.removeItem(AUTH_STORAGE_KEY);
  }

  function authHeaders() {
    return state.auth ? { Authorization: `Bearer ${state.auth.token}` } : {};
  }

  async function authedFetch(url, options = {}) {
    const headers = { ...(options.headers || {}), ...authHeaders() };
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401) {
      clearAuth();
      showAuthGate();
      throw new Error('Session expired. Please log in again.');
    }
    return res;
  }

  // ==================================================================
  // Auth: gate UI
  // ==================================================================
  function showAuthGate() {
    els.authGate.hidden = false;
    els.appShell.hidden = true;
    stopEverything();
  }

  function showApp() {
    els.authGate.hidden = true;
    els.appShell.hidden = false;
    els.usernameText.textContent = state.auth.username;
    els.roleBadge.textContent = state.auth.role;
    els.historyScopeSwitch.hidden = state.auth.role !== 'admin';
    state.historyScope = state.auth.role === 'admin' ? 'all' : 'mine';
    checkHealth();
    loadHistory();
  }

  els.authTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      els.authTabs.forEach((t) => {
        t.classList.toggle('active', t === tab);
        t.setAttribute('aria-selected', t === tab ? 'true' : 'false');
      });
      els.authForms.forEach((f) => f.classList.toggle('active', f.dataset.form === tab.dataset.tab));
    });
  });

  function setAuthError(el, message) {
    if (!message) {
      el.hidden = true;
      el.textContent = '';
      return;
    }
    el.hidden = false;
    el.textContent = message;
  }

  els.loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    setAuthError(els.loginError, null);
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Login failed');
      saveAuth({ token: data.token, username: data.username, email: data.email, role: data.role });
      showApp();
    } catch (err) {
      setAuthError(els.loginError, err.message);
    }
  });

  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const PASSWORD_RE = /^(?=.*[A-Za-z])(?=.*\d).{8,}$/;

  els.registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    setAuthError(els.registerError, null);
    const username = document.getElementById('registerUsername').value.trim();
    const email = document.getElementById('registerEmail').value.trim();
    const password = document.getElementById('registerPassword').value;

    if (!EMAIL_RE.test(email)) {
      setAuthError(els.registerError, 'Please enter a valid email address.');
      return;
    }
    if (!PASSWORD_RE.test(password)) {
      setAuthError(els.registerError, 'Password must be at least 8 characters and include a letter and a number.');
      return;
    }

    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Registration failed');
      saveAuth({ token: data.token, username: data.username, email: data.email, role: data.role });
      showApp();
    } catch (err) {
      setAuthError(els.registerError, err.message);
    }
  });

  els.logoutBtn.addEventListener('click', async () => {
    try { await authedFetch('/api/auth/logout', { method: 'POST' }); } catch {}
    clearAuth();
    showAuthGate();
  });

  async function initAuth() {
    const saved = loadAuth();
    if (!saved) {
      showAuthGate();
      return;
    }
    state.auth = saved;
    try {
      const res = await fetch('/api/auth/me', { headers: authHeaders() });
      if (!res.ok) throw new Error('invalid session');
      showApp();
    } catch {
      clearAuth();
      showAuthGate();
    }
  }

  // ==================================================================
  // Clock
  // ==================================================================
  function tickClock() {
    els.clockChip.textContent = new Date().toLocaleTimeString();
  }
  setInterval(tickClock, 1000);
  tickClock();

  // ==================================================================
  // Engine status
  // ==================================================================
  function setEngineStatus(kind, text) {
    els.engineDot.className = 'dot ' + kind;
    els.engineStatusText.textContent = text;
    state.engineReady = kind === 'dot-live';
    updateStartButtonsAvailability();
  }

  function updateStartButtonsAvailability() {
    const ready = state.engineReady;
    els.webcamStartBtn.disabled = !ready;
    els.webcamStartBtnLabel.textContent = ready ? 'Start Webcam Monitoring' : 'Waiting for engine…';
    if (state.videoFile) els.videoStartBtn.disabled = !ready;
    if (state.imageFile) els.imageStartBtn.disabled = !ready;
  }

  function checkHealth() {
    fetch('/api/health').then((r) => r.json()).then((data) => {
      if (data.engineReady) {
        setEngineStatus('dot-live', data.demoMode ? 'Engine ready (demo mode)' : 'Engine ready');
      } else {
        setEngineStatus('dot-pending', 'Loading models…');
      }
    }).catch(() => setEngineStatus('dot-error', 'Backend unreachable'));
  }

  // ==================================================================
  // Mode switching
  // ==================================================================
  els.modeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      if (mode === state.mode) return;
      stopEverything();
      state.mode = mode;
      els.modeBtns.forEach((b) => {
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
      });
      els.sourcePanes.forEach((p) => p.classList.toggle('active', p.dataset.pane === mode));
      resetViewfinder();
    });
  });

  function resetViewfinder() {
    els.viewfinder.classList.remove('is-live');
    els.feedImage.hidden = true;
    els.vfEmpty.hidden = false;
    els.vfBadgeRow.hidden = true;
  }

  function showLiveFeed() {
    els.viewfinder.classList.add('is-live');
    els.vfEmpty.hidden = true;
    els.vfBadgeRow.hidden = false;
  }

  // ==================================================================
  // WebSocket plumbing
  // ==================================================================
  function openSocket(sessionId) {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const token = encodeURIComponent(state.auth?.token || '');
    const ws = new WebSocket(`${proto}://${location.host}/ws?sessionId=${sessionId}&token=${token}`);

    ws.addEventListener('message', (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch { return; }
      handleServerMessage(msg);
    });
    ws.addEventListener('close', (evt) => {
      if (evt.code === 1008) {
        showToast('Connection closed: ' + (evt.reason || 'authentication required'));
      }
    });

    state.ws = ws;
    return ws;
  }

  function handleServerMessage(msg) {
    switch (msg.type) {
      case 'connected':
        if (msg.engineReady) {
          setEngineStatus('dot-live', msg.demoMode ? 'Engine ready (demo mode)' : 'Engine ready');
        }
        state.demoMode = msg.demoMode;
        break;
      case 'engine_ready':
        setEngineStatus('dot-live', msg.demoMode ? 'Engine ready (demo mode)' : 'Engine ready');
        state.demoMode = msg.demoMode;
        break;
      case 'result':
        renderResult(msg);
        break;
      case 'violation':
        attachViolationThumbnail(msg.violation);
        prependGalleryCard(msg.violation);
        break;
      case 'video_done':
        onVideoDone();
        break;
      case 'error':
        showToast(`Inference error: ${msg.message}`);
        break;
      default:
        break;
    }
  }

  // ==================================================================
  // Rendering a detection result
  // ==================================================================
  function renderResult(msg) {
    if (msg.annotated) {
      els.feedImage.src = msg.annotated;
      els.feedImage.hidden = false;
      showLiveFeed();
    }

    const stats = msg.stats || {};
    els.fpsBadge.textContent = `${(stats.fps || 0).toFixed(1)} FPS`;
    els.demoBadge.hidden = !stats.demoMode;

    els.statTotal.textContent = stats.totalWorkers ?? 0;
    els.statSafe.textContent = stats.safeCount ?? 0;
    els.statUnsafe.textContent = stats.unsafeCount ?? 0;
    els.statFps.innerHTML = `${(stats.fps || 0).toFixed(1)}<small> fps</small>`;

    const rate = stats.complianceRatePct ?? 100;
    els.complianceValue.textContent = `${rate}%`;
    els.complianceFill.style.width = `${rate}%`;
    els.complianceFill.classList.remove('warn', 'danger');
    if (rate < 50) els.complianceFill.classList.add('danger');
    else if (rate < 85) els.complianceFill.classList.add('warn');

    updateWorkerTable(msg.detections || []);

    els.exportWorkersCsvBtn.disabled = !state.sessionId;
    els.exportWorkersJsonBtn.disabled = !state.sessionId;

    (stats.newAlerts || []).forEach((trackId) => {
      const w = state.workers.get(trackId);
      addAlert(trackId, w);
      showToast(`Worker #${trackId} flagged: missing PPE`);
    });
  }

  function formatDuration(seconds) {
    if (!seconds || seconds < 0) seconds = 0;
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  function evictStaleWorkers() {
    if (state.workers.size <= MAX_TRACKED_WORKERS) return;
    const entries = Array.from(state.workers.entries()).sort((a, b) => (a[1].lastSeen || 0) - (b[1].lastSeen || 0));
    const toRemove = entries.length - MAX_TRACKED_WORKERS;
    for (let i = 0; i < toRemove; i++) {
      const trackId = entries[i][0];
      state.workers.delete(trackId);
      state.alertElementsByTrackId.delete(trackId);
    }
  }

  function updateWorkerTable(detections) {
    detections.forEach((d) => state.workers.set(d.trackId, d));
    evictStaleWorkers(); // safety cap - a flaky fallback tracker can otherwise mint unbounded IDs

    const rows = Array.from(state.workers.values()).sort((a, b) => a.trackId - b.trackId);
    els.trackedCountHint.textContent = `${rows.length} tracked`;

    if (rows.length === 0) {
      els.workerTableBody.innerHTML = '<tr class="table-empty-row"><td colspan="8">Worker tracking data will appear here once monitoring starts.</td></tr>';
      return;
    }

    els.workerTableBody.innerHTML = rows.map((w) => {
      const flash = w.isNewAlert ? ' row-flash' : '';
      const rowClass = (w.compliant ? '' : 'row-violation') + flash;
      const firstSeenLabel = w.firstSeen ? new Date(w.firstSeen * 1000).toLocaleTimeString() : '—';
      const durationLabel = w.firstSeen && w.lastSeen ? formatDuration(w.lastSeen - w.firstSeen) : '—';
      return `
        <tr class="${rowClass}">
          <td>#${w.trackId}</td>
          <td>${w.compliant
            ? '<span class="pill pill-safe">● SAFE</span>'
            : '<span class="pill pill-unsafe">● VIOLATION</span>'}</td>
          <td class="${w.helmet ? 'pill-ok' : 'pill-missing'}">${w.helmet ? 'OK' : 'MISSING'}</td>
          <td class="${w.vest ? 'pill-ok' : 'pill-missing'}">${w.vest ? 'OK' : 'MISSING'}</td>
          <td>${w.complianceRate}%</td>
          <td>${w.framesSeen}</td>
          <td>${firstSeenLabel}</td>
          <td>${durationLabel}</td>
        </tr>`;
    }).join('');
  }

  function addAlert(trackId, workerData) {
    state.alertCount += 1;
    els.alertCount.textContent = state.alertCount;

    const emptyEl = els.alertFeed.querySelector('.alert-empty');
    if (emptyEl) emptyEl.remove();

    const li = document.createElement('li');
    li.className = 'alert-item';
    const missing = [];
    if (workerData && !workerData.helmet) missing.push('helmet');
    if (workerData && !workerData.vest) missing.push('vest');
    li.innerHTML = `
      <span class="dot"></span>
      <span class="alert-text">Worker #${trackId} missing ${missing.join(' & ') || 'PPE'}</span>
      <span class="alert-time">${new Date().toLocaleTimeString()}</span>
    `;
    els.alertFeed.prepend(li);
    state.alertElementsByTrackId.set(trackId, li);

    while (els.alertFeed.children.length > 30) {
      els.alertFeed.removeChild(els.alertFeed.lastChild);
    }
  }

  function attachViolationThumbnail(violation) {
    if (!violation || !violation.snapshot) return;
    const li = state.alertElementsByTrackId.get(violation.trackId);
    if (!li || li.querySelector('.alert-thumb')) return;
    const img = document.createElement('img');
    img.className = 'alert-thumb';
    img.src = violation.snapshot;
    img.alt = `Evidence snapshot for worker #${violation.trackId}`;
    li.prepend(img);
  }

  function showToast(text) {
    const now = Date.now();
    if (text === state.lastToastMessage && now - state.lastToastAt < 3000) {
      return; // suppress rapid-fire duplicates (e.g. many frames failing the same way in a row)
    }
    state.lastToastMessage = text;
    state.lastToastAt = now;

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<strong>⚠ Safety Alert</strong><br>${text}`;
    els.toastStack.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);

    while (els.toastStack.children.length > 5) {
      els.toastStack.removeChild(els.toastStack.firstChild);
    }
  }

  // ==================================================================
  // Session History & Reports
  // ==================================================================
  async function loadHistory() {
    try {
      const mine = state.historyScope === 'mine';
      const [sessionsRes, violationsRes] = await Promise.all([
        authedFetch(`/api/logs/sessions${mine ? '?mine=true' : ''}`),
        authedFetch(`/api/logs/violations?limit=20${mine ? '&mine=true' : ''}`),
      ]);
      const sessions = await sessionsRes.json();
      const violations = await violationsRes.json();
      renderSessionHistory(sessions);
      renderViolationGallery(violations);
    } catch (e) {
      // silent - history is a nice-to-have, don't block the dashboard on it
    }
  }

  els.historyScopeSwitch.addEventListener('click', (e) => {
    const btn = e.target.closest('.mode-btn');
    if (!btn) return;
    state.historyScope = btn.dataset.scope;
    els.historyScopeSwitch.querySelectorAll('.mode-btn').forEach((b) => b.classList.toggle('active', b === btn));
    loadHistory();
  });

  function renderSessionHistory(sessions) {
    if (!sessions.length) {
      els.sessionHistoryBody.innerHTML = '<tr class="table-empty-row"><td colspan="7">No sessions yet. Run a webcam, video, or image scan to see it here.</td></tr>';
      return;
    }
    els.sessionHistoryBody.innerHTML = sessions.slice(0, 25).map((s) => {
      const started = new Date(s.startedAt).toLocaleString();
      const workers = s.summary?.totalWorkers ?? '—';
      const violations = s.summary?.violationCount ?? '—';
      const compliance = s.summary?.complianceRatePct != null ? `${s.summary.complianceRatePct}%` : '—';
      return `
        <tr>
          <td>${started}</td>
          <td>${s.username || '—'}</td>
          <td>${s.mode}${s.demoMode ? ' <span class="vf-badge demo" style="display:inline;padding:1px 6px;font-size:10px;">DEMO</span>' : ''}</td>
          <td>${workers}</td>
          <td>${violations}</td>
          <td>${compliance}</td>
          <td><button class="link-btn" data-export-session="${s.id}">Export CSV</button></td>
        </tr>`;
    }).join('');

    els.sessionHistoryBody.querySelectorAll('[data-export-session]').forEach((btn) => {
      btn.addEventListener('click', () => {
        exportUrl(`/api/logs/sessions/${btn.dataset.exportSession}/workers/export?format=csv`);
      });
    });
  }

  function violationCard(v) {
    const time = new Date(v.timestamp).toLocaleString();
    const missing = [!v.helmet && 'helmet', !v.vest && 'vest'].filter(Boolean).join(' & ') || 'PPE';
    const img = v.snapshot
      ? `<img src="${v.snapshot}" alt="Violation evidence for worker #${v.trackId}" />`
      : `<div style="aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:var(--text-faint);font-size:11px;">No snapshot</div>`;
    return `
      <li class="violation-card">
        ${img}
        <div class="violation-card-body">
          <span class="vc-title">Worker #${v.trackId} — missing ${missing}</span>
          <span class="vc-time">${time}</span>
        </div>
      </li>`;
  }

  function renderViolationGallery(violations) {
    if (!violations.length) {
      els.violationGallery.innerHTML = '<li class="alert-empty">No violations logged yet.</li>';
      return;
    }
    els.violationGallery.innerHTML = violations.map(violationCard).join('');
  }

  function prependGalleryCard(v) {
    const empty = els.violationGallery.querySelector('.alert-empty');
    if (empty) empty.remove();
    els.violationGallery.insertAdjacentHTML('afterbegin', violationCard(v));
    while (els.violationGallery.children.length > 20) {
      els.violationGallery.removeChild(els.violationGallery.lastChild);
    }
  }

  function exportUrl(path) {
    const token = encodeURIComponent(state.auth?.token || '');
    const sep = path.includes('?') ? '&' : '?';
    window.open(`${path}${sep}token=${token}`, '_blank');
  }

  els.exportAllViolationsCsvBtn.addEventListener('click', () => exportUrl('/api/logs/violations/export?format=csv&mine=true'));
  els.exportAllViolationsJsonBtn.addEventListener('click', () => exportUrl('/api/logs/violations/export?format=json&mine=true'));
  els.refreshHistoryBtn.addEventListener('click', loadHistory);
  els.exportWorkersCsvBtn.addEventListener('click', () => {
    if (state.sessionId) exportUrl(`/api/logs/sessions/${state.sessionId}/workers/export?format=csv`);
  });
  els.exportWorkersJsonBtn.addEventListener('click', () => {
    if (state.sessionId) exportUrl(`/api/logs/sessions/${state.sessionId}/workers/export?format=json`);
  });

  // ==================================================================
  // Webcam mode
  // ==================================================================
  els.webcamStartBtn.addEventListener('click', startWebcam);
  els.webcamStopBtn.addEventListener('click', stopWebcam);

  async function startWebcam() {
    try {
      state.webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 } });
    } catch (e) {
      showToast('Could not access webcam: ' + e.message);
      return;
    }

    state.sessionId = crypto.randomUUID();
    els.feedVideoRaw.srcObject = state.webcamStream;
    openSocket(state.sessionId);

    els.webcamStartBtn.hidden = true;
    els.webcamStopBtn.hidden = false;

    const canvas = els.captureCanvas;
    const video = els.feedVideoRaw;

    state.webcamCaptureTimer = setInterval(() => {
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
      if (!video.videoWidth) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      state.ws.send(JSON.stringify({ type: 'frame', mode: 'webcam', image: dataUrl }));
    }, WEBCAM_CAPTURE_INTERVAL_MS);
  }

  function stopWebcam() {
    if (state.webcamCaptureTimer) clearInterval(state.webcamCaptureTimer);
    state.webcamCaptureTimer = null;
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((t) => t.stop());
      state.webcamStream = null;
    }
    els.feedVideoRaw.srcObject = null;
    closeSocket();
    els.webcamStartBtn.hidden = false;
    els.webcamStopBtn.hidden = true;
    resetViewfinder();
    loadHistory();
  }

  // ==================================================================
  // Video upload mode
  // ==================================================================
  els.videoDropzone.addEventListener('click', () => els.videoInput.click());
  els.videoDropzone.addEventListener('dragover', (e) => { e.preventDefault(); });
  els.videoDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) onVideoFileChosen(e.dataTransfer.files[0]);
  });
  els.videoInput.addEventListener('change', () => {
    if (els.videoInput.files.length) onVideoFileChosen(els.videoInput.files[0]);
  });

  function onVideoFileChosen(file) {
    state.videoFile = file;
    els.videoDropzoneLabel.textContent = file.name;
    els.videoDropzone.classList.add('has-file');
    els.videoStartBtn.disabled = !state.engineReady;
  }

  els.videoStartBtn.addEventListener('click', startVideoProcessing);
  els.videoStopBtn.addEventListener('click', stopVideoProcessing);

  async function startVideoProcessing() {
    if (!state.videoFile) return;
    els.videoStartBtn.disabled = true;
    els.videoStartBtn.textContent = 'Uploading…';

    const formData = new FormData();
    formData.append('video', state.videoFile);

    try {
      const uploadRes = await authedFetch('/api/upload/video', { method: 'POST', body: formData });
      const uploadData = await uploadRes.json();
      if (!uploadRes.ok) throw new Error(uploadData.error || 'Upload failed');

      state.sessionId = uploadData.sessionId;
      state.videoServerPath = uploadData.path;
      openSocket(state.sessionId);

      // Give the socket a brief moment to establish before starting the stream
      await new Promise((resolve) => setTimeout(resolve, 250));

      const startRes = await authedFetch(`/api/session/${state.sessionId}/start-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: state.videoServerPath }),
      });
      const startData = await startRes.json();
      if (!startRes.ok) throw new Error(startData.error || 'Could not start processing');

      els.videoStartBtn.hidden = true;
      els.videoStopBtn.hidden = false;
      els.videoStartBtn.textContent = 'Process Video';
    } catch (e) {
      showToast('Video processing failed: ' + e.message);
      els.videoStartBtn.disabled = false;
      els.videoStartBtn.textContent = 'Process Video';
    }
  }

  function stopVideoProcessing() {
    if (state.sessionId) {
      authedFetch(`/api/session/${state.sessionId}/stop`, { method: 'POST' }).catch(() => {});
    }
    onVideoDone();
  }

  function onVideoDone() {
    closeSocket();
    els.videoStartBtn.hidden = false;
    els.videoStopBtn.hidden = true;
    els.videoStartBtn.disabled = !state.videoFile;
    loadHistory();
  }

  // ==================================================================
  // Image upload mode
  // ==================================================================
  els.imageDropzone.addEventListener('click', () => els.imageInput.click());
  els.imageDropzone.addEventListener('dragover', (e) => { e.preventDefault(); });
  els.imageDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) onImageFileChosen(e.dataTransfer.files[0]);
  });
  els.imageInput.addEventListener('change', () => {
    if (els.imageInput.files.length) onImageFileChosen(els.imageInput.files[0]);
  });

  function onImageFileChosen(file) {
    state.imageFile = file;
    els.imageDropzoneLabel.textContent = file.name;
    els.imageDropzone.classList.add('has-file');
    els.imageStartBtn.disabled = !state.engineReady;
  }

  els.imageStartBtn.addEventListener('click', analyzeImage);

  async function analyzeImage() {
    if (!state.imageFile) return;
    els.imageStartBtn.disabled = true;
    els.imageStartBtn.textContent = 'Analyzing…';

    const formData = new FormData();
    formData.append('image', state.imageFile);

    try {
      const uploadRes = await authedFetch('/api/upload/image', { method: 'POST', body: formData });
      const uploadData = await uploadRes.json();
      if (!uploadRes.ok) throw new Error(uploadData.error || 'Upload failed');

      state.sessionId = uploadData.sessionId;
      state.workers.clear();
      openSocket(state.sessionId);
      await new Promise((resolve) => setTimeout(resolve, 200));

      state.ws.send(JSON.stringify({ type: 'frame', mode: 'image', image: uploadData.image }));
      setTimeout(loadHistory, 800);
    } catch (e) {
      showToast('Image analysis failed: ' + e.message);
    } finally {
      els.imageStartBtn.disabled = false;
      els.imageStartBtn.textContent = 'Analyze Image';
    }
  }

  // ==================================================================
  // Shared helpers
  // ==================================================================
  function closeSocket() {
    if (state.ws) {
      try { state.ws.close(); } catch {}
      state.ws = null;
    }
  }

  function stopEverything() {
    if (state.webcamCaptureTimer) clearInterval(state.webcamCaptureTimer);
    state.webcamCaptureTimer = null;
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((t) => t.stop());
      state.webcamStream = null;
    }
    if (state.sessionId && state.auth) {
      fetch(`/api/session/${state.sessionId}/stop`, { method: 'POST', headers: authHeaders() }).catch(() => {});
    }
    closeSocket();
  }

  window.addEventListener('beforeunload', stopEverything);

  // ==================================================================
  // Boot
  // ==================================================================
  initAuth();
})();
