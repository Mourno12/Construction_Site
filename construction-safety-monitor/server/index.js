/**
 * index.js
 * Entry point for the Construction Site Safety Monitoring System backend.
 *
 *  - Serves the dashboard (static files in /public)
 *  - Auth: register/login/logout (JWT), gating everything except the login screen
 *  - REST endpoints for uploading video/image and controlling sessions
 *  - A WebSocket server (auth-checked) that streams live detection results
 *    (bounding boxes, PPE compliance, worker IDs, stats) to the dashboard
 *  - Bridges everything to a persistent Python worker process that runs
 *    YOLOv12n + ViT + ByteTrack
 *  - Persists sessions / per-worker history / violations (with evidence
 *    snapshots) to disk, with CSV/JSON export via /api/logs
 */

const express = require('express');
const http = require('http');
const path = require('path');
const cors = require('cors');
const { WebSocketServer } = require('ws');
const { URL } = require('url');

const pythonBridge = require('./utils/pythonBridge');
const sessionManager = require('./utils/sessionManager');
const db = require('./utils/db');
const { verifyToken } = require('./utils/auth');
const requireAuth = require('./middleware/requireAuth');

const authRoutes = require('./routes/auth');
const uploadRoutes = require('./routes/upload');
const sessionRoutes = require('./routes/session');
const logsRoutes = require('./routes/logs');

const PORT = process.env.PORT || 3000;

const app = express();
app.use(cors());
app.use(express.json({ limit: '25mb' }));
app.use(express.static(path.join(__dirname, '..', 'public')));
app.use('/violations', express.static(db.VIOLATIONS_DIR)); // evidence snapshot images

app.use('/api/auth', authRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/session', sessionRoutes);
app.use('/api/logs', requireAuth, logsRoutes);

app.get('/api/health', (req, res) => {
  res.json({ ok: true, engineReady: pythonBridge.ready, demoMode: pythonBridge.demoMode });
});

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

// Tracks the most recent stats seen for each active session, so we can
// write a sensible summary to the session record when it ends.
const lastStatsBySession = new Map();

wss.on('connection', (ws, req) => {
  const { searchParams } = new URL(req.url, `http://${req.headers.host}`);
  const sessionId = searchParams.get('sessionId');
  const token = searchParams.get('token');

  if (!sessionId) {
    ws.close(1008, 'sessionId query param is required');
    return;
  }

  const payload = token ? verifyToken(token) : null;
  if (!payload) {
    ws.close(1008, 'Authentication required or token expired - please log in again');
    return;
  }
  const user = { id: payload.sub, username: payload.username, role: payload.role };

  sessionManager.register(sessionId, ws, user);
  ws.send(JSON.stringify({
    type: 'connected',
    sessionId,
    engineReady: pythonBridge.ready,
    demoMode: pythonBridge.demoMode,
  }));

  ws.on('message', (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch (e) {
      ws.send(JSON.stringify({ type: 'error', message: 'Invalid JSON message' }));
      return;
    }

    if (msg.type === 'frame') {
      // Live webcam frame captured client-side, forwarded to the inference engine.
      if (!pythonBridge.ready) {
        ws.send(JSON.stringify({ type: 'error', sessionId, message: 'Inference engine not ready yet' }));
        return;
      }
      if (!db.getSession(sessionId)) {
        db.startSession({ sessionId, userId: user.id, username: user.username, mode: msg.mode || 'webcam' });
      }
      try {
        pythonBridge.send({
          type: 'frame',
          sessionId,
          mode: msg.mode || 'webcam',
          image: msg.image,
        });
      } catch (e) {
        ws.send(JSON.stringify({ type: 'error', sessionId, message: e.message }));
      }
    } else if (msg.type === 'ping') {
      ws.send(JSON.stringify({ type: 'pong' }));
    }
  });

  ws.on('close', () => {
    finalizeSession(sessionId);
    sessionManager.unregister(sessionId);
  });
});

function finalizeSession(sessionId) {
  const stats = lastStatsBySession.get(sessionId);
  const violations = db.listViolations({ sessionId });
  db.endSession(sessionId, {
    totalWorkers: stats?.totalWorkers ?? 0,
    complianceRatePct: stats?.complianceRatePct ?? null,
    violationCount: violations.length,
  });
  db.persistWorkers();
  lastStatsBySession.delete(sessionId);
}

// Route results coming back from the Python worker to the right browser tab,
// and persist sessions / worker history / violation evidence as we go.
pythonBridge.on('message', (msg) => {
  if (!msg.sessionId) return;
  sessionManager.send(msg.sessionId, msg);

  if (msg.type === 'result') {
    const user = sessionManager.getUser(msg.sessionId);

    if (!db.getSession(msg.sessionId)) {
      db.startSession({
        sessionId: msg.sessionId,
        userId: user?.id,
        username: user?.username,
        mode: msg.mode,
      });
    }
    if (msg.stats) {
      db.setSessionDemoMode(msg.sessionId, msg.stats.demoMode);
      lastStatsBySession.set(msg.sessionId, msg.stats);
    }

    const detectionsById = new Map((msg.detections || []).map((d) => [d.trackId, d]));
    for (const d of msg.detections || []) {
      db.upsertWorker(msg.sessionId, d.trackId, {
        framesSeen: d.framesSeen,
        complianceRate: d.complianceRate,
        lastHelmet: d.helmet,
        lastVest: d.vest,
        compliant: d.compliant,
        firstSeen: d.firstSeen ? new Date(d.firstSeen * 1000).toISOString() : undefined,
        lastSeen: d.lastSeen ? new Date(d.lastSeen * 1000).toISOString() : undefined,
      });
    }

    for (const trackId of msg.stats?.newAlerts || []) {
      const d = detectionsById.get(trackId);
      const violation = db.addViolation({
        sessionId: msg.sessionId,
        userId: user?.id,
        trackId,
        helmet: d ? d.helmet : false,
        vest: d ? d.vest : false,
        snapshotBase64: msg.annotated,
      });
      // Push the persisted violation (with its snapshot URL) straight to the
      // browser so the alert feed can show the evidence thumbnail immediately.
      sessionManager.send(msg.sessionId, { type: 'violation', violation });
    }
  }

  if (msg.type === 'video_done') {
    finalizeSession(msg.sessionId);
  }
});

pythonBridge.on('ready', (msg) => {
  // Notify every currently-connected client that the engine has finished
  // loading models (useful if a client connected before startup finished).
  wss.clients.forEach((ws) => {
    if (ws.readyState === ws.OPEN) {
      ws.send(JSON.stringify({ type: 'engine_ready', demoMode: msg.demoMode, detector: msg.detector, classifier: msg.classifier }));
    }
  });
});

async function main() {
  // sql.js loads its WASM module asynchronously - everything that touches
  // the database (routes, the pythonBridge message handler above, etc.)
  // assumes it's already usable, so nothing is allowed to start accepting
  // connections until this resolves.
  await db.init();

  pythonBridge.start();

  server.listen(PORT, () => {
    console.log(`Construction Site Safety Monitoring System running at http://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error('Fatal error during startup:', err);
  process.exit(1);
});

process.on('SIGINT', () => {
  console.log('\nShutting down...');
  pythonBridge.stop();
  server.close(() => process.exit(0));
});
