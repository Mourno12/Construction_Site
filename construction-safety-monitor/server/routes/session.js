/**
 * session.js
 * Lifecycle control for a monitoring session: kicking off server-side
 * frame-by-frame processing of an uploaded video, and stopping any running
 * session (video playback loop, or a live webcam feed).
 */

const express = require('express');
const router = express.Router();
const pythonBridge = require('../utils/pythonBridge');
const sessionManager = require('../utils/sessionManager');
const requireAuth = require('../middleware/requireAuth');

// GET /api/session/status - basic health info about the inference engine (public, used pre-login)
router.get('/status', (req, res) => {
  res.json({
    engineReady: pythonBridge.ready,
    demoMode: pythonBridge.demoMode,
  });
});

router.use(requireAuth);

// Only the session's own owner (or an admin) may control it - the WS
// connection registered the owner in sessionManager the moment it opened,
// before any frame/db record necessarily exists yet.
function requireSessionOwner(req, res, next) {
  const owner = sessionManager.getUser(req.params.sessionId);
  if (owner && req.user.role !== 'admin' && owner.id !== req.user.id) {
    return res.status(403).json({ error: "You don't have access to this session." });
  }
  next();
}

// POST /api/session/:sessionId/start-video   body: { path: "<abs path from upload>" }
router.post('/:sessionId/start-video', requireSessionOwner, (req, res) => {
  const { sessionId } = req.params;
  const { path: videoPath } = req.body;
  if (!videoPath) return res.status(400).json({ error: 'Missing "path" in request body' });

  if (!pythonBridge.ready) {
    return res.status(503).json({ error: 'Inference engine is still starting up, try again shortly' });
  }

  try {
    pythonBridge.send({ type: 'start_video', sessionId, path: videoPath });
    res.json({ started: true, sessionId });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// POST /api/session/:sessionId/stop
router.post('/:sessionId/stop', requireSessionOwner, (req, res) => {
  const { sessionId } = req.params;
  try {
    pythonBridge.send({ type: 'stop', sessionId });
    res.json({ stopped: true, sessionId });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

module.exports = router;
