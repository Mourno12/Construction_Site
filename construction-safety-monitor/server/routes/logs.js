/**
 * logs.js
 * Read access to everything SiteGuard has persisted: past sessions, the
 * per-worker compliance history for a session, and the violation log
 * (with evidence snapshots) - plus CSV/JSON export of any of the above.
 *
 * Role enforcement (this is the actual access-control boundary, not just a
 * UI convenience):
 *   - operator: ALWAYS scoped to their own sessions/violations, regardless
 *     of query params - an operator cannot see or export another user's data
 *     by guessing a sessionId or omitting `mine`.
 *   - admin: sees everything by default; can pass `mine=true` to filter
 *     down to just their own sessions, same as an operator would see.
 */

const express = require('express');
const router = express.Router();
const db = require('../utils/db');

function sendExport(res, rows, format, filenameBase) {
  if (format === 'csv') {
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="${filenameBase}.csv"`);
    return res.send(db.toCSV(rows));
  }
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Content-Disposition', `attachment; filename="${filenameBase}.json"`);
  return res.send(JSON.stringify(rows, null, 2));
}

// Resolves which userId (if any) results should be scoped to. Operators are
// hard-scoped to themselves no matter what the client asks for; only an
// admin can ever see other users' data, and only when they don't opt into
// `mine=true` themselves.
function scopeUserId(req) {
  if (req.user.role !== 'admin') return req.user.id;
  return req.query.mine === 'true' ? req.user.id : undefined;
}

// Blocks an operator from reading/exporting a session that isn't theirs,
// even if they know (or guess) its sessionId. Admins can access any session.
function requireSessionAccess(req, res, next) {
  const session = db.getSession(req.params.sessionId);
  if (!session) return res.status(404).json({ error: 'Session not found.' });
  if (req.user.role !== 'admin' && session.userId !== req.user.id) {
    return res.status(403).json({ error: "You don't have access to this session." });
  }
  next();
}

// GET /api/logs/sessions?mine=true
router.get('/sessions', (req, res) => {
  res.json(db.listSessions({ userId: scopeUserId(req) }));
});

// GET /api/logs/sessions/export?format=csv|json&mine=true
router.get('/sessions/export', (req, res) => {
  const rows = db.listSessions({ userId: scopeUserId(req) }).map((s) => ({
    sessionId: s.id,
    username: s.username,
    mode: s.mode,
    startedAt: s.startedAt,
    endedAt: s.endedAt,
    demoMode: s.demoMode,
    totalWorkers: s.summary?.totalWorkers ?? '',
    finalComplianceRatePct: s.summary?.complianceRatePct ?? '',
    violationCount: s.summary?.violationCount ?? '',
  }));
  sendExport(res, rows, req.query.format === 'csv' ? 'csv' : 'json', 'siteguard-sessions');
});

// GET /api/logs/sessions/:sessionId/workers
router.get('/sessions/:sessionId/workers', requireSessionAccess, (req, res) => {
  res.json(db.listWorkers(req.params.sessionId));
});

// GET /api/logs/sessions/:sessionId/workers/export?format=csv|json
router.get('/sessions/:sessionId/workers/export', requireSessionAccess, (req, res) => {
  const rows = db.listWorkers(req.params.sessionId);
  sendExport(res, rows, req.query.format === 'csv' ? 'csv' : 'json', `siteguard-workers-${req.params.sessionId}`);
});

// GET /api/logs/violations?sessionId=&mine=true&limit=200
router.get('/violations', (req, res) => {
  if (req.query.sessionId) {
    const session = db.getSession(req.query.sessionId);
    if (session && req.user.role !== 'admin' && session.userId !== req.user.id) {
      return res.status(403).json({ error: "You don't have access to this session." });
    }
  }
  const violations = db.listViolations({
    sessionId: req.query.sessionId,
    userId: scopeUserId(req),
    limit: req.query.limit ? parseInt(req.query.limit, 10) : 200,
  });
  res.json(violations);
});

// GET /api/logs/violations/export?format=csv|json&sessionId=&mine=true
router.get('/violations/export', (req, res) => {
  const rows = db.listViolations({ sessionId: req.query.sessionId, userId: scopeUserId(req) });
  sendExport(res, rows, req.query.format === 'csv' ? 'csv' : 'json', 'siteguard-violations');
});

module.exports = router;
