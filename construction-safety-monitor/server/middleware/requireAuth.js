/**
 * requireAuth.js
 * Protects REST routes: expects `Authorization: Bearer <token>`.
 * Attaches the decoded token payload to req.user on success.
 */

const { verifyToken } = require('../utils/auth');

function requireAuth(req, res, next) {
  const header = req.headers.authorization || '';
  const headerToken = header.startsWith('Bearer ') ? header.slice(7) : null;
  const token = headerToken || req.query.token || null;

  if (!token) {
    return res.status(401).json({ error: 'Not authenticated. Please log in.' });
  }

  const payload = verifyToken(token);
  if (!payload) {
    return res.status(401).json({ error: 'Session expired or invalid. Please log in again.' });
  }

  req.user = { id: payload.sub, username: payload.username, role: payload.role };
  next();
}

module.exports = requireAuth;
