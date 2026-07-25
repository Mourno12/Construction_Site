/**
 * sessionManager.js
 * Tracks which WebSocket connection (and which logged-in user) belongs to
 * which sessionId, so results coming back from the Python worker (tagged
 * with a sessionId) can be routed to the correct browser tab, and so
 * persisted logs can be attributed to the right account.
 */

const clients = new Map(); // sessionId -> { ws, user }

function register(sessionId, ws, user) {
  clients.set(sessionId, { ws, user: user || null });
}

function unregister(sessionId) {
  clients.delete(sessionId);
}

function send(sessionId, payload) {
  const entry = clients.get(sessionId);
  if (entry && entry.ws && entry.ws.readyState === entry.ws.OPEN) {
    entry.ws.send(JSON.stringify(payload));
    return true;
  }
  return false;
}

function has(sessionId) {
  return clients.has(sessionId);
}

function getUser(sessionId) {
  const entry = clients.get(sessionId);
  return entry ? entry.user : null;
}

module.exports = { register, unregister, send, has, getUser };
