/**
 * auth.js (routes)
 * Register, login, current-user, and logout endpoints.
 *
 * Security measures in place:
 *  - Passwords hashed with bcrypt, never stored or logged in plain text.
 *  - Password policy: 8+ characters, at least one letter and one number.
 *  - Email format validated and required to be unique (stored, never sent -
 *    see the "email" question this project settled on: profile field only).
 *  - Generic error messages on login ("Incorrect username or password")
 *    regardless of whether the username exists, to avoid user enumeration.
 *  - Per-IP rate limiting on both endpoints (express-rate-limit) to blunt
 *    rapid-fire brute force / credential-stuffing bursts.
 *  - Per-username lockout (server/utils/db.js) after repeated failed
 *    logins, independent of the IP rate limiter, so a slow/distributed
 *    attack against one account still gets throttled.
 *
 * Logout is stateless: JWTs aren't stored server-side, so "logging out"
 * just means the client discards its token. The endpoint exists mainly for
 * a consistent API and a place to hook in token revocation later if needed.
 */

const express = require('express');
const rateLimit = require('express-rate-limit');
const router = express.Router();
const db = require('../utils/db');
const { hashPassword, verifyPassword, issueToken } = require('../utils/auth');
const requireAuth = require('../middleware/requireAuth');

const USERNAME_RE = /^[a-zA-Z0-9_.-]{3,32}$/;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
// At least 8 characters, at least one letter and one number.
const PASSWORD_RE = /^(?=.*[A-Za-z])(?=.*\d).{8,}$/;

// Rate limits are intentionally generous enough not to bother a normal user
// who fat-fingers a password a couple times, but tight enough to blunt an
// automated brute-force burst against this endpoint.
const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many login attempts from this network. Please wait a few minutes and try again.' },
});

const registerLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  limit: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many accounts created from this network. Please wait a while and try again.' },
});

function publicUser(user) {
  return { username: user.username, email: user.email, role: user.role };
}

// POST /api/auth/register  { username, email, password }
router.post('/register', registerLimiter, async (req, res) => {
  const { username, email, password } = req.body || {};

  if (!username || !email || !password) {
    return res.status(400).json({ error: 'Username, email, and password are required.' });
  }
  if (!USERNAME_RE.test(username)) {
    return res.status(400).json({ error: 'Username must be 3-32 characters (letters, numbers, _ . -).' });
  }
  if (!EMAIL_RE.test(String(email).trim())) {
    return res.status(400).json({ error: 'Please enter a valid email address.' });
  }
  if (!PASSWORD_RE.test(String(password))) {
    return res.status(400).json({ error: 'Password must be at least 8 characters and include at least one letter and one number.' });
  }
  if (db.findUserByUsername(username)) {
    return res.status(409).json({ error: 'That username is already taken.' });
  }
  if (db.findUserByEmail(email)) {
    return res.status(409).json({ error: 'That email is already registered.' });
  }

  // First registered user becomes admin; everyone after is an operator.
  const role = db.countUsers() === 0 ? 'admin' : 'operator';
  const passwordHash = await hashPassword(password);
  const user = db.createUser({ username, email: String(email).trim().toLowerCase(), passwordHash, role });
  const token = issueToken(user);

  res.status(201).json({ token, ...publicUser(user) });
});

// POST /api/auth/login  { username, password }
router.post('/login', loginLimiter, async (req, res) => {
  const { username, password } = req.body || {};
  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required.' });
  }

  const lockStatus = db.isLockedOut(username);
  if (lockStatus.locked) {
    const minutes = Math.ceil(lockStatus.retryAfterMs / 60000);
    return res.status(429).json({ error: `Too many failed attempts. Try again in about ${minutes} minute(s).` });
  }

  const user = db.findUserByUsername(username);
  const ok = user ? await verifyPassword(password, user.passwordHash) : false;

  if (!ok) {
    db.recordFailedLogin(username);
    return res.status(401).json({ error: 'Incorrect username or password.' });
  }

  db.resetLoginAttempts(username);
  const token = issueToken(user);
  res.json({ token, ...publicUser(user) });
});

// GET /api/auth/me
router.get('/me', requireAuth, (req, res) => {
  const user = db.findUserById(req.user.id);
  if (!user) return res.status(401).json({ error: 'Account no longer exists.' });
  res.json(publicUser(user));
});

// POST /api/auth/logout - stateless, see file header comment.
router.post('/logout', requireAuth, (req, res) => {
  res.json({ loggedOut: true });
});

module.exports = router;
