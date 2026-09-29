/**
 * db.js
 * Real SQLite persistence for EdgeGuard, via sql.js - SQLite compiled to
 * WebAssembly. Deliberately NOT using a native module (better-sqlite3,
 * node-sqlite3, etc.): those require a matching prebuilt binary or a C++
 * build toolchain on install, which is exactly the kind of environment
 * trap this project has already hit once (Python's `python`/`python3`/`py`
 * naming mess on Windows). sql.js is pure WebAssembly - `npm install` can't
 * fail to find a compiler for it on any platform.
 *
 * Trade-off: sql.js runs entirely in memory and has no built-in file
 * persistence, so this module keeps the same "load into memory, debounce
 * writes to disk" pattern as before - except the on-disk file is now a real
 * SQLite database (data/edgeguard.db) instead of a hand-rolled JSON blob,
 * with actual schema, indexes, and SQL queries doing the filtering that
 * used to be done by hand in JavaScript.
 *
 * IMPORTANT: call and await `init()` once at server startup, before
 * mounting anything that touches the database - sql.js's WASM module has
 * to load asynchronously first. Every function below is synchronous once
 * `init()` has resolved.
 */

const fs = require('fs');
const path = require('path');
const initSqlJs = require('sql.js');
const { v4: uuidv4 } = require('uuid');

const DATA_DIR = path.join(__dirname, '..', '..', 'data');
const DB_FILE = path.join(DATA_DIR, 'edgeguard.db');
const VIOLATIONS_DIR = path.join(DATA_DIR, 'violations');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(VIOLATIONS_DIR)) fs.mkdirSync(VIOLATIONS_DIR, { recursive: true });

const SCHEMA_SQL = `
  CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email         TEXT UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'operator',
    created_at    TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS login_attempts (
    username     TEXT PRIMARY KEY COLLATE NOCASE,
    count        INTEGER NOT NULL DEFAULT 0,
    locked_until INTEGER
  );

  CREATE TABLE IF NOT EXISTS sessions (
    id                   TEXT PRIMARY KEY,
    user_id              TEXT,
    username             TEXT,
    mode                 TEXT,
    started_at           TEXT NOT NULL,
    ended_at             TEXT,
    demo_mode            INTEGER,
    total_workers        INTEGER,
    compliance_rate_pct  REAL,
    violation_count      INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id)
  );
  CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
  CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

  CREATE TABLE IF NOT EXISTS workers (
    session_id       TEXT NOT NULL,
    track_id         INTEGER NOT NULL,
    first_seen       TEXT,
    last_seen        TEXT,
    frames_seen      INTEGER,
    compliance_rate  REAL,
    last_helmet      INTEGER,
    last_vest        INTEGER,
    compliant        INTEGER,
    PRIMARY KEY (session_id, track_id)
  );
  CREATE INDEX IF NOT EXISTS idx_workers_session ON workers(session_id);
  CREATE INDEX IF NOT EXISTS idx_workers_lastseen ON workers(last_seen);

  CREATE TABLE IF NOT EXISTS violations (
    id         TEXT PRIMARY KEY,
    session_id TEXT,
    user_id    TEXT,
    track_id   INTEGER,
    helmet     INTEGER,
    vest       INTEGER,
    timestamp  TEXT NOT NULL,
    snapshot   TEXT
  );
  CREATE INDEX IF NOT EXISTS idx_violations_session ON violations(session_id);
  CREATE INDEX IF NOT EXISTS idx_violations_user ON violations(user_id);
  CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
`;

// Columns added after the original schema shipped. CREATE TABLE IF NOT
// EXISTS above only helps on a brand-new database file - an existing
// edgeguard.db from before these fields existed needs each column added
// in place, or every query naming them would fail against old files.
const SCHEMA_MIGRATIONS = [
  ['workers', 'last_gloves', 'INTEGER'],
  ['workers', 'last_boots', 'INTEGER'],
  ['workers', 'last_mask', 'INTEGER'],
  ['workers', 'last_fallen', 'INTEGER'],
  ['violations', 'kind', "TEXT DEFAULT 'ppe'"],       // 'ppe' | 'fall' | 'object_fall'
  ['violations', 'missing_items', 'TEXT'],            // JSON array, e.g. ["gloves","mask"]
  ['violations', 'bbox', 'TEXT'],                     // JSON [x1,y1,x2,y2], used by object_fall (no trackId-based PPE info)
];

function runMigrations() {
  for (const [table, column, columnDef] of SCHEMA_MIGRATIONS) {
    try {
      _sqlDb.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${columnDef}`);
    } catch (e) {
      // Column already exists (this file was created after the migration
      // landed, or it already ran once) - nothing to do.
    }
  }
}

let _sqlDb = null;
let _ready = false;

async function init() {
  if (_ready) return;
  const SQL = await initSqlJs();

  try {
    if (fs.existsSync(DB_FILE)) {
      _sqlDb = new SQL.Database(fs.readFileSync(DB_FILE));
    } else {
      _sqlDb = new SQL.Database();
    }
    _sqlDb.exec(SCHEMA_SQL); // corruption in an existing file often only surfaces here, not on open
  } catch (e) {
    // Corrupted file (e.g. the process was killed mid-write). Don't crash
    // the whole server over it - move the bad file aside for forensics
    // and start fresh, the same way the old JSON store used to degrade.
    const backupPath = `${DB_FILE}.corrupt-${Date.now()}`;
    console.error(`[db] ${DB_FILE} is corrupted (${e.message}). Backing it up to ${backupPath} and starting a fresh database.`);
    if (fs.existsSync(DB_FILE)) fs.renameSync(DB_FILE, backupPath);
    _sqlDb = new SQL.Database();
    _sqlDb.exec(SCHEMA_SQL);
  }

  runMigrations();

  _ready = true;
  save();
  console.log(`[db] SQLite store ready at ${DB_FILE}`);
}

function assertReady() {
  if (!_ready) {
    throw new Error('db.init() must be awaited before using the database (see server/index.js startup order)');
  }
}

function save() {
  if (!_sqlDb) return;
  fs.writeFileSync(DB_FILE, Buffer.from(_sqlDb.export()));
}

let _saveTimer = null;
function scheduleSave() {
  if (_saveTimer) return;
  _saveTimer = setTimeout(() => {
    _saveTimer = null;
    save();
  }, 800);
}

// ---------------------------------------------------------------------------
// Tiny query helpers on top of sql.js's prepare/step/getAsObject API
// ---------------------------------------------------------------------------
function run(sql, params = []) {
  assertReady();
  const stmt = _sqlDb.prepare(sql);
  stmt.bind(params);
  stmt.step();
  stmt.free();
}

function get(sql, params = []) {
  assertReady();
  const stmt = _sqlDb.prepare(sql);
  stmt.bind(params);
  let row;
  if (stmt.step()) row = stmt.getAsObject();
  stmt.free();
  return row;
}

function all(sql, params = []) {
  assertReady();
  const stmt = _sqlDb.prepare(sql);
  stmt.bind(params);
  const rows = [];
  while (stmt.step()) rows.push(stmt.getAsObject());
  stmt.free();
  return rows;
}

// ---------------------------------------------------------------------------
// Row <-> app-shape mappers (snake_case columns -> the camelCase shapes the
// rest of the app already expects, so no other file needs to change)
// ---------------------------------------------------------------------------
function mapUserRow(row) {
  return {
    id: row.id,
    username: row.username,
    email: row.email,
    passwordHash: row.password_hash,
    role: row.role,
    createdAt: row.created_at,
  };
}

function mapSessionRow(row) {
  return {
    id: row.id,
    userId: row.user_id,
    username: row.username,
    mode: row.mode,
    startedAt: row.started_at,
    endedAt: row.ended_at,
    demoMode: row.demo_mode === null ? null : !!row.demo_mode,
    summary: row.ended_at ? {
      totalWorkers: row.total_workers,
      complianceRatePct: row.compliance_rate_pct,
      violationCount: row.violation_count,
    } : null,
  };
}

function mapWorkerRow(row) {
  return {
    sessionId: row.session_id,
    trackId: row.track_id,
    firstSeen: row.first_seen,
    lastSeen: row.last_seen,
    framesSeen: row.frames_seen,
    complianceRate: row.compliance_rate,
    lastHelmet: !!row.last_helmet,
    lastVest: !!row.last_vest,
    lastGloves: !!row.last_gloves,
    lastBoots: !!row.last_boots,
    lastMask: !!row.last_mask,
    lastFallen: !!row.last_fallen,
    compliant: !!row.compliant,
  };
}

function mapViolationRow(row) {
  return {
    id: row.id,
    sessionId: row.session_id,
    userId: row.user_id,
    trackId: row.track_id,
    helmet: !!row.helmet,
    vest: !!row.vest,
    timestamp: row.timestamp,
    snapshot: row.snapshot,
    kind: row.kind || 'ppe', // 'ppe' | 'fall' | 'object_fall' - absent on rows written before this field existed
    missingItems: row.missing_items ? JSON.parse(row.missing_items) : [],
    bbox: row.bbox ? JSON.parse(row.bbox) : null,
  };
}

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------
function createUser({ username, email, passwordHash, role = 'operator' }) {
  const id = uuidv4();
  const createdAt = new Date().toISOString();
  run(
    `INSERT INTO users (id, username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
    [id, username, email || null, passwordHash, role, createdAt]
  );
  save(); // account creation is rare + important - flush immediately, don't debounce
  return { id, username, email: email || null, passwordHash, role, createdAt };
}

function findUserByUsername(username) {
  const row = get(`SELECT * FROM users WHERE username = ? COLLATE NOCASE`, [username]);
  return row ? mapUserRow(row) : undefined;
}

function findUserByEmail(email) {
  if (!email) return undefined;
  const row = get(`SELECT * FROM users WHERE email = ? COLLATE NOCASE`, [email]);
  return row ? mapUserRow(row) : undefined;
}

function findUserById(id) {
  const row = get(`SELECT * FROM users WHERE id = ?`, [id]);
  return row ? mapUserRow(row) : undefined;
}

function countUsers() {
  return get(`SELECT COUNT(*) as c FROM users`).c;
}

// ---------------------------------------------------------------------------
// Login attempt tracking (per-username lockout, independent of the request
// rate limiter - this catches slow/targeted brute force attempts spread out
// over time, not just rapid-fire ones).
// ---------------------------------------------------------------------------
const MAX_FAILED_ATTEMPTS = 5;
const LOCKOUT_DURATION_MS = 15 * 60 * 1000; // 15 minutes

function isLockedOut(username) {
  const row = get(`SELECT * FROM login_attempts WHERE username = ? COLLATE NOCASE`, [username]);
  if (row && row.locked_until && Date.now() < row.locked_until) {
    return { locked: true, retryAfterMs: row.locked_until - Date.now() };
  }
  return { locked: false };
}

function recordFailedLogin(username) {
  const row = get(`SELECT * FROM login_attempts WHERE username = ? COLLATE NOCASE`, [username]);
  let count = (row ? row.count : 0) + 1;
  let lockedUntil = row ? row.locked_until : null;
  if (count >= MAX_FAILED_ATTEMPTS) {
    lockedUntil = Date.now() + LOCKOUT_DURATION_MS;
    count = 0; // reset counter once locked, so the next window starts clean
  }
  if (row) {
    run(`UPDATE login_attempts SET count = ?, locked_until = ? WHERE username = ? COLLATE NOCASE`, [count, lockedUntil, username]);
  } else {
    run(`INSERT INTO login_attempts (username, count, locked_until) VALUES (?, ?, ?)`, [String(username).toLowerCase(), count, lockedUntil]);
  }
  save(); // security-relevant - flush immediately
}

function resetLoginAttempts(username) {
  run(`DELETE FROM login_attempts WHERE username = ? COLLATE NOCASE`, [username]);
  scheduleSave();
}

// ---------------------------------------------------------------------------
// Sessions (one per webcam run / video upload / image upload)
// ---------------------------------------------------------------------------
function startSession({ sessionId, userId, username, mode }) {
  const existing = getSession(sessionId);
  if (existing) return existing;
  const startedAt = new Date().toISOString();
  run(
    `INSERT INTO sessions (id, user_id, username, mode, started_at) VALUES (?, ?, ?, ?, ?)`,
    [sessionId, userId || null, username || 'anonymous', mode, startedAt]
  );
  scheduleSave();
  return getSession(sessionId);
}

function endSession(sessionId, summary = {}) {
  const session = getSession(sessionId);
  if (!session) return null;
  const endedAt = new Date().toISOString();
  run(
    `UPDATE sessions SET ended_at = ?, total_workers = ?, compliance_rate_pct = ?, violation_count = ? WHERE id = ?`,
    [endedAt, summary.totalWorkers ?? null, summary.complianceRatePct ?? null, summary.violationCount ?? null, sessionId]
  );
  save(); // session end is a natural checkpoint - flush immediately
  return getSession(sessionId);
}

function setSessionDemoMode(sessionId, demoMode) {
  const session = getSession(sessionId);
  if (!session || session.demoMode !== null) return; // only set once
  run(`UPDATE sessions SET demo_mode = ? WHERE id = ?`, [demoMode ? 1 : 0, sessionId]);
  scheduleSave();
}

function listSessions({ userId } = {}) {
  const rows = userId
    ? all(`SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC`, [userId])
    : all(`SELECT * FROM sessions ORDER BY started_at DESC`);
  return rows.map(mapSessionRow);
}

function getSession(sessionId) {
  const row = get(`SELECT * FROM sessions WHERE id = ?`, [sessionId]);
  return row ? mapSessionRow(row) : undefined;
}

// ---------------------------------------------------------------------------
// Workers (per-session tracked-worker summary - upserted every frame)
// ---------------------------------------------------------------------------
const MAX_WORKERS_PER_SESSION = 500; // safety cap - a flaky fallback tracker can otherwise mint unbounded IDs

function upsertWorker(sessionId, trackId, patch) {
  const now = new Date().toISOString();
  const existing = get(`SELECT * FROM workers WHERE session_id = ? AND track_id = ?`, [sessionId, trackId]);

  if (!existing) {
    run(
      `INSERT INTO workers (session_id, track_id, first_seen, last_seen, frames_seen, compliance_rate, last_helmet, last_vest, last_gloves, last_boots, last_mask, last_fallen, compliant)
       VALUES (?, ?, ?, ?, 0, 100, 1, 1, 1, 1, 1, 0, 1)`,
      [sessionId, trackId, patch.firstSeen || now, now]
    );
    evictExcessWorkers(sessionId);
  }

  // present-by-default (1) for PPE items, absent-by-default (0) for fallen -
  // matches the same "optimistic until proven otherwise" fallback the
  // original helmet/vest columns used.
  const bit = (value, existingCol, fallback) =>
    value !== undefined ? (value ? 1 : 0) : (existing ? existing[existingCol] : fallback);

  run(
    `UPDATE workers SET last_seen = ?, frames_seen = ?, compliance_rate = ?,
       last_helmet = ?, last_vest = ?, last_gloves = ?, last_boots = ?, last_mask = ?, last_fallen = ?, compliant = ?
     WHERE session_id = ? AND track_id = ?`,
    [
      patch.lastSeen || now,
      patch.framesSeen ?? (existing ? existing.frames_seen : 0),
      patch.complianceRate ?? (existing ? existing.compliance_rate : 100),
      bit(patch.lastHelmet, 'last_helmet', 1),
      bit(patch.lastVest, 'last_vest', 1),
      bit(patch.lastGloves, 'last_gloves', 1),
      bit(patch.lastBoots, 'last_boots', 1),
      bit(patch.lastMask, 'last_mask', 1),
      bit(patch.lastFallen, 'last_fallen', 0),
      bit(patch.compliant, 'compliant', 1),
      sessionId,
      trackId,
    ]
  );
  scheduleSave(); // hot path (called per frame) - debounced, not written every call

  const row = get(`SELECT * FROM workers WHERE session_id = ? AND track_id = ?`, [sessionId, trackId]);
  return mapWorkerRow(row);
}

// A real SQL query replaces what used to be manual JS array sort+slice+filter.
function evictExcessWorkers(sessionId) {
  const total = get(`SELECT COUNT(*) as c FROM workers WHERE session_id = ?`, [sessionId]).c;
  if (total <= MAX_WORKERS_PER_SESSION) return;
  run(
    `DELETE FROM workers WHERE rowid IN (
       SELECT rowid FROM workers WHERE session_id = ? ORDER BY last_seen ASC LIMIT ?
     )`,
    [sessionId, total - MAX_WORKERS_PER_SESSION]
  );
}

function persistWorkers() {
  save(); // force an immediate flush (e.g. when a session ends)
}

function listWorkers(sessionId) {
  return all(`SELECT * FROM workers WHERE session_id = ? ORDER BY track_id ASC`, [sessionId]).map(mapWorkerRow);
}

// ---------------------------------------------------------------------------
// Violations (one row per alert - i.e. a worker crossing the "missing PPE
// for N consecutive frames" threshold), with an optional evidence snapshot.
// ---------------------------------------------------------------------------
function addViolation({ sessionId, userId, trackId, kind = 'ppe', helmet, vest, missingItems, bbox, snapshotBase64 }) {
  const id = uuidv4();
  let snapshotFile = null;

  if (snapshotBase64) {
    try {
      const base64 = snapshotBase64.includes(',') ? snapshotBase64.split(',')[1] : snapshotBase64;
      snapshotFile = `${id}.jpg`;
      fs.writeFileSync(path.join(VIOLATIONS_DIR, snapshotFile), Buffer.from(base64, 'base64'));
    } catch (e) {
      console.error('[db] failed to save violation snapshot:', e.message);
      snapshotFile = null;
    }
  }

  const timestamp = new Date().toISOString();
  const snapshot = snapshotFile ? `/violations/${snapshotFile}` : null;
  const missingItemsJson = missingItems && missingItems.length ? JSON.stringify(missingItems) : null;
  const bboxJson = bbox ? JSON.stringify(bbox) : null;
  run(
    `INSERT INTO violations (id, session_id, user_id, track_id, helmet, vest, timestamp, snapshot, kind, missing_items, bbox)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [id, sessionId, userId || null, trackId, helmet ? 1 : 0, vest ? 1 : 0, timestamp, snapshot, kind, missingItemsJson, bboxJson]
  );
  save(); // violations are comparatively rare and important - flush immediately

  return {
    id, sessionId, userId: userId || null, trackId, helmet: !!helmet, vest: !!vest, timestamp, snapshot,
    kind, missingItems: missingItems || [], bbox: bbox || null,
  };
}

function listViolations({ sessionId, userId, limit } = {}) {
  const conditions = [];
  const params = [];
  if (sessionId) { conditions.push('session_id = ?'); params.push(sessionId); }
  if (userId) { conditions.push('user_id = ?'); params.push(userId); }

  let sql = 'SELECT * FROM violations';
  if (conditions.length) sql += ' WHERE ' + conditions.join(' AND ');
  sql += ' ORDER BY timestamp DESC';
  if (limit) { sql += ' LIMIT ?'; params.push(limit); }

  return all(sql, params).map(mapViolationRow);
}

// ---------------------------------------------------------------------------
// CSV helpers (used by the export routes)
// ---------------------------------------------------------------------------
function csvEscape(val) {
  if (val === null || val === undefined) return '';
  if (typeof val === 'boolean') return val ? 'Yes' : 'No';
  const str = String(val);
  return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
}

// camelCase field name -> "Title Case With Spaces" column header, so a
// plain data dump (sessionId, framesSeen, lastHelmet, ...) reads like a
// report instead of a variable listing.
function prettifyHeader(key) {
  return key
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/^./, (c) => c.toUpperCase());
}

// Generic: dumps whatever shape of row it's given, one column per key,
// headers prettified. Fine for sessions/workers exports, which don't need
// any row-specific formatting (no embedded links, no enum labels).
function toCSV(rows) {
  if (!rows.length) return '';
  const keys = Object.keys(rows[0]);
  const lines = [keys.map(prettifyHeader).join(',')];
  for (const row of rows) {
    lines.push(keys.map((k) => csvEscape(row[k])).join(','));
  }
  return lines.join('\n');
}

const VIOLATION_KIND_LABELS = { ppe: 'PPE Violation', fall: 'Worker Fallen', object_fall: 'Falling Object' };

// Violations get their own formatter rather than the generic one above,
// because a raw dump of this table is genuinely hard to read as a report:
// a bare relative path ("/violations/<uuid>.jpg") isn't a working link once
// it leaves the browser (no host, and plain CSV text isn't clickable at
// all), the "kind" enum reads as code not English, and the most
// review-relevant columns (when, who, what) were buried after id/session/
// user UUIDs that matter for lookups but not for a human scanning the log.
// baseUrl (e.g. "http://localhost:3000") turns the stored relative snapshot
// path into a full URL, wrapped in Excel/Sheets' own HYPERLINK() formula so
// it opens as an actual clickable link, not inert text.
function violationsToCSV(rows, baseUrl) {
  const headers = [
    'Timestamp', 'Worker ID', 'Type', 'Missing PPE Items',
    'Helmet', 'Vest', 'Photo Preview', 'Evidence Snapshot',
    'Violation ID', 'Session ID', 'User ID', 'Object BBox',
  ];
  const lines = [headers.join(',')];
  for (const v of rows) {
    const snapshotUrl = v.snapshot ? `${baseUrl}${v.snapshot}` : null;
    // IMAGE() renders an actual thumbnail in the cell - only works in
    // Excel for Microsoft 365 (desktop), and only while this server is
    // running, since it fetches from baseUrl (localhost) live when the
    // sheet is opened/recalculated. Older Excel and Google Sheets (which
    // can't reach localhost at all) will show #NAME?/broken - HYPERLINK
    // next to it is the fallback that works everywhere.
    const photoCell = snapshotUrl ? `=IMAGE("${snapshotUrl}","Evidence photo",0)` : 'No photo';
    const snapshotCell = snapshotUrl ? `=HYPERLINK("${snapshotUrl}","View Snapshot")` : 'No snapshot';
    const cells = [
      new Date(v.timestamp).toLocaleString(),
      v.trackId,
      VIOLATION_KIND_LABELS[v.kind] || v.kind,
      v.missingItems.join(', '),
      v.kind === 'object_fall' ? '' : (v.helmet ? 'Yes' : 'No'),
      v.kind === 'object_fall' ? '' : (v.vest ? 'Yes' : 'No'),
      photoCell,
      snapshotCell,
      v.id,
      v.sessionId,
      v.userId || '',
      v.bbox ? JSON.stringify(v.bbox) : '',
    ];
    // photoCell/snapshotCell are the two cells allowed to start with "="
    // (IMAGE/HYPERLINK formulas); csvEscape would otherwise treat them
    // like any other string, which is fine - it only quotes on
    // comma/quote/newline and these formula strings contain commas, so
    // they get quoted, which Excel/Sheets still parses correctly as a
    // formula.
    lines.push(cells.map(csvEscape).join(','));
  }
  return lines.join('\n');
}

module.exports = {
  VIOLATIONS_DIR,
  init,
  createUser,
  findUserByUsername,
  findUserByEmail,
  findUserById,
  countUsers,
  isLockedOut,
  recordFailedLogin,
  resetLoginAttempts,
  startSession,
  endSession,
  setSessionDemoMode,
  listSessions,
  getSession,
  upsertWorker,
  persistWorkers,
  listWorkers,
  addViolation,
  listViolations,
  toCSV,
  violationsToCSV,
};
