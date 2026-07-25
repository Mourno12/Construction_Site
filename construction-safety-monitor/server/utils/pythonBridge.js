/**
 * pythonBridge.js
 * Spawns the long-lived Python inference worker (python/inference_worker.py)
 * once at server startup, and exposes a simple event-based API for the rest
 * of the Node backend to send commands / receive results, speaking NDJSON
 * (newline-delimited JSON) over the child process's stdin/stdout.
 */

const { spawn, spawnSync } = require('child_process');
const path = require('path');
const { EventEmitter } = require('events');

const PYTHON_DIR = path.join(__dirname, '..', '..', 'python');
const WORKER_SCRIPT = path.join(PYTHON_DIR, 'inference_worker.py');

/**
 * Figures out which command actually runs Python on this machine.
 * Different installs expose different names - Windows commonly only has
 * `py` (the launcher), some Linux distros only have `python3`, some have
 * both. Rather than hardcoding one guess per platform (which breaks with a
 * confusing `spawn ENOENT` the moment it's wrong), try a short list of
 * candidates and use the first one that actually runs.
 *
 * An explicit PYTHON_BIN env var (e.g. for a venv) always wins.
 */
function resolvePythonBin() {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;

  const candidates = process.platform === 'win32'
    ? ['py', 'python', 'python3']
    : ['python3', 'python'];

  for (const candidate of candidates) {
    const result = spawnSync(candidate, ['--version'], { stdio: 'ignore' });
    if (!result.error) return candidate;
  }

  // Nothing worked - fall back to the most likely name so the resulting
  // error message at least names something the person can search for.
  return candidates[0];
}

const PYTHON_BIN = resolvePythonBin();

class PythonBridge extends EventEmitter {
  constructor() {
    super();
    this.proc = null;
    this.ready = false;
    this.demoMode = null;
    this._stdoutBuffer = '';
    this._stderrBuffer = '';
  }

  start() {
    console.log(`[pythonBridge] Spawning inference worker: ${PYTHON_BIN} ${WORKER_SCRIPT}`);
    this.proc = spawn(PYTHON_BIN, ['-u', WORKER_SCRIPT], {
      cwd: PYTHON_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    this.proc.stdout.setEncoding('utf8');
    this.proc.stdout.on('data', (chunk) => this._handleStdout(chunk));

    this.proc.stderr.setEncoding('utf8');
    this.proc.stderr.on('data', (chunk) => this._handleStderr(chunk));

    this.proc.on('exit', (code, signal) => {
      console.error(`[pythonBridge] worker exited (code=${code}, signal=${signal})`);
      this.ready = false;
      this.emit('exit', { code, signal });
    });

    this.proc.on('error', (err) => {
      console.error('[pythonBridge] failed to spawn worker:', err.message);
      if (err.code === 'ENOENT') {
        console.error(
          `[pythonBridge] Could not run "${PYTHON_BIN}". Make sure Python is installed and on your PATH, ` +
          'or set the PYTHON_BIN environment variable to the exact command that works on your machine ' +
          '(e.g. "py" on Windows, or the full path to python.exe).'
        );
      }
      this.emit('spawnError', err);
    });

    return this;
  }

  _handleStdout(chunk) {
    this._stdoutBuffer += chunk;
    let lines = this._stdoutBuffer.split('\n');
    this._stdoutBuffer = lines.pop(); // last (possibly incomplete) line stays buffered

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let msg;
      try {
        msg = JSON.parse(trimmed);
      } catch (e) {
        console.warn('[pythonBridge] non-JSON stdout line ignored:', trimmed.slice(0, 200));
        continue;
      }

      if (msg.type === 'ready') {
        this.ready = true;
        this.demoMode = msg.demoMode;
        console.log(`[pythonBridge] worker ready. detector=${msg.detector} classifier=${msg.classifier} demoMode=${msg.demoMode}`);
        this.emit('ready', msg);
        continue;
      }
      this.emit('message', msg);
    }
  }

  _handleStderr(chunk) {
    this._stderrBuffer += chunk;
    let lines = this._stderrBuffer.split('\n');
    this._stderrBuffer = lines.pop();
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const msg = JSON.parse(trimmed);
        if (msg.type === 'log') {
          const level = msg.level === 'error' ? 'error' : msg.level === 'warn' ? 'warn' : 'log';
          console[level](`[python] ${msg.message}`);
          continue;
        }
      } catch (e) {
        // not JSON - just a raw stderr line (e.g. a python traceback)
      }
      console.error(`[python:stderr] ${trimmed}`);
    }
  }

  send(obj) {
    if (!this.proc || this.proc.killed) {
      throw new Error('Python worker is not running');
    }
    this.proc.stdin.write(JSON.stringify(obj) + '\n');
  }

  stop() {
    if (this.proc) {
      this.proc.kill();
    }
  }
}

module.exports = new PythonBridge();
