# SiteGuard — Construction Site Safety Monitoring System

Real-time AI-powered PPE (helmet + safety vest) compliance monitoring for
construction sites. Detects workers, checks PPE compliance, tracks each
worker with a persistent ID, and streams everything to a live dashboard —
from a webcam, an uploaded video, or an uploaded image.

## Highlights

- **YOLOv12n** (Ultralytics) person detection — attention-centric architecture for strong accuracy
- **Fine-tuned Vision Transformer (ViT)** for helmet + vest classification (~95% accuracy on a properly trained checkpoint)
- **ByteTrack** persistent worker IDs — a worker keeps the same ID even after leaving and re-entering frame
- **Node.js + Express + WebSocket** backend — sub-100ms streaming to the UI
- **3 input modes**: live webcam, video upload, image upload
- **Accounts**: register/log in/log out (JWT-based); the first account created becomes the site admin
- **Persistent logs**: every session, per-worker compliance history, and violation (with an evidence snapshot) is saved to disk and survives server restarts
- **CSV/JSON export**: download the violation log, a session's worker log, or the full session history straight from the dashboard
- **Full ViT training pipeline** included (`python/train_vit.py`) for transfer learning on your own PPE dataset
- **Runs out of the box in "Demo Mode"** — even before you install the full ML stack or train a model, the app runs end-to-end using an OpenCV person detector + colour-heuristic PPE classifier, so you always have something to demo. Swap in the real models any time.

> **Heads up on YOLOv12:** Ultralytics' own docs note that YOLO12's
> attention-centric blocks trade some CPU throughput and training stability
> for accuracy compared to YOLO11/YOLO26, and recommend YOLO11 or YOLO26 for
> most production workloads. It's a great fit here on a GPU-equipped demo
> machine; if you deploy to a CPU-only device and notice a speed drop, set
> `YOLO_WEIGHTS=yolo11n.pt` in `python/config.py` (or as an env var) to swap
> back with no other code changes.

## Architecture

```
Browser (dashboard)
   │  login/register (JWT) → webcam frames / upload files
   ▼
Node.js + Express + WebSocket  (server/)
   │  NDJSON over stdin/stdout          │  reads/writes
   ▼                                     ▼
Python inference worker              data/siteguard.db + data/violations/*.jpg
(python/inference_worker.py)         (users, sessions, worker history,
   ├─ detector.py                     violations with evidence snapshots)
   ├─ ppe_classifier.py
   ├─ tracker.py
   └─ drawing.py
```

The Node server spawns **one long-lived Python process** at startup and talks
to it via newline-delimited JSON over stdin/stdout — this avoids the
overhead of spawning a new Python process per frame, which is what makes the
sub-100ms round trip possible. Every result is also written to a small
JSON-file-backed store (`server/utils/db.js`) so sessions, worker history,
and violations (with evidence snapshots) survive a server restart.

## Project structure

```
construction-safety-monitor/
├── server/                 Node.js/Express backend + WebSocket server
│   ├── index.js
│   ├── middleware/            requireAuth.js
│   ├── routes/                 auth.js, upload.js, session.js, logs.js
│   └── utils/                  pythonBridge.js, sessionManager.js, db.js, auth.js
├── python/                 ML pipeline
│   ├── inference_worker.py  main NDJSON worker process (spawned by Node)
│   ├── detector.py          YOLOv12n + fallback HOG detector
│   ├── ppe_classifier.py    fine-tuned ViT + fallback colour heuristic
│   ├── tracker.py            fallback IOU tracker
│   ├── drawing.py            annotation / HUD drawing helpers
│   ├── train_vit.py          full ViT training pipeline
│   ├── dataset.py            PPE dataset loader + skeleton generator
│   ├── config.py             all thresholds / paths / performance knobs
│   ├── requirements.txt
│   └── models/                put your trained checkpoint at models/vit_ppe_classifier/
├── public/                  Dashboard frontend (vanilla HTML/CSS/JS)
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── data/                    persisted store (auto-created)
│   ├── siteguard.db           users, sessions, worker history, violations (SQLite)
│   └── violations/            evidence snapshot images (.jpg)
├── uploads/                 uploaded videos/images land here
├── package.json
└── README.md
```

## Setup

### 1. Install Node dependencies

```bash
npm install
```

### 2. Install Python dependencies

```bash
cd python
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

> **No GPU or don't want to install the full ML stack right now?** Skip this
> step. The app still runs — see **Demo Mode** below.

### 3. Run it

```bash
npm start
```

Then open **http://localhost:3000** and register an account — the first
account created becomes the site admin (see **Accounts, Logs & Exports**
below).

On first run, `ultralytics` will auto-download `yolo12n.pt` (a few MB), and
if you've trained your own ViT checkpoint it will be picked up automatically
from `python/models/vit_ppe_classifier/`.

## Accounts, Roles & Security

**Accounts.** Registration asks for a username, email, and password
(JWT-based auth via `bcryptjs` + `jsonwebtoken`). Email is stored as a
profile field only — nothing is ever sent to it, no verification email, no
password-reset email. The first account you create becomes `admin`;
everyone after is an `operator`.

**Roles are enforced server-side, not just labeled:**
- **Operator** can only ever see, export, or control their own sessions and
  violations — this is checked on every request (`server/routes/logs.js`,
  `server/routes/session.js`), not just hidden in the UI. An operator gets a
  `403` if they try to reach another user's session directly, even by
  guessing/knowing its ID.
- **Admin** sees every session and violation across all operators by
  default, with a toggle in the Session History panel to filter down to
  just their own. Only an admin can access any session's data directly.

**Security measures in place:**
- Passwords hashed with bcrypt; policy requires 8+ characters with at least
  one letter and one number.
- Login errors are generic ("Incorrect username or password") regardless of
  whether the username exists, to avoid leaking which accounts are real.
- Per-IP rate limiting on `/api/auth/login` and `/api/auth/register`
  (`express-rate-limit`) to blunt rapid brute-force bursts.
- Per-username account lockout (15 minutes after 5 failed logins),
  independent of the IP rate limiter, so a slow/distributed attack against
  one account still gets throttled. This is tracked in `data/siteguard.db` and
  survives a server restart.
- Username and email uniqueness enforced at registration.

Log out with the button in the top bar; logging out just discards the token
client-side (tokens also expire server-side after 12h regardless).

Set a real `JWT_SECRET` environment variable before deploying this anywhere
beyond your own machine:
```bash
JWT_SECRET="something-long-and-random" npm start
```

**What this auth setup is *not*:** there's no email verification, no
password reset flow, and no 2FA. Tokens live in `localStorage`, which is
fine for a local demo but is more exposed to XSS than an httpOnly cookie
would be — worth revisiting before any real deployment.

**Persistent logs.** Every monitoring session (webcam run, video upload, or
image scan), the per-worker compliance history within it, and every
violation (the moment a worker crosses the "missing PPE for N consecutive
frames" threshold) is written to a real **SQLite database**
(`data/siteguard.db`), with a full-frame evidence snapshot saved to
`data/violations/`. It uses [sql.js](https://github.com/sql-js/sql.js)
(SQLite compiled to WebAssembly) rather than a native module like
`better-sqlite3` — that's a deliberate choice so `npm install` never needs a
C++ build toolchain or a matching prebuilt binary on any OS; it's plain
WebAssembly, so it just works everywhere `npm install` does.

The file is a genuine SQLite database — open it with any standard SQLite
tool/library (`sqlite3 data/siteguard.db`, DB Browser for SQLite, Python's
`sqlite3` module, etc.) to inspect or query it directly. Real schema, real
foreign keys, real indexes (`server/utils/db.js`), not a hand-rolled JSON
blob. The one trade-off: sql.js keeps the whole database in memory and
writes it back to disk on change (debounced on hot paths like per-frame
worker updates, immediate on things like account creation or a violation),
same general shape as before but with actual SQL doing the querying now
instead of manual JS array filtering.

**Exports.** From the dashboard:
- The **Worker Tracking Log** panel can export the *current* session's
  worker history as CSV or JSON.
- The **Session History & Reports** panel lists your past sessions (with a
  per-session CSV export) and a scrollable gallery of violation evidence
  snapshots, plus buttons to export *all* your violations as CSV or JSON.

These also work as plain authenticated REST endpoints, e.g.:
```bash
curl "http://localhost:3000/api/logs/violations/export?format=csv&mine=true" \
  -H "Authorization: Bearer <token>"
```

## Demo Mode

If `ultralytics`/`torch`/`transformers` aren't installed, or no fine-tuned
ViT checkpoint is found yet, the pipeline **automatically falls back** to:

| Stage       | Full pipeline              | Demo Mode fallback                     |
|-------------|-----------------------------|-----------------------------------------|
| Detection   | YOLOv12n                   | OpenCV HOG person detector             |
| Tracking    | ByteTrack (via ultralytics) | Simple IOU-matching tracker            |
| PPE check   | Fine-tuned ViT               | HSV colour-heuristic (hard-hat/hi-vis colours) |

A **"DEMO MODE"** badge appears on the live feed and dashboard whenever any
fallback is active, so it's always clear which mode you're running in. This
means you can demo the entire product — UI, tracking, alerts, worker log,
accounts, and persisted history — before you've trained anything.

## Training your own ViT PPE classifier

1. Generate the dataset skeleton:
   ```bash
   cd python
   python dataset.py
   ```
2. Drop your worker-crop images into `python/dataset/images/`, and create
   `train_annotations.csv` / `val_annotations.csv` with columns:
   ```
   filename,helmet,vest
   img_0001.jpg,1,1
   img_0002.jpg,0,1
   ```
   Tip: run the app in Demo Mode on your own site footage first, export the
   per-worker crops, and hand-correct the heuristic labels — much faster
   than annotating from a blank slate.
3. Train:
   ```bash
   python train_vit.py --epochs 15 --batch_size 32
   ```
   The best checkpoint (by validation F1) is saved to
   `python/models/vit_ppe_classifier/`.
4. Restart `npm start` — the app will detect the checkpoint and switch out
   of Demo Mode automatically.

## Using your own YOLO weights

Fine-tuned a custom YOLOv12n on your own site's camera angles/PPE colours?
Point `python/config.py`'s `YOLO_WEIGHTS` at your `.pt` file (or set the
`YOLO_WEIGHTS` environment variable before starting the server).

## Notes on the 3 input modes

- **Webcam**: captured client-side (browser `getUserMedia`) and streamed to
  the backend frame-by-frame over WebSocket — works with any device camera,
  not just one attached to the server.
- **Upload Video**: uploaded via REST, then processed server-side frame by
  frame; results stream back over the same WebSocket session in real time.
- **Upload Image**: uploaded and analyzed once; result renders immediately.

## Tech stack

Python, PyTorch, YOLOv12n (Ultralytics), Vision Transformer (ViT), Hugging
Face Transformers, ByteTrack, OpenCV, Node.js, Express, WebSocket (`ws`),
JWT auth (`jsonwebtoken`, `bcryptjs`), HTML, CSS, JavaScript.

## License

MIT — built for hackathon / academic use. The JSON-file store and JWT auth
here are meant to be "good enough to demo," not a production security
posture — set a real `JWT_SECRET`, add HTTPS, and consider a real database
before deploying this to an actual job site.
