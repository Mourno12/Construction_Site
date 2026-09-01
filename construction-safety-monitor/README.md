# EdgeGuard — Construction Site Safety Monitoring System

Real-time AI-powered construction site safety monitoring. Detects workers,
checks 5 PPE items (helmet, vest, gloves, boots, mask), flags a worker who's
fallen down, flags a falling object, tracks each worker with a persistent
ID, and streams everything to a live dashboard — from a webcam, an uploaded
video, or an uploaded image.

## Highlights

- **8 things detected/tracked per frame:**
  1. Helmet or no helmet
  2. Vest or no vest
  3. Gloves or no gloves
  4. Boots or no boots
  5. Mask or no mask
  6. Persistent worker tracking (ByteTrack / IOU fallback)
  7. Worker fallen down
  8. Object falling (tools/materials/debris)
- **YOLOv12n** (Ultralytics) person + (optionally) object detection — attention-centric architecture for strong accuracy
- **Fine-tuned Vision Transformer (ViT)**, one model, six sigmoid outputs: helmet, vest, gloves, boots, mask, and "fallen" — all trained the same way, on labelled worker-crop images (see **Training your own ViT classifier** below)
- **ByteTrack** persistent worker IDs — a worker keeps the same ID even after leaving and re-entering frame
- **Object-fall detection**: a velocity-based rule on top of the (trainable) object detector + tracker — falling is a property of motion across frames, not of one static image, so there's no single-frame model to train for "is this falling"; see **Object-fall detection** below for how the detector side is trained
- **Full-screen alerts**: a pulsing red screen-edge flash + a large banner (title + detail, differs for PPE violation / worker down / falling object), plus a 30-second soft repeating audio pulse — built to be noticed from across a room, not just by whoever's looking at the screen; a mute toggle lives in the top bar
- **Node.js + Express + WebSocket** backend — sub-100ms streaming to the UI
- **3 input modes**: live webcam, video upload, image upload
- **Accounts**: register/log in/log out (JWT-based); the first account created becomes the site admin
- **Persistent logs**: every session, per-worker compliance history, and violation (PPE / fall / object-fall, each with an evidence snapshot) is saved to disk and survives server restarts
- **CSV/JSON export**: download the violation log, a session's worker log, or the full session history straight from the dashboard
- **Full ViT training pipeline** included (`python/train_vit.py`) for transfer learning on your own PPE + fall dataset
- **Runs out of the box in "Demo Mode"** — even before you install the full ML stack or train a model, the app runs end-to-end using an OpenCV person detector + a colour/geometry-heuristic classifier (PPE colours + bounding-box aspect ratio for "fallen"), so you always have something to demo. Swap in the real trained models any time.

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
Python inference worker              data/edgeguard.db + data/violations/*.jpg
(python/inference_worker.py)         (users, sessions, worker history,
   ├─ detector.py (person + object)   violations with evidence snapshots,
   ├─ ppe_classifier.py (6 labels)     each tagged kind: ppe | fall | object_fall)
   ├─ tracker.py (fallback)
   ├─ fall_events.py (object velocity)
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
│   ├── detector.py          YOLOv12n (person + optional object class) + fallback HOG detector
│   ├── ppe_classifier.py    fine-tuned ViT (6 labels) + fallback colour/geometry heuristic
│   ├── tracker.py            fallback IOU tracker (person-only, Demo Mode)
│   ├── fall_events.py        velocity-based object-fall detector for tracked objects
│   ├── drawing.py            annotation / HUD drawing helpers
│   ├── train_vit.py          full ViT training pipeline (helmet/vest/gloves/boots/mask/fallen)
│   ├── dataset.py            dataset loader + skeleton generator
│   ├── config.py             all labels / thresholds / paths / performance knobs
│   ├── requirements.txt
│   └── models/                put your trained checkpoint at models/vit_ppe_classifier/
├── public/                  Dashboard frontend (vanilla HTML/CSS/JS)
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── data/                    persisted store (auto-created)
│   ├── edgeguard.db           users, sessions, worker history, violations (SQLite)
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
  one account still gets throttled. This is tracked in `data/edgeguard.db` and
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
(`data/edgeguard.db`), with a full-frame evidence snapshot saved to
`data/violations/`. It uses [sql.js](https://github.com/sql-js/sql.js)
(SQLite compiled to WebAssembly) rather than a native module like
`better-sqlite3` — that's a deliberate choice so `npm install` never needs a
C++ build toolchain or a matching prebuilt binary on any OS; it's plain
WebAssembly, so it just works everywhere `npm install` does.

The file is a genuine SQLite database — open it with any standard SQLite
tool/library (`sqlite3 data/edgeguard.db`, DB Browser for SQLite, Python's
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

| Stage             | Full pipeline                | Demo Mode fallback                                          |
|-------------------|-------------------------------|----------------------------------------------------------------|
| Detection         | YOLOv12n                     | OpenCV HOG person detector                                    |
| Tracking          | ByteTrack (via ultralytics)   | Simple IOU-matching tracker                                    |
| PPE check (×5)    | Fine-tuned ViT                 | HSV colour-heuristic per item (hard-hat/hi-vis/glove/boot/mask colours) |
| Fallen detection  | Fine-tuned ViT ("fallen" output) | Bounding-box aspect ratio (wider-than-tall = fallen)         |
| Object-fall       | Same tracked-object velocity rule either way — see below | (n/a) |

A **"DEMO MODE"** badge appears on the live feed and dashboard whenever any
fallback is active, so it's always clear which mode you're running in. This
means you can demo the entire product — UI, tracking, alerts, worker log,
accounts, and persisted history — before you've trained anything.

Note that **object-fall detection only ever runs on the real YOLOv12n path**:
the fallback HOG detector only finds people, so there's nothing for it to
track as a falling object in Demo Mode (see **Object-fall detection**
below).

## Alerts

Three alert kinds, all persisted to the violation log with an evidence
snapshot: `ppe` (missing required PPE for `ALERT_STREAK_FRAMES` consecutive
frames), `fall` (worker detected as fallen for the same streak), and
`object_fall` (a tracked object falling — see **Object-fall detection**).
Each one:
- Flashes a pulsing red glow around the whole viewport plus a large banner
  (title + specifics, e.g. "Worker #4 missing helmet & vest" /
  "Worker Down" / "Falling Object") — sized to be noticed from across a
  room, not just by whoever's looking at the screen right now.
- Plays a soft repeating audio pulse for 30 seconds (a fresh alert re-arms
  the full 30s rather than queuing behind an old one).
- Adds an entry to the Violation Alerts feed and the Session History
  evidence gallery, colour-coded per kind (red = PPE, magenta = fall,
  amber = object fall).

A speaker-icon button in the top bar mutes/unmutes the sound (persisted
across sessions); muting also cuts an in-progress 30s pulse immediately.

## Training your own ViT classifier

One ViT checkpoint handles all 6 labels — helmet, vest, gloves, boots,
mask, and fallen — as six sigmoid outputs, in the order defined by
`config.ALL_TRAINABLE_LABELS`.

1. Generate the dataset skeleton:
   ```bash
   cd python
   python dataset.py
   ```
2. Drop your worker-crop images into `python/dataset/images/`, and create
   `train_annotations.csv` / `val_annotations.csv` with columns:
   ```
   filename,helmet,vest,gloves,boots,mask,fallen
   img_0001.jpg,1,1,0,1,0,0
   img_0002.jpg,0,1,1,1,0,0
   img_0003.jpg,1,1,1,1,1,1
   ```
   `fallen` = 1 for a crop where the worker is down/lying/slumped, not
   standing. Tip: run the app in Demo Mode on your own site footage first,
   export the per-worker crops, and hand-correct the heuristic labels —
   much faster than annotating from a blank slate.
3. Train:
   ```bash
   python train_vit.py --epochs 15 --batch_size 32
   ```
   The best checkpoint (by validation F1, computed across all 6 labels) is
   saved to `python/models/vit_ppe_classifier/`.
4. Restart `npm start` — the app will detect the checkpoint and switch out
   of Demo Mode automatically for PPE + fall detection.

Want to track a different set of items (e.g. drop `mask`, add `earmuffs`)?
Edit `PPE_LABELS` in `python/config.py` — `dataset.py` and `train_vit.py`
both read that list dynamically, so the CSV columns and model output count
follow automatically. Also check `REQUIRED_PPE_ITEMS` there: it controls
which items must be present for a worker to count as "compliant" (green,
not just logged) — trim it if e.g. gloves/mask aren't mandatory on your site.

## Object-fall detection

Detecting an object mid-fall doesn't work like PPE classification: a single
still crop of an object in mid-air often looks identical to one sitting in
a normal position — "falling" is a property of *motion across frames*, not
of one frame. So this feature is split into two parts:

1. **Trainable part — detecting the object at all.** The stock
   `yolo12n.pt` COCO weights only know "person" (plus generic COCO
   classes); there's no "construction debris/tool" class. Fine-tune your
   own YOLOv12n with an extra class for whatever you want watched (a
   dropped tool, loose material, a falling brick, etc.), the same way
   you'd train any Ultralytics YOLO model, outside this repo:
   ```bash
   yolo detect train model=yolo12n.pt data=your_data.yaml epochs=100 imgsz=640
   ```
   `your_data.yaml` needs images + bounding-box label files for your object
   class(es), plus `person` if you want to keep detecting people with the
   same model (or point `YOLO_WEIGHTS` at this new model and keep
   `YOLO_PERSON_CLASS_ID` matching whatever id your dataset assigns to
   "person").
2. **Physics rule — deciding it's "falling".** Once that model can detect
   and track the object (via the same ByteTrack path used for people),
   `python/fall_events.py` watches each tracked object's vertical bbox-centre
   position frame over frame. If it's moving down fast enough
   (`OBJECT_FALL_MIN_DOWNWARD_PX_PER_FRAME`) for long enough
   (`OBJECT_FALL_STREAK_FRAMES`, both in `config.py`), that's an
   "object_fall" alert — debounced the same way PPE violations are, so a
   sustained fall reports once, not once per frame.

To turn it on: set `YOLO_OBJECT_CLASS_IDS` in `config.py` (or the
`YOLO_OBJECT_CLASS_IDS` env var, comma-separated) to your fine-tuned
model's object class id(s), and point `YOLO_WEIGHTS` at that model. Left
empty (the default), object-fall detection is simply inert — no object
tracks ever exist to evaluate, and person detection/PPE/fall checking work
exactly as before.

## Using your own YOLO weights

Fine-tuned a custom YOLOv12n on your own site's camera angles/PPE colours,
or added an object class for object-fall detection? Point
`python/config.py`'s `YOLO_WEIGHTS` at your `.pt` file (or set the
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
