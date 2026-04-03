// server.js -

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const http = require('http');
const WebSocket = require('ws');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = 5000;

// ✅ IMPORTANT PATH FIX
const ROOT_DIR = path.join(__dirname, '..'); // go to Dashboard root
const DETECTION_SCRIPT = path.join(ROOT_DIR, 'Detection', 'detection_api.py');

const UPLOAD_DIR = path.join(__dirname, 'uploads');
const OUTPUT_DIR = path.join(__dirname, 'output');

// ✅ safer python command (Windows)
const PYTHON_PATH = 'python';

// ===================== MIDDLEWARE =====================
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/output', express.static(OUTPUT_DIR));
app.use('/uploads', express.static(UPLOAD_DIR));
app.use(express.static(path.join(__dirname, 'public')));

// Create folders
[UPLOAD_DIR, OUTPUT_DIR].forEach(dir => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

// ===================== MULTER =====================
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOAD_DIR),
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname);
        cb(null, `${file.fieldname}_${Date.now()}${ext}`);
    }
});
const upload = multer({ storage });

// ===================== GLOBAL STATE =====================
let trackingData = {
    worker_logs: [],
    compliance_history: [],
    stats: {
        total_workers: 0,
        safe_workers: 0,
        unsafe_workers: 0,
        compliance_rate: 0,
        fps: 0
    }
};

// ===================== PYTHON EXECUTION =====================
function executePython(mode, input, output) {
    return new Promise((resolve, reject) => {

        console.log("▶ Running Python:", DETECTION_SCRIPT);

        const py = spawn(PYTHON_PATH, [
            DETECTION_SCRIPT,
            mode,
            input,
            output
        ], {
            cwd: ROOT_DIR   // ✅ VERY IMPORTANT
        });

        let stdout = '';
        let stderr = '';

        py.stdout.on('data', (data) => {
            stdout += data.toString();
            console.log("PYTHON OUT:", data.toString());
        });

        py.stderr.on('data', (data) => {
            stderr += data.toString();
            console.error("PYTHON ERR:", data.toString());
        });

        py.on('close', (code) => {
            console.log("Python exit code:", code);

            if (code !== 0) {
                return reject(stderr || "Python failed");
            }

            // ✅ Extract JSON safely
            const match = stdout.match(/\{[\s\S]*\}/);

            if (!match) {
                return reject("No JSON output from Python");
            }

            try {
                const json = JSON.parse(match[0]);
                resolve(json);
            } catch (e) {
                reject("Invalid JSON from Python");
            }
        });

        py.on('error', (err) => {
            reject("Failed to start Python: " + err.message);
        });
    });
}

// ===================== COMPUTE STATS =====================
function computeStats(workerLogs) {
    let total = workerLogs.length;
    let safe = 0;
    let unsafe = 0;

    workerLogs.forEach(w => {
        if ((w.compliance_rate || 100) >= 80) safe++;
        else unsafe++;
    });

    const compliance = total ? (safe / total) * 100 : 0;

    return {
        total_workers: total,
        safe_workers: safe,
        unsafe_workers: unsafe,
        compliance_rate: compliance
    };
}

// ===================== ROUTES =====================

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'dashboard.html'));
});

// ===================== IMAGE =====================
app.post('/api/process_image', upload.single('image'), async (req, res) => {
    try {
        const input = req.file.path;
        const outputFile = `image_${Date.now()}.jpg`;
        const output = path.join(OUTPUT_DIR, outputFile);

        const result = await executePython('image', input, output);

        res.json({
            success: true,
            output_path: `/output/${outputFile}`,
            analysis: result.analysis || {}
        });

    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: err.toString() });
    }
});

// ===================== VIDEO =====================
app.post('/api/process_video', upload.single('video'), async (req, res) => {
    try {
        const input = req.file.path;
        const outputFile = `video_${Date.now()}.mp4`;
        const output = path.join(OUTPUT_DIR, outputFile);

        const result = await executePython('video', input, output);

        if (result.tracking_data) {
            const logs = result.tracking_data.worker_logs || [];
            const stats = computeStats(logs);

            trackingData = {
                worker_logs: logs,
                compliance_history: result.tracking_data.compliance_history || [],
                stats: {
                    ...stats,
                    fps: result.tracking_data.average_fps || 0
                }
            };

            broadcast();
        }

        res.json({
            success: true,
            video_url: `/output/${outputFile}`
        });

    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: err.toString() });
    }
});

// ===================== RESET =====================
app.post('/api/reset_stats', (req, res) => {
    trackingData = {
        worker_logs: [],
        compliance_history: [],
        stats: {
            total_workers: 0,
            safe_workers: 0,
            unsafe_workers: 0,
            compliance_rate: 0,
            fps: 0
        }
    };

    broadcast();
    res.json({ success: true });
});

// ===================== WEBSOCKET =====================
function broadcast() {
    wss.clients.forEach(c => {
        if (c.readyState === WebSocket.OPEN) {
            c.send(JSON.stringify({
                type: "tracking_update",
                data: trackingData
            }));
        }
    });
}

wss.on('connection', ws => {
    console.log("✓ WS connected");

    ws.send(JSON.stringify({
        type: "tracking_update",
        data: trackingData
    }));
});

// ===================== START =====================
server.listen(PORT, () => {
    console.log(`🚀 Server running: http://localhost:${PORT}`);
});