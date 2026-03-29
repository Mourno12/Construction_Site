// server.js - FIXED + REAL-TIME + BETTER STATS + WEBCAM READY

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
const UPLOAD_DIR = path.join(__dirname, 'uploads');
const OUTPUT_DIR = path.join(__dirname, 'output');
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
function executePython(script, args = []) {
    return new Promise((resolve, reject) => {
        const py = spawn(PYTHON_PATH, [script, ...args]);

        let output = '';
        let error = '';

        py.stdout.on('data', d => output += d.toString());
        py.stderr.on('data', d => error += d.toString());

        py.on('close', code => {
            if (code !== 0) {
                console.error("Python Error:", error);
                return reject(error);
            }

            // Extract JSON safely
            const match = output.match(/\{[\s\S]*\}/);
            if (!match) return reject("No JSON output");

            try {
                resolve(JSON.parse(match[0]));
            } catch (e) {
                reject("Invalid JSON");
            }
        });
    });
}

// ===================== COMPUTE STATS =====================
function computeStats(workerLogs) {
    let total = workerLogs.length;
    let safe = 0;
    let unsafe = 0;

    workerLogs.forEach(w => {
        if (w.compliance_rate >= 80) safe++;
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

// ===================== RUN DETECTION =====================
async function runDetection(mode, input, output) {
    return await executePython('Detection/detection_api.py', [mode, input, output]);
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

        const result = await runDetection('image', input, output);

        res.json({
            success: true,
            output_path: `/output/${outputFile}`,
            analysis: result.analysis || {}
        });

    } catch (err) {
        res.status(500).json({ success: false, message: err.toString() });
    }
});

// ===================== VIDEO =====================
app.post('/api/process_video', upload.single('video'), async (req, res) => {
    try {
        const input = req.file.path;
        const outputFile = `video_${Date.now()}.mp4`;
        const output = path.join(OUTPUT_DIR, outputFile);

        const result = await runDetection('video', input, output);

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
        res.status(500).json({ success: false, message: err.toString() });
    }
});

// ===================== WEBCAM (NEW 🔥) =====================
app.get('/api/webcam', (req, res) => {
    res.json({
        success: true,
        stream_url: "http://localhost:5000/webcam_stream"
    });
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