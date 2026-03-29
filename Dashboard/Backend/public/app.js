const API_BASE = "http://localhost:5000";

// ================= WEBSOCKET =================
const ws = new WebSocket("ws://localhost:5000");

ws.onmessage = (event) => {
    console.log("WS:", event.data); // DEBUG

    const message = JSON.parse(event.data);

    if (message.type === "tracking_update") {
        updateTrackingData(message.data);
    }
};

// ================= TRACKING =================
function updateTrackingData(data) {

    if (data.stats) {
        document.getElementById("statTotalWorkers").textContent = data.stats.total_workers || 0;
        document.getElementById("statSafe").textContent = data.stats.safe_workers || 0;
        document.getElementById("statUnsafe").textContent = data.stats.unsafe_workers || 0;

        const c = data.stats.compliance_rate || 0;
        document.getElementById("statCompliance").textContent = c.toFixed(1) + "%";

        const bar = document.getElementById("complianceBar");
        bar.style.width = c + "%";
        bar.textContent = c.toFixed(1) + "%";

        if (data.stats.fps) {
            document.getElementById("fpsBadge").textContent = "FPS: " + data.stats.fps.toFixed(1);
        }
    }

    if (data.worker_logs) {
        updateWorkerTable(data.worker_logs);
    }
}

// ================= TABLE FIX =================
function updateWorkerTable(workers) {

    const tbody = document.getElementById("workerTableBody");

    if (!workers || workers.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7">No workers</td></tr>`;
        return;
    }

    tbody.innerHTML = workers.map(w => {

        const id = w.worker_id ?? w.id ?? "N/A";
        const total = w.total_frames ?? w.frames ?? 0;
        const safe = w.safe_frames ?? 0;
        const helmet = w.no_helmet_frames ?? 0;
        const vest = w.no_vest_frames ?? 0;
        const comp = w.compliance_rate ?? 0;

        let status = "✅";
        if (comp < 50) status = "❌";
        else if (comp < 80) status = "⚠️";

        return `
        <tr>
            <td>${id}</td>
            <td>${total}</td>
            <td>${safe}</td>
            <td>${helmet}</td>
            <td>${vest}</td>
            <td>${comp.toFixed(1)}%</td>
            <td>${status}</td>
        </tr>
        `;
    }).join("");
}

// ================= IMAGE =================
async function uploadImage() {
    const file = document.getElementById("imageInput").files[0];
    if (!file) return alert("Select image");

    setMode("processing");

    const fd = new FormData();
    fd.append("image", file);

    const res = await fetch(API_BASE + "/api/process_image", {
        method: "POST",
        body: fd
    });

    const data = await res.json();

    if (data.success) {
        showImage(data.output_path);
    }
}

// ================= VIDEO =================
async function uploadVideo() {
    const file = document.getElementById("videoInput").files[0];
    if (!file) return alert("Select video");

    setMode("processing");

    const fd = new FormData();
    fd.append("video", file);

    const res = await fetch(API_BASE + "/api/process_video", {
        method: "POST",
        body: fd
    });

    const data = await res.json();

    if (data.success) {
        showVideo(data.video_url);
    }
}

// ================= WEBCAM =================
let stream = null;

async function startWebcam() {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });

    const video = document.getElementById("webcamOutput");
    video.srcObject = stream;

    hideAll();
    video.style.display = "block";

    setMode("webcam");

    sendFrames(video);
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }
}

function sendFrames(video) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    setInterval(async () => {

        if (!video.videoWidth) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        ctx.drawImage(video, 0, 0);

        canvas.toBlob(async blob => {
            const fd = new FormData();
            fd.append("frame", blob);

            await fetch(API_BASE + "/api/process_frame", {
                method: "POST",
                body: fd
            });

        }, "image/jpeg");

    }, 300);
}

// ================= UI =================
function hideAll() {
    document.getElementById("imageOutput").style.display = "none";
    document.getElementById("videoOutput").style.display = "none";
    document.getElementById("webcamOutput").style.display = "none";
    document.getElementById("outputPlaceholder").style.display = "none";
}

function showImage(src) {
    hideAll();
    const img = document.getElementById("imageOutput");
    img.src = src;
    img.style.display = "block";
    setMode("image");
}

function showVideo(src) {
    hideAll();
    const vid = document.getElementById("videoOutput");
    vid.src = src;
    vid.style.display = "block";
    setMode("video");
}

function setMode(m) {
    document.getElementById("modeBadge").textContent = m.toUpperCase();
}

// ================= RESET =================
function resetStats() {
    location.reload();
}