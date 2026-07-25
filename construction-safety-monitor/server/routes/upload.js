/**
 * upload.js
 * Handles video / image uploads used by the "Upload Video" and
 * "Upload Image" input modes.
 */

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const requireAuth = require('../middleware/requireAuth');

const router = express.Router();
router.use(requireAuth);

const UPLOAD_DIR = path.join(__dirname, '..', '..', 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${Date.now()}-${uuidv4()}${ext}`);
  },
});

const MAX_FILE_SIZE_MB = 500;
const upload = multer({
  storage,
  limits: { fileSize: MAX_FILE_SIZE_MB * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const okVideo = /^video\//.test(file.mimetype);
    const okImage = /^image\//.test(file.mimetype);
    if (req.path.includes('video') && !okVideo) {
      return cb(new Error('Only video files are accepted on this endpoint'));
    }
    if (req.path.includes('image') && !okImage) {
      return cb(new Error('Only image files are accepted on this endpoint'));
    }
    cb(null, true);
  },
});

// POST /api/upload/video
router.post('/video', upload.single('video'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No video file received' });
  const sessionId = uuidv4();
  res.json({
    sessionId,
    filename: req.file.filename,
    originalName: req.file.originalname,
    sizeBytes: req.file.size,
    path: req.file.path,
  });
});

// POST /api/upload/image
router.post('/image', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image file received' });
  const sessionId = uuidv4();
  const imageBase64 = fs.readFileSync(req.file.path, { encoding: 'base64' });
  res.json({
    sessionId,
    filename: req.file.filename,
    originalName: req.file.originalname,
    sizeBytes: req.file.size,
    image: `data:${req.file.mimetype};base64,${imageBase64}`,
  });
});

module.exports = router;
