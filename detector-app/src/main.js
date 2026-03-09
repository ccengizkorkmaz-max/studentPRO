import { ObjectDetector, FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './style.css';
import labelsTr from './labels.js';

// DOM Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas-overlay');
const ctx = canvas.getContext('2d');
const loadingScreen = document.getElementById('loading-screen');
const loadingText = document.getElementById('loading-text');
const toggleBtn = document.getElementById('toggle-camera');

// Stats Elements
const occupancyVal = document.getElementById('occupancy-val');
const avgFocusVal = document.getElementById('avg-focus-val');
const activeAlarms = document.getElementById('active-alarms');
const moodIndicator = document.getElementById('mood-indicator');
const moodProgress = document.getElementById('mood-progress');
const focusProgress = document.getElementById('focus-progress');
const activityFeed = document.getElementById('activity-feed');
const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');

// Global Engines
let objectDetector = null;
let faceLandmarker = null;
let isAnalysing = false;
let lastVideoTime = -1;
let minConfidence = 0.6;

/**
 * Initialize MediaPipe AI Models
 */
async function initAI() {
  try {
    loadingText.innerText = "Azure AI Core Hazırlanıyor...";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    loadingText.innerText = "EfficientDet Pro Sınıf Modeli Yükleniyor...";
    objectDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
        delegate: "GPU"
      },
      scoreThreshold: minConfidence,
      runningMode: "VIDEO"
    });

    loadingText.innerText = "Duygu Analiz Motoru Yükleniyor...";
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      outputFaceBlendshapes: true,
      runningMode: "VIDEO",
      numFaces: 1
    });

    loadingText.innerText = "Kurumsal Sistem Hazır!";
    setTimeout(() => {
      loadingScreen.style.opacity = '0';
      setTimeout(() => loadingScreen.style.display = 'none', 800);
    }, 1000);

  } catch (error) {
    console.error("AI Init Error:", error);
    loadingText.innerText = "Hata! Lütfen bağlantıyı kontrol edin.";
  }
}

/**
 * Start Camera
 */
async function triggerCamera() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false
    });
    video.srcObject = stream;
    return new Promise(r => video.onloadedmetadata = r);
  }
}

/**
 * Main AI Pipeline
 */
async function runPulse() {
  if (!isAnalysing) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();

    // 1. Detect Objects
    const objResults = objectDetector.detectForVideo(video, now);

    // 2. Detect Face & Emotions
    const faceResults = faceLandmarker.detectForVideo(video, now);

    // Clear Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    processDetections(objResults.detections, faceResults);
  }

  requestAnimationFrame(runPulse);
}

/**
 * Process AI Results
 */
function processDetections(objects, faces) {
  const displayW = canvas.width;
  const displayH = canvas.height;
  const videoW = video.videoWidth;
  const videoH = video.videoHeight;

  // Filter & Overlap Cleaning
  const validObjects = objects
    .filter(det => ['person', 'cell phone'].includes(det.categories[0].categoryName))
    .sort((a, b) => (b.boundingBox.width * b.boundingBox.height) - (a.boundingBox.width * a.boundingBox.height));

  let personCount = 0;
  let hasPhone = false;
  let personDrawn = false;

  validObjects.forEach(det => {
    const label = det.categories[0].categoryName;

    if (label === 'person') {
      if (personDrawn) return;
      personCount++;
      personDrawn = true;
    }
    if (label === 'cell phone') hasPhone = true;

    const { originX, originY, width, height } = det.boundingBox;
    const sx = (originX / videoW) * displayW;
    const sy = (originY / videoH) * displayH;
    const sw = (width / videoW) * displayW;
    const sh = (height / videoH) * displayH;

    // Microsoft Blue / Danger Red
    ctx.strokeStyle = label === 'cell phone' ? '#a4262c' : '#0078d4';
    ctx.lineWidth = 2;
    ctx.strokeRect(sx, sy, sw, sh);

    // Refined Label
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "bold 11px 'Segoe UI'";
    const trName = labelsTr[label] || label;
    ctx.fillRect(sx, sy - 20, ctx.measureText(trName).width + 10, 20);
    ctx.fillStyle = "white";
    ctx.fillText(trName, sx + 5, sy - 5);
  });

  // Mood Logic
  let faceVisible = false;
  if (faces.faceLandmarks && faces.faceLandmarks.length > 0) {
    faceVisible = true;
    const blendshapes = faces.faceBlendshapes[0].categories;
    const smile = (blendshapes.find(b => b.categoryName === 'mouthSmileLeft').score +
      blendshapes.find(b => b.categoryName === 'mouthSmileRight').score) / 2;

    if (smile > 0.4) {
      updateMood("Mutlu 😊", smile * 100);
    } else {
      updateMood("Nötr 😐", 40 + (smile * 100));
    }
  }

  updateRealtimeStats(personCount, faceVisible, hasPhone);
}

function updateMood(label, score) {
  moodIndicator.innerText = label;
  moodProgress.style.width = `${score}%`;
}

function updateRealtimeStats(count, faceVisible, phone) {
  occupancyVal.innerText = `${count} / 30`;

  const focus = faceVisible ? 95 : 15;
  focusProgress.style.width = `${focus}%`;
  avgFocusVal.innerText = `%${focus}`;

  if (phone) {
    activeAlarms.innerText = "TELEFON!";
    activeAlarms.classList.add('pulse-text');
    addFeedItem("⚠️ KRİTİK: Öğrenci telefonla ilgileniyor!");
  } else if (!faceVisible && count > 0) {
    activeAlarms.innerText = "ODAK KAYBI";
    activeAlarms.classList.remove('pulse-text');
    addFeedItem("❗ UYARI: Öğrenci odağını kaybetti.");
  } else {
    activeAlarms.innerText = "Yok";
    activeAlarms.classList.remove('pulse-text');
  }
}

function addFeedItem(text) {
  if (activityFeed.firstChild && activityFeed.firstChild.innerText.includes(text)) return;
  const div = document.createElement('div');
  div.className = 'feed-item';
  div.innerHTML = `<span class="fi-time">${new Date().toLocaleTimeString()}</span><span class="fi-text">${text}</span>`;
  activityFeed.prepend(div);
  if (activityFeed.children.length > 5) activityFeed.lastChild.remove();
}

// Event Listeners
toggleBtn.addEventListener('click', async () => {
  if (!isAnalysing) {
    await triggerCamera();
    video.play();
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    isAnalysing = true;
    toggleBtn.innerHTML = "⏹️ Analizi Durdur";
    runPulse();
  } else {
    isAnalysing = false;
    const stream = video.srcObject;
    if (stream) stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    toggleBtn.innerHTML = "🔘 Analizi Başlat";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
});

confSlider.addEventListener('input', (e) => {
  minConfidence = e.target.value / 100;
  confVal.innerText = `${e.target.value}%`;
  if (objectDetector) objectDetector.setOptions({ scoreThreshold: minConfidence });
});

// Clock Sync
setInterval(() => {
  document.getElementById('current-time').innerText = new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
}, 1000);

initAI();
