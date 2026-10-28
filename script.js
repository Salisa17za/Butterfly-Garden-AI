/*
==================================================
 Butterfly Detection — YOLO11 + ONNX Runtime Web

 ขั้นตอนการทำงาน
 1. โหลดโมเดล Butterfly.onnx และ labels.json
 2. ผู้ใช้เลือกรูป
 3. Preprocess : ย่อรูปแบบ letterbox → Tensor [1,3,640,640]
 4. Inference  : ส่งเข้าโมเดล ได้ output [1,7,8400]
 5. Postprocess: แปลงเป็นกล่อง + กรองความมั่นใจ + NMS
 6. วาดกรอบบนรูป และแสดงผลด้านขวา

 โมเดล : YOLO11n (Ultralytics) export เป็น ONNX
 Input : "images"  [1, 3, 640, 640]
 Output: "output0" [1, 7, 8400]  (7 = x, y, w, h + 3 คลาส)
==================================================
*/

// ==================================================
// 0. ตั้งค่า
// ==================================================

const MODEL_PATH  = "Butterfly.onnx";   // ชื่อไฟล์โมเดล (ต้องอยู่โฟลเดอร์เดียวกับ index.html)
const LABELS_PATH = "labels.json";      // ชื่อคลาส เรียงตาม class id ของโมเดล
const INPUT_SIZE  = 640;                // ขนาด input ตอน export (imgsz=640)
const IOU_THRESHOLD = 0.45;             // ใช้ใน NMS

// ชื่อภาษาไทย + สีของแต่ละคลาส (key ต้องตรงกับ labels.json)
const CLASS_INFO = {
  Monarch:     { th: "ผีเสื้อโมนาร์ค",     color: "#f08a4b" },
  blue_morpho: { th: "ผีเสื้อมอร์โฟสีฟ้า",  color: "#4f8ef0" },
  swallowtail: { th: "ผีเสื้อหางติ่ง",      color: "#d9a514" },
};

// ==================================================
// 1. อ้างอิง Element ใน HTML
// ==================================================

const imageInput = document.getElementById("imageInput");
const canvas     = document.getElementById("canvas");
const ctx        = canvas.getContext("2d");
const dropZone   = document.getElementById("drop");
const labelBox   = document.getElementById("label");
const statusEl   = document.getElementById("status");
const statusText = document.getElementById("statusText");
const confSlider = document.getElementById("conf");
const confVal    = document.getElementById("confVal");
const resetBtn   = document.getElementById("resetBtn");
const classChips = document.getElementById("classChips");

let session = null;     // ONNX session
let labels  = [];       // ชื่อคลาส
let lastImage = null;   // รูปล่าสุด (ไว้คำนวณใหม่ตอนเลื่อน slider)
let lastRaw   = null;   // output ดิบล่าสุด
let lastInferMs = 0;

// ==================================================
// 2. โหลดโมเดลและ labels
// ==================================================

async function loadModel() {
  try {
    const res = await fetch(LABELS_PATH);
    labels = await res.json();
    renderClassChips();

    // wasm = ใช้ CPU ของเบราว์เซอร์ (ใช้ได้ทุกเครื่อง)
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });

    console.log("Model loaded", session.inputNames, session.outputNames);
    setStatus("ready", "โมเดลพร้อมใช้งานแล้ว");
  } catch (err) {
    console.error(err);
    setStatus("error", "โหลดโมเดลไม่สำเร็จ — ต้องเปิดผ่าน Live Server / http server");
  }
}
loadModel();

function setStatus(type, text) {
  statusEl.className = "status " + type;
  statusText.textContent = text;
}

function renderClassChips() {
  classChips.innerHTML = labels
    .map((name) => {
      const info = CLASS_INFO[name] || { th: name, color: "#e07a9b" };
      return `<span class="chip" style="border-color:${info.color}55">
                <span style="color:${info.color}">●</span> ${info.th}
              </span>`;
    })
    .join("");
}

// ==================================================
// 3. รับรูปจากผู้ใช้ (คลิกเลือก หรือ ลากวาง)
// ==================================================

imageInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("over"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("over"); })
);
dropZone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

resetBtn.addEventListener("click", () => imageInput.click());

confSlider.addEventListener("input", () => {
  confVal.textContent = confSlider.value + "%";
  if (lastRaw) showResult(); // คำนวณกล่องใหม่โดยไม่ต้องรันโมเดลซ้ำ
});

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const img = new Image();
  img.onload = () => detect(img);
  img.src = URL.createObjectURL(file);
}

// ==================================================
// 4. รันการตรวจจับ
// ==================================================

async function detect(img) {
  if (!session) {
    setStatus("error", "โมเดลยังโหลดไม่เสร็จ ลองใหม่อีกครั้ง");
    return;
  }
  setStatus("", "กำลังทำนาย…");

  const { tensor, scale, padX, padY } = preprocessImage(img);

  const feeds = { [session.inputNames[0]]: tensor };
  const t0 = performance.now();
  const result = await session.run(feeds);
  lastInferMs = performance.now() - t0;

  lastRaw = { output: result[session.outputNames[0]], scale, padX, padY };
  lastImage = img;

  showResult();
  setStatus("ready", "ทำนายเสร็จแล้ว");
}

function showResult() {
  const threshold = confSlider.value / 100;
  const { output, scale, padX, padY } = lastRaw;

  let boxes = postprocessOutput(output, threshold);
  boxes = nonMaxSuppression(boxes, IOU_THRESHOLD);

  // แปลงพิกัดจาก 640x640 (letterbox) กลับเป็นพิกัดของรูปจริง
  boxes.forEach((b) => {
    b.x = (b.x - padX) / scale;
    b.y = (b.y - padY) / scale;
    b.width  /= scale;
    b.height /= scale;
  });

  drawBoxes(lastImage, boxes);
  renderResultList(boxes);
}

// ==================================================
// 5. Preprocess : รูป → Tensor (letterbox)
//    ย่อรูปโดยรักษาสัดส่วน แล้วเติมขอบสีเทา (114)
//    เหมือนที่ Ultralytics ทำตอนทำนาย
// ==================================================

function preprocessImage(img) {
  const size = INPUT_SIZE;
  const scale = Math.min(size / img.width, size / img.height);
  const newW = Math.round(img.width * scale);
  const newH = Math.round(img.height * scale);
  const padX = (size - newW) / 2;
  const padY = (size - newH) / 2;

  const temp = document.createElement("canvas");
  temp.width = size;
  temp.height = size;
  const tctx = temp.getContext("2d");
  tctx.fillStyle = "rgb(114,114,114)";
  tctx.fillRect(0, 0, size, size);
  tctx.drawImage(img, padX, padY, newW, newH);

  const data = tctx.getImageData(0, 0, size, size).data;

  // จาก RGBA (H x W x 4) → Float32 [1, 3, H, W] ค่า 0–1
  const area = size * size;
  const float32 = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    float32[i]            = data[i * 4]     / 255; // R
    float32[i + area]     = data[i * 4 + 1] / 255; // G
    float32[i + area * 2] = data[i * 4 + 2] / 255; // B
  }

  return {
    tensor: new ort.Tensor("float32", float32, [1, 3, size, size]),
    scale, padX, padY,
  };
}

// ==================================================
// 6. Postprocess : Output → Bounding Boxes
//    output [1, 7, 8400]
//    แถว 0–3 = cx, cy, w, h  | แถว 4–6 = คะแนนของแต่ละคลาส
// ==================================================

function postprocessOutput(output, threshold) {
  const data = output.data;
  const numAttrs = output.dims[1]; // 7
  const numBoxes = output.dims[2]; // 8400
  const boxes = [];

  for (let i = 0; i < numBoxes; i++) {
    let bestScore = 0;
    let classId = 0;
    for (let c = 4; c < numAttrs; c++) {
      const s = data[numBoxes * c + i];
      if (s > bestScore) { bestScore = s; classId = c - 4; }
    }
    if (bestScore < threshold) continue;

    const cx = data[i];
    const cy = data[numBoxes + i];
    const w  = data[numBoxes * 2 + i];
    const h  = data[numBoxes * 3 + i];

    boxes.push({ x: cx - w / 2, y: cy - h / 2, width: w, height: h, score: bestScore, classId });
  }
  return boxes;
}

// ==================================================
// 7. NMS : ลบกล่องที่ซ้อนกัน (คลาสเดียวกัน)
// ==================================================

function nonMaxSuppression(boxes, iouThreshold) {
  boxes.sort((a, b) => b.score - a.score);
  const selected = [];
  while (boxes.length) {
    const current = boxes.shift();
    selected.push(current);
    boxes = boxes.filter(
      (b) => b.classId !== current.classId || calculateIoU(current, b) < iouThreshold
    );
  }
  return selected;
}

function calculateIoU(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width,  b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.width * a.height + b.width * b.height - inter);
}

// ==================================================
// 8. วาดรูป + กรอบ บน Canvas
// ==================================================

function drawBoxes(img, boxes) {
  // canvas ขนาดเท่ารูปจริง (CSS จะย่อให้พอดีกล่องเอง)
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  dropZone.classList.add("has-img");

  const lw = Math.max(3, Math.round(img.width / 200));
  const fontSize = Math.max(16, Math.round(img.width / 35));
  ctx.font = `600 ${fontSize}px Mali, sans-serif`;
  ctx.lineWidth = lw;
  ctx.lineJoin = "round";

  boxes.forEach((box) => {
    const name = labels[box.classId] ?? `class ${box.classId}`;
    const info = CLASS_INFO[name] || { color: "#e07a9b" };
    const text = `${name} ${(box.score * 100).toFixed(1)}%`;

    // กรอบ
    ctx.strokeStyle = info.color;
    roundRect(box.x, box.y, box.width, box.height, lw * 3);
    ctx.stroke();

    // ป้ายชื่อ
    const pad = fontSize * 0.4;
    const tw = ctx.measureText(text).width + pad * 2;
    const th = fontSize + pad * 1.2;
    const ty = box.y - th < 0 ? box.y : box.y - th;
    ctx.fillStyle = info.color;
    roundRect(box.x, ty, tw, th, th / 2);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, box.x + pad, ty + th / 2);
  });
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ==================================================
// 9. แสดงรายการผลทางขวา
// ==================================================

function renderResultList(boxes) {
  if (!boxes.length) {
    labelBox.innerHTML = `<div class="result-empty">
        ไม่พบผีเสื้อที่ความมั่นใจ ≥ ${confSlider.value}%<br>ลองเลื่อนค่าความมั่นใจให้ต่ำลงดูนะ
      </div>`;
    return;
  }

  labelBox.innerHTML =
    boxes
      .map((b) => {
        const name = labels[b.classId] ?? `class ${b.classId}`;
        const info = CLASS_INFO[name] || { th: name, color: "#e07a9b" };
        const pct = (b.score * 100).toFixed(1);
        return `<div class="item">
          <div class="swatch" style="background:${info.color}22">🦋</div>
          <div style="flex:1;min-width:0">
            <div class="name">${info.th}</div>
            <div class="en">${name}</div>
            <div class="bar"><i style="width:${pct}%;background:${info.color}"></i></div>
          </div>
          <div class="pct" style="color:${info.color}">${pct}%</div>
        </div>`;
      })
      .join("") +
    `<div class="meta">พบ ${boxes.length} ตัว · ใช้เวลา ${lastInferMs.toFixed(0)} ms</div>`;
}
