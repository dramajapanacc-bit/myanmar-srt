// ============================================================
// Myanmar SRT + One Clips Movie Recap Server
// ============================================================

const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");
const { spawn } = require("child_process");

const app = express();

const PORT = process.env.PORT || 10000;

const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, "public");
const WORK_DIR = path.join(os.tmpdir(), "myanmar-srt-work");

fs.mkdirSync(WORK_DIR, { recursive: true });

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(express.static(PUBLIC_DIR));

// ------------------------------------------------------------
// Multer
// ------------------------------------------------------------

const upload = multer({
  dest: WORK_DIR,
  limits: {
    fileSize: 1024 * 1024 * 1024
  }
});

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

function uid(prefix = "job") {
  return `${prefix}_${Date.now()}_${crypto.randomBytes(4).toString("hex")}`;
}

function safeName(name) {
  return String(name || "file")
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .slice(0, 100);
}

function clamp(n, min, max) {
  n = Number(n);
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, n));
}

function jsonError(res, status, message, extra = {}) {
  return res.status(status).json({
    ok: false,
    error: message,
    ...extra
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ------------------------------------------------------------
// FFmpeg
// ------------------------------------------------------------

function runProcess(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd || ROOT,
      env: process.env
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", d => {
      stdout += d.toString();
    });

    child.stderr.on("data", d => {
      stderr += d.toString();
    });

    child.on("error", err => {
      reject(err);
    });

    child.on("close", code => {
      if (code === 0) {
        resolve({
          stdout,
          stderr
        });
      } else {
        const err = new Error(
          `${command} exited with code ${code}\n${stderr.slice(-10000)}`
        );
        err.code = code;
        reject(err);
      }
    });
  });
}

async function ffmpeg(args) {
  return runProcess("ffmpeg", ["-y", "-hide_banner", "-loglevel", "error", ...args]);
}

async function ffprobe(args) {
  return runProcess("ffprobe", args);
}

async function getVideoDuration(file) {
  const result = await ffprobe([
    "-v",
    "error",
    "-show_entries",
    "format=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    file
  ]);

  const duration = Number(result.stdout.trim());

  if (!Number.isFinite(duration)) {
    throw new Error("Video duration မဖတ်နိုင်ပါ။");
  }

  return duration;
}

async function getVideoSize(file) {
  const result = await ffprobe([
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=width,height",
    "-of",
    "csv=p=0:s=x",
    file
  ]);

  const [width, height] = result.stdout.trim().split("x").map(Number);

  return {
    width: Number.isFinite(width) ? width : 1920,
    height: Number.isFinite(height) ? height : 1080
  };
}

// ------------------------------------------------------------
// Gemini REST
// ------------------------------------------------------------

async function geminiGenerate(model, body) {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY မတွေ့ပါ။ Render Environment Variables ကိုစစ်ပါ။");
  }

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent` +
    `?key=${encodeURIComponent(GEMINI_API_KEY)}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  const text = await response.text();

  let data;

  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(
      `Gemini response JSON မဟုတ်ပါ။ HTTP ${response.status}: ${text.slice(0, 1000)}`
    );
  }

  if (!response.ok) {
    throw new Error(
      `Gemini API error ${response.status}: ${
        data?.error?.message || text.slice(0, 2000)
      }`
    );
  }

  return data;
}

function extractGeminiText(data) {
  const parts =
    data?.candidates?.[0]?.content?.parts ||
    [];

  return parts
    .filter(p => typeof p.text === "string")
    .map(p => p.text)
    .join("\n")
    .trim();
}

// ------------------------------------------------------------
// Gemini image / scene analysis
// ------------------------------------------------------------

async function extractFrames(videoPath, jobDir, duration) {
  const framesDir = path.join(jobDir, "frames");
  fs.mkdirSync(framesDir, { recursive: true });

  // Maximum 24 frames.
  const count = Math.min(24, Math.max(8, Math.ceil(duration / 5)));

  const fps = count / Math.max(duration, 1);

  await ffmpeg([
    "-i",
    videoPath,
    "-vf",
    `fps=${fps.toFixed(6)},scale=768:-2`,
    "-q:v",
    "4",
    path.join(framesDir, "frame_%03d.jpg")
  ]);

  const files = fs
    .readdirSync(framesDir)
    .filter(f => f.toLowerCase().endsWith(".jpg"))
    .sort();

  return files.map(file => path.join(framesDir, file));
}

async function analyzeVideoScenes(videoPath, jobDir) {
  const duration = await getVideoDuration(videoPath);

  const frames = await extractFrames(videoPath, jobDir, duration);

  if (!frames.length) {
    throw new Error("Video scene frame မရပါ။");
  }

  const imageParts = [];

  for (const frame of frames) {
    const buffer = fs.readFileSync(frame);

    imageParts.push({
      inlineData: {
        mimeType: "image/jpeg",
        data: buffer.toString("base64")
      }
    });
  }

  const prompt = `
You are a professional movie recap writer.

The attached images are sequential frames sampled from ONE movie/video.

Analyze the actual visible scenes.

IMPORTANT:
- Do NOT invent unrelated scenes.
- Do NOT write a generic movie summary.
- Follow the actual visual sequence.
- Identify characters, locations, actions, emotions and important events visible in the frames.
- Write a natural Myanmar movie recap narration.
- The narration should sound like a Myanmar movie recap YouTuber.
- Do not mention that you are analyzing frames.
- Do not describe camera technical details.
- Do not use English unless a name or unavoidable proper noun requires it.
- Keep the narration suitable for voice-over.
- Make the narration coherent from beginning to end.
- Avoid excessive dialogue.
- Focus on what happens in the movie.

Video duration:
${duration.toFixed(2)} seconds

Return ONLY the Myanmar narration script.
`;

  const models = [
    "gemini-3.8-flash",
    "gemini-3.1-flash",
    "gemini-2.5-flash"
  ];

  let lastError = null;

  for (const model of models) {
    try {
      const data = await geminiGenerate(model, {
        contents: [
          {
            role: "user",
            parts: [
              { text: prompt },
              ...imageParts
            ]
          }
        ],
        generationConfig: {
          temperature: 0.55,
          maxOutputTokens: 10000
        }
      });

      const text = extractGeminiText(data);

      if (text) {
        return {
          script: cleanRecapScript(text),
          duration,
          frameCount: frames.length
        };
      }
    } catch (err) {
      lastError = err;
    }
  }

  throw lastError || new Error("Scene analysis failed.");
}

function cleanRecapScript(text) {
  return String(text || "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/^Myanmar\s*movie\s*recap\s*script\s*:?\s*/i, "")
    .replace(/^ဇာတ်လမ်းအကျဉ်း\s*:?\s*/i, "")
    .trim();
}

// ------------------------------------------------------------
// Gemini TTS
// ------------------------------------------------------------

const DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts";

const GEMINI_VOICES = [
  "Zephyr",
  "Puck",
  "Charon",
  "Kore",
  "Fenrir",
  "Leda",
  "Orus",
  "Aoede",
  "Callirrhoe",
  "Autonoe",
  "Enceladus",
  "Iapetus",
  "Umbriel",
  "Algieba",
  "Despina",
  "Erinome",
  "Algenib",
  "Rasalgethi",
  "Laomedeia",
  "Achernar",
  "Alnilam",
  "Schedar",
  "Gacrux",
  "Pulcherrima",
  "Achird",
  "Zubenelgenubi",
  "Vindemiatrix",
  "Sadachbia",
  "Sadaltager",
  "Sulafat"
];

async function geminiTTS({
  script,
  voice = "Kore",
  jobDir,
  speed = 1,
  pitch = 0,
  volume = 1
}) {
  if (!script.trim()) {
    throw new Error("Recap script မရှိပါ။");
  }

  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY မရှိပါ။");
  }

  if (!GEMINI_VOICES.includes(voice)) {
    voice = "Kore";
  }

  const response = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/interactions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
      },
      body: JSON.stringify({
        model: DEFAULT_TTS_MODEL,
        input: [
          {
            type: "user_input",
            content: [
              {
                type: "text",
                text: script,
                annotations: [
                  {
                    type: "speech_metadata",
                    style:
                      "Natural Myanmar movie recap narrator. Clear Burmese pronunciation. Smooth storytelling. Medium pace. Expressive but not theatrical."
                  }
                ]
              }
            ]
          }
        ],
        response_format: {
          type: "audio"
        },
        generation_config: {
          speech_config: [
            {
              voice
            }
          ]
        }
      })
    }
  );

  const responseText = await response.text();

  let data;

  try {
    data = JSON.parse(responseText);
  } catch {
    throw new Error(
      `Gemini TTS JSON မရပါ။ HTTP ${response.status}: ${responseText.slice(0, 2000)}`
    );
  }

  if (!response.ok) {
    throw new Error(
      `Gemini TTS error ${response.status}: ${
        data?.error?.message || responseText.slice(0, 2000)
      }`
    );
  }

  let audioBase64 = null;

  // New Interactions API convenience shape.
  if (data?.output_audio?.data) {
    audioBase64 = data.output_audio.data;
  }

  // REST raw response shape.
  if (!audioBase64 && Array.isArray(data?.steps)) {
    for (let i = data.steps.length - 1; i >= 0; i--) {
      const step = data.steps[i];

      if (!Array.isArray(step?.content)) continue;

      for (let j = step.content.length - 1; j >= 0; j--) {
        const item = step.content[j];

        if (item?.type === "audio" && item?.data) {
          audioBase64 = item.data;
          break;
        }
      }

      if (audioBase64) break;
    }
  }

  if (!audioBase64) {
    throw new Error("Gemini TTS audio data မတွေ့ပါ။");
  }

  const rawWav = path.join(jobDir, "gemini_raw.wav");

  fs.writeFileSync(
    rawWav,
    Buffer.from(audioBase64, "base64")
  );

  // Apply speed / pitch / volume AFTER TTS.
  const finalWav = path.join(jobDir, "voice.wav");

  const filters = [];

  const safeVolume = clamp(volume, 0, 3);
  if (Math.abs(safeVolume - 1) > 0.01) {
    filters.push(`volume=${safeVolume}`);
  }

  const safeSpeed = clamp(speed, 0.5, 2);

  if (Math.abs(safeSpeed - 1) > 0.01) {
    filters.push(`atempo=${safeSpeed}`);
  }

  const safePitch = clamp(pitch, -12, 12);

  if (Math.abs(safePitch) > 0.01) {
    const ratio = Math.pow(2, safePitch / 12);

    filters.push(
      `asetrate=24000*${ratio.toFixed(6)}`,
      "aresample=24000",
      `atempo=${(1 / ratio).toFixed(6)}`
    );
  }

  if (filters.length) {
    await ffmpeg([
      "-i",
      rawWav,
      "-af",
      filters.join(","),
      "-ar",
      "24000",
      "-ac",
      "1",
      "-c:a",
      "pcm_s16le",
      finalWav
    ]);
  } else {
    fs.copyFileSync(rawWav, finalWav);
  }

  return {
    audioPath: finalWav,
    voice,
    model: DEFAULT_TTS_MODEL
  };
}

// ------------------------------------------------------------
// Groq Whisper timestamps
// ------------------------------------------------------------

async function transcribeVoiceWithGroq(audioPath, script) {
  if (!GROQ_API_KEY) {
    throw new Error(
      "GROQ_API_KEY မရှိပါ။ Voice-sync subtitle အတွက် GROQ_API_KEY လိုပါတယ်။"
    );
  }

  const audioBuffer = fs.readFileSync(audioPath);

  const form = new FormData();

  form.append(
    "file",
    new Blob([audioBuffer], { type: "audio/wav" }),
    "voice.wav"
  );

  form.append("model", "whisper-large-v3-turbo");
  form.append("response_format", "verbose_json");
  form.append("timestamp_granularities[]", "word");
  form.append("timestamp_granularities[]", "segment");
  form.append("language", "my");
  form.append("temperature", "0");

  if (script) {
    form.append(
      "prompt",
      script.slice(0, 800)
    );
  }

  const response = await fetch(
    "https://api.groq.com/openai/v1/audio/transcriptions",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`
      },
      body: form
    }
  );

  const text = await response.text();

  let data;

  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(
      `Groq transcription JSON မရပါ။ HTTP ${response.status}: ${text.slice(0, 2000)}`
    );
  }

  if (!response.ok) {
    throw new Error(
      `Groq transcription error ${response.status}: ${
        data?.error?.message || text.slice(0, 2000)
      }`
    );
  }

  return data;
}

// ------------------------------------------------------------
// Subtitle helpers
// ------------------------------------------------------------

function assTime(seconds) {
  seconds = Math.max(0, Number(seconds) || 0);

  const h = Math.floor(seconds / 3600);

  const m = Math.floor(
    (seconds % 3600) / 60
  );

  const s = seconds % 60;

  const cs = Math.floor(
    (s - Math.floor(s)) * 100
  );

  return `${h}:${String(m).padStart(2, "0")}:${String(
    Math.floor(s)
  ).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
}

function escapeAssText(text) {
  return String(text || "")
    .replace(/\\/g, "\\\\")
    .replace(/\{/g, "\\{")
    .replace(/\}/g, "\\}")
    .replace(/\r?\n/g, "\\N");
}

function assColor(hex) {
  let value = String(hex || "#FFFFFF")
    .replace("#", "")
    .trim();

  if (!/^[0-9a-fA-F]{6}$/.test(value)) {
    value = "FFFFFF";
  }

  const r = value.substring(0, 2);
  const g = value.substring(2, 4);
  const b = value.substring(4, 6);

  return `&H00${b}${g}${r}`;
}

function makeAssHeader({
  fontName = "Noto Sans Myanmar",
  fontSize = 25,
  outline = 2,
  textColor = "#FFFFFF",
  position = "bottom",
  width = 1920,
  height = 1080
}) {
  let alignment = 2;

  if (position === "top") {
    alignment = 8;
  }

  if (position === "middle") {
    alignment = 5;
  }

  return `[Script Info]
ScriptType: v4.00+
PlayResX: ${width}
PlayResY: ${height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Recap,${fontName},${fontSize},${assColor(textColor)},${assColor("#FFFFFF")},&H00000000,&H88000000,0,0,0,0,100,100,0,0,1,${outline},0,${alignment},40,40,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;
}

function createSubtitleEvents(transcription) {
  const events = [];

  // Prefer segment timestamps because they are more stable
  // for readable subtitles.
  if (Array.isArray(transcription?.segments)) {
    for (const segment of transcription.segments) {
      const text = String(segment.text || "").trim();

      if (!text) continue;

      events.push({
        start: Number(segment.start) || 0,
        end: Number(segment.end) || 0,
        text
      });
    }
  }

  // Fallback to word timestamps.
  if (!events.length && Array.isArray(transcription?.words)) {
    let current = null;

    for (const word of transcription.words) {
      const text = String(word.word || "").trim();

      if (!text) continue;

      const start = Number(word.start) || 0;
      const end = Number(word.end) || start + 0.2;

      if (!current) {
        current = {
          start,
          end,
          text
        };
      } else {
        current.end = end;
        current.text += " " + text;
      }

      // Keep subtitles readable.
      if (
        current.text.length >= 48 ||
        current.end - current.start >= 4
      ) {
        events.push(current);
        current = null;
      }
    }

    if (current) {
      events.push(current);
    }
  }

  return events;
}

function makeAssFile({
  transcription,
  outputPath,
  fontName,
  fontSize,
  outline,
  textColor,
  position,
  width,
  height,
  subtitleX,
  subtitleY
}) {
  const header = makeAssHeader({
    fontName,
    fontSize,
    outline,
    textColor,
    position,
    width,
    height
  });

  let body = "";

  const events = createSubtitleEvents(transcription);

  for (const event of events) {
    let text = escapeAssText(event.text);

    // If user dragged subtitle freely, use exact position.
    const x = Number(subtitleX);
    const y = Number(subtitleY);

    if (
      Number.isFinite(x) &&
      Number.isFinite(y) &&
      x >= 0 &&
      y >= 0
    ) {
      text = `{\\pos(${Math.round(x)},${Math.round(y)})}` + text;
    }

    body +=
      `Dialogue: 0,${assTime(event.start)},${assTime(event.end)},Recap,,0,0,0,,${text}\n`;
  }

  fs.writeFileSync(
    outputPath,
    header + body,
    "utf8"
  );

  return events;
}

// ------------------------------------------------------------
// ASS path escaping
// ------------------------------------------------------------

function escapeFilterPath(file) {
  return file
    .replace(/\\/g, "\\\\")
    .replace(/:/g, "\\:")
    .replace(/'/g, "\\'");
}

// ------------------------------------------------------------
// Blur area
// ------------------------------------------------------------

function buildBlurFilter({
  inputWidth,
  inputHeight,
  blurOriginal,
  blurX,
  blurY,
  blurWidth,
  blurHeight
}) {
  if (!blurOriginal) {
    return null;
  }

  const x = clamp(blurX, 0, 100);
  const y = clamp(blurY, 0, 100);
  const w = clamp(blurWidth, 1, 100);
  const h = clamp(blurHeight, 1, 100);

  let cropW = Math.round(
    inputWidth * (w / 100)
  );

  let cropH = Math.round(
    inputHeight * (h / 100)
  );

  let cropX = Math.round(
    inputWidth * (x / 100)
  );

  let cropY = Math.round(
    inputHeight * (y / 100)
  );

  cropW = Math.max(2, Math.min(cropW, inputWidth));
  cropH = Math.max(2, Math.min(cropH, inputHeight));

  cropX = Math.max(
    0,
    Math.min(cropX, inputWidth - cropW)
  );

  cropY = Math.max(
    0,
    Math.min(cropY, inputHeight - cropH)
  );

  return {
    cropW,
    cropH,
    cropX,
    cropY
  };
}

// ------------------------------------------------------------
// Output size
// ------------------------------------------------------------

function outputScaleFilter(outputSize) {
  if (outputSize === "9:16") {
    return "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2";
  }

  if (outputSize === "1:1") {
    return "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2";
  }

  if (outputSize === "16:9") {
    return "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2";
  }

  return null;
}

// ------------------------------------------------------------
// Render final recap
// ------------------------------------------------------------

async function renderRecap({
  videoPath,
  audioPath,
  transcription,
  jobDir,
  settings
}) {
  const originalSize = await getVideoSize(videoPath);

  let width = originalSize.width;
  let height = originalSize.height;

  if (settings.outputSize === "9:16") {
    width = 1080;
    height = 1920;
  } else if (settings.outputSize === "1:1") {
    width = 1080;
    height = 1080;
  } else if (settings.outputSize === "16:9") {
    width = 1920;
    height = 1080;
  }

  const assPath = path.join(jobDir, "recap.ass");

  makeAssFile({
    transcription,
    outputPath: assPath,
    fontName: settings.fontName,
    fontSize: settings.fontSize,
    outline: settings.outline,
    textColor: settings.textColor,
    position: settings.position,
    width,
    height,
    subtitleX: settings.subtitleX,
    subtitleY: settings.subtitleY
  });

  const finalPath = path.join(
    jobDir,
    "final_movie_recap.mp4"
  );

  const blur = buildBlurFilter({
    inputWidth: originalSize.width,
    inputHeight: originalSize.height,
    blurOriginal: settings.blurOriginal,
    blurX: settings.blurX,
    blurY: settings.blurY,
    blurWidth: settings.blurWidth,
    blurHeight: settings.blurHeight
  });

  const filters = [];

  // ----------------------------------------------------------
  // Blur original subtitle
  // ----------------------------------------------------------

  if (blur) {
    filters.push(
      `[0:v]split=2[base][blurSrc]`,
      `[blurSrc]crop=${blur.cropW}:${blur.cropH}:${blur.cropX}:${blur.cropY},boxblur=12:2[blurred]`,
      `[base][blurred]overlay=${blur.cropX}:${blur.cropY}:enable='between(t,0,999999)'[blurVideo]`
    );

    const outputScale = outputScaleFilter(settings.outputSize);

    if (outputScale) {
      filters.push(
        `[blurVideo]${outputScale}[scaled]`
      );
    } else {
      filters.push(
        `[blurVideo]null[scaled]`
      );
    }

    const assEscaped = escapeFilterPath(assPath);

    filters.push(
      `[scaled]ass='${assEscaped}'[vout]`
    );
  } else {
    const outputScale = outputScaleFilter(settings.outputSize);

    if (outputScale) {
      filters.push(
        `[0:v]${outputScale}[scaled]`
      );
    } else {
      filters.push(
        `[0:v]null[scaled]`
      );
    }

    const assEscaped = escapeFilterPath(assPath);

    filters.push(
      `[scaled]ass='${assEscaped}'[vout]`
    );
  }

  const filterComplex = filters.join(";");

  const args = [
    "-i",
    videoPath,
    "-i",
    audioPath,
    "-filter_complex",
    filterComplex,
    "-map",
    "[vout]",
    "-map",
    "1:a:0",
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    "-c:a",
    "aac",
    "-b:a",
    "192k",
    "-shortest",
    "-movflags",
    "+faststart",
    finalPath
  ];

  await ffmpeg(args);

  return finalPath;
}

// ------------------------------------------------------------
// ONE-CLICK PIPELINE
// ------------------------------------------------------------

app.post(
  "/api/recap/one-click",
  upload.single("video"),
  async (req, res) => {
    const jobId = uid("recap");
    const jobDir = path.join(WORK_DIR, jobId);

    fs.mkdirSync(jobDir, { recursive: true });

    try {
      if (!req.file) {
        return jsonError(
          res,
          400,
          "Video file မတွေ့ပါ။"
        );
      }

      if (!GEMINI_API_KEY) {
        return jsonError(
          res,
          500,
          "GEMINI_API_KEY မရှိပါ။"
        );
      }

      if (!GROQ_API_KEY) {
        return jsonError(
          res,
          500,
          "GROQ_API_KEY မရှိပါ။ Voice subtitle sync အတွက်လိုပါတယ်။"
        );
      }

      const originalVideo = req.file.path;

      const videoPath = path.join(
        jobDir,
        safeName(req.file.originalname || "input.mp4")
      );

      fs.copyFileSync(
        originalVideo,
        videoPath
      );

      // ------------------------------------------------------
      // Settings from frontend
      // ------------------------------------------------------

      const settings = {
        voice:
          req.body.voice ||
          "Kore",

        voiceSpeed:
          clamp(
            req.body.voiceSpeed || 1,
            0.5,
            2
          ),

        voiceVolume:
          clamp(
            req.body.voiceVolume || 1,
            0,
            3
          ),

        voicePitch:
          clamp(
            req.body.voicePitch || 0,
            -12,
            12
          ),

        blurOriginal:
          String(
            req.body.blurOriginal || ""
          ) === "true",

        blurX:
          clamp(
            req.body.blurX || 0,
            0,
            100
          ),

        blurY:
          clamp(
            req.body.blurY || 0,
            0,
            100
          ),

        blurWidth:
          clamp(
            req.body.blurWidth || 100,
            1,
            100
          ),

        blurHeight:
          clamp(
            req.body.blurHeight || 20,
            1,
            100
          ),

        subtitleX:
          req.body.subtitleX !== undefined
            ? Number(req.body.subtitleX)
            : null,

        subtitleY:
          req.body.subtitleY !== undefined
            ? Number(req.body.subtitleY)
            : null,

        fontName:
          req.body.fontName ||
          "Noto Sans Myanmar",

        fontSize:
          clamp(
            req.body.fontSize || 25,
            10,
            100
          ),

        outline:
          clamp(
            req.body.outline || 2,
            0,
            20
          ),

        textColor:
          req.body.textColor ||
          "#FFFFFF",

        position:
          req.body.position ||
          "bottom",

        outputSize:
          req.body.outputSize ||
          "original"
      };

      // ------------------------------------------------------
      // STEP 1
      // Actual video scene analysis
      // ------------------------------------------------------

      console.log(
        `[${jobId}] STEP 1: analyzing video scenes`
      );

      const sceneResult =
        await analyzeVideoScenes(
          videoPath,
          jobDir
        );

      const script =
        sceneResult.script;

      fs.writeFileSync(
        path.join(jobDir, "recap_script.txt"),
        script,
        "utf8"
      );

      // ------------------------------------------------------
      // STEP 2
      // Gemini TTS
      // ------------------------------------------------------

      console.log(
        `[${jobId}] STEP 2: generating Gemini TTS`
      );

      const tts =
        await geminiTTS({
          script,
          voice: settings.voice,
          jobDir,
          speed: settings.voiceSpeed,
          pitch: settings.voicePitch,
          volume: settings.voiceVolume
        });

      // ------------------------------------------------------
      // STEP 3
      // Exact voice timestamps
      // ------------------------------------------------------

      console.log(
        `[${jobId}] STEP 3: transcribing generated voice`
      );

      const transcription =
        await transcribeVoiceWithGroq(
          tts.audioPath,
          script
        );

      fs.writeFileSync(
        path.join(jobDir, "voice_timestamps.json"),
        JSON.stringify(
          transcription,
          null,
          2
        ),
        "utf8"
      );

      // ------------------------------------------------------
      // STEP 4
      // Render final MP4
      // ------------------------------------------------------

      console.log(
        `[${jobId}] STEP 4: rendering final MP4`
      );

      const finalPath =
        await renderRecap({
          videoPath,
          audioPath: tts.audioPath,
          transcription,
          jobDir,
          settings
        });

      // ------------------------------------------------------
      // Return final video
      // ------------------------------------------------------

      console.log(
        `[${jobId}] DONE`
      );

      res.setHeader(
        "Content-Type",
        "video/mp4"
      );

      res.setHeader(
        "Content-Disposition",
        'attachment; filename="myanmar_movie_recap.mp4"'
      );

      const stream =
        fs.createReadStream(finalPath);

      stream.on("error", err => {
        console.error(err);
        if (!res.headersSent) {
          res.status(500).json({
            ok: false,
            error: err.message
          });
        }
      });

      stream.pipe(res);

    } catch (error) {
      console.error(
        `[${jobId}] ERROR`,
        error
      );

      if (!res.headersSent) {
        return jsonError(
          res,
          500,
          error.message || "One-Click Recap failed.",
          {
            jobId
          }
        );
      }
    } finally {
      // Keep files for a short time so stream can finish.
      setTimeout(() => {
        try {
          fs.rmSync(jobDir, {
            recursive: true,
            force: true
          });
        } catch {}
      }, 10 * 60 * 1000);

      try {
        fs.unlinkSync(req.file.path);
      } catch {}
    }
  }
);

// ------------------------------------------------------------
// Recap scene analysis endpoint
// Useful for P1/P2 preview workflow
// ------------------------------------------------------------

app.post(
  "/api/recap/analyze",
  upload.single("video"),
  async (req, res) => {
    const jobDir = path.join(
      WORK_DIR,
      uid("analyze")
    );

    fs.mkdirSync(jobDir, {
      recursive: true
    });

    try {
      if (!req.file) {
        return jsonError(
          res,
          400,
          "Video file မတွေ့ပါ။"
        );
      }

      const result =
        await analyzeVideoScenes(
          req.file.path,
          jobDir
        );

      return res.json({
        ok: true,
        script: result.script,
        duration: result.duration,
        frameCount: result.frameCount
      });

    } catch (error) {
      return jsonError(
        res,
        500,
        error.message
      );
    } finally {
      try {
        fs.unlinkSync(req.file.path);
      } catch {}

      setTimeout(() => {
        try {
          fs.rmSync(jobDir, {
            recursive: true,
            force: true
          });
        } catch {}
      }, 60000);
    }
  }
);

// ------------------------------------------------------------
// Recap TTS endpoint
// ------------------------------------------------------------

app.post(
  "/api/recap/tts",
  async (req, res) => {
    const jobDir = path.join(
      WORK_DIR,
      uid("tts")
    );

    fs.mkdirSync(jobDir, {
      recursive: true
    });

    try {
      const script =
        String(req.body.script || "").trim();

      if (!script) {
        return jsonError(
          res,
          400,
          "Script မရှိပါ။"
        );
      }

      const result =
        await geminiTTS({
          script,
          voice:
            req.body.voice ||
            "Kore",
          jobDir,
          speed:
            clamp(
              req.body.voiceSpeed || 1,
              0.5,
              2
            ),
          pitch:
            clamp(
              req.body.voicePitch || 0,
              -12,
              12
            ),
          volume:
            clamp(
              req.body.voiceVolume || 1,
              0,
              3
            )
        });

      // Copy to a public temporary location.
      const publicAudioDir =
        path.join(
          PUBLIC_DIR,
          "generated-audio"
        );

      fs.mkdirSync(
        publicAudioDir,
        {
          recursive: true
        }
      );

      const fileName =
        `${uid("voice")}.wav`;

      const destination =
        path.join(
          publicAudioDir,
          fileName
        );

      fs.copyFileSync(
        result.audioPath,
        destination
      );

      return res.json({
        ok: true,
        voice: result.voice,
        model: result.model,
        audioUrl:
          `/generated-audio/${fileName}`
      });

    } catch (error) {
      return jsonError(
        res,
        500,
        error.message
      );
    } finally {
      setTimeout(() => {
        try {
          fs.rmSync(jobDir, {
            recursive: true,
            force: true
          });
        } catch {}
      }, 10 * 60 * 1000);
    }
  }
);

// ------------------------------------------------------------
// Voice list
// ------------------------------------------------------------

app.get(
  "/api/recap/voices",
  (req, res) => {
    res.json({
      ok: true,
      model: DEFAULT_TTS_MODEL,
      voices: GEMINI_VOICES
    });
  }
);

// ------------------------------------------------------------
// Voice timestamp endpoint
// ------------------------------------------------------------

app.post(
  "/api/recap/voice-sync",
  upload.single("audio"),
  async (req, res) => {
    try {
      if (!req.file) {
        return jsonError(
          res,
          400,
          "Audio file မတွေ့ပါ။"
        );
      }

      const transcription =
        await transcribeVoiceWithGroq(
          req.file.path,
          req.body.script || ""
        );

      return res.json({
        ok: true,
        transcription,
        events:
          createSubtitleEvents(
            transcription
          )
      });

    } catch (error) {
      return jsonError(
        res,
        500,
        error.message
      );
    } finally {
      try {
        fs.unlinkSync(req.file.path);
      } catch {}
    }
  }
);

// ------------------------------------------------------------
// OLD SRT: Groq transcription
// ------------------------------------------------------------

app.post(
  "/api/transcribe",
  upload.single("video"),
  async (req, res) => {
    try {
      if (!req.file) {
        return jsonError(
          res,
          400,
          "Video file မတွေ့ပါ။"
        );
      }

      if (!GROQ_API_KEY) {
        return jsonError(
          res,
          500,
          "GROQ_API_KEY မရှိပါ။"
        );
      }

      const audioPath =
        path.join(
          WORK_DIR,
          `${uid("audio")}.wav`
        );

      await ffmpeg([
        "-i",
        req.file.path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        audioPath
      ]);

      const transcription =
        await transcribeVoiceWithGroq(
          audioPath,
          ""
        );

      return res.json({
        ok: true,
        text:
          transcription.text || "",
        segments:
          transcription.segments || [],
        words:
          transcription.words || []
      });

    } catch (error) {
      return jsonError(
        res,
        500,
        error.message
      );
    } finally {
      try {
        fs.unlinkSync(req.file.path);
      } catch {}
    }
  }
);

// ------------------------------------------------------------
// OLD SRT: Gemini translation
// ------------------------------------------------------------

app.post(
  "/api/translate",
  async (req, res) => {
    try {
      const text =
        String(req.body.text || "").trim();

      if (!text) {
        return jsonError(
          res,
          400,
          "Translate text မရှိပါ။"
        );
      }

      const prompt = `
Translate the following subtitle transcript into natural Myanmar Burmese.

Rules:
- Preserve the meaning.
- Keep subtitle-style short sentences.
- Do not add explanations.
- Do not add timestamps.
- Return ONLY the Myanmar translation.

TEXT:
${text}
`;

      const models = [
        "gemini-3.8-flash",
        "gemini-3.1-flash",
        "gemini-2.5-flash"
      ];

      let lastError = null;

      for (const model of models) {
        try {
          const data =
            await geminiGenerate(
              model,
              {
                contents: [
                  {
                    role: "user",
                    parts: [
                      {
                        text: prompt
                      }
                    ]
                  }
                ],
                generationConfig: {
                  temperature: 0.2
                }
              }
            );

          const translated =
            extractGeminiText(data);

          if (translated) {
            return res.json({
              ok: true,
              text: translated
            });
          }

        } catch (error) {
          lastError = error;
        }
      }

      throw (
        lastError ||
        new Error("Translation failed.")
      );

    } catch (error) {
      return jsonError(
        res,
        500,
        error.message
      );
    }
  }
);

// ------------------------------------------------------------
// OLD SRT render
// ------------------------------------------------------------

app.post(
  "/api/render",
  upload.single("video"),
  async (req, res) => {
    const jobDir = path.join(
      WORK_DIR,
      uid("render")
    );

    fs.mkdirSync(jobDir, {
      recursive: true
    });

    try {
      if (!req.file) {
        return jsonError(
          res,
          400,
          "Video file မတွေ့ပါ။"
        );
      }

      const videoPath =
        req.file.path;

      const originalSize =
        await getVideoSize(videoPath);

      const assPath =
        path.join(
          jobDir,
          "subtitle.ass"
        );

      let transcription = {
        segments: []
      };

      // Accept frontend-generated subtitle segments.
      if (Array.isArray(req.body.segments)) {
        transcription.segments =
          req.body.segments;
      } else if (
        typeof req.body.segments === "string"
      ) {
        try {
          transcription.segments =
            JSON.parse(
              req.body.segments
            );
        } catch {}
      }

      makeAssFile({
        transcription,
        outputPath: assPath,
        fontName:
          req.body.fontName ||
          "Noto Sans Myanmar",
        fontSize:
          clamp(
            req.body.fontSize || 25,
            10,
            100
          ),
        outline:
          clamp(
            req.body.outline || 2,
            0,
            20
          ),
        textColor:
          req.body.textColor ||
          "#FFFFFF",
        position:
          req.body.position ||
          "bottom",
        width:
          originalSize.width,
        height:
          originalSize.height,
        subtitleX:
          req.body.subtitleX,
        subtitleY:
          req.body.subtitleY
      });

      const output =
        path.join(
          jobDir,
          "rendered.mp4"
        );

      const assEscaped =
        escapeFilterPath(
          assPath
        );

      await ffmpeg([
        "-i",
        videoPath,
        "-vf",
        `ass='${assEscaped}'`,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        output
      ]);

      res.setHeader(
        "Content-Type",
        "video/mp4"
      );

      res.setHeader(
        "Content-Disposition",
        'attachment; filename="rendered.mp4"'
      );

      fs.createReadStream(
        output
      ).pipe(res);

    } catch (error) {
      return jsonError(
        res,
        500,
        error.message
      );
    }
  }
);

// ------------------------------------------------------------
// Health
// ------------------------------------------------------------

app.get(
  "/api/health",
  (req, res) => {
    res.json({
      ok: true,
      service: "myanmar-srt",
      movieRecap: true,
      oneClick: true,
      gemini: Boolean(GEMINI_API_KEY),
      groq: Boolean(GROQ_API_KEY),
      ttsModel: DEFAULT_TTS_MODEL
    });
  }
);

// ------------------------------------------------------------
// SPA fallback
// ------------------------------------------------------------

app.use((req, res) => {
  const indexPath = path.join(
    PUBLIC_DIR,
    "index.html"
  );

  if (fs.existsSync(indexPath)) {
    return res.sendFile(indexPath);
  }

  return res.status(404).send(
    "index.html not found"
  );
});

// ------------------------------------------------------------
// Start
// ------------------------------------------------------------

app.listen(
  PORT,
  "0.0.0.0",
  () => {
    console.log(
      `Myanmar SRT server running on port ${PORT}`
    );

    console.log(
      `Gemini API: ${
        GEMINI_API_KEY ? "OK" : "MISSING"
      }`
    );

    console.log(
      `Groq API: ${
        GROQ_API_KEY ? "OK" : "MISSING"
      }`
    );

    console.log(
      `Movie Recap One-Click: ENABLED`
    );
  }
);
