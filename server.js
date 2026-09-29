// ============================================================
// BURMESE SRT + AI MOVIE RECAP
// Full Auto:
// Video -> AI Scene Understanding -> Myanmar Script
// -> Gemini TTS -> Auto Sync -> Groq Whisper Timing
// -> Sync SRT -> Blur -> Myanmar Subtitle -> Final MP4
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
const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "output");
const WORK_DIR = path.join(os.tmpdir(), "myanmar-srt-work");

fs.mkdirSync(PUBLIC_DIR, { recursive: true });
fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.mkdirSync(WORK_DIR, { recursive: true });

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(express.static(PUBLIC_DIR));

const upload = multer({
  dest: WORK_DIR,
  limits: {
    fileSize: 1024 * 1024 * 1024
  }
});


// ============================================================
// HELPERS
// ============================================================

function uid(prefix = "job") {
  return (
    prefix +
    "_" +
    Date.now() +
    "_" +
    crypto.randomBytes(5).toString("hex")
  );
}

function clamp(value, min, max) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return min;
  }

  return Math.max(min, Math.min(max, n));
}

function jsonError(res, status, message, extra = {}) {
  return res.status(status).json({
    ok: false,
    error: message,
    ...extra
  });
}

function getGroqKey(req) {
  return String(
    req.headers["x-groq-api-key"] ||
    req.body?.groqApiKey ||
    process.env.GROQ_API_KEY ||
    ""
  ).trim();
}

function getGeminiKey(req) {
  return String(
    req.headers["x-gemini-api-key"] ||
    req.body?.geminiApiKey ||
    process.env.GEMINI_API_KEY ||
    ""
  ).trim();
}

function safeFilename(name) {
  return String(name || "file")
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .slice(0, 120);
}

function removeFile(file) {
  try {
    if (file) {
      fs.rmSync(file, { force: true });
    }
  } catch (_) {}
}

function removeDir(dir) {
  try {
    if (dir) {
      fs.rmSync(dir, {
        recursive: true,
        force: true
      });
    }
  } catch (_) {}
}


// ============================================================
// PROCESS / FFMPEG / FFPROBE
// ============================================================

function runProcess(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      command,
      args,
      {
        cwd: ROOT,
        env: process.env
      }
    );

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", data => {
      stdout += data.toString();
    });

    child.stderr.on("data", data => {
      stderr += data.toString();
    });

    child.on("error", error => {
      reject(error);
    });

    child.on("close", code => {
      if (code === 0) {
        resolve({
          stdout,
          stderr
        });
      } else {
        reject(
          new Error(
            `${command} exited with code ${code}\n` +
            stderr.slice(-12000)
          )
        );
      }
    });
  });
}

async function ffmpeg(args) {
  return runProcess(
    "ffmpeg",
    [
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      ...args
    ]
  );
}

async function ffprobe(args) {
  return runProcess(
    "ffprobe",
    args
  );
}


async function videoDuration(file) {

  const result = await ffprobe([
    "-v",
    "error",
    "-show_entries",
    "format=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    file
  ]);

  const duration =
    Number(result.stdout.trim());

  if (!Number.isFinite(duration)) {
    throw new Error(
      "Video duration မဖတ်နိုင်ပါ။"
    );
  }

  return duration;
}


async function mediaDuration(file) {

  const result = await ffprobe([
    "-v",
    "error",
    "-show_entries",
    "format=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    file
  ]);

  const duration =
    Number(result.stdout.trim());

  if (!Number.isFinite(duration)) {
    throw new Error(
      "Media duration မဖတ်နိုင်ပါ။"
    );
  }

  return duration;
}


async function videoSize(file) {

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

  const parts =
    result.stdout
      .trim()
      .split("x")
      .map(Number);

  const width =
    Number.isFinite(parts[0])
      ? parts[0]
      : 1920;

  const height =
    Number.isFinite(parts[1])
      ? parts[1]
      : 1080;

  return {
    width,
    height
  };
}


// ============================================================
// AUDIO TEMPO
// ============================================================

function makeAtempoFilters(factor) {

  let value =
    Number(factor);

  if (!Number.isFinite(value) || value <= 0) {
    value = 1;
  }

  const filters = [];

  /*
    FFmpeg atempo supports 0.5 -> 2.0.

    If sync factor is outside that range,
    split it into multiple atempo filters.
  */

  while (value > 2) {
    filters.push("atempo=2");
    value /= 2;
  }

  while (value < 0.5) {
    filters.push("atempo=0.5");
    value /= 0.5;
  }

  filters.push(
    `atempo=${value.toFixed(8)}`
  );

  return filters;
}


async function syncAudioToVideo(
  audioPath,
  targetDuration,
  outputPath
) {

  const audioDuration =
    await mediaDuration(audioPath);

  if (
    !Number.isFinite(audioDuration) ||
    audioDuration <= 0
  ) {
    throw new Error(
      "TTS Audio duration မဖတ်နိုင်ပါ။"
    );
  }

  if (
    !Number.isFinite(targetDuration) ||
    targetDuration <= 0
  ) {
    throw new Error(
      "Video duration မမှန်ပါ။"
    );
  }

  /*
    Example:

    Video = 120 sec
    Audio = 138 sec

    factor = 138 / 120
            = 1.15

    atempo=1.15

    Audio becomes approximately 120 sec.
  */

  const factor =
    audioDuration / targetDuration;

  const filters =
    makeAtempoFilters(factor);

  /*
    apad + atrim guarantees that the final
    audio does not exceed the target timeline.
  */

  const filterComplex =
    [
      ...filters,
      "apad",
      `atrim=duration=${targetDuration.toFixed(3)}`,
      "asetpts=N/SR/TB"
    ].join(",");

  await ffmpeg([
    "-i",
    audioPath,

    "-af",
    filterComplex,

    "-ar",
    "24000",

    "-ac",
    "1",

    "-c:a",
    "pcm_s16le",

    outputPath
  ]);

  const finalDuration =
    await mediaDuration(outputPath);

  return {
    inputDuration: audioDuration,
    targetDuration,
    outputDuration: finalDuration,
    tempoFactor: factor
  };
}


// ============================================================
// GEMINI TEXT
// ============================================================

async function geminiGenerate(
  model,
  body,
  apiKey
) {

  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY မရှိပါ။"
    );
  }

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${encodeURIComponent(model)}:generateContent?key=` +
    `${encodeURIComponent(apiKey)}`;

  const response =
    await fetch(
      url,
      {
        method: "POST",
        headers: {
          "Content-Type":
            "application/json"
        },
        body: JSON.stringify(body)
      }
    );

  const raw =
    await response.text();

  let data;

  try {
    data =
      JSON.parse(raw);
  } catch (_) {
    throw new Error(
      `Gemini JSON မရပါ။ HTTP ${response.status}: ` +
      raw.slice(0, 2000)
    );
  }

  if (!response.ok) {
    throw new Error(
      `Gemini API ${response.status}: ` +
      (
        data?.error?.message ||
        raw.slice(0, 2000)
      )
    );
  }

  return data;
}


function geminiText(data) {

  return (
    data?.candidates?.[0]?.content?.parts || []
  )
    .filter(
      part =>
        typeof part.text === "string"
    )
    .map(
      part => part.text
    )
    .join("\n")
    .trim();
}


// ============================================================
// AI VIDEO SCENE ANALYSIS
// ============================================================

async function extractFrames(
  video,
  jobDir,
  duration
) {

  const framesDir =
    path.join(
      jobDir,
      "frames"
    );

  fs.mkdirSync(
    framesDir,
    {
      recursive: true
    }
  );

  /*
    Maximum 24 frames.

    For short videos:
    minimum 8 frames.

    For longer videos:
    approximately one frame per 5 sec,
    capped at 24.
  */

  const count =
    Math.min(
      24,
      Math.max(
        8,
        Math.ceil(
          duration / 5
        )
      )
    );

  const fps =
    count /
    Math.max(
      duration,
      1
    );

  await ffmpeg([
    "-i",
    video,

    "-vf",
    `fps=${fps.toFixed(6)},scale=768:-2`,

    "-q:v",
    "4",

    path.join(
      framesDir,
      "frame_%03d.jpg"
    )
  ]);

  return fs
    .readdirSync(
      framesDir
    )
    .filter(
      name =>
        name.endsWith(".jpg")
    )
    .sort()
    .map(
      name =>
        path.join(
          framesDir,
          name
        )
    );
}


function cleanScript(text) {

  return String(text || "")
    .replace(
      /```[\s\S]*?```/g,
      ""
    )
    .replace(
      /^Myanmar\s*movie\s*recap\s*script\s*:?\s*/i,
      ""
    )
    .trim();
}


async function analyzeVideoScenes(
  video,
  jobDir,
  apiKey
) {

  const duration =
    await videoDuration(video);

  const frames =
    await extractFrames(
      video,
      jobDir,
      duration
    );

  if (!frames.length) {
    throw new Error(
      "Video scene frame မရပါ။"
    );
  }

  const imageParts =
    frames.map(
      frame => ({
        inlineData: {
          mimeType: "image/jpeg",
          data:
            fs
              .readFileSync(frame)
              .toString("base64")
        }
      })
    );


  /*
    IMPORTANT:

    AI ကို "scene description" သာမက
    Movie Recap narrator အဖြစ်
    ဇာတ်လမ်းကို ဆက်စပ်ပြီး ရေးစေမယ်။
  */

  const prompt = `
You are an expert Myanmar movie recap narrator and screenplay summarizer.

You are given sequential visual frames from ONE video.

Your job is to understand the story from the visible scenes and write a natural Myanmar-language movie recap narration for voice-over.

IMPORTANT RULES:

1. Follow the actual visual order.
2. Only describe events supported by what is visible.
3. Do NOT invent characters, locations, events, relationships, or outcomes that cannot reasonably be supported by the video.
4. Connect scenes naturally so the narration feels like one continuous story.
5. Explain important actions and story developments clearly.
6. Use natural spoken Myanmar.
7. Write like a professional Myanmar movie recap narrator speaking to viewers.
8. Do not sound like a literal translation.
9. Avoid robotic wording.
10. Avoid excessive dialogue.
11. Do not mention "frame", "image", "camera", "AI", "scene analysis", or this prompt.
12. Do not use numbered sections.
13. Do not add a title.
14. Return ONLY the Myanmar narration script.
15. Make the script suitable for Gemini TTS.

Video duration:
${duration.toFixed(2)} seconds
`;


  let lastError = null;

  const models = [
    "gemini-3.8-flash",
    "gemini-3.1-flash",
    "gemini-2.5-flash"
  ];


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
                  },
                  ...imageParts
                ]
              }
            ],

            generationConfig: {
              temperature: 0.55,
              maxOutputTokens: 10000
            }
          },
          apiKey
        );


      const script =
        cleanScript(
          geminiText(data)
        );


      if (script) {

        return {
          script,
          duration,
          frameCount:
            frames.length,
          model
        };

      }

    } catch (error) {

      lastError = error;

    }

  }


  throw (
    lastError ||
    new Error(
      "AI Movie Recap Script မထုတ်နိုင်ပါ။"
    )
  );
}


// ============================================================
// GEMINI VOICES
// ============================================================

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

const TTS_MODEL =
  "gemini-3.8-flash-tts";


function emotionInstruction(
  emotion
) {

  const styles = {

    natural:
      "Natural, warm and conversational Myanmar movie recap narration. Clear pronunciation. Smooth storytelling. Medium pace. Expressive but not theatrical.",

    storytelling:
      "Professional Myanmar movie recap storyteller. Natural storytelling rhythm. Clear pronunciation. Smooth transitions. Engaging but believable.",

    dramatic:
      "Dramatic Myanmar movie recap narration. Build tension naturally. Use expressive emphasis on important moments, but remain believable and clear.",

    emotional:
      "Emotionally expressive Myanmar movie recap narration. Convey sadness, fear, hope and tension naturally according to the story. Do not overact.",

    calm:
      "Calm and controlled Myanmar movie recap narration. Smooth, gentle and easy to understand.",

    excited:
      "Energetic Myanmar movie recap narration. Engaging and lively while keeping Myanmar pronunciation clear and natural."
  };

  return (
    styles[emotion] ||
    styles.natural
  );
}


// ============================================================
// GEMINI TTS
// ============================================================

async function geminiTTS({
  script,
  voice = "Kore",
  emotion = "natural",
  jobDir,
  speed = 1,
  pitch = 0,
  volume = 1,
  apiKey
}) {

  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY မရှိပါ။"
    );
  }

  if (!String(script).trim()) {
    throw new Error(
      "Recap Script မရှိပါ။"
    );
  }

  if (
    !GEMINI_VOICES.includes(
      voice
    )
  ) {
    voice = "Kore";
  }


  const style =
    emotionInstruction(
      emotion
    );


  const response =
    await fetch(
      "https://generativelanguage.googleapis.com/v1beta/interactions",
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",

          "x-goog-api-key":
            apiKey
        },

        body: JSON.stringify({

          model:
            TTS_MODEL,

          input: [
            {
              type:
                "user_input",

              content: [
                {
                  type:
                    "text",

                  text:
                    String(script),

                  annotations: [
                    {
                      type:
                        "speech_metadata",

                      style
                    }
                  ]
                }
              ]
            }
          ],

          response_format: {
            type:
              "audio"
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


  const raw =
    await response.text();

  let data;

  try {
    data =
      JSON.parse(raw);
  } catch (_) {

    throw new Error(
      `Gemini TTS JSON မရပါ။ HTTP ${response.status}: ` +
      raw.slice(0, 2000)
    );

  }


  if (!response.ok) {

    throw new Error(
      `Gemini TTS ${response.status}: ` +
      (
        data?.error?.message ||
        raw.slice(0, 2000)
      )
    );

  }


  let audioBase64 =
    data?.output_audio?.data;


  /*
    Some Gemini responses return audio
    inside steps/content.
  */

  if (
    !audioBase64 &&
    Array.isArray(
      data?.steps
    )
  ) {

    for (
      let i =
        data.steps.length - 1;

      i >= 0 &&
      !audioBase64;

      i--
    ) {

      const content =
        data
          .steps[i]
          ?.content || [];

      for (
        let j =
          content.length - 1;

        j >= 0;

        j--
      ) {

        const item =
          content[j];

        if (
          item?.type ===
            "audio" &&
          item?.data
        ) {

          audioBase64 =
            item.data;

          break;

        }

      }

    }

  }


  if (!audioBase64) {

    throw new Error(
      "Gemini TTS audio data မတွေ့ပါ။"
    );

  }


  const rawWav =
    path.join(
      jobDir,
      "gemini_raw.wav"
    );

  const outWav =
    path.join(
      jobDir,
      "voice.wav"
    );


  fs.writeFileSync(
    rawWav,
    Buffer.from(
      audioBase64,
      "base64"
    )
  );


  const filters = [];

  const vol =
    clamp(
      volume,
      0,
      3
    );

  const spd =
    clamp(
      speed,
      0.5,
      2
    );

  const pit =
    clamp(
      pitch,
      -12,
      12
    );


  if (
    Math.abs(vol - 1) >
    0.01
  ) {

    filters.push(
      `volume=${vol}`
    );

  }


  /*
    Pitch changes frequency while
    compensating timing.
  */

  if (
    Math.abs(pit) >
    0.01
  ) {

    const ratio =
      Math.pow(
        2,
        pit / 12
      );

    filters.push(
      `asetrate=24000*${ratio.toFixed(6)}`
    );

    filters.push(
      "aresample=24000"
    );

    filters.push(
      ...makeAtempoFilters(
        1 / ratio
      )
    );

  }


  if (
    Math.abs(spd - 1) >
    0.01
  ) {

    filters.push(
      ...makeAtempoFilters(
        spd
      )
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

      outWav
    ]);

  } else {

    fs.copyFileSync(
      rawWav,
      outWav
    );

  }


  return {
    audioPath:
      outWav,

    voice,

    emotion,

    model:
      TTS_MODEL
  };
}


// ============================================================
// GROQ WHISPER
// ============================================================

async function groqTranscribe(
  audioPath,
  script,
  apiKey
) {

  if (!apiKey) {
    throw new Error(
      "GROQ_API_KEY မရှိပါ။"
    );
  }


  const form =
    new FormData();


  form.append(
    "file",
    new Blob(
      [
        fs.readFileSync(
          audioPath
        )
      ],
      {
        type:
          "audio/wav"
      }
    ),
    "voice.wav"
  );


  form.append(
    "model",
    "whisper-large-v3-turbo"
  );

  form.append(
    "response_format",
    "verbose_json"
  );

  form.append(
    "timestamp_granularities[]",
    "word"
  );

  form.append(
    "timestamp_granularities[]",
    "segment"
  );

  form.append(
    "language",
    "my"
  );

  form.append(
    "temperature",
    "0"
  );


  if (script) {

    form.append(
      "prompt",
      String(
        script
      ).slice(
        0,
        800
      )
    );

  }


  const response =
    await fetch(
      "https://api.groq.com/openai/v1/audio/transcriptions",
      {
        method:
          "POST",

        headers: {
          Authorization:
            `Bearer ${apiKey}`
        },

        body:
          form
      }
    );


  const raw =
    await response.text();

  let data;

  try {
    data =
      JSON.parse(raw);
  } catch (_) {

    throw new Error(
      `Groq JSON မရပါ။ HTTP ${response.status}: ` +
      raw.slice(0, 2000)
    );

  }


  if (!response.ok) {

    throw new Error(
      `Groq ${response.status}: ` +
      (
        data?.error?.message ||
        raw.slice(0, 2000)
      )
    );

  }


  return data;
}


// ============================================================
// SRT
// ============================================================

function formatSrtTime(
  seconds
) {

  const totalMs =
    Math.max(
      0,
      Math.round(
        Number(seconds) *
        1000
      )
    );

  const hours =
    Math.floor(
      totalMs /
      3600000
    );

  const minutes =
    Math.floor(
      (totalMs %
        3600000) /
      60000
    );

  const secondsPart =
    Math.floor(
      (totalMs %
        60000) /
      1000
    );

  const ms =
    totalMs %
    1000;


  return (
    String(hours)
      .padStart(2, "0") +

    ":" +

    String(minutes)
      .padStart(2, "0") +

    ":" +

    String(secondsPart)
      .padStart(2, "0") +

    "," +

    String(ms)
      .padStart(3, "0")
  );
}


function eventsFromTranscription(
  transcription
) {

  const output = [];


  if (
    Array.isArray(
      transcription?.segments
    )
  ) {

    for (
      const segment
      of transcription.segments
    ) {

      const text =
        String(
          segment.text ||
          ""
        ).trim();

      if (!text) {
        continue;
      }


      const start =
        Number(
          segment.start
        );

      const end =
        Number(
          segment.end
        );


      if (
        !Number.isFinite(start) ||
        !Number.isFinite(end)
      ) {
        continue;
      }


      output.push({
        start,
        end:
          Math.max(
            end,
            start + 0.15
          ),
        text
      });

    }

  }


  /*
    Fallback to word timestamps
    if segments are missing.
  */

  if (
    !output.length &&
    Array.isArray(
      transcription?.words
    )
  ) {

    let current =
      null;


    for (
      const word
      of transcription.words
    ) {

      const text =
        String(
          word.word ||
          ""
        ).trim();

      if (!text) {
        continue;
      }


      const start =
        Number(
          word.start
        ) || 0;

      const end =
        Number(
          word.end
        ) ||
        start + 0.2;


      if (!current) {

        current = {
          start,
          end,
          text
        };

      } else {

        current.end =
          end;

        current.text +=
          " " +
          text;

      }


      if (
        current.text.length >=
          55 ||
        current.end -
          current.start >=
          4
      ) {

        output.push(
          current
        );

        current =
          null;

      }

    }


    if (current) {
      output.push(
        current
      );
    }

  }


  return output;
}


function makeSrt(
  transcription
) {

  const events =
    eventsFromTranscription(
      transcription
    );


  return events
    .map(
      (event, index) =>
`${index + 1}
${formatSrtTime(event.start)} --> ${formatSrtTime(event.end)}
${event.text}
`
    )
    .join("\n");
}


// ============================================================
// ASS SUBTITLE
// ============================================================

function assTime(seconds) {

  const s =
    Math.max(
      0,
      Number(seconds) || 0
    );

  const hours =
    Math.floor(
      s / 3600
    );

  const minutes =
    Math.floor(
      (s % 3600) / 60
    );

  const secondsWhole =
    Math.floor(
      s % 60
    );

  const centiseconds =
    Math.floor(
      (
        s -
        Math.floor(s)
      ) *
      100
    );


  return (
    `${hours}:` +
    `${String(minutes).padStart(2,"0")}:` +
    `${String(secondsWhole).padStart(2,"0")}.` +
    `${String(centiseconds).padStart(2,"0")}`
  );
}


function assColor(hex) {

  let value =
    String(
      hex ||
      "#FFFFFF"
    )
      .replace(
        "#",
        ""
      );


  if (
    !/^[0-9a-fA-F]{6}$/
      .test(value)
  ) {

    value =
      "FFFFFF";

  }


  return (
    "&H00" +
    value.slice(4,6) +
    value.slice(2,4) +
    value.slice(0,2)
  );
}


function escAss(text) {

  return String(
    text || ""
  )
    .replace(
      /\\/g,
      "\\\\"
    )
    .replace(
      /\{/g,
      "\\{"
    )
    .replace(
      /\}/g,
      "\\}"
    )
    .replace(
      /\r?\n/g,
      "\\N"
    );
}


function makeAss({
  transcription,
  file,
  fontName = "Noto Sans Myanmar",
  fontSize = 28,
  outline = 2,
  textColor = "#FFFFFF",
  position = "bottom",
  width = 1920,
  height = 1080,
  subtitleX,
  subtitleY
}) {

  const hasFreePosition =
    Number.isFinite(
      Number(subtitleX)
    ) &&
    Number.isFinite(
      Number(subtitleY)
    );


  let alignment = 2;

  if (
    position === "top"
  ) {
    alignment = 8;
  }

  if (
    position === "middle"
  ) {
    alignment = 5;
  }

  if (
    hasFreePosition
  ) {
    alignment = 7;
  }


  let ass =
`[Script Info]
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


  const events =
    eventsFromTranscription(
      transcription
    );


  let x = null;
  let y = null;


  if (hasFreePosition) {

    x =
      (
        clamp(
          subtitleX,
          0,
          100
        ) /
        100
      ) *
      width;

    y =
      (
        clamp(
          subtitleY,
          0,
          100
        ) /
        100
      ) *
      height;

  }


  for (
    const event
    of events
  ) {

    let text =
      escAss(
        event.text
      );


    if (
      hasFreePosition
    ) {

      text =
        `{\\pos(${Math.round(x)},${Math.round(y)})}` +
        text;

    }


    ass +=
`Dialogue: 0,${assTime(event.start)},${assTime(event.end)},Recap,,0,0,0,,${text}
`;

  }


  fs.writeFileSync(
    file,
    ass,
    "utf8"
  );
}


// ============================================================
// BLUR
// ============================================================

function filterPath(file) {

  return String(file)
    .replace(
      /\\/g,
      "\\\\"
    )
    .replace(
      /:/g,
      "\\:"
    )
    .replace(
      /'/g,
      "\\'"
    );
}


function blurBox({
  width,
  height,
  on,
  x,
  y,
  w,
  h
}) {

  if (!on) {
    return null;
  }


  const bw =
    Math.max(
      2,
      Math.round(
        width *
        clamp(w,1,100) /
        100
      )
    );


  const bh =
    Math.max(
      2,
      Math.round(
        height *
        clamp(h,1,100) /
        100
      )
    );


  const bx =
    Math.max(
      0,
      Math.min(
        Math.round(
          width *
          clamp(x,0,100) /
          100
        ),
        width - bw
      )
    );


  const by =
    Math.max(
      0,
      Math.min(
        Math.round(
          height *
          clamp(y,0,100) /
          100
        ),
        height - bh
      )
    );


  return {
    bw,
    bh,
    bx,
    by
  };
}


// ============================================================
// OUTPUT SIZE
// ============================================================

function outputSpec(
  size,
  original
) {

  if (
    size === "9:16"
  ) {

    return {
      width: 1080,
      height: 1920,

      scale:
        "scale=1080:1920:force_original_aspect_ratio=decrease," +
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    };

  }


  if (
    size === "1:1"
  ) {

    return {
      width: 1080,
      height: 1080,

      scale:
        "scale=1080:1080:force_original_aspect_ratio=decrease," +
        "pad=1080:1080:(ow-iw)/2:(oh-ih)/2"
    };

  }


  if (
    size === "16:9"
  ) {

    return {
      width: 1920,
      height: 1080,

      scale:
        "scale=1920:1080:force_original_aspect_ratio=decrease," +
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
    };

  }


  return {
    width:
      original.width,

    height:
      original.height,

    scale:
      null
  };
}


// ============================================================
// FINAL VIDEO RENDER
// ============================================================

async function renderRecap({
  video,
  audio,
  transcription,
  settings,
  jobDir
}) {

  if (!audio) {

    throw new Error(
      "Render အတွက် audio မရှိပါ။"
    );

  }


  const original =
    await videoSize(
      video
    );


  const output =
    outputSpec(
      settings.outputSize,
      original
    );


  const assFile =
    path.join(
      jobDir,
      "subtitles.ass"
    );


  makeAss({
    transcription,
    file:
      assFile,

    fontName:
      settings.fontName ||
      "Noto Sans Myanmar",

    fontSize:
      Number(
        settings.fontSize
      ) || 28,

    outline:
      Number(
        settings.outline
      ) || 2,

    textColor:
      settings.textColor ||
      "#FFFFFF",

    position:
      settings.position ||
      "bottom",

    width:
      output.width,

    height:
      output.height,

    subtitleX:
      settings.subtitleX,

    subtitleY:
      settings.subtitleY
  });


  const box =
    blurBox({
      width:
        original.width,

      height:
        original.height,

      on:
        settings.blurOriginal,

      x:
        settings.blurX,

      y:
        settings.blurY,

      w:
        settings.blurWidth,

      h:
        settings.blurHeight
    });


  const assPath =
    filterPath(
      assFile
    );


  const filters = [];


  if (box) {

    filters.push(
      "[0:v]split=2[base][src]"
    );

    filters.push(
      `[src]crop=${box.bw}:${box.bh}:${box.bx}:${box.by},boxblur=20:10[blur]`
    );

    filters.push(
      `[base][blur]overlay=${box.bx}:${box.by}[v0]`
    );


    if (
      output.scale
    ) {

      filters.push(
        `[v0]${output.scale}[v1]`
      );

    } else {

      filters.push(
        "[v0]null[v1]"
      );

    }

  } else {

    if (
      output.scale
    ) {

      filters.push(
        `[0:v]${output.scale}[v1]`
      );

    } else {

      filters.push(
        "[0:v]null[v1]"
      );

    }

  }


  filters.push(
    `[v1]subtitles='${assPath}'[vout]`
  );


  const outputFile =
    path.join(
      jobDir,
      "myanmar_movie_recap.mp4"
    );


  await ffmpeg([
    "-i",
    video,

    "-i",
    audio,

    "-filter_complex",
    filters.join(";"),

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

    "-c:a",
    "aac",

    "-b:a",
    "192k",

    "-t",
    String(
      await videoDuration(
        video
      )
    ),

    "-movflags",
    "+faststart",

    outputFile
  ]);


  return outputFile;
}


// ============================================================
// PUBLIC OUTPUT
// ============================================================

function publicOutputFile(
  prefix,
  extension
) {

  const name =
    `${prefix}_${Date.now()}_${crypto
      .randomBytes(4)
      .toString("hex")}.${extension}`;

  return {
    filename:
      name,

    path:
      path.join(
        OUTPUT_DIR,
        name
      ),

    url:
      `/output/${name}`
  };
}


function writeScriptFile(
  script
) {

  const result =
    publicOutputFile(
      "movie_recap_script",
      "txt"
    );


  fs.writeFileSync(
    result.path,
    String(script || ""),
    "utf8"
  );


  return result;
}


function writeSrtFile(
  transcription
) {

  const result =
    publicOutputFile(
      "movie_recap_sync",
      "srt"
    );


  fs.writeFileSync(
    result.path,
    makeSrt(
      transcription
    ),
    "utf8"
  );


  return result;
}


function copyOutput(
  source,
  prefix,
  extension
) {

  const result =
    publicOutputFile(
      prefix,
      extension
    );


  fs.copyFileSync(
    source,
    result.path
  );


  return result;
}


// ============================================================
// API: HEALTH
// ============================================================

app.get(
  "/api/health",
  (req,res) => {

    res.json({
      ok:
        true,

      service:
        "Burmese SRT + AI Movie Recap",

      movieRecap:
        true,

      autoSync:
        true
    });

  }
);


// ============================================================
// API: VOICES
// ============================================================

app.get(
  "/api/recap/voices",
  (req,res) => {

    res.json({
      ok:
        true,

      voices:
        GEMINI_VOICES,

      model:
        TTS_MODEL
    });

  }
);


// ============================================================
// API: TRANSCRIBE
// ============================================================

app.post(
  "/api/transcribe",
  upload.single("video"),

  async (req,res) => {

    if (!req.file) {

      return jsonError(
        res,
        400,
        "Video မရှိပါ။"
      );

    }


    const key =
      getGroqKey(req);


    if (!key) {

      removeFile(
        req.file.path
      );

      return jsonError(
        res,
        400,
        "Groq API Key မရှိပါ။"
      );

    }


    try {

      const data =
        await groqTranscribe(
          req.file.path,
          "",
          key
        );


      res.json({
        ok:
          true,

        ...data
      });

    } catch (error) {

      jsonError(
        res,
        500,
        error.message
      );

    } finally {

      removeFile(
        req.file.path
      );

    }

  }
);


// ============================================================
// API: TRANSLATE
// ============================================================

app.post(
  "/api/translate",

  async (req,res) => {

    const key =
      getGeminiKey(req);

    const text =
      String(
        req.body?.text ??
        req.body?.transcript ??
        ""
      ).trim();


    if (!text) {

      return jsonError(
        res,
        400,
        "Translate text မရှိပါ။"
      );

    }


    if (!key) {

      return jsonError(
        res,
        400,
        "Gemini API Key မရှိပါ။"
      );

    }


    try {

      const data =
        await geminiGenerate(
          "gemini-3.8-flash",

          {
            contents: [
              {
                role:
                  "user",

                parts: [
                  {
                    text:
`Translate the following text into natural Myanmar subtitle language.

Preserve the original meaning.

Do not add information.

Return only the Myanmar translation.

${text}`
                  }
                ]
              }
            ],

            generationConfig: {
              temperature:
                0.2,

              maxOutputTokens:
                8000
            }
          },

          key
        );


      res.json({
        ok:
          true,

        text:
          geminiText(data)
      });


    } catch (error) {

      jsonError(
        res,
        500,
        error.message
      );

    }

  }
);


// ============================================================
// API: RECAP ANALYZE
// ============================================================

app.post(
  "/api/recap/analyze",
  upload.single("video"),

  async (req,res) => {

    if (!req.file) {

      return jsonError(
        res,
        400,
        "Video မရှိပါ။"
      );

    }


    const key =
      getGeminiKey(req);

    const job =
      path.join(
        WORK_DIR,
        uid("analyze")
      );


    fs.mkdirSync(
      job,
      {
        recursive:
          true
      }
    );


    try {

      const result =
        await analyzeVideoScenes(
          req.file.path,
          job,
          key
        );


      res.json({
        ok:
          true,

        ...result
      });


    } catch (error) {

      jsonError(
        res,
        500,
        error.message
      );


    } finally {

      removeFile(
        req.file.path
      );

      setTimeout(
        () =>
          removeDir(job),
        10 * 60 * 1000
      );

    }

  }
);


// ============================================================
// API: OLD TTS
// ============================================================

app.post(
  "/api/recap/tts",

  async (req,res) => {

    const key =
      getGeminiKey(req);

    const job =
      path.join(
        WORK_DIR,
        uid("tts")
      );


    fs.mkdirSync(
      job,
      {
        recursive:
          true
      }
    );


    try {

      const result =
        await geminiTTS({

          script:
            String(
              req.body?.script ||
              ""
            ),

          voice:
            req.body?.voice ||
            "Kore",

          emotion:
            req.body?.emotion ||
            "natural",

          speed:
            req.body?.voiceSpeed ||
            1,

          pitch:
            req.body?.voicePitch ||
            0,

          volume:
            req.body?.voiceVolume ||
            1,

          jobDir:
            job,

          apiKey:
            key

        });


      res.sendFile(
        result.audioPath,
        {
          headers: {
            "Content-Type":
              "audio/wav",

            "Content-Disposition":
              'inline; filename="voice.wav"'
          }
        }
      );


    } catch (error) {

      jsonError(
        res,
        500,
        error.message
      );

    } finally {

      setTimeout(
        () =>
          removeDir(job),
        10 * 60 * 1000
      );

    }

  }
);


// ============================================================
// API: OLD VOICE SYNC
// ============================================================

app.post(
  "/api/recap/voice-sync",
  upload.single("audio"),

  async (req,res) => {

    if (!req.file) {

      return jsonError(
        res,
        400,
        "Audio မရှိပါ။"
      );

    }


    const key =
      getGroqKey(req);


    try {

      const result =
        await groqTranscribe(
          req.file.path,
          req.body?.script ||
            "",
          key
        );


      res.json({
        ok:
          true,

        ...result
      });


    } catch (error) {

      jsonError(
        res,
        500,
        error.message
      );


    } finally {

      removeFile(
        req.file.path
      );

    }

  }
);


// ============================================================
// API: OLD RENDER
// ============================================================

app.post(
  "/api/render",
  upload.single("video"),

  async (req,res) => {

    if (!req.file) {

      return jsonError(
        res,
        400,
        "Video မရှိပါ။"
      );

    }


    const job =
      path.join(
        WORK_DIR,
        uid("render")
      );


    fs.mkdirSync(
      job,
      {
        recursive:
          true
      }
    );


    try {

      let transcription =
        req.body?.transcription ||
        {};


      if (
        typeof transcription ===
        "string"
      ) {

        transcription =
          JSON.parse(
            transcription
          );

      }


      const settings = {

        ...req.body,

        blurOriginal:
          req.body.blurOriginal ===
            "true" ||
          req.body.blurOriginal ===
            true,

        fontSize:
          Number(
            req.body.fontSize
          ) || 28,

        outline:
          Number(
            req.body.outline
          ) || 2

      };


      const audioPath =
        req.body.audioPath;


      if (!audioPath) {

        return jsonError(
          res,
          400,
          "Render audioPath မရှိပါ။"
        );

      }


      const result =
        await renderRecap({

          video:
            req.file.path,

          audio:
            audioPath,

          transcription,

          settings,

          jobDir:
            job

        });


      res.sendFile(
        result,
        {
          headers: {
            "Content-Type":
              "video/mp4",

            "Content-Disposition":
              'attachment; filename="myanmar_movie_recap.mp4"'
          }
        }
      );


    } catch (error) {

      jsonError(
        res,
        500,
        error.message
      );


    } finally {

      removeFile(
        req.file.path
      );

      setTimeout(
        () =>
          removeDir(job),
        10 * 60 * 1000
      );

    }

  }
);


// ============================================================
// NEW API
// 1. VIDEO -> AI SCRIPT
// ============================================================

app.post(
  "/api/movie-recap",
  upload.single("video"),

  async (req,res) => {

    if (!req.file) {

      return jsonError(
        res,
        400,
        "Movie Recap Video မရှိပါ။"
      );

    }


    const geminiKey =
      getGeminiKey(req);


    if (!geminiKey) {

      removeFile(
        req.file.path
      );

      return jsonError(
        res,
        400,
        "Gemini API Key မရှိပါ။"
      );

    }


    const job =
      path.join(
        WORK_DIR,
        uid("movie_script")
      );


    fs.mkdirSync(
      job,
      {
        recursive:
          true
      }
    );


    try {

      console.log(
        "MOVIE RECAP: AI scene analysis started"
      );


      const result =
        await analyzeVideoScenes(
          req.file.path,
          job,
          geminiKey
        );


      console.log(
        "MOVIE RECAP: AI script generated"
      );


      res.json({
        ok:
          true,

        script:
          result.script,

        duration:
          result.duration,

        frameCount:
          result.frameCount,

        model:
          result.model
      });


    } catch (error) {

      console.error(
        "MOVIE RECAP SCRIPT ERROR:",
        error
      );


      jsonError(
        res,
        500,
        error.message
      );


    } finally {

      removeFile(
        req.file.path
      );

      setTimeout(
        () =>
          removeDir(job),
        10 * 60 * 1000
      );

    }

  }
);


// ============================================================
// NEW API
// 2. SCRIPT -> GEMINI VOICE
// ============================================================

app.post(
  "/api/movie-voice",

  async (req,res) => {

    const geminiKey =
      getGeminiKey(req);


    if (!geminiKey) {

      return jsonError(
        res,
        400,
        "Gemini API Key မရှိပါ။"
      );

    }


    const script =
      String(
        req.body?.script ||
        ""
      ).trim();


    if (!script) {

      return jsonError(
        res,
        400,
        "Movie Recap Script မရှိပါ။"
      );

    }


    const job =
      path.join(
        WORK_DIR,
        uid("movie_voice")
      );


    fs.mkdirSync(
      job,
      {
        recursive:
          true
      }
    );


    try {

      const result =
        await geminiTTS({

          script,

          voice:
            req.body?.voice ||
            "Kore",

          emotion:
            req.body?.emotion ||
            "natural",

          speed:
            Number(
              req.body?.speed ??
              req.body?.voiceSpeed ??
              1
            ),

          pitch:
            Number(
              req.body?.pitch ??
              req.body?.voicePitch ??
              0
            ),

          volume:
            Number(
              req.body?.volume ??
              req.body?.voiceVolume ??
              1
            ),

          jobDir:
            job,

          apiKey:
            geminiKey

        });


      /*
        Copy TTS audio into public/output.

        Browser can then send the URL back
        to /api/movie-render.
      */

      const publicAudio =
        copyOutput(
          result.audioPath,
          "movie_recap_tts",
          "wav"
        );


      const duration =
        await mediaDuration(
          result.audioPath
        );


      console.log(
        "MOVIE RECAP: TTS generated:",
        duration,
        "seconds"
      );


      res.json({

        ok:
          true,

        audioUrl:
          publicAudio.url,

        duration,

        voice:
          result.voice,

        emotion:
          result.emotion,

        model:
          result.model

      });


    } catch (error) {

      console.error(
        "MOVIE RECAP TTS ERROR:",
        error
      );


      jsonError(
        res,
        500,
        error.message
      );


    } finally {

      setTimeout(
        () =>
          removeDir(job),
        10 * 60 * 1000
      );

    }

  }
);


// ============================================================
// RESOLVE PUBLIC AUDIO URL
// ============================================================

function resolvePublicOutputUrl(
  value
) {

  const raw =
    String(
      value ||
      ""
    );


  if (!raw) {
    return null;
  }


  /*
    Only allow /output/<filename>
    so arbitrary server filesystem paths
    cannot be requested.
  */

  let pathname =
    raw;


  try {

    if (
      pathname.startsWith("http://") ||
      pathname.startsWith("https://")
    ) {

      const url =
        new URL(
          pathname
        );

      pathname =
        url.pathname;

    }

  } catch (_) {}


  if (
    !pathname.startsWith(
      "/output/"
    )
  ) {

    return null;

  }


  const filename =
    path.basename(
      pathname
    );


  const resolved =
    path.join(
      OUTPUT_DIR,
      filename
    );


  const relative =
    path.relative(
      OUTPUT_DIR,
      resolved
    );


  if (
    relative.startsWith("..") ||
    path.isAbsolute(relative)
  ) {

    return null;

  }


  if (
    !fs.existsSync(
      resolved
    )
  ) {

    return null;

  }


  return resolved;
}


// ============================================================
// NEW API
// 3. AUTO SYNC + SRT + FINAL MP4
// ============================================================

app.post(
  "/api/movie-render",
  upload.single("video"),

  async (req,res) => {

    if (!req.file) {

      return jsonError(
        res,
        400,
        "Movie Recap Video မရှိပါ။"
      );

    }


    const geminiKey =
      getGeminiKey(req);

    const groqKey =
      getGroqKey(req);


    if (!geminiKey) {

      removeFile(
        req.file.path
      );

      return jsonError(
        res,
        400,
        "Gemini API Key မရှိပါ။"
      );

    }


    if (!groqKey) {

      removeFile(
        req.file.path
      );

      return jsonError(
        res,
        400,
        "Groq API Key မရှိပါ။"
      );

    }


    const job =
      path.join(
        WORK_DIR,
        uid("movie_render")
      );


    fs.mkdirSync(
      job,
      {
        recursive:
          true
      }
    );


    try {

      const script =
        String(
          req.body?.script ||
          ""
        ).trim();


      if (!script) {

        throw new Error(
          "Movie Recap Script မရှိပါ။"
        );

      }


      /*
        ------------------------------------------------------
        STEP A
        Original Video Duration
        ------------------------------------------------------
      */

      console.log(
        "MOVIE RECAP: reading video duration"
      );


      const targetDuration =
        await videoDuration(
          req.file.path
        );


      /*
        ------------------------------------------------------
        STEP B
        Resolve generated TTS audio
        ------------------------------------------------------
      */

      const voiceUrl =
        req.body?.voiceUrl;


      let sourceAudio =
        resolvePublicOutputUrl(
          voiceUrl
        );


      /*
        If browser sent no public URL,
        generate TTS again server-side.
      */

      if (!sourceAudio) {

        console.log(
          "MOVIE RECAP: TTS URL unavailable, generating TTS server-side"
        );


        const tts =
          await geminiTTS({

            script,

            voice:
              req.body?.voice ||
              "Kore",

            emotion:
              req.body?.emotion ||
              "natural",

            speed:
              Number(
                req.body?.voiceSpeed
              ) || 1,

            pitch:
              Number(
                req.body?.voicePitch
              ) || 0,

            volume:
              Number(
                req.body?.voiceVolume
              ) || 1,

            jobDir:
              job,

            apiKey:
              geminiKey

          });


        sourceAudio =
          tts.audioPath;

      }


      /*
        ------------------------------------------------------
        STEP C
        AUTO SYNC
        ------------------------------------------------------
      */

      console.log(
        "MOVIE RECAP: auto syncing TTS to video"
      );


      const syncedAudio =
        path.join(
          job,
          "synced_voice.wav"
        );


      const syncInfo =
        await syncAudioToVideo(
          sourceAudio,
          targetDuration,
          syncedAudio
        );


      console.log(
        "MOVIE RECAP: sync factor:",
        syncInfo.tempoFactor
      );


      /*
        ------------------------------------------------------
        STEP D
        Groq Whisper on the ACTUAL SYNCED AUDIO
        ------------------------------------------------------
      */

      console.log(
        "MOVIE RECAP: generating actual speech timing"
      );


      const whisper =
        await groqTranscribe(
          syncedAudio,
          script,
          groqKey
        );


      const events =
        eventsFromTranscription(
          whisper
        );


      if (!events.length) {

        throw new Error(
          "Sync Voice အတွက် Whisper timing မရပါ။"
        );

      }


      /*
        ------------------------------------------------------
        STEP E
        Sync SRT
        ------------------------------------------------------
      */

      const transcription =
        {
          ...whisper,

          segments:
            events
        };


      const srtFile =
        writeSrtFile(
          transcription
        );


      /*
        ------------------------------------------------------
        STEP F
        Save AI Script
        ------------------------------------------------------
      */

      const scriptFile =
        writeScriptFile(
          script
        );


      /*
        ------------------------------------------------------
        STEP G
        Save synced audio
        ------------------------------------------------------
      */

      const audioFile =
        copyOutput(
          syncedAudio,
          "movie_recap_synced_voice",
          "wav"
        );


      /*
        ------------------------------------------------------
        STEP H
        Render Final MP4
        ------------------------------------------------------
      */

      const settings = {

        blurOriginal:
          req.body?.blurOriginal ===
            "true" ||
          req.body?.blurOriginal ===
            true,

        blurX:
          clamp(
            req.body?.blurX,
            0,
            100
          ),

        blurY:
          clamp(
            req.body?.blurY,
            0,
            100
          ),

        blurWidth:
          clamp(
            req.body?.blurW ??
            req.body?.blurWidth,
            1,
            100
          ),

        blurHeight:
          clamp(
            req.body?.blurH ??
            req.body?.blurHeight,
            1,
            100
          ),

        subtitleX:
          req.body?.subtitleX !==
            "" &&
          req.body?.subtitleX !=
            null
            ? clamp(
                req.body.subtitleX,
                0,
                100
              )
            : NaN,

        subtitleY:
          req.body?.subtitleY !==
            "" &&
          req.body?.subtitleY !=
            null
            ? clamp(
                req.body.subtitleY,
                0,
                100
              )
            : NaN,

        fontName:
          req.body?.fontName ||
          "Noto Sans Myanmar",

        fontSize:
          clamp(
            req.body?.fontSize,
            10,
            120
          ),

        outline:
          clamp(
            req.body?.outline,
            0,
            10
          ),

        textColor:
          req.body?.textColor ||
          "#FFFFFF",

        position:
          req.body?.position ||
          "bottom",

        outputSize:
          req.body?.outputSize ||
          "original"

      };


      console.log(
        "MOVIE RECAP: rendering final MP4"
      );


      const finalVideo =
        await renderRecap({

          video:
            req.file.path,

          audio:
            syncedAudio,

          transcription,

          settings,

          jobDir:
            job

        });


      /*
        ------------------------------------------------------
        STEP I
        Copy Final MP4 to public/output
        ------------------------------------------------------
      */

      const finalFile =
        copyOutput(
          finalVideo,
          "movie_recap_final",
          "mp4"
        );


      console.log(
        "MOVIE RECAP: FINAL MP4 READY"
      );


      /*
        ------------------------------------------------------
        FINAL RESPONSE
        ------------------------------------------------------
      */

      res.json({

        ok:
          true,

        message:
          "Movie Recap အားလုံးပြီးပါပြီ။",

        downloadUrl:
          finalFile.url,

        audioUrl:
          audioFile.url,

        srtUrl:
          srtFile.url,

        scriptUrl:
          scriptFile.url,

        videoDuration:
          targetDuration,

        originalTtsDuration:
          syncInfo.inputDuration,

        syncedAudioDuration:
          syncInfo.outputDuration,

        syncFactor:
          syncInfo.tempoFactor,

        subtitleCount:
          events.length

      });


    } catch (error) {

      console.error(
        "MOVIE RECAP RENDER ERROR:",
        error
      );


      jsonError(
        res,
        500,
        error.message
      );


    } finally {

      removeFile(
        req.file.path
      );


      /*
        Keep temporary job for a while
        for debugging / downloads.
      */

      setTimeout(
        () =>
          removeDir(job),
        15 * 60 * 1000
      );

    }

  }
);


// ============================================================
// OLD ONE-CLICK
// ============================================================

app.post(
  "/api/recap/one-click",
  upload.single("video"),

  async (req,res) => {

    if (!req.file) {

      return jsonError(
        res,
        400,
        "Video မရှိပါ။"
      );

    }


    const geminiKey =
      getGeminiKey(req);

    const groqKey =
      getGroqKey(req);


    if (!geminiKey) {

      removeFile(
        req.file.path
      );

      return jsonError(
        res,
        400,
        "Gemini API Key မရှိပါ။"
      );

    }


    if (!groqKey) {

      removeFile(
        req.file.path
      );

      return jsonError(
        res,
        400,
        "Groq API Key မရှိပါ။"
      );

    }


    const job =
      path.join(
        WORK_DIR,
        uid("oneclick")
      );


    fs.mkdirSync(
      job,
      {
        recursive:
          true
      }
    );


    try {

      /*
        Keep old endpoint working.

        Video
        -> AI Script
        -> TTS
        -> Sync
        -> Whisper
        -> Final MP4
      */

      const analysis =
        await analyzeVideoScenes(
          req.file.path,
          job,
          geminiKey
        );


      const targetDuration =
        await videoDuration(
          req.file.path
        );


      const tts =
        await geminiTTS({

          script:
            analysis.script,

          voice:
            req.body?.voice ||
            "Kore",

          emotion:
            req.body?.emotion ||
            "natural",

          speed:
            Number(
              req.body?.voiceSpeed
            ) || 1,

          pitch:
            Number(
              req.body?.voicePitch
            ) || 0,

          volume:
            Number(
              req.body?.voiceVolume
            ) || 1,

          jobDir:
            job,

          apiKey:
            geminiKey

        });


      const syncedAudio =
        path.join(
          job,
          "synced_voice.wav"
        );


      await syncAudioToVideo(
        tts.audioPath,
        targetDuration,
        syncedAudio
      );


      const whisper =
        await groqTranscribe(
          syncedAudio,
          analysis.script,
          groqKey
        );


      const transcription =
        {
          ...whisper,

          segments:
            eventsFromTranscription(
              whisper
            )
        };


      const settings = {

        blurOriginal:
          req.body?.blurOriginal ===
            "true" ||
          req.body?.blurOriginal ===
            true,

        blurX:
          clamp(
            req.body?.blurX,
            0,
            100
          ),

        blurY:
          clamp(
            req.body?.blurY,
            0,
            100
          ),

        blurWidth:
          clamp(
            req.body?.blurWidth,
            1,
            100
          ),

        blurHeight:
          clamp(
            req.body?.blurHeight,
            1,
            100
          ),

        subtitleX:
          req.body?.subtitleX !==
            "" &&
          req.body?.subtitleX !=
            null
            ? clamp(
                req.body.subtitleX,
                0,
                100
              )
            : NaN,

        subtitleY:
          req.body?.subtitleY !==
            "" &&
          req.body?.subtitleY !=
            null
            ? clamp(
                req.body.subtitleY,
                0,
                100
              )
            : NaN,

        fontName:
          req.body?.fontName ||
          "Noto Sans Myanmar",

        fontSize:
          clamp(
            req.body?.fontSize,
            10,
            120
          ),

        outline:
          clamp(
            req.body?.outline,
            0,
            10
          ),

        textColor:
          req.body?.textColor ||
          "#FFFFFF",

        position:
          req.body?.position ||
          "bottom",

        outputSize:
          req.body?.outputSize ||
          "original"

      };


      const finalVideo =
        await renderRecap({

          video:
            req.file.path,

          audio:
            syncedAudio,

          transcription,

          settings,

          jobDir:
            job

        });


      res.sendFile(
        finalVideo,
        {
          headers: {
            "Content-Type":
              "video/mp4",

            "Content-Disposition":
              'attachment; filename="myanmar_movie_recap.mp4"'
          }
        }
      );


    } catch (error) {

      console.error(
        "ONE CLICK ERROR:",
        error
      );


      jsonError(
        res,
        500,
        error.message
      );


    } finally {

      removeFile(
        req.file.path
      );


      setTimeout(
        () =>
          removeDir(job),
        15 * 60 * 1000
      );

    }

  }
);


// ============================================================
// CLEAN OLD OUTPUT FILES
// ============================================================

function cleanOldOutputs() {

  try {

    const files =
      fs.readdirSync(
        OUTPUT_DIR
      );

    const now =
      Date.now();


    for (
      const file
      of files
    ) {

      const full =
        path.join(
          OUTPUT_DIR,
          file
        );


      const stat =
        fs.statSync(
          full
        );


      /*
        Delete files older than 6 hours.
      */

      if (
        now -
        stat.mtimeMs >
        6 *
        60 *
        60 *
        1000
      ) {

        fs.rmSync(
          full,
          {
            force:
              true
          }
        );

      }

    }

  } catch (_) {}

}


setInterval(
  cleanOldOutputs,
  30 * 60 * 1000
);


// ============================================================
// STATIC FALLBACK
// ============================================================

app.use(
  (req,res,next) => {

    if (
      req.method ===
        "GET" &&
      !req.path.startsWith(
        "/api/"
      )
    ) {

      return res.sendFile(
        path.join(
          PUBLIC_DIR,
          "index.html"
        )
      );

    }

    next();

  }
);


// ============================================================
// ERROR HANDLER
// ============================================================

app.use(
  (
    error,
    req,
    res,
    next
  ) => {

    console.error(
      "SERVER ERROR:",
      error
    );


    if (
      res.headersSent
    ) {

      return next(
        error
      );

    }


    return jsonError(
      res,
      500,
      error.message ||
        "Server error"
    );

  }
);


// ============================================================
// START
// ============================================================

app.listen(
  PORT,
  () => {

    console.log(
      `Myanmar SRT server running on port ${PORT}`
    );

    console.log(
      "Gemini API:",
      process.env.GEMINI_API_KEY
        ? "OK"
        : "MISSING"
    );

    console.log(
      "Groq API:",
      process.env.GROQ_API_KEY
        ? "OK"
        : "MISSING"
    );

    console.log(
      "AI Movie Recap:",
      "ENABLED"
    );

    console.log(
      "Auto Sync:",
      "ENABLED"
    );

  }
);
