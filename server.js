const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const crypto = require("crypto");

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.static("public"));

/* =========================
   DIRECTORIES
========================= */

const uploadDir = path.join(__dirname, "uploads");
const outputDir = path.join(__dirname, "public", "output");

fs.mkdirSync(uploadDir, { recursive: true });
fs.mkdirSync(outputDir, { recursive: true });

/* =========================
   UPLOAD
========================= */

const upload = multer({
  dest: uploadDir,
  limits: {
    fileSize: 300 * 1024 * 1024
  }
});

/* =========================
   LIMITS
========================= */

const MAX_SIZE = 300 * 1024 * 1024;
const MAX_MINUTES = 5;

/* =========================
   API KEY HELPERS
========================= */

function getGroqKey(req) {
  return (
    req.headers["x-groq-api-key"] ||
    process.env.GROQ_API_KEY ||
    ""
  ).trim();
}

function getGeminiKey(req) {
  return (
    req.headers["x-gemini-api-key"] ||
    process.env.GEMINI_API_KEY ||
    ""
  ).trim();
}

/* =========================
   HEALTH
========================= */

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    message: "Myanmar SRT backend is running"
  });
});

/* =========================================================
   GROQ TRANSCRIPTION
========================================================= */

app.post(
  "/api/transcribe",
  upload.single("video"),
  async (req, res) => {
    let filePath = null;

    try {
      if (!req.file) {
        return res.status(400).json({
          error: "Video file မတွေ့ပါ"
        });
      }

      filePath = req.file.path;

      if (req.file.size > MAX_SIZE) {
        return res.status(400).json({
          error: "Video size က 300MB ထက်မကျော်ရပါ"
        });
      }

      const groqKey = getGroqKey(req);

      if (!groqKey) {
        return res.status(400).json({
          error: "Groq API Key ထည့်ပေးပါ"
        });
      }

      const videoBuffer = fs.readFileSync(filePath);

      const form = new FormData();

      const blob = new Blob(
        [videoBuffer],
        {
          type: req.file.mimetype || "video/mp4"
        }
      );

      form.append(
        "file",
        blob,
        req.file.originalname
      );

      form.append(
        "model",
        "whisper-large-v3"
      );

      form.append(
        "response_format",
        "verbose_json"
      );

      form.append(
        "timestamp_granularities[]",
        "segment"
      );

      form.append(
        "temperature",
        "0"
      );

      const groqResponse = await fetch(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${groqKey}`
          },
          body: form
        }
      );

      const data = await groqResponse.json();

      if (!groqResponse.ok) {
        console.error(
          "Groq request failed:",
          data?.error?.message || "Unknown error"
        );

        return res.status(502).json({
          error:
            data?.error?.message ||
            "Groq transcription မအောင်မြင်ပါ"
        });
      }

      const segments =
        Array.isArray(data.segments)
          ? data.segments
          : [];

      const transcript =
        segments
          .map(segment => ({
            start: Number(segment.start || 0),
            end: Number(segment.end || 0),
            text: String(segment.text || "").trim()
          }))
          .filter(item => item.text);

      if (!transcript.length) {
        return res.status(502).json({
          error: "Groq က transcript မရပါ"
        });
      }

      const lastEnd =
        transcript[transcript.length - 1].end;

      if (lastEnd / 60 > MAX_MINUTES + 0.25) {
        return res.status(400).json({
          error:
            "Video က 5 မိနစ်ထက်ရှည်နေပါတယ်"
        });
      }

      return res.json({
        ok: true,
        durationSeconds: lastEnd,
        transcript
      });

    } catch (error) {
      console.error(
        "Transcription error:",
        error.message
      );

      return res.status(500).json({
        error:
          error.message ||
          "Transcription error"
      });

    } finally {
      if (filePath) {
        try {
          fs.unlinkSync(filePath);
        } catch {}
      }
    }
  }
);

/* =========================================================
   GEMINI BASIC HELPER
========================================================= */

async function callGemini(
  model,
  geminiKey,
  prompt
) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": geminiKey
      },

      body: JSON.stringify({
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
          responseMimeType:
            "application/json"
        }
      })
    }
  );

  const raw = await response.text();

  let data;

  try {
    data = JSON.parse(raw);
  } catch {
    return {
      ok: false,
      status: response.status,
      error: "Gemini JSON response မမှန်ပါ"
    };
  }

  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      error:
        data?.error?.message ||
        "Gemini request failed"
    };
  }

  const responseText =
    data
      ?.candidates?.[0]
      ?.content?.parts?.[0]
      ?.text;

  if (!responseText) {
    return {
      ok: false,
      status: 502,
      error: "Gemini response မရပါ"
    };
  }

  let parsed;

  try {
    parsed = JSON.parse(responseText);
  } catch {
    return {
      ok: false,
      status: 502,
      error: "Gemini JSON မမှန်ပါ"
    };
  }

  return {
    ok: true,
    data: parsed
  };
}

/* =========================================================
   GEMINI MYANMAR TRANSLATION
========================================================= */

app.post(
  "/api/translate",
  async (req, res) => {
    try {
      const geminiKey = getGeminiKey(req);

      if (!geminiKey) {
        return res.status(400).json({
          error: "Gemini API Key ထည့်ပေးပါ"
        });
      }

      const transcript =
        req.body?.transcript;

      if (
        !Array.isArray(transcript) ||
        transcript.length === 0
      ) {
        return res.status(400).json({
          error: "Transcript မတွေ့ပါ"
        });
      }

      const source =
        transcript
          .map(
            (item, index) =>
              `${index + 1}. ` +
              `[${item.start} --> ${item.end}] ` +
              `${item.text}`
          )
          .join("\n");

      const prompt = `
You are a professional Myanmar movie subtitle translator.

Translate the following subtitle transcript into natural, fluent Myanmar language.

Rules:
- Translate every subtitle line.
- Keep the exact same order.
- Do not remove any line.
- Do not merge lines.
- Do not add new lines.
- Do not add explanations.
- Do not add English translation.
- Keep names and proper nouns natural.
- Use natural spoken Myanmar suitable for movies and dramas.
- Preserve meaning and emotion.
- Keep subtitles reasonably short.
- Return JSON only.

Required JSON:

{
  "segments": [
    {
      "id": 1,
      "text": "မြန်မာဘာသာပြန်"
    }
  ]
}

SOURCE:

${source}
`;

      const models = [
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite"
      ];

      let finalResult = null;
      let lastError = null;

      for (const model of models) {
        try {
          const result =
            await callGemini(
              model,
              geminiKey,
              prompt
            );

          if (result.ok) {
            finalResult = result.data;
            break;
          }

          lastError = result.error;

        } catch (error) {
          lastError = error.message;
        }
      }

      if (!finalResult) {
        return res.status(502).json({
          error:
            lastError ||
            "Gemini translation မအောင်မြင်ပါ"
        });
      }

      const translatedSegments =
        Array.isArray(finalResult?.segments)
          ? finalResult.segments
          : [];

      const result =
        transcript
          .map((original, index) => {
            const translated =
              translatedSegments.find(
                item =>
                  Number(item.id) ===
                  index + 1
              );

            if (!translated) {
              return null;
            }

            const text =
              String(
                translated.text || ""
              ).trim();

            if (!text) {
              return null;
            }

            return {
              start:
                Number(original.start),
              end:
                Number(original.end),
              text
            };
          })
          .filter(Boolean);

      if (!result.length) {
        return res.status(502).json({
          error:
            "Myanmar translation မရပါ"
        });
      }

      return res.json({
        ok: true,
        transcript: result
      });

    } catch (error) {
      console.error(
        "Translation error:",
        error.message
      );

      return res.status(500).json({
        error:
          error.message ||
          "Translation error"
      });
    }
  }
);

/* =========================================================
   MOVIE RECAP - GEMINI VIDEO UPLOAD
========================================================= */

async function uploadGeminiFile(
  filePath,
  mimeType,
  geminiKey
) {
  const buffer =
    fs.readFileSync(filePath);

  const uploadResponse =
    await fetch(
      "https://generativelanguage.googleapis.com/upload/v1beta/files",
      {
        method: "POST",

        headers: {
          "x-goog-api-key":
            geminiKey,

          "Content-Type":
            "application/octet-stream",

          "X-Goog-Upload-Protocol":
            "raw",

          "X-Goog-Upload-File-Name":
            path.basename(filePath),

          "X-Goog-Upload-Header-Content-Length":
            String(buffer.length),

          "X-Goog-Upload-Header-Content-Type":
            mimeType || "video/mp4"
        },

        body: buffer
      }
    );

  const text =
    await uploadResponse.text();

  let data;

  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(
      "Gemini File upload response မမှန်ပါ"
    );
  }

  if (!uploadResponse.ok) {
    throw new Error(
      data?.error?.message ||
      "Gemini video upload မအောင်မြင်ပါ"
    );
  }

  return data.file || data;
}

/* =========================================================
   MOVIE RECAP - WAIT FOR FILE
========================================================= */

async function waitForGeminiFile(
  fileName,
  geminiKey
) {
  for (let i = 0; i < 60; i++) {

    const response =
      await fetch(
        `https://generativelanguage.googleapis.com/v1beta/${fileName}`,
        {
          headers: {
            "x-goog-api-key":
              geminiKey
          }
        }
      );

    const data =
      await response.json();

    if (!response.ok) {
      throw new Error(
        data?.error?.message ||
        "Gemini file status မရပါ"
      );
    }

    const state =
      data?.state ||
      data?.file?.state;

    if (
      state === "ACTIVE" ||
      state?.name === "ACTIVE"
    ) {
      return data;
    }

    if (
      state === "FAILED" ||
      state?.name === "FAILED"
    ) {
      throw new Error(
        "Gemini video processing failed"
      );
    }

    await new Promise(
      resolve =>
        setTimeout(resolve, 3000)
    );
  }

  throw new Error(
    "Gemini video processing timeout"
  );
}

/* =========================================================
   MOVIE RECAP - P2
========================================================= */

app.post(
  "/api/movie-recap",
  upload.single("video"),
  async (req, res) => {

    let filePath = null;

    try {

      if (!req.file) {
        return res.status(400).json({
          error:
            "Movie Recap video မတွေ့ပါ"
        });
      }

      filePath = req.file.path;

      if (req.file.size > MAX_SIZE) {
        return res.status(400).json({
          error:
            "Video size က 300MB ထက်မကျော်ရပါ"
        });
      }

      const geminiKey =
        getGeminiKey(req);

      if (!geminiKey) {
        return res.status(400).json({
          error:
            "Gemini API Key ထည့်ပေးပါ"
        });
      }

      console.log(
        "Uploading Movie Recap video to Gemini..."
      );

      const uploaded =
        await uploadGeminiFile(
          filePath,
          req.file.mimetype,
          geminiKey
        );

      const fileName =
        uploaded?.name;

      const fileUri =
        uploaded?.uri ||
        uploaded?.fileUri;

      if (!fileName || !fileUri) {
        throw new Error(
          "Gemini file reference မရပါ"
        );
      }

      const activeFile =
        await waitForGeminiFile(
          fileName,
          geminiKey
        );

      const activeUri =
        activeFile?.uri ||
        activeFile?.file?.uri ||
        fileUri;

      const prompt = `
You are a professional Myanmar movie recap script writer.

Watch and understand the entire video.

Create a natural Myanmar-language movie recap narration.

Requirements:
- Explain what happens in the video naturally.
- Follow the actual story order.
- Mention important characters and actions.
- Keep the narration interesting and easy to understand.
- Do not invent events that are not shown.
- Do not describe every tiny visual detail.
- Write like a professional Myanmar movie recap narrator.
- Use spoken Myanmar that sounds natural when read aloud.
- Do not use bullet points.
- Do not add headings.
- Return only the narration script.
`;

      const response =
        await fetch(
          "https://generativelanguage.googleapis.com/v1beta/interactions",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",

              "x-goog-api-key":
                geminiKey
            },

            body: JSON.stringify({
              model:
                "gemini-3.8-flash",

              input: [
                {
                  type: "video",
                  uri: activeUri,
                  mime_type:
                    req.file.mimetype ||
                    "video/mp4"
                },

                {
                  type: "text",
                  text: prompt
                }
              ]
            })
          }
        );

      const raw =
        await response.text();

      let data;

      try {
        data = JSON.parse(raw);
      } catch {
        throw new Error(
          "Gemini recap response မမှန်ပါ"
        );
      }

      if (!response.ok) {
        throw new Error(
          data?.error?.message ||
          "Gemini Movie Recap မအောင်မြင်ပါ"
        );
      }

      let script =
        data?.output_text ||
        "";

      if (!script) {

        const parts =
          data?.steps
            ?.flatMap(
              step =>
                Array.isArray(step.content)
                  ? step.content
                  : []
            )
            ?.filter(
              part =>
                part?.type === "text"
            )
            ?.map(
              part => part.text
            ) ||
            [];

        script =
          parts.join("\n");
      }

      script =
        String(script)
          .trim();

      if (!script) {
        throw new Error(
          "Movie Recap Script မရပါ"
        );
      }

      return res.json({
        ok: true,
        script
      });

    } catch (error) {

      console.error(
        "Movie Recap error:",
        error.message
      );

      return res.status(500).json({
        error:
          error.message ||
          "Movie Recap မအောင်မြင်ပါ"
      });

    } finally {

      if (filePath) {
        try {
          fs.unlinkSync(filePath);
        } catch {}
      }
    }
  }
);

/* =========================================================
   GEMINI TTS
========================================================= */

const MOVIE_VOICES = {
  thiha: "Kore",
  nila: "Aoede"
};

/*
   IMPORTANT:
   TTS သို့ script စာသားကိုပဲ ပို့သည်။
   style instruction ကို spoken text ထဲ မထည့်တော့ပါ။
*/

async function generateGeminiTTS(
  text,
  geminiKey,
  voiceName,
  speed,
  pitch
) {

  const safeSpeed =
    Math.min(
      2,
      Math.max(
        0.5,
        Number(speed || 1)
      )
    );

  const safePitch =
    Math.min(
      12,
      Math.max(
        -12,
        Number(pitch || 0)
      )
    );

  /*
     Gemini TTS ကို
     "script ကိုပဲဖတ်ပါ"
     လို့ သီးခြား instruction ပေးပြီး
     spoken content ထဲမှာ script သီးသန့်ထားသည်။
  */

  const response =
    await fetch(
      "https://generativelanguage.googleapis.com/v1beta/interactions",
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",

          "x-goog-api-key":
            geminiKey
        },

        body: JSON.stringify({

          model:
            "gemini-3.8-flash-tts",

          input: [

            {
              type: "user_input",

              content: [

                {
                  type: "text",

                  text:
                    "Read the following Myanmar movie recap script verbatim. " +
                    "Speak ONLY the script. " +
                    "Do not read these instructions. " +
                    "Do not add any words, explanations, introductions, or endings.\n\n" +
                    "SCRIPT:\n" +
                    text
                }

              ]
            }

          ],

          response_format: {
            type: "audio",
            mime_type: "audio/wav",
            sample_rate: 24000
          },

          generation_config: {
            speech_config: [
              {
                voice:
                  voiceName
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
    data = JSON.parse(raw);
  } catch {
    throw new Error(
      "Gemini TTS response မမှန်ပါ"
    );
  }

  if (!response.ok) {
    throw new Error(
      data?.error?.message ||
      "Gemini TTS မအောင်မြင်ပါ"
    );
  }

  let audioBase64 =
    data?.output_audio?.data;

  if (!audioBase64) {

    const audioParts =
      data?.steps
        ?.flatMap(
          step =>
            Array.isArray(step.content)
              ? step.content
              : []
        )
        ?.filter(
          part =>
            part?.type === "audio"
        ) || [];

    const lastAudio =
      audioParts[audioParts.length - 1];

    audioBase64 =
      lastAudio?.data;
  }

  if (!audioBase64) {
    throw new Error(
      "TTS audio မရပါ"
    );
  }

  return {

    audio:
      Buffer.from(
        audioBase64,
        "base64"
      ),

    speed:
      safeSpeed,

    pitch:
      safePitch
  };
}

/* =========================================================
   MOVIE RECAP - P3 VOICE
========================================================= */

app.post(
  "/api/movie-voice",
  async (req, res) => {

    try {

      const geminiKey =
        getGeminiKey(req);

      if (!geminiKey) {
        return res.status(400).json({
          error:
            "Gemini API Key ထည့်ပေးပါ"
        });
      }

      const script =
        String(
          req.body?.script ||
          ""
        ).trim();

      if (!script) {
        return res.status(400).json({
          error:
            "Movie Recap Script မတွေ့ပါ"
        });
      }

      const voice =
        String(
          req.body?.voice ||
          "thiha"
        );

      const voiceName =
        MOVIE_VOICES[voice] ||
        MOVIE_VOICES.thiha;

      const speed =
        Number(
          req.body?.voiceSpeed ||
          1
        );

      const volume =
        Math.min(
          2,
          Math.max(
            0,
            Number(
              req.body?.voiceVolume ||
              1
            )
          )
        );

      const pitch =
        Number(
          req.body?.voicePitch ||
          0
        );

      console.log(
        "Generating Movie Recap voice:",
        voice,
        voiceName
      );

      const result =
        await generateGeminiTTS(
          script,
          geminiKey,
          voiceName,
          speed,
          pitch
        );

      const jobId =
        crypto.randomUUID();

      const audioPath =
        path.join(
          outputDir,
          `${jobId}.wav`
        );

      fs.writeFileSync(
        audioPath,
        result.audio
      );

      return res.json({
        ok: true,

        audioUrl:
          `/output/${path.basename(
            audioPath
          )}`,

        voice,
        voiceName,

        speed:
          result.speed,

        volume,

        pitch
      });

    } catch (error) {

      console.error(
        "Movie voice error:",
        error.message
      );

      return res.status(500).json({
        error:
          error.message ||
          "Movie voice မအောင်မြင်ပါ"
      });
    }
  }
);

/* =========================================================
   SRT TIME FORMAT
========================================================= */

function formatSrtTime(seconds) {

  const totalMs =
    Math.max(
      0,
      Math.round(
        Number(seconds) * 1000
      )
    );

  const hours =
    Math.floor(
      totalMs / 3600000
    );

  const minutes =
    Math.floor(
      (totalMs % 3600000) /
      60000
    );

  const secs =
    Math.floor(
      (totalMs % 60000) /
      1000
    );

  const ms =
    totalMs % 1000;

  return (
    String(hours).padStart(2, "0") +
    ":" +
    String(minutes).padStart(2, "0") +
    ":" +
    String(secs).padStart(2, "0") +
    "," +
    String(ms).padStart(3, "0")
  );
}

/* =========================================================
   CREATE SRT
========================================================= */

function makeSrt(transcript) {

  return transcript
    .map(
      (item, index) =>
        `${index + 1}\n` +
        `${formatSrtTime(item.start)} --> ` +
        `${formatSrtTime(item.end)}\n` +
        `${String(item.text || "").trim()}\n`
    )
    .join("\n");
}

/* =========================================================
   FFMPEG
========================================================= */

function runFfmpeg(args) {

  return new Promise(
    (resolve, reject) => {

      const ffmpeg =
        spawn(
          "ffmpeg",
          args
        );

      let stderr = "";

      ffmpeg.stderr.on(
        "data",
        data => {
          stderr +=
            data.toString();
        }
      );

      ffmpeg.on(
        "error",
        error => {
          reject(error);
        }
      );

      ffmpeg.on(
        "close",
        code => {

          if (code === 0) {
            resolve();
          } else {
            reject(
              new Error(
                stderr.slice(-8000) ||
                `FFmpeg exited with code ${code}`
              )
            );
          }
        }
      );
    }
  );
}

/* =========================================================
   OUTPUT SIZE
========================================================= */

function getOutputFilter(
  outputSize
) {

  switch (outputSize) {

    case "9:16":
      return (
        "scale=720:1280:force_original_aspect_ratio=increase," +
        "crop=720:1280"
      );

    case "1:1":
      return (
        "scale=1080:1080:force_original_aspect_ratio=increase," +
        "crop=1080:1080"
      );

    case "16:9":
      return (
        "scale=1280:720:force_original_aspect_ratio=increase," +
        "crop=1280:720"
      );

    default:
      return null;
  }
}

/* =========================================================
   HEX COLOR -> ASS COLOR
========================================================= */

function hexToAssColor(hex) {

  let value =
    String(hex || "FFFFFF")
      .replace(
        /[^A-Fa-f0-9]/g,
        ""
      )
      .slice(0, 6);

  if (value.length !== 6) {
    value = "FFFFFF";
  }

  const rr =
    value.slice(0, 2);

  const gg =
    value.slice(2, 4);

  const bb =
    value.slice(4, 6);

  /*
     ASS uses AABBGGRR
     Alpha = 00
  */

  return `&H00${bb}${gg}${rr}`;
}

/* =========================================================
   ASS TEXT ESCAPE
========================================================= */

function escapeAssText(text) {

  return String(text || "")
    .replace(/\r/g, "")
    .replace(/\n/g, "\\N")
    .replace(/\{/g, "\\{")
    .replace(/\}/g, "\\}");
}

/* =========================================================
   ASS TIME FORMAT
========================================================= */

function formatAssTime(seconds) {

  const total =
    Math.max(
      0,
      Number(seconds || 0)
    );

  const hours =
    Math.floor(
      total / 3600
    );

  const minutes =
    Math.floor(
      (total % 3600) / 60
    );

  const secs =
    Math.floor(
      total % 60
    );

  const centiseconds =
    Math.floor(
      (total - Math.floor(total)) * 100
    );

  return (
    `${hours}:` +
    `${String(minutes).padStart(2, "0")}:` +
    `${String(secs).padStart(2, "0")}.` +
    `${String(centiseconds).padStart(2, "0")}`
  );
}

/* =========================================================
   CREATE ASS SUBTITLE
========================================================= */

function makeAss(
  transcript,
  options
) {

  const fontName =
    String(
      options.fontName ||
      "Noto Sans Myanmar"
    );

  const fontSize =
    Number(
      options.fontSize ||
      25
    );

  const outline =
    Number(
      options.outline ||
      2
    );

  const textColor =
    hexToAssColor(
      options.textColor ||
      "FFFFFF"
    );

  const position =
    String(
      options.position ||
      "bottom"
    );

  let alignment = 2;

  if (position === "top") {
    alignment = 8;
  }

  if (position === "middle") {
    alignment = 5;
  }

  const header = `
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Recap,${fontName},${fontSize},${textColor},${textColor},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,${outline},0,${alignment},40,40,45,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`.trim() + "\n";

  const lines =
    transcript.map(
      item => {

        const start =
          formatAssTime(
            item.start
          );

        const end =
          formatAssTime(
            item.end
          );

        const text =
          escapeAssText(
            item.text
          );

        return (
          `Dialogue: 0,${start},${end},Recap,,0,0,0,,${text}`
        );
      }
    );

  return (
    header +
    lines.join("\n") +
    "\n"
  );
}

/* =========================================================
   ASS FILTER PATH
========================================================= */

function escapeAssFilterPath(
  filePath
) {

  return String(filePath)
    .replace(/\\/g, "/")
    .replace(/'/g, "\\'");
}

/* =========================================================
   SAFE NUMBER
========================================================= */

function clamp(
  value,
  min,
  max,
  fallback
) {

  const number =
    Number(value);

  if (!Number.isFinite(number)) {
    return fallback;
  }

  return Math.min(
    max,
    Math.max(
      min,
      number
    )
  );
}

/* =========================================================
   MOVIE RECAP - P5 FINAL MP4
========================================================= */

app.post(
  "/api/movie-render",
  upload.single("video"),
  async (req, res) => {

    let inputPath = null;
    let audioPath = null;
    let assPath = null;
    let outputPath = null;

    try {

      if (!req.file) {
        return res.status(400).json({
          error:
            "Movie Recap video မတွေ့ပါ"
        });
      }

      inputPath =
        req.file.path;

      if (req.file.size > MAX_SIZE) {
        return res.status(400).json({
          error:
            "Video size က 300MB ထက်မကျော်ရပါ"
        });
      }

      const script =
        String(
          req.body?.script ||
          ""
        ).trim();

      if (!script) {
        return res.status(400).json({
          error:
            "Movie Recap Script မတွေ့ပါ"
        });
      }

      /* =========================
         STYLE
      ========================= */

      const fontName =
        String(
          req.body?.fontName ||
          "Noto Sans Myanmar"
        );

      const fontSize =
        clamp(
          req.body?.fontSize,
          16,
          72,
          25
        );

      const outline =
        clamp(
          req.body?.outline,
          0,
          10,
          2
        );

      const textColor =
        String(
          req.body?.textColor ||
          "FFFFFF"
        )
          .replace(
            /[^A-Fa-f0-9]/g,
            ""
          )
          .slice(0, 6) ||
          "FFFFFF";

      const position =
        String(
          req.body?.position ||
          "bottom"
        );

      const outputSize =
        String(
          req.body?.outputSize ||
          "original"
        );

      /* =========================
         BLUR
      ========================= */

      const blurOriginal =
        String(
          req.body?.blurOriginal ||
          "false"
        ) === "true";

      /*
         Frontend က percentage ဖြင့် ပို့သည်။
         Default:
         X  = 5%
         Y  = 78%
         W  = 90%
         H  = 22%
      */

      const blurX =
        clamp(
          req.body?.blurX,
          0,
          100,
          5
        );

      const blurY =
        clamp(
          req.body?.blurY,
          0,
          100,
          78
        );

      const blurWidth =
        clamp(
          req.body?.blurWidth,
          1,
          100,
          90
        );

      const blurHeight =
        clamp(
          req.body?.blurHeight,
          1,
          100,
          22
        );

      /* =========================
         VOICE
      ========================= */

      const geminiKey =
        getGeminiKey(req);

      if (!geminiKey) {
        return res.status(400).json({
          error:
            "Gemini API Key ထည့်ပေးပါ"
        });
      }

      const voice =
        String(
          req.body?.voice ||
          "thiha"
        );

      const voiceName =
        MOVIE_VOICES[voice] ||
        MOVIE_VOICES.thiha;

      const voiceSpeed =
        clamp(
          req.body?.voiceSpeed,
          0.5,
          2,
          1
        );

      const voiceVolume =
        clamp(
          req.body?.voiceVolume,
          0,
          2,
          1
        );

      const voicePitch =
        clamp(
          req.body?.voicePitch,
          -12,
          12,
          0
        );

      /* =========================
         GENERATE VOICE
      ========================= */

      console.log(
        "Generating Movie Recap voice for final render:",
        voice,
        voiceName
      );

      const tts =
        await generateGeminiTTS(
          script,
          geminiKey,
          voiceName,
          voiceSpeed,
          voicePitch
        );

      const jobId =
        crypto.randomUUID();

      audioPath =
        path.join(
          uploadDir,
          `${jobId}.wav`
        );

      fs.writeFileSync(
        audioPath,
        tts.audio
      );

      /* =========================
         SCRIPT TIMING
      ========================= */

      const scriptLines =
        script
          .split(/\n+/)
          .map(
            line =>
              line.trim()
          )
          .filter(Boolean);

      const audioSeconds =
        await getMediaDuration(
          audioPath
        );

      const lineDuration =
        scriptLines.length
          ? audioSeconds /
            scriptLines.length
          : audioSeconds;

      const recapTranscript =
        scriptLines.map(
          (text, index) => ({
            start:
              index *
              lineDuration,

            end:
              index ===
              scriptLines.length - 1
                ? audioSeconds
                : (index + 1) *
                  lineDuration,

            text
          })
        );

      /* =========================
         ASS SUBTITLE FILE
         FIXES PRIMARYCOLOUR ERROR
      ========================= */

      const assText =
        makeAss(
          recapTranscript,
          {
            fontName,
            fontSize,
            outline,
            textColor,
            position
          }
        );

      assPath =
        path.join(
          uploadDir,
          `${jobId}.ass`
        );

      fs.writeFileSync(
        assPath,
        assText,
        "utf8"
      );

      console.log(
        "ASS subtitle created:",
        assPath
      );

      /* =========================
         OUTPUT
      ========================= */

      outputPath =
        path.join(
          outputDir,
          `${jobId}.mp4`
        );

      /* =========================
         VIDEO FILTER
      ========================= */

      const assFilterPath =
        escapeAssFilterPath(
          assPath
        );

      const sizeFilter =
        getOutputFilter(
          outputSize
        );

      let videoFilter = "";

      /*
         1.
         Output size ကို အရင်ပြောင်းမယ်။
      */

      if (sizeFilter) {

        videoFilter =
          `[0:v]${sizeFilter}[sized]`;

      } else {

        videoFilter =
          `[0:v]null[sized]`;
      }

      /*
         2.
         Blur area ကို output video
         ပေါ်မှာ percentage ဖြင့်
         သတ်မှတ်မယ်။
      */

      if (blurOriginal) {

        videoFilter +=
          `;[sized]split=2[base][blur];`;

        videoFilter +=
          `[blur]` +
          `crop=` +
          `w=iw*${blurWidth / 100}:` +
          `h=ih*${blurHeight / 100}:` +
          `x=iw*${blurX / 100}:` +
          `y=ih*${blurY / 100},` +
          `boxblur=20:10[blurred];`;

        videoFilter +=
          `[base][blurred]` +
          `overlay=` +
          `x=W*${blurX / 100}:` +
          `y=H*${blurY / 100}` +
          `[blurvideo];`;

        /*
           3.
           ASS subtitles
        */

        videoFilter +=
          `[blurvideo]` +
          `ass='${assFilterPath}'` +
          `[vout]`;

      } else {

        /*
           No blur
        */

        videoFilter +=
          `;[sized]` +
          `ass='${assFilterPath}'` +
          `[vout]`;
      }

      /* =========================
         AUDIO FILTER
      ========================= */

      const audioFilters = [];

      if (voiceVolume !== 1) {

        audioFilters.push(
          `volume=${voiceVolume}`
        );
      }

      /*
         Pitch control:
         FFmpeg မှာ rubberband မရှိနိုင်တာကြောင့်
         basic pitch adjustment ကို
         asetrate + aresample + atempo ဖြင့်လုပ်သည်။
      */

      if (voicePitch !== 0) {

        const factor =
          Math.pow(
            2,
            voicePitch / 12
          );

        const safeFactor =
          Math.max(
            0.5,
            Math.min(
              2,
              factor
            )
          );

        const atempo =
          1 / safeFactor;

        audioFilters.push(
          `asetrate=24000*${safeFactor}`,
          `aresample=24000`,
          `atempo=${atempo}`
        );
      }

      /*
         Speed
      */

      if (voiceSpeed !== 1) {

        let remaining =
          voiceSpeed;

        /*
           atempo supports
           approximately 0.5 - 2.0
        */

        while (remaining > 2) {

          audioFilters.push(
            "atempo=2"
          );

          remaining /= 2;
        }

        while (remaining < 0.5) {

          audioFilters.push(
            "atempo=0.5"
          );

          remaining /= 0.5;
        }

        audioFilters.push(
          `atempo=${remaining}`
        );
      }

      /* =========================
         FFMPEG ARGS
      ========================= */

      const args = [

        "-y",

        "-i",
        inputPath,

        "-i",
        audioPath,

        "-filter_complex",
        videoFilter,

        "-map",
        "[vout]",

        "-map",
        "1:a:0",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "23",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "aac",

        "-b:a",
        "128k"
      ];

      if (audioFilters.length) {

        args.push(
          "-af",
          audioFilters.join(",")
        );
      }

      args.push(
        "-shortest",

        "-movflags",
        "+faststart",

        outputPath
      );

      console.log(
        "Starting Movie Recap FFmpeg..."
      );

      console.log(
        "FFmpeg filter:",
        videoFilter
      );

      console.log(
        "FFmpeg args:",
        args.join(" ")
      );

      await runFfmpeg(
        args
      );

      if (
        !fs.existsSync(
          outputPath
        )
      ) {
        throw new Error(
          "Final MP4 မထွက်လာပါ"
        );
      }

      /* =========================
         CLEAN TEMP FILES
      ========================= */

      for (
        const file of [
          inputPath,
          audioPath,
          assPath
        ]
      ) {

        try {

          if (
            file &&
            fs.existsSync(file)
          ) {
            fs.unlinkSync(file);
          }

        } catch {}
      }

      inputPath = null;
      audioPath = null;
      assPath = null;

      /* =========================
         AUTO DELETE OUTPUT
      ========================= */

      setTimeout(
        () => {

          try {

            if (
              fs.existsSync(
                outputPath
              )
            ) {
              fs.unlinkSync(
                outputPath
              );
            }

          } catch {}

        },
        10 * 60 * 1000
      );

      return res.json({

        ok: true,

        message:
          "Movie Recap Final MP4 အောင်မြင်ပါပြီ",

        downloadUrl:
          `/output/${path.basename(
            outputPath
          )}`,

        outputSize,

        voice,

        voiceSpeed,

        voiceVolume,

        voicePitch,

        fontName,

        fontSize,

        outline,

        textColor,

        position,

        blurOriginal,

        blurX,

        blurY,

        blurWidth,

        blurHeight
      });

    } catch (error) {

      console.error(
        "Movie render error:",
        error.message
      );

      for (
        const file of [
          inputPath,
          audioPath,
          assPath,
          outputPath
        ]
      ) {

        try {

          if (
            file &&
            fs.existsSync(file)
          ) {
            fs.unlinkSync(file);
          }

        } catch {}
      }

      return res.status(500).json({
        error:
          error.message ||
          "Movie Recap render မအောင်မြင်ပါ"
      });
    }
  }
);

/* =========================================================
   MEDIA DURATION
========================================================= */

function getMediaDuration(
  filePath
) {

  return new Promise(
    (resolve, reject) => {

      const ffprobe =
        spawn(
          "ffprobe",
          [
            "-v",
            "error",

            "-show_entries",
            "format=duration",

            "-of",
            "default=noprint_wrappers=1:nokey=1",

            filePath
          ]
        );

      let output = "";
      let errorOutput = "";

      ffprobe.stdout.on(
        "data",
        data => {
          output +=
            data.toString();
        }
      );

      ffprobe.stderr.on(
        "data",
        data => {
          errorOutput +=
            data.toString();
        }
      );

      ffprobe.on(
        "error",
        error => {
          reject(error);
        }
      );

      ffprobe.on(
        "close",
        code => {

          if (code !== 0) {

            reject(
              new Error(
                errorOutput ||
                "ffprobe failed"
              )
            );

            return;
          }

          const duration =
            Number(
              output.trim()
            );

          if (
            !Number.isFinite(
              duration
            )
          ) {

            reject(
              new Error(
                "Media duration မဖတ်နိုင်ပါ"
              )
            );

            return;
          }

          resolve(
            duration
          );
        }
      );
    }
  );
}

/* =========================================================
   OLD SRT RENDER
========================================================= */

app.post(
  "/api/render",
  upload.single("video"),
  async (req, res) => {

    let inputPath = null;
    let srtPath = null;
    let assPath = null;
    let outputPath = null;

    try {

      if (!req.file) {
        return res.status(400).json({
          error:
            "Render လုပ်ရန် video file မတွေ့ပါ"
        });
      }

      inputPath =
        req.file.path;

      let transcript;

      try {

        transcript =
          JSON.parse(
            req.body?.transcript ||
            "[]"
          );

      } catch {

        return res.status(400).json({
          error:
            "Subtitle data မမှန်ပါ"
        });
      }

      if (
        !Array.isArray(transcript) ||
        transcript.length === 0
      ) {
        return res.status(400).json({
          error:
            "Myanmar subtitle မတွေ့ပါ"
        });
      }

      const jobId =
        crypto.randomUUID();

      srtPath =
        path.join(
          uploadDir,
          `${jobId}.srt`
        );

      assPath =
        path.join(
          uploadDir,
          `${jobId}.ass`
        );

      outputPath =
        path.join(
          outputDir,
          `${jobId}.mp4`
        );

      fs.writeFileSync(
        srtPath,
        makeSrt(transcript),
        "utf8"
      );

      const fontName =
        String(
          req.body?.fontName ||
          "Noto Sans Myanmar"
        );

      const fontSize =
        clamp(
          req.body?.fontSize,
          16,
          72,
          25
        );

      const textColor =
        String(
          req.body?.textColor ||
          "FFFFFF"
        )
          .replace(
            /[^A-Fa-f0-9]/g,
            ""
          )
          .slice(0, 6) ||
          "FFFFFF";

      const outline =
        clamp(
          req.body?.outline,
          0,
          10,
          2
        );

      const position =
        String(
          req.body?.position ||
          "bottom"
        );

      fs.writeFileSync(
        assPath,
        makeAss(
          transcript,
          {
            fontName,
            fontSize,
            outline,
            textColor,
            position
          }
        ),
        "utf8"
      );

      const assFilterPath =
        escapeAssFilterPath(
          assPath
        );

      const blurOriginal =
        String(
          req.body?.blurOriginal ||
          "false"
        ) === "true";

      const blurX =
        clamp(
          req.body?.blurX,
          0,
          100,
          5
        );

      const blurY =
        clamp(
          req.body?.blurY,
          0,
          100,
          78
        );

      const blurWidth =
        clamp(
          req.body?.blurWidth,
          1,
          100,
          90
        );

      const blurHeight =
        clamp(
          req.body?.blurHeight,
          1,
          100,
          22
        );

      let videoFilter;

      if (blurOriginal) {

        videoFilter =
          `[0:v]split=2[base][blur];` +

          `[blur]` +
          `crop=` +
          `w=iw*${blurWidth / 100}:` +
          `h=ih*${blurHeight / 100}:` +
          `x=iw*${blurX / 100}:` +
          `y=ih*${blurY / 100},` +
          `boxblur=20:10[blurred];` +

          `[base][blurred]` +
          `overlay=` +
          `x=W*${blurX / 100}:` +
          `y=H*${blurY / 100}` +
          `[blurvideo];` +

          `[blurvideo]` +
          `ass='${assFilterPath}'` +
          `[vout]`;

      } else {

        videoFilter =
          `[0:v]` +
          `ass='${assFilterPath}'` +
          `[vout]`;
      }

      const args = [

        "-y",

        "-i",
        inputPath,

        "-filter_complex",
        videoFilter,

        "-map",
        "[vout]",

        "-map",
        "0:a?",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "23",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "aac",

        "-b:a",
        "128k",

        "-movflags",
        "+faststart",

        outputPath
      ];

      await runFfmpeg(args);

      if (
        !fs.existsSync(
          outputPath
        )
      ) {
        throw new Error(
          "Rendered MP4 မထွက်လာပါ"
        );
      }

      for (
        const file of [
          inputPath,
          srtPath,
          assPath
        ]
      ) {

        try {

          if (
            file &&
            fs.existsSync(file)
          ) {
            fs.unlinkSync(file);
          }

        } catch {}
      }

      inputPath = null;
      srtPath = null;
      assPath = null;

      setTimeout(
        () => {

          try {

            if (
              fs.existsSync(
                outputPath
              )
            ) {
              fs.unlinkSync(
                outputPath
              );
            }

          } catch {}
        },
        10 * 60 * 1000
      );

      return res.json({

        ok: true,

        message:
          "MP4 render အောင်မြင်ပါပြီ",

        downloadUrl:
          `/output/${path.basename(
            outputPath
          )}`
      });

    } catch (error) {

      console.error(
        "Render error:",
        error.message
      );

      for (
        const file of [
          inputPath,
          srtPath,
          assPath,
          outputPath
        ]
      ) {

        try {

          if (
            file &&
            fs.existsSync(file)
          ) {
            fs.unlinkSync(file);
          }

        } catch {}
      }

      return res.status(500).json({
        error:
          error.message ||
          "MP4 render မအောင်မြင်ပါ"
      });
    }
  }
);

/* =========================================================
   SERVER
========================================================= */

const PORT =
  process.env.PORT || 3000;

app.listen(
  PORT,
  "0.0.0.0",
  () => {

    console.log(
      `Myanmar SRT server running on port ${PORT}`
    );

  }
);
