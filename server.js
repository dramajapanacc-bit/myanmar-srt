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
    fileSize: 100 * 1024 * 1024
  }
});


/* =========================
   LIMITS
========================= */

const MAX_SIZE =
  100 * 1024 * 1024;

const MAX_MINUTES = 5;


/* =========================
   API KEY HELPER
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
          error:
            "Video size က 100MB ထက်မကျော်ရပါ"
        });

      }


      const groqKey =
        getGroqKey(req);


      if (!groqKey) {

        return res.status(400).json({
          error:
            "Groq API Key ထည့်ပေးပါ"
        });

      }


      const videoBuffer =
        fs.readFileSync(filePath);


      const form =
        new FormData();


      const blob =
        new Blob(
          [videoBuffer],
          {
            type:
              req.file.mimetype ||
              "video/mp4"
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


      const groqResponse =
        await fetch(
          "https://api.groq.com/openai/v1/audio/transcriptions",
          {
            method: "POST",

            headers: {
              Authorization:
                `Bearer ${groqKey}`
            },

            body: form
          }
        );


      const data =
        await groqResponse.json();


      if (!groqResponse.ok) {

        console.error(
          "Groq request failed:",
          data?.error?.message ||
          "Unknown error"
        );


        return res.status(502).json({
          error:
            data?.error?.message ||
            "Groq transcription မအောင်မြင်ပါ"
        });

      }


      const segments =
        Array.isArray(
          data.segments
        )
          ? data.segments
          : [];


      const transcript =
        segments
          .map(segment => ({

            start:
              Number(
                segment.start || 0
              ),

            end:
              Number(
                segment.end || 0
              ),

            text:
              String(
                segment.text || ""
              ).trim()

          }))
          .filter(
            item =>
              item.text
          );


      if (!transcript.length) {

        return res.status(502).json({
          error:
            "Groq က transcript မရပါ"
        });

      }


      const lastEnd =
        transcript[
          transcript.length - 1
        ].end;


      if (
        lastEnd / 60 >
        MAX_MINUTES + 0.25
      ) {

        return res.status(400).json({
          error:
            "Video က 5 မိနစ်ထက်ရှည်နေပါတယ်"
        });

      }


      return res.json({

        ok: true,

        durationSeconds:
          lastEnd,

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
   GEMINI HELPER
========================================================= */

async function callGemini(
  model,
  geminiKey,
  prompt
) {

  console.log(
    `Trying Gemini model: ${model}`
  );


  const response =
    await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
      {

        method: "POST",

        headers: {

          "Content-Type":
            "application/json",

          "x-goog-api-key":
            geminiKey

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
              "application/json",

            responseSchema: {

              type: "OBJECT",

              properties: {

                segments: {

                  type: "ARRAY",

                  items: {

                    type: "OBJECT",

                    properties: {

                      id: {
                        type: "INTEGER"
                      },

                      text: {
                        type: "STRING"
                      }

                    },

                    required: [
                      "id",
                      "text"
                    ]

                  }

                }

              },

              required: [
                "segments"
              ]

            }

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

  } catch {

    return {

      ok: false,

      status:
        response.status,

      error:
        "Gemini က JSON response မပြန်ပါ"

    };

  }


  if (!response.ok) {

    return {

      ok: false,

      status:
        response.status,

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

      error:
        "Gemini response မရပါ"

    };

  }


  let translated;


  try {

    translated =
      JSON.parse(
        responseText
      );

  } catch {

    return {

      ok: false,

      status: 502,

      error:
        "Gemini translation JSON မမှန်ပါ"

    };

  }


  return {

    ok: true,

    data: translated

  };

}


/* =========================================================
   GEMINI MYANMAR TRANSLATION
========================================================= */

app.post(
  "/api/translate",
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


      const transcript =
        req.body?.transcript;


      if (
        !Array.isArray(transcript) ||
        transcript.length === 0
      ) {

        return res.status(400).json({
          error:
            "Transcript မတွေ့ပါ"
        });

      }


      const source =
        transcript
          .map(
            (item, index) => {

              return (
                `${index + 1}. ` +
                `[${item.start} --> ${item.end}] ` +
                `${item.text}`
              );

            }
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
- Preserve the meaning and emotion of the original dialogue.
- Keep each subtitle reasonably short and readable.
- Return JSON only.
- The "id" must match the original subtitle number exactly.

Required JSON format:

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


      /* =========================
         GEMINI FALLBACK MODELS
      ========================= */

      const models = [

        "gemini-3.5-flash",

        "gemini-3.1-flash-lite"

      ];


      let finalResult = null;

      let lastError = null;


      for (
        const model of models
      ) {

        try {

          const result =
            await callGemini(
              model,
              geminiKey,
              prompt
            );


          if (result.ok) {

            finalResult =
              result.data;

            console.log(
              `Gemini success: ${model}`
            );

            break;

          }


          lastError =
            result.error;


          console.error(
            `Gemini ${model} failed:`,
            result.error
          );


        } catch (error) {

          lastError =
            error.message;


          console.error(
            `Gemini ${model} exception:`,
            error.message
          );

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
        Array.isArray(
          finalResult?.segments
        )
          ? finalResult.segments
          : [];


      if (
        translatedSegments.length === 0
      ) {

        return res.status(502).json({
          error:
            "Gemini က translation lines မပြန်ပါ"
        });

      }


      const result =
        transcript
          .map(
            (original, index) => {

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
                  Number(
                    original.start
                  ),

                end:
                  Number(
                    original.end
                  ),

                text

              };

            }
          )
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
      (item, index) => {

        return (
          `${index + 1}\n` +
          `${formatSrtTime(item.start)} --> ` +
          `${formatSrtTime(item.end)}\n` +
          `${String(item.text || "").trim()}\n`
        );

      }
    )
    .join("\n");

}


/* =========================================================
   FFMPEG RENDER
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
                stderr.slice(-4000) ||
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
   RENDER MP4
========================================================= */

app.post(
  "/api/render",
  upload.single("video"),
  async (req, res) => {

    let inputPath = null;
    let srtPath = null;
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


      if (req.file.size > MAX_SIZE) {

        return res.status(400).json({
          error:
            "Video size က 100MB ထက်မကျော်ရပါ"
        });

      }


      /* =========================
         RECEIVE SRT
      ========================= */

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


      const srtText =
        makeSrt(
          transcript
        );


      /* =========================
         TEMP SRT
      ========================= */

      const jobId =
        crypto.randomUUID();


      srtPath =
        path.join(
          uploadDir,
          `${jobId}.srt`
        );


      outputPath =
        path.join(
          outputDir,
          `${jobId}.mp4`
        );


      fs.writeFileSync(
        srtPath,
        srtText,
        "utf8"
      );


      /* =========================
         STYLE SETTINGS
      ========================= */

      const fontSize =
        Math.min(
          72,
          Math.max(
            16,
            Number(
              req.body?.fontSize ||
              28
            )
          )
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
        Math.min(
          10,
          Math.max(
            0,
            Number(
              req.body?.outline ||
              2
            )
          )
        );


      const position =
        String(
          req.body?.position ||
          "bottom"
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


      /* =========================
         SRT FILTER
      ========================= */

      const subtitlePath =
        srtPath
          .replace(
            /\\/g,
            "/"
          )
          .replace(
            /:/g,
            "\\:"
          )
          .replace(
            /'/g,
            "\\'"
          );


      const subtitleFilter =
        `subtitles='${subtitlePath}':` +
        `force_style=` +
        `FontSize=${fontSize},` +
        `PrimaryColour=&H00${textColor},` +
        `OutlineColour=&H00000000,` +
        `Outline=${outline},` +
        `Shadow=1,` +
        `Alignment=${alignment},` +
        `MarginV=35`;


      /* =========================
         BLUR ORIGINAL SUBTITLE
      ========================= */

      const blurOriginal =
        String(
          req.body?.blurOriginal ||
          "false"
        ) === "true";


      let videoFilter;


      if (blurOriginal) {

        /*
          Blur the lower 22% of the original video.
          This is a practical subtitle-area blur.
        */

        videoFilter =
          `[0:v]split=2[base][blur];` +
          `[blur]crop=w=iw:h=ih*0.22:y=ih*0.78,` +
          `boxblur=10:1[blurred];` +
          `[base][blurred]overlay=0:H-h[tmp];` +
          `[tmp]${subtitleFilter}`;

      } else {

        videoFilter =
          subtitleFilter;

      }


      /* =========================
         FFMPEG
      ========================= */

      const args = [

        "-y",

        "-i",
        inputPath,

        "-vf",
        videoFilter,

        "-map",
        "0:v:0",

        "-map",
        "0:a?",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "23",

        "-c:a",
        "aac",

        "-b:a",
        "128k",

        "-movflags",
        "+faststart",

        outputPath

      ];


      console.log(
        "Starting FFmpeg render..."
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
          "Rendered MP4 မထွက်လာပါ"
        );

      }


      /* =========================
         CLEAN TEMP FILES
      ========================= */

      try {
        fs.unlinkSync(
          inputPath
        );
      } catch {}


      try {
        fs.unlinkSync(
          srtPath
        );
      } catch {}


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


      /* =========================
         RESPONSE
      ========================= */

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


      if (inputPath) {

        try {
          fs.unlinkSync(
            inputPath
          );
        } catch {}

      }


      if (srtPath) {

        try {
          fs.unlinkSync(
            srtPath
          );
        } catch {}

      }


      if (outputPath) {

        try {
          fs.unlinkSync(
            outputPath
          );
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
