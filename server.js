const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.static("public"));

const upload = multer({
  dest: "uploads/",
  limits: {
    fileSize: 100 * 1024 * 1024
  }
});

const MAX_SIZE = 100 * 1024 * 1024;
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


/* =========================
   GROQ TRANSCRIPT
========================= */

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
          error: "Video size က 100MB ထက်မကျော်ရပါ"
        });
      }

      const groqKey = getGroqKey(req);

      if (!groqKey) {
        return res.status(400).json({
          error: "Groq API Key ထည့်ပေးပါ"
        });
      }

      const videoBuffer =
        fs.readFileSync(filePath);

      const form = new FormData();

      const blob = new Blob(
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
        Array.isArray(data.segments)
          ? data.segments
          : [];

      const transcript =
        segments
          .map((segment) => ({
            start:
              Number(segment.start || 0),

            end:
              Number(segment.end || 0),

            text:
              String(
                segment.text || ""
              ).trim()
          }))
          .filter(
            item => item.text
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


/* =========================
   GEMINI REQUEST HELPER
========================= */

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
    data = JSON.parse(raw);
  } catch {

    return {
      ok: false,

      status:
        response.status,

      error:
        "Gemini က JSON response မပြန်ပါ",

      raw
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
      JSON.parse(responseText);

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


/* =========================
   GEMINI MYANMAR TRANSLATION
========================= */

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


      /* =========================
         BUILD SOURCE
      ========================= */

      const source =
        transcript
          .map((item, index) => {

            return (
              `${index + 1}. ` +
              `[${item.start} --> ${item.end}] ` +
              `${item.text}`
            );

          })
          .join("\n");


      /* =========================
         TRANSLATION PROMPT
      ========================= */

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

The "id" must match the original subtitle number exactly.

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
         GEMINI MODEL FALLBACK
      ========================= */

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


      /* =========================
         ALL MODELS FAILED
      ========================= */

      if (!finalResult) {

        return res.status(502).json({
          error:
            lastError ||
            "Gemini translation မအောင်မြင်ပါ"
        });
      }


      /* =========================
         GET TRANSLATED SEGMENTS
      ========================= */

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


      /* =========================
         MATCH TIMESTAMPS
      ========================= */

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
                Number(
                  original.start
                ),

              end:
                Number(
                  original.end
                ),

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


      /* =========================
         RETURN RESULT
      ========================= */

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


/* =========================
   SERVER
========================= */

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
