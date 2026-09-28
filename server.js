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

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    message: "Myanmar SRT backend is running"
  });
});


/* =========================
   GROQ TRANSCRIPTION
========================= */

app.post("/api/transcribe", upload.single("video"), async (req, res) => {
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

    if (!process.env.GROQ_API_KEY) {
      return res.status(500).json({
        error: "GROQ_API_KEY မသတ်မှတ်ရသေးပါ"
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
          Authorization:
            `Bearer ${process.env.GROQ_API_KEY}`
        },
        body: form
      }
    );

    const data = await groqResponse.json();

    if (!groqResponse.ok) {
      console.error("Groq error:", data);

      return res.status(502).json({
        error:
          data?.error?.message ||
          "Groq transcription မအောင်မြင်ပါ"
      });
    }

    const segments = Array.isArray(data.segments)
      ? data.segments
      : [];

    const transcript = segments
      .map((segment) => ({
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

    const durationMinutes =
      lastEnd / 60;

    if (durationMinutes > MAX_MINUTES + 0.25) {
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
    console.error(error);

    return res.status(500).json({
      error: error.message
    });

  } finally {
    if (filePath) {
      try {
        fs.unlinkSync(filePath);
      } catch {}
    }
  }
});


/* =========================
   GEMINI MYANMAR TRANSLATION
========================= */

app.post("/api/translate", async (req, res) => {

  try {

    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        error: "GEMINI_API_KEY မသတ်မှတ်ရသေးပါ"
      });
    }

    const transcript = req.body?.transcript;

    if (!Array.isArray(transcript) || !transcript.length) {
      return res.status(400).json({
        error: "Transcript မတွေ့ပါ"
      });
    }

    const cleanTranscript = transcript.map((item, index) => ({
      id: index + 1,
      start: Number(item.start || 0),
      end: Number(item.end || 0),
      text: String(item.text || "")
    }));


    const prompt = `
You are a professional subtitle translator.

Translate the following subtitle transcript into natural, fluent Myanmar (Burmese).

IMPORTANT RULES:

1. Translate ONLY the dialogue text.
2. Keep every subtitle ID exactly the same.
3. Keep start and end timestamps exactly the same.
4. Do NOT merge subtitles.
5. Do NOT split subtitles.
6. Do NOT add explanations.
7. Do NOT add English text.
8. Preserve names and proper nouns naturally.
9. Make the Myanmar translation suitable for movie/drama subtitles.
10. Return ONLY valid JSON.

Input:
${JSON.stringify(cleanTranscript, null, 2)}

Return JSON in this exact structure:

{
  "segments": [
    {
      "id": 1,
      "start": 0,
      "end": 3,
      "text": "မြန်မာဘာသာပြန်"
    }
  ]
}
`;


    const geminiResponse = await fetch(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent?key=" +
      encodeURIComponent(process.env.GEMINI_API_KEY),
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json"
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
            temperature: 0.2,
            responseMimeType: "application/json",

            responseSchema: {
              type: "object",

              properties: {
                segments: {
                  type: "array",

                  items: {
                    type: "object",

                    properties: {
                      id: {
                        type: "integer"
                      },

                      start: {
                        type: "number"
                      },

                      end: {
                        type: "number"
                      },

                      text: {
                        type: "string"
                      }
                    },

                    required: [
                      "id",
                      "start",
                      "end",
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


    const geminiData =
      await geminiResponse.json();


    if (!geminiResponse.ok) {

      console.error(
        "Gemini error:",
        geminiData
      );

      return res.status(502).json({
        error:
          geminiData?.error?.message ||
          "Gemini translation မအောင်မြင်ပါ"
      });
    }


    const outputText =
      geminiData
        ?.candidates?.[0]
        ?.content?.parts?.[0]
        ?.text;


    if (!outputText) {
      return res.status(502).json({
        error:
          "Gemini က ဘာသာပြန်စာ မပြန်ပေးပါ"
      });
    }


    let translated;

    try {

      translated =
        JSON.parse(outputText);

    } catch (error) {

      console.error(
        "Gemini JSON parse error:",
        outputText
      );

      return res.status(502).json({
        error:
          "Gemini response ကို JSON အဖြစ်ဖတ်မရပါ"
      });
    }


    if (
      !Array.isArray(
        translated.segments
      )
    ) {
      return res.status(502).json({
        error:
          "Gemini translation format မမှန်ပါ"
      });
    }


    const result =
      translated.segments.map(
        (item, index) => {

          const original =
            cleanTranscript[index];

          return {
            id:
              original?.id ??
              item.id ??
              index + 1,

            start:
              original?.start ??
              item.start ??
              0,

            end:
              original?.end ??
              item.end ??
              0,

            text:
              String(
                item.text || ""
              ).trim()
          };
        }
      );


    return res.json({
      ok: true,
      segments: result
    });


  } catch (error) {

    console.error(error);

    return res.status(500).json({
      error: error.message
    });
  }
});


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
