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
   HEALTH CHECK
========================= */

app.get("/api/health", (req, res) => {

  res.json({
    ok: true,
    message: "Myanmar SRT backend is running"
  });

});


/* =========================
   GROQ TRANSCRIPTION
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


      if (!process.env.GROQ_API_KEY) {

        return res.status(500).json({
          error: "GROQ_API_KEY မသတ်မှတ်ရသေးပါ"
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
                `Bearer ${process.env.GROQ_API_KEY}`
            },

            body: form
          }
        );


      const data =
        await groqResponse.json();


      if (!groqResponse.ok) {

        console.error(
          "Groq error:",
          data
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
          error: "Groq က transcript မရပါ"
        });

      }


      const lastEnd =
        transcript[
          transcript.length - 1
        ].end;


      const durationMinutes =
        lastEnd / 60;


      if (
        durationMinutes >
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
        error
      );


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

  }
);


/* =========================
   GEMINI MYANMAR TRANSLATION
========================= */

app.post(
  "/api/translate",
  async (req, res) => {

    try {

      if (!process.env.GEMINI_API_KEY) {

        return res.status(500).json({
          error:
            "GEMINI_API_KEY မသတ်မှတ်ရသေးပါ"
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


      const inputText =
        transcript
          .map(
            (item, index) =>
              `${index + 1}. [${item.start} --> ${item.end}] ${item.text}`
          )
          .join("\n");


      const prompt = `
You are a professional Myanmar subtitle translator.

Translate the following movie/drama transcript into natural, fluent Myanmar language.

IMPORTANT RULES:

1. Translate every subtitle.
2. Keep the same subtitle order.
3. Keep the original start and end timestamps.
4. Do not add explanations.
5. Do not add English translation.
6. Do not remove subtitle lines.
7. Preserve names and proper nouns naturally.
8. Make the Myanmar language sound natural for movie/drama subtitles.
9. Return JSON only.

SOURCE TRANSCRIPT:

${inputText}
`;


      const geminiUrl =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent?key=" +
        encodeURIComponent(
          process.env.GEMINI_API_KEY
        );


      const geminiResponse =
        await fetch(
          geminiUrl,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
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


      const rawText =
        await geminiResponse.text();


      let geminiData;


      try {

        geminiData =
          JSON.parse(rawText);

      } catch {

        console.error(
          "Gemini raw response:",
          rawText
        );

        return res.status(502).json({
          error:
            "Gemini က JSON response မပြန်နိုင်ပါ"
        });

      }


      if (!geminiResponse.ok) {

        console.error(
          "Gemini API error:",
          geminiData
        );


        return res.status(
          geminiResponse.status
        ).json({

          error:
            geminiData?.error?.message ||
            "Gemini translation မအောင်မြင်ပါ"

        });

      }


      const candidate =
        geminiData?.candidates?.[0];


      const responseText =
        candidate?.content?.parts?.[0]?.text;


      if (!responseText) {

        return res.status(502).json({
          error:
            "Gemini response မရပါ"
        });

      }


      let translatedData;


      try {

        translatedData =
          JSON.parse(responseText);

      } catch {

        console.error(
          "Gemini text:",
          responseText
        );

        return res.status(502).json({
          error:
            "Gemini translation JSON မမှန်ပါ"
        });

      }


      const translatedSegments =
        Array.isArray(
          translatedData.segments
        )
          ? translatedData.segments
          : [];


      if (!translatedSegments.length) {

        return res.status(502).json({
          error:
            "Gemini translation မရပါ"
        });

      }


      /*
        Gemini က timestamp မပြန်ရင်
        original transcript ထဲက timestamp
        ကို ပြန်ယူမယ်။
      */

      const result =
        translatedSegments
          .map((item, index) => {

            const original =
              transcript[
                Number(item.id) - 1
              ] ||
              transcript[index];


            if (!original) {
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

              text:
                String(
                  item.text || ""
                ).trim()

            };

          })
          .filter(
            item =>
              item &&
              item.text
          );


      if (!result.length) {

        return res.status(502).json({
          error:
            "Myanmar subtitle မထွက်ပါ"
        });

      }


      return res.json({

        ok: true,

        transcript: result

      });


    } catch (error) {

      console.error(
        "Translation error:",
        error
      );


      return res.status(500).json({

        error:
          error.message ||
          "Gemini translation error"

      });

    }

  }
);


/* =========================
   START SERVER
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
