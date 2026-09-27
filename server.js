const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");

const app = express();

app.use(cors());
app.use(express.json());
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

    const transcript = segments.map((segment) => ({
      start: Number(segment.start || 0),
      end: Number(segment.end || 0),
      text: String(segment.text || "").trim()
    })).filter(item => item.text);

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

const PORT =
  process.env.PORT || 3000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(
    `Myanmar SRT server running on port ${PORT}`
  );
});
