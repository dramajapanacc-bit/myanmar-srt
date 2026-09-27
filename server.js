const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

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

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    message: "Myanmar SRT backend is running"
  });
});

app.post("/api/transcribe", upload.single("video"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "Video file မတွေ့ပါ"
      });
    }

    res.json({
      ok: true,
      message: "Video ရရှိပါပြီ",
      filename: req.file.originalname,
      size: req.file.size
    });

  } catch (error) {
    console.error(error);

    res.status(500).json({
      error: error.message
    });
  }
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Myanmar SRT server running on port ${PORT}`);
});
