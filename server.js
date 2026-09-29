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

const WORK_DIR = path.join(
  os.tmpdir(),
  "myanmar-srt-work"
);

const PUBLIC_DIR = path.join(
  __dirname,
  "public"
);

fs.mkdirSync(WORK_DIR, {
  recursive: true
});

fs.mkdirSync(PUBLIC_DIR, {
  recursive: true
});

app.use(cors());

app.use(
  express.json({
    limit: "10mb"
  })
);

app.use(
  express.urlencoded({
    extended: true,
    limit: "10mb"
  })
);

app.use(
  express.static(PUBLIC_DIR)
);


/* =========================================================
   UPLOAD
========================================================= */

const upload = multer({
  dest: WORK_DIR,

  limits: {
    fileSize: 300 * 1024 * 1024
  }
});


/* =========================================================
   API KEY
========================================================= */

function getGroqKey(req) {
  return (
    req.headers["x-groq-api-key"] ||
    req.body?.groqApiKey ||
    process.env.GROQ_API_KEY ||
    ""
  )
    .toString()
    .trim();
}


function getGeminiKey(req) {
  return (
    req.headers["x-gemini-api-key"] ||
    req.body?.geminiApiKey ||
    process.env.GEMINI_API_KEY ||
    ""
  )
    .toString()
    .trim();
}


/* =========================================================
   HELPERS
========================================================= */

function safeFileName(name) {
  return String(name || "upload")
    .replace(
      /[^a-zA-Z0-9._-]/g,
      "_"
    )
    .slice(0, 120);
}


function cleanupFile(filePath) {
  try {
    if (
      filePath &&
      fs.existsSync(filePath)
    ) {
      fs.unlinkSync(filePath);
    }
  } catch (_) {}
}


function parseJsonSafely(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    return null;
  }
}


function extractJsonObject(text) {
  const value = String(
    text || ""
  ).trim();

  const direct =
    parseJsonSafely(value);

  if (direct) {
    return direct;
  }

  const fenced = value
    .replace(
      /^```json\s*/i,
      ""
    )
    .replace(
      /^```\s*/i,
      ""
    )
    .replace(
      /```\s*$/i,
      ""
    )
    .trim();

  const fencedJson =
    parseJsonSafely(fenced);

  if (fencedJson) {
    return fencedJson;
  }

  const start =
    fenced.indexOf("{");

  const end =
    fenced.lastIndexOf("}");

  if (
    start !== -1 &&
    end > start
  ) {
    const obj =
      parseJsonSafely(
        fenced.slice(
          start,
          end + 1
        )
      );

    if (obj) {
      return obj;
    }
  }

  return null;
}


/* =========================================================
   COMMAND RUNNER
========================================================= */

function runCommand(
  command,
  args,
  options = {}
) {
  return new Promise(
    (resolve, reject) => {
      const child =
        spawn(
          command,
          args,
          {
            windowsHide: true,
            ...options
          }
        );

      let stdout = "";
      let stderr = "";

      child.stdout?.on(
        "data",
        data => {
          stdout +=
            data.toString();
        }
      );

      child.stderr?.on(
        "data",
        data => {
          stderr +=
            data.toString();
        }
      );

      child.on(
        "error",
        reject
      );

      child.on(
        "close",
        code => {
          if (code === 0) {
            resolve({
              stdout,
              stderr
            });
          } else {
            reject(
              new Error(
                `${command} exited with code ${code}\n${stderr.slice(
                  -4000
                )}`
              )
            );
          }
        }
      );
    }
  );
}


/* =========================================================
   VIDEO DURATION
========================================================= */

async function videoDuration(
  filePath
) {
  const result =
    await runCommand(
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

  const duration =
    Number(
      result.stdout.trim()
    );

  if (
    !Number.isFinite(
      duration
    )
  ) {
    throw new Error(
      "Video duration မဖတ်နိုင်ပါ"
    );
  }

  return duration;
}


/* =========================================================
   TEXT CLEANING
========================================================= */

function cleanTranscriptText(
  text
) {
  let value =
    String(
      text || ""
    ).trim();

  /*
    Remove accidental prompt/instruction
    text sometimes returned by Whisper.
  */

  value =
    value.replace(
      /\bReturn accurate segment\.?/gi,
      ""
    );

  value =
    value.replace(
      /\bReturn the spoken word\.?/gi,
      ""
    );

  value =
    value.replace(
      /\bReturn accurate transcript\.?/gi,
      ""
    );

  value =
    value.replace(
      /\s{2,}/g,
      " "
    );

  return value.trim();
}


/* =========================================================
   WORD NORMALIZATION
========================================================= */

function normalizeWord(
  word
) {
  if (!word) {
    return null;
  }

  const start =
    Number(
      word.start
    );

  const end =
    Number(
      word.end
    );

  const text =
    cleanTranscriptText(
      word.word
    );

  if (
    !text ||
    !Number.isFinite(start) ||
    !Number.isFinite(end)
  ) {
    return null;
  }

  return {
    start,
    end,
    text
  };
}


/* =========================================================
   SENTENCE END CHECK
========================================================= */

function isSentenceEnd(
  text
) {
  const value =
    String(
      text || ""
    ).trim();

  return /[.!?။၊!?]$/.test(
    value
  );
}


/* =========================================================
   SUBTITLE SPLITTER
=========================================================

   Main goal:

   - Not too many words
   - Not too many characters
   - Not too long
   - Prefer natural sentence breaks
   - Keep real Groq word timestamps

========================================================= */

function splitWordsIntoSubtitles(
  words
) {
  const result = [];

  let current = [];

  const MAX_CHARS = 42;

  const MAX_WORDS = 10;

  const MAX_DURATION = 5.0;


  function flush() {
    if (!current.length) {
      return;
    }

    const start =
      current[0].start;

    const end =
      current[
        current.length - 1
      ].end;

    const text =
      current
        .map(
          item =>
            item.text
        )
        .join(" ")
        .replace(
          /\s+/g,
          " "
        )
        .trim();

    if (text) {
      result.push({
        start,
        end,
        text
      });
    }

    current = [];
  }


  for (
    const word of words
  ) {
    if (!word) {
      continue;
    }

    if (!current.length) {
      current.push(word);
      continue;
    }

    const currentText =
      current
        .map(
          item =>
            item.text
        )
        .join(" ");

    const candidateText =
      currentText +
      " " +
      word.text;

    const candidateStart =
      current[0].start;

    const candidateEnd =
      word.end;

    const candidateDuration =
      candidateEnd -
      candidateStart;

    const tooManyChars =
      candidateText.length >
      MAX_CHARS;

    const tooManyWords =
      current.length >=
      MAX_WORDS;

    const tooLong =
      candidateDuration >
      MAX_DURATION;

    /*
      If current line already ends naturally,
      prefer starting a new subtitle.
    */

    const naturalBreak =
      isSentenceEnd(
        currentText
      );


    if (
      tooManyChars ||
      tooManyWords ||
      tooLong ||
      naturalBreak
    ) {
      flush();

      current.push(word);
    } else {
      current.push(word);
    }
  }

  flush();

  return result;
}


/* =========================================================
   FALLBACK SEGMENT SPLITTER
========================================================= */

function splitSegmentByText(
  segment
) {
  const text =
    cleanTranscriptText(
      segment.text
    );

  if (!text) {
    return [];
  }

  const words =
    text.split(
      /\s+/
    );

  if (
    words.length <= 8 &&
    (
      Number(segment.end) -
      Number(segment.start)
    ) <= 5
  ) {
    return [
      {
        start:
          Number(segment.start),

        end:
          Number(segment.end),

        text
      }
    ];
  }

  const duration =
    Math.max(
      0.1,
      Number(segment.end) -
        Number(segment.start)
    );

  const totalChars =
    Math.max(
      1,
      text.length
    );

  const chunks = [];

  let current = [];

  let currentChars = 0;

  for (
    const word of words
  ) {
    const nextChars =
      currentChars === 0
        ? word.length
        : currentChars +
          1 +
          word.length;

    if (
      current.length >= 8 ||
      nextChars > 42
    ) {
      if (current.length) {
        chunks.push(
          current.join(" ")
        );
      }

      current = [
        word
      ];

      currentChars =
        word.length;
    } else {
      current.push(word);

      currentChars =
        nextChars;
    }
  }

  if (current.length) {
    chunks.push(
      current.join(" ")
    );
  }

  let offsetChars = 0;

  return chunks.map(
    chunk => {
      const ratioStart =
        offsetChars /
        totalChars;

      offsetChars +=
        chunk.length;

      const ratioEnd =
        Math.min(
          1,
          offsetChars /
            totalChars
        );

      return {
        start:
          Number(
            segment.start
          ) +
          duration *
            ratioStart,

        end:
          Number(
            segment.start
          ) +
          duration *
            ratioEnd,

        text: chunk
      };
    }
  );
}


/* =========================================================
   BUILD FINAL TRANSCRIPT
========================================================= */

function buildTranscript(
  data,
  duration
) {
  const words = Array.isArray(
    data?.words
  )
    ? data.words
        .map(
          normalizeWord
        )
        .filter(Boolean)
    : [];


  /*
    BEST METHOD:
    Use real word timestamps.
  */

  if (words.length) {
    const subtitles =
      splitWordsIntoSubtitles(
        words
      );

    return subtitles
      .map(
        (item, index) => ({
          id: index + 1,

          start:
            Number(
              item.start.toFixed(3)
            ),

          end:
            Number(
              item.end.toFixed(3)
            ),

          text:
            cleanTranscriptText(
              item.text
            )
        })
      )
      .filter(
        item =>
          item.text &&
          item.end >
            item.start
      );
  }


  /*
    FALLBACK:
    Use Groq segment timestamps.
  */

  const segments =
    Array.isArray(
      data?.segments
    )
      ? data.segments
      : [];


  const result = [];

  for (
    const segment of segments
  ) {
    const text =
      cleanTranscriptText(
        segment.text
      );

    const start =
      Number(
        segment.start
      );

    const end =
      Number(
        segment.end
      );

    if (
      !text ||
      !Number.isFinite(
        start
      ) ||
      !Number.isFinite(
        end
      ) ||
      end <= start
    ) {
      continue;
    }

    const pieces =
      splitSegmentByText({
        start,
        end,
        text
      });

    for (
      const piece of pieces
    ) {
      result.push({
        id:
          result.length + 1,

        start:
          Number(
            piece.start.toFixed(3)
          ),

        end:
          Number(
            piece.end.toFixed(3)
          ),

        text:
          cleanTranscriptText(
            piece.text
          )
      });
    }
  }


  /*
    LAST FALLBACK
  */

  if (
    !result.length &&
    data?.text
  ) {
    const fallbackText =
      cleanTranscriptText(
        data.text
      );

    if (fallbackText) {
      result.push({
        id: 1,
        start: 0,
        end: duration,
        text: fallbackText
      });
    }
  }

  return result;
}


/* =========================================================
   HEALTH
========================================================= */

app.get(
  "/api/health",
  async (req, res) => {
    res.json({
      ok: true,

      service:
        "myanmar-srt",

      srtOnly: true,

      movieRecap: false,

      render: false,

      groqConfigured:
        Boolean(
          process.env.GROQ_API_KEY
        ),

      geminiConfigured:
        Boolean(
          process.env.GEMINI_API_KEY
        ),

      timestamp:
        new Date().toISOString()
    });
  }
);


/* =========================================================
   GROQ TRANSCRIBE
========================================================= */

app.post(
  "/api/transcribe",

  upload.single("video"),

  async (
    req,
    res
  ) => {
    let filePath = null;

    try {
      const key =
        getGroqKey(req);

      if (!key) {
        return res
          .status(400)
          .json({
            error:
              "Groq API Key မရှိပါ"
          });
      }

      if (!req.file) {
        return res
          .status(400)
          .json({
            error:
              "Video file မရှိပါ"
          });
      }

      filePath =
        req.file.path;


      /*
        Check video duration.
      */

      const duration =
        await videoDuration(
          filePath
        );

      if (
        duration > 300
      ) {
        return res
          .status(400)
          .json({
            error:
              "Video က 5 မိနစ်ထက်ကျော်နေပါတယ်"
          });
      }


      /*
        Upload video to Groq.
      */

      const form =
        new FormData();

      const bytes =
        fs.readFileSync(
          filePath
        );

      const blob =
        new Blob(
          [
            bytes
          ],
          {
            type:
              req.file.mimetype ||
              "video/mp4"
          }
        );

      form.append(
        "file",
        blob,
        safeFileName(
          req.file.originalname
        )
      );


      /*
        Whisper model.
      */

      form.append(
        "model",
        "whisper-large-v3-turbo"
      );


      /*
        Verbose JSON is required
        for timestamps.
      */

      form.append(
        "response_format",
        "verbose_json"
      );


      /*
        IMPORTANT:

        Request WORD timestamps.

        This is what lets us split
        long subtitles properly.
      */

      form.append(
        "timestamp_granularities[]",
        "word"
      );

      form.append(
        "timestamp_granularities[]",
        "segment"
      );


      /*
        Do NOT send a prompt containing
        "Return accurate segment".

        That text can sometimes appear
        in the transcription.
      */

      form.append(
        "language",
        "en"
      );


      const response =
        await fetch(
          "https://api.groq.com/openai/v1/audio/transcriptions",
          {
            method:
              "POST",

            headers: {
              Authorization:
                `Bearer ${key}`
            },

            body:
              form
          }
        );


      const rawText =
        await response.text();

      const data =
        parseJsonSafely(
          rawText
        );


      if (!response.ok) {
        throw new Error(
          data?.error?.message ||
          data?.error ||
          rawText ||
          "Groq transcription failed"
        );
      }


      /*
        Build clean subtitle segments.
      */

      const transcript =
        buildTranscript(
          data,
          duration
        );


      if (!transcript.length) {
        throw new Error(
          "Groq က စာသားမတွေ့ပါ။ Video ထဲမှာ အသံရှိ/မရှိ စစ်ပေးပါ"
        );
      }


      /*
        Full transcript text.
      */

      const fullText =
        transcript
          .map(
            item =>
              item.text
          )
          .join(" ");


      console.log(
        `Groq transcription complete: ${transcript.length} subtitle segments`
      );


      res.json({
        ok: true,

        duration,

        language:
          data?.language ||
          "en",

        text:
          data?.text ||
          fullText,

        transcript
      });


    } catch (error) {
      console.error(
        "TRANSCRIBE ERROR:",
        error
      );

      res
        .status(500)
        .json({
          error:
            error.message ||
            "Groq Transcript Error"
        });


    } finally {
      cleanupFile(
        filePath
      );
    }
  }
);


/* =========================================================
   GEMINI TRANSLATION
========================================================= */

app.post(
  "/api/translate",

  async (
    req,
    res
  ) => {

    try {

      const key =
        getGeminiKey(req);


      if (!key) {
        return res
          .status(400)
          .json({
            error:
              "Gemini API Key မရှိပါ"
          });
      }


      const transcript =
        Array.isArray(
          req.body?.transcript
        )
          ? req.body.transcript
          : [];


      if (!transcript.length) {
        return res
          .status(400)
          .json({
            error:
              "Transcript မရှိပါ"
          });
      }


      /*
        Preserve Groq timing.
      */

      const inputSegments =
        transcript.map(
          (item, index) => ({
            id:
              index + 1,

            start:
              Number(
                item.start
              ) || 0,

            end:
              Number(
                item.end
              ) || 0,

            text:
              String(
                item.text ||
                ""
              ).trim()
          })
        );


      /*
        Gemini should ONLY translate.

        It must NOT create timing.

        It must NOT merge.

        It must NOT split.

      */

      const prompt = `
You are a professional Myanmar subtitle translator.

Translate each English dialogue line into natural, clear Myanmar Burmese.

VERY IMPORTANT:

- Translate ONLY the dialogue text.
- Do NOT add explanations.
- Do NOT add comments.
- Do NOT add instructions.
- Do NOT invent dialogue.
- Do NOT repeat the English dialogue.
- Do NOT merge subtitle lines.
- Do NOT split subtitle lines.
- Keep exactly the same IDs.
- Return JSON only.

The "start" and "end" values are provided only for reference.
DO NOT change them.

Return exactly this structure:

{
  "transcript": [
    {
      "id": 1,
      "text": "မြန်မာဘာသာပြန်စာ"
    }
  ]
}

SOURCE:
${JSON.stringify(
  inputSegments.map(
    item => ({
      id:
        item.id,

      text:
        item.text
    })
  )
)}
`;


      /*
        Gemini Interactions API
      */

      const response =
        await fetch(
          "https://generativelanguage.googleapis.com/v1beta/interactions",
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",

              "x-goog-api-key":
                key
            },

            body:
              JSON.stringify({
                model:
                  "gemini-3.8-flash",

                input:
                  prompt
              })
          }
        );


      const rawText =
        await response.text();

      const data =
        parseJsonSafely(
          rawText
        );


      /*
        Handle Gemini quota
        separately.
      */

      if (
        response.status === 429
      ) {

        console.error(
          "GEMINI RATE LIMIT:",
          rawText
        );

        return res
          .status(429)
          .json({
            error:
              "Gemini Free Tier quota ပြည့်နေပါတယ်။ ခဏစောင့်ပြီး နောက်မှ Translate ပြန်နှိပ်ပါ။ Auto Retry မလုပ်ထားပါ။"
          });
      }


      if (!response.ok) {

        throw new Error(
          data?.error?.message ||
          data?.error ||
          rawText ||
          "Gemini translation failed"
        );
      }


      const outputText =
        extractInteractionText(
          data
        );


      const parsed =
        extractJsonObject(
          outputText
        );


      if (!parsed) {
        throw new Error(
          "Gemini က valid JSON translation မပြန်ပါ"
        );
      }


      const translated =
        Array.isArray(
          parsed.transcript
        )
          ? parsed.transcript
          : Array.isArray(
              parsed
            )
            ? parsed
            : [];


      if (!translated.length) {
        throw new Error(
          "Gemini Myanmar Translation data မမှန်ပါ"
        );
      }


      /*
        Match translation by ID.
      */

      const translatedById =
        new Map();


      translated.forEach(
        (item, index) => {

          const id =
            Number(
              item.id
            ) ||
            index + 1;


          const text =
            String(
              item.text ||
              ""
            )
              .trim();


          translatedById.set(
            id,
            text
          );
        }
      );


      /*
        ALWAYS keep original
        Groq timing.
      */

      const result =
        inputSegments
          .map(
            item => ({
              id:
                item.id,

              start:
                item.start,

              end:
                item.end,

              text:
                translatedById.get(
                  item.id
                ) ||
                item.text
            })
          )
          .filter(
            item =>
              item.text
          );


      console.log(
        `Gemini translation complete: ${result.length} subtitle segments`
      );


      res.json({
        ok: true,

        transcript:
          result
      });


    } catch (error) {

      console.error(
        "TRANSLATE ERROR:",
        error
      );


      res
        .status(500)
        .json({
          error:
            error.message ||
            "Gemini Myanmar Translation Error"
        });
    }
  }
);


/* =========================================================
   GEMINI OUTPUT EXTRACTION
========================================================= */

function extractInteractionText(
  data
) {

  if (!data) {
    return "";
  }


  if (
    typeof data.output_text ===
    "string"
  ) {
    return data.output_text;
  }


  if (
    typeof data.text ===
    "string"
  ) {
    return data.text;
  }


  if (
    typeof data.output ===
    "string"
  ) {
    return data.output;
  }


  if (
    Array.isArray(
      data.output
    )
  ) {

    const parts = [];


    for (
      const item of
      data.output
    ) {

      if (
        typeof item ===
        "string"
      ) {
        parts.push(
          item
        );

        continue;
      }


      if (
        typeof item?.text ===
        "string"
      ) {
        parts.push(
          item.text
        );

        continue;
      }


      if (
        Array.isArray(
          item?.content
        )
      ) {

        for (
          const content of
          item.content
        ) {

          if (
            typeof content ===
            "string"
          ) {

            parts.push(
              content
            );

          } else if (
            typeof content?.text ===
            "string"
          ) {

            parts.push(
              content.text
            );
          }
        }
      }
    }


    return parts.join("");
  }


  /*
    Compatibility with
    candidate-style responses.
  */

  if (
    Array.isArray(
      data.candidates
    )
  ) {

    const parts = [];


    for (
      const candidate of
      data.candidates
    ) {

      const content =
        candidate?.content;


      if (
        typeof content?.text ===
        "string"
      ) {

        parts.push(
          content.text
        );
      }


      if (
        Array.isArray(
          content?.parts
        )
      ) {

        for (
          const part of
          content.parts
        ) {

          if (
            typeof part?.text ===
            "string"
          ) {

            parts.push(
              part.text
            );
          }
        }
      }
    }


    return parts.join("");
  }


  return "";
}


/* =========================================================
   MOVIE RECAP DISABLED
========================================================= */

app.all(
  "/api/movie-auto",
  (req, res) => {

    res
      .status(410)
      .json({
        error:
          "Movie Recap feature ကို SRT-only version မှာ ပိတ်ထားပါတယ်"
      });
  }
);


app.all(
  "/api/movie-recap",
  (req, res) => {

    res
      .status(410)
      .json({
        error:
          "Movie Recap feature ကို SRT-only version မှာ ပိတ်ထားပါတယ်"
      });
  }
);


app.all(
  "/api/movie-voice",
  (req, res) => {

    res
      .status(410)
      .json({
        error:
          "Movie Recap feature ကို SRT-only version မှာ ပိတ်ထားပါတယ်"
      });
  }
);


app.all(
  "/api/movie-render",
  (req, res) => {

    res
      .status(410)
      .json({
        error:
          "Movie Recap feature ကို SRT-only version မှာ ပိတ်ထားပါတယ်"
      });
  }
);


app.all(
  "/api/recap/one-click",
  (req, res) => {

    res
      .status(410)
      .json({
        error:
          "Movie Recap feature ကို SRT-only version မှာ ပိတ်ထားပါတယ်"
      });
  }
);


app.all(
  "/api/recap/analyze",
  (req, res) => {

    res
      .status(410)
      .json({
        error:
          "Movie Recap feature ကို SRT-only version မှာ ပိတ်ထားပါတယ်"
      });
  }
);


app.all(
  "/api/recap/tts",
  (req, res) => {

    res
      .status(410)
      .json({
        error:
          "Movie Recap feature ကို SRT-only version မှာ ပိတ်ထားပါတယ်"
      });
  }
);


app.all(
  "/api/recap/voice-sync",
  (req, res) => {

    res
      .status(410)
      .json({
        error:
          "Movie Recap feature ကို SRT-only version မှာ ပိတ်ထားပါတယ်"
      });
  }
);


app.all(
  "/api/render",
  (req, res) => {

    res
      .status(410)
      .json({
        error:
          "MP4 Render feature ကို SRT-only version မှာ ပိတ်ထားပါတယ်"
      });
  }
);


/* =========================================================
   API 404
========================================================= */

app.use(
  "/api",
  (req, res) => {

    res
      .status(404)
      .json({
        error:
          "API endpoint မတွေ့ပါ"
      });
  }
);


/* =========================================================
   ERROR HANDLER
========================================================= */

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


    res
      .status(500)
      .json({
        error:
          error.message ||
          "Server Error"
      });
  }
);


/* =========================================================
   START SERVER
========================================================= */

app.listen(
  PORT,
  () => {

    console.log(
      `Myanmar SRT server running on port ${PORT}`
    );

    console.log(
      "SRT ONLY MODE: ENABLED"
    );

    console.log(
      "Groq model: whisper-large-v3-turbo"
    );

    console.log(
      "Groq timestamps: WORD + SEGMENT"
    );

    console.log(
      "Subtitle split: ENABLED"
    );

    console.log(
      "Gemini model: gemini-3.8-flash"
    );

    console.log(
      "Gemini auto-retry: DISABLED"
    );

    console.log(
      "Movie Recap: DISABLED"
    );

    console.log(
      "MP4 Render: DISABLED"
    );
  }
);
