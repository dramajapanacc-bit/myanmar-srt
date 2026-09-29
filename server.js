const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const os = require("os");
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


/* =========================================================
   EXPRESS
========================================================= */

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
   MULTER
========================================================= */

const upload = multer({
  dest: WORK_DIR,

  limits: {
    fileSize:
      300 * 1024 * 1024
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
   FILE CLEANUP
========================================================= */

function cleanupFile(filePath) {

  try {

    if (
      filePath &&
      fs.existsSync(filePath)
    ) {

      fs.unlinkSync(filePath);

    }

  } catch (error) {

    console.error(
      "Cleanup error:",
      error.message
    );

  }

}


/* =========================================================
   JSON HELPER
========================================================= */

function parseJson(text) {

  try {

    return JSON.parse(text);

  } catch (error) {

    return null;

  }

}


/* =========================================================
   EXTRACT JSON FROM GEMINI
========================================================= */

function extractJson(text) {

  if (!text) {
    return null;
  }


  let value =
    String(text).trim();


  /* Direct JSON */

  let parsed =
    parseJson(value);

  if (parsed) {
    return parsed;
  }


  /* Remove markdown code block */

  value = value
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


  parsed =
    parseJson(value);

  if (parsed) {
    return parsed;
  }


  /* Find object */

  const start =
    value.indexOf("{");

  const end =
    value.lastIndexOf("}");


  if (
    start !== -1 &&
    end > start
  ) {

    parsed =
      parseJson(
        value.slice(
          start,
          end + 1
        )
      );

    if (parsed) {
      return parsed;
    }

  }


  return null;

}


/* =========================================================
   COMMAND RUNNER
========================================================= */

function runCommand(
  command,
  args
) {

  return new Promise(
    (resolve, reject) => {

      const child =
        spawn(
          command,
          args,
          {
            windowsHide: true
          }
        );


      let stdout = "";
      let stderr = "";


      child.stdout.on(
        "data",
        data => {

          stdout +=
            data.toString();

        }
      );


      child.stderr.on(
        "data",
        data => {

          stderr +=
            data.toString();

        }
      );


      child.on(
        "error",
        error => {

          reject(error);

        }
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
                `${command} exited with code ${code}\n${stderr.slice(-4000)}`
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

async function getVideoDuration(
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
    !Number.isFinite(duration)
  ) {

    throw new Error(
      "Video duration မဖတ်နိုင်ပါ"
    );

  }


  return duration;

}


/* =========================================================
   HEALTH
========================================================= */

app.get(
  "/api/health",
  (req, res) => {

    res.json({

      ok: true,

      service:
        "myanmar-srt",

      mode:
        "SRT ONLY",

      groqModel:
        "whisper-large-v3-turbo",

      geminiModel:
        "gemini-3.8-flash",

      movieRecap:
        false,

      mp4Render:
        false,

      groqConfigured:
        Boolean(
          process.env.GROQ_API_KEY
        ),

      geminiConfigured:
        Boolean(
          process.env.GEMINI_API_KEY
        ),

      time:
        new Date().toISOString()

    });

  }
);


/* =========================================================
   GROQ TRANSCRIPT
========================================================= */

app.post(
  "/api/transcribe",

  upload.single("video"),

  async (req, res) => {

    let filePath = null;


    try {

      /* API KEY */

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


      /* VIDEO */

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


      /* DURATION */

      const duration =
        await getVideoDuration(
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


      /* =================================================
         GROQ FORM
      ================================================= */

      const form =
        new FormData();


      const bytes =
        fs.readFileSync(
          filePath
        );


      const blob =
        new Blob(
          [bytes],
          {
            type:
              req.file.mimetype ||
              "video/mp4"
          }
        );


      form.append(
        "file",
        blob,
        req.file.originalname ||
          "video.mp4"
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
        "segment"
      );


      form.append(
        "language",
        "my"
      );


      form.append(
        "prompt",

        "Transcribe the spoken dialogue accurately. Preserve Burmese words, names, numbers and natural speech. Return accurate segment timestamps."
      );


      /* =================================================
         GROQ REQUEST
      ================================================= */

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
        parseJson(
          rawText
        );


      if (!response.ok) {

        throw new Error(

          data?.error?.message ||

          data?.error ||

          rawText ||

          "Groq Transcript Error"

        );

      }


      /* =================================================
         SEGMENTS
      ================================================= */

      const segments =
        Array.isArray(
          data?.segments
        )
          ? data.segments
          : [];


      const transcript =
        segments

          .map(
            (segment, index) => {

              const start =
                Number(
                  segment.start
                );

              const end =
                Number(
                  segment.end
                );

              const text =
                String(
                  segment.text ||
                  ""
                ).trim();


              return {

                id:
                  index + 1,

                start:
                  Number.isFinite(
                    start
                  )
                    ? start
                    : 0,

                end:
                  Number.isFinite(
                    end
                  )
                    ? end
                    : (
                      Number.isFinite(
                        start
                      )
                        ? start + 1
                        : 1
                    ),

                text

              };

            }
          )

          .filter(
            item =>
              item.text
          );


      /* =================================================
         FALLBACK
      ================================================= */

      if (
        !transcript.length &&
        data?.text
      ) {

        transcript.push({

          id: 1,

          start: 0,

          end:
            duration,

          text:
            String(
              data.text
            ).trim()

        });

      }


      /* =================================================
         RESPONSE
      ================================================= */

      res.json({

        ok: true,

        duration,

        language:
          data?.language ||
          "my",

        text:
          data?.text ||
          "",

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
   GEMINI MYANMAR TRANSLATION
========================================================= */

app.post(
  "/api/translate",

  async (req, res) => {

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


      if (
        !transcript.length
      ) {

        return res
          .status(400)
          .json({

            error:
              "Transcript မရှိပါ"

          });

      }


      /* =================================================
         NORMALIZE INPUT
      ================================================= */

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
              )

          })
        );


      /* =================================================
         GEMINI PROMPT
      ================================================= */

      const prompt = `

You are a professional Myanmar movie subtitle translator.

Translate the following dialogue into natural,
clear and easy-to-read Myanmar Burmese.

IMPORTANT RULES:

1. Return JSON only.
2. Do not add explanations.
3. Do not remove any segment.
4. Keep exactly the same IDs.
5. Do not merge segments.
6. Do not split segments.
7. Translate only the text.
8. Keep every timestamp exactly unchanged.
9. Keep names and important English terms naturally.
10. Make the Myanmar subtitle short and natural.

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
  inputSegments
)}

`;


      /* =================================================
         GEMINI INTERACTIONS API
      ================================================= */

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
        parseJson(
          rawText
        );


      if (!response.ok) {

        throw new Error(

          data?.error?.message ||

          data?.error ||

          rawText ||

          "Gemini Translation Error"

        );

      }


      /* =================================================
         GEMINI OUTPUT TEXT
      ================================================= */

      let outputText =
        "";


      if (
        typeof data?.output_text ===
        "string"
      ) {

        outputText =
          data.output_text;

      }


      else if (
        typeof data?.text ===
        "string"
      ) {

        outputText =
          data.text;

      }


      else if (
        Array.isArray(
          data?.output
        )
      ) {

        const parts = [];


        for (
          const item
          of data.output
        ) {

          if (
            typeof item ===
            "string"
          ) {

            parts.push(
              item
            );

          }


          else if (
            typeof item?.text ===
            "string"
          ) {

            parts.push(
              item.text
            );

          }


          else if (
            Array.isArray(
              item?.content
            )
          ) {

            for (
              const content
              of item.content
            ) {

              if (
                typeof content ===
                "string"
              ) {

                parts.push(
                  content
                );

              }


              else if (
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


        outputText =
          parts.join("");

      }


      if (
        !outputText
      ) {

        throw new Error(
          "Gemini output မရပါ"
        );

      }


      /* =================================================
         PARSE JSON
      ================================================= */

      const parsed =
        extractJson(
          outputText
        );


      if (!parsed) {

        throw new Error(
          "Gemini က valid JSON မပြန်ပါ"
        );

      }


      const translated =
        Array.isArray(
          parsed.transcript
        )
          ? parsed.transcript
          : [];


      if (
        !translated.length
      ) {

        throw new Error(
          "Myanmar Translation data မရပါ"
        );

      }


      /* =================================================
         MATCH TRANSLATION
         WITH ORIGINAL TIMING
      ================================================= */

      const translationMap =
        new Map();


      translated.forEach(
        (item, index) => {

          const id =
            Number(
              item.id
            ) ||
            index + 1;


          translationMap.set(

            id,

            String(
              item.text ||
              ""
            ).trim()

          );

        }
      );


      /*
        IMPORTANT:

        Gemini timing ကို မယုံပါ။

        Groq ရဲ့ original
        start/end ကိုပဲ
        ပြန်သုံးပါတယ်။
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
                translationMap.get(
                  item.id
                ) ||
                item.text

            })
          )

          .filter(
            item =>
              item.text
          );


      /* =================================================
         RESPONSE
      ================================================= */

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
   MOVIE RECAP DISABLED
========================================================= */

function movieRecapDisabled(
  req,
  res
) {

  res
    .status(410)
    .json({

      error:
        "Movie Recap feature ကို ပိတ်ထားပါတယ်။ SRT feature ကိုပဲ အသုံးပြုနိုင်ပါတယ်။"

    });

}


app.all(
  "/api/movie-auto",
  movieRecapDisabled
);

app.all(
  "/api/movie-recap",
  movieRecapDisabled
);

app.all(
  "/api/movie-voice",
  movieRecapDisabled
);

app.all(
  "/api/movie-render",
  movieRecapDisabled
);

app.all(
  "/api/recap/one-click",
  movieRecapDisabled
);

app.all(
  "/api/recap/analyze",
  movieRecapDisabled
);

app.all(
  "/api/recap/tts",
  movieRecapDisabled
);

app.all(
  "/api/recap/voice-sync",
  movieRecapDisabled
);

app.all(
  "/api/render",
  movieRecapDisabled
);


/* =========================================================
   404 API
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
      "Groq: whisper-large-v3-turbo"
    );

    console.log(
      "Gemini: gemini-3.8-flash"
    );

    console.log(
      "Movie Recap: DISABLED"
    );

    console.log(
      "MP4 Render: DISABLED"
    );

  }
);
