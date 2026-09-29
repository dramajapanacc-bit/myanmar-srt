import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';

import Groq from 'groq-sdk';
import { GoogleGenAI } from '@google/genai';
import ffprobeStatic from 'ffprobe-static';


const app = express();

const PORT =
  Number(process.env.PORT || 10000);

const MAX_BYTES =
  300 * 1024 * 1024;

const MAX_SECONDS =
  5 * 60;

const ROOT =
  process.cwd();

const PUBLIC =
  path.join(ROOT, 'public');

const TMP =
  path.join(
    os.tmpdir(),
    'myanmar-srt'
  );


await fs.mkdir(
  TMP,
  { recursive: true }
);


// ============================================================
// MODELS
// ============================================================

const GROQ_MODEL =
  'whisper-large-v3-turbo';

const GEMINI_MODEL =
  'gemini-3.8-flash';


// ============================================================
// ALLOWED FILE TYPES
// ============================================================

const allowedExt =
  new Set([
    '.flac',
    '.mp3',
    '.mp4',
    '.mpeg',
    '.mpga',
    '.m4a',
    '.ogg',
    '.opus',
    '.wav',
    '.webm'
  ]);


// ============================================================
// MULTER
// ============================================================

const upload =
  multer({

    storage:
      multer.diskStorage({

        destination:
          (_req, _file, cb) => {
            cb(
              null,
              TMP
            );
          },

        filename:
          (_req, file, cb) => {

            const originalExt =
              path
                .extname(
                  file.originalname || ''
                )
                .toLowerCase();

            const ext =
              allowedExt.has(
                originalExt
              )
                ? originalExt
                : '.mp4';

            cb(
              null,
              `${crypto.randomUUID()}${ext}`
            );
          }
      }),

    limits: {
      fileSize:
        MAX_BYTES,

      files:
        1
    },

    fileFilter:
      (_req, file, cb) => {

        const ext =
          path
            .extname(
              file.originalname || ''
            )
            .toLowerCase();

        if (
          !allowedExt.has(ext)
        ) {

          return cb(
            new Error(
              'MP4 သို့မဟုတ် WEBM video ကိုသုံးပါ။'
            )
          );
        }

        cb(
          null,
          true
        );
      }
  });


// ============================================================
// EXPRESS
// ============================================================

app.use(
  express.json({
    limit: '10mb'
  })
);

app.use(
  express.urlencoded({
    extended: true,
    limit: '10mb'
  })
);

app.use(
  express.static(
    PUBLIC
  )
);


// ============================================================
// API KEY
// ============================================================

function getKey(
  req,
  headerName,
  bodyName,
  envName,
  label
) {

  const key =
    String(
      req.headers[headerName] ||
      req.body?.[bodyName] ||
      process.env[envName] ||
      ''
    ).trim();

  if (!key) {

    throw new Error(
      `${label} မရှိပါ`
    );
  }

  return key;
}


// ============================================================
// CLEAN TEXT
// ============================================================

function cleanText(text) {

  return String(
    text || ''
  )
    .replace(
      /Return accurate segment[.!]?/gi,
      ''
    )
    .replace(
      /Return accurate transcript[.!]?/gi,
      ''
    )
    .replace(
      /Return the spoken word[.!]?/gi,
      ''
    )
    .replace(
      /\s{2,}/g,
      ' '
    )
    .trim();
}


// ============================================================
// PROBE VIDEO
// ============================================================

async function probeVideo(
  file
) {

  const result =
    await new Promise(
      (
        resolve,
        reject
      ) => {

        const child =
          spawn(
            ffprobeStatic.path,
            [
              '-v',
              'error',

              '-show_entries',
              'format=duration,size',

              '-show_streams',

              '-of',
              'json',

              file
            ]
          );

        let stdout =
          '';

        let stderr =
          '';

        child.stdout.on(
          'data',
          d => {
            stdout +=
              d.toString();
          }
        );

        child.stderr.on(
          'data',
          d => {
            stderr +=
              d.toString();
          }
        );

        child.on(
          'error',
          reject
        );

        child.on(
          'close',
          code => {

            if (
              code === 0
            ) {

              resolve({
                stdout,
                stderr
              });

            } else {

              reject(
                new Error(
                  stderr ||
                  `ffprobe exited with code ${code}`
                )
              );
            }
          }
        );
      }
    );


  const data =
    JSON.parse(
      result.stdout
    );


  const duration =
    Number(
      data?.format?.duration ||
      0
    );


  if (
    !Number.isFinite(
      duration
    ) ||
    duration <= 0
  ) {

    throw new Error(
      'Video duration မဖတ်နိုင်ပါ'
    );
  }


  return {
    duration
  };
}


// ============================================================
// GROQ TRANSCRIPTION
// ============================================================

async function transcribeGroq(
  file,
  apiKey
) {

  const groq =
    new Groq({
      apiKey
    });


  return groq
    .audio
    .transcriptions
    .create({

      file:
        createReadStream(
          file
        ),

      model:
        GROQ_MODEL,

      response_format:
        'verbose_json',

      timestamp_granularities:
        [
          'word',
          'segment'
        ],

      temperature:
        0
    });
}


// ============================================================
// BUILD TRANSCRIPT SEGMENTS
// ============================================================

function buildSegments(
  result,
  duration
) {

  const segments =
    Array.isArray(
      result?.segments
    )
      ? result.segments
      : [];


  const output =
    segments

      .map(
        (
          s,
          i
        ) => ({

          id:
            i + 1,

          start:
            Math.max(
              0,
              Number(
                s.start
              ) || 0
            ),

          end:
            Math.max(
              0,
              Number(
                s.end
              ) || 0
            ),

          text:
            cleanText(
              s.text
            )
        })
      )

      .filter(
        s =>
          s.text &&
          s.end > s.start
      );


  if (
    output.length
  ) {

    return output;
  }


  const fullText =
    cleanText(
      result?.text
    );


  if (
    fullText
  ) {

    return [
      {
        id:
          1,

        start:
          0,

        end:
          duration,

        text:
          fullText
      }
    ];
  }


  return [];
}


// ============================================================
// EXTRACT JSON
// ============================================================

function extractJson(
  text
) {

  let raw =
    String(
      text || ''
    ).trim();


  raw =
    raw
      .replace(
        /^```json\s*/i,
        ''
      )
      .replace(
        /^```\s*/i,
        ''
      )
      .replace(
        /```\s*$/i,
        ''
      )
      .trim();


  try {

    return JSON.parse(
      raw
    );

  } catch {}


  const objectStart =
    raw.indexOf(
      '{'
    );

  const objectEnd =
    raw.lastIndexOf(
      '}'
    );


  if (
    objectStart >= 0 &&
    objectEnd > objectStart
  ) {

    try {

      return JSON.parse(
        raw.slice(
          objectStart,
          objectEnd + 1
        )
      );

    } catch {}
  }


  const arrayStart =
    raw.indexOf(
      '['
    );

  const arrayEnd =
    raw.lastIndexOf(
      ']'
    );


  if (
    arrayStart >= 0 &&
    arrayEnd > arrayStart
  ) {

    try {

      return JSON.parse(
        raw.slice(
          arrayStart,
          arrayEnd + 1
        )
      );

    } catch {}
  }


  return null;
}


// ============================================================
// GEMINI TRANSLATION
// ============================================================

async function translateChunkGemini(
  segments,
  apiKey
) {

  const ai =
    new GoogleGenAI({
      apiKey
    });


  const input =
    segments.map(
      s => ({

        id:
          s.id,

        text:
          s.text
      })
    );


  const prompt = `
You are a professional Myanmar subtitle translator.

Translate EVERY source dialogue line into natural,
clear and concise Myanmar Unicode.

The source language may be:
English, Chinese, Japanese, Korean,
or any other spoken language.

Return ONLY valid JSON.

Required format:

{
  "translations": [
    {
      "id": 1,
      "translation": "မြန်မာဘာသာပြန်"
    }
  ]
}

Rules:

- Keep every id exactly the same.
- Do not remove lines.
- Do not merge lines.
- Do not reorder lines.
- Do not invent dialogue.
- Translate only spoken dialogue.
- No explanation.
- No markdown.
- No comments.
- Keep names naturally.
- Keep subtitles short.
- Prefer under 42 characters when possible.
- Preserve meaning and emotion.

INPUT:

${JSON.stringify(input)}
`;


  let response;


  try {

    response =
      await ai.models.generateContent({

        model:
          GEMINI_MODEL,

        contents:
          prompt,

        config: {

          temperature:
            0.2,

          responseMimeType:
            'application/json'
        }
      });

  } catch (error) {

    const message =
      String(
        error?.message ||
        error
      );


    if (
      /429|rate.?limit|quota|resource.?exhausted/i
        .test(message)
    ) {

      throw new Error(
        'Gemini Free Tier quota ပြည့်နေပါတယ်။ ခဏစောင့်ပြီး နောက်မှ Translate ပြန်နှိပ်ပါ။'
      );
    }


    throw error;
  }


  const parsed =
    extractJson(
      response?.text ||
      ''
    );


  let list = [];


  if (
    Array.isArray(
      parsed
    )
  ) {

    list =
      parsed;
  }


  if (
    !list.length &&
    Array.isArray(
      parsed?.translations
    )
  ) {

    list =
      parsed.translations;
  }


  if (
    !list.length
  ) {

    throw new Error(
      'Gemini က valid translation JSON မပြန်ပါ'
    );
  }


  const byId =
    new Map();


  for (
    const item of list
  ) {

    const id =
      Number(
        item?.id
      );


    const translation =
      String(
        item?.translation ||
        item?.text ||
        item?.myanmar ||
        ''
      ).trim();


    if (
      Number.isFinite(
        id
      ) &&
      translation
    ) {

      byId.set(
        id,
        translation
      );
    }
  }


  if (
    !byId.size
  ) {

    throw new Error(
      'Gemini translation data မမှန်ပါ'
    );
  }


  return segments.map(
    s => ({

      ...s,

      translation:
        byId.get(
          s.id
        ) ||
        s.text
    })
  );
}


// ============================================================
// TRANSLATE ALL WITH GEMINI ONLY
// ============================================================

async function translateAllGemini(
  segments,
  apiKey
) {

  // Smaller chunks reduce Gemini request size.
  const max =
    40;

  const results =
    [];


  for (
    let i = 0;
    i < segments.length;
    i += max
  ) {

    const chunk =
      segments.slice(
        i,
        i + max
      );


    console.log(
      `Gemini translation chunk ${Math.floor(i / max) + 1}`
    );


    const translated =
      await translateChunkGemini(
        chunk,
        apiKey
      );


    results.push(
      ...translated
    );
  }


  return results;
}


// ============================================================
// SPLIT SUBTITLE
// ============================================================

function splitSubtitle(
  text,
  maxChars = 42,
  maxWords = 10
) {

  const value =
    String(
      text || ''
    ).trim();


  if (!value) {

    return [];
  }


  if (
    value.length <=
      maxChars &&
    value.split(
      /\s+/
    ).length <=
      maxWords
  ) {

    return [
      value
    ];
  }


  const words =
    value
      .split(
        /\s+/
      )
      .filter(Boolean);


  const parts =
    [];

  let current =
    '';


  for (
    const word of words
  ) {

    const next =
      current
        ? `${current} ${word}`
        : word;


    if (
      current &&
      (
        next.length >
          maxChars ||
        next.split(
          /\s+/
        ).length >
          maxWords
      )
    ) {

      parts.push(
        current.trim()
      );

      current =
        word;

    } else {

      current =
        next;
    }
  }


  if (
    current
  ) {

    parts.push(
      current.trim()
    );
  }


  if (
    parts.length === 1 &&
    parts[0].length >
      maxChars
  ) {

    const chunks =
      [];


    for (
      let i = 0;
      i < parts[0].length;
      i += maxChars
    ) {

      chunks.push(
        parts[0]
          .slice(
            i,
            i + maxChars
          )
          .trim()
      );
    }


    return chunks.filter(
      Boolean
    );
  }


  return parts;
}


// ============================================================
// SPLIT TRANSLATED SEGMENTS
// ============================================================

function splitTranslatedSegments(
  segments
) {

  const output =
    [];

  let id =
    1;


  for (
    const s of segments
  ) {

    const text =
      String(
        s.translation ||
        s.text ||
        ''
      ).trim();


    const parts =
      splitSubtitle(
        text
      );


    if (
      parts.length <= 1
    ) {

      output.push({

        id:
          id++,

        start:
          s.start,

        end:
          s.end,

        text,

        translation:
          text
      });


      continue;
    }


    const total =
      parts.reduce(
        (
          sum,
          part
        ) =>
          sum +
          Math.max(
            1,
            part.length
          ),
        0
      );


    const duration =
      Math.max(
        0.2,
        s.end -
          s.start
      );


    let cursor =
      s.start;


    parts.forEach(
      (
        part,
        index
      ) => {

        const end =
          index ===
          parts.length - 1

            ? s.end

            : cursor +
              duration *
                (
                  Math.max(
                    1,
                    part.length
                  ) /
                  total
                );


        output.push({

          id:
            id++,

          start:
            cursor,

          end:
            Math.min(
              end,
              s.end
            ),

          text:
            part,

          translation:
            part
        });


        cursor =
          Math.min(
            end,
            s.end
          );
      }
    );
  }


  return output;
}


// ============================================================
// SRT TIME
// ============================================================

function srtTime(
  seconds
) {

  const ms =
    Math.max(
      0,
      Math.round(
        Number(
          seconds || 0
        ) * 1000
      )
    );


  const h =
    Math.floor(
      ms /
      3600000
    );


  const m =
    Math.floor(
      (
        ms %
        3600000
      ) /
      60000
    );


  const s =
    Math.floor(
      (
        ms %
        60000
      ) /
      1000
    );


  const milli =
    ms % 1000;


  return (
    `${String(h).padStart(2, '0')}:` +
    `${String(m).padStart(2, '0')}:` +
    `${String(s).padStart(2, '0')},` +
    `${String(milli).padStart(3, '0')}`
  );
}


// ============================================================
// MAKE SRT
// ============================================================

function makeSrt(
  segments
) {

  return segments
    .map(
      (
        s,
        i
      ) => (

        `${i + 1}\n` +

        `${srtTime(s.start)} --> ${srtTime(s.end)}\n` +

        `${s.translation || s.text}\n`
      )
    )
    .join('\n');
}


// ============================================================
// CLEANUP
// ============================================================

function cleanup(
  file
) {

  if (
    file
  ) {

    fs.rm(
      file,
      {
        force:
          true
      }
    ).catch(
      () => {}
    );
  }
}


// ============================================================
// HEALTH
// ============================================================

app.get(
  '/api/health',
  (_req, res) => {

    res.json({

      ok:
        true,

      service:
        'myanmar-srt',

      srtOnly:
        true,

      movieRecap:
        false,

      render:
        false,

      groqConfigured:
        Boolean(
          process.env.GROQ_API_KEY
        ),

      geminiConfigured:
        Boolean(
          process.env.GEMINI_API_KEY
        ),

      groqModel:
        GROQ_MODEL,

      geminiModel:
        GEMINI_MODEL,

      translation:
        'Gemini Only'
    });
  }
);


// ============================================================
// GROQ TRANSCRIBE
// ============================================================

app.post(
  '/api/transcribe',

  upload.single(
    'video'
  ),

  async (
    req,
    res
  ) => {

    const file =
      req.file?.path;


    try {

      if (
        !file
      ) {

        throw new Error(
          'Video file မရှိပါ'
        );
      }


      const groqKey =
        getKey(
          req,
          'x-groq-api-key',
          'groqApiKey',
          'GROQ_API_KEY',
          'Groq API Key'
        );


      const {
        duration
      } =
        await probeVideo(
          file
        );


      if (
        duration >
        MAX_SECONDS
      ) {

        throw new Error(
          'Video က 5 မိနစ်ထက်ကျော်နေပါတယ်'
        );
      }


      const result =
        await transcribeGroq(
          file,
          groqKey
        );


      const transcript =
        buildSegments(
          result,
          duration
        );


      if (
        !transcript.length
      ) {

        throw new Error(
          'Groq က စာတန်းမထုတ်ပေးနိုင်ပါ'
        );
      }


      res.json({

        ok:
          true,

        duration,

        language:
          result?.language ||
          'auto',

        text:
          result?.text ||
          transcript
            .map(
              x =>
                x.text
            )
            .join(' '),

        transcript
      });


    } catch (
      error
    ) {

      console.error(
        'TRANSCRIBE ERROR:',
        error
      );


      res.status(
        400
      ).json({

        ok:
          false,

        error:
          error?.message ||
          'Groq Transcript Error'
      });


    } finally {

      cleanup(
        file
      );
    }
  }
);


// ============================================================
// GEMINI TRANSLATE
// ============================================================

app.post(
  '/api/translate',

  async (
    req,
    res
  ) => {

    try {

      const geminiKey =
        getKey(
          req,
          'x-gemini-api-key',
          'geminiApiKey',
          'GEMINI_API_KEY',
          'Gemini API Key'
        );


      const input =
        Array.isArray(
          req.body?.transcript
        )
          ? req.body.transcript
          : [];


      if (
        !input.length
      ) {

        throw new Error(
          'Transcript မရှိပါ'
        );
      }


      const normalized =
        input

          .map(
            (
              s,
              i
            ) => ({

              id:
                i + 1,

              start:
                Number(
                  s.start
                ) || 0,

              end:
                Number(
                  s.end
                ) || 0,

              text:
                cleanText(
                  s.text
                )
            })
          )

          .filter(
            s =>
              s.text &&
              s.end >
                s.start
          );


      if (
        !normalized.length
      ) {

        throw new Error(
          'ဘာသာပြန်ရန် Transcript data မမှန်ပါ'
        );
      }


      const translated =
        await translateAllGemini(
          normalized,
          geminiKey
        );


      const subtitleSegments =
        splitTranslatedSegments(
          translated
        );


      res.json({

        ok:
          true,

        provider:
          'gemini',

        transcript:
          subtitleSegments,

        srt:
          makeSrt(
            subtitleSegments
          )
      });


    } catch (
      error
    ) {

      console.error(
        'TRANSLATE ERROR:',
        error
      );


      res.status(
        400
      ).json({

        ok:
          false,

        error:
          error?.message ||
          'Gemini Myanmar Translation Error'
      });
    }
  }
);


// ============================================================
// OLD PROCESS API
// ============================================================

app.post(
  '/api/process',

  upload.single(
    'video'
  ),

  async (
    req,
    res
  ) => {

    const file =
      req.file?.path;


    try {

      if (
        !file
      ) {

        throw new Error(
          'Video file မရှိပါ'
        );
      }


      const groqKey =
        getKey(
          req,
          'x-groq-api-key',
          'groqKey',
          'GROQ_API_KEY',
          'Groq API Key'
        );


      const geminiKey =
        getKey(
          req,
          'x-gemini-api-key',
          'geminiKey',
          'GEMINI_API_KEY',
          'Gemini API Key'
        );


      const {
        duration
      } =
        await probeVideo(
          file
        );


      if (
        duration >
        MAX_SECONDS
      ) {

        throw new Error(
          'Video က 5 မိနစ်ထက်ကျော်နေပါတယ်'
        );
      }


      const result =
        await transcribeGroq(
          file,
          groqKey
        );


      const original =
        buildSegments(
          result,
          duration
        );


      const translated =
        await translateAllGemini(
          original,
          geminiKey
        );


      const segments =
        splitTranslatedSegments(
          translated
        );


      res.json({

        ok:
          true,

        duration,

        provider:
          'gemini',

        segments,

        srt:
          makeSrt(
            segments
          )
      });


    } catch (
      error
    ) {

      console.error(
        'PROCESS ERROR:',
        error
      );


      res.status(
        400
      ).json({

        ok:
          false,

        error:
          error?.message ||
          String(error)
      });


    } finally {

      cleanup(
        file
      );
    }
  }
);


// ============================================================
// DISABLED MOVIE RECAP / RENDER
// ============================================================

for (
  const route of [
    '/api/movie-auto',
    '/api/movie-recap',
    '/api/movie-voice',
    '/api/movie-render',
    '/api/recap/one-click',
    '/api/recap/analyze',
    '/api/recap/tts',
    '/api/recap/voice-sync',
    '/api/render'
  ]
) {

  app.all(
    route,
    (
      _req,
      res
    ) =>
      res.status(
        410
      ).json({

        ok:
          false,

        error:
          'Movie Recap / MP4 Render ကို SRT-only version မှာ ပိတ်ထားပါတယ်'
      })
  );
}


// ============================================================
// UNKNOWN API
// ============================================================

app.use(
  '/api',
  (
    _req,
    res
  ) =>
    res.status(
      404
    ).json({

      ok:
        false,

      error:
        'API endpoint မတွေ့ပါ'
    })
);


// ============================================================
// SERVER ERROR
// ============================================================

app.use(
  (
    error,
    req,
    res,
    next
  ) => {

    console.error(
      'SERVER ERROR:',
      error
    );


    if (
      res.headersSent
    ) {

      return next(
        error
      );
    }


    res.status(
      500
    ).json({

      ok:
        false,

      error:
        error?.message ||
        'Server Error'
    });
  }
);


// ============================================================
// START
// ============================================================

app.listen(
  PORT,
  '0.0.0.0',
  () => {

    console.log(
      `Myanmar SRT server running on port ${PORT}`
    );

    console.log(
      'SRT ONLY MODE: ENABLED'
    );

    console.log(
      `Groq: ${GROQ_MODEL}`
    );

    console.log(
      `Gemini: ${GEMINI_MODEL}`
    );

    console.log(
      'Translation: GEMINI ONLY'
    );

    console.log(
      'OpenRouter: DISABLED'
    );

    console.log(
      'Movie Recap: DISABLED'
    );

    console.log(
      'MP4 Render: DISABLED'
    );
  }
);
