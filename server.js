import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import Groq from 'groq-sdk';
import { GoogleGenAI } from '@google/genai';
import ffprobeStatic from 'ffprobe-static';

const app = express();

const PORT = Number(process.env.PORT || 10000);

const MAX_BYTES = 300 * 1024 * 1024;
const MAX_SECONDS = 5 * 60;

const ROOT = process.cwd();
const PUBLIC = path.join(ROOT, 'public');
const TMP = path.join(os.tmpdir(), 'myanmar-srt');

await fs.mkdir(TMP, { recursive: true });

/* =========================
   UPLOAD
========================= */

const allowedExtensions = new Set([
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

const upload = multer({
  storage: multer.diskStorage({

    destination: (_req, _file, cb) => {
      cb(null, TMP);
    },

    filename: (_req, file, cb) => {

      const originalExt =
        path.extname(
          file.originalname || ''
        ).toLowerCase();

      const ext =
        allowedExtensions.has(originalExt)
          ? originalExt
          : '.mp4';

      cb(
        null,
        `${crypto.randomUUID()}${ext}`
      );

    }

  }),

  limits: {
    fileSize: MAX_BYTES,
    files: 1
  },

  fileFilter: (_req, file, cb) => {

    const ext =
      path.extname(
        file.originalname || ''
      ).toLowerCase();

    if (!allowedExtensions.has(ext)) {

      return cb(
        new Error(
          'Groq မထောက်ပံ့တဲ့ video/audio format ဖြစ်ပါတယ်။ MP4 သို့မဟုတ် WEBM ကိုသုံးပါ။'
        )
      );

    }

    cb(null, true);

  }

});

/* =========================
   EXPRESS
========================= */

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
  express.static(PUBLIC)
);

/* =========================
   API KEY
========================= */

function getKey(
  req,
  headerName,
  bodyName,
  envName,
  label
) {

  const key = String(

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

/* =========================
   JSON HELPERS
========================= */

function parseJson(text) {

  try {

    return JSON.parse(text);

  } catch {

    return null;

  }

}

function extractJson(text) {

  const raw =
    String(text || '').trim();

  const direct =
    parseJson(raw);

  if (direct !== null) {
    return direct;
  }

  const cleaned =
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

  const fenced =
    parseJson(cleaned);

  if (fenced !== null) {
    return fenced;
  }

  const start =
    cleaned.indexOf('[');

  const end =
    cleaned.lastIndexOf(']');

  if (
    start >= 0 &&
    end > start
  ) {

    const value =
      parseJson(
        cleaned.slice(
          start,
          end + 1
        )
      );

    if (value !== null) {
      return value;
    }

  }

  const objStart =
    cleaned.indexOf('{');

  const objEnd =
    cleaned.lastIndexOf('}');

  if (
    objStart >= 0 &&
    objEnd > objStart
  ) {

    const value =
      parseJson(
        cleaned.slice(
          objStart,
          objEnd + 1
        )
      );

    if (value !== null) {
      return value;
    }

  }

  return null;

}

/* =========================
   FFPROBE
========================= */

async function probeVideo(file) {

  const result =
    await new Promise(
      (resolve, reject) => {

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

        let stdout = '';
        let stderr = '';

        child.stdout.on(
          'data',
          data => {

            stdout +=
              data.toString();

          }
        );

        child.stderr.on(
          'data',
          data => {

            stderr +=
              data.toString();

          }
        );

        child.on(
          'error',
          reject
        );

        child.on(
          'close',
          code => {

            if (code === 0) {

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
      data?.format?.duration || 0
    );

  if (
    !Number.isFinite(duration) ||
    duration <= 0
  ) {

    throw new Error(
      'Video duration မဖတ်နိုင်ပါ'
    );

  }

  return {
    data,
    duration
  };

}

/* =========================
   CLEAN TRANSCRIPT
========================= */

function cleanTranscriptText(text) {

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

/* =========================
   GROQ TRANSCRIPTION
========================= */

async function transcribeGroq(
  file,
  apiKey
) {

  const groq =
    new Groq({
      apiKey
    });

  const result =
    await groq.audio.transcriptions.create({

      file:
        createReadStream(file),

      model:
        'whisper-large-v3-turbo',

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

  return result;

}

/* =========================
   BUILD SEGMENTS
========================= */

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

  const cleaned =
    segments

      .map(
        (
          s,
          index
        ) => {

          const start =
            Number(
              s.start
            );

          const end =
            Number(
              s.end
            );

          const text =
            cleanTranscriptText(
              s.text
            );

          return {

            id:
              index + 1,

            start:
              Number.isFinite(start)
                ? Math.max(
                    0,
                    start
                  )
                : 0,

            end:
              Number.isFinite(end)
                ? Math.max(
                    0,
                    end
                  )
                : 0,

            text

          };

        }
      )

      .filter(
        s =>
          s.text &&
          s.end > s.start
      );

  if (
    cleaned.length
  ) {

    return cleaned;

  }

  const fullText =
    cleanTranscriptText(
      result?.text
    );

  if (!fullText) {

    return [];

  }

  return [

    {

      id: 1,

      start: 0,

      end: duration,

      text: fullText

    }

  ];

}

/* =========================
   GEMINI TRANSLATION
========================= */

async function translateChunk(
  segments,
  apiKey,
  model
) {

  const ai =
    new GoogleGenAI({
      apiKey
    });

  const source =
    segments.map(
      s => ({

        id:
          s.id,

        text:
          s.text

      })
    );

  const prompt = `You are a professional Myanmar subtitle translator.

Translate each English dialogue line into natural, concise Myanmar Unicode.

Return ONLY a JSON array.

Each array item must contain exactly:
id
translation

Rules:
1. Keep the same id.
2. Do not remove lines.
3. Do not merge lines.
4. Do not reorder lines.
5. Do not invent dialogue.
6. Translate only the spoken dialogue.
7. Do not output instructions.
8. Do not output explanations.
9. Keep names naturally.
10. Keep subtitles short and easy to read.
11. Prefer under 42 characters per subtitle when possible.

INPUT:

${JSON.stringify(source)}
`;

  let response;

  try {

    response =
      await ai.models.generateContent({

        model,

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
        'Gemini Free Tier quota ပြည့်နေပါတယ်။ ခဏစောင့်ပြီး နောက်မှ Translate ပြန်နှိပ်ပါ။ Auto Retry မလုပ်ထားပါ။'
      );

    }

    throw error;

  }

  const parsed =
    extractJson(
      response?.text || ''
    );

  const list =
    Array.isArray(parsed)

      ? parsed

      : Array.isArray(
          parsed?.transcript
        )

        ? parsed.transcript

        : [];

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
        ''
      ).trim();

    if (
      Number.isFinite(id) &&
      translation
    ) {

      byId.set(
        id,
        translation
      );

    }

  }

  return segments.map(
    s => ({

      ...s,

      translation:
        byId.get(s.id) ||
        s.text

    })
  );

}

/* =========================
   TRANSLATE ALL
========================= */

async function translateAll(
  segments,
  apiKey,
  model = 'gemini-3.8-flash'
) {

  const MAX_PER_REQUEST =
    120;

  if (
    segments.length <=
    MAX_PER_REQUEST
  ) {

    return translateChunk(
      segments,
      apiKey,
      model
    );

  }

  const first =
    segments.slice(
      0,
      MAX_PER_REQUEST
    );

  const second =
    segments.slice(
      MAX_PER_REQUEST
    );

  const a =
    await translateChunk(
      first,
      apiKey,
      model
    );

  const b =
    await translateChunk(
      second,
      apiKey,
      model
    );

  return [
    ...a,
    ...b
  ];

}

/* =========================
   SUBTITLE SPLIT
========================= */

function splitLongSubtitle(
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
    value.length <= maxChars &&
    value.split(/\s+/).length <= maxWords
  ) {

    return [
      value
    ];

  }

  const words =
    value
      .split(/\s+/)
      .filter(Boolean);

  const parts = [];

  let current = '';

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
        next.length > maxChars ||
        next.split(/\s+/).length >
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

  if (current) {

    parts.push(
      current.trim()
    );

  }

  if (
    parts.length === 1 &&
    parts[0].length > maxChars
  ) {

    const chars = [];

    for (
      let i = 0;
      i < parts[0].length;
      i += maxChars
    ) {

      chars.push(
        parts[0]
          .slice(
            i,
            i + maxChars
          )
          .trim()
      );

    }

    return chars.filter(Boolean);

  }

  return parts;

}

/* =========================
   TIMED SUBTITLE SEGMENTS
========================= */

function splitTranslatedSegments(
  segments
) {

  const output = [];

  let nextId = 1;

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
      splitLongSubtitle(
        text
      );

    if (
      parts.length <= 1
    ) {

      output.push({

        id:
          nextId++,

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

    const totalWeight =
      parts.reduce(
        (
          sum,
          p
        ) =>
          sum +
          Math.max(
            1,
            p.length
          ),
        0
      );

    const duration =
      Math.max(
        0.2,
        s.end - s.start
      );

    let cursor =
      s.start;

    parts.forEach(
      (
        part,
        index
      ) => {

        const weight =
          Math.max(
            1,
            part.length
          ) /
          totalWeight;

        let end =
          index ===
          parts.length - 1

            ? s.end

            : cursor +
              duration *
              weight;

        if (
          end <= cursor
        ) {

          end =
            cursor + 0.2;

        }

        if (
          end > s.end
        ) {

          end =
            s.end;

        }

        output.push({

          id:
            nextId++,

          start:
            cursor,

          end,

          text:
            part,

          translation:
            part

        });

        cursor =
          end;

      }
    );

  }

  return output;

}

/* =========================
   SRT TIME
========================= */

function srtTime(
  seconds
) {

  const ms =
    Math.max(
      0,
      Math.round(
        Number(
          seconds || 0
        ) *
        1000
      )
    );

  const h =
    Math.floor(
      ms / 3600000
    );

  const m =
    Math.floor(
      (ms % 3600000) /
      60000
    );

  const s =
    Math.floor(
      (ms % 60000) /
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

/* =========================
   MAKE SRT
========================= */

function makeSrt(
  segments
) {

  return segments

    .map(
      (
        s,
        i
      ) => {

        return (

          `${i + 1}\n` +

          `${srtTime(s.start)} --> ${srtTime(s.end)}\n` +

          `${s.translation || s.text}\n`

        );

      }
    )

    .join('\n');

}

/* =========================
   CLEANUP
========================= */

function cleanup(file) {

  if (file) {

    fs.rm(
      file,
      {
        force: true
      }
    ).catch(
      () => {}
    );

  }

}

/* =========================
   HEALTH
========================= */

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
        'whisper-large-v3-turbo',

      geminiModel:
        'gemini-3.8-flash'

    });

  }
);

/* =========================
   GROQ TRANSCRIBE
========================= */

app.post(
  '/api/transcribe',
  upload.single('video'),

  async (
    req,
    res
  ) => {

    const file =
      req.file?.path;

    try {

      if (!file) {

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
              x => x.text
            )
            .join(' '),

        transcript

      });

    } catch (error) {

      console.error(
        'TRANSCRIBE ERROR:',
        error
      );

      res.status(400).json({

        ok:
          false,

        error:
          error?.message ||
          'Groq Transcript Error'

      });

    } finally {

      cleanup(file);

    }

  }
);

/* =========================
   GEMINI TRANSLATE
========================= */

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
                cleanTranscriptText(
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
        !normalized.length
      ) {

        throw new Error(
          'ဘာသာပြန်ရန် Transcript data မမှန်ပါ'
        );

      }

      const model =
        String(
          req.body?.geminiModel ||
          'gemini-3.8-flash'
        ).trim();

      const translated =
        await translateAll(
          normalized,
          geminiKey,
          model
        );

      const subtitleSegments =
        splitTranslatedSegments(
          translated
        );

      res.json({

        ok:
          true,

        transcript:
          subtitleSegments,

        srt:
          makeSrt(
            subtitleSegments
          )

      });

    } catch (error) {

      console.error(
        'TRANSLATE ERROR:',
        error
      );

      res.status(400).json({

        ok:
          false,

        error:
          error?.message ||
          'Gemini Myanmar Translation Error'

      });

    }

  }
);

/* =========================
   COMPATIBILITY PROCESS API
========================= */

app.post(
  '/api/process',

  upload.single('video'),

  async (
    req,
    res
  ) => {

    const file =
      req.file?.path;

    try {

      if (!file) {

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

      const model =
        String(
          req.body?.geminiModel ||
          'gemini-3.8-flash'
        ).trim();

      const translated =
        await translateAll(
          original,
          geminiKey,
          model
        );

      const subtitleSegments =
        splitTranslatedSegments(
          translated
        );

      const id =
        crypto.randomUUID();

      const work =
        path.join(
          TMP,
          id
        );

      await fs.mkdir(
        work,
        {
          recursive:
            true
        }
      );

      await fs.writeFile(
        path.join(
          work,
          'myanmar.srt'
        ),
        makeSrt(
          subtitleSegments
        ),
        'utf8'
      );

      await fs.writeFile(
        path.join(
          work,
          'original.json'
        ),
        JSON.stringify(
          {
            duration,
            segments:
              original
          },
          null,
          2
        ),
        'utf8'
      );

      res.json({

        ok:
          true,

        duration,

        segments:
          subtitleSegments,

        files: {

          srt:
            `/api/download/${id}/myanmar.srt`,

          transcript:
            `/api/download/${id}/original.json`

        }

      });

    } catch (error) {

      console.error(
        'PROCESS ERROR:',
        error
      );

      res.status(400).json({

        ok:
          false,

        error:
          error?.message ||
          String(error)

      });

    } finally {

      cleanup(file);

    }

  }
);

/* =========================
   DOWNLOAD
========================= */

app.get(
  '/api/download/:id/:file',

  async (
    req,
    res
  ) => {

    const allowed =
      new Set([
        'myanmar.srt',
        'original.json'
      ]);

    if (
      !allowed.has(
        req.params.file
      )
    ) {

      return res
        .status(404)
        .end();

    }

    const file =
      path.join(
        TMP,
        req.params.id,
        req.params.file
      );

    try {

      await fs.access(
        file
      );

      res.download(
        file
      );

    } catch {

      res.status(404).json({

        error:
          'File not found or expired.'

      });

    }

  }
);

/* =========================
   MOVIE RECAP DISABLED
========================= */

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
    ) => {

      res.status(410).json({

        ok:
          false,

        error:
          'Movie Recap / MP4 Render ကို SRT-only version မှာ ပိတ်ထားပါတယ်'

      });

    }
  );

}

/* =========================
   UNKNOWN API
========================= */

app.use(
  '/api',

  (
    _req,
    res
  ) => {

    res.status(404).json({

      ok:
        false,

      error:
        'API endpoint မတွေ့ပါ'

    });

  }
);

/* =========================
   ERROR HANDLER
========================= */

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

    res.status(500).json({

      ok:
        false,

      error:
        error?.message ||
        'Server Error'

    });

  }
);

/* =========================
   START SERVER
========================= */

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
      'Groq: whisper-large-v3-turbo'
    );

    console.log(
      'Gemini: gemini-3.8-flash'
    );

    console.log(
      'Groq API Key: TRANSCRIPTION'
    );

    console.log(
      'Gemini API Key: MYANMAR TRANSLATION'
    );

    console.log(
      'Gemini request batching: 1 request normally / 2 max'
    );

    console.log(
      'Movie Recap: DISABLED'
    );

    console.log(
      'MP4 Render: DISABLED'
    );

  }
);
