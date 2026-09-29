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

const PORT = Number(process.env.PORT || 10000);

const MAX_BYTES = 300 * 1024 * 1024;
const MAX_SECONDS = 5 * 60;

const ROOT = process.cwd();
const PUBLIC = path.join(ROOT, 'public');
const TMP = path.join(os.tmpdir(), 'myanmar-srt');

await fs.mkdir(TMP, { recursive: true });


// ============================================================
// ALLOWED VIDEO / AUDIO EXTENSIONS
// ============================================================

const allowedExt = new Set([
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
// MULTER UPLOAD
// IMPORTANT:
// Keep original extension because Groq checks file extension.
// ============================================================

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => {
      cb(null, TMP);
    },

    filename: (_req, file, cb) => {
      const originalExt = path
        .extname(file.originalname || '')
        .toLowerCase();

      const ext = allowedExt.has(originalExt)
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
    const ext = path
      .extname(file.originalname || '')
      .toLowerCase();

    if (!allowedExt.has(ext)) {
      return cb(
        new Error(
          'MP4 သို့မဟုတ် WEBM video ကိုသုံးပါ။ Groq မထောက်ပံ့တဲ့ format ဖြစ်ပါတယ်။'
        )
      );
    }

    cb(null, true);
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
  express.static(PUBLIC)
);


// ============================================================
// API KEY HELPER
// ============================================================

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
    throw new Error(`${label} မရှိပါ`);
  }

  return key;
}


// ============================================================
// TEXT CLEAN
// ============================================================

function cleanText(text) {
  return String(text || '')
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
// FFMPEG / FFPROBE VIDEO CHECK
// ============================================================

async function probeVideo(file) {
  const result = await new Promise(
    (resolve, reject) => {
      const child = spawn(
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
        d => {
          stdout += d.toString();
        }
      );

      child.stderr.on(
        'data',
        d => {
          stderr += d.toString();
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

  const data = JSON.parse(
    result.stdout
  );

  const duration = Number(
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
  const groq = new Groq({
    apiKey
  });

  return groq.audio.transcriptions.create({
    file: createReadStream(file),

    model:
      'whisper-large-v3-turbo',

    response_format:
      'verbose_json',

    timestamp_granularities: [
      'word',
      'segment'
    ],

    temperature: 0
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
    Array.isArray(result?.segments)
      ? result.segments
      : [];

  const output =
    segments
      .map((s, i) => ({
        id: i + 1,

        start:
          Math.max(
            0,
            Number(s.start) || 0
          ),

        end:
          Math.max(
            0,
            Number(s.end) || 0
          ),

        text:
          cleanText(s.text)
      }))
      .filter(
        s =>
          s.text &&
          s.end > s.start
      );

  if (output.length) {
    return output;
  }

  const fullText =
    cleanText(result?.text);

  if (fullText) {
    return [
      {
        id: 1,
        start: 0,
        end: duration,
        text: fullText
      }
    ];
  }

  return [];
}


// ============================================================
// JSON EXTRACTOR
// ============================================================

function extractJson(text) {
  const raw =
    String(text || '').trim();

  const tries = [
    raw,

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
      .trim()
  ];

  for (const value of tries) {
    try {
      return JSON.parse(value);
    } catch {}
  }

  const a =
    raw.indexOf('[');

  const b =
    raw.lastIndexOf(']');

  if (
    a >= 0 &&
    b > a
  ) {
    try {
      return JSON.parse(
        raw.slice(
          a,
          b + 1
        )
      );
    } catch {}
  }

  return null;
}


// ============================================================
// CHECK IF GEMINI ERROR SHOULD FALLBACK
// ============================================================

function shouldFallbackToOpenRouter(
  error
) {
  const message =
    String(
      error?.message ||
      error ||
      ''
    ).toLowerCase();

  return (
    /429/.test(message) ||
    /500/.test(message) ||
    /503/.test(message) ||
    /529/.test(message) ||
    /quota/.test(message) ||
    /resource.?exhausted/.test(message) ||
    /rate.?limit/.test(message) ||
    /unavailable/.test(message) ||
    /overloaded/.test(message) ||
    /high demand/.test(message)
  );
}


// ============================================================
// GEMINI TRANSLATION
// ============================================================

async function translateWithGemini(
  segments,
  apiKey,
  model
) {
  const ai =
    new GoogleGenAI({
      apiKey
    });

  const input =
    segments.map(s => ({
      id: s.id,
      text: s.text
    }));

  const prompt = `
You are a professional Myanmar subtitle translator.

Translate EVERY source dialogue line into natural,
concise Myanmar Unicode.

The source language may be:
English, Chinese, Japanese, Korean,
or any other spoken language.

Return ONLY a JSON array.

Each item must contain exactly:
id
translation

Rules:
- Keep the same id.
- Do not remove lines.
- Do not merge lines.
- Do not reorder lines.
- Do not invent lines.
- Translate only spoken dialogue.
- Do not add explanations.
- Do not add translator notes.
- Keep names naturally.
- Keep Myanmar subtitles short and easy to read.
- Prefer under 42 characters when possible.

INPUT:
${JSON.stringify(input)}
`;

  try {
    const response =
      await ai.models.generateContent({
        model,

        contents: prompt,

        config: {
          temperature: 0.2,

          responseMimeType:
            'application/json'
        }
      });

    const parsed =
      extractJson(
        response?.text || ''
      );

    const list =
      Array.isArray(parsed)
        ? parsed
        : [];

    if (!list.length) {
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
        Number(item?.id);

      const translation =
        String(
          item?.translation || ''
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

  } catch (error) {

    console.error(
      'GEMINI ERROR:',
      error
    );

    if (
      shouldFallbackToOpenRouter(
        error
      )
    ) {
      throw new Error(
        `OPENROUTER_FALLBACK_REQUIRED: ${
          error?.message || error
        }`
      );
    }

    throw error;
  }
}


// ============================================================
// OPENROUTER FREE TRANSLATION
// ============================================================

async function translateWithOpenRouter(
  segments,
  apiKey
) {
  if (!apiKey) {
    throw new Error(
      'OPENROUTER_API_KEY မရှိပါ'
    );
  }

  const input =
    segments.map(s => ({
      id: s.id,
      text: s.text
    }));

  const prompt = `
You are a professional Myanmar subtitle translator.

Translate EVERY source dialogue line into natural,
clear, concise Myanmar Unicode.

The source language may be:
English, Chinese, Japanese, Korean,
or any other language.

Return ONLY valid JSON.

Required format:
[
  {
    "id": 1,
    "translation": "မြန်မာဘာသာပြန်"
  }
]

Rules:
- Keep every id exactly the same.
- Do not remove any line.
- Do not merge lines.
- Do not reorder lines.
- Do not invent dialogue.
- Translate only the dialogue.
- No explanation.
- No markdown.
- No comments.
- Keep names naturally.
- Make subtitles easy to read.
- Prefer short Myanmar subtitles.
- Preserve the meaning and emotion of the original dialogue.

INPUT:
${JSON.stringify(input)}
`;

  const response =
    await fetch(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        method: 'POST',

        headers: {
          'Authorization':
            `Bearer ${apiKey}`,

          'Content-Type':
            'application/json',

          'HTTP-Referer':
            'https://myanmar-srt-txdb.onrender.com',

          'X-Title':
            'Myanmar SRT'
        },

        body: JSON.stringify({
          model:
            'openrouter/free',

          messages: [
            {
              role: 'user',
              content: prompt
            }
          ],

          temperature: 0.2
        })
      }
    );

  const raw =
    await response.text();

  if (!response.ok) {
    throw new Error(
      `OpenRouter Error ${response.status}: ${raw}`
    );
  }

  let data;

  try {
    data =
      JSON.parse(raw);
  } catch {
    throw new Error(
      'OpenRouter response JSON မဖတ်နိုင်ပါ'
    );
  }

  const content =
    data?.choices?.[0]?.message?.content;

  if (!content) {
    throw new Error(
      'OpenRouter က translation မပြန်ပါ'
    );
  }

  const parsed =
    extractJson(content);

  const list =
    Array.isArray(parsed)
      ? parsed
      : [];

  if (!list.length) {
    throw new Error(
      'OpenRouter က valid translation JSON မပြန်ပါ'
    );
  }

  const byId =
    new Map();

  for (
    const item of list
  ) {
    const id =
      Number(item?.id);

    const translation =
      String(
        item?.translation || ''
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


// ============================================================
// TRANSLATE CHUNK
// GEMINI FIRST
// OPENROUTER FALLBACK
// ============================================================

async function translateChunk(
  segments,
  geminiKey,
  geminiModel,
  openRouterKey
) {

  // ----------------------------------------------------------
  // 1. TRY GEMINI
  // ----------------------------------------------------------

  try {

    const translated =
      await translateWithGemini(
        segments,
        geminiKey,
        geminiModel
      );

    return {
      translated,
      provider: 'gemini'
    };

  } catch (geminiError) {

    console.error(
      'Gemini failed.'
    );

    console.error(
      geminiError?.message ||
      geminiError
    );


    // --------------------------------------------------------
    // 2. TRY OPENROUTER
    // --------------------------------------------------------

    if (
      shouldFallbackToOpenRouter(
        geminiError
      ) ||
      String(
        geminiError?.message || ''
      ).includes(
        'OPENROUTER_FALLBACK_REQUIRED'
      )
    ) {

      if (!openRouterKey) {

        throw new Error(
          'Gemini မရပါ။ OpenRouter API Key လည်း မရှိပါ။'
        );

      }

      console.log(
        '🔵 Falling back to OpenRouter Free...'
      );

      try {

        const translated =
          await translateWithOpenRouter(
            segments,
            openRouterKey
          );

        return {
          translated,
          provider:
            'openrouter'
        };

      } catch (openRouterError) {

        console.error(
          'OPENROUTER ERROR:',
          openRouterError
        );

        throw new Error(
          `Gemini မရပါ။ OpenRouter လည်း မအောင်မြင်ပါ။ ${
            openRouterError?.message ||
            openRouterError
          }`
        );
      }
    }

    throw geminiError;
  }
}


// ============================================================
// TRANSLATE ALL
// ============================================================

async function translateAll(
  segments,
  geminiKey,
  geminiModel =
    'gemini-3.8-flash',
  openRouterKey
) {

  // Keep requests small.
  // This also makes fallback safer.

  const max = 60;

  const chunks = [];

  for (
    let i = 0;
    i < segments.length;
    i += max
  ) {
    chunks.push(
      segments.slice(
        i,
        i + max
      )
    );
  }

  const results = [];

  let usedProvider =
    'gemini';

  for (
    const chunk of chunks
  ) {

    const result =
      await translateChunk(
        chunk,
        geminiKey,
        geminiModel,
        openRouterKey
      );

    results.push(
      ...result.translated
    );

    if (
      result.provider ===
      'openrouter'
    ) {
      usedProvider =
        'openrouter';
    }
  }

  return {
    segments: results,
    provider:
      usedProvider
  };
}


// ============================================================
// SUBTITLE SPLITTER
// ============================================================

function splitSubtitle(
  text,
  maxChars = 42,
  maxWords = 10
) {
  const value =
    String(text || '').trim();

  if (!value) {
    return [];
  }

  if (
    value.length <= maxChars &&
    value.split(/\s+/).length <= maxWords
  ) {
    return [value];
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

      current = word;

    } else {

      current = next;

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

    const chunks = [];

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

    return chunks.filter(Boolean);
  }

  return parts;
}


// ============================================================
// SPLIT TRANSLATED SEGMENTS
// ============================================================

function splitTranslatedSegments(
  segments
) {
  const output = [];

  let id = 1;

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
      splitSubtitle(text);

    if (
      parts.length <= 1
    ) {

      output.push({
        id: id++,
        start: s.start,
        end: s.end,
        text,
        translation: text
      });

      continue;
    }

    const total =
      parts.reduce(
        (sum, p) =>
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
                  ) / total
                );

        output.push({
          id: id++,

          start:
            cursor,

          end:
            Math.min(
              end,
              s.end
            ),

          text: part,

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
        Number(seconds || 0) *
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


// ============================================================
// MAKE SRT
// ============================================================

function makeSrt(
  segments
) {
  return segments
    .map(
      (s, i) =>
        `${i + 1}\n` +
        `${srtTime(s.start)} --> ` +
        `${srtTime(s.end)}\n` +
        `${s.translation || s.text}\n`
    )
    .join('\n');
}


// ============================================================
// CLEAN TEMP FILE
// ============================================================

function cleanup(
  file
) {
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


// ============================================================
// HEALTH CHECK
// ============================================================

app.get(
  '/api/health',
  (_req, res) => {

    res.json({
      ok: true,

      service:
        'myanmar-srt',

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

      openRouterConfigured:
        Boolean(
          process.env.OPENROUTER_API_KEY
        ),

      groqModel:
        'whisper-large-v3-turbo',

      geminiModel:
        'gemini-3.8-flash',

      openRouterModel:
        'openrouter/free'
    });
  }
);


// ============================================================
// TRANSCRIBE API
// ============================================================

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
        ok: true,

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
        ok: false,

        error:
          error?.message ||
          'Groq Transcript Error'
      });

    } finally {

      cleanup(file);

    }
  }
);


// ============================================================
// TRANSLATE API
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

      const openRouterKey =
        String(
          req.headers[
            'x-openrouter-api-key'
          ] ||
          req.body?.openRouterApiKey ||
          process.env.OPENROUTER_API_KEY ||
          ''
        ).trim();

      const input =
        Array.isArray(
          req.body?.transcript
        )
          ? req.body.transcript
          : [];

      if (!input.length) {

        throw new Error(
          'Transcript မရှိပါ'
        );
      }

      const normalized =
        input
          .map(
            (s, i) => ({
              id: i + 1,

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

      const result =
        await translateAll(
          normalized,
          geminiKey,
          model,
          openRouterKey
        );

      const subtitleSegments =
        splitTranslatedSegments(
          result.segments
        );

      res.json({
        ok: true,

        provider:
          result.provider,

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
        ok: false,

        error:
          error?.message ||
          'Myanmar Translation Error'
      });
    }
  }
);


// ============================================================
// OLD PROCESS API
// KEPT FOR COMPATIBILITY
// ============================================================

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

      const openRouterKey =
        String(
          req.headers[
            'x-openrouter-api-key'
          ] ||
          req.body?.openRouterApiKey ||
          process.env.OPENROUTER_API_KEY ||
          ''
        ).trim();

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
          model,
          openRouterKey
        );

      const segments =
        splitTranslatedSegments(
          translated.segments
        );

      res.json({
        ok: true,

        provider:
          translated.provider,

        duration,

        segments,

        srt:
          makeSrt(
            segments
          )
      });

    } catch (error) {

      console.error(
        'PROCESS ERROR:',
        error
      );

      res.status(400).json({
        ok: false,

        error:
          error?.message ||
          String(error)
      });

    } finally {

      cleanup(file);

    }
  }
);


// ============================================================
// DISABLED MOVIE RECAP / RENDER ROUTES
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
    (_req, res) =>
      res.status(410).json({
        ok: false,

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
  (_req, res) =>
    res.status(404).json({
      ok: false,

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
      return next(error);
    }

    res.status(500).json({
      ok: false,

      error:
        error?.message ||
        'Server Error'
    });
  }
);


// ============================================================
// START SERVER
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
      'Groq: whisper-large-v3-turbo'
    );

    console.log(
      'Gemini: gemini-3.8-flash'
    );

    console.log(
      'OpenRouter: openrouter/free'
    );

    console.log(
      'Groq API Key: TRANSCRIPTION'
    );

    console.log(
      'Gemini API Key: MYANMAR TRANSLATION'
    );

    console.log(
      'OpenRouter API Key: FALLBACK TRANSLATION'
    );

    console.log(
      'Translation fallback: GEMINI -> OPENROUTER'
    );

    console.log(
      'Movie Recap: DISABLED'
    );

    console.log(
      'MP4 Render: DISABLED'
    );
  }
);
