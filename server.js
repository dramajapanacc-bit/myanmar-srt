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

const GROQ_MODEL = 'whisper-large-v3-turbo';

// Gemini model
const GEMINI_MODEL = 'gemini-3.5-flash-lite';

const ALLOWED_EXTENSIONS = new Set([
  '.mp4',
  '.mov',
  '.mkv',
  '.webm',
  '.avi',
  '.m4v',
  '.flv',
  '.wmv',
  '.mpeg',
  '.mpg'
]);

app.use(express.json({ limit: '2mb' }));
app.use(express.static(PUBLIC));


// ============================================================
// MULTER
// ============================================================

const storage = multer.diskStorage({
  destination: TMP,

  filename: (req, file, cb) => {
    const originalExt = path.extname(file.originalname || '').toLowerCase();

    const ext = ALLOWED_EXTENSIONS.has(originalExt)
      ? originalExt
      : '.mp4';

    const filename =
      `${Date.now()}-${crypto.randomBytes(8).toString('hex')}${ext}`;

    cb(null, filename);
  }
});

const upload = multer({
  storage,

  limits: {
    fileSize: MAX_BYTES
  },

  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase();

    if (!ALLOWED_EXTENSIONS.has(ext)) {
      return cb(
        new Error(
          'Unsupported video format. Please upload MP4, MOV, MKV, WEBM, AVI or another supported video file.'
        )
      );
    }

    cb(null, true);
  }
});


// ============================================================
// HELPERS
// ============================================================

function getKey(req, headerName, envName) {
  const headerValue = req.get(headerName);

  if (headerValue && headerValue.trim()) {
    return headerValue.trim();
  }

  const envValue = process.env[envName];

  if (envValue && envValue.trim()) {
    return envValue.trim();
  }

  return '';
}


function cleanText(value) {
  return String(value ?? '')
    .replace(/\r/g, ' ')
    .replace(/\n+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


function isRetryableGeminiError(error) {
  const message = String(
    error?.message ||
    error?.error?.message ||
    error ||
    ''
  ).toLowerCase();

  const status =
    error?.status ||
    error?.code ||
    error?.error?.status ||
    error?.error?.code;

  if (
    Number(status) === 429 ||
    Number(status) === 408 ||
    Number(status) === 500 ||
    Number(status) === 502 ||
    Number(status) === 503 ||
    Number(status) === 504
  ) {
    return true;
  }

  return (
    message.includes('429') ||
    message.includes('resource_exhausted') ||
    message.includes('rate limit') ||
    message.includes('rate_limit') ||
    message.includes('quota') ||
    message.includes('503') ||
    message.includes('unavailable') ||
    message.includes('high demand') ||
    message.includes('temporarily unavailable') ||
    message.includes('500') ||
    message.includes('502') ||
    message.includes('504') ||
    message.includes('timeout')
  );
}


function isQuotaError(error) {
  const message = String(
    error?.message ||
    error?.error?.message ||
    error ||
    ''
  ).toLowerCase();

  const status =
    error?.status ||
    error?.code ||
    error?.error?.status ||
    error?.error?.code;

  return (
    Number(status) === 429 ||
    message.includes('resource_exhausted') ||
    message.includes('quota exceeded') ||
    message.includes('rate limit') ||
    message.includes('rate_limit')
  );
}


// ============================================================
// FFPROBE
// ============================================================

function runProcess(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args);

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', chunk => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', chunk => {
      stderr += chunk.toString();
    });

    child.on('error', reject);

    child.on('close', code => {
      if (code === 0) {
        resolve({
          stdout,
          stderr
        });
      } else {
        reject(
          new Error(
            stderr ||
            `Process exited with code ${code}`
          )
        );
      }
    });
  });
}


async function probeVideo(filePath) {
  const result = await runProcess(
    ffprobeStatic.path,
    [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      filePath
    ]
  );

  const duration = Number(
    String(result.stdout).trim()
  );

  if (!Number.isFinite(duration)) {
    throw new Error('Could not determine video duration.');
  }

  return duration;
}


// ============================================================
// GROQ TRANSCRIPTION
// ============================================================

async function transcribeGroq(filePath, apiKey) {
  if (!apiKey) {
    throw new Error('Groq API Key မထည့်ရသေးပါ။');
  }

  const groq = new Groq({
    apiKey
  });

  const fileStream = createReadStream(filePath);

  const result = await groq.audio.transcriptions.create({
    file: fileStream,

    model: GROQ_MODEL,

    response_format: 'verbose_json',

    timestamp_granularities: [
      'segment'
    ]
  });

  return result;
}


// ============================================================
// SEGMENTS
// ============================================================

function buildSegments(transcript) {
  const sourceSegments =
    Array.isArray(transcript?.segments)
      ? transcript.segments
      : [];

  return sourceSegments
    .map((segment, index) => {
      const start = Number(segment.start);
      const end = Number(segment.end);

      const text = cleanText(segment.text);

      return {
        id: index + 1,

        start: Number.isFinite(start)
          ? start
          : 0,

        end: Number.isFinite(end)
          ? end
          : Math.max(
              0,
              Number.isFinite(start)
                ? start + 2
                : 2
            ),

        text
      };
    })
    .filter(item => item.text);
}


// ============================================================
// JSON PARSER
// ============================================================

function extractJson(text) {
  const raw = String(text || '').trim();

  if (!raw) {
    throw new Error('Gemini က empty response ပြန်ပေးပါတယ်။');
  }

  // Remove markdown code fence
  const cleaned = raw
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch {}

  // Find JSON object
  const objectStart = cleaned.indexOf('{');
  const objectEnd = cleaned.lastIndexOf('}');

  if (
    objectStart >= 0 &&
    objectEnd > objectStart
  ) {
    const objectText = cleaned.slice(
      objectStart,
      objectEnd + 1
    );

    try {
      return JSON.parse(objectText);
    } catch {}
  }

  // Find JSON array
  const arrayStart = cleaned.indexOf('[');
  const arrayEnd = cleaned.lastIndexOf(']');

  if (
    arrayStart >= 0 &&
    arrayEnd > arrayStart
  ) {
    const arrayText = cleaned.slice(
      arrayStart,
      arrayEnd + 1
    );

    try {
      return JSON.parse(arrayText);
    } catch {}
  }

  throw new Error(
    'Gemini က valid JSON မပြန်ပါ။'
  );
}


// ============================================================
// GEMINI TRANSLATION
// ============================================================

async function generateGeminiWithRetry(
  ai,
  prompt
) {
  const MAX_RETRIES = 4;

  let lastError = null;

  for (
    let attempt = 0;
    attempt <= MAX_RETRIES;
    attempt++
  ) {
    try {
      console.log(
        `Gemini request attempt ${attempt + 1}/${MAX_RETRIES + 1}`
      );

      const response =
        await ai.models.generateContent({
          model: GEMINI_MODEL,

          contents: prompt,

          config: {
            temperature: 0.2,

            responseMimeType:
              'application/json',

            maxOutputTokens: 8192
          }
        });

      return response;

    } catch (error) {
      lastError = error;

      const retryable =
        isRetryableGeminiError(error);

      if (
        !retryable ||
        attempt >= MAX_RETRIES
      ) {
        throw error;
      }

      // Exponential backoff:
      // ~1.5s → ~3s → ~6s → ~12s
      const baseDelay =
        1200 * Math.pow(2, attempt);

      const jitter =
        Math.floor(Math.random() * 500);

      const delay =
        Math.min(
          baseDelay + jitter,
          15000
        );

      console.log(
        `Gemini temporary error. Retry ${
          attempt + 1
        }/${MAX_RETRIES} after ${delay}ms`
      );

      await sleep(delay);
    }
  }

  throw lastError;
}


async function translateChunkGemini(
  chunk,
  apiKey
) {
  if (!apiKey) {
    throw new Error(
      'Gemini API Key မထည့်ရသေးပါ။'
    );
  }

  const ai = new GoogleGenAI({
    apiKey
  });

  const input = chunk.map(item => ({
    id: item.id,
    text: item.text
  }));

  const prompt = `
You are a professional subtitle translator.

Translate the following video subtitle segments into natural Myanmar Burmese.

Rules:
1. Translate ONLY the subtitle text.
2. Keep the exact id unchanged.
3. Do not add explanations.
4. Do not summarize.
5. Keep each subtitle concise and natural.
6. Preserve names, places, numbers and important terms.
7. Return valid JSON only.
8. Return an array with objects containing:
   id
   text

Source subtitles:
${JSON.stringify(input, null, 2)}
`;

  try {
    const response =
      await generateGeminiWithRetry(
        ai,
        prompt
      );

    const text =
      response?.text ??
      response?.candidates?.[0]?.content?.parts
        ?.map(part => part.text || '')
        .join('') ??
      '';

    const parsed = extractJson(text);

    let items = [];

    if (Array.isArray(parsed)) {
      items = parsed;
    } else if (
      Array.isArray(parsed?.translations)
    ) {
      items = parsed.translations;
    } else if (
      Array.isArray(parsed?.results)
    ) {
      items = parsed.results;
    } else {
      throw new Error(
        'Gemini response format မမှန်ပါ။'
      );
    }

    const byId = new Map();

    for (const item of items) {
      const id = Number(item?.id);

      if (!Number.isFinite(id)) {
        continue;
      }

      byId.set(
        id,
        cleanText(item?.text)
      );
    }

    return chunk.map(item => ({
      ...item,

      text:
        byId.get(item.id) ||
        item.text
    }));

  } catch (error) {
    if (isQuotaError(error)) {
      throw new Error(
        'Gemini Free Tier quota/rate limit ပြည့်နေပါတယ်။ ခဏစောင့်ပြီး နောက်မှ Translate ပြန်နှိပ်ပါ။'
      );
    }

    const message =
      error?.message ||
      String(error);

    throw new Error(
      `Gemini Error: ${message}`
    );
  }
}


// ============================================================
// TRANSLATE ALL
// ============================================================

async function translateAllGemini(
  segments,
  apiKey
) {
  const result = [];

  // Smaller chunks reduce request size
  // and help avoid temporary overload.
  const CHUNK_SIZE = 30;

  for (
    let i = 0;
    i < segments.length;
    i += CHUNK_SIZE
  ) {
    const chunk =
      segments.slice(
        i,
        i + CHUNK_SIZE
      );

    console.log(
      `Gemini translating ${i + 1}-${Math.min(
        i + CHUNK_SIZE,
        segments.length
      )} / ${segments.length}`
    );

    const translated =
      await translateChunkGemini(
        chunk,
        apiKey
      );

    result.push(...translated);

    // Small pause between chunks
    if (
      i + CHUNK_SIZE <
      segments.length
    ) {
      await sleep(300);
    }
  }

  return result;
}


// ============================================================
// SUBTITLE TEXT SPLIT
// ============================================================

function splitSubtitleText(
  text,
  maxChars = 42
) {
  const clean = cleanText(text);

  if (!clean) {
    return [];
  }

  if (clean.length <= maxChars) {
    return [clean];
  }

  const words = clean.split(' ');

  const lines = [];
  let current = '';

  for (const word of words) {
    const candidate =
      current
        ? `${current} ${word}`
        : word;

    if (
      candidate.length <= maxChars
    ) {
      current = candidate;
    } else {
      if (current) {
        lines.push(current);
      }

      current = word;
    }
  }

  if (current) {
    lines.push(current);
  }

  return lines;
}


// ============================================================
// TIME FORMAT
// ============================================================

function formatSrtTime(seconds) {
  let totalMs =
    Math.max(
      0,
      Math.round(
        Number(seconds || 0) * 1000
      )
    );

  const hours =
    Math.floor(
      totalMs / 3600000
    );

  totalMs %= 3600000;

  const minutes =
    Math.floor(
      totalMs / 60000
    );

  totalMs %= 60000;

  const secs =
    Math.floor(
      totalMs / 1000
    );

  const ms =
    totalMs % 1000;

  return (
    String(hours).padStart(2, '0') +
    ':' +
    String(minutes).padStart(2, '0') +
    ':' +
    String(secs).padStart(2, '0') +
    ',' +
    String(ms).padStart(3, '0')
  );
}


// ============================================================
// SRT
// ============================================================

function makeSrt(segments) {
  const blocks = [];

  let subtitleNumber = 1;

  for (const segment of segments) {
    const start =
      Number(segment.start);

    const end =
      Number(segment.end);

    if (
      !Number.isFinite(start) ||
      !Number.isFinite(end)
    ) {
      continue;
    }

    const text =
      cleanText(segment.text);

    if (!text) {
      continue;
    }

    const lines =
      splitSubtitleText(text);

    if (!lines.length) {
      continue;
    }

    blocks.push(
      [
        String(subtitleNumber++),

        `${formatSrtTime(start)} --> ${formatSrtTime(end)}`,

        lines.join('\n'),

        ''
      ].join('\n')
    );
  }

  return blocks.join('\n');
}


// ============================================================
// CLEANUP
// ============================================================

async function cleanupFile(filePath) {
  if (!filePath) {
    return;
  }

  try {
    await fs.unlink(filePath);
  } catch {}
}


// ============================================================
// HEALTH
// ============================================================

app.get('/health', (req, res) => {
  res.json({
    ok: true,
    service: 'myanmar-srt',
    geminiModel: GEMINI_MODEL,
    groqModel: GROQ_MODEL
  });
});


// ============================================================
// TRANSCRIBE
// ============================================================

app.post(
  '/api/transcribe',
  upload.single('video'),
  async (req, res) => {
    let filePath = null;

    try {
      if (!req.file) {
        return res.status(400).json({
          error: 'Video file မတွေ့ပါ။'
        });
      }

      filePath =
        req.file.path;

      const groqKey =
        getKey(
          req,
          'x-groq-api-key',
          'GROQ_API_KEY'
        );

      if (!groqKey) {
        return res.status(400).json({
          error:
            'Groq API Key မထည့်ရသေးပါ။'
        });
      }

      console.log(
        `Uploaded video: ${req.file.originalname}`
      );

      const duration =
        await probeVideo(filePath);

      console.log(
        `Video duration: ${duration}s`
      );

      if (
        duration > MAX_SECONDS
      ) {
        return res.status(400).json({
          error:
            'Video length က 5 မိနစ်ထက် မကျော်ရပါ။'
        });
      }

      const transcript =
        await transcribeGroq(
          filePath,
          groqKey
        );

      const segments =
        buildSegments(transcript);

      return res.json({
        ok: true,

        duration,

        language:
          transcript?.language ||
          null,

        text:
          cleanText(
            transcript?.text
          ),

        segments
      });

    } catch (error) {
      console.error(
        'TRANSCRIBE ERROR:',
        error
      );

      return res.status(500).json({
        error:
          error?.message ||
          'Groq transcription failed.'
      });

    } finally {
      await cleanupFile(filePath);
    }
  }
);


// ============================================================
// TRANSLATE
// ============================================================

app.post(
  '/api/translate',
  async (req, res) => {
    try {
      const geminiKey =
        getKey(
          req,
          'x-gemini-api-key',
          'GEMINI_API_KEY'
        );

      if (!geminiKey) {
        return res.status(400).json({
          error:
            'Gemini API Key မထည့်ရသေးပါ။'
        });
      }

      const segments =
        Array.isArray(req.body?.segments)
          ? req.body.segments
          : [];

      if (!segments.length) {
        return res.status(400).json({
          error:
            'Translate လုပ်ရန် transcript မရှိပါ။'
        });
      }

      const cleanedSegments =
        segments
          .map((item, index) => ({
            id:
              Number.isFinite(
                Number(item?.id)
              )
                ? Number(item.id)
                : index + 1,

            start:
              Number(item?.start) || 0,

            end:
              Number(item?.end) || 0,

            text:
              cleanText(item?.text)
          }))
          .filter(
            item => item.text
          );

      console.log(
        `Starting Gemini translation: ${cleanedSegments.length} segments`
      );

      const translated =
        await translateAllGemini(
          cleanedSegments,
          geminiKey
        );

      return res.json({
        ok: true,

        segments: translated,

        srt:
          makeSrt(translated)
      });

    } catch (error) {
      console.error(
        'TRANSLATE ERROR:',
        error
      );

      const message =
        error?.message ||
        String(error);

      return res.status(500).json({
        error: message
      });
    }
  }
);


// ============================================================
// OLD PROCESS ROUTE
// ============================================================

app.post(
  '/api/process',
  (req, res) => {
    res.status(410).json({
      error:
        'Movie Recap / MP4 Render feature has been disabled. This app is SRT only.'
    });
  }
);


// ============================================================
// DISABLED MOVIE ROUTES
// ============================================================

app.post(
  '/api/render',
  (req, res) => {
    res.status(410).json({
      error:
        'MP4 rendering is disabled. This app is SRT only.'
    });
  }
);

app.post(
  '/api/recap',
  (req, res) => {
    res.status(410).json({
      error:
        'Movie Recap is disabled. This app is SRT only.'
    });
  }
);


// ============================================================
// 404 API
// ============================================================

app.use(
  '/api',
  (req, res) => {
    res.status(404).json({
      error:
        'API endpoint not found.'
    });
  }
);


// ============================================================
// GLOBAL ERROR
// ============================================================

app.use(
  (error, req, res, next) => {
    console.error(
      'SERVER ERROR:',
      error
    );

    if (
      error?.code ===
      'LIMIT_FILE_SIZE'
    ) {
      return res.status(413).json({
        error:
          'Video file size က 300MB ထက် မကျော်ရပါ။'
      });
    }

    return res.status(500).json({
      error:
        error?.message ||
        'Server error.'
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
      `Gemini model: ${GEMINI_MODEL}`
    );

    console.log(
      `Groq model: ${GROQ_MODEL}`
    );
  }
);
