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
// MODELS
// ============================================================

const GROQ_MODEL = 'whisper-large-v3-turbo';

const GEMINI_MODEL = 'gemini-3.8-flash';

// OpenRouter FREE AUTO ROUTER
const OPENROUTER_MODEL = 'openrouter/free';


// ============================================================
// ALLOWED FILE TYPES
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
// UPLOAD
// ============================================================

const upload = multer({
  storage: multer.diskStorage({

    destination: (_req, _file, cb) => {
      cb(null, TMP);
    },

    filename: (_req, file, cb) => {

      const originalExt =
        path
          .extname(file.originalname || '')
          .toLowerCase();

      const ext =
        allowedExt.has(originalExt)
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
      path
        .extname(file.originalname || '')
        .toLowerCase();

    if (!allowedExt.has(ext)) {

      return cb(
        new Error(
          'MP4 သို့မဟုတ် WEBM video ကိုသုံးပါ။'
        )
      );
    }

    cb(null, true);
  }
});


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
// FFPROBE
// ============================================================

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

  return groq.audio.transcriptions.create({

    file:
      createReadStream(file),

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
// BUILD SEGMENTS
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
        (s, i) => ({

          id:
            i + 1,

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

  if (output.length) {

    return output;
  }

  const fullText =
    cleanText(
      result?.text
    );

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
// EXTRACT JSON
// ============================================================

function extractJson(text) {

  let raw =
    String(text || '')
      .trim();

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

    return JSON.parse(raw);

  } catch {}


  const objectStart =
    raw.indexOf('{');

  const objectEnd =
    raw.lastIndexOf('}');

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
    raw.indexOf('[');

  const arrayEnd =
    raw.lastIndexOf(']');

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
// NORMALIZE TRANSLATION
// ============================================================

function normalizeTranslationResult(
  parsed,
  segments
) {

  let list = [];


  if (Array.isArray(parsed)) {

    list = parsed;
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
    !list.length &&
    Array.isArray(
      parsed?.result
    )
  ) {

    list =
      parsed.result;
  }


  if (!list.length) {

    return null;
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
      Number.isFinite(id) &&
      translation
    ) {

      byId.set(
        id,
        translation
      );
    }
  }


  if (!byId.size) {

    return null;
  }


  const result =
    segments.map(
      s => ({

        ...s,

        translation:
          byId.get(s.id) ||
          s.text
      })
    );


  // Require ALL lines
  const missing =
    result.some(
      s =>
        !byId.has(s.id)
    );


  if (missing) {

    return null;
  }


  return result;
}


// ============================================================
// GEMINI FALLBACK DETECTION
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

    /high demand/.test(message) ||

    /internal error/.test(message)
  );
}


// ============================================================
// GEMINI TRANSLATION
// ============================================================

async function translateWithGemini(
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

The source language may be English, Chinese,
Japanese, Korean, or any other spoken language.

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


  try {

    const response =
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


    const parsed =
      extractJson(
        response?.text || ''
      );


    const translated =
      normalizeTranslationResult(
        parsed,
        segments
      );


    if (!translated) {

      throw new Error(
        'Gemini valid translation JSON မပြန်ပါ'
      );
    }


    return translated;

  } catch (error) {

    console.error(
      'GEMINI ERROR:',
      error
    );

    throw error;
  }
}


// ============================================================
// OPENROUTER FREE AUTO
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

The source language may be English, Chinese,
Japanese, Korean, or any other spoken language.

IMPORTANT:

Return ONLY valid JSON.

Required exact structure:

{
  "translations": [
    {
      "id": 1,
      "translation": "မြန်မာဘာသာပြန်"
    }
  ]
}

Rules:

- Every input line MUST have exactly one output item.
- Keep every id exactly unchanged.
- Never remove a line.
- Never merge lines.
- Never reorder lines.
- Never invent dialogue.
- Translate only spoken dialogue.
- No explanation.
- No markdown.
- No code fences.
- No comments.
- Keep names naturally.
- Keep subtitles short and natural.
- Prefer under 42 characters when possible.
- Preserve original meaning and emotion.

INPUT:

${JSON.stringify(input)}
`;


  const body = {

    model:
      OPENROUTER_MODEL,

    messages: [

      {
        role:
          'system',

        content:
          'You are a reliable Myanmar subtitle translation engine. Return valid JSON only.'
      },

      {
        role:
          'user',

        content:
          prompt
      }
    ],

    temperature:
      0.1,

    max_tokens:
      4000,

    response_format: {

      type:
        'json_object'
    },

    provider: {

      allow_fallbacks:
        true,

      require_parameters:
        true,

      data_collection:
        'deny'
    }
  };


  let lastError =
    null;


  // Retry up to 3 times
  for (
    let attempt = 1;
    attempt <= 3;
    attempt++
  ) {

    try {

      console.log(
        `🔵 OpenRouter FREE attempt ${attempt}/3`
      );


      const response =
        await fetch(
          'https://openrouter.ai/api/v1/chat/completions',
          {

            method:
              'POST',

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

            body:
              JSON.stringify(body)
          }
        );


      const raw =
        await response.text();


      console.log(
        'OPENROUTER STATUS:',
        response.status
      );


      if (!response.ok) {

        console.error(
          'OPENROUTER ERROR:',
          raw
        );


        lastError =
          new Error(
            `OpenRouter Error ${response.status}: ${raw}`
          );


        // Wait before retrying 429/5xx
        if (
          response.status === 429 ||
          response.status === 500 ||
          response.status === 502 ||
          response.status === 503 ||
          response.status === 504
        ) {

          const wait =
            attempt * 1500;

          await new Promise(
            resolve =>
              setTimeout(
                resolve,
                wait
              )
          );

          continue;
        }


        throw lastError;
      }


      let data;


      try {

        data =
          JSON.parse(raw);

      } catch {

        throw new Error(
          'OpenRouter API response JSON မဖတ်နိုင်ပါ'
        );
      }


      const content =
        data
          ?.choices
          ?.[0]
          ?.message
          ?.content;


      if (!content) {

        throw new Error(
          'OpenRouter က empty response ပြန်ပါသည်'
        );
      }


      console.log(
        'OPENROUTER CONTENT:',
        String(content).slice(
          0,
          1500
        )
      );


      const parsed =
        extractJson(
          content
        );


      const translated =
        normalizeTranslationResult(
          parsed,
          segments
        );


      if (!translated) {

        lastError =
          new Error(
            'OpenRouter valid translation JSON မပြန်ပါ'
          );


        await new Promise(
          resolve =>
            setTimeout(
              resolve,
              attempt * 1000
            )
        );

        continue;
      }


      return translated;

    } catch (error) {

      lastError =
        error;


      console.error(
        'OPENROUTER ATTEMPT ERROR:',
        error?.message ||
        error
      );


      if (
        attempt < 3
      ) {

        await new Promise(
          resolve =>
            setTimeout(
              resolve,
              attempt * 1200
            )
        );

        continue;
      }
    }
  }


  throw (
    lastError ||
    new Error(
      'OpenRouter Free translation မအောင်မြင်ပါ'
    )
  );
}


// ============================================================
// TRANSLATE CHUNK
// ============================================================

async function translateChunk(
  segments,
  geminiKey,
  openRouterKey
) {

  try {

    console.log(
      '🟢 Trying Gemini...'
    );


    const translated =
      await translateWithGemini(
        segments,
        geminiKey
      );


    console.log(
      '🟢 Gemini translation successful'
    );


    return {

      translated,

      provider:
        'gemini'
    };


  } catch (geminiError) {

    console.error(
      'Gemini failed:',
      geminiError?.message ||
      geminiError
    );


    if (
      !shouldFallbackToOpenRouter(
        geminiError
      )
    ) {

      throw geminiError;
    }


    if (!openRouterKey) {

      throw new Error(
        'Gemini မရပါ။ OPENROUTER_API_KEY မရှိပါ'
      );
    }


    console.log(
      '🔵 Falling back to OpenRouter FREE AUTO...'
    );


    const translated =
      await translateWithOpenRouter(
        segments,
        openRouterKey
      );


    console.log(
      '🔵 OpenRouter FREE AUTO successful'
    );


    return {

      translated,

      provider:
        'openrouter-free-auto'
    };
  }
}


// ============================================================
// TRANSLATE ALL
// ============================================================

async function translateAll(
  segments,
  geminiKey,
  openRouterKey
) {

  // Smaller chunks = faster and more stable
  const max = 25;

  const results = [];

  let provider =
    'gemini';


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
      `Translation chunk ${Math.floor(i / max) + 1}`
    );


    const result =
      await translateChunk(
        chunk,
        geminiKey,
        openRouterKey
      );


    results.push(
      ...result.translated
    );


    if (
      result.provider !==
      'gemini'
    ) {

      provider =
        result.provider;
    }
  }


  return {

    segments:
      results,

    provider
  };
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
    String(text || '')
      .trim();


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
        next.length > maxChars ||
        next.split(/\s+/).length > maxWords
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


    return chunks
      .filter(Boolean);
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
      ms / 3600000
    );


  const m =
    Math.floor(
      (
        ms % 3600000
      ) / 60000
    );


  const s =
    Math.floor(
      (
        ms % 60000
      ) / 1000
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
      ) =>

        `${i + 1}\n` +

        `${srtTime(s.start)} --> ${srtTime(s.end)}\n` +

        `${s.translation || s.text}\n`
    )

    .join('\n');
}


// ============================================================
// CLEANUP
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

      openRouterConfigured:
        Boolean(
          process.env.OPENROUTER_API_KEY
        ),

      groqModel:
        GROQ_MODEL,

      geminiModel:
        GEMINI_MODEL,

      openRouterModel:
        OPENROUTER_MODEL
    });
  }
);


// ============================================================
// TRANSCRIBE
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
              x =>
                x.text
            )
            .join(' '),

        transcript
      });


    } catch (error) {

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
// TRANSLATE
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
              s.end > s.start
          );


      if (
        !normalized.length
      ) {

        throw new Error(
          'ဘာသာပြန်ရန် Transcript data မမှန်ပါ'
        );
      }


      const result =
        await translateAll(
          normalized,
          geminiKey,
          openRouterKey
        );


      const subtitleSegments =
        splitTranslatedSegments(
          result.segments
        );


      res.json({

        ok:
          true,

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


      res.status(
        400
      ).json({

        ok:
          false,

        error:
          error?.message ||
          'Myanmar Translation Error'
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


      const translated =
        await translateAll(
          original,
          geminiKey,
          openRouterKey
        );


      const segments =
        splitTranslatedSegments(
          translated.segments
        );


      res.json({

        ok:
          true,

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
    (_req, res) =>
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
  (_req, res) =>
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
      `OpenRouter: ${OPENROUTER_MODEL}`
    );

    console.log(
      'Translation fallback: GEMINI -> OPENROUTER FREE AUTO'
    );

    console.log(
      'OpenRouter provider fallback: ENABLED'
    );

    console.log(
      'Movie Recap: DISABLED'
    );

    console.log(
      'MP4 Render: DISABLED'
    );
  }
);
