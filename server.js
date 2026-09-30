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

import ffmpegStatic from 'ffmpeg-static';
import ffprobeStatic from 'ffprobe-static';


// ============================================================
// APP
// ============================================================

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
  {
    recursive: true
  }
);


// ============================================================
// MODELS
// ============================================================

// Accuracy-first Whisper model
const GROQ_MODEL =
  'whisper-large-v3';

// Gemini translation model
const GEMINI_MODEL =
  'gemini-3.5-flash-lite';


// ============================================================
// EXPRESS
// ============================================================

app.use(
  express.json({
    limit: '2mb'
  })
);

app.use(
  express.static(PUBLIC)
);


// ============================================================
// ALLOWED VIDEO TYPES
// ============================================================

const ALLOWED_EXTENSIONS =
  new Set([
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


// ============================================================
// MULTER
// ============================================================

const storage =
  multer.diskStorage({

    destination:
      TMP,

    filename:
      (req, file, cb) => {

        const originalExt =
          path.extname(
            file.originalname || ''
          ).toLowerCase();

        const ext =
          ALLOWED_EXTENSIONS.has(
            originalExt
          )
            ? originalExt
            : '.mp4';

        const filename =
          `${Date.now()}-${crypto.randomBytes(8).toString('hex')}${ext}`;

        cb(
          null,
          filename
        );
      }

  });


const upload =
  multer({

    storage,

    limits: {
      fileSize:
        MAX_BYTES
    },

    fileFilter:
      (req, file, cb) => {

        const ext =
          path.extname(
            file.originalname || ''
          ).toLowerCase();

        if (
          !ALLOWED_EXTENSIONS.has(
            ext
          )
        ) {

          return cb(
            new Error(
              'Unsupported video format.'
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
// HELPERS
// ============================================================

function getKey(
  req,
  headerName,
  envName
) {

  const headerValue =
    req.get(
      headerName
    );

  if (
    headerValue &&
    headerValue.trim()
  ) {

    return headerValue.trim();
  }

  const envValue =
    process.env[
      envName
    ];

  if (
    envValue &&
    envValue.trim()
  ) {

    return envValue.trim();
  }

  return '';
}


function cleanText(
  value
) {

  return String(
    value ?? ''
  )
    .replace(
      /\r/g,
      ' '
    )
    .replace(
      /\n+/g,
      ' '
    )
    .replace(
      /\s+/g,
      ' '
    )
    .trim();
}


function sleep(
  ms
) {

  return new Promise(
    resolve =>
      setTimeout(
        resolve,
        ms
      )
  );
}


// ============================================================
// RUN PROCESS
// ============================================================

function runProcess(
  command,
  args
) {

  return new Promise(
    (
      resolve,
      reject
    ) => {

      const child =
        spawn(
          command,
          args
        );

      let stdout =
        '';

      let stderr =
        '';

      child.stdout.on(
        'data',
        chunk => {

          stdout +=
            chunk.toString();

        }
      );

      child.stderr.on(
        'data',
        chunk => {

          stderr +=
            chunk.toString();

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
                `Process exited with code ${code}`
              )
            );
          }

        }
      );

    }
  );
}


// ============================================================
// VIDEO DURATION
// ============================================================

async function probeVideo(
  filePath
) {

  const result =
    await runProcess(
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

  const duration =
    Number(
      String(
        result.stdout
      ).trim()
    );

  if (
    !Number.isFinite(
      duration
    )
  ) {

    throw new Error(
      'Video duration ကို မဖတ်နိုင်ပါ။'
    );
  }

  return duration;
}


// ============================================================
// EXTRACT AUDIO
//
// Video
//   ↓
// WAV
//   ↓
// 16kHz
// Mono
// PCM
//
// WAV is used to preserve speech quality.
// ============================================================

async function extractAudio(
  videoPath
) {

  const audioPath =
    path.join(
      TMP,
      `${Date.now()}-${crypto.randomBytes(8).toString('hex')}.wav`
    );

  console.log(
    'Extracting audio from video...'
  );

  await runProcess(
    ffmpegStatic,
    [
      '-y',

      '-i',
      videoPath,

      // First audio track
      '-map',
      '0:a:0',

      // Speech-friendly format
      '-ar',
      '16000',

      '-ac',
      '1',

      // PCM WAV
      '-c:a',
      'pcm_s16le',

      audioPath
    ]
  );

  console.log(
    `Audio extracted: ${audioPath}`
  );

  return audioPath;
}


// ============================================================
// GROQ TRANSCRIPTION
//
// IMPORTANT:
// We request BOTH:
//
//   word
//   segment
//
// Word timestamps are then used to create
// subtitle timelines.
//
// This means subtitle timing is based on
// actual spoken words rather than guessed
// fixed durations.
// ============================================================

async function transcribeGroq(
  audioPath,
  apiKey
) {

  if (!apiKey) {

    throw new Error(
      'Groq API Key မထည့်ရသေးပါ။'
    );
  }

  const groq =
    new Groq({
      apiKey
    });

  console.log(
    'Sending extracted audio to Groq Whisper...'
  );

  const fileStream =
    createReadStream(
      audioPath
    );

  const result =
    await groq.audio.transcriptions.create({

      file:
        fileStream,

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

  console.log(
    `Whisper language: ${
      result?.language ||
      'unknown'
    }`
  );

  console.log(
    `Whisper raw segments: ${
      Array.isArray(
        result?.segments
      )
        ? result.segments.length
        : 0
    }`
  );

  console.log(
    `Whisper raw words: ${
      Array.isArray(
        result?.words
      )
        ? result.words.length
        : 0
    }`
  );

  return result;
}


// ============================================================
// NORMALIZE WORD
// ============================================================

function normalizeWord(
  item
) {

  if (!item) {
    return null;
  }

  const word =
    cleanText(
      item.word
    );

  const start =
    Number(
      item.start
    );

  const end =
    Number(
      item.end
    );

  if (
    !word ||
    !Number.isFinite(
      start
    ) ||
    !Number.isFinite(
      end
    )
  ) {

    return null;
  }

  return {

    word,

    start,

    end

  };
}


// ============================================================
// CHECK SENTENCE END
// ============================================================

function endsSentence(
  text
) {

  const value =
    String(
      text || ''
    ).trim();

  if (!value) {
    return false;
  }

  return /[.!?。！？！？…]+$/.test(
    value
  );
}


// ============================================================
// BUILD PRECISE SUBTITLE SEGMENTS
//
// Uses word timestamps.
//
// A subtitle ends when:
//
// 1. Sentence punctuation appears
// 2. There is a natural speech pause
// 3. Cue becomes too long
// 4. Cue becomes too long in time
// 5. Too many words are inside one cue
//
// Start/end timestamps ALWAYS come from
// the actual first/last spoken words.
// ============================================================

function buildPreciseSegments(
  transcript
) {

  const rawWords =
    Array.isArray(
      transcript?.words
    )
      ? transcript.words
      : [];

  const words =
    rawWords
      .map(
        normalizeWord
      )
      .filter(Boolean);


  // ----------------------------------------------------------
  // WORD TIMESTAMPS AVAILABLE
  // ----------------------------------------------------------

  if (
    words.length
  ) {

    const result = [];

    let currentWords = [];

    let currentStart =
      null;

    let currentEnd =
      null;


    function flush() {

      if (
        !currentWords.length
      ) {
        return;
      }

      const text =
        currentWords
          .map(
            item =>
              item.word
          )
          .join(' ')
          .replace(
            /\s+([,.!?;:，。！？；：])/g,
            '$1'
          )
          .trim();

      if (
        text
      ) {

        result.push({

          id:
            result.length + 1,

          start:
            currentStart,

          end:
            currentEnd,

          text

        });
      }

      currentWords =
        [];

      currentStart =
        null;

      currentEnd =
        null;
    }


    for (
      let i = 0;
      i < words.length;
      i++
    ) {

      const word =
        words[i];

      const previous =
        words[i - 1];

      const gap =
        previous
          ? Math.max(
              0,
              word.start -
              previous.end
            )
          : 0;


      // ------------------------------------------------------
      // Start new cue
      // ------------------------------------------------------

      if (
        !currentWords.length
      ) {

        currentStart =
          word.start;

        currentEnd =
          word.end;

        currentWords.push(
          word
        );

        continue;
      }


      const currentText =
        currentWords
          .map(
            item =>
              item.word
          )
          .join(' ')
          .replace(
            /\s+([,.!?;:，。！？；：])/g,
            '$1'
          )
          .trim();


      const candidateText =
        `${currentText} ${word.word}`
          .replace(
            /\s+([,.!?;:，。！？；：])/g,
            '$1'
          )
          .trim();


      const currentDuration =
        word.end -
        currentStart;


      // ------------------------------------------------------
      // Split conditions
      // ------------------------------------------------------

      const naturalPause =
        gap >= 0.65;

      const sentenceFinished =
        endsSentence(
          currentText
        );

      const tooLong =
        currentDuration >= 5.5;

      const tooManyChars =
        candidateText.length > 58;

      const tooManyWords =
        currentWords.length >= 12;


      if (
        naturalPause ||
        sentenceFinished ||
        tooLong ||
        tooManyChars ||
        tooManyWords
      ) {

        flush();

        currentStart =
          word.start;

        currentEnd =
          word.end;

        currentWords.push(
          word
        );

      } else {

        currentWords.push(
          word
        );

        currentEnd =
          word.end;
      }

    }


    flush();


    console.log(
      `Precise subtitle cues created: ${result.length}`
    );

    return result;
  }


  // ----------------------------------------------------------
  // FALLBACK TO SEGMENT TIMESTAMPS
  // ----------------------------------------------------------

  const rawSegments =
    Array.isArray(
      transcript?.segments
    )
      ? transcript.segments
      : [];


  const fallback =
    rawSegments
      .map(
        (
          segment,
          index
        ) => {

          const start =
            Number(
              segment.start
            );

          const end =
            Number(
              segment.end
            );

          const text =
            cleanText(
              segment.text
            );

          if (
            !text ||
            !Number.isFinite(
              start
            ) ||
            !Number.isFinite(
              end
            )
          ) {

            return null;
          }

          return {

            id:
              index + 1,

            start,

            end,

            text

          };

        }
      )
      .filter(Boolean);


  console.log(
    `Fallback segment cues created: ${fallback.length}`
  );

  return fallback;
}


// ============================================================
// GEMINI ERROR CHECK
// ============================================================

function isRetryableGeminiError(
  error
) {

  const message =
    String(
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
    Number(status) === 408 ||
    Number(status) === 429 ||
    Number(status) === 500 ||
    Number(status) === 502 ||
    Number(status) === 503 ||
    Number(status) === 504
  ) {

    return true;
  }


  return (

    message.includes('429') ||

    message.includes(
      'resource_exhausted'
    ) ||

    message.includes(
      'rate limit'
    ) ||

    message.includes(
      'rate_limit'
    ) ||

    message.includes(
      '503'
    ) ||

    message.includes(
      'unavailable'
    ) ||

    message.includes(
      'high demand'
    ) ||

    message.includes(
      'temporarily unavailable'
    ) ||

    message.includes(
      '500'
    ) ||

    message.includes(
      '502'
    ) ||

    message.includes(
      '504'
    ) ||

    message.includes(
      'timeout'
    )
  );
}


function isQuotaError(
  error
) {

  const message =
    String(
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

    message.includes(
      'resource_exhausted'
    ) ||

    message.includes(
      'quota exceeded'
    ) ||

    message.includes(
      'rate limit'
    ) ||

    message.includes(
      'rate_limit'
    )
  );
}


// ============================================================
// GEMINI REQUEST WITH RETRY
// ============================================================

async function generateGeminiWithRetry(
  ai,
  prompt
) {

  const MAX_RETRIES =
    4;

  let lastError =
    null;


  for (
    let attempt = 0;
    attempt <= MAX_RETRIES;
    attempt++
  ) {

    try {

      console.log(
        `Gemini attempt ${
          attempt + 1
        }/${MAX_RETRIES + 1}`
      );


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
              'application/json',

            maxOutputTokens:
              8192

          }

        });


      return response;

    } catch (
      error
    ) {

      lastError =
        error;


      if (
        !isRetryableGeminiError(
          error
        ) ||
        attempt >=
          MAX_RETRIES
      ) {

        throw error;
      }


      const baseDelay =
        1200 *
        Math.pow(
          2,
          attempt
        );

      const jitter =
        Math.floor(
          Math.random() *
          500
        );

      const delay =
        Math.min(
          baseDelay +
          jitter,
          15000
        );


      console.log(
        `Gemini temporary error. Retry after ${delay}ms`
      );


      await sleep(
        delay
      );
    }
  }


  throw lastError;
}


// ============================================================
// JSON PARSER
// ============================================================

function extractJson(
  text
) {

  const raw =
    String(
      text || ''
    ).trim();


  if (!raw) {

    throw new Error(
      'Gemini က empty response ပြန်ပေးပါတယ်။'
    );
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
        /\s*```$/i,
        ''
      )
      .trim();


  try {

    return JSON.parse(
      cleaned
    );

  } catch {}


  const objectStart =
    cleaned.indexOf(
      '{'
    );

  const objectEnd =
    cleaned.lastIndexOf(
      '}'
    );


  if (
    objectStart >= 0 &&
    objectEnd >
      objectStart
  ) {

    try {

      return JSON.parse(
        cleaned.slice(
          objectStart,
          objectEnd + 1
        )
      );

    } catch {}
  }


  const arrayStart =
    cleaned.indexOf(
      '['
    );

  const arrayEnd =
    cleaned.lastIndexOf(
      ']'
    );


  if (
    arrayStart >= 0 &&
    arrayEnd >
      arrayStart
  ) {

    try {

      return JSON.parse(
        cleaned.slice(
          arrayStart,
          arrayEnd + 1
        )
      );

    } catch {}
  }


  throw new Error(
    'Gemini က valid JSON မပြန်ပါ။'
  );
}


// ============================================================
// GEMINI TRANSLATE ONE CHUNK
// ============================================================

async function translateChunkGemini(
  chunk,
  apiKey
) {

  if (!apiKey) {

    throw new Error(
      'Gemini API Key မထည့်ရသေးပါ။'
    );
  }


  const ai =
    new GoogleGenAI({
      apiKey
    });


  const input =
    chunk.map(
      item => ({

        id:
          item.id,

        text:
          item.text

      })
    );


  const prompt = `
You are a professional subtitle translator.

Translate the following spoken dialogue into natural Myanmar Burmese.

IMPORTANT RULES:

1. Translate ONLY the spoken dialogue.
2. Keep every id exactly unchanged.
3. Do not remove any item.
4. Do not merge items.
5. Do not create new items.
6. Do not summarize.
7. Do not explain anything.
8. Keep names, places and important terms.
9. Make the Myanmar subtitle natural and easy to read.
10. Return JSON only.
11. Return an array.
12. Every input id must appear exactly once.
13. Each object must contain:
    id
    text

SOURCE:
${JSON.stringify(
  input,
  null,
  2
)}
`;


  try {

    const response =
      await generateGeminiWithRetry(
        ai,
        prompt
      );


    const text =
      response?.text ??
      response
        ?.candidates?.[0]
        ?.content?.parts
        ?.map(
          part =>
            part.text || ''
        )
        .join('') ??
      '';


    const parsed =
      extractJson(
        text
      );


    let items = [];


    if (
      Array.isArray(
        parsed
      )
    ) {

      items =
        parsed;

    } else if (
      Array.isArray(
        parsed?.translations
      )
    ) {

      items =
        parsed.translations;

    } else if (
      Array.isArray(
        parsed?.results
      )
    ) {

      items =
        parsed.results;

    } else {

      throw new Error(
        'Gemini response format မမှန်ပါ။'
      );
    }


    const byId =
      new Map();


    for (
      const item of items
    ) {

      const id =
        Number(
          item?.id
        );


      if (
        !Number.isFinite(
          id
        )
      ) {

        continue;
      }


      byId.set(
        id,
        cleanText(
          item?.text
        )
      );
    }


    return chunk.map(
      item => ({

        ...item,

        text:
          byId.get(
            item.id
          ) ||
          item.text

      })
    );


  } catch (
    error
  ) {

    if (
      isQuotaError(
        error
      )
    ) {

      throw new Error(
        'Gemini Free Tier quota/rate limit ပြည့်နေပါတယ်။ ခဏစောင့်ပြီး နောက်မှ Translate ပြန်နှိပ်ပါ။'
      );
    }


    throw new Error(
      `Gemini Error: ${
        error?.message ||
        String(error)
      }`
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

  const result =
    [];

  const CHUNK_SIZE =
    30;


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
      `Gemini translating ${
        i + 1
      }-${
        Math.min(
          i +
            CHUNK_SIZE,
          segments.length
        )
      } / ${
        segments.length
      }`
    );


    const translated =
      await translateChunkGemini(
        chunk,
        apiKey
      );


    result.push(
      ...translated
    );


    if (
      i +
        CHUNK_SIZE <
      segments.length
    ) {

      await sleep(
        300
      );
    }
  }


  return result;
}


// ============================================================
// SRT TIME
// ============================================================

function formatSrtTime(
  seconds
) {

  let totalMs =
    Math.max(
      0,
      Math.round(
        Number(
          seconds || 0
        ) *
        1000
      )
    );


  const hours =
    Math.floor(
      totalMs /
      3600000
    );


  totalMs %=
    3600000;


  const minutes =
    Math.floor(
      totalMs /
      60000
    );


  totalMs %=
    60000;


  const secs =
    Math.floor(
      totalMs /
      1000
    );


  const ms =
    totalMs %
    1000;


  return (

    String(hours)
      .padStart(
        2,
        '0'
      ) +

    ':' +

    String(minutes)
      .padStart(
        2,
        '0'
      ) +

    ':' +

    String(secs)
      .padStart(
        2,
        '0'
      ) +

    ',' +

    String(ms)
      .padStart(
        3,
        '0'
      )

  );
}


// ============================================================
// SPLIT SUBTITLE TEXT
//
// IMPORTANT:
// Splitting text does NOT change the original
// speech timing in the main segment.
//
// We keep one timing block per speech cue.
// ============================================================

function splitSubtitleText(
  text,
  maxChars = 48
) {

  const clean =
    cleanText(
      text
    );


  if (!clean) {
    return [];
  }


  if (
    clean.length <=
    maxChars
  ) {

    return [
      clean
    ];
  }


  const words =
    clean.split(
      ' '
    );


  const lines =
    [];

  let current =
    '';


  for (
    const word of words
  ) {

    const candidate =
      current
        ? `${current} ${word}`
        : word;


    if (
      candidate.length <=
      maxChars
    ) {

      current =
        candidate;

    } else {

      if (
        current
      ) {

        lines.push(
          current
        );
      }


      current =
        word;
    }
  }


  if (
    current
  ) {

    lines.push(
      current
    );
  }


  return lines;
}


// ============================================================
// MAKE SRT
// ============================================================

function makeSrt(
  segments
) {

  const blocks =
    [];

  let number =
    1;


  for (
    const segment of segments
  ) {

    const start =
      Number(
        segment.start
      );

    const end =
      Number(
        segment.end
      );

    const text =
      cleanText(
        segment.text
      );


    if (
      !Number.isFinite(
        start
      ) ||
      !Number.isFinite(
        end
      ) ||
      !text
    ) {

      continue;
    }


    const lines =
      splitSubtitleText(
        text
      );


    if (
      !lines.length
    ) {

      continue;
    }


    blocks.push(

      [

        String(
          number++
        ),

        `${formatSrtTime(
          start
        )} --> ${formatSrtTime(
          end
        )}`,

        lines.join(
          '\n'
        ),

        ''

      ].join(
        '\n'
      )

    );
  }


  return blocks.join(
    '\n'
  );
}


// ============================================================
// CLEAN FILE
// ============================================================

async function cleanupFile(
  filePath
) {

  if (!filePath) {
    return;
  }


  try {

    await fs.unlink(
      filePath
    );

  } catch {}
}


// ============================================================
// HEALTH
// ============================================================

app.get(
  '/health',
  (req, res) => {

    res.json({

      ok:
        true,

      service:
        'myanmar-srt',

      groqModel:
        GROQ_MODEL,

      geminiModel:
        GEMINI_MODEL,

      audioExtraction:
        true,

      preciseWordTimeline:
        true

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

    let videoPath =
      null;

    let audioPath =
      null;


    try {

      if (
        !req.file
      ) {

        return res
          .status(400)
          .json({

            error:
              'Video file မတွေ့ပါ။'

          });
      }


      videoPath =
        req.file.path;


      const groqKey =
        getKey(
          req,
          'x-groq-api-key',
          'GROQ_API_KEY'
        );


      if (!groqKey) {

        return res
          .status(400)
          .json({

            error:
              'Groq API Key မထည့်ရသေးပါ။'

          });
      }


      console.log(
        `Uploaded video: ${
          req.file.originalname
        }`
      );


      // ------------------------------------------------------
      // VIDEO DURATION
      // ------------------------------------------------------

      const duration =
        await probeVideo(
          videoPath
        );


      console.log(
        `Video duration: ${
          duration
        }s`
      );


      if (
        duration >
        MAX_SECONDS
      ) {

        return res
          .status(400)
          .json({

            error:
              'Video length က 5 မိနစ်ထက် မကျော်ရပါ။'

          });
      }


      // ------------------------------------------------------
      // EXTRACT AUDIO
      // ------------------------------------------------------

      audioPath =
        await extractAudio(
          videoPath
        );


      // ------------------------------------------------------
      // WHISPER
      // ------------------------------------------------------

      const transcript =
        await transcribeGroq(
          audioPath,
          groqKey
        );


      // ------------------------------------------------------
      // PRECISE WORD TIMELINE
      // ------------------------------------------------------

      const segments =
        buildPreciseSegments(
          transcript
        );


      if (
        !segments.length
      ) {

        return res
          .status(400)
          .json({

            error:
              'Groq က စကားပြော Transcript မရပါ။'

          });
      }


      console.log(
        `Final subtitle segments: ${
          segments.length
        }`
      );


      // Print first few cues for debugging
      console.log(
        'First subtitle cues:',
        JSON.stringify(
          segments.slice(
            0,
            5
          ),
          null,
          2
        )
      );


      return res.json({

        ok:
          true,

        duration,

        language:
          transcript?.language ||
          null,

        // Full original text
        text:
          cleanText(
            transcript?.text
          ) ||
          segments
            .map(
              item =>
                item.text
            )
            .join(
              ' '
            ),

        // Precise speech timeline
        segments

      });


    } catch (
      error
    ) {

      console.error(
        'TRANSCRIBE ERROR:',
        error
      );


      return res
        .status(500)
        .json({

          error:
            error?.message ||
            'Transcription failed.'

        });


    } finally {

      await cleanupFile(
        audioPath
      );

      await cleanupFile(
        videoPath
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
          'GEMINI_API_KEY'
        );


      if (!geminiKey) {

        return res
          .status(400)
          .json({

            error:
              'Gemini API Key မထည့်ရသေးပါ။'

          });
      }


      const segments =
        Array.isArray(
          req.body?.segments
        )
          ? req.body.segments
          : [];


      if (
        !segments.length
      ) {

        return res
          .status(400)
          .json({

            error:
              'Translate လုပ်ရန် transcript မရှိပါ။'

          });
      }


      const cleanedSegments =
        segments
          .map(
            (
              item,
              index
            ) => ({

              id:
                Number.isFinite(
                  Number(
                    item?.id
                  )
                )
                  ? Number(
                      item.id
                    )
                  : index + 1,

              start:
                Number(
                  item?.start
                ) || 0,

              end:
                Number(
                  item?.end
                ) || 0,

              text:
                cleanText(
                  item?.text
                )

            })
          )
          .filter(
            item =>
              item.text
          );


      if (
        !cleanedSegments.length
      ) {

        return res
          .status(400)
          .json({

            error:
              'Transcript စာသား မရှိပါ။'

          });
      }


      console.log(
        `Starting Gemini translation: ${
          cleanedSegments.length
        } segments`
      );


      const translated =
        await translateAllGemini(
          cleanedSegments,
          geminiKey
        );


      return res.json({

        ok:
          true,

        // Timing is preserved exactly
        // from Groq.
        segments:
          translated,

        srt:
          makeSrt(
            translated
          )

      });


    } catch (
      error
    ) {

      console.error(
        'TRANSLATE ERROR:',
        error
      );


      return res
        .status(500)
        .json({

          error:
            error?.message ||
            String(error)

        });
    }
  }
);


// ============================================================
// DISABLED OLD ROUTES
// ============================================================

app.post(
  '/api/process',
  (req, res) => {

    res
      .status(410)
      .json({

        error:
          'Movie Recap / MP4 Render feature has been disabled. This app is SRT only.'

      });
  }
);


app.post(
  '/api/render',
  (req, res) => {

    res
      .status(410)
      .json({

        error:
          'MP4 rendering is disabled. This app is SRT only.'

      });
  }
);


app.post(
  '/api/recap',
  (req, res) => {

    res
      .status(410)
      .json({

        error:
          'Movie Recap is disabled. This app is SRT only.'

      });
  }
);


// ============================================================
// API 404
// ============================================================

app.use(
  '/api',
  (req, res) => {

    res
      .status(404)
      .json({

        error:
          'API endpoint not found.'

      });
  }
);


// ============================================================
// GLOBAL ERROR
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
      error?.code ===
      'LIMIT_FILE_SIZE'
    ) {

      return res
        .status(413)
        .json({

          error:
            'Video file size က 300MB ထက် မကျော်ရပါ။'

        });
    }


    return res
      .status(500)
      .json({

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
      `Groq model: ${GROQ_MODEL}`
    );

    console.log(
      `Gemini model: ${GEMINI_MODEL}`
    );

    console.log(
      'Audio extraction: ENABLED'
    );

    console.log(
      'Precise word timeline: ENABLED'
    );

  }
);
