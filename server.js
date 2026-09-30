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


/* =========================================================
   BASIC CONFIG
========================================================= */

const app = express();

const PORT = Number(process.env.PORT || 3000);

const MAX_BYTES =
  300 * 1024 * 1024;

const MAX_SECONDS =
  5 * 60;

const TMP =
  path.join(
    os.tmpdir(),
    'myanmar-srt'
  );

const FONT_DIR =
  path.join(
    TMP,
    'fonts'
  );

const MYANMAR_FONT =
  path.join(
    FONT_DIR,
    'NotoSansMyanmar-Regular.ttf'
  );

const GROQ_MODEL =
  'whisper-large-v3';

const GEMINI_MODEL =
  'gemini-3.5-flash-lite';


/* =========================================================
   EXPRESS
========================================================= */

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


/* =========================================================
   MULTER
========================================================= */

const upload =
  multer({
    dest: TMP,
    limits: {
      fileSize: MAX_BYTES
    }
  });


/* =========================================================
   TEMP DOWNLOAD STORAGE
========================================================= */

const downloadFiles =
  new Map();

const DOWNLOAD_TTL =
  30 * 60 * 1000;


/*
  Final MP4 ကို render ပြီးတာနဲ့ ချက်ချင်းမဖျက်ဘူး။

  Browser က /api/download/:token ကိုခေါ်တဲ့အချိန်မှ
  MP4 ကို download လုပ်ပေးပြီး ပြီးရင်ဖျက်မယ်။
*/

function registerDownload(
  file,
  filename
) {
  const token =
    crypto.randomUUID();

  downloadFiles.set(
    token,
    {
      file,
      filename,
      expiresAt:
        Date.now() +
        DOWNLOAD_TTL
    }
  );

  return token;
}


/*
  အသုံးမပြုဘဲကျန်နေတဲ့ download file
  တွေကို 1 မိနစ်တစ်ကြိမ် စစ်ပြီးဖျက်မယ်။
*/

const downloadCleanupTimer =
  setInterval(
    async () => {
      const now =
        Date.now();

      for (
        const [
          token,
          item
        ] of downloadFiles
      ) {
        if (
          item.expiresAt <
          now
        ) {
          downloadFiles.delete(
            token
          );

          try {
            await fs.unlink(
              item.file
            );
          } catch {}
        }
      }
    },
    60 * 1000
  );

downloadCleanupTimer.unref?.();


/* =========================================================
   FILE HELPERS
========================================================= */

async function ensureDirs() {
  await fs.mkdir(
    TMP,
    {
      recursive: true
    }
  );

  await fs.mkdir(
    FONT_DIR,
    {
      recursive: true
    }
  );
}


async function cleanup(
  file
) {
  if (!file) return;

  try {
    await fs.unlink(
      file
    );
  } catch {}
}


/* =========================================================
   RUN PROCESS
========================================================= */

function runProcess(
  command,
  args,
  options = {}
) {
  return new Promise(
    (
      resolve,
      reject
    ) => {
      const child =
        spawn(
          command,
          args,
          {
            ...options,
            windowsHide: true
          }
        );

      let stdout = '';
      let stderr = '';

      child.stdout?.on(
        'data',
        data => {
          stdout +=
            data.toString();
        }
      );

      child.stderr?.on(
        'data',
        data => {
          stderr +=
            data.toString();
        }
      );

      child.on(
        'error',
        error => {
          reject(error);
        }
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
            const error =
              new Error(
                `Process failed with code ${code}\n${stderr.slice(-6000)}`
              );

            error.code =
              code;

            error.stderr =
              stderr;

            reject(error);
          }
        }
      );
    }
  );
}


/* =========================================================
   FONT
========================================================= */

async function ensureMyanmarFont() {
  try {
    await fs.access(
      MYANMAR_FONT
    );

    return;
  } catch {}


  /*
    Google Fonts GitHub repository မှ
    Noto Sans Myanmar font ကို download လုပ်မယ်။
  */

  const url =
    'https://raw.githubusercontent.com/google/fonts/main/ofl/notosansmyanmar/NotoSansMyanmar%5Bwght%5D.ttf';

  try {
    const response =
      await fetch(
        url
      );

    if (
      !response.ok
    ) {
      throw new Error(
        `Font download HTTP ${response.status}`
      );
    }

    const buffer =
      Buffer.from(
        await response.arrayBuffer()
      );

    if (
      buffer.length <
      10000
    ) {
      throw new Error(
        'Downloaded font file is too small'
      );
    }

    await fs.writeFile(
      MYANMAR_FONT,
      buffer
    );

    console.log(
      'Myanmar font downloaded'
    );
  } catch (error) {
    console.error(
      'Myanmar font download failed:',
      error.message
    );

    /*
      Render ကို ဒီနေရာမှာမရပ်ဘူး။
      Server မှာ system font ရှိရင် ဆက်သုံးနိုင်အောင်ထားတယ်။
    */
  }
}


/* =========================================================
   VIDEO PROBE
========================================================= */

async function probeVideo(
  file
) {
  const result =
    await runProcess(
      ffprobeStatic.path,
      [
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-show_entries',
        'stream=width,height,codec_type',
        '-of',
        'json',
        file
      ]
    );

  const data =
    JSON.parse(
      result.stdout
    );

  const videoStream =
    (
      data.streams ||
      []
    ).find(
      s =>
        s.codec_type ===
        'video'
    );

  const duration =
    Number(
      data.format?.duration ||
      0
    );

  return {
    duration,
    width:
      Number(
        videoStream?.width ||
        1920
      ),
    height:
      Number(
        videoStream?.height ||
        1080
      )
  };
}


/* =========================================================
   AUDIO EXTRACTION
========================================================= */

async function extractAudio(
  video,
  wav
) {
  await runProcess(
    ffmpegStatic,
    [
      '-y',
      '-i',
      video,

      '-vn',

      '-ac',
      '1',

      '-ar',
      '16000',

      '-c:a',
      'pcm_s16le',

      wav
    ]
  );
}


/* =========================================================
   GROQ TRANSCRIPTION
========================================================= */

async function transcribeGroq(
  wav,
  apiKey
) {
  if (!apiKey) {
    throw new Error(
      'Groq API Key မရှိပါ'
    );
  }

  const groq =
    new Groq({
      apiKey
    });

  const stream =
    createReadStream(
      wav
    );

  const result =
    await groq.audio.transcriptions.create(
      {
        file: stream,

        model:
          GROQ_MODEL,

        response_format:
          'verbose_json',

        timestamp_granularities:
          [
            'word',
            'segment'
          ],

        temperature: 0
      }
    );

  return result;
}


/* =========================================================
   WORD / SEGMENT PROCESSING
========================================================= */

function cleanText(
  text
) {
  return String(
    text || ''
  )
    .replace(
      /\s+/g,
      ' '
    )
    .trim();
}


function punctuationEnd(
  text
) {
  return /[.!?။！？]$/.test(
    String(text || '').trim()
  );
}


function makePreciseSegments(
  transcription
) {
  const words =
    Array.isArray(
      transcription?.words
    )
      ? transcription.words
      : [];

  /*
    Word timestamps ရှိရင်
    word အလိုက် subtitle ခွဲမယ်။
  */

  if (!words.length) {
    return (
      Array.isArray(
        transcription?.segments
      )
        ? transcription.segments
        : []
    )
      .map(
        s => ({
          start:
            Number(
              s.start || 0
            ),

          end:
            Number(
              s.end || 0
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
          s.end >
            s.start
      );
  }


  const output = [];

  let current = null;


  for (
    const rawWord of words
  ) {
    const word =
      cleanText(
        rawWord.word
      );

    const start =
      Number(
        rawWord.start
      );

    const end =
      Number(
        rawWord.end
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
      continue;
    }


    if (!current) {
      current = {
        start,
        end,
        words: [word]
      };

      continue;
    }


    const pause =
      start -
      current.end;

    const wordCount =
      current.words.length;

    const charCount =
      current.words.join(
        ' '
      ).length;


    const shouldSplit =
      pause >
        0.75 ||

      end -
          current.start >
        5 ||

      wordCount >=
        11 ||

      charCount >=
        52 ||

      punctuationEnd(
        current.words[
          current.words.length - 1
        ]
      );


    if (
      shouldSplit
    ) {
      output.push({
        start:
          current.start,

        end:
          current.end,

        text:
          current.words.join(
            ' '
          )
      });

      current = {
        start,
        end,
        words: [word]
      };
    } else {
      current.words.push(
        word
      );

      current.end =
        end;
    }
  }


  if (current) {
    output.push({
      start:
        current.start,

      end:
        current.end,

      text:
        current.words.join(
          ' '
        )
    });
  }


  return removeDuplicateOverlap(
    output
  );
}


/* =========================================================
   REMOVE DUPLICATE OVERLAP
========================================================= */

function removeDuplicateOverlap(
  segments
) {
  const output = [];

  for (
    const item of segments
  ) {
    const text =
      cleanText(
        item.text
      );

    if (
      !text ||
      item.end <=
        item.start
    ) {
      continue;
    }

    const previous =
      output[
        output.length - 1
      ];

    if (
      previous &&
      previous.text
        .toLowerCase()
        .replace(/\s+/g, ' ')
        ===
      text
        .toLowerCase()
        .replace(/\s+/g, ' ')
    ) {
      previous.end =
        Math.max(
          previous.end,
          item.end
        );

      continue;
    }

    output.push({
      start:
        Number(
          item.start
        ),

      end:
        Number(
          item.end
        ),

      text
    });
  }

  return output;
}


/* =========================================================
   GEMINI
========================================================= */

function getGemini(
  apiKey
) {
  if (!apiKey) {
    throw new Error(
      'Gemini API Key မရှိပါ'
    );
  }

  return new GoogleGenAI({
    apiKey
  });
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


function isRetryableGeminiError(
  error
) {
  const text =
    String(
      error?.message ||
      error ||
      ''
    ).toLowerCase();

  return (
    text.includes('429') ||
    text.includes('503') ||
    text.includes('500') ||
    text.includes('502') ||
    text.includes('504') ||
    text.includes('timeout') ||
    text.includes('temporarily')
  );
}


async function generateGemini(
  ai,
  contents
) {
  let lastError =
    null;

  for (
    let attempt = 0;
    attempt < 4;
    attempt++
  ) {
    try {
      const response =
        await ai.models.generateContent(
          {
            model:
              GEMINI_MODEL,

            contents,

            config: {
              temperature: 0.15
            }
          }
        );

      return response;
    } catch (error) {
      lastError =
        error;

      if (
        !isRetryableGeminiError(
          error
        ) ||
        attempt === 3
      ) {
        throw error;
      }

      await sleep(
        1200 *
          (attempt + 1)
      );
    }
  }

  throw lastError;
}


/* =========================================================
   JSON EXTRACTION
========================================================= */

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
        /\s*```$/i,
        ''
      )
      .trim();


  try {
    return JSON.parse(
      raw
    );
  } catch {}


  const start =
    raw.indexOf(
      '['
    );

  const end =
    raw.lastIndexOf(
      ']'
    );

  if (
    start >= 0 &&
    end > start
  ) {
    return JSON.parse(
      raw.slice(
        start,
        end + 1
      )
    );
  }


  throw new Error(
    'Gemini JSON response မမှန်ပါ'
  );
}


/* =========================================================
   MYANMAR TEXT CLEANING
========================================================= */

function cleanMyanmarText(
  text
) {
  return String(
    text || ''
  )
    .replace(
      /\r/g,
      ''
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


function hasMyanmar(
  text
) {
  return /[\u1000-\u109F]/.test(
    String(text || '')
  );
}


/* =========================================================
   TRANSLATE CHUNK
========================================================= */

async function translateChunk(
  ai,
  segments
) {
  const input =
    segments.map(
      (
        item,
        index
      ) => ({
        id:
          index + 1,

        start:
          item.start,

        end:
          item.end,

        text:
          item.text
      })
    );


  const prompt = `
You are a professional Myanmar subtitle translator.

Translate the following subtitle dialogue into natural, easy-to-read Myanmar Burmese.

IMPORTANT RULES:

1. Return ONLY valid JSON.
2. Return an array.
3. Keep the same id, start and end values.
4. Do NOT change timestamps.
5. Translate every dialogue line.
6. Use natural spoken Myanmar.
7. Do not add explanations.
8. Do not add English translation.
9. Do not add speaker labels unless they already exist.
10. Do not merge subtitle entries.
11. Do not split subtitle entries.
12. The "text" field should contain Myanmar subtitle text.
13. Preserve names and technical terms when appropriate.
14. Avoid unnecessary English words.
15. Keep the subtitle concise enough to read on screen.

Input:

${JSON.stringify(
  input
)}

Output format:

[
  {
    "id": 1,
    "start": 0,
    "end": 2.5,
    "text": "မြန်မာစာ"
  }
]
`;


  const response =
    await generateGemini(
      ai,
      prompt
    );


  const data =
    extractJson(
      response.text
    );


  if (
    !Array.isArray(
      data
    )
  ) {
    throw new Error(
      'Gemini response array မဟုတ်ပါ'
    );
  }


  return data.map(
    (
      item,
      index
    ) => ({
      start:
        Number(
          segments[index]?.start ??
          item.start ??
          0
        ),

      end:
        Number(
          segments[index]?.end ??
          item.end ??
          0
        ),

      text:
        cleanMyanmarText(
          item.text
        )
    })
  );
}


/* =========================================================
   TRANSLATE ALL
========================================================= */

async function translateAll(
  segments,
  apiKey
) {
  const ai =
    getGemini(
      apiKey
    );

  const result = [];

  const chunkSize =
    35;

  for (
    let i = 0;
    i < segments.length;
    i += chunkSize
  ) {
    const chunk =
      segments.slice(
        i,
        i + chunkSize
      );

    const translated =
      await translateChunk(
        ai,
        chunk
      );

    for (
      let j = 0;
      j < chunk.length;
      j++
    ) {
      const original =
        chunk[j];

      const translatedItem =
        translated[j];

      let text =
        cleanMyanmarText(
          translatedItem?.text
        );

      /*
        Myanmar မပါရင် original ကို
        မပြန်ထည့်ဘဲ Gemini ကို retry
        မလုပ်နိုင်တဲ့အတွက် အနည်းဆုံး
        original text ကိုထားပေးမယ်။
      */

      if (!text) {
        text =
          original.text;
      }

      result.push({
        start:
          original.start,

        end:
          original.end,

        text
      });
    }
  }

  return result;
}


/* =========================================================
   SUBTITLE SPLIT
========================================================= */

function splitSubtitleText(
  text,
  maxChars = 34
) {
  const value =
    cleanMyanmarText(
      text
    );

  if (
    value.length <=
    maxChars
  ) {
    return value;
  }

  const words =
    value.split(
      ' '
    );

  const lines = [];

  let current = '';

  for (
    const word of words
  ) {
    const next =
      current
        ? `${current} ${word}`
        : word;

    if (
      next.length >
        maxChars &&
      current
    ) {
      lines.push(
        current
      );

      current =
        word;
    } else {
      current =
        next;
    }
  }

  if (current) {
    lines.push(
      current
    );
  }

  return lines
    .slice(0, 3)
    .join('\\N');
}


/* =========================================================
   SRT
========================================================= */

function formatSrtTime(
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

  const x =
    ms % 1000;

  return (
    `${String(h).padStart(2, '0')}:` +
    `${String(m).padStart(2, '0')}:` +
    `${String(s).padStart(2, '0')},` +
    `${String(x).padStart(3, '0')}`
  );
}


function makeSrt(
  segments
) {
  return segments
    .map(
      (
        s,
        index
      ) =>
        `${index + 1}\n` +
        `${formatSrtTime(s.start)} --> ${formatSrtTime(s.end)}\n` +
        `${s.text}\n`
    )
    .join('\n');
}


/* =========================================================
   ASS HELPERS
========================================================= */

function assTime(
  seconds
) {
  const total =
    Math.max(
      0,
      Number(
        seconds || 0
      )
    );

  const h =
    Math.floor(
      total / 3600
    );

  const m =
    Math.floor(
      (
        total %
        3600
      ) / 60
    );

  const s =
    Math.floor(
      total % 60
    );

  const cs =
    Math.floor(
      (
        total -
        Math.floor(total)
      ) *
        100
    );

  return (
    `${h}:` +
    `${String(m).padStart(2, '0')}:` +
    `${String(s).padStart(2, '0')}.` +
    `${String(cs).padStart(2, '0')}`
  );
}


function assEscape(
  text
) {
  return String(
    text || ''
  )
    .replace(
      /\\/g,
      '\\\\'
    )
    .replace(
      /{/g,
      '\\{'
    )
    .replace(
      /}/g,
      '\\}'
    )
    .replace(
      /\r?\n/g,
      '\\N'
    );
}


function hexToAssColor(
  hex
) {
  let value =
    String(
      hex || '#FFFFFF'
    )
      .trim()
      .replace(
        '#',
        ''
      );

  if (
    !/^[0-9a-fA-F]{6}$/.test(
      value
    )
  ) {
    value =
      'FFFFFF';
  }

  const r =
    value.slice(
      0,
      2
    );

  const g =
    value.slice(
      2,
      4
    );

  const b =
    value.slice(
      4,
      6
    );

  return (
    `&H00${b}${g}${r}`
  );
}


function escapeFilterPath(
  file
) {
  return String(
    file
  )
    .replace(
      /\\/g,
      '\\\\'
    )
    .replace(
      /:/g,
      '\\:'
    )
    .replace(
      /'/g,
      "\\'"
    )
    .replace(
      /,/g,
      '\\,'
    );
}


/* =========================================================
   SUBTITLE BOX
========================================================= */

function safeSubtitleBox(
  raw
) {
  if (
    !raw ||
    typeof raw !==
      'object'
  ) {
    return null;
  }

  let w =
    Number(
      raw.w
    );

  let h =
    Number(
      raw.h
    );

  let x =
    Number(
      raw.x
    );

  let y =
    Number(
      raw.y
    );


  if (
    !Number.isFinite(w)
  ) {
    w = 0.9;
  }

  if (
    !Number.isFinite(h)
  ) {
    h = 0.18;
  }

  if (
    !Number.isFinite(x)
  ) {
    x = 0.05;
  }

  if (
    !Number.isFinite(y)
  ) {
    y = 0.68;
  }


  w =
    Math.max(
      0.12,
      Math.min(
        1,
        w
      )
    );

  h =
    Math.max(
      0.08,
      Math.min(
        1,
        h
      )
    );


  x =
    Math.max(
      0,
      Math.min(
        1 - w,
        x
      )
    );

  y =
    Math.max(
      0,
      Math.min(
        1 - h,
        y
      )
    );


  return {
    x,
    y,
    w,
    h
  };
}


/* =========================================================
   ASS BUILD
========================================================= */

function buildAss(
  segments,
  options,
  width,
  height
) {
  const videoWidth =
    Number(width) ||
    1920;

  const videoHeight =
    Number(height) ||
    1080;


  const fontSize =
    Math.max(
      22,
      Math.min(
        80,
        Number(
          options.fontSize
        ) || 42
      )
    );


  const outline =
    Math.max(
      0,
      Math.min(
        8,
        Number(
          options.outline
        ) || 3
      )
    );


  const position =
    [
      'top',
      'middle',
      'bottom'
    ].includes(
      options.position
    )
      ? options.position
      : 'bottom';


  const color =
    hexToAssColor(
      options.color ||
      '#FFFFFF'
    );


  const border =
    hexToAssColor(
      '#000000'
    );


  const font =
    'Noto Sans Myanmar';


  const box =
    options.subtitleBox;


  let alignment;
  let marginL;
  let marginR;
  let marginV;


  if (
    box &&
    typeof box ===
      'object'
  ) {
    const x =
      Math.max(
        0,
        Math.min(
          1,
          Number(
            box.x
          ) || 0
        )
      );

    const y =
      Math.max(
        0,
        Math.min(
          1,
          Number(
            box.y
          ) || 0
        )
      );

    const w =
      Math.max(
        0.12,
        Math.min(
          1 - x,
          Number(
            box.w
          ) || 0.9
        )
      );


    alignment = 7;


    marginL =
      Math.round(
        x *
          videoWidth
      );


    marginR =
      Math.max(
        0,
        Math.round(
          (
            1 -
            x -
            w
          ) *
            videoWidth
        )
      );


    marginV =
      Math.round(
        y *
          videoHeight
      );
  } else {
    alignment =
      position === 'top'
        ? 8
        : position === 'middle'
          ? 5
          : 2;


    marginL = 50;
    marginR = 50;


    marginV =
      position === 'top'
        ? 55
        : position === 'middle'
          ? 0
          : 55;
  }


  const events =
    segments
      .map(
        s =>
          `Dialogue: 0,${assTime(
            s.start
          )},${assTime(
            s.end
          )},Default,,0,0,0,,${assEscape(
            splitSubtitleText(
              s.text
            )
          )}`
      )
      .join('\n');


  return `[Script Info]
ScriptType: v4.00+
PlayResX: ${videoWidth}
PlayResY: ${videoHeight}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,${font},${fontSize},${color},${color},${border},&H99000000&,0,0,0,0,100,100,0,0,1,${outline},1,${alignment},${marginL},${marginR},${marginV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
${events}
`;
}


/* =========================================================
   BLUR HELPERS
========================================================= */

function clamp01(
  value
) {
  const n =
    Number(
      value
    );

  if (
    !Number.isFinite(n)
  ) {
    return 0;
  }

  return Math.max(
    0,
    Math.min(
      1,
      n
    )
  );
}


function safeBlurRegions(
  regions,
  width,
  height
) {
  if (
    !Array.isArray(
      regions
    ) ||
    !width ||
    !height
  ) {
    return [];
  }


  return regions
    .slice(
      0,
      3
    )
    .map(
      r => {
        const xNorm =
          clamp01(
            r.x
          );

        const yNorm =
          clamp01(
            r.y
          );


        let wNorm =
          clamp01(
            r.w
          );

        let hNorm =
          clamp01(
            r.h
          );


        wNorm =
          Math.max(
            0.03,
            Math.min(
              1 - xNorm,
              wNorm
            )
          );


        hNorm =
          Math.max(
            0.03,
            Math.min(
              1 - yNorm,
              hNorm
            )
          );


        const x =
          Math.round(
            xNorm *
              width
          );

        const y =
          Math.round(
            yNorm *
              height
          );


        const w =
          Math.max(
            8,
            Math.round(
              wNorm *
                width
            )
          );


        const h =
          Math.max(
            8,
            Math.round(
              hNorm *
                height
            )
          );


        return {
          x:
            Math.min(
              x,
              Math.max(
                0,
                width - 8
              )
            ),

          y:
            Math.min(
              y,
              Math.max(
                0,
                height - 8
              )
            ),

          w:
            Math.min(
              w,
              width
            ),

          h:
            Math.min(
              h,
              height
            )
        };
      }
    )
    .filter(
      r =>
        r.w >= 8 &&
        r.h >= 8
    );
}


/* =========================================================
   BLUR FILTER
========================================================= */

function makeBlurFilter(
  regions
) {
  if (
    !regions.length
  ) {
    return (
      '[0:v]null[video]'
    );
  }


  let filter =
    '[0:v]split=' +
    (
      regions.length + 1
    );


  for (
    let i = 0;
    i <
      regions.length + 1;
    i++
  ) {
    filter +=
      `[s${i}]`;
  }


  filter +=
    ';';


  let base =
    '[s0]';


  for (
    let i = 0;
    i <
      regions.length;
    i++
  ) {
    const r =
      regions[i];


    const crop =
      `[s${i + 1}]crop=${r.w}:${r.h}:${r.x}:${r.y},boxblur=18:2[b${i}]`;


    filter +=
      crop +
      ';';


    const next =
      `[o${i}]`;


    filter +=
      `${base}[b${i}]overlay=${r.x}:${r.y}${next};`;


    base =
      next;
  }


  /*
    copy filter မသုံးဘူး။
    null filter သုံးထားတယ်။
  */

  filter +=
    `${base}null[video]`;


  return filter;
}


/* =========================================================
   HEALTH
========================================================= */

app.get(
  '/api/health',
  async (
    req,
    res
  ) => {
    res.json({
      ok: true,
      service:
        'Myanmar SRT',
      groqModel:
        GROQ_MODEL,
      geminiModel:
        GEMINI_MODEL,
      maxVideoMB:
        300,
      maxVideoMinutes:
        5
    });
  }
);


/* =========================================================
   TRANSCRIBE API
========================================================= */

app.post(
  '/api/transcribe',
  upload.single('video'),
  async (
    req,
    res
  ) => {
    const video =
      req.file?.path;

    let wav = null;


    try {
      if (!video) {
        throw new Error(
          'Video file မရှိပါ'
        );
      }


      if (
        req.file.size >
        MAX_BYTES
      ) {
        throw new Error(
          'Video က 300MB ထက်ကြီးနေပါတယ်'
        );
      }


      const info =
        await probeVideo(
          video
        );


      if (
        info.duration >
        MAX_SECONDS
      ) {
        throw new Error(
          'Video က 5 မိနစ်ထက်ကျော်နေပါတယ်'
        );
      }


      const groqKey =
        String(
          req.headers[
            'x-groq-api-key'
          ] ||
          process.env.GROQ_API_KEY ||
          ''
        ).trim();


      if (!groqKey) {
        throw new Error(
          'Groq API Key မရှိပါ'
        );
      }


      wav =
        path.join(
          TMP,
          `${crypto.randomUUID()}.wav`
        );


      console.log(
        'Extracting audio...'
      );


      await extractAudio(
        video,
        wav
      );


      console.log(
        'Groq transcription...'
      );


      const transcription =
        await transcribeGroq(
          wav,
          groqKey
        );


      const segments =
        makePreciseSegments(
          transcription
        );


      res.json({
        ok: true,

        duration:
          info.duration,

        width:
          info.width,

        height:
          info.height,

        transcript:
          segments
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
        ok: false,

        error:
          error?.message ||
          'Transcript မအောင်မြင်ပါ'
      });
    } finally {
      await cleanup(
        video
      );

      await cleanup(
        wav
      );
    }
  }
);


/* =========================================================
   TRANSLATE API
========================================================= */

app.post(
  '/api/translate',
  async (
    req,
    res
  ) => {
    try {
      const geminiKey =
        String(
          req.headers[
            'x-gemini-api-key'
          ] ||
          process.env.GEMINI_API_KEY ||
          ''
        ).trim();


      if (!geminiKey) {
        throw new Error(
          'Gemini API Key မရှိပါ'
        );
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
        throw new Error(
          'Original Transcript မရှိပါ'
        );
      }


      const cleanSegments =
        transcript
          .map(
            s => ({
              start:
                Number(
                  s.start || 0
                ),

              end:
                Number(
                  s.end || 0
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
              s.end >
                s.start
          );


      const translated =
        await translateAll(
          cleanSegments,
          geminiKey
        );


      res.json({
        ok: true,

        transcript:
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


      res.status(
        400
      ).json({
        ok: false,

        error:
          error?.message ||
          'Myanmar Translation မအောင်မြင်ပါ'
      });
    }
  }
);


/* =========================================================
   RENDER API
========================================================= */

app.post(
  '/api/render',
  upload.single('video'),
  async (
    req,
    res
  ) => {
    const video =
      req.file?.path;

    let ass = null;

    let output = null;

    let registeredOutput =
      false;


    try {
      if (!video) {
        throw new Error(
          'Video file မရှိပါ'
        );
      }


      if (
        req.file.size >
        MAX_BYTES
      ) {
        throw new Error(
          'Video က 300MB ထက်ကြီးနေပါတယ်'
        );
      }


      const info =
        await probeVideo(
          video
        );


      if (
        info.duration >
        MAX_SECONDS
      ) {
        throw new Error(
          'Video က 5 မိနစ်ထက်ကျော်နေပါတယ်'
        );
      }


      /*
        Myanmar segments
      */

      let segments = [];


      try {
        segments =
          Array.isArray(
            req.body?.segments
          )
            ? req.body.segments
            : JSON.parse(
                String(
                  req.body?.segments ||
                  '[]'
                )
              );
      } catch {
        segments = [];
      }


      if (
        !segments.length
      ) {
        throw new Error(
          'Myanmar Subtitle မရှိပါ'
        );
      }


      /*
        Subtitle Box
      */

      let subtitleBox =
        null;


      try {
        const raw =
          JSON.parse(
            String(
              req.body
                ?.subtitleBox ||
              'null'
            )
          );


        subtitleBox =
          safeSubtitleBox(
            raw
          );
      } catch {
        subtitleBox =
          null;
      }


      /*
        Render options
      */

      const options = {
        fontSize:
          Number(
            req.body?.fontSize
          ) || 42,

        outline:
          Number(
            req.body?.outline
          ) || 3,

        color:
          String(
            req.body?.color ||
            req.body?.textColor ||
            '#FFFFFF'
          ),

        position:
          String(
            req.body?.position ||
            'bottom'
          ),

        subtitleBox
      };


      /*
        Blur Regions
      */

      let blurInput =
        [];


      try {
        blurInput =
          JSON.parse(
            String(
              req.body
                ?.blurRegions ||
              '[]'
            )
          );
      } catch {
        blurInput =
          [];
      }


      const regions =
        safeBlurRegions(
          blurInput,
          info.width,
          info.height
        );


      /*
        Temporary files
      */

      ass =
        path.join(
          TMP,
          `${crypto.randomUUID()}.ass`
        );


      output =
        path.join(
          TMP,
          `${crypto.randomUUID()}.mp4`
        );


      /*
        ASS subtitle
      */

      const assText =
        buildAss(
          segments,
          options,
          info.width,
          info.height
        );


      await fs.writeFile(
        ass,
        assText,
        'utf8'
      );


      /*
        Subtitle filter
      */

      const fontsDir =
        FONT_DIR;


      const subtitleFilter =
        `subtitles=filename='${escapeFilterPath(
          ass
        )}':fontsdir='${escapeFilterPath(
          fontsDir
        )}'`;


      let videoFilter =
        '';


      if (
        regions.length
      ) {
        const blur =
          makeBlurFilter(
            regions
          );


        videoFilter =
          `${blur};[video]${subtitleFilter}[outv]`;
      } else {
        videoFilter =
          `[0:v]${subtitleFilter}[outv]`;
      }


      console.log(
        'Starting FFmpeg render...'
      );


      await runProcess(
        ffmpegStatic,
        [
          '-y',

          '-i',
          video,

          '-filter_complex',
          videoFilter,

          '-map',
          '[outv]',

          '-map',
          '0:a?',

          '-c:v',
          'libx264',

          '-preset',
          'veryfast',

          '-crf',
          '22',

          '-c:a',
          'aac',

          '-b:a',
          '128k',

          '-movflags',
          '+faststart',

          output
        ]
      );


      /*
        Check output
      */

      const stat =
        await fs.stat(
          output
        );


      if (
        !stat.size ||
        stat.size <
          1000
      ) {
        throw new Error(
          'Final Video file မထွက်ပါ'
        );
      }


      console.log(
        `Render completed: ${stat.size} bytes`
      );


      /*
        IMPORTANT:

        MP4 ကို ဒီနေရာမှာမဖျက်ဘူး။

        Download token တစ်ခုဖန်တီးပြီး
        Browser ကို URL ပြန်ပေးမယ်။
      */

      const token =
        registerDownload(
          output,
          'Burmese-YNT-SRT.mp4'
        );


      registeredOutput =
        true;


      /*
        output cleanup ကို catch ထဲက
        မလုပ်အောင် null ထားမယ်။
      */

      output = null;


      /*
        Input video + ASS ကိုတော့
        အခုချက်ချင်းဖျက်နိုင်တယ်။
      */

      await cleanup(
        video
      );

      await cleanup(
        ass
      );


      video = undefined;
      ass = undefined;


      res.json({
        ok: true,

        filename:
          'Burmese-YNT-SRT.mp4',

        downloadUrl:
          `/api/download/${token}`
      });
    } catch (
      error
    ) {
      console.error(
        'RENDER ERROR:',
        error
      );


      await cleanup(
        video
      );

      await cleanup(
        ass
      );


      if (
        !registeredOutput
      ) {
        await cleanup(
          output
        );
      }


      res.status(
        400
      ).json({
        ok: false,

        error:
          error?.message ||
          'Video render မအောင်မြင်ပါ'
      });
    }
  }
);


/* =========================================================
   FINAL VIDEO DOWNLOAD
========================================================= */

app.get(
  '/api/download/:token',
  async (
    req,
    res
  ) => {
    const token =
      String(
        req.params.token ||
        ''
      );


    const item =
      downloadFiles.get(
        token
      );


    if (!item) {
      return res
        .status(404)
        .send(
          'Download link မရှိတော့ပါ။ Render ကို ပြန်လုပ်ပေးပါ။'
        );
    }


    if (
      Date.now() >
      item.expiresAt
    ) {
      downloadFiles.delete(
        token
      );

      await cleanup(
        item.file
      );

      return res
        .status(410)
        .send(
          'Download link သက်တမ်းကုန်သွားပါပြီ။'
        );
    }


    /*
      Token ကို တစ်ကြိမ်သုံးပြီးတာနဲ့
      နောက်တစ်ခါအသုံးမပြုနိုင်အောင်
      ချက်ချင်း remove လုပ်မယ်။
    */

    downloadFiles.delete(
      token
    );


    try {
      await fs.access(
        item.file
      );
    } catch {
      return res
        .status(404)
        .send(
          'Final Video file မတွေ့တော့ပါ။'
        );
    }


    /*
      Android / Browser Download အတွက်
      attachment header ပါအောင်
      Express res.download() သုံးတယ်။
    */

    res.setHeader(
      'Cache-Control',
      'no-store, no-cache, must-revalidate'
    );

    res.setHeader(
      'Pragma',
      'no-cache'
    );


    res.download(
      item.file,
      item.filename,
      {
        headers: {
          'Content-Type':
            'video/mp4'
        }
      },
      async error => {
        await cleanup(
          item.file
        );


        if (error) {
          console.error(
            'DOWNLOAD ERROR:',
            error
          );
        } else {
          console.log(
            'Final video download completed'
          );
        }
      }
    );
  }
);


/* =========================================================
   STATIC FRONTEND
========================================================= */

app.use(
  express.static(
    process.cwd(),
    {
      index:
        'index.html'
    }
  )
);


/* =========================================================
   404
========================================================= */

app.use(
  (
    req,
    res
  ) => {
    if (
      req.path.startsWith(
        '/api/'
      )
    ) {
      return res
        .status(404)
        .json({
          ok: false,
          error:
            'API endpoint မတွေ့ပါ'
        });
    }


    res
      .status(404)
      .send(
        'Page မတွေ့ပါ'
      );
  }
);


/* =========================================================
   GLOBAL ERROR
========================================================= */

app.use(
  (
    error,
    req,
    res,
    next
  ) => {
    console.error(
      'GLOBAL ERROR:',
      error
    );


    if (
      error instanceof
      multer.MulterError
    ) {
      if (
        error.code ===
        'LIMIT_FILE_SIZE'
      ) {
        return res
          .status(413)
          .json({
            ok: false,
            error:
              'Video က 300MB ထက်ကြီးနေပါတယ်'
          });
      }
    }


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
        ok: false,
        error:
          error?.message ||
          'Server Error'
      });
  }
);


/* =========================================================
   START
========================================================= */

async function start() {
  await ensureDirs();

  await ensureMyanmarFont();

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
        `Max video: 300MB / 5 minutes`
      );
    }
  );
}


start().catch(
  error => {
    console.error(
      'SERVER START ERROR:',
      error
    );

    process.exit(
      1
    );
  }
);
