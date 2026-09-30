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

const app = express();

const PORT = Number(process.env.PORT || 10000);
const MAX_BYTES = 300 * 1024 * 1024;
const MAX_SECONDS = 5 * 60;

const ROOT = process.cwd();
const PUBLIC = path.join(ROOT, 'public');
const TMP = path.join(os.tmpdir(), 'myanmar-srt');

await fs.mkdir(TMP, { recursive: true });

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

  const videoStream =
    Array.isArray(data?.streams)
      ? data.streams.find(
          s => s.codec_type === 'video'
        )
      : null;

  const width = Number(
    videoStream?.width || 1920
  );

  const height = Number(
    videoStream?.height || 1080
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
    duration,
    width,
    height
  };
}

async function transcribeGroq(
  file,
  apiKey
) {
  const groq = new Groq({
    apiKey
  });

  return groq.audio.transcriptions.create({
    file: createReadStream(file),
    model: 'whisper-large-v3-turbo',
    response_format: 'verbose_json',
    timestamp_granularities: [
      'word',
      'segment'
    ],
    temperature: 0
  });
}

function buildSegments(
  result,
  duration
) {
  const segments =
    Array.isArray(result?.segments)
      ? result.segments
      : [];

  const words =
    Array.isArray(result?.words)
      ? result.words
      : [];

  const output = segments
    .map((s, i) => {
      const start = Math.max(
        0,
        Number(s.start) || 0
      );

      let end = Math.max(
        0,
        Number(s.end) || 0
      );

      const wordsInSegment =
        words.filter(w => {
          const ws = Number(
            w?.start
          );

          const we = Number(
            w?.end
          );

          return (
            Number.isFinite(ws) &&
            Number.isFinite(we) &&
            we > ws &&
            ws >= start - 0.15 &&
            ws <= end + 0.15
          );
        });

      if (wordsInSegment.length) {
        const lastWordEnd =
          Math.max(
            ...wordsInSegment.map(
              w => Number(w.end)
            )
          );

        end = Math.max(
          end,
          lastWordEnd
        );
      }

      return {
        id: i + 1,
        start,
        end: Math.min(
          Math.max(
            start + 0.05,
            end
          ),
          duration
        ),
        text: cleanText(
          s.text
        )
      };
    })
    .filter(
      s =>
        s.text &&
        s.end > s.start
    );

  if (output.length) {
    const last =
      output[
        output.length - 1
      ];

    const finalWordEnds =
      words
        .map(
          w =>
            Number(w?.end)
        )
        .filter(
          n =>
            Number.isFinite(n) &&
            n > 0 &&
            n <= duration
        );

    if (finalWordEnds.length) {
      last.end = Math.min(
        duration,
        Math.max(
          last.end,
          Math.max(
            ...finalWordEnds
          ) + 0.15
        )
      );
    }

    return output;
  }

  const fullText = cleanText(
    result?.text
  );

  return fullText
    ? [
        {
          id: 1,
          start: 0,
          end: duration,
          text: fullText
        }
      ]
    : [];
}

function extractJson(text) {
  const raw = String(
    text || ''
  ).trim();

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
      return JSON.parse(
        value
      );
    } catch {}
  }

  const a =
    raw.indexOf('[');

  const b =
    raw.lastIndexOf(']');

  if (a >= 0 && b > a) {
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

async function translateChunk(
  segments,
  apiKey,
  model
) {
  const ai =
    new GoogleGenAI({
      apiKey
    });

  const input =
    segments.map(
      s => ({
        id: s.id,
        text: s.text
      })
    );

  const prompt = `
You are a professional Myanmar subtitle translator.

Translate every English dialogue line into natural, concise Myanmar Unicode.

Return ONLY a JSON array.

Each item must contain exactly:
id, translation

Rules:
- keep the same id
- do not remove lines
- do not merge lines
- do not reorder lines
- do not invent lines
- translate only spoken dialogue
- no instructions
- no explanations
- keep names naturally
- keep subtitles short and easy to read
- prefer under 42 characters when possible

INPUT:
${JSON.stringify(input)}
`;

  let response;

  try {
    response =
      await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          temperature: 0.2,
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
      /429|rate.?limit|quota|resource.?exhausted/i.test(
        message
      )
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
      : [];

  if (!list.length) {
    throw new Error(
      'Gemini က valid translation JSON မပြန်ပါ'
    );
  }

  const byId =
    new Map();

  for (const item of list) {
    const id = Number(
      item?.id
    );

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

async function translateAll(
  segments,
  apiKey,
  model = 'gemini-3.8-flash'
) {
  const max = 120;

  if (
    segments.length <= max
  ) {
    return translateChunk(
      segments,
      apiKey,
      model
    );
  }

  const a =
    await translateChunk(
      segments.slice(
        0,
        max
      ),
      apiKey,
      model
    );

  const b =
    await translateChunk(
      segments.slice(max),
      apiKey,
      model
    );

  return [
    ...a,
    ...b
  ];
}

function splitSubtitle(
  text,
  maxChars = 42,
  maxWords = 10
) {
  const value =
    String(text || '')
      .trim();

  if (!value) return [];

  if (
    value.length <= maxChars &&
    value.split(/\s+/).length <=
      maxWords
  ) {
    return [value];
  }

  const words =
    value
      .split(/\s+/)
      .filter(Boolean);

  const parts = [];

  let current = '';

  for (const word of words) {
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

    return chunks.filter(
      Boolean
    );
  }

  return parts;
}

function splitTranslatedSegments(
  segments
) {
  const output = [];

  let id = 1;

  for (const s of segments) {
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
      (part, index) => {
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
          id: id++,
          start: cursor,
          end: Math.min(
            end,
            s.end
          ),
          text: part,
          translation: part
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

function srtTime(seconds) {
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

function makeSrt(
  segments
) {
  return segments
    .map(
      (s, i) =>
        `${i + 1}\n` +
        `${srtTime(s.start)} --> ${srtTime(s.end)}\n` +
        `${s.translation || s.text}\n`
    )
    .join('\n');
}

/* =========================
   MP4 RENDER HELPERS
========================= */

function clamp01(value) {
  return Math.max(
    0,
    Math.min(
      1,
      Number(value) || 0
    )
  );
}

function safeBlurRegions(
  regions,
  width,
  height
) {
  if (
    !Array.isArray(regions) ||
    !width ||
    !height
  ) {
    return [];
  }

  return regions
    .slice(0, 3)
    .map(r => {
      const x =
        Math.round(
          clamp01(r.x) *
            width
        );

      const y =
        Math.round(
          clamp01(r.y) *
            height
        );

      const w =
        Math.max(
          8,
          Math.round(
            clamp01(r.w) *
              width
          )
        );

      const h =
        Math.max(
          8,
          Math.round(
            clamp01(r.h) *
              height
          )
        );

      return {
        x: Math.min(
          x,
          Math.max(
            0,
            width - 8
          )
        ),

        y: Math.min(
          y,
          Math.max(
            0,
            height - 8
          )
        ),

        w: Math.min(
          w,
          width
        ),

        h: Math.min(
          h,
          height
        )
      };
    })
    .filter(
      r =>
        r.w >= 8 &&
        r.h >= 8
    );
}

function makeBlurFilter(
  regions
) {
  if (
    !regions.length
  ) {
    return '[0:v]null[video]';
  }

  let filter =
    '[0:v]split=' +
    (regions.length + 1);

  for (
    let i = 0;
    i < regions.length + 1;
    i++
  ) {
    filter += `[s${i}]`;
  }

  filter += ';';

  let base =
    '[s0]';

  for (
    let i = 0;
    i < regions.length;
    i++
  ) {
    const r =
      regions[i];

    const crop =
      `[s${i + 1}]` +
      `crop=${r.w}:${r.h}:${r.x}:${r.y},` +
      `boxblur=18:2[b${i}]`;

    filter +=
      crop + ';';

    const next =
      `[o${i}]`;

    filter +=
      `${base}` +
      `[b${i}]` +
      `overlay=${r.x}:${r.y}` +
      `${next};`;

    base = next;
  }

  filter +=
    `${base}copy[video]`;

  return filter;
}

function hexToAssColor(
  hex
) {
  const value =
    String(hex || '#FFFFFF')
      .replace('#', '')
      .trim();

  const safe =
    /^[0-9a-fA-F]{6}$/.test(
      value
    )
      ? value
      : 'FFFFFF';

  const r =
    safe.slice(0, 2);

  const g =
    safe.slice(2, 4);

  const b =
    safe.slice(4, 6);

  return `&H00${b}${g}${r}`;
}

function assTime(seconds) {
  const total =
    Math.max(
      0,
      Number(seconds) || 0
    );

  const h =
    Math.floor(
      total / 3600
    );

  const m =
    Math.floor(
      (total % 3600) / 60
    );

  const s =
    total % 60;

  return (
    `${h}:` +
    `${String(m).padStart(2, '0')}:` +
    `${s.toFixed(2).padStart(5, '0')}`
  );
}

function assEscape(text) {
  return String(text || '')
    .replace(/\\/g, '\\\\')
    .replace(/\{/g, '\\{')
    .replace(/\}/g, '\\}');
}

/*
  Subtitle Box:
  x/y/w/h are normalized 0-1 values.

  When the frontend sends subtitleBox,
  subtitles are placed inside that area.

  If subtitleBox is missing,
  the old top/middle/bottom system
  continues to work.
*/
function buildAss(
  segments,
  options,
  width,
  height
) {
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

  let alignment = 2;
  let marginL = 50;
  let marginR = 50;
  let marginV = 55;

  const box =
    options.subtitleBox;

  if (
    box &&
    width &&
    height
  ) {
    const x =
      clamp01(
        box.x
      );

    const y =
      clamp01(
        box.y
      );

    const w =
      Math.max(
        0.12,
        Math.min(
          1,
          Number(box.w) ||
            0.9
        )
      );

    const h =
      Math.max(
        0.08,
        Math.min(
          1,
          Number(box.h) ||
            0.18
        )
      );

    const safeW =
      Math.min(
        w,
        1 - x
      );

    const safeH =
      Math.min(
        h,
        1 - y
      );

    marginL =
      Math.max(
        0,
        Math.round(
          x * width
        )
      );

    marginR =
      Math.max(
        0,
        Math.round(
          (
            1 -
            x -
            safeW
          ) * width
        )
      );

    marginV =
      Math.max(
        0,
        Math.round(
          y * height
        )
      );

    /*
      Top-left alignment allows
      the subtitle box to behave
      like the frontend editor.
    */
    alignment = 7;

    /*
      Keep a reasonable minimum
      right margin so ASS doesn't
      overflow the video.
    */
    marginR =
      Math.max(
        20,
        marginR
      );

    /*
      The height is mainly controlled
      by the subtitle line wrapping.
      The frontend editor uses h for
      visual positioning and resizing.
    */
    void safeH;
  } else {
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

    alignment =
      position === 'top'
        ? 8
        : position === 'middle'
          ? 5
          : 2;

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
            s.text
          )}`
      )
      .join('\n');

  return `[Script Info]
ScriptType: v4.00+
PlayResX: ${width || 1920}
PlayResY: ${height || 1080}
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

function escapeFilterPath(
  value
) {
  return String(value)
    .replace(/\\/g, '\\\\')
    .replace(/:/g, '\\:')
    .replace(/'/g, "\\'");
}

async function runProcess(
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
            stdio: [
              'ignore',
              'pipe',
              'pipe'
            ]
          }
        );

      let stderr = '';

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
            resolve();
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

/* =========================
   HEALTH
========================= */

app.get(
  '/api/health',
  (_req, res) => {
    res.json({
      ok: true,
      service:
        'myanmar-srt',

      srtOnly: false,

      movieRecap: false,

      render: true,

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
   TRANSCRIBE
========================= */

app.post(
  '/api/transcribe',
  upload.single('video'),
  async (req, res) => {
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

      res
        .status(400)
        .json({
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

/* =========================
   TRANSLATE
========================= */

app.post(
  '/api/translate',
  async (req, res) => {
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
        ok: true,

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

      res
        .status(400)
        .json({
          ok: false,
          error:
            error?.message ||
            'Gemini Myanmar Translation Error'
        });
    }
  }
);

/* =========================
   OLD PROCESS COMPATIBILITY
========================= */

app.post(
  '/api/process',
  upload.single('video'),
  async (req, res) => {
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

      const segments =
        splitTranslatedSegments(
          translated
        );

      res.json({
        ok: true,
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

      res
        .status(400)
        .json({
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

/* =========================
   MP4 RENDER
========================= */

app.post(
  '/api/render',
  upload.single('video'),
  async (req, res) => {
    const video =
      req.file?.path;

    let ass = null;
    let output = null;

    try {
      if (!video) {
        throw new Error(
          'Video file မရှိပါ'
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

      const segments =
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

      if (
        !segments.length
      ) {
        throw new Error(
          'Myanmar Subtitle မရှိပါ'
        );
      }

      /* -------------------------
         Subtitle options
      ------------------------- */

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
          )
      };

      /* -------------------------
         Blur Regions
      ------------------------- */

      let blurInput = [];

      try {
        blurInput =
          JSON.parse(
            String(
              req.body?.blurRegions ||
              '[]'
            )
          );
      } catch {
        blurInput = [];
      }

      const regions =
        safeBlurRegions(
          blurInput,
          info.width,
          info.height
        );

      /* -------------------------
         Subtitle Box
      ------------------------- */

      let subtitleBox = null;

      try {
        const raw =
          JSON.parse(
            String(
              req.body?.subtitleBox ||
              'null'
            )
          );

        if (
          raw &&
          typeof raw ===
            'object'
        ) {
          const rawW =
            Number(
              raw.w
            ) || 0.9;

          const rawH =
            Number(
              raw.h
            ) || 0.18;

          const w =
            Math.max(
              0.12,
              Math.min(
                1,
                rawW
              )
            );

          const h =
            Math.max(
              0.08,
              Math.min(
                1,
                rawH
              )
            );

          const x =
            Math.max(
              0,
              Math.min(
                1 - w,
                Number(
                  raw.x
                ) || 0.05
              )
            );

          const y =
            Math.max(
              0,
              Math.min(
                1 - h,
                Number(
                  raw.y
                ) || 0.68
              )
            );

          subtitleBox = {
            x,
            y,
            w,
            h
          };
        }
      } catch {
        subtitleBox = null;
      }

      options.subtitleBox =
        subtitleBox;

      /* -------------------------
         Temporary files
      ------------------------- */

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

      /* -------------------------
         Build ASS
      ------------------------- */

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

      /* -------------------------
         Subtitle filter
      ------------------------- */

      const fontsDir =
        path.dirname(
          ffprobeStatic.path
        );

      const subtitleFilter =
        `subtitles=filename='${escapeFilterPath(
          ass
        )}'`;

      /*
        If Noto Sans Myanmar font
        exists in public/fonts or
        server font directory,
        FFmpeg can use it through
        fontsdir.
      */

      const possibleFontDirs = [
        path.join(
          ROOT,
          'fonts'
        ),
        path.join(
          PUBLIC,
          'fonts'
        ),
        fontsDir
      ];

      const fontDir =
        possibleFontDirs.find(
          dir => {
            try {
              return true;
            } catch {
              return false;
            }
          }
        ) ||
        fontsDir;

      const subtitleFilterWithFont =
        `${subtitleFilter}:fontsdir='${escapeFilterPath(
          fontDir
        )}'`;

      /* -------------------------
         Video filter
      ------------------------- */

      let videoFilter = '';

      if (
        regions.length
      ) {
        const blur =
          makeBlurFilter(
            regions
          );

        videoFilter =
          `${blur};` +
          `[video]${subtitleFilterWithFont}[outv]`;
      } else {
        videoFilter =
          `[0:v]${subtitleFilterWithFont}[outv]`;
      }

      /* -------------------------
         FFmpeg Render
      ------------------------- */

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

      /* -------------------------
         Download MP4
      ------------------------- */

      res.download(
        output,
        'Burmese-YNT-SRT.mp4',
        async () => {
          cleanup(video);
          cleanup(ass);
          cleanup(output);
        }
      );
    } catch (error) {
      console.error(
        'RENDER ERROR:',
        error
      );

      cleanup(video);
      cleanup(ass);
      cleanup(output);

      if (
        !res.headersSent
      ) {
        res
          .status(400)
          .json({
            ok: false,
            error:
              error?.message ||
              'Video render မအောင်မြင်ပါ'
          });
      }
    }
  }
);

/* =========================
   UNKNOWN API
========================= */

app.use(
  '/api',
  (_req, res) => {
    res
      .status(404)
      .json({
        ok: false,
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
      'SRT MODE: ENABLED'
    );

    console.log(
      'MP4 RENDER: ENABLED'
    );

    console.log(
      'BLUR AREA: ENABLED'
    );

    console.log(
      'CUSTOM SUBTITLE BOX: ENABLED'
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
  }
);
