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
const TMP = path.join(os.tmpdir(), 'burmese-ynt-srt');
const FONT_DIR = path.join(TMP, 'fonts');
const MYANMAR_FONT = path.join(FONT_DIR, 'NotoSansMyanmar.ttf');

await fs.mkdir(TMP, { recursive: true });
await fs.mkdir(FONT_DIR, { recursive: true });

async function ensureMyanmarFont() {
  try {
    await fs.access(MYANMAR_FONT);
    return;
  } catch {}

  const urls = [
    'https://raw.githubusercontent.com/google/fonts/main/ofl/notosansmyanmar/NotoSansMyanmar%5Bwdth%2Cwght%5D.ttf',
    'https://raw.githubusercontent.com/google/fonts/main/ofl/notosansmyanmar/NotoSansMyanmar-Regular.ttf'
  ];

  for (const url of urls) {
    try {
      const response = await fetch(url);
      if (!response.ok) continue;
      const buffer = Buffer.from(await response.arrayBuffer());
      if (buffer.length > 100000) {
        await fs.writeFile(MYANMAR_FONT, buffer);
        console.log('Myanmar font ready.');
        return;
      }
    } catch (error) {
      console.log('Myanmar font download skipped:', error?.message || error);
    }
  }
}

await ensureMyanmarFont();

const GROQ_MODEL = 'whisper-large-v3';
const GEMINI_MODEL = 'gemini-3.8-flash';
const THUMBNAIL_ANALYSIS_MODEL = 'gemini-3.8-flash';
const THUMBNAIL_IMAGE_MODEL = 'gemini-3.1-flash-image';

const allowedExt = new Set([
  '.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.flv', '.wmv',
  '.mpeg', '.mpg', '.mp3', '.wav', '.m4a', '.ogg', '.opus'
]);

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, TMP),
    filename: (_req, file, cb) => {
      const ext0 = path.extname(file.originalname || '').toLowerCase();
      const ext = allowedExt.has(ext0) ? ext0 : '.mp4';
      cb(null, `${Date.now()}-${crypto.randomBytes(8).toString('hex')}${ext}`);
    }
  }),
  limits: { fileSize: MAX_BYTES, files: 1 },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase();
    if (!allowedExt.has(ext)) return cb(new Error('MP4 / MOV / MKV / WEBM video ကိုသုံးပါ။'));
    cb(null, true);
  }
});

app.use(express.json({ limit: '12mb' }));
app.use(express.urlencoded({ extended: true, limit: '12mb' }));
app.use(express.static(PUBLIC));

function getKey(req, headerName, bodyName, envName, label) {
  const value = String(
    req.get(headerName) || req.body?.[bodyName] || process.env[envName] || ''
  ).trim();
  if (!value) throw new Error(`${label} မရှိပါ`);
  return value;
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

function runProcess(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args);
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', d => { stdout += d.toString(); });
    child.stderr.on('data', d => { stderr += d.toString(); });
    child.on('error', reject);
    child.on('close', code => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(stderr || `${command} exited with code ${code}`));
    });
  });
}

async function probeVideo(file) {
  const result = await runProcess(ffprobeStatic.path, [
    '-v', 'error',
    '-show_entries', 'format=duration:stream=index,codec_type,width,height',
    '-of', 'json',
    file
  ]);
  const data = JSON.parse(result.stdout);
  const duration = Number(data?.format?.duration || 0);
  const video = (data?.streams || []).find(s => s.codec_type === 'video');
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error('Video duration မဖတ်နိုင်ပါ');
  }
  return {
    duration,
    width: Number(video?.width || 0),
    height: Number(video?.height || 0)
  };
}

async function extractAudio(videoPath) {
  const out = path.join(TMP, `${crypto.randomUUID()}.wav`);
  await runProcess(ffmpegStatic, [
    '-y', '-i', videoPath,
    '-map', '0:a:0',
    '-vn',
    '-ar', '16000',
    '-ac', '1',
    '-c:a', 'pcm_s16le',
    out
  ]);
  return out;
}

async function transcribeGroq(audioPath, apiKey) {
  const groq = new Groq({ apiKey });
  return groq.audio.transcriptions.create({
    file: createReadStream(audioPath),
    model: GROQ_MODEL,
    response_format: 'verbose_json',
    timestamp_granularities: ['word', 'segment'],
    temperature: 0
  });
}

function normalizeWord(item) {
  const word = cleanText(item?.word);
  const start = Number(item?.start);
  const end = Number(item?.end);
  if (!word || !Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  return { word, start, end };
}

function endsSentence(text) {
  return /[.!?。！？…]+$/.test(String(text || '').trim());
}

function makePreciseSegments(result, duration) {
  const rawWords = Array.isArray(result?.words) ? result.words : [];
  const words = rawWords.map(normalizeWord).filter(Boolean);

  if (!words.length) {
    const raw = Array.isArray(result?.segments) ? result.segments : [];
    return raw.map((s, i) => ({
      id: i + 1,
      start: Math.max(0, Number(s.start) || 0),
      end: Math.min(duration, Number(s.end) || 0),
      text: cleanText(s.text)
    })).filter(s => s.text && s.end > s.start);
  }

  const output = [];
  let current = [];
  let currentStart = 0;
  let currentEnd = 0;

  const flush = () => {
    if (!current.length) return;
    const text = current.map(x => x.word).join(' ')
      .replace(/\s+([,.!?;:，。！？；：])/g, '$1')
      .trim();
    if (text && currentEnd > currentStart) {
      output.push({
        id: output.length + 1,
        start: currentStart,
        end: Math.min(duration, currentEnd),
        text
      });
    }
    current = [];
  };

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const prev = words[i - 1];
    const gap = prev ? Math.max(0, w.start - prev.end) : 0;

    if (!current.length) {
      currentStart = w.start;
      currentEnd = w.end;
      current.push(w);
      continue;
    }

    const currentText = current.map(x => x.word).join(' ')
      .replace(/\s+([,.!?;:，。！？；：])/g, '$1')
      .trim();
    const candidate = `${currentText} ${w.word}`
      .replace(/\s+([,.!?;:，。！？；：])/g, '$1')
      .trim();

    const shouldBreak =
      gap >= 0.55 ||
      endsSentence(currentText) ||
      (w.end - currentStart) >= 5.0 ||
      candidate.length > 52 ||
      current.length >= 11;

    if (shouldBreak) {
      flush();
      currentStart = w.start;
      currentEnd = w.end;
      current.push(w);
    } else {
      current.push(w);
      currentEnd = w.end;
    }
  }
  flush();

  return removeDuplicateOverlap(output);
}

function similarity(a, b) {
  const aa = cleanText(a).toLowerCase();
  const bb = cleanText(b).toLowerCase();
  if (!aa || !bb) return 0;
  if (aa === bb) return 1;
  const sa = new Set(aa.split(/\s+/));
  const sb = new Set(bb.split(/\s+/));
  const inter = [...sa].filter(x => sb.has(x)).length;
  return inter / Math.max(sa.size, sb.size);
}

function removeDuplicateOverlap(segments) {
  const out = [];
  for (const s of segments) {
    const prev = out[out.length - 1];
    if (!prev) {
      out.push(s);
      continue;
    }

    if (s.start < prev.end) {
      if (similarity(s.text, prev.text) >= 0.75) {
        prev.end = Math.max(prev.end, s.end);
        continue;
      }
      s.start = prev.end;
      if (s.end <= s.start) continue;
    }

    out.push(s);
  }
  return out.map((s, i) => ({ ...s, id: i + 1 }));
}

function extractJson(text) {
  const raw = String(text || '').trim();
  const candidates = [
    raw,
    raw.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/```\s*$/i, '').trim()
  ];

  for (const value of candidates) {
    try { return JSON.parse(value); } catch {}
  }

  const a = raw.indexOf('[');
  const b = raw.lastIndexOf(']');
  if (a >= 0 && b > a) {
    try { return JSON.parse(raw.slice(a, b + 1)); } catch {}
  }
  return null;
}

function keepMyanmarOnly(text) {
  let value = String(text || '').trim();

  // Remove non-Myanmar scripts/Latin words while preserving digits and common punctuation.
  value = value
    .replace(/[A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]/g, '')
    .replace(/[^\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF\u104A\u104B\u104C\u104D\u104E\u104F0-9၀-၉၊။!?,.؟…'"“”‘’()\-\s]/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim();

  return value;
}

async function generateGemini(ai, prompt) {
  const maxRetries = 4;
  let last = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await ai.models.generateContent({
        model: GEMINI_MODEL,
        contents: prompt,
        config: {
          temperature: 0.15,
          responseMimeType: 'application/json',
          maxOutputTokens: 8192
        }
      });
    } catch (error) {
      last = error;
      const msg = String(error?.message || error).toLowerCase();
      const retryable =
        /429|resource_exhausted|rate.?limit|503|unavailable|high demand|500|502|504|timeout/.test(msg);
      if (!retryable || attempt >= maxRetries) throw error;
      await sleep(Math.min(1200 * (2 ** attempt) + Math.floor(Math.random() * 400), 12000));
    }
  }
  throw last;
}

async function translateChunk(segments, apiKey) {
  const ai = new GoogleGenAI({ apiKey });
  const input = segments.map(s => ({ id: s.id, text: s.text }));

  const prompt = `You are a professional Myanmar Burmese subtitle translator.

Translate EVERY spoken-dialogue line into natural, clear Myanmar Unicode.

STRICT RULES:
1. Return ONLY a JSON array.
2. Each item must be exactly: {"id": number, "translation": "မြန်မာစာ"}.
3. Keep every id exactly.
4. Do not remove, merge, reorder, duplicate, or invent any line.
5. Translate ONLY the spoken dialogue.
6. The translation field MUST contain Myanmar Burmese only.
7. Do NOT output English, Chinese, Japanese, Korean, Thai or any other script.
8. Do not leave foreign words unchanged. Transliterate names/places into natural Myanmar when needed.
9. Do not add explanations, notes, labels, emojis, or quotation marks.
10. Keep each subtitle concise and natural.
11. Never return the original language.
12. If the source is unclear, still produce the best Myanmar translation instead of copying the source.

INPUT:
${JSON.stringify(input)}`;

  const response = await generateGemini(ai, prompt);
  const parsed = extractJson(response?.text || '');
  const list = Array.isArray(parsed) ? parsed : [];
  if (!list.length) throw new Error('Gemini က valid Myanmar translation JSON မပြန်ပါ');

  const byId = new Map();
  for (const item of list) {
    const id = Number(item?.id);
    const translation = keepMyanmarOnly(item?.translation);
    if (Number.isFinite(id) && translation) byId.set(id, translation);
  }

  return segments.map(s => ({
    ...s,
    translation: byId.get(s.id) || ''
  }));
}

async function translateAll(segments, apiKey) {
  const chunkSize = 30;
  const out = [];
  for (let i = 0; i < segments.length; i += chunkSize) {
    const chunk = segments.slice(i, i + chunkSize);
    out.push(...await translateChunk(chunk, apiKey));
  }
  return out;
}

function splitSubtitleText(text, maxChars = 46, maxWords = 11) {
  const value = cleanText(text);
  if (!value) return [];
  if (value.length <= maxChars && value.split(/\s+/).length <= maxWords) return [value];

  const words = value.split(/\s+/).filter(Boolean);
  const parts = [];
  let current = '';

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (current && (next.length > maxChars || next.split(/\s+/).length > maxWords)) {
      parts.push(current);
      current = word;
    } else {
      current = next;
    }
  }
  if (current) parts.push(current);
  return parts;
}

function buildMyanmarSegments(translated) {
  const out = [];
  let id = 1;

  for (const s of translated) {
    const text = keepMyanmarOnly(s.translation);
    if (!text) continue;

    const parts = splitSubtitleText(text);
    if (parts.length <= 1) {
      out.push({ id: id++, start: s.start, end: s.end, text });
      continue;
    }

    const total = parts.reduce((sum, p) => sum + Math.max(1, p.length), 0);
    const duration = Math.max(0.2, s.end - s.start);
    let cursor = s.start;

    parts.forEach((part, index) => {
      const end = index === parts.length - 1
        ? s.end
        : cursor + duration * (Math.max(1, part.length) / total);

      out.push({
        id: id++,
        start: cursor,
        end: Math.min(end, s.end),
        text: part
      });
      cursor = Math.min(end, s.end);
    });
  }

  return removeDuplicateOverlap(out);
}

function srtTime(seconds) {
  const ms = Math.max(0, Math.round(Number(seconds || 0) * 1000));
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const milli = ms % 1000;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(milli).padStart(3, '0')}`;
}

function makeSrt(segments) {
  return segments.map((s, i) =>
    `${i + 1}\n${srtTime(s.start)} --> ${srtTime(s.end)}\n${s.text}\n`
  ).join('\n');
}

function assTime(seconds) {
  const cs = Math.max(0, Math.round(Number(seconds || 0) * 100));
  const h = Math.floor(cs / 360000);
  const m = Math.floor((cs % 360000) / 6000);
  const s = Math.floor((cs % 6000) / 100);
  const c = cs % 100;
  return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(c).padStart(2, '0')}`;
}

function assEscape(text) {
  return String(text || '')
    .replace(/\\/g, '\\N')
    .replace(/\r?\n/g, '\\N')
    .replace(/\{/g, '\\{')
    .replace(/\}/g, '\\}');
}

function hexToAssColor(hex, alpha = '00') {
  const clean = String(hex || '#FFFFFF').replace('#', '');
  const r = clean.slice(0, 2) || 'FF';
  const g = clean.slice(2, 4) || 'FF';
  const b = clean.slice(4, 6) || 'FF';
  return `&H${alpha}${b}${g}${r}`;
}

function buildAss(segments, options, width, height) {
  const fontSize = Math.max(22, Math.min(80, Number(options.fontSize) || 42));
  const outline = Math.max(0, Math.min(8, Number(options.outline) || 3));
  const position = ['top', 'middle', 'bottom'].includes(options.position) ? options.position : 'bottom';
  const alignment = position === 'top' ? 8 : position === 'middle' ? 5 : 2;
  const marginV = position === 'top' ? 55 : position === 'middle' ? 0 : 55;
  const color = hexToAssColor(options.color || '#FFFFFF');
  const border = hexToAssColor('#000000');
  const font = 'Noto Sans Myanmar';

  const events = segments.map(s =>
    `Dialogue: 0,${assTime(s.start)},${assTime(s.end)},Default,,0,0,0,,${assEscape(s.text)}`
  ).join('\n');

  return `[Script Info]
ScriptType: v4.00+
PlayResX: ${width || 1920}
PlayResY: ${height || 1080}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,${font},${fontSize},${color},${color},${border},&H99000000&,0,0,0,0,100,100,0,0,1,${outline},1,${alignment},50,50,${marginV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
${events}
`;
}

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(v) || 0));
}

function safeBlurRegions(regions, width, height) {
  if (!Array.isArray(regions) || !width || !height) return [];
  return regions.slice(0, 3).map(r => {
    const x = Math.round(clamp01(r.x) * width);
    const y = Math.round(clamp01(r.y) * height);
    const w = Math.max(8, Math.round(clamp01(r.w) * width));
    const h = Math.max(8, Math.round(clamp01(r.h) * height));
    return {
      x: Math.min(x, Math.max(0, width - 8)),
      y: Math.min(y, Math.max(0, height - 8)),
      w: Math.min(w, width),
      h: Math.min(h, height)
    };
  }).filter(r => r.w >= 8 && r.h >= 8);
}

function makeBlurFilter(regions) {
  if (!regions.length) return '[0:v]null[video]';

  let filter = '[0:v]split=' + (regions.length + 1);
  for (let i = 0; i < regions.length + 1; i++) filter += `[s${i}]`;
  filter += ';';

  let base = '[s0]';
  for (let i = 0; i < regions.length; i++) {
    const r = regions[i];
    const crop = `[s${i + 1}]crop=${r.w}:${r.h}:${r.x}:${r.y},boxblur=18:2[b${i}]`;
    filter += crop + ';';
    const next = `[o${i}]`;
    filter += `${base}[b${i}]overlay=${r.x}:${r.y}${next};`;
    base = next;
  }
  filter += `${base}null[video]`;
  return filter;
}


/* =========================================================
   RECAP THUMBNAIL STUDIO
   Separate workflow from Myanmar SRT.
========================================================= */

const thumbnailJobs = new Map();
const thumbnailImages = new Map();
const THUMBNAIL_JOB_TTL = 30 * 60 * 1000;

function rememberThumbnailJob(id, data) {
  thumbnailJobs.set(id, { ...data, createdAt: Date.now() });
  setTimeout(() => {
    const job = thumbnailJobs.get(id);
    if (!job) return;
    thumbnailJobs.delete(id);
    cleanup(job.videoPath);
  }, THUMBNAIL_JOB_TTL).unref?.();
}

function getMimeType(file) {
  const ext = path.extname(file || '').toLowerCase();
  if (ext === '.mov') return 'video/mov';
  if (ext === '.webm') return 'video/webm';
  if (ext === '.avi') return 'video/avi';
  if (ext === '.mpg' || ext === '.mpeg') return 'video/mpeg';
  return 'video/mp4';
}

async function uploadGeminiVideo(videoPath, apiKey, mimeType) {
  const ai = new GoogleGenAI({ apiKey });
  const uploaded = await ai.files.upload({
    file: videoPath,
    config: { mimeType }
  });

  let file = uploaded;
  for (let i = 0; i < 120; i++) {
    file = await ai.files.get({ name: uploaded.name });
    const state = String(file?.state || '').toUpperCase();
    if (state === 'ACTIVE') return { ai, file };
    if (state === 'FAILED') throw new Error('Gemini Video File processing failed');
    await sleep(2500);
  }
  throw new Error('Gemini Video processing အချိန်ကျော်သွားပါတယ်');
}

function extractInteractionText(interaction) {
  if (interaction?.output_text) return String(interaction.output_text);
  const steps = Array.isArray(interaction?.steps) ? interaction.steps : [];
  const texts = [];
  for (const step of steps) {
    const content = Array.isArray(step?.content) ? step.content : [];
    for (const item of content) {
      if (typeof item?.text === 'string') texts.push(item.text);
    }
  }
  return texts.join('\n');
}

function extractJsonObject(text) {
  const raw = String(text || '').trim();
  const candidates = [
    raw,
    raw.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/```\s*$/i, '').trim()
  ];
  for (const value of candidates) {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === 'object') return parsed;
    } catch {}
  }
  const a = raw.indexOf('{');
  const b = raw.lastIndexOf('}');
  if (a >= 0 && b > a) {
    try {
      const parsed = JSON.parse(raw.slice(a, b + 1));
      if (parsed && typeof parsed === 'object') return parsed;
    } catch {}
  }
  return null;
}

function safeTitle(value) {
  return cleanText(value)
    .replace(/[<>\\/|{}\[\]`]/g, '')
    .slice(0, 120)
    .trim();
}

function safeRatio(value) {
  return ['16:9', '9:16', '1:1'].includes(String(value)) ? String(value) : '16:9';
}

async function analyzeRecapVideo(videoPath, apiKey, mimeType) {
  const { ai, file } = await uploadGeminiVideo(videoPath, apiKey, mimeType);
  const prompt = `
You are analyzing a short drama/movie recap video for a social-media cover thumbnail.
Watch and understand the WHOLE video, including visual events and spoken dialogue.
Return ONLY valid JSON with this exact structure:
{
  "title": "one short compelling Myanmar Unicode recap title",
  "summary": "one concise Myanmar summary of the story",
  "thumbnailScene": "MM:SS",
  "thumbnailReason": "why this moment is the strongest cover moment"
}

Rules:
- The title MUST be natural Myanmar Unicode.
- Do not use English, Chinese, Japanese, Korean or Thai in the title.
- Keep the title short enough for a thumbnail (ideally 6-16 Myanmar words).
- Make it describe the actual story, not a generic title.
- Pick the most visually/emotionally important moment for thumbnailScene.
- Use a real timestamp from the video in MM:SS format.
- Do not invent characters or events that are not in the video.
`;

  const interaction = await ai.interactions.create({
    model: THUMBNAIL_ANALYSIS_MODEL,
    input: [
      { type: 'video', uri: file.uri, mime_type: file.mimeType || mimeType },
      { type: 'text', text: prompt }
    ]
  });

  const parsed = extractJsonObject(extractInteractionText(interaction));
  if (!parsed) throw new Error('Gemini က Thumbnail Analysis JSON မပြန်ပါ');

  return {
    title: safeTitle(parsed.title) || 'ဒီဇာတ်လမ်းရဲ့ မမျှော်လင့်ထားတဲ့ အလှည့်အပြောင်း',
    summary: cleanText(parsed.summary),
    thumbnailScene: /^\d{1,2}:\d{2}$/.test(String(parsed.thumbnailScene || ''))
      ? String(parsed.thumbnailScene)
      : '00:00',
    thumbnailReason: cleanText(parsed.thumbnailReason),
    fileUri: file.uri,
    mimeType: file.mimeType || mimeType
  };
}

async function generateRecapThumbnail(videoPath, apiKey, mimeType, title, style, ratio, scene, summary) {
  const { ai, file } = await uploadGeminiVideo(videoPath, apiKey, mimeType);
  const styleText = {
    dramatic: 'high-impact Korean drama recap poster, cinematic lighting, dramatic faces, rich contrast, premium streaming poster feel',
    emotional: 'emotional Korean drama recap poster, expressive faces, soft cinematic lighting, strong emotional storytelling',
    mystery: 'mystery thriller recap poster, suspenseful composition, moody cinematic lighting, dramatic shadows',
    romance: 'romantic Korean drama recap poster, elegant cinematic lighting, expressive couple-focused composition'
  }[String(style)] || 'cinematic Korean drama recap poster';

  const prompt = `
Create ONE polished social-media recap cover image from this video.
Focus primarily on the strongest important visual moment around ${scene || '00:00'}.
Use the video as the visual source and preserve the actual characters, setting, wardrobe and events shown in it.

STYLE:
${styleText}

STORY CONTEXT:
${summary || 'Use the actual video context.'}

TITLE TO INSPIRE THE COMPOSITION:
${title}

IMPORTANT:
- Create a clean cover image with the characters/scenes as the focal point.
- Leave a clear lower-third or central negative-space area where a title can be placed later.
- DO NOT render any words, letters, captions, logos, watermarks, UI, or fake text inside the image.
- Do not invent a different story.
- Make it look like a high-quality drama/movie recap thumbnail, not a generic AI poster.
`;

  const interaction = await ai.interactions.create({
    model: THUMBNAIL_IMAGE_MODEL,
    input: [
      { type: 'video', uri: file.uri, mime_type: file.mimeType || mimeType },
      { type: 'text', text: prompt }
    ],
    response_format: {
      type: 'image',
      mime_type: 'image/png',
      aspect_ratio: safeRatio(ratio),
      image_size: '1K'
    }
  });

  const image = interaction?.output_image;
  if (!image?.data) throw new Error('Gemini က Thumbnail Image မထုတ်ပေးနိုင်ပါ');
  return Buffer.from(image.data, 'base64');
}

function wrapThumbnailTitle(title, maxChars) {
  const words = String(title || '').trim().split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (line && next.length > maxChars) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines.slice(0, 2).join('\n');
}

async function burnThumbnailTitle(imageBuffer, title, ratio) {
  const imagePath = path.join(TMP, `${crypto.randomUUID()}.png`);
  const titlePath = path.join(TMP, `${crypto.randomUUID()}.txt`);
  const outputPath = path.join(TMP, `${crypto.randomUUID()}-thumbnail.png`);

  await fs.writeFile(imagePath, imageBuffer);

  const maxChars = ratio === '9:16' ? 15 : ratio === '1:1' ? 19 : 24;
  const wrappedTitle = wrapThumbnailTitle(title, maxChars);
  await fs.writeFile(titlePath, wrappedTitle, 'utf8');

  const fontSize = ratio === '9:16' ? 44 : ratio === '1:1' ? 48 : 54;
  const boxHeight = ratio === '9:16' ? 285 : ratio === '1:1' ? 205 : 180;
  const bottomMargin = ratio === '9:16' ? 55 : ratio === '1:1' ? 42 : 38;
  const fontFile = escapeFilterPath(MYANMAR_FONT);
  const titleFile = escapeFilterPath(titlePath);

  await runProcess(ffmpegStatic, [
    '-y',
    '-i', imagePath,
    '-vf',
    `drawbox=x=0:y=h-${boxHeight}:w=iw:h=${boxHeight}:color=black@0.38:t=fill,` +
    `drawtext=fontfile='${fontFile}':textfile='${titleFile}':fontcolor=0xFFE76A:fontsize=${fontSize}:line_spacing=8:borderw=4:bordercolor=black@0.95:x=(w-text_w)/2:y=h-text_h-${bottomMargin}:shadowx=2:shadowy=2`,
    '-frames:v', '1',
    '-c:v', 'png',
    outputPath
  ]);

  const output = await fs.readFile(outputPath);
  cleanup(imagePath);
  cleanup(titlePath);
  cleanup(outputPath);
  return output;
}

function cleanup(file) {
  if (file) fs.rm(file, { force: true }).catch(() => {});
}

function escapeFilterPath(file) {
  return file.replace(/\\/g, '/').replace(/:/g, '\\:').replace(/'/g, "\\'");
}

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'burmese-ynt-srt',
    groqModel: GROQ_MODEL,
    geminiModel: GEMINI_MODEL,
    burnVideo: true,
    blurOriginal: true
  });
});

app.post('/api/transcribe', upload.single('video'), async (req, res) => {
  const video = req.file?.path;
  let audio = null;

  try {
    if (!video) throw new Error('Video file မရှိပါ');
    const groqKey = getKey(req, 'x-groq-api-key', 'groqApiKey', 'GROQ_API_KEY', 'Groq API Key');

    const info = await probeVideo(video);
    if (info.duration > MAX_SECONDS) throw new Error('Video က 5 မိနစ်ထက်ကျော်နေပါတယ်');
    if (!info.width || !info.height) throw new Error('Video size မဖတ်နိုင်ပါ');

    audio = await extractAudio(video);
    const result = await transcribeGroq(audio, groqKey);
    const transcript = makePreciseSegments(result, info.duration);

    if (!transcript.length) throw new Error('Groq က စကားပြောစာတန်း မထုတ်ပေးနိုင်ပါ');

    res.json({
      ok: true,
      duration: info.duration,
      width: info.width,
      height: info.height,
      language: result?.language || 'auto',
      text: result?.text || transcript.map(x => x.text).join(' '),
      transcript
    });
  } catch (error) {
    console.error('TRANSCRIBE ERROR:', error);
    res.status(400).json({ ok: false, error: error?.message || 'Groq Transcript Error' });
  } finally {
    cleanup(video);
    cleanup(audio);
  }
});

app.post('/api/translate', async (req, res) => {
  try {
    const geminiKey = getKey(req, 'x-gemini-api-key', 'geminiApiKey', 'GEMINI_API_KEY', 'Gemini API Key');
    const input = Array.isArray(req.body?.transcript) ? req.body.transcript : [];
    if (!input.length) throw new Error('Transcript မရှိပါ');

    const normalized = input.map((s, i) => ({
      id: i + 1,
      start: Number(s.start) || 0,
      end: Number(s.end) || 0,
      text: cleanText(s.text)
    })).filter(s => s.text && s.end > s.start);

    const translated = await translateAll(normalized, geminiKey);
    const transcript = buildMyanmarSegments(translated);

    if (!transcript.length) throw new Error('မြန်မာစာ ဘာသာပြန်ရလဒ် မရပါ');

    res.json({
      ok: true,
      transcript,
      srt: makeSrt(transcript)
    });
  } catch (error) {
    console.error('TRANSLATE ERROR:', error);
    res.status(400).json({ ok: false, error: error?.message || 'Gemini Myanmar Translation Error' });
  }
});


app.post('/api/thumbnail/analyze', upload.single('video'), async (req, res) => {
  const video = req.file?.path;
  try {
    if (!video) throw new Error('Recap Video မရှိပါ');
    const geminiKey = getKey(req, 'x-gemini-api-key', 'geminiApiKey', 'GEMINI_API_KEY', 'Gemini API Key');
    const info = await probeVideo(video);
    if (info.duration > MAX_SECONDS) throw new Error('Recap Video က 5 မိနစ်ထက်ကျော်နေပါတယ်');

    const mimeType = getMimeType(video);
    const analysis = await analyzeRecapVideo(video, geminiKey, mimeType);
    const jobId = crypto.randomUUID();

    rememberThumbnailJob(jobId, {
      videoPath: video,
      mimeType,
      duration: info.duration,
      width: info.width,
      height: info.height,
      title: analysis.title,
      summary: analysis.summary,
      thumbnailScene: analysis.thumbnailScene,
      thumbnailReason: analysis.thumbnailReason
    });

    res.json({
      ok: true,
      jobId,
      title: analysis.title,
      summary: analysis.summary,
      thumbnailScene: analysis.thumbnailScene,
      thumbnailReason: analysis.thumbnailReason
    });
  } catch (error) {
    console.error('THUMBNAIL ANALYZE ERROR:', error);
    cleanup(video);
    res.status(400).json({ ok: false, error: error?.message || 'Thumbnail Analysis Error' });
  }
});

app.post('/api/thumbnail/generate', async (req, res) => {
  try {
    const jobId = String(req.body?.jobId || '').trim();
    const job = thumbnailJobs.get(jobId);
    if (!job) throw new Error('Thumbnail Analysis session မတွေ့ပါ။ Video ကို Analyze ပြန်လုပ်ပါ');

    const geminiKey = getKey(req, 'x-gemini-api-key', 'geminiApiKey', 'GEMINI_API_KEY', 'Gemini API Key');
    const title = safeTitle(req.body?.title) || job.title;
    const style = String(req.body?.style || 'dramatic');
    const ratio = safeRatio(req.body?.ratio);

    const imageBuffer = await generateRecapThumbnail(
      job.videoPath,
      geminiKey,
      job.mimeType,
      title,
      style,
      ratio,
      job.thumbnailScene,
      job.summary
    );

    const finalImage = await burnThumbnailTitle(imageBuffer, title, ratio);
    const imageId = crypto.randomUUID();
    const imagePath = path.join(TMP, `${imageId}.png`);
    await fs.writeFile(imagePath, finalImage);
    thumbnailImages.set(imageId, { imagePath, filename: 'Recap-Thumbnail.png', createdAt: Date.now() });

    setTimeout(() => {
      const item = thumbnailImages.get(imageId);
      if (!item) return;
      thumbnailImages.delete(imageId);
      cleanup(item.imagePath);
    }, THUMBNAIL_JOB_TTL).unref?.();

    thumbnailJobs.delete(jobId);
    cleanup(job.videoPath);

    res.json({
      ok: true,
      imageUrl: `/api/thumbnail/download/${imageId}`,
      filename: 'Recap-Thumbnail.png',
      title,
      ratio,
      thumbnailScene: job.thumbnailScene
    });
  } catch (error) {
    console.error('THUMBNAIL GENERATE ERROR:', error);
    res.status(400).json({ ok: false, error: error?.message || 'Thumbnail Generate Error' });
  }
});

app.get('/api/thumbnail/download/:id', async (req, res) => {
  const item = thumbnailImages.get(String(req.params.id || ''));
  if (!item) return res.status(404).json({ ok: false, error: 'Thumbnail မတွေ့ပါ' });
  try {
    await fs.access(item.imagePath);
    res.download(item.imagePath, item.filename, () => {
      thumbnailImages.delete(String(req.params.id || ''));
      cleanup(item.imagePath);
    });
  } catch (error) {
    thumbnailImages.delete(String(req.params.id || ''));
    cleanup(item.imagePath);
    res.status(404).json({ ok: false, error: 'Thumbnail Download မရပါ' });
  }
});

app.post('/api/render', upload.single('video'), async (req, res) => {
  const video = req.file?.path;
  let ass = null;
  let output = null;

  try {
    if (!video) throw new Error('Video file မရှိပါ');

    const info = await probeVideo(video);
    if (info.duration > MAX_SECONDS) throw new Error('Video က 5 မိနစ်ထက်ကျော်နေပါတယ်');

    const segments = Array.isArray(req.body?.segments)
      ? req.body.segments
      : JSON.parse(String(req.body?.segments || '[]'));

    if (!segments.length) throw new Error('Myanmar Subtitle မရှိပါ');

    const options = {
      fontSize: Number(req.body?.fontSize) || 42,
      outline: Number(req.body?.outline) || 3,
      color: String(req.body?.color || '#FFFFFF'),
      position: String(req.body?.position || 'bottom')
    };

    const regions = safeBlurRegions(
      JSON.parse(String(req.body?.blurRegions || '[]')),
      info.width,
      info.height
    );

    ass = path.join(TMP, `${crypto.randomUUID()}.ass`);
    output = path.join(TMP, `${crypto.randomUUID()}.mp4`);

    const assText = buildAss(segments, options, info.width, info.height);
    await fs.writeFile(ass, assText, 'utf8');

    const fontsDir = FONT_DIR;
    const subtitleFilter = `subtitles=filename='${escapeFilterPath(ass)}':fontsdir='${escapeFilterPath(fontsDir)}'`;

    let videoFilter = '';
    if (regions.length) {
      const blur = makeBlurFilter(regions);
      videoFilter = `${blur};[video]${subtitleFilter}[outv]`;
    } else {
      videoFilter = `[0:v]${subtitleFilter}[outv]`;
    }

    await runProcess(ffmpegStatic, [
      '-y',
      '-i', video,
      '-filter_complex', videoFilter,
      '-map', '[outv]',
      '-map', '0:a?',
      '-c:v', 'libx264',
      '-preset', 'veryfast',
      '-crf', '22',
      '-c:a', 'aac',
      '-b:a', '128k',
      '-movflags', '+faststart',
      output
    ]);

    res.download(output, 'Burmese-YNT-SRT.mp4', async () => {
      cleanup(video);
      cleanup(ass);
      cleanup(output);
    });
  } catch (error) {
    console.error('RENDER ERROR:', error);
    cleanup(video);
    cleanup(ass);
    cleanup(output);
    res.status(400).json({ ok: false, error: error?.message || 'Video render မအောင်မြင်ပါ' });
  }
});

app.use('/api', (_req, res) => {
  res.status(404).json({ ok: false, error: 'API endpoint မတွေ့ပါ' });
});

app.use((error, _req, res, _next) => {
  console.error('SERVER ERROR:', error);
  if (res.headersSent) return;
  res.status(500).json({ ok: false, error: error?.message || 'Server Error' });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Burmese YNT SRT server running on port ${PORT}`);
  console.log(`Groq: ${GROQ_MODEL}`);
  console.log(`Gemini: ${GEMINI_MODEL}`);
  console.log('Video Burn: ENABLED');
  console.log('Original Text Blur: ENABLED');
});
