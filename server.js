// ============================================================
// Burmese SRT + One Clips Movie Recap Server
// ============================================================

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const { spawn } = require('child_process');

const app = express();

const PORT = process.env.PORT || 10000;
const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const WORK_DIR = path.join(os.tmpdir(), 'myanmar-srt-work');

fs.mkdirSync(WORK_DIR, { recursive: true });

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({
  extended: true,
  limit: '50mb'
}));
app.use(express.static(PUBLIC_DIR));

const upload = multer({
  dest: WORK_DIR,
  limits: {
    fileSize: 1024 * 1024 * 1024
  }
});

// ============================================================
// Helpers
// ============================================================

function uid(prefix = 'job') {
  return `${prefix}_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
}

function clamp(value, min, max) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return min;
  }

  return Math.max(min, Math.min(max, n));
}

function jsonError(res, status, message, extra = {}) {
  return res.status(status).json({
    ok: false,
    error: message,
    ...extra
  });
}

// Website ကပို့တဲ့ API key ကိုသုံးမယ်.
// မရှိရင် Render Environment Variable ကိုသုံးမယ်.
function getGroqKey(req) {
  return String(
    req.headers['x-groq-api-key'] ||
    req.body?.groqApiKey ||
    process.env.GROQ_API_KEY ||
    ''
  ).trim();
}

function getGeminiKey(req) {
  return String(
    req.headers['x-gemini-api-key'] ||
    req.body?.geminiApiKey ||
    process.env.GEMINI_API_KEY ||
    ''
  ).trim();
}

// ============================================================
// Process / FFmpeg
// ============================================================

function runProcess(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT,
      env: process.env
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', data => {
      stdout += data.toString();
    });

    child.stderr.on('data', data => {
      stderr += data.toString();
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
            `${command} exited with code ${code}\n${stderr.slice(-10000)}`
          )
        );
      }
    });
  });
}

async function ffmpeg(args) {
  return runProcess(
    'ffmpeg',
    [
      '-y',
      '-hide_banner',
      '-loglevel',
      'error',
      ...args
    ]
  );
}

async function ffprobe(args) {
  return runProcess('ffprobe', args);
}

async function videoDuration(file) {
  const result = await ffprobe([
    '-v',
    'error',
    '-show_entries',
    'format=duration',
    '-of',
    'default=noprint_wrappers=1:nokey=1',
    file
  ]);

  const duration = Number(result.stdout.trim());

  if (!Number.isFinite(duration)) {
    throw new Error('Video duration မဖတ်နိုင်ပါ။');
  }

  return duration;
}

async function videoSize(file) {
  const result = await ffprobe([
    '-v',
    'error',
    '-select_streams',
    'v:0',
    '-show_entries',
    'stream=width,height',
    '-of',
    'csv=p=0:s=x',
    file
  ]);

  const parts = result.stdout.trim().split('x');

  const width = Number(parts[0]);
  const height = Number(parts[1]);

  return {
    width: Number.isFinite(width) ? width : 1920,
    height: Number.isFinite(height) ? height : 1080
  };
}

// ============================================================
// Gemini
// ============================================================

async function geminiGenerate(model, body, apiKey) {
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY မရှိပါ။');
  }

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${encodeURIComponent(model)}:generateContent?key=` +
    `${encodeURIComponent(apiKey)}`;

  const response = await fetch(url, {
    method: 'POST',

    headers: {
      'Content-Type': 'application/json'
    },

    body: JSON.stringify(body)
  });

  const raw = await response.text();

  let data;

  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error(
      `Gemini JSON မရပါ။ HTTP ${response.status}: ${raw.slice(0, 1000)}`
    );
  }

  if (!response.ok) {
    throw new Error(
      `Gemini API ${response.status}: ` +
      `${data?.error?.message || raw.slice(0, 1500)}`
    );
  }

  return data;
}

function geminiText(data) {
  return (
    data?.candidates?.[0]?.content?.parts || []
  )
    .filter(part => typeof part.text === 'string')
    .map(part => part.text)
    .join('\n')
    .trim();
}

// ============================================================
// Video Scene Analysis
// ============================================================

async function extractFrames(video, dir, duration) {
  const framesDir = path.join(dir, 'frames');

  fs.mkdirSync(framesDir, {
    recursive: true
  });

  const count = Math.min(
    24,
    Math.max(8, Math.ceil(duration / 5))
  );

  const fps =
    count / Math.max(duration, 1);

  await ffmpeg([
    '-i',
    video,

    '-vf',
    `fps=${fps.toFixed(6)},scale=768:-2`,

    '-q:v',
    '4',

    path.join(
      framesDir,
      'frame_%03d.jpg'
    )
  ]);

  return fs
    .readdirSync(framesDir)
    .filter(file => file.endsWith('.jpg'))
    .sort()
    .map(file =>
      path.join(framesDir, file)
    );
}

function cleanScript(script) {
  return String(script || '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(
      /^Myanmar\s*movie\s*recap\s*script\s*:?\s*/i,
      ''
    )
    .trim();
}

async function analyzeVideoScenes(
  video,
  jobDir,
  apiKey
) {
  const duration =
    await videoDuration(video);

  const frames =
    await extractFrames(
      video,
      jobDir,
      duration
    );

  const imageParts =
    frames.map(file => ({
      inlineData: {
        mimeType: 'image/jpeg',
        data: fs
          .readFileSync(file)
          .toString('base64')
      }
    }));

  const prompt = `
You are a professional Myanmar movie recap writer.

These are sequential frames from ONE video.

Analyze only what is actually visible.

Do not invent unrelated events.

Follow the visual order.

Identify:
- visible characters
- locations
- actions
- important events
- changes between scenes

Write a natural Myanmar movie recap narration suitable for voice-over.

It should sound like a Myanmar movie recap YouTuber.

Do not mention:
- frames
- camera
- AI
- image analysis
- technical details

Avoid excessive dialogue.

Return ONLY the Myanmar narration script.

Video duration:
${duration.toFixed(2)} seconds.
`;

  let lastError;

  const models = [
    'gemini-3.8-flash',
    'gemini-3.1-flash',
    'gemini-2.5-flash'
  ];

  for (const model of models) {
    try {
      const data =
        await geminiGenerate(
          model,

          {
            contents: [
              {
                role: 'user',

                parts: [
                  {
                    text: prompt
                  },

                  ...imageParts
                ]
              }
            ],

            generationConfig: {
              temperature: 0.55,
              maxOutputTokens: 10000
            }
          },

          apiKey
        );

      const script =
        cleanScript(
          geminiText(data)
        );

      if (script) {
        return {
          script,
          duration,
          frameCount: frames.length
        };
      }

    } catch (error) {
      lastError = error;
    }
  }

  throw (
    lastError ||
    new Error('Scene analysis failed.')
  );
}

// ============================================================
// Gemini TTS
// ============================================================

const GEMINI_VOICES = [
  'Zephyr',
  'Puck',
  'Charon',
  'Kore',
  'Fenrir',
  'Leda',
  'Orus',
  'Aoede',
  'Callirrhoe',
  'Autonoe',
  'Enceladus',
  'Iapetus',
  'Umbriel',
  'Algieba',
  'Despina',
  'Erinome',
  'Algenib',
  'Rasalgethi',
  'Laomedeia',
  'Achernar',
  'Alnilam',
  'Schedar',
  'Gacrux',
  'Pulcherrima',
  'Achird',
  'Zubenelgenubi',
  'Vindemiatrix',
  'Sadachbia',
  'Sadaltager',
  'Sulafat'
];

const TTS_MODEL =
  'gemini-3.8-flash-tts';

async function geminiTTS({
  script,
  voice = 'Kore',
  jobDir,
  speed = 1,
  pitch = 0,
  volume = 1,
  apiKey
}) {
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY မရှိပါ။'
    );
  }

  if (!String(script).trim()) {
    throw new Error(
      'Recap script မရှိပါ။'
    );
  }

  if (!GEMINI_VOICES.includes(voice)) {
    voice = 'Kore';
  }

  const response =
    await fetch(
      'https://generativelanguage.googleapis.com/v1beta/interactions',
      {
        method: 'POST',

        headers: {
          'Content-Type':
            'application/json',

          'x-goog-api-key':
            apiKey
        },

        body: JSON.stringify({
          model: TTS_MODEL,

          input: [
            {
              type: 'user_input',

              content: [
                {
                  type: 'text',

                  text: String(script),

                  annotations: [
                    {
                      type:
                        'speech_metadata',

                      style:
                        'Natural Myanmar movie recap narrator. Clear Burmese pronunciation. Smooth storytelling. Medium pace. Expressive but not theatrical.'
                    }
                  ]
                }
              ]
            }
          ],

          response_format: {
            type: 'audio'
          },

          generation_config: {
            speech_config: [
              {
                voice
              }
            ]
          }
        })
      }
    );

  const raw =
    await response.text();

  let data;

  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error(
      `Gemini TTS JSON မရပါ။ HTTP ${response.status}: ${raw.slice(0, 1500)}`
    );
  }

  if (!response.ok) {
    throw new Error(
      `Gemini TTS ${response.status}: ` +
      `${data?.error?.message || raw.slice(0, 1500)}`
    );
  }

  let base64 =
    data?.output_audio?.data;

  if (
    !base64 &&
    Array.isArray(data?.steps)
  ) {
    for (
      let i = data.steps.length - 1;
      i >= 0 && !base64;
      i--
    ) {
      const content =
        data.steps[i]?.content || [];

      for (
        let j = content.length - 1;
        j >= 0;
        j--
      ) {
        const item = content[j];

        if (
          item?.type === 'audio' &&
          item?.data
        ) {
          base64 = item.data;
          break;
        }
      }
    }
  }

  if (!base64) {
    throw new Error(
      'Gemini TTS audio data မတွေ့ပါ။'
    );
  }

  const rawWav =
    path.join(
      jobDir,
      'gemini_raw.wav'
    );

  const outputWav =
    path.join(
      jobDir,
      'voice.wav'
    );

  fs.writeFileSync(
    rawWav,
    Buffer.from(
      base64,
      'base64'
    )
  );

  const filters = [];

  const vol =
    clamp(volume, 0, 3);

  const spd =
    clamp(speed, 0.5, 2);

  const pit =
    clamp(pitch, -12, 12);

  if (
    Math.abs(vol - 1) > 0.01
  ) {
    filters.push(
      `volume=${vol}`
    );
  }

  if (
    Math.abs(pit) > 0.01
  ) {
    const ratio =
      Math.pow(
        2,
        pit / 12
      );

    filters.push(
      `asetrate=24000*${ratio.toFixed(6)}`
    );

    filters.push(
      'aresample=24000'
    );

    filters.push(
      `atempo=${(1 / ratio).toFixed(6)}`
    );
  }

  if (
    Math.abs(spd - 1) > 0.01
  ) {
    filters.push(
      `atempo=${spd}`
    );
  }

  if (filters.length) {
    await ffmpeg([
      '-i',
      rawWav,

      '-af',
      filters.join(','),

      '-ar',
      '24000',

      '-ac',
      '1',

      '-c:a',
      'pcm_s16le',

      outputWav
    ]);
  } else {
    fs.copyFileSync(
      rawWav,
      outputWav
    );
  }

  return {
    audioPath: outputWav,
    voice,
    model: TTS_MODEL
  };
}

// ============================================================
// Groq Whisper
// ============================================================

async function groqTranscribe(
  audioPath,
  script,
  apiKey
) {
  if (!apiKey) {
    throw new Error(
      'GROQ_API_KEY မရှိပါ။'
    );
  }

  const form =
    new FormData();

  form.append(
    'file',

    new Blob(
      [
        fs.readFileSync(
          audioPath
        )
      ],
      {
        type: 'audio/wav'
      }
    ),

    'voice.wav'
  );

  form.append(
    'model',
    'whisper-large-v3-turbo'
  );

  form.append(
    'response_format',
    'verbose_json'
  );

  form.append(
    'timestamp_granularities[]',
    'word'
  );

  form.append(
    'timestamp_granularities[]',
    'segment'
  );

  form.append(
    'language',
    'my'
  );

  form.append(
    'temperature',
    '0'
  );

  if (script) {
    form.append(
      'prompt',
      String(script).slice(
        0,
        800
      )
    );
  }

  const response =
    await fetch(
      'https://api.groq.com/openai/v1/audio/transcriptions',
      {
        method: 'POST',

        headers: {
          Authorization:
            `Bearer ${apiKey}`
        },

        body: form
      }
    );

  const raw =
    await response.text();

  let data;

  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error(
      `Groq JSON မရပါ။ HTTP ${response.status}: ${raw.slice(0, 1500)}`
    );
  }

  if (!response.ok) {
    throw new Error(
      `Groq ${response.status}: ` +
      `${data?.error?.message || raw.slice(0, 1500)}`
    );
  }

  return data;
}

// ============================================================
// ASS Subtitle
// ============================================================

function assTime(seconds) {
  seconds =
    Math.max(
      0,
      Number(seconds) || 0
    );

  const hours =
    Math.floor(
      seconds / 3600
    );

  const minutes =
    Math.floor(
      (seconds % 3600) / 60
    );

  const sec =
    seconds % 60;

  const centiseconds =
    Math.floor(
      (sec - Math.floor(sec)) * 100
    );

  return (
    `${hours}:` +
    `${String(minutes).padStart(2, '0')}:` +
    `${String(Math.floor(sec)).padStart(2, '0')}.` +
    `${String(centiseconds).padStart(2, '0')}`
  );
}

function assColor(hex) {
  let value =
    String(
      hex || '#FFFFFF'
    ).replace('#', '');

  if (
    !/^[0-9a-fA-F]{6}$/.test(
      value
    )
  ) {
    value = 'FFFFFF';
  }

  return (
    `&H00` +
    value.slice(4, 6) +
    value.slice(2, 4) +
    value.slice(0, 2)
  );
}

function escAss(text) {
  return String(text || '')
    .replace(/\\/g, '\\\\')
    .replace(/\{/g, '\\{')
    .replace(/\}/g, '\\}')
    .replace(/\r?\n/g, '\\N');
}

function eventsFromTranscription(
  transcription
) {
  const events = [];

  if (
    Array.isArray(
      transcription?.segments
    )
  ) {
    for (
      const segment of
      transcription.segments
    ) {
      const text =
        String(
          segment.text || ''
        ).trim();

      if (!text) {
        continue;
      }

      events.push({
        start:
          Number(
            segment.start
          ) || 0,

        end:
          Number(
            segment.end
          ) || 0,

        text
      });
    }
  }

  if (
    !events.length &&
    Array.isArray(
      transcription?.words
    )
  ) {
    let current = null;

    for (
      const word of
      transcription.words
    ) {
      const text =
        String(
          word.word || ''
        ).trim();

      if (!text) {
        continue;
      }

      const start =
        Number(word.start) || 0;

      const end =
        Number(word.end) ||
        start + 0.2;

      if (!current) {
        current = {
          start,
          end,
          text
        };
      } else {
        current.end = end;
        current.text +=
          ' ' + text;
      }

      if (
        current.text.length >= 48 ||
        current.end -
          current.start >= 4
      ) {
        events.push(current);
        current = null;
      }
    }

    if (current) {
      events.push(current);
    }
  }

  return events;
}

function assHeader({
  fontName,
  fontSize,
  outline,
  textColor,
  position,
  width,
  height,
  free
}) {
  let alignment = 2;

  if (position === 'top') {
    alignment = 8;
  }

  if (position === 'middle') {
    alignment = 5;
  }

  if (free) {
    alignment = 7;
  }

  return (
`[Script Info]
ScriptType: v4.00+
PlayResX: ${width}
PlayResY: ${height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Recap,${fontName},${fontSize},${assColor(textColor)},${assColor('#FFFFFF')},&H00000000,&H88000000,0,0,0,0,100,100,0,0,1,${outline},0,${alignment},40,40,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`
  );
}

function makeAss({
  transcription,
  file,
  fontName = 'Noto Sans Myanmar',
  fontSize = 25,
  outline = 2,
  textColor = '#FFFFFF',
  position = 'bottom',
  width = 1920,
  height = 1080,
  subtitleX,
  subtitleY
}) {
  const free =
    Number.isFinite(
      Number(subtitleX)
    ) &&
    Number.isFinite(
      Number(subtitleY)
    );

  const x =
    clamp(
      subtitleX,
      0,
      100
    ) /
    100 *
    width;

  const y =
    clamp(
      subtitleY,
      0,
      100
    ) /
    100 *
    height;

  let body =
    assHeader({
      fontName,
      fontSize,
      outline,
      textColor,
      position,
      width,
      height,
      free
    });

  for (
    const event of
    eventsFromTranscription(
      transcription
    )
  ) {
    let text =
      escAss(
        event.text
      );

    if (free) {
      text =
        `{\\pos(${Math.round(x)},${Math.round(y)})}` +
        text;
    }

    body +=
      `Dialogue: 0,${assTime(event.start)},${assTime(event.end)},Recap,,0,0,0,,${text}\n`;
  }

  fs.writeFileSync(
    file,
    body,
    'utf8'
  );
}

// ============================================================
// Blur / Output
// ============================================================

function filterPath(file) {
  return file
    .replace(/\\/g, '\\\\')
    .replace(/:/g, '\\:')
    .replace(/'/g, "\\'");
}

function blurBox({
  width,
  height,
  on,
  x,
  y,
  w,
  h
}) {
  if (!on) {
    return null;
  }

  const bw =
    Math.max(
      2,
      Math.round(
        width *
        clamp(w, 1, 100) /
        100
      )
    );

  const bh =
    Math.max(
      2,
      Math.round(
        height *
        clamp(h, 1, 100) /
        100
      )
    );

  const bx =
    Math.max(
      0,
      Math.min(
        Math.round(
          width *
          clamp(x, 0, 100) /
          100
        ),
        width - bw
      )
    );

  const by =
    Math.max(
      0,
      Math.min(
        Math.round(
          height *
          clamp(y, 0, 100) /
          100
        ),
        height - bh
      )
    );

  return {
    bw,
    bh,
    bx,
    by
  };
}

function outputSpec(
  size,
  original
) {
  if (size === '9:16') {
    return {
      width: 1080,
      height: 1920,

      scale:
        'scale=1080:1920:force_original_aspect_ratio=decrease,' +
        'pad=1080:1920:(ow-iw)/2:(oh-ih)/2'
    };
  }

  if (size === '1:1') {
    return {
      width: 1080,
      height: 1080,

      scale:
        'scale=1080:1080:force_original_aspect_ratio=decrease,' +
        'pad=1080:1080:(ow-iw)/2:(oh-ih)/2'
    };
  }

  if (size === '16:9') {
    return {
      width: 1920,
      height: 1080,

      scale:
        'scale=1920:1080:force_original_aspect_ratio=decrease,' +
        'pad=1920:1080:(ow-iw)/2:(oh-ih)/2'
    };
  }

  return {
    width: original.width,
    height: original.height,
    scale: null
  };
}

// ============================================================
// Final Render
// ============================================================

async function renderRecap({
  video,
  audio,
  transcription,
  settings,
  jobDir
}) {
  const original =
    await videoSize(video);

  const output =
    outputSpec(
      settings.outputSize,
      original
    );

  const ass =
    path.join(
      jobDir,
      'subtitles.ass'
    );

  makeAss({
    transcription,
    file: ass,

    fontName:
      settings.fontName,

    fontSize:
      settings.fontSize,

    outline:
      settings.outline,

    textColor:
      settings.textColor,

    position:
      settings.position,

    width:
      output.width,

    height:
      output.height,

    subtitleX:
      settings.subtitleX,

    subtitleY:
      settings.subtitleY
  });

  const blur =
    blurBox({
      width:
        original.width,

      height:
        original.height,

      on:
        settings.blurOriginal,

      x:
        settings.blurX,

      y:
        settings.blurY,

      w:
        settings.blurWidth,

      h:
        settings.blurHeight
    });

  const assPath =
    filterPath(ass);

  const filters = [];

  if (blur) {
    filters.push(
      `[0:v]split=2[base][src]`
    );

    filters.push(
      `[src]crop=${blur.bw}:${blur.bh}:${blur.bx}:${blur.by},boxblur=20:10[blur]`
    );

    filters.push(
      `[base][blur]overlay=${blur.bx}:${blur.by}[v0]`
    );

    if (output.scale) {
      filters.push(
        `[v0]${output.scale}[v1]`
      );
    } else {
      filters.push(
        `[v0]null[v1]`
      );
    }

  } else {
    if (output.scale) {
      filters.push(
        `[0:v]${output.scale}[v1]`
      );
    } else {
      filters.push(
        `[0:v]null[v1]`
      );
    }
  }

  filters.push(
    `[v1]subtitles='${assPath}'[vout]`
  );

  const outputFile =
    path.join(
      jobDir,
      'myanmar_movie_recap.mp4'
    );

  await ffmpeg([
    '-i',
    video,

    '-i',
    audio,

    '-filter_complex',
    filters.join(';'),

    '-map',
    '[vout]',

    '-map',
    '1:a:0',

    '-c:v',
    'libx264',

    '-preset',
    'veryfast',

    '-crf',
    '20',

    '-c:a',
    'aac',

    '-b:a',
    '192k',

    '-shortest',

    '-movflags',
    '+faststart',

    outputFile
  ]);

  return outputFile;
}

// ============================================================
// Basic API
// ============================================================

app.get(
  '/api/health',
  (req, res) => {
    res.json({
      ok: true,
      service: 'Burmese SRT'
    });
  }
);

app.get(
  '/api/recap/voices',
  (req, res) => {
    res.json({
      ok: true,
      voices: GEMINI_VOICES,
      model: TTS_MODEL
    });
  }
);

// ============================================================
// Old SRT - Transcribe
// ============================================================

app.post(
  '/api/transcribe',
  upload.single('video'),

  async (req, res) => {
    if (!req.file) {
      return jsonError(
        res,
        400,
        'Video မရှိပါ။'
      );
    }

    const key =
      getGroqKey(req);

    if (!key) {
      return jsonError(
        res,
        400,
        'Groq API Key မရှိပါ။'
      );
    }

    try {
      const data =
        await groqTranscribe(
          req.file.path,
          '',
          key
        );

      res.json({
        ok: true,
        ...data
      });

    } catch (error) {
      jsonError(
        res,
        500,
        error.message
      );

    } finally {
      fs.rm(
        req.file.path,
        {
          force: true
        },
        () => {}
      );
    }
  }
);

// ============================================================
// Old SRT - Translate
// ============================================================

app.post(
  '/api/translate',

  async (req, res) => {
    const text =
      String(
        req.body?.text ??
        req.body?.transcript ??
        ''
      ).trim();

    const key =
      getGeminiKey(req);

    if (!text) {
      return jsonError(
        res,
        400,
        'Translate text မရှိပါ။'
      );
    }

    if (!key) {
      return jsonError(
        res,
        400,
        'Gemini API Key မရှိပါ။'
      );
    }

    try {
      const data =
        await geminiGenerate(
          'gemini-3.8-flash',

          {
            contents: [
              {
                role: 'user',

                parts: [
                  {
                    text:
`Translate the following text into natural Myanmar subtitle language.

Preserve the meaning.

Do not add information.

Return only the Myanmar translation.

TEXT:

${text}`
                  }
                ]
              }
            ],

            generationConfig: {
              temperature: 0.2,
              maxOutputTokens: 8000
            }
          },

          key
        );

      res.json({
        ok: true,
        text:
          geminiText(data)
      });

    } catch (error) {
      jsonError(
        res,
        500,
        error.message
      );
    }
  }
);

// ============================================================
// One Clips - Scene Analysis
// ============================================================

app.post(
  '/api/recap/analyze',
  upload.single('video'),

  async (req, res) => {
    if (!req.file) {
      return jsonError(
        res,
        400,
        'Video မရှိပါ။'
      );
    }

    const key =
      getGeminiKey(req);

    const job =
      path.join(
        WORK_DIR,
        uid('analyze')
      );

    fs.mkdirSync(
      job,
      {
        recursive: true
      }
    );

    try {
      const result =
        await analyzeVideoScenes(
          req.file.path,
          job,
          key
        );

      res.json({
        ok: true,
        ...result
      });

    } catch (error) {
      jsonError(
        res,
        500,
        error.message
      );

    } finally {
      fs.rm(
        req.file.path,
        {
          force: true
        },
        () => {}
      );

      setTimeout(
        () => {
          fs.rm(
            job,
            {
              recursive: true,
              force: true
            },
            () => {}
          );
        },
        600000
      );
    }
  }
);

// ============================================================
// One Clips - Gemini TTS
// ============================================================

app.post(
  '/api/recap/tts',

  async (req, res) => {
    const key =
      getGeminiKey(req);

    const job =
      path.join(
        WORK_DIR,
        uid('tts')
      );

    fs.mkdirSync(
      job,
      {
        recursive: true
      }
    );

    try {
      const result =
        await geminiTTS({
          script:
            String(
              req.body?.script ||
              ''
            ),

          voice:
            req.body?.voice,

          speed:
            req.body?.voiceSpeed,

          pitch:
            req.body?.voicePitch,

          volume:
            req.body?.voiceVolume,

          jobDir:
            job,

          apiKey:
            key
        });

      res.sendFile(
        result.audioPath,

        {
          headers: {
            'Content-Type':
              'audio/wav',

            'Content-Disposition':
              'inline; filename="voice.wav"'
          }
        }
      );

    } catch (error) {
      jsonError(
        res,
        500,
        error.message
      );

    } finally {
      setTimeout(
        () => {
          fs.rm(
            job,
            {
              recursive: true,
              force: true
            },
            () => {}
          );
        },
        600000
      );
    }
  }
);

// ============================================================
// One Clips - Voice Sync
// ============================================================

app.post(
  '/api/recap/voice-sync',
  upload.single('audio'),

  async (req, res) => {
    if (!req.file) {
      return jsonError(
        res,
        400,
        'Audio မရှိပါ။'
      );
    }

    const key =
      getGroqKey(req);

    try {
      const result =
        await groqTranscribe(
          req.file.path,

          req.body?.script ||
            '',

          key
        );

      res.json({
        ok: true,
        ...result
      });

    } catch (error) {
      jsonError(
        res,
        500,
        error.message
      );

    } finally {
      fs.rm(
        req.file.path,
        {
          force: true
        },
        () => {}
      );
    }
  }
);

// ============================================================
// Render Endpoint
// ============================================================

app.post(
  '/api/render',
  upload.single('video'),

  async (req, res) => {
    if (!req.file) {
      return jsonError(
        res,
        400,
        'Video မရှိပါ။'
      );
    }

    const job =
      path.join(
        WORK_DIR,
        uid('render')
      );

    fs.mkdirSync(
      job,
      {
        recursive: true
      }
    );

    try {
      let transcription = {};

      if (
        typeof req.body?.transcription ===
        'string'
      ) {
        transcription =
          JSON.parse(
            req.body.transcription
          );
      } else {
        transcription =
          req.body?.transcription ||
          {};
      }

      const settings = {
        ...req.body,

        blurOriginal:
          req.body.blurOriginal ===
            'true' ||
          req.body.blurOriginal ===
            true,

        fontSize:
          Number(
            req.body.fontSize
          ) || 25,

        outline:
          Number(
            req.body.outline
          ) || 2
      };

      const output =
        await renderRecap({
          video:
            req.file.path,

          audio:
            req.body.audioPath,

          transcription,

          settings,

          jobDir:
            job
        });

      res.sendFile(
        output,

        {
          headers: {
            'Content-Type':
              'video/mp4',

            'Content-Disposition':
              'attachment; filename="myanmar_movie_recap.mp4"'
          }
        }
      );

    } catch (error) {
      jsonError(
        res,
        500,
        error.message
      );

    } finally {
      fs.rm(
        req.file.path,
        {
          force: true
        },
        () => {}
      );

      setTimeout(
        () => {
          fs.rm(
            job,
            {
              recursive: true,
              force: true
            },
            () => {}
          );
        },
        600000
      );
    }
  }
);

// ============================================================
// ONE CLICK
//
// Video
//   ↓
// Gemini Scene Analysis
//   ↓
// Myanmar Recap Script
//   ↓
// Gemini TTS
//   ↓
// Groq Whisper Timestamp
//   ↓
// Subtitle
//   ↓
// Original Subtitle Blur
//   ↓
// Font / Color / Outline
//   ↓
// Output Size
//   ↓
// Final MP4
// ============================================================

app.post(
  '/api/recap/one-click',
  upload.single('video'),

  async (req, res) => {
    if (!req.file) {
      return jsonError(
        res,
        400,
        'Video မရှိပါ။'
      );
    }

    const geminiKey =
      getGeminiKey(req);

    const groqKey =
      getGroqKey(req);

    if (!geminiKey) {
      return jsonError(
        res,
        400,
        'Gemini API Key မရှိပါ။'
      );
    }

    if (!groqKey) {
      return jsonError(
        res,
        400,
        'Groq API Key မရှိပါ။'
      );
    }

    const job =
      path.join(
        WORK_DIR,
        uid('oneclick')
      );

    fs.mkdirSync(
      job,
      {
        recursive: true
      }
    );

    try {

      // ------------------------------------------------------
      // 1. Analyze actual video scenes
      // ------------------------------------------------------

      const analysis =
        await analyzeVideoScenes(
          req.file.path,
          job,
          geminiKey
        );

      // ------------------------------------------------------
      // 2. Gemini TTS
      // ------------------------------------------------------

      const tts =
        await geminiTTS({
          script:
            analysis.script,

          voice:
            req.body.voice ||
            'Kore',

          speed:
            Number(
              req.body.voiceSpeed
            ) || 1,

          pitch:
            Number(
              req.body.voicePitch
            ) || 0,

          volume:
            Number(
              req.body.voiceVolume
            ) || 1,

          jobDir:
            job,

          apiKey:
            geminiKey
        });

      // ------------------------------------------------------
      // 3. Groq Whisper
      //
      // Generated voice ကိုပြန် transcribe လုပ်ပြီး
      // exact subtitle timing ရယူတယ်.
      // ------------------------------------------------------

      const sync =
        await groqTranscribe(
          tts.audioPath,

          analysis.script,

          groqKey
        );

      // ------------------------------------------------------
      // 4. Render settings
      // ------------------------------------------------------

      const settings = {

        blurOriginal:
          req.body.blurOriginal ===
            'true' ||
          req.body.blurOriginal ===
            true,

        blurX:
          clamp(
            req.body.blurX,
            0,
            100
          ),

        blurY:
          clamp(
            req.body.blurY,
            0,
            100
          ),

        blurWidth:
          clamp(
            req.body.blurWidth,
            1,
            100
          ),

        blurHeight:
          clamp(
            req.body.blurHeight,
            1,
            100
          ),

        subtitleX:
          req.body.subtitleX !==
            '' &&
          req.body.subtitleX !=
            null

            ? clamp(
                req.body.subtitleX,
                0,
                100
              )

            : NaN,

        subtitleY:
          req.body.subtitleY !==
            '' &&
          req.body.subtitleY !=
            null

            ? clamp(
                req.body.subtitleY,
                0,
                100
              )

            : NaN,

        fontName:
          req.body.fontName ||
          'Noto Sans Myanmar',

        fontSize:
          clamp(
            req.body.fontSize,
            10,
            120
          ),

        outline:
          clamp(
            req.body.outline,
            0,
            10
          ),

        textColor:
          req.body.textColor ||
          '#FFFFFF',

        position:
          req.body.position ||
          'bottom',

        outputSize:
          req.body.outputSize ||
          'original'
      };

      // ------------------------------------------------------
      // 5. Final MP4
      // ------------------------------------------------------

      const output =
        await renderRecap({
          video:
            req.file.path,

          audio:
            tts.audioPath,

          transcription:
            sync,

          settings,

          jobDir:
            job
        });

      res.sendFile(
        output,

        {
          headers: {
            'Content-Type':
              'video/mp4',

            'Content-Disposition':
              'attachment; filename="myanmar_movie_recap.mp4"'
          }
        }
      );

    } catch (error) {

      console.error(
        'ONE CLICK ERROR:',
        error
      );

      jsonError(
        res,
        500,
        error.message
      );

    } finally {

      fs.rm(
        req.file.path,
        {
          force: true
        },
        () => {}
      );

      setTimeout(
        () => {
          fs.rm(
            job,
            {
              recursive: true,
              force: true
            },
            () => {}
          );
        },
        600000
      );
    }
  }
);

// ============================================================
// Frontend fallback
// ============================================================

app.use(
  (req, res, next) => {

    if (
      req.method === 'GET' &&
      !req.path.startsWith('/api/')
    ) {
      return res.sendFile(
        path.join(
          PUBLIC_DIR,
          'index.html'
        )
      );
    }

    next();
  }
);

// ============================================================
// Error Handler
// ============================================================

app.use(
  (err, req, res, next) => {

    console.error(err);

    if (res.headersSent) {
      return next(err);
    }

    jsonError(
      res,
      500,
      err.message ||
        'Server error'
    );
  }
);

// ============================================================
// Start
// ============================================================

app.listen(
  PORT,
  () => {
    console.log(
      `Myanmar SRT server running on port ${PORT}`
    );
  }
);
