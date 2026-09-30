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
   APP
========================================================= */

const app = express();

const PORT =
  Number(process.env.PORT || 10000);

const ROOT =
  process.cwd();

const PUBLIC =
  path.join(ROOT, 'public');

const TMP =
  path.join(
    os.tmpdir(),
    'myanmar-srt'
  );

const MAX_BYTES =
  300 * 1024 * 1024;

const MAX_SECONDS =
  5 * 60;

await fs.mkdir(
  TMP,
  {
    recursive:true
  }
);


/* =========================================================
   MODELS
========================================================= */

const GROQ_MODEL =
  'whisper-large-v3';

const GEMINI_MODEL =
  'gemini-3.5-flash-lite';


/* =========================================================
   EXPRESS
========================================================= */

app.use(
  express.json({
    limit:'5mb'
  })
);

app.use(
  express.static(PUBLIC)
);


/* =========================================================
   ALLOWED VIDEO
========================================================= */

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


/* =========================================================
   MULTER
========================================================= */

const storage =
  multer.diskStorage({

    destination:
      TMP,

    filename:
      (req,file,cb)=>{

        const originalExt =
          path
            .extname(
              file.originalname || ''
            )
            .toLowerCase();

        const ext =
          ALLOWED_EXTENSIONS.has(
            originalExt
          )
            ? originalExt
            : '.mp4';

        const filename =
          `${Date.now()}-` +
          `${crypto.randomBytes(8).toString('hex')}` +
          `${ext}`;

        cb(
          null,
          filename
        );

      }

  });


const upload =
  multer({

    storage,

    limits:{
      fileSize:
        MAX_BYTES
    },

    fileFilter:
      (req,file,cb)=>{

        const ext =
          path
            .extname(
              file.originalname || ''
            )
            .toLowerCase();

        if(
          !ALLOWED_EXTENSIONS.has(ext)
        ){

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


/* =========================================================
   HELPERS
========================================================= */

function getKey(
  req,
  headerName,
  envName
){

  const headerValue =
    req.get(headerName);

  if(
    headerValue &&
    headerValue.trim()
  ){

    return headerValue.trim();

  }

  const envValue =
    process.env[envName];

  if(
    envValue &&
    envValue.trim()
  ){

    return envValue.trim();

  }

  return '';

}


function cleanText(value){

  return String(
    value ?? ''
  )
    .replace(/\r/g,' ')
    .replace(/\n+/g,' ')
    .replace(/\s+/g,' ')
    .trim();

}


function sleep(ms){

  return new Promise(
    resolve =>
      setTimeout(
        resolve,
        ms
      )
  );

}


function safeNumber(
  value,
  fallback = 0
){

  const n =
    Number(value);

  return Number.isFinite(n)
    ? n
    : fallback;

}


/* =========================================================
   PROCESS
========================================================= */

function runProcess(
  command,
  args
){

  return new Promise(
    (resolve,reject)=>{

      const child =
        spawn(
          command,
          args
        );

      let stdout = '';
      let stderr = '';

      child.stdout.on(
        'data',
        chunk=>{
          stdout +=
            chunk.toString();
        }
      );

      child.stderr.on(
        'data',
        chunk=>{
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
        code=>{

          if(
            code === 0
          ){

            resolve({
              stdout,
              stderr
            });

          }else{

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


/* =========================================================
   VIDEO PROBE
========================================================= */

async function probeVideo(
  filePath
){

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

  if(
    !Number.isFinite(duration)
  ){

    throw new Error(
      'Video duration ကို မဖတ်နိုင်ပါ။'
    );

  }

  return duration;

}


/* =========================================================
   EXTRACT AUDIO
========================================================= */

async function extractAudio(
  videoPath
){

  const audioPath =
    path.join(
      TMP,
      `${Date.now()}-` +
      `${crypto.randomBytes(8).toString('hex')}.wav`
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

      '-map',
      '0:a:0',

      '-ar',
      '16000',

      '-ac',
      '1',

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


/* =========================================================
   GROQ WHISPER
========================================================= */

async function transcribeGroq(
  audioPath,
  apiKey
){

  if(!apiKey){

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
      Array.isArray(result?.segments)
        ? result.segments.length
        : 0
    }`
  );

  console.log(
    `Whisper raw words: ${
      Array.isArray(result?.words)
        ? result.words.length
        : 0
    }`
  );

  return result;

}


/* =========================================================
   WORD NORMALIZE
========================================================= */

function normalizeWord(
  item
){

  if(!item){
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

  if(
    !word ||
    !Number.isFinite(start) ||
    !Number.isFinite(end)
  ){

    return null;

  }

  if(
    end <= start
  ){

    return null;

  }

  return {
    word,
    start,
    end
  };

}


/* =========================================================
   SENTENCE
========================================================= */

function endsSentence(
  text
){

  const value =
    String(
      text || ''
    ).trim();

  if(!value){
    return false;
  }

  return /[.!?。！？…]+$/.test(
    value
  );

}


/* =========================================================
   CLEAN DUPLICATE TEXT
========================================================= */

function normalizeCompare(
  text
){

  return cleanText(
    text
  )
    .toLowerCase()
    .replace(
      /[.,!?;:'"“”‘’၊။！？]/g,
      ''
    )
    .replace(
      /\s+/g,
      ' '
    )
    .trim();

}


function isDuplicateCue(
  previous,
  current
){

  if(
    !previous ||
    !current
  ){

    return false;

  }

  const a =
    normalizeCompare(
      previous.text
    );

  const b =
    normalizeCompare(
      current.text
    );

  if(
    !a ||
    !b
  ){

    return false;

  }

  if(
    a === b
  ){

    return true;

  }

  if(
    a.length > 10 &&
    b.length > 10
  ){

    if(
      a.includes(b) ||
      b.includes(a)
    ){

      const ratio =
        Math.min(
          a.length,
          b.length
        ) /
        Math.max(
          a.length,
          b.length
        );

      if(
        ratio >= 0.82
      ){

        return true;

      }

    }

  }

  return false;

}


/* =========================================================
   PRECISE SUBTITLE SEGMENTS
========================================================= */

function buildPreciseSegments(
  transcript
){

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

  if(words.length){

    const result = [];

    let currentWords = [];

    let currentStart =
      null;

    let currentEnd =
      null;


    function flush(){

      if(
        !currentWords.length
      ){

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
            /\s+([,.!?;:၊။！？；：])/g,
            '$1'
          )
          .trim();

      if(
        text &&
        currentStart !== null &&
        currentEnd !== null
      ){

        const cue = {
          id:
            result.length + 1,

          start:
            currentStart,

          end:
            currentEnd,

          text
        };

        if(
          !isDuplicateCue(
            result[result.length - 1],
            cue
          )
        ){

          result.push(
            cue
          );

        }

      }

      currentWords = [];

      currentStart =
        null;

      currentEnd =
        null;

    }


    for(
      let i = 0;
      i < words.length;
      i++
    ){

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


      if(
        !currentWords.length
      ){

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
            /\s+([,.!?;:၊။！？；：])/g,
            '$1'
          )
          .trim();


      const candidateText =
        `${currentText} ${word.word}`
          .replace(
            /\s+([,.!?;:၊။！？；：])/g,
            '$1'
          )
          .trim();


      const currentDuration =
        word.end -
        currentStart;


      const naturalPause =
        gap >= 0.55;

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


      if(
        naturalPause ||
        sentenceFinished ||
        tooLong ||
        tooManyChars ||
        tooManyWords
      ){

        flush();

        currentStart =
          word.start;

        currentEnd =
          word.end;

        currentWords.push(
          word
        );

      }else{

        currentWords.push(
          word
        );

        currentEnd =
          word.end;

      }

    }

    flush();


    /*
      အနီးကပ် cue တွေကို အလွန်တိုလွန်းရင်
      မလိုအပ်ဘဲ မခွဲစေရန် ထပ်မံစစ်ဆေးခြင်း
    */

    const merged = [];

    for(
      const cue of result
    ){

      const previous =
        merged[
          merged.length - 1
        ];

      if(
        previous &&
        cue.start -
        previous.end < 0.20 &&
        previous.text.length < 28 &&
        cue.text.length < 28
      ){

        previous.end =
          cue.end;

        previous.text =
          `${previous.text} ${cue.text}`
            .replace(
              /\s+([,.!?;:၊။！？；：])/g,
              '$1'
            )
            .trim();

      }else{

        merged.push({
          ...cue
        });

      }

    }


    const finalResult =
      merged.map(
        (item,index)=>({
          id:
            index + 1,

          start:
            Number(
              item.start.toFixed(3)
            ),

          end:
            Number(
              item.end.toFixed(3)
            ),

          text:
            cleanText(
              item.text
            )
        })
      );


    console.log(
      `Precise subtitle cues created: ${finalResult.length}`
    );

    return finalResult;

  }


  /* =====================================================
     FALLBACK SEGMENTS
  ===================================================== */

  const rawSegments =
    Array.isArray(
      transcript?.segments
    )
      ? transcript.segments
      : [];


  const fallback =
    rawSegments
      .map(
        (segment,index)=>{

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

          if(
            !text ||
            !Number.isFinite(start) ||
            !Number.isFinite(end)
          ){

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


/* =========================================================
   GEMINI ERROR
========================================================= */

function isRetryableGeminiError(
  error
){

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


  if(
    [
      408,
      429,
      500,
      502,
      503,
      504
    ].includes(
      Number(status)
    )
  ){

    return true;

  }


  return (
    message.includes('429') ||
    message.includes('resource_exhausted') ||
    message.includes('rate limit') ||
    message.includes('rate_limit') ||
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


function isQuotaError(
  error
){

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


/* =========================================================
   GEMINI RETRY
========================================================= */

async function generateGeminiWithRetry(
  ai,
  prompt
){

  const MAX_RETRIES =
    4;

  let lastError =
    null;


  for(
    let attempt = 0;
    attempt <= MAX_RETRIES;
    attempt++
  ){

    try{

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

          config:{
            temperature:
              0.1,

            responseMimeType:
              'application/json',

            maxOutputTokens:
              8192
          }

        });


      return response;


    }catch(error){

      lastError =
        error;


      if(
        !isRetryableGeminiError(
          error
        ) ||
        attempt >= MAX_RETRIES
      ){

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
          baseDelay + jitter,
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


/* =========================================================
   EXTRACT JSON
========================================================= */

function extractJson(
  text
){

  let value =
    String(
      text || ''
    ).trim();


  value =
    value
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


  try{

    return JSON.parse(
      value
    );

  }catch(_){}


  const start =
    value.indexOf('[');

  const end =
    value.lastIndexOf(']');


  if(
    start >= 0 &&
    end > start
  ){

    const jsonText =
      value.slice(
        start,
        end + 1
      );

    try{

      return JSON.parse(
        jsonText
      );

    }catch(_){}

  }


  const objectStart =
    value.indexOf('{');

  const objectEnd =
    value.lastIndexOf('}');


  if(
    objectStart >= 0 &&
    objectEnd > objectStart
  ){

    const jsonText =
      value.slice(
        objectStart,
        objectEnd + 1
      );

    try{

      return JSON.parse(
        jsonText
      );

    }catch(_){}

  }


  throw new Error(
    'Gemini က valid JSON မပြန်ပါ။'
  );

}


/* =========================================================
   GEMINI TRANSLATE CHUNK
========================================================= */

async function translateChunkGemini(
  chunk,
  apiKey
){

  if(!apiKey){

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
      item=>({

        id:
          item.id,

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

Translate the following subtitle dialogue into NATURAL, ACCURATE BURMESE (MYANMAR LANGUAGE).

STRICT RULES:

1. Output Burmese/Myanmar language ONLY.
2. Do NOT output Chinese characters.
3. Do NOT output Korean characters.
4. Do NOT output Japanese characters.
5. Do NOT output English sentences.
6. Do NOT copy the original dialogue.
7. Do NOT add explanations.
8. Do NOT add notes.
9. Do NOT add translator comments.
10. Preserve every subtitle ID exactly.
11. Preserve start and end timestamps exactly.
12. Translate every dialogue item.
13. Keep the translation natural for subtitles.
14. Do not merge different IDs.
15. Do not create new IDs.
16. Do not remove IDs.
17. Proper names may remain unchanged only when absolutely necessary.
18. Everything else must be Myanmar language.
19. Return ONLY valid JSON.
20. Return an array.

INPUT:
${JSON.stringify(input)}

OUTPUT FORMAT:
[
  {
    "id": 1,
    "start": 0,
    "end": 2.5,
    "text": "မြန်မာဘာသာပြန်စာ"
  }
]
`;


  const response =
    await generateGeminiWithRetry(
      ai,
      prompt
    );


  const rawText =
    response?.text ||
    response?.candidates?.[0]?.content?.parts
      ?.map(
        part =>
          part.text || ''
      )
      .join('') ||
    '';


  const parsed =
    extractJson(
      rawText
    );


  const array =
    Array.isArray(parsed)
      ? parsed
      : (
          Array.isArray(
            parsed?.segments
          )
            ? parsed.segments
            : []
        );


  if(
    !array.length
  ){

    throw new Error(
      'Gemini ဘာသာပြန်ရလဒ် အလွတ်ဖြစ်နေပါတယ်။'
    );

  }


  const result =
    chunk.map(
      original=>{

        const found =
          array.find(
            item =>
              Number(
                item.id
              ) ===
              Number(
                original.id
              )
          );


        const translated =
          cleanText(
            found?.text ||
            found?.translation ||
            ''
          );


        return {

          id:
            original.id,

          start:
            original.start,

          end:
            original.end,

          text:
            translated ||
            original.text

        };

      }
    );


  return result;

}


/* =========================================================
   TRANSLATE ALL
========================================================= */

async function translateAllGemini(
  segments,
  apiKey
){

  const CHUNK_SIZE =
    30;

  const output = [];


  for(
    let i = 0;
    i < segments.length;
    i += CHUNK_SIZE
  ){

    const chunk =
      segments.slice(
        i,
        i + CHUNK_SIZE
      );


    console.log(
      `Gemini translation chunk ${
        Math.floor(
          i / CHUNK_SIZE
        ) + 1
      } / ${
        Math.ceil(
          segments.length /
          CHUNK_SIZE
        )
      }`
    );


    const translated =
      await translateChunkGemini(
        chunk,
        apiKey
      );


    output.push(
      ...translated
    );


    if(
      i + CHUNK_SIZE <
      segments.length
    ){

      await sleep(
        250
      );

    }

  }


  return output.map(
    (item,index)=>({

      id:
        index + 1,

      start:
        item.start,

      end:
        item.end,

      text:
        cleanText(
          item.text
        )

    })
  );

}


/* =========================================================
   SRT
========================================================= */

function formatSrtTime(
  seconds
){

  const totalMs =
    Math.max(
      0,
      Math.round(
        Number(seconds || 0) *
        1000
      )
    );


  const hours =
    Math.floor(
      totalMs /
      3600000
    );


  const minutes =
    Math.floor(
      (
        totalMs %
        3600000
      ) /
      60000
    );


  const secs =
    Math.floor(
      (
        totalMs %
        60000
      ) /
      1000
    );


  const ms =
    totalMs %
    1000;


  return (
    String(hours).padStart(2,'0') +
    ':' +
    String(minutes).padStart(2,'0') +
    ':' +
    String(secs).padStart(2,'0') +
    ',' +
    String(ms).padStart(3,'0')
  );

}


function makeSrt(
  segments
){

  return segments
    .map(
      (item,index)=>{

        const text =
          cleanText(
            item.text
          );

        return (
          `${index + 1}\n` +
          `${formatSrtTime(item.start)} --> ` +
          `${formatSrtTime(item.end)}\n` +
          `${text}\n`
        );

      }
    )
    .join('\n');

}


/* =========================================================
   ASS ESCAPE
========================================================= */

function escapeAssText(
  text
){

  return String(
    text || ''
  )
    .replace(
      /\\/g,
      '\\\\'
    )
    .replace(
      /\{/g,
      '\\{'
    )
    .replace(
      /\}/g,
      '\\}'
    )
    .replace(
      /\r?\n/g,
      '\\N'
    );

}


/* =========================================================
   HEX -> ASS COLOR
========================================================= */

function hexToAss(
  hex
){

  const value =
    String(
      hex || '#FFFFFF'
    )
      .replace(
        '#',
        ''
      )
      .trim();


  if(
    !/^[0-9a-fA-F]{6}$/.test(
      value
    )
  ){

    return '&H00FFFFFF';

  }


  const rr =
    value.slice(0,2);

  const gg =
    value.slice(2,4);

  const bb =
    value.slice(4,6);


  return (
    '&H00' +
    bb +
    gg +
    rr
  );

}


/* =========================================================
   ASS TIME
========================================================= */

function formatAssTime(
  seconds
){

  const total =
    Math.max(
      0,
      Number(seconds || 0)
    );


  const hours =
    Math.floor(
      total / 3600
    );


  const minutes =
    Math.floor(
      (
        total % 3600
      ) / 60
    );


  const secs =
    Math.floor(
      total % 60
    );


  const centis =
    Math.floor(
      (
        total -
        Math.floor(total)
      ) * 100
    );


  return (
    `${hours}:` +
    `${String(minutes).padStart(2,'0')}:` +
    `${String(secs).padStart(2,'0')}.` +
    `${String(centis).padStart(2,'0')}`
  );

}


/* =========================================================
   FIND FONT
========================================================= */

async function findMyanmarFont(){

  const possibleFonts = [

    '/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf',

    '/usr/share/fonts/opentype/noto/NotoSansMyanmar-Regular.ttf',

    '/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.otf',

    '/usr/share/fonts/truetype/noto/NotoSansMyanmarUI-Regular.ttf',

    '/usr/share/fonts/truetype/noto/NotoSansMyanmarUI-Regular.otf',

    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

  ];


  for(
    const font of possibleFonts
  ){

    try{

      await fs.access(
        font
      );

      return font;

    }catch(_){}

  }


  return null;

}


/* =========================================================
   CREATE ASS
========================================================= */

async function createAssFile(
  segments,
  options
){

  const assPath =
    path.join(
      TMP,
      `${Date.now()}-` +
      `${crypto.randomBytes(8).toString('hex')}.ass`
    );


  const width =
    safeNumber(
      options.videoWidth,
      1080
    );


  const height =
    safeNumber(
      options.videoHeight,
      1920
    );


  const fontSize =
    Math.max(
      20,
      Math.min(
        80,
        safeNumber(
          options.fontSize,
          42
        )
      )
    );


  const outline =
    Math.max(
      0,
      Math.min(
        8,
        safeNumber(
          options.outline,
          3
        )
      )
    );


  const color =
    hexToAss(
      options.color
    );


  let alignment = 2;

  let marginV = 70;


  if(
    options.position === 'top'
  ){

    alignment = 8;
    marginV = 55;

  }else if(
    options.position === 'middle'
  ){

    alignment = 5;
    marginV = 0;

  }else{

    alignment = 2;
    marginV = 65;

  }


  const fontFile =
    await findMyanmarFont();


  const fontName =
    'Noto Sans Myanmar';


  const lines = [];


  lines.push(
    '[Script Info]'
  );

  lines.push(
    'ScriptType: v4.00+'
  );

  lines.push(
    'PlayResX: ' + width
  );

  lines.push(
    'PlayResY: ' + height
  );

  lines.push(
    'ScaledBorderAndShadow: yes'
  );

  lines.push('');

  lines.push(
    '[V4+ Styles]'
  );

  lines.push(
    'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding'
  );

  lines.push(
    `Style: Default,${fontName},${fontSize},${color},${color},&H00000000,&H99000000,1,0,0,0,100,100,0,0,1,${outline},1,${alignment},45,45,${marginV},1`
  );

  lines.push('');

  lines.push(
    '[Events]'
  );

  lines.push(
    'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text'
  );


  for(
    const item of segments
  ){

    const text =
      escapeAssText(
        item.text
      );


    if(!text){
      continue;
    }


    lines.push(
      `Dialogue: 0,${formatAssTime(item.start)},${formatAssTime(item.end)},Default,,0,0,0,,${text}`
    );

  }


  await fs.writeFile(
    assPath,
    lines.join('\n'),
    'utf8'
  );


  return {
    assPath,
    fontFile
  };

}


/* =========================================================
   BUILD BLUR FILTER
========================================================= */

function buildBlurFilter(
  regions
){

  if(
    !Array.isArray(regions) ||
    !regions.length
  ){

    return null;

  }


  const safeRegions =
    regions
      .slice(0,3)
      .map(
        r=>({

          x:
            Math.max(
              0,
              Math.min(
                1,
                safeNumber(
                  r.x
                )
              )
            ),

          y:
            Math.max(
              0,
              Math.min(
                1,
                safeNumber(
                  r.y
                )
              )
            ),

          w:
            Math.max(
              0,
              Math.min(
                1,
                safeNumber(
                  r.w
                )
              )
            ),

          h:
            Math.max(
              0,
              Math.min(
                1,
                safeNumber(
                  r.h
                )
              )
            )

        })
      )
      .filter(
        r =>
          r.w >= 0.01 &&
          r.h >= 0.01
      );


  if(
    !safeRegions.length
  ){

    return null;

  }


  const filters = [];


  filters.push(
    '[0:v]split=' +
    (safeRegions.length + 1) +
    '[base]' +
    safeRegions
      .map(
        (_,i)=>
          `[blur${i}]`
      )
      .join('')
  );


  /*
    split output naming ကို ffmpeg
    filter graph အတွက် ပြန်တည်ဆောက်ရန်
  */

  const splitLabels =
    [
      '[base]'
    ];


  for(
    let i = 0;
    i < safeRegions.length;
    i++
  ){

    splitLabels.push(
      `[blur${i}]`
    );

  }


  filters[0] =
    `[0:v]split=${safeRegions.length + 1}${splitLabels.join('')}`;


  for(
    let i = 0;
    i < safeRegions.length;
    i++
  ){

    const r =
      safeRegions[i];


    filters.push(
      `[blur${i}]` +
      `crop=` +
      `iw*${r.w}:` +
      `ih*${r.h}:` +
      `iw*${r.x}:` +
      `ih*${r.y},` +
      `boxblur=12:2` +
      `[b${i}]`
    );

  }


  let current =
    '[base]';


  for(
    let i = 0;
    i < safeRegions.length;
    i++
  ){

    const r =
      safeRegions[i];


    const next =
      i === safeRegions.length - 1
        ? '[blurred]'
        : `[mix${i}]`;


    filters.push(
      `${current}[b${i}]` +
      `overlay=` +
      `x=W*${r.x}:` +
      `y=H*${r.y}` +
      `${next}`
    );


    current =
      next;

  }


  return filters.join(';');

}


/* =========================================================
   RENDER VIDEO
========================================================= */

async function renderVideo(
  inputPath,
  segments,
  options,
  blurRegions
){

  const duration =
    await probeVideo(
      inputPath
    );


  if(
    duration > MAX_SECONDS + 2
  ){

    throw new Error(
      'Video က 5 မိနစ်ထက်ရှည်နေပါတယ်။'
    );

  }


  const outputPath =
    path.join(
      TMP,
      `${Date.now()}-` +
      `${crypto.randomBytes(8).toString('hex')}.mp4`
    );


  /*
    Video size ကို ffprobe နဲ့ယူမယ်
  */

  const probe =
    await runProcess(
      ffprobeStatic.path,
      [
        '-v',
        'error',

        '-select_streams',
        'v:0',

        '-show_entries',
        'stream=width,height',

        '-of',
        'csv=s=x:p=0',

        inputPath
      ]
    );


  const dimensions =
    String(
      probe.stdout || ''
    ).trim();


  let videoWidth =
    1080;

  let videoHeight =
    1920;


  if(
    /^\d+x\d+$/.test(
      dimensions
    )
  ){

    const parts =
      dimensions
        .split('x')
        .map(Number);

    if(
      Number.isFinite(parts[0]) &&
      Number.isFinite(parts[1])
    ){

      videoWidth =
        parts[0];

      videoHeight =
        parts[1];

    }

  }


  /*
    ASS
  */

  const ass =
    await createAssFile(
      segments,
      {
        ...options,
        videoWidth,
        videoHeight
      }
    );


  /*
    Filter
  */

  const blurFilter =
    buildBlurFilter(
      blurRegions
    );


  const filterParts = [];


  if(
    blurFilter
  ){

    filterParts.push(
      blurFilter
    );

  }


  /*
    ASS subtitle filter

    Windows path မဟုတ်တဲ့အတွက်
    / path ကိုသုံးမယ်။
  */

  const escapedAss =
    ass.assPath
      .replace(
        /\\/g,
        '/'
      )
      .replace(
        /:/g,
        '\\:'
      )
      .replace(
        /'/g,
        "\\'"
      );


  if(
    blurFilter
  ){

    filterParts.push(
      `[blurred]subtitles='${escapedAss}'[vout]`
    );

  }else{

    filterParts.push(
      `[0:v]subtitles='${escapedAss}'[vout]`
    );

  }


  /*
    Blur filter က [blurred] output ဖြစ်ပြီး
    မရှိရင် [0:v] ကို subtitle ထဲပို့တယ်။
  */

  const filterComplex =
    filterParts.join(';');


  console.log(
    'Starting final MP4 render...'
  );


  await runProcess(
    ffmpegStatic,
    [

      '-y',

      '-i',
      inputPath,

      '-filter_complex',
      filterComplex,

      '-map',
      '[vout]',

      '-map',
      '0:a?',

      '-c:v',
      'libx264',

      '-preset',
      'veryfast',

      '-crf',
      '20',

      '-pix_fmt',
      'yuv420p',

      '-c:a',
      'aac',

      '-b:a',
      '128k',

      '-movflags',
      '+faststart',

      outputPath

    ]
  );


  console.log(
    `Final video created: ${outputPath}`
  );


  /*
    temporary ASS ဖိုင်ဖျက်
  */

  try{

    await fs.unlink(
      ass.assPath
    );

  }catch(_){}


  return outputPath;

}


/* =========================================================
   TRANSCRIBE ROUTE
========================================================= */

app.post(
  '/api/transcribe',

  upload.single('video'),

  async (req,res)=>{

    let audioPath =
      null;


    try{

      if(!req.file){

        return res
          .status(400)
          .json({
            error:
              'Video file မတွေ့ပါ။'
          });

      }


      const groqKey =
        getKey(
          req,
          'x-groq-api-key',
          'GROQ_API_KEY'
        );


      if(!groqKey){

        return res
          .status(400)
          .json({
            error:
              'Groq API Key မထည့်ရသေးပါ။'
          });

      }


      const duration =
        await probeVideo(
          req.file.path
        );


      console.log(
        `Uploaded video: ${req.file.originalname}`
      );

      console.log(
        `Video duration: ${duration}s`
      );


      if(
        duration > MAX_SECONDS + 2
      ){

        return res
          .status(400)
          .json({
            error:
              'Video က 5 မိနစ်ထက်ရှည်နေပါတယ်။'
          });

      }


      if(
        duration <= 0
      ){

        return res
          .status(400)
          .json({
            error:
              'Video duration မမှန်ပါ။'
          });

      }


      audioPath =
        await extractAudio(
          req.file.path
        );


      const transcript =
        await transcribeGroq(
          audioPath,
          groqKey
        );


      const segments =
        buildPreciseSegments(
          transcript
        );


      if(
        !segments.length
      ){

        return res
          .status(422)
          .json({
            error:
              'Video ထဲမှာ မှတ်သားနိုင်တဲ့ စကားပြောသံ မတွေ့ပါ။'
          });

      }


      console.log(
        `Final subtitle segments: ${segments.length}`
      );


      for(
        const item of segments.slice(
          0,
          5
        )
      ){

        console.log(
          `[${item.start} - ${item.end}] ${item.text}`
        );

      }


      return res.json({

        ok:
          true,

        duration,

        language:
          transcript?.language ||
          null,

        text:
          transcript?.text ||
          segments
            .map(
              s => s.text
            )
            .join(' '),

        segments,

        /*
          HTML က transcript ကို
          ဒီ key ကနေယူနိုင်အောင်
          transcript alias ပါထည့်ထားတယ်
        */

        transcript:
          segments

      });


    }catch(error){

      console.error(
        'Transcribe error:',
        error
      );


      return res
        .status(500)
        .json({
          error:
            error?.message ||
            'Transcription မအောင်မြင်ပါ။'
        });


    }finally{

      if(audioPath){

        try{

          await fs.unlink(
            audioPath
          );

        }catch(_){}

      }


      if(req.file?.path){

        try{

          await fs.unlink(
            req.file.path
          );

        }catch(_){}

      }

    }

  }

);


/* =========================================================
   TRANSLATE ROUTE
========================================================= */

app.post(
  '/api/translate',

  async (req,res)=>{

    try{

      const geminiKey =
        getKey(
          req,
          'x-gemini-api-key',
          'GEMINI_API_KEY'
        );


      if(!geminiKey){

        return res
          .status(400)
          .json({
            error:
              'Gemini API Key မထည့်ရသေးပါ။'
          });

      }


      const segments =
        Array.isArray(
          req.body?.transcript
        )
          ? req.body.transcript
          : (
              Array.isArray(
                req.body?.segments
              )
                ? req.body.segments
                : []
            );


      if(
        !segments.length
      ){

        return res
          .status(400)
          .json({
            error:
              'Transcript မတွေ့ပါ။'
          });

      }


      const cleaned =
        segments
          .map(
            (item,index)=>({

              id:
                Number(
                  item.id ||
                  index + 1
                ),

              start:
                Number(
                  item.start
                ),

              end:
                Number(
                  item.end
                ),

              text:
                cleanText(
                  item.text
                )

            })
          )
          .filter(
            item =>
              item.text &&
              Number.isFinite(
                item.start
              ) &&
              Number.isFinite(
                item.end
              )
          );


      if(
        !cleaned.length
      ){

        return res
          .status(400)
          .json({
            error:
              'မှန်ကန်တဲ့ Transcript မရှိပါ။'
          });

      }


      console.log(
        `Translating ${cleaned.length} subtitle cues...`
      );


      const translated =
        await translateAllGemini(
          cleaned,
          geminiKey
        );


      const srt =
        makeSrt(
          translated
        );


      return res.json({

        ok:
          true,

        segments:
          translated,

        transcript:
          translated,

        srt

      });


    }catch(error){

      console.error(
        'Translation error:',
        error
      );


      if(
        isQuotaError(
          error
        )
      ){

        return res
          .status(429)
          .json({
            error:
              'Gemini API quota / rate limit ပြည့်နေပါတယ်။ ခဏစောင့်ပြီး ပြန်စမ်းပါ။'
          });

      }


      return res
        .status(500)
        .json({
          error:
            error?.message ||
            'Gemini Translation မအောင်မြင်ပါ။'
        });

    }

  }

);


/* =========================================================
   RENDER ROUTE
========================================================= */

app.post(
  '/api/render',

  upload.single('video'),

  async (req,res)=>{

    let outputPath =
      null;


    try{

      if(!req.file){

        return res
          .status(400)
          .json({
            error:
              'Video file မတွေ့ပါ။'
          });

      }


      const duration =
        await probeVideo(
          req.file.path
        );


      if(
        duration > MAX_SECONDS + 2
      ){

        return res
          .status(400)
          .json({
            error:
              'Video က 5 မိနစ်ထက်ရှည်နေပါတယ်။'
          });

      }


      let segments = [];


      try{

        segments =
          JSON.parse(
            req.body.segments ||
            '[]'
          );

      }catch(_){

        return res
          .status(400)
          .json({
            error:
              'Subtitle segments JSON မမှန်ပါ။'
          });

      }


      if(
        !Array.isArray(segments) ||
        !segments.length
      ){

        return res
          .status(400)
          .json({
            error:
              'မြန်မာ Subtitle မရှိသေးပါ။'
          });

      }


      let blurRegions = [];


      try{

        blurRegions =
          JSON.parse(
            req.body.blurRegions ||
            '[]'
          );

      }catch(_){

        blurRegions = [];

      }


      /*
        Subtitle text ကို
        Myanmar-only safe clean လုပ်
      */

      segments =
        segments
          .map(
            (item,index)=>({

              id:
                index + 1,

              start:
                Math.max(
                  0,
                  Number(
                    item.start
                  )
                ),

              end:
                Math.min(
                  duration,
                  Number(
                    item.end
                  )
                ),

              text:
                cleanText(
                  item.text ||
                  item.translation ||
                  ''
                )

            })
          )
          .filter(
            item =>
              item.text &&
              item.end >
              item.start
          );


      if(
        !segments.length
      ){

        return res
          .status(400)
          .json({
            error:
              'Render လုပ်ရန် Subtitle မရှိပါ။'
          });

      }


      const options = {

        fontSize:
          req.body.fontSize,

        outline:
          req.body.outline,

        color:
          req.body.color ||
          '#FFFFFF',

        position:
          req.body.position ||
          'bottom'

      };


      console.log(
        `Rendering ${segments.length} subtitle cues`
      );

      console.log(
        `Blur regions: ${blurRegions.length}`
      );


      outputPath =
        await renderVideo(
          req.file.path,
          segments,
          options,
          blurRegions
        );


      const stat =
        await fs.stat(
          outputPath
        );


      if(
        !stat.size
      ){

        throw new Error(
          'Output Video အလွတ်ဖြစ်နေပါတယ်။'
        );

      }


      res.setHeader(
        'Content-Type',
        'video/mp4'
      );

      res.setHeader(
        'Content-Length',
        String(
          stat.size
        )
      );

      res.setHeader(
        'Content-Disposition',
        'attachment; filename="Burmese-YNT-SRT.mp4"'
      );


      const buffer =
        await fs.readFile(
          outputPath
        );


      return res.end(
        buffer
      );


    }catch(error){

      console.error(
        'Render error:',
        error
      );


      return res
        .status(500)
        .json({
          error:
            error?.message ||
            'Video Render မအောင်မြင်ပါ။'
        });


    }finally{

      if(req.file?.path){

        try{

          await fs.unlink(
            req.file.path
          );

        }catch(_){}

      }


      if(outputPath){

        try{

          await fs.unlink(
            outputPath
          );

        }catch(_){}

      }

    }

  }

);


/* =========================================================
   HEALTH
========================================================= */

app.get(
  '/api/health',
  (req,res)=>{

    res.json({

      ok:
        true,

      service:
        'Burmese YNT SRT',

      groqModel:
        GROQ_MODEL,

      geminiModel:
        GEMINI_MODEL,

      videoRender:
        true,

      blur:
        true,

      subtitleBurn:
        true

    });

  }
);


/* =========================================================
   ROOT
========================================================= */

app.get(
  '*',
  (req,res,next)=>{

    if(
      req.path.startsWith('/api/')
    ){

      return next();

    }

    res.sendFile(
      path.join(
        PUBLIC,
        'index.html'
      )
    );

  }
);


/* =========================================================
   ERROR HANDLER
========================================================= */

app.use(
  (error,req,res,next)=>{

    console.error(
      'Unhandled error:',
      error
    );


    if(
      error instanceof multer.MulterError
    ){

      if(
        error.code ===
        'LIMIT_FILE_SIZE'
      ){

        return res
          .status(413)
          .json({
            error:
              'Video က 300MB ထက်ကြီးနေပါတယ်။'
          });

      }

    }


    return res
      .status(500)
      .json({
        error:
          error?.message ||
          'Server error ဖြစ်သွားပါတယ်။'
      });

  }
);


/* =========================================================
   START
========================================================= */

app.listen(
  PORT,
  ()=>{
    console.log(
      `Burmese YNT SRT server running on port ${PORT}`
    );

    console.log(
      `Groq model: ${GROQ_MODEL}`
    );

    console.log(
      `Gemini model: ${GEMINI_MODEL}`
    );

    console.log(
      'Video render: ENABLED'
    );

    console.log(
      'Original subtitle blur: ENABLED'
    );

  }
);
