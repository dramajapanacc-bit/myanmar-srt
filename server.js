import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { GoogleGenAI } from '@google/genai';

const app = express();
const PORT = Number(process.env.PORT || 10000);
const MAX_BYTES = Number(process.env.MAX_VIDEO_SIZE || 500) * 1024 * 1024;
const MAX_SECONDS = Number(process.env.MAX_VIDEO_SECONDS || 1800);
const MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
const TMP = path.join(os.tmpdir(), 'myanmar-movie-script');
await fs.mkdir(TMP, { recursive: true });

const videoExt = new Set(['.mp4','.mov','.mkv','.webm','.avi','.m4v','.mpeg','.mpg','.wmv','.flv','.3gp']);
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, TMP),
  filename: (_req, file, cb) => cb(null, `${Date.now()}-${crypto.randomBytes(8).toString('hex')}${path.extname(file.originalname || '').toLowerCase()}`)
});
const uploadVideo = multer({
  storage,
  limits: { fileSize: MAX_BYTES, files: 1 },
  fileFilter: (_req, file, cb) => videoExt.has(path.extname(file.originalname || '').toLowerCase())
    ? cb(null, true) : cb(new Error('MP4, MOV, MKV, WEBM, AVI video ဖိုင်ကိုသာ အသုံးပြုပါ။'))
});
app.use(express.json({ limit: '15mb' }));
app.use(express.static(path.join(process.cwd(), 'public')));

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const remove = file => file && fs.rm(file, { force: true }).catch(() => {});
function getApiKey(req) {
  const key = String(req.get('x-gemini-api-key') || process.env.GEMINI_API_KEY || '').trim();
  if (!key) throw new Error('Gemini API Key မထည့်ရသေးပါ။ Settings မှာ API Key ထည့်ပါ။');
  return key;
}
function getMime(file) {
  const ext = path.extname(file.originalname || '').toLowerCase();
  return ({ '.mp4':'video/mp4', '.mov':'video/quicktime', '.mkv':'video/x-matroska', '.webm':'video/webm', '.avi':'video/avi', '.m4v':'video/mp4', '.mpeg':'video/mpeg', '.mpg':'video/mpg', '.wmv':'video/x-ms-wmv', '.flv':'video/x-flv', '.3gp':'video/3gpp' })[ext] || 'video/mp4';
}
function cleanError(error) {
  const message = String(error?.message || error || 'Unknown error');
  if (/429|resource.?exhausted|quota/i.test(message)) return 'Gemini API quota ပြည့်နေပါတယ်။ Google AI Studio မှာ quota ပြန်စစ်ပြီး နောက်မှ ပြန်ကြိုးစားပါ။';
  if (/api.?key|unauthenticated|permission.?denied|403/i.test(message)) return 'Gemini API Key မမှန်ပါ သို့မဟုတ် API အသုံးပြုခွင့်မရှိပါ။';
  if (/503|unavailable|overloaded/i.test(message)) return 'Gemini server ခဏအလုပ်များနေပါတယ်။ ခဏနားပြီး ပြန်ကြိုးစားပါ။';
  return message.slice(0, 700);
}

function buildScriptPrompt(mode) {
  const base = `သင်သည် မြန်မာဘာသာဖြင့် Movie Recap / ဇာတ်လမ်းပြန်လည်ပြောပြသည့် Script ရေးသားရာတွင် ကျွမ်းကျင်သော ဇာတ်လမ်းပြောသူတစ်ဦးဖြစ်သည်။ ပေးထားသော VIDEO တစ်ခုလုံးကို အသံနှင့် ရုပ်ပုံနှစ်မျိုးလုံးအပေါ် အခြေခံ၍ သေချာကြည့်ရှုနားလည်ပြီး အောက်ပါစည်းမျဉ်းများအတိုင်း မြန်မာလို Script ရေးပါ။

အရေးကြီးသော စည်းမျဉ်းများ
1. Video ထဲတွင် တကယ်ဖြစ်ပျက်သော အကြောင်းအရာ၊ ဇာတ်ကောင်၊ ဆက်ဆံရေး၊ နေရာနှင့် အဖြစ်အပျက်များကိုသာ အသုံးပြုပါ။ မမြင်ရ၊ မကြားရသော အချက်အလက်များကို မတီထွင်ပါနှင့်။ မသေချာလျှင် သတိထားရေးပါ။
2. စကားလုံးချင်း ဘာသာပြန်ခြင်းမဟုတ်ပါ။ မြန်မာလူမျိုးတစ်ယောက်က ရုပ်ရှင်ဇာတ်လမ်းကို ပရိသတ်အား ပြန်ပြောပြနေသကဲ့သို့ သဘာဝကျကျ၊ ချောမွေ့စွာ၊ နားထောင်ကောင်းအောင် ရေးပါ။
3. ဇာတ်လမ်းအစမှ Video ထဲရှိ နောက်ဆုံးအထိ ဖြစ်စဉ်အစီအစဉ်မပျက် ရေးပါ။ အရေးကြီးသော အဖြစ်အပျက်၊ အကြောင်းရင်း၊ အကျိုးဆက်၊ ဇာတ်ကောင်တုံ့ပြန်မှုနှင့် အလှည့်အပြောင်းများကို မကျော်ပါနှင့်။
4. ဇာတ်ကောင်အမည်ကို သိရှိပါက တစ်သမတ်တည်းသုံးပါ။ မသိလျှင် သင့်တော်သည့် ဖော်ပြချက်ကို သုံးပါ။
5. Dialogue များကို စာကြောင်းလိုက် တိုက်ရိုက်မကူးဘဲ အဓိပ္ပာယ်နှင့် ခံစားချက်ကို ဇာတ်ကြောင်းထဲ သဘာဝကျကျ ပေါင်းစပ်ပါ။
6. မြန်မာ Unicode ဖြင့်သာ ရေးပါ။ AI မှတ်ချက်၊ အချိန်တံဆိပ်၊ SRT ပုံစံ မထည့်ပါနှင့်။
7. အကျဉ်းချုပ်လွန်းခြင်းမလုပ်ပါနှင့်။ အရေးကြီးသော scene များကို ပရိသတ်နားလည်နိုင်အောင် ဖော်ပြပါ။ မလိုအပ်သော ထပ်ကျော့မှုကိုသာ လျှော့ပါ။
8. Video အတွင်း အစပိုင်းက စိတ်ဝင်စားစေမည့် Hook တစ်ခု ထည့်ပါ။ Video ထဲက အဆုံးသတ်အတိုင်းသာ ရေးပြီး အပိုဇာတ်သိမ်း မတီထွင်ပါနှင့်။
9. ဇာတ်လမ်းထဲက အကြမ်းဖက်မှု သို့မဟုတ် အရွယ်ရောက်အကြောင်းအရာများကို လိုအပ်သည့်အတိုင်း သင့်တော်စွာဖော်ပြပါ။

အသေးစိတ် Recap ပုံစံ: အဖြစ်အပျက်များကို အစဉ်လိုက် အသေးစိတ်ရှင်းပြပါ။ ဇာတ်ကောင်များ ဘာကြောင့် ထိုသို့လုပ်ဆောင်သည်၊ အဖြစ်အပျက်တစ်ခုက နောက်တစ်ခုကို မည်သို့ဖြစ်စေသည်ကို နားလည်လွယ်အောင် ဆက်စပ်ရေးပါ။

YouTube Recap ပုံစံ: ပထမစာပိုဒ်မှာ စိတ်ဝင်စားစရာ Hook ပါရမည်။ စကားပြောသံစဉ်ဖြင့် နားထောင်ကောင်းအောင် ရေးပါ။ စာကြောင်းများကို သွက်လက်စွာ ဆက်စပ်ပြီး ပရိသတ်ကို နောက်တစ်ဆင့် ဘာဖြစ်မလဲ သိချင်စေပါ။ သို့သော် မူရင်း Video ထဲမရှိသော အချက်အလက် မထည့်ပါနှင့်။

ထုတ်ပေးမည့် Script အမျိုးအစား: ${mode === 'both' ? 'အသေးစိတ် Recap နှင့် YouTube Recap နှစ်မျိုးလုံး' : mode === 'youtube' ? 'YouTube Recap တစ်မျိုးတည်း' : 'အသေးစိတ် Recap တစ်မျိုးတည်း'}။

အောက်ပါ JSON ပုံစံအတိုင်းသာ ပြန်ပေးပါ။ Markdown code fence မထည့်ပါနှင့်။
{
  "title": "မြန်မာလို ဇာတ်လမ်းခေါင်းစဉ်တို",
  "summary": "ဇာတ်လမ်းအကျဉ်း ၂ မှ ၄ ကြောင်း",
  "characters": ["Video မှ အတည်ပြုနိုင်သော ဇာတ်ကောင်အမည်နှင့် အကျဉ်းဖော်ပြချက်"],
  "detailedScript": "အသေးစိတ် Recap Script",
  "youtubeScript": "YouTube Recap Narration Script"
}

မလိုအပ်သော Script field ကို mode အလိုက် စာသားအလွတ်ထားနိုင်သည်။ Script များကို စာပိုဒ်ခွဲပြီး newline ဖြင့် ရေးပါ။`;
  return base;
}

function parseModelJson(raw) {
  const text = String(raw || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '').trim();
  try { return JSON.parse(text); } catch {}
  const first = text.indexOf('{');
  const last = text.lastIndexOf('}');
  if (first >= 0 && last > first) {
    try { return JSON.parse(text.slice(first, last + 1)); } catch {}
  }
  return null;
}

app.get('/api/health', (_req, res) => res.json({ ok: true, app: 'Myanmar Movie Recap Script', model: MODEL, maxVideoMB: Math.round(MAX_BYTES / 1024 / 1024), maxVideoSeconds: MAX_SECONDS }));

app.post('/api/script/generate', uploadVideo.single('video'), async (req, res) => {
  const videoPath = req.file?.path;
  let geminiFile;
  try {
    if (!req.file) throw new Error('Video ဖိုင်ရွေးချယ်ပါ။');
    const mode = ['both', 'detailed', 'youtube'].includes(String(req.body?.scriptMode || 'both')) ? String(req.body?.scriptMode || 'both') : 'both';
    const ai = new GoogleGenAI({ apiKey: getApiKey(req) });
    const mimeType = getMime(req.file);
    geminiFile = await ai.files.upload({ file: videoPath, config: { mimeType, displayName: req.file.originalname } });
    for (let attempt = 0; attempt < 180; attempt++) {
      const state = String(geminiFile?.state || '').toUpperCase();
      if (state === 'ACTIVE') break;
      if (state === 'FAILED') throw new Error('Gemini က Video ကို process မလုပ်နိုင်ပါ။ အခြား video format ဖြင့် ပြန်စမ်းပါ။');
      if (attempt === 179) throw new Error('Gemini Video processing အချိန်ကြာလွန်းနေပါတယ်။');
      await sleep(3000);
      geminiFile = await ai.files.get({ name: geminiFile.name });
    }
    const response = await ai.models.generateContent({
      model: MODEL,
      contents: [{
        role: 'user',
        parts: [
          { fileData: { fileUri: geminiFile.uri, mimeType: geminiFile.mimeType || mimeType } },
          { text: buildScriptPrompt(mode) }
        ]
      }],
      config: { temperature: 0.55, maxOutputTokens: 24000, responseMimeType: 'application/json' }
    });
    const result = parseModelJson(response?.text || '');
    if (!result) throw new Error('Gemini က Script JSON ကို မှန်ကန်စွာ မထုတ်ပေးနိုင်ပါ။ နောက်တစ်ကြိမ် ပြန်ကြိုးစားပါ။');
    const detailedScript = String(result.detailedScript || (mode === 'detailed' ? result.script || '' : '')).trim();
    const youtubeScript = String(result.youtubeScript || (mode === 'youtube' ? result.script || '' : '')).trim();
    if ((mode === 'both' && !detailedScript && !youtubeScript) || (mode === 'detailed' && !detailedScript) || (mode === 'youtube' && !youtubeScript)) {
      throw new Error('Gemini က တောင်းဆိုထားသော Script အပြည့်အစုံ မထုတ်ပေးနိုင်ပါ။ နောက်တစ်ကြိမ် ပြန်ကြိုးစားပါ။');
    }
    res.json({ ok: true, model: MODEL, title: String(result.title || 'Movie Recap Script'), summary: String(result.summary || ''), characters: Array.isArray(result.characters) ? result.characters : [], detailedScript, youtubeScript, script: mode === 'youtube' ? youtubeScript : detailedScript, filename: req.file.originalname });
  } catch (error) {
    console.error('SCRIPT GENERATION ERROR:', error);
    res.status(400).json({ ok: false, error: cleanError(error) });
  } finally {
    remove(videoPath);
    // Gemini Files are retained temporarily by Google; delete after generation where supported.
    if (geminiFile?.name) {
      try { const ai = new GoogleGenAI({ apiKey: getApiKey(req) }); await ai.files.delete({ name: geminiFile.name }); } catch {}
    }
  }
});

app.post('/api/script/export', (req, res) => {
  const script = String(req.body?.script || '').trim();
  const title = String(req.body?.title || 'movie-recap-script').replace(/[\\/:*?"<>|\r\n]+/g, '-').slice(0, 80);
  if (!script) return res.status(400).json({ ok: false, error: 'Download လုပ်ရန် Script မရှိပါ။' });
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename="${encodeURIComponent(title)}.txt"`);
  res.send(script);
});

app.use((err, _req, res, _next) => {
  console.error('REQUEST ERROR:', err);
  const message = err?.code === 'LIMIT_FILE_SIZE' ? `Video ဖိုင်အရွယ်အစား ${Math.round(MAX_BYTES / 1024 / 1024)} MB ထက် မကြီးရပါ။` : cleanError(err);
  res.status(400).json({ ok: false, error: message });
});
app.get('/{*splat}', (_req, res) => res.sendFile(path.join(process.cwd(), 'public', 'index.html')));
app.listen(PORT, () => console.log(`StoryWave Studio running on port ${PORT} | model=${MODEL}`));
