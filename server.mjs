import express from 'express';
import multer from 'multer';
import dotenv from 'dotenv';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const PORT = process.env.PORT || 3000;
const PRIMARY_MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
const FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL || 'gemini-3.1-flash-lite';
const PRIMARY_RETRIES = Math.max(0, Number(process.env.GEMINI_PRIMARY_RETRIES || 1));
const FALLBACK_RETRIES = Math.max(0, Number(process.env.GEMINI_FALLBACK_RETRIES || 1));
const RETRY_BASE_MS = Math.max(250, Number(process.env.GEMINI_RETRY_BASE_MS || 1000));
const RETRY_MAX_MS = Math.max(RETRY_BASE_MS, Number(process.env.GEMINI_RETRY_MAX_MS || 8000));
const MAX_FILE_MB = Number(process.env.MAX_FILE_MB || 20);
const MAX_BYTES = MAX_FILE_MB * 1024 * 1024;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_BYTES, files: 1 },
  fileFilter: (_req, file, cb) => {
    const allowed = new Set(['application/pdf', 'image/jpeg', 'image/png', 'image/webp']);
    cb(null, allowed.has(file.mimetype));
  }
});

app.use(express.json({ limit: '2mb' }));
app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'carex.html'));
});

function getClient() {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return null;
  return new GoogleGenAI({ apiKey: key });
}

function getErrorStatus(err) {
  const direct = Number(err?.status ?? err?.statusCode ?? err?.response?.status ?? err?.error?.code ?? err?.code);
  if (Number.isFinite(direct) && direct > 0) return direct;
  const message = String(err?.message || err || '');
  if (/\b503\b/.test(message) || /UNAVAILABLE|service_unavailable/i.test(message)) return 503;
  return 0;
}

function is503(err) {
  return getErrorStatus(err) === 503;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function backoffDelay(attempt) {
  const exponential = Math.min(RETRY_MAX_MS, RETRY_BASE_MS * (2 ** attempt));
  const jitter = Math.floor(Math.random() * Math.min(500, RETRY_BASE_MS));
  return exponential + jitter;
}

async function generateWithFallback(ai, requestConfig) {
  let last503 = null;

  for (let attempt = 0; attempt <= PRIMARY_RETRIES; attempt++) {
    try {
      const response = await ai.models.generateContent({ ...requestConfig, model: PRIMARY_MODEL });
      return { response, model: PRIMARY_MODEL, fallbackUsed: false };
    } catch (err) {
      if (!is503(err)) throw err;
      last503 = err;
      if (attempt < PRIMARY_RETRIES) await sleep(backoffDelay(attempt));
    }
  }

  for (let attempt = 0; attempt <= FALLBACK_RETRIES; attempt++) {
    try {
      const response = await ai.models.generateContent({ ...requestConfig, model: FALLBACK_MODEL });
      return { response, model: FALLBACK_MODEL, fallbackUsed: true };
    } catch (err) {
      if (!is503(err)) throw err;
      last503 = err;
      if (attempt < FALLBACK_RETRIES) await sleep(backoffDelay(attempt));
    }
  }

  const error = new Error(`CareX AI is temporarily unavailable. Both ${PRIMARY_MODEL} and ${FALLBACK_MODEL} returned 503 Service Unavailable.`);
  error.status = 503;
  error.cause = last503;
  throw error;
}

function parseJson(text) {
  try { return JSON.parse(text); } catch {}
  const match = text?.match(/\{[\s\S]*\}/);
  if (match) {
    try { return JSON.parse(match[0]); } catch {}
  }
  throw new Error('AI returned an invalid response format.');
}

const medicalSystemInstruction = `
You are CareX, a cautious health-information assistant for a kiosk in India.
Your job is to extract and explain medical information, not to replace a clinician.
Always answer in clear, simple Telugu, while preserving medicine names, test names, units, and standard medical terms in English when that is clearer.
Never invent a diagnosis, lab value, medicine, dosage, or medical history.
When analyzing a report, distinguish clearly between what is explicitly present and what is uncertain.
Never recommend starting, stopping, changing, or dispensing prescription medicines based only on your own reasoning.
If the report contains a prescription, transcribe medicines/doses as written and say that dispensing must be based on clinician-verified instructions.
For potentially life-threatening symptoms, clearly advise urgent/emergency medical care.
`;

const reportSchema = {
  type: 'object',
  properties: {
    patientName: { type: 'string' },
    diagnosis: { type: 'string' },
    summary_telugu: { type: 'string' },
    findings: { type: 'array', items: { type: 'string' } },
    medicines: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          strength: { type: 'string' },
          dosage: { type: 'string' },
          frequency: { type: 'string' },
          duration: { type: 'string' }
        },
        required: ['name', 'strength', 'dosage', 'frequency', 'duration']
      }
    },
    confidence_note_telugu: { type: 'string' },
    urgency: { type: 'string', enum: ['EMERGENCY', 'URGENT', 'ROUTINE'] }
  },
  required: ['patientName', 'diagnosis', 'summary_telugu', 'findings', 'medicines', 'confidence_note_telugu', 'urgency']
};

const answerSchema = {
  type: 'object',
  properties: {
    answer_telugu: { type: 'string' },
    urgency: { type: 'string', enum: ['EMERGENCY', 'URGENT', 'ROUTINE'] },
    related_topic: { type: 'string' },
    caution_telugu: { type: 'string' }
  },
  required: ['answer_telugu', 'urgency', 'related_topic', 'caution_telugu']
};

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    aiConfigured: Boolean(process.env.GEMINI_API_KEY),
    model: PRIMARY_MODEL,
    primaryModel: PRIMARY_MODEL,
    fallbackModel: FALLBACK_MODEL,
    retryPolicy: { primaryRetries: PRIMARY_RETRIES, fallbackRetries: FALLBACK_RETRIES, baseDelayMs: RETRY_BASE_MS },
    provider: 'Google Gemini'
  });
});

app.post('/api/analyze-report', upload.single('report'), async (req, res) => {
  try {
    const ai = getClient();
    if (!ai) return res.status(503).json({ error: 'Gemini API key is not configured. Add GEMINI_API_KEY to the server environment.' });
    if (!req.file) return res.status(400).json({ error: 'Please upload a PDF or supported image.' });

    const prompt = `Analyze the attached medical report for CareX. Extract only information visible or clearly stated in the report.
Return JSON matching the supplied schema.
- patientName: patient name if visible; otherwise "Not visible"
- diagnosis: diagnosis/condition if explicitly stated; otherwise "Not stated"
- summary_telugu: a simple Telugu explanation of what the report says
- findings: important test findings or observations; include high/low flags only when the report itself gives a reference range or clear marker
- medicines: prescription medicines visible in the report; do not invent any. If none, return an empty array.
- confidence_note_telugu: say what could not be read or what should be confirmed by a clinician/pharmacist
- urgency: EMERGENCY only if the report itself contains a clearly urgent/life-threatening finding; URGENT for findings that warrant prompt medical review; otherwise ROUTINE.`;

    const contents = [
      { text: prompt },
      { inlineData: { mimeType: req.file.mimetype, data: req.file.buffer.toString('base64') } }
    ];

    const aiRun = await generateWithFallback(ai, {
      contents,
      config: {
        systemInstruction: medicalSystemInstruction,
        responseMimeType: 'application/json',
        responseSchema: reportSchema,
        temperature: 0.1
      }
    });

    const data = parseJson(aiRun.response.text);
    res.set('X-CareX-AI-Model', aiRun.model);
    res.set('X-CareX-AI-Fallback', String(aiRun.fallbackUsed));
    res.json(data);
  } catch (err) {
    console.error('analyze-report error', err);
    const message = err?.message || 'Unable to analyze the report.';
    res.status(getErrorStatus(err) === 503 ? 503 : 500).json({ error: message });
  }
});

app.post('/api/ask', async (req, res) => {
  try {
    const ai = getClient();
    if (!ai) return res.status(503).json({ error: 'Gemini API key is not configured. Add GEMINI_API_KEY to the server environment.' });
    const question = String(req.body?.question || '').trim();
    if (!question) return res.status(400).json({ error: 'Please enter a health question.' });

    const reportContext = req.body?.reportContext ? JSON.stringify(req.body.reportContext) : 'No report has been uploaded in this session.';
    const prompt = `Answer this user's health question.
User question: ${question}

Current CareX report context (may be absent; treat it only as user-provided context):
${reportContext}

Return JSON matching the supplied schema.
- answer_telugu: direct, simple Telugu answer. Do not pretend to diagnose.
- urgency: EMERGENCY for immediately dangerous symptoms; URGENT for symptoms needing prompt medical attention; ROUTINE otherwise.
- related_topic: concise medical topic, or "General health information".
- caution_telugu: brief safety note. Do not provide a medication prescription or dose change.`;

    const aiRun = await generateWithFallback(ai, {
      contents: prompt,
      config: {
        systemInstruction: medicalSystemInstruction,
        responseMimeType: 'application/json',
        responseSchema: answerSchema,
        temperature: 0.2
      }
    });

    const data = parseJson(aiRun.response.text);
    res.set('X-CareX-AI-Model', aiRun.model);
    res.set('X-CareX-AI-Fallback', String(aiRun.fallbackUsed));
    res.json(data);
  } catch (err) {
    console.error('ask error', err);
    res.status(getErrorStatus(err) === 503 ? 503 : 500).json({ error: err?.message || 'Unable to get an AI answer.' });
  }
});

app.get(/.*/, (_req, res) => res.sendFile(path.join(__dirname, 'carex.html')));

app.listen(PORT, () => {
  console.log(`CareX running at http://localhost:${PORT}`);
  console.log(`Gemini configured: ${Boolean(process.env.GEMINI_API_KEY)} | Primary: ${PRIMARY_MODEL} | Fallback: ${FALLBACK_MODEL}`);
});
