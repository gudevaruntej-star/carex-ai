# CareX AI — Automatic 503 Fallback

CareX is an AI-backed healthcare information kiosk prototype. The browser calls the CareX Node/Express backend; the Gemini API key stays server-side in `.env`.

## Run
1. `npm install`
2. Put your Gemini key in `.env` as `GEMINI_API_KEY=...`
3. `npm start`
4. Open `http://localhost:3000`

## AI reliability
- Primary model: `gemini-3.8-flash`
- Fallback model: `gemini-3.1-flash-lite`
- On HTTP 503 / `UNAVAILABLE`, CareX retries the primary model with exponential backoff + jitter, then automatically switches to the fallback model and retries it.
- Retry settings can be changed with `GEMINI_PRIMARY_RETRIES`, `GEMINI_FALLBACK_RETRIES`, `GEMINI_RETRY_BASE_MS`, and `GEMINI_RETRY_MAX_MS`.

Google documents 503 as a transient service-unavailable condition and recommends exponential backoff. Both configured models are listed in the current Gemini API model/structured-output documentation.

## Safety
CareX is for health information and report explanation. It should not be treated as a medical diagnosis or autonomous prescribing/dispensing system. Keep the API key private and rotate it if it is ever exposed publicly.
