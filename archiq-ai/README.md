# ARCHIQ AI

AI-powered construction norms assistant for Kazakhstan (РК).

## Project Structure

```
archiq-ai/
├── .github/workflows/deploy.yml   # Auto-deploy backend to Render
├── backend/
│   ├── main.py                     # FastAPI + Gemini + HF OCR
│   ├── requirements.txt
│   └── Dockerfile
├── mobile/
│   ├── package.json                # Capacitor deps
│   └── www/index.html              # Web UI (becomes mobile app)
├── render.yaml                     # Render Blueprint config
└── README.md                       # This file
```

## Quick Start

### 1. Upload to GitHub
1. Create a new **public** repository named `archiq-ai`
2. Upload all files from this folder
3. Commit

### 2. Get API Keys

| Secret | Where to get | Purpose |
|--------|--------------|---------|
| `GEMINI_API_KEY` | [Google AI Studio](https://makersuite.google.com/app/apikey) | AI analysis |
| `HF_API_KEY` | [HuggingFace Tokens](https://huggingface.co/settings/tokens) | OCR documents |
| `RENDER_API_KEY` | [Render Dashboard](https://dashboard.render.com) → Account → API Keys | Hosting deploys |

### 3. Add GitHub Secrets
In your repo: **Settings → Secrets and variables → Actions → New repository secret**

Add:
- `GEMINI_API_KEY`
- `HF_API_KEY`
- `RENDER_API_KEY`
- `RENDER_SERVICE_ID` — leave empty (auto-filled on first deploy)

### 4. Deploy Backend
Push to `main` branch — GitHub Actions will auto-deploy to Render via `render.yaml` Blueprint.

After first deploy:
1. Open Render Dashboard
2. Find service `archiq-ai-api`
3. Copy Service ID (from URL or settings)
4. Save it as `RENDER_SERVICE_ID` secret in GitHub

### 5. Build Mobile App (Capacitor)
```bash
cd mobile
npm install
npx cap add android   # or ios
npm run build
npx cap sync
npx cap open android
```

### 6. Google Play Auto-Deploy (Optional)
1. [Google Play Console](https://play.google.com/console) → Setup → API Access → Create Service Account
2. Role: **Editor**
3. Keys → Add Key → JSON → save as `archiq-deployer-xxx.json`
4. Add to GitHub Secrets as `PLAY_STORE_JSON_KEY` (paste full JSON text)
5. Create keystore and add as `KEYSTORE_BASE64` secret

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service status |
| GET | `/norms?search=` | Search construction norms (РК) |
| POST | `/ask` | Ask AI with norm context |
| POST | `/ocr` | OCR via HuggingFace |
| POST | `/analyze-document` | OCR + AI analysis |
| GET | `/documents` | List uploaded docs |

## Environment Variables

| Var | Default | Description |
|-----|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `HF_API_KEY` | — | HuggingFace API token |
| `HF_OCR_MODEL` | `microsoft/trocr-base-printed` | OCR model on HF |
| `PORT` | `8000` | Server port |
| `DB_PATH` | `norms.db` | SQLite path |

## Contact
For issues contact: **evratnikov83@gmail.com**
