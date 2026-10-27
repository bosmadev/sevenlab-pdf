# sevenlab-pdf

Internal SevenLab PDF chat application. FastAPI + Next.js 16 with Cohere V5-Pro embeddings and MongoDB Atlas hybrid search.

- **Backend:** FastAPI (Python 3.11), binds `127.0.0.1` only
- **Frontend:** Next.js 16 (App Router), TypeScript, Tailwind
- **Vector DB:** MongoDB Atlas Vector Search (2048-dim)
- **Embeddings:** Cohere Embed V5-Pro on Azure
- **OCR:** Proximity captions -> Tesseract -> Azure Computer Vision v4
- **Domain:** `sevenlab.bosma.dev` (deployed at `/opt/sevenlab`)

Plan: `~/.claude/plans/SEVENLAB-PDFCHAT.md`

## Quick start

```bash
# Backend
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env   # fill in secrets
uvicorn app.main:app --host 127.0.0.1 --port 8700 --reload

# Frontend
cd frontend
npm install
cp .env.sample .env.local
npm run dev
```

External traffic terminates at Caddy -> `127.0.0.1:<port>`. Never expose FastAPI on `0.0.0.0`.
