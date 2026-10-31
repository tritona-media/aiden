# Aiden backend

The FastAPI API checks `backend/cv` for a CV named with the user's browser UUID
(`{user_id}.pdf` or `{user_id}.docx`) before calling the LangGraph supervisor.
When no CV is found, Aiden introduces itself, explains it needs the user's
profile first, and asks for a CV. The frontend displays the upload button below
that message. After upload, Aiden reads the CV and asks how it can help.
The frontend uses `GET /api/users/{user_id}/intro/stream` and
`POST /api/users/{user_id}/cv/stream` to render LangGraph model tokens as they
arrive. Streaming events use newline-delimited JSON: a CV metadata event,
individual text tokens, then a completion event.

## Run locally

Install the requirements and configure `OPENAI_API_KEY` in `backend/.env`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` and
`AIDEN_ALLOWED_ORIGINS` in `backend/.env`. Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Run the frontend separately with `npm run dev`. By default it connects to
`http://127.0.0.1:8000`; copy `.env.local.example` to `.env.local` to configure
`NEXT_PUBLIC_AIDEN_API_URL`.

## CV storage and privacy

Original PDF/DOCX files are stored under `backend/cv` with a server-generated
UUID filename; this directory is excluded from Git. Uploads are limited to
10 MB. The backend extracts text from the CV and sends it to the configured
LangChain model to generate a greeting.

The browser UUID is an identifier, not authentication. Add authentication and
authorization before using this for multiple real users or exposing it outside
a trusted local environment. Restrict access to CV files and define retention
and deletion policies before deploying with real personal data.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests
```
