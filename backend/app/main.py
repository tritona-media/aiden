import logging
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx
from openai import APIError
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from backend.app.agent import get_agent_graph, introduce, stream_introduction
from backend.app.cv import extract_cv_text, find_user_cv, save_user_cv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

app = FastAPI(title="Aiden Job Assistant API", version="0.1.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "AIDEN_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class IntroductionResponse(BaseModel):
    message: str
    has_cv: bool


async def create_introduction(cv_text: str | None) -> str:
    try:
        return await introduce(cv_text)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def encode_stream_event(event: dict[str, object]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


async def introduction_events(
    has_cv: bool,
) -> AsyncIterator[bytes]:
    """Stream introduction events as JSON lines:
    {"type": "meta", "has_cv": true/false}
    {"type": "token", "text": "..."}
    {"type": "done"} or {"type": "error", "message": message}
    """
    yield encode_stream_event({"type": "meta", "has_cv": has_cv})
    received_text = False
    try:
        async for text in stream_introduction(has_cv):
            received_text = True
            yield encode_stream_event({"type": "token", "text": text})
        if not received_text:
            raise RuntimeError("The supervisor returned an empty introduction.")
        yield encode_stream_event({"type": "done"})
    except (RuntimeError, APIError, httpx.HTTPError) as error:
        logger.exception("Could not stream Aiden's introduction")
        message = (
            str(error)
            if isinstance(error, RuntimeError)
            else "Aiden kann gerade keine Antwort generieren. Bitte versuche es erneut."
        )
        yield encode_stream_event({"type": "error", "message": message})


def stream_response(has_cv: bool) -> StreamingResponse:
    # check if agent graph is ok
    try:
        get_agent_graph()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    # stream jsons line by line
    return StreamingResponse(
        introduction_events(has_cv),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/users/{user_id}/intro/stream")
async def get_introduction_stream(user_id: UUID) -> StreamingResponse:
    cv_path = find_user_cv(user_id)
    return stream_response(has_cv=cv_path is not None)


@app.post("/api/users/{user_id}/cv", response_model=IntroductionResponse)
async def upload_cv(
    user_id: UUID,
    file: UploadFile = File(...),
) -> IntroductionResponse:
    _, cv_text = await save_user_cv(user_id, file)
    message = await create_introduction(cv_text)
    return IntroductionResponse(message=message, has_cv=True)


@app.post("/api/users/{user_id}/cv/stream")
async def upload_cv_stream(
    user_id: UUID,
    file: UploadFile = File(...),
) -> StreamingResponse:
    _, cv_text = await save_user_cv(user_id, file)
    return stream_response(cv_text, has_cv=True)
