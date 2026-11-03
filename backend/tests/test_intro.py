import asyncio
import io
import json
from uuid import uuid4

from docx import Document
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from backend.app import cv, main
from backend.app.agent import (
    SYSTEM_PROMPT,
    create_agent_graph,
    get_agent_graph,
    stream_introduction,
)


def create_docx(*paragraphs: str) -> bytes:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    content = io.BytesIO()
    document.save(content)
    return content.getvalue()


def test_agent_graph_uses_system_prompt() -> None:
    graph = create_agent_graph(
        FakeListChatModel(responses=["Hallo, ich bin Aiden. Bitte lade deinen CV hoch."])
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Ich bin neu und habe noch keinen Lebenslauf hochgeladen.",
                    }
                ]
            }
        )
    )
    assert result["messages"][-1].content.startswith("Hallo")
    assert result["messages"][-1].content.startswith("Hallo")
    assert "persönlicher Job- und Bewerbungsassistent" in SYSTEM_PROMPT
    assert "Antworte ausschließlich auf Deutsch" in SYSTEM_PROMPT


def test_supervisor_streams_model_tokens(monkeypatch) -> None:
    graph = create_agent_graph(
        FakeListChatModel(responses=["Hallo, ich bin Aiden. Willkommen!"])
    )
    monkeypatch.setattr("backend.app.agent.get_agent_graph", lambda: graph)

    async def collect_chunks() -> list[str]:
        return [chunk async for chunk in stream_introduction(False)]

    chunks = asyncio.run(collect_chunks())

    assert "".join(chunks) == "Hallo, ich bin Aiden. Willkommen!"
    assert len(chunks) > 1


def test_intro_stream_sends_metadata_tokens_and_completion(monkeypatch) -> None:
    user_id = uuid4()
    monkeypatch.setattr(main, "find_user_cv", lambda _: None)
    graph = create_agent_graph(
        FakeListChatModel(responses=["Hallo, ich bin Aiden. Willkommen!"])
    )
    monkeypatch.setattr("backend.app.agent.get_agent_graph", lambda: graph)
    monkeypatch.setattr(main, "get_agent_graph", lambda: graph)

    response = TestClient(main.app).get(
        f"/api/users/{user_id}/intro/stream"
    )

    events = [json.loads(line) for line in response.text.splitlines()]
    token_events = [event["text"] for event in events if event["type"] == "token"]
    assert events[0] == {"type": "meta", "has_cv": False}
    assert "".join(token_events) == "Hallo, ich bin Aiden. Willkommen!"
    assert events[-1] == {"type": "done"}


def test_cv_upload_streams_personalized_greeting(monkeypatch, tmp_path) -> None:
    user_id = uuid4()
    monkeypatch.setattr(cv, "CV_DIRECTORY", tmp_path / "cv")
    graph = create_agent_graph(
        FakeListChatModel(responses=["Hallo Ada, wie kann ich dir helfen?"])
    )
    monkeypatch.setattr("backend.app.agent.get_agent_graph", lambda: graph)
    monkeypatch.setattr(main, "get_agent_graph", lambda: graph)

    response = TestClient(main.app).post(
        f"/api/users/{user_id}/cv/stream",
        files={
            "file": (
                "lebenslauf.docx",
                create_docx("Ada Beispiel", "Softwareentwicklerin"),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    events = [json.loads(line) for line in response.text.splitlines()]
    token_events = [event["text"] for event in events if event["type"] == "token"]
    assert events[0] == {"type": "meta", "has_cv": True}
    assert "".join(token_events) == "Hallo Ada, wie kann ich dir helfen?"
    assert events[-1] == {"type": "done"}
    assert (tmp_path / "cv" / f"{user_id}.docx").is_file()


def test_pdf_upload_stores_file_without_extracting_text(monkeypatch, tmp_path) -> None:
    user_id = uuid4()
    cv_directory = tmp_path / "cv"
    pdf_content = b"%PDF-1.7\nPDF CV contents"
    monkeypatch.setattr(cv, "CV_DIRECTORY", cv_directory)

    async def fake_stream_introduction(has_cv: bool):
        assert has_cv
        yield "Hallo Ada, wie kann ich dir helfen?"

    monkeypatch.setattr(main, "get_agent_graph", lambda: object())
    monkeypatch.setattr(main, "stream_introduction", fake_stream_introduction)
    response = TestClient(main.app).post(
        f"/api/users/{user_id}/cv/stream",
        files={"file": ("lebenslauf.pdf", pdf_content, "application/pdf")},
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0] == {"type": "meta", "has_cv": True}
    assert events[-1] == {"type": "done"}
    assert (cv_directory / f"{user_id}.pdf").read_bytes() == pdf_content


def test_upload_rejects_unsupported_file() -> None:
    response = TestClient(main.app).post(
        f"/api/users/{uuid4()}/cv/stream",
        files={"file": ("resume.txt", b"not a CV", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_reports_missing_model_configuration(monkeypatch, tmp_path) -> None:
    async def clear_cached_graph() -> None:
        get_agent_graph.cache_clear()

    monkeypatch.setattr(cv, "CV_DIRECTORY", tmp_path / "cv")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    asyncio.run(clear_cached_graph())

    response = TestClient(main.app).post(
        f"/api/users/{uuid4()}/cv/stream",
        files={
            "file": (
                "lebenslauf.docx",
                create_docx("Ada Beispiel"),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "OPENAI_API_KEY is not configured."
