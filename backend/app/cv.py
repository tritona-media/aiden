import asyncio
import io
import zipfile
from pathlib import Path
from uuid import UUID, uuid4

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from fastapi import HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

CV_DIRECTORY = Path(__file__).resolve().parents[1] / "cv"
ALLOWED_EXTENSIONS = (".pdf", ".docx")
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_TEXT_CHARS = 100_000
MAX_DOCX_UNCOMPRESSED_BYTES = 25 * 1024 * 1024


def find_user_cv(user_id: UUID) -> Path | None:
    for extension in ALLOWED_EXTENSIONS:
        candidate = CV_DIRECTORY / f"{user_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _extract_text(content: bytes, extension: str) -> str:
    try:
        if extension == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            if len(reader.pages) > 50:
                raise HTTPException(
                    status_code=413,
                    detail="Der Lebenslauf darf höchstens 50 Seiten enthalten.",
                )
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members = archive.infolist()
                if (
                    len(members) > 1000
                    or sum(member.file_size for member in members)
                    > MAX_DOCX_UNCOMPRESSED_BYTES
                ):
                    raise HTTPException(
                        status_code=413,
                        detail="Der entpackte Lebenslauf ist zu groß.",
                    )
            document = Document(io.BytesIO(content))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    except (PdfReadError, PackageNotFoundError, zipfile.BadZipFile) as error:
        raise HTTPException(
            status_code=422,
            detail="Die Datei ist kein lesbarer PDF- oder DOCX-Lebenslauf.",
        ) from error

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Der Lebenslauf enthält keinen lesbaren Text.",
        )
    if len(text) > MAX_EXTRACTED_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail="Der extrahierte Lebenslauftext ist zu umfangreich.",
        )
    return text


async def extract_cv_text(path: Path) -> str:
    content = await asyncio.to_thread(path.read_bytes)
    return await asyncio.to_thread(_extract_text, content, path.suffix.lower())


async def save_user_cv(user_id: UUID, upload: UploadFile) -> tuple[Path, str]:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Bitte lade deinen Lebenslauf als PDF oder DOCX hoch.",
        )

    content = await upload.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Die Datei darf maximal 10 MB groß sein.",
        )
    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="Die PDF-Datei ist ungültig.")
    if extension == ".docx" and not content.startswith(b"PK"):
        raise HTTPException(status_code=415, detail="Die DOCX-Datei ist ungültig.")

    text = await asyncio.to_thread(_extract_text, content, extension)
    await asyncio.to_thread(CV_DIRECTORY.mkdir, parents=True, exist_ok=True)

    target_path = CV_DIRECTORY / f"{user_id}{extension}"
    temporary_path = CV_DIRECTORY / f".{uuid4()}.tmp"
    try:
        await asyncio.to_thread(temporary_path.write_bytes, content)
        await asyncio.to_thread(temporary_path.replace, target_path)
    except OSError as error:
        if temporary_path.exists():
            await asyncio.to_thread(temporary_path.unlink)
        raise HTTPException(
            status_code=500,
            detail="Der Lebenslauf konnte nicht gespeichert werden.",
        ) from error
    for other_extension in ALLOWED_EXTENSIONS:
        if other_extension != extension:
            (CV_DIRECTORY / f"{user_id}{other_extension}").unlink(missing_ok=True)
    return target_path, text
