import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile

CV_DIRECTORY = Path(__file__).resolve().parents[1] / "cv"
ALLOWED_EXTENSIONS = (".pdf", ".docx")
MAX_FILE_BYTES = 10 * 1024 * 1024


def find_user_cv(user_id: UUID) -> Path | None:
    candidate = CV_DIRECTORY / f"{user_id}.json"
    if candidate.is_file():
        return candidate
    return None


async def save_user_cv(user_id: UUID, upload: UploadFile) -> Path:
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
    return target_path
