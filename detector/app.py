import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from detector.service import Detector
from detector.documents import extract_document, MAX_FILE_BYTES, SUPPORTED, jsonl_records
from detector.text import MAX_CHARS
from detector.language import suggest_language
from detector.reporting import ReportRequest, render_report

STATIC = Path(__file__).parent / "static"
MAX_BODY = MAX_FILE_BYTES + 64 * 1024
Language = Literal["en", "hi", "ta", "kn", "hi_en", "ta_en", "kn_en"]


class BodyLimit:
    """Bound bodies, including chunked uploads, before multipart parsing."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > MAX_BODY:
                return await JSONResponse({"detail": "Upload is too large. Use a document under 10 MB."}, status_code=413)(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


@asynccontextmanager
async def lifespan(app):
    app.state.detector = None
    app.state.busy = asyncio.Lock()
    try:
        app.state.detector = await run_in_threadpool(Detector)
    except Exception:
        logging.exception("Model initialization failed. Run the documented setup steps.")
    yield


app = FastAPI(title="Lexora · AI text detection", lifespan=lifespan)
app.add_middleware(BodyLimit)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/fonts", StaticFiles(directory=Path(__file__).parent / "fonts"), name="fonts")


@app.middleware("http")
async def response_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    if request.url.path.startswith(("/api/", "/static/")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    ready = getattr(app.state, "detector", None) is not None
    return JSONResponse({"status": "ready" if ready else "unavailable", "model": "MuRIL + character n-grams"}, status_code=200 if ready else 503)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    language: Language


async def analyze(text, language):
    detector = getattr(app.state, "detector", None)
    if detector is None:
        raise HTTPException(503, "The model is not ready. Complete setup and restart the server.")
    if app.state.busy.locked():
        raise HTTPException(429, "Another passage is being analyzed. Please try again shortly.")
    async with app.state.busy:
        try:
            return await run_in_threadpool(detector.predict, text, language)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e


@app.post("/api/analyze")
async def analyze_text(body: TextRequest):
    return await analyze(body.text, body.language)


@app.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...), language: Language = Form(...)):
    try:
        if Path(file.filename or "").suffix.lower() not in SUPPORTED:
            raise HTTPException(415, "Upload a .txt, .md, .jsonl, .pdf, .docx or .doc file.")
        content = await file.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(413, "Please upload a file under 10 MB.")
        try:
            text, notes = await run_in_threadpool(extract_document, file.filename, content)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        result = await analyze(text, language)
        result["warnings"].extend(notes)
        return result
    finally:
        await file.close()


@app.post("/api/report.pdf")
async def export_pdf(report: ReportRequest):
    # Render the already-computed result without rerunning the encoder or
    # storing uploaded text/reports. This is an export, not signed evidence.
    content = await run_in_threadpool(render_report, report)
    return Response(content, media_type="application/pdf", headers={
        "Content-Disposition": 'attachment; filename="lexora-analysis.pdf"',
        "Cache-Control": "no-store",
    })


class SuggestRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)


@app.post("/api/suggest-language")
def language_suggestion(body: SuggestRequest):
    return suggest_language(body.text)


@app.post("/api/preview-file")
async def preview_file(file: UploadFile = File(...), jsonl_mode: Literal['combined', 'independent'] = Form('combined')):
    """Extract without running the detector or saving a document on the server."""
    try:
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(415, 'Choose a TXT, MD, JSONL, PDF, DOCX or DOC file.')
        content = await file.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(413, 'Please upload a file under 10 MB.')
        try:
            if suffix == '.jsonl' and jsonl_mode == 'independent':
                rows = await run_in_threadpool(jsonl_records, content)
                if len(rows) > 100:
                    raise ValueError('Use up to 100 independent JSONL records per batch.')
                documents = [dict(text=row['text'], name=f"Line {row['line']}" + (f" · ID {row['id']}" if row['id'] else ''),
                                  notes=['JSONL record analyzed independently. Labels and metadata are excluded.']) for row in rows]
            else:
                text, notes = await run_in_threadpool(extract_document, file.filename, content)
                documents = [dict(text=text, name='', notes=notes)]
        except (ValueError, UnicodeDecodeError) as error:
            raise HTTPException(422, str(error)) from error
        for document in documents:
            document['suggestion'] = suggest_language(document['text'])
        return {'documents': documents}
    finally:
        await file.close()
