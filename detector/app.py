import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from detector.service import Detector

STATIC = Path(__file__).parent / "static"
MAX_BODY = 64 * 1024
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
                return await JSONResponse({"detail": "Upload is too large. Use a UTF-8 text file under 48 KB."}, status_code=413)(scope, receive, send)
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


app = FastAPI(title="Origin · AI text detection", lifespan=lifespan)
app.add_middleware(BodyLimit)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def response_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    if request.url.path.startswith("/api/"):
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
    text: str = Field(min_length=1, max_length=12000)
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
        if Path(file.filename or "").suffix.lower() not in (".txt", ".md"):
            raise HTTPException(415, "Upload a UTF-8 .txt or .md file.")
        content = await file.read(48001)
        if len(content) > 48000:
            raise HTTPException(413, "Please upload a text file under 48 KB.")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise HTTPException(422, "The file must be UTF-8 text.") from e
        if "\x00" in text:
            raise HTTPException(422, "This file contains binary data. Upload plain text.")
        return await analyze(text, language)
    finally:
        await file.close()
