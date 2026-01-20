import os
import time
from typing import Optional, Dict, Any, List, Tuple

import requests
from fastapi import FastAPI, Request, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ---------------- CONFIG ----------------
PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()  # openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

# Knowledge base (OpenAI Vector Store)
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()

# Secrets
# NOTE:
# - ADMIN_SECRET: always required for /admin/*
# - CHATBOT_SECRET: optional for /chat (recommended), but we accept it in multiple ways (header/body/query)
CHATBOT_SECRET = os.getenv("CHATBOT_SECRET", "").strip()
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "").strip()

# CORS (only your domains)
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "https://videmark.hu,https://www.videmark.hu").split(",")
    if o.strip()
]

SYSTEM_PROMPT = """
Te a Videmark weboldal hivatalos asszisztense vagy.

Feladatod:
- Ügyféltájékoztatás: drón videó/fotó, reklámvideó, short tartalom (TikTok/Reels/Shorts), fotózás, utómunka,
  social média támogatás, AI megoldások.
- Leadgyűjtés: ha a felhasználó ajánlatot kér vagy érdeklődik, kérd be:
  Név, Email, Telefonszám (opcionális), Helyszín, Határidő, Mire kell (videó/fotó/drón/short/fotózás), mennyiség.

Stílus:
- magyarul, tömören, érthetően
- max 2 kérdés egyszerre
- javasolj következő lépést (pl. "kérsz gyors árajánlatot?")

Fontos szabály:
- Ha a tudásbázisban nem találsz releváns infót, ezt mondd ki őszintén, és kérj pontosítást.
""".strip()

# Rate limit (simple in-memory)
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))  # 30 req/min/IP
_ip_hits: Dict[str, List[float]] = {}

# File search tuning
FILE_SEARCH_MAX_RESULTS = int(os.getenv("FILE_SEARCH_MAX_RESULTS", "5"))
OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "45"))
DEBUG_FILE_SEARCH = os.getenv("DEBUG_FILE_SEARCH", "0").strip() == "1"


# ---------------- APP ----------------
app = FastAPI(title="Videmark Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatReq(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[List[Dict[str, Any]]] = None  # [{role:"user/assistant", content:"..."}]
    # NEW: token can be sent in body as well (so frontend can avoid custom headers if needed)
    chatbot_secret: Optional[str] = None


class ChatResp(BaseModel):
    reply: str


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "videmark-chatbot-api",
        "provider": PROVIDER,
        "model": OPENAI_MODEL,
        "vector_store_configured": bool(OPENAI_VECTOR_STORE_ID),
        "chat_secret_required": bool(CHATBOT_SECRET),
        "allowed_origins": ALLOWED_ORIGINS,
    }


def _rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0
    hits = _ip_hits.get(ip, [])
    hits = [t for t in hits if now - t < window]
    if len(hits) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Túl sok kérés. Próbáld újra kicsit később.")
    hits.append(now)
    _ip_hits[ip] = hits


def _require_admin_secret(x_admin_secret: str):
    if not ADMIN_SECRET:
        raise HTTPException(status_code=500, detail="ADMIN_SECRET nincs beállítva a szerveren.")
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized (admin secret).")


def _get_chat_token(
    header_token: str,
    query_token: str,
    body_token: Optional[str],
) -> str:
    """
    Accept token from:
    - Header: X-Chatbot-Secret: ...
    - Query: ?chatbot_secret=...
    - Body: {"chatbot_secret": "..."}
    """
    if header_token:
        return header_token.strip()
    if query_token:
        return query_token.strip()
    if body_token:
        return str(body_token).strip()
    return ""


def _require_chat_secret(token: str):
    # If CHATBOT_SECRET is empty, allow public chat (not recommended, but convenient)
    if CHATBOT_SECRET and token != CHATBOT_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized (chat secret).")


def _openai_headers() -> Dict[str, str]:
    if not OPENAI_API_KEY:
        return {}
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}


def _ok_status(code: int) -> bool:
    return 200 <= code < 300


def _extract_output_text(data: Dict[str, Any]) -> str:
    """
    Robust extraction for Responses API output.

    Common patterns:
    - data["output_text"]
    - data["output"] -> items -> content -> output_text parts
    """
    if not isinstance(data, dict):
        return "Nem tudtam választ generálni (OpenAI válasz formátum)."

    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    out = data.get("output", [])
    if isinstance(out, list):
        # Walk through output items and collect output_text
        chunks: List[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    t = part["text"].strip()
                    if t:
                        chunks.append(t)
        if chunks:
            return "\n".join(chunks)

    return "Nem tudtam választ generálni. (Nincs output_text a válaszban.)"


def _build_responses_payload(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    input_msgs = [{"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]}]
    for m in messages:
        input_msgs.append({"role": m["role"], "content": [{"type": "input_text", "text": m["content"]}]})

    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": input_msgs,
        "temperature": 0.4,
        "max_output_tokens": 650,
    }

    if OPENAI_VECTOR_STORE_ID:
        payload["tools"] = [{
            "type": "file_search",
            "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
            "max_num_results": FILE_SEARCH_MAX_RESULTS,
        }]

        # Debug: include search results in response (ONLY if you set DEBUG_FILE_SEARCH=1)
        if DEBUG_FILE_SEARCH:
            payload["include"] = ["file_search_call.results"]

    return payload


def call_openai_with_filesearch(messages: List[Dict[str, str]]) -> Tuple[str, Optional[str]]:
    """
    Returns: (reply_text, debug_info_optional)
    debug_info is only present if DEBUG_FILE_SEARCH enabled.
    """
    if not OPENAI_API_KEY:
        return "Hiányzik az OPENAI_API_KEY a szerveren.", None

    url = "https://api.openai.com/v1/responses"
    headers = {**_openai_headers(), "Content-Type": "application/json"}
    payload = _build_responses_payload(messages)

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SEC)
    except Exception as e:
        return f"Hiba az OpenAI hívásnál (kapcsolat): {str(e)[:200]}", None

    if not _ok_status(r.status_code):
        # Helpful error
        return f"Hiba az OpenAI hívásnál: {r.status_code} – {r.text[:400]}", None

    data = r.json()
    reply = _extract_output_text(data)

    dbg = None
    if DEBUG_FILE_SEARCH:
        # lightweight debug snippet
        try:
            dbg = str(data.get("output", ""))[:800]
        except Exception:
            dbg = None

    return reply, dbg


# ---------------- PUBLIC CHAT ----------------
@app.post("/chat", response_model=ChatResp)
def chat(
    req: ChatReq,
    request: Request,
    # Accept token via header name: X-Chatbot-Secret
    x_chatbot_secret: str = Header(default=""),
    # Accept token via query: ?chatbot_secret=...
    chatbot_secret: str = Query(default=""),
):
    _rate_limit(request)

    # Allow token from header OR query OR body
    token = _get_chat_token(x_chatbot_secret, chatbot_secret, req.chatbot_secret)
    _require_chat_secret(token)

    history = req.context or []
    messages: List[Dict[str, str]] = []

    # Keep last 10 messages max (avoid token bloat)
    for m in history[-10:]:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": req.message})

    reply, _dbg = call_openai_with_filesearch(messages)
    return ChatResp(reply=reply)


# ---------------- ADMIN: VECTOR STORE + UPLOAD ----------------
@app.post("/admin/create_vector_store")
def admin_create_vector_store(
    name: str = "videmark_knowledge_base",
    x_admin_secret: str = Header(default=""),
):
    _require_admin_secret(x_admin_secret)

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY nincs beállítva.")

    url = "https://api.openai.com/v1/vector_stores"
    headers = {**_openai_headers(), "Content-Type": "application/json"}

    r = requests.post(url, headers=headers, json={"name": name}, timeout=OPENAI_TIMEOUT_SEC)
    if not _ok_status(r.status_code):
        raise HTTPException(status_code=500, detail=f"Vector store create hiba: {r.status_code} – {r.text[:400]}")

    vs = r.json()
    return {
        "vector_store_id": vs.get("id"),
        "note": "Ezt másold be Render ENV-be: OPENAI_VECTOR_STORE_ID, majd redeploy.",
    }


@app.post("/admin/upload")
def admin_upload(
    file: UploadFile = File(...),
    x_admin_secret: str = Header(default=""),
):
    _require_admin_secret(x_admin_secret)

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY nincs beállítva.")
    if not OPENAI_VECTOR_STORE_ID:
        raise HTTPException(status_code=400, detail="Nincs OPENAI_VECTOR_STORE_ID. Előbb hozz létre vector store-t.")

    # 1) Upload file to OpenAI Files API
    files_url = "https://api.openai.com/v1/files"
    headers_files = _openai_headers()  # requests sets multipart content-type
    data = {"purpose": "assistants"}

    try:
        fbytes = file.file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Nem tudtam beolvasni a feltöltött fájlt.")

    files = {"file": (file.filename, fbytes, file.content_type or "application/octet-stream")}

    r1 = requests.post(files_url, headers=headers_files, data=data, files=files, timeout=60)
    if not _ok_status(r1.status_code):
        raise HTTPException(status_code=500, detail=f"File upload hiba: {r1.status_code} – {r1.text[:400]}")

    file_obj = r1.json()
    file_id = file_obj.get("id")
    if not file_id:
        raise HTTPException(status_code=500, detail="OpenAI file upload: nem kaptam file_id-t.")

    # 2) Attach file to vector store
    vs_file_url = f"https://api.openai.com/v1/vector_stores/{OPENAI_VECTOR_STORE_ID}/files"
    headers_vs = {**_openai_headers(), "Content-Type": "application/json"}

    r2 = requests.post(vs_file_url, headers=headers_vs, json={"file_id": file_id}, timeout=60)
    if not _ok_status(r2.status_code):
        raise HTTPException(status_code=500, detail=f"Vector store attach hiba: {r2.status_code} – {r2.text[:400]}")

    return {"ok": True, "uploaded_file_id": file_id, "vector_store_id": OPENAI_VECTOR_STORE_ID}


@app.get("/admin/files")
def admin_list_files(x_admin_secret: str = Header(default="")):
    _require_admin_secret(x_admin_secret)

    if not OPENAI_VECTOR_STORE_ID:
        raise HTTPException(status_code=400, detail="Nincs OPENAI_VECTOR_STORE_ID.")

    url = f"https://api.openai.com/v1/vector_stores/{OPENAI_VECTOR_STORE_ID}/files"
    headers = _openai_headers()
    r = requests.get(url, headers=headers, timeout=OPENAI_TIMEOUT_SEC)
    if not _ok_status(r.status_code):
        raise HTTPException(status_code=500, detail=f"List hiba: {r.status_code} – {r.text[:400]}")
    return r.json()
