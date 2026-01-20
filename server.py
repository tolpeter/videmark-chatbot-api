import os
import time
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, Request, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- CONFIG ----------------
PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()  # openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

# Knowledge base (OpenAI Vector Store)
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()

# Secrets
CHATBOT_SECRET = os.getenv("CHATBOT_SECRET", "").strip()  # public chat calls (from WP)
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "").strip()      # admin upload calls (server-side)

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

Ha a tudásbázis fájlokból találsz releváns infót, arra támaszkodj.
""".strip()

# Rate limit (simple in-memory)
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))  # 30 req/min/IP
_ip_hits: Dict[str, List[float]] = {}

# ---------------- APP ----------------
app = FastAPI(title="Videmark Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatReq(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[List[Dict[str, Any]]] = None  # [{role:"user/assistant", content:"..."}]

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

def _require_chat_secret(x_chatbot_secret: str):
    # If CHATBOT_SECRET is empty, we allow without secret (not recommended)
    if CHATBOT_SECRET and x_chatbot_secret != CHATBOT_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized (chat secret).")

def _require_admin_secret(x_admin_secret: str):
    if not ADMIN_SECRET:
        raise HTTPException(status_code=500, detail="ADMIN_SECRET nincs beállítva a szerveren.")
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized (admin secret).")

def _openai_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}

def _extract_output_text(data: Dict[str, Any]) -> str:
    # Most common: "output_text"
    if isinstance(data, dict) and data.get("output_text"):
        return data["output_text"]

    # Fallback parse
    try:
        out = data.get("output", [])
        if out and out[0].get("content"):
            for part in out[0]["content"]:
                if isinstance(part, dict) and "text" in part and part["text"]:
                    return part["text"]
    except Exception:
        pass

    return "Nem tudtam választ generálni (OpenAI válasz formátum)."

def call_openai_with_filesearch(messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        return "Hiányzik az OPENAI_API_KEY a szerveren."

    url = "https://api.openai.com/v1/responses"
    headers = {**_openai_headers(), "Content-Type": "application/json"}

    input_msgs = [{"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]}]
    for m in messages:
        input_msgs.append({"role": m["role"], "content": [{"type": "input_text", "text": m["content"]}]})

    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": input_msgs,
        "temperature": 0.4,
        "max_output_tokens": 650,
    }

    # Enable file_search if vector store is configured
    if OPENAI_VECTOR_STORE_ID:
        payload["tools"] = [{
            "type": "file_search",
            "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
        }]  # docs: file_search with vector_store_ids :contentReference[oaicite:2]{index=2}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=45)
    except Exception as e:
        return f"Hiba az OpenAI hívásnál (kapcsolat): {str(e)[:200]}"

    if r.status_code != 200:
        return f"Hiba az OpenAI hívásnál: {r.status_code} – {r.text[:400]}"

    return _extract_output_text(r.json())

# ---------------- PUBLIC CHAT ----------------
@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, request: Request, x_chatbot_secret: str = Header(default="")):
    _rate_limit(request)
    _require_chat_secret(x_chatbot_secret)

    history = req.context or []
    messages: List[Dict[str, str]] = []

    for m in history[-10:]:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": req.message})

    reply = call_openai_with_filesearch(messages)
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

    r = requests.post(url, headers=headers, json={"name": name}, timeout=45)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Vector store create hiba: {r.status_code} – {r.text[:400]}")

    vs = r.json()
    return {
        "vector_store_id": vs.get("id"),
        "note": "Ezt másold be Render ENV-be: OPENAI_VECTOR_STORE_ID",
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
    fbytes = file.file.read()
    files = {"file": (file.filename, fbytes, file.content_type or "application/octet-stream")}

    r1 = requests.post(files_url, headers=headers_files, data=data, files=files, timeout=60)
    if r1.status_code != 200:
        raise HTTPException(status_code=500, detail=f"File upload hiba: {r1.status_code} – {r1.text[:400]}")
    file_obj = r1.json()
    file_id = file_obj.get("id")

    # 2) Attach file to vector store
    vs_file_url = f"https://api.openai.com/v1/vector_stores/{OPENAI_VECTOR_STORE_ID}/files"
    headers_vs = {**_openai_headers(), "Content-Type": "application/json"}
    r2 = requests.post(vs_file_url, headers=headers_vs, json={"file_id": file_id}, timeout=60)
    if r2.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Vector store attach hiba: {r2.status_code} – {r2.text[:400]}")

    return {"ok": True, "uploaded_file_id": file_id, "vector_store_id": OPENAI_VECTOR_STORE_ID}

@app.get("/admin/files")
def admin_list_files(x_admin_secret: str = Header(default="")):
    _require_admin_secret(x_admin_secret)
    if not OPENAI_VECTOR_STORE_ID:
        raise HTTPException(status_code=400, detail="Nincs OPENAI_VECTOR_STORE_ID.")

    url = f"https://api.openai.com/v1/vector_stores/{OPENAI_VECTOR_STORE_ID}/files"
    headers = _openai_headers()
    r = requests.get(url, headers=headers, timeout=45)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"List hiba: {r.status_code} – {r.text[:400]}")
    return r.json()

