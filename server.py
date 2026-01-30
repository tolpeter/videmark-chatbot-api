import os
import time
import json
import re
import smtplib
import base64
import html
from email.mime.text import MIMEText
from typing import Optional, Dict, List, Any

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import requests


# ---------------- CONFIG ----------------
PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()  # "openai" (most ezt használjuk)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID", "").strip()
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

CHATBOT_SECRET = os.getenv("CHATBOT_SECRET", "").strip()
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "").strip()

# WP lead webhook (WordPress plugin REST endpoint)
WP_LEAD_WEBHOOK_URL = os.getenv("WP_LEAD_WEBHOOK_URL", "https://videmark.hu/wp-json/vmkb/v1/lead").strip()

# Email beállítások
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
EMAIL_TO = os.getenv("EMAIL_TO", "").strip()  # opcionális: ha üres, akkor SMTP_USER-re megy

# Korlátok / lista lapozás
MAX_ADMIN_LIST_LIMIT = 10

# ---------------- GLOBALS ----------------
client = OpenAI(api_key=OPENAI_API_KEY)

# session -> thread
_thread_map: Dict[str, str] = {}

# ha lead pending, akkor ne hívja többször
_lead_pending: Dict[str, bool] = {}

app = FastAPI(title="Videmark Chatbot API v4.7 (+ WP lead webhook)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- OPENAI COMPAT ----------------
def vs_api(_client: OpenAI):
    if hasattr(_client, "beta") and hasattr(_client.beta, "vector_stores"):
        return _client.beta.vector_stores
    if hasattr(_client, "vector_stores"):
        return _client.vector_stores
    raise HTTPException(
        500,
        "OpenAI python csomag túl régi: nincs vector_stores. Frissítsd: openai>=1.55.0 és Renderen 'Clear build cache & deploy'."
    )


def assistants_api(_client: OpenAI):
    if hasattr(_client, "beta") and hasattr(_client.beta, "assistants"):
        return _client.beta.assistants
    raise HTTPException(500, "OpenAI beta assistants nem elérhető. Frissíts: openai>=1.55.0")


def threads_api(_client: OpenAI):
    if hasattr(_client, "beta") and hasattr(_client.beta, "threads"):
        return _client.beta.threads
    raise HTTPException(500, "OpenAI beta threads nem elérhető. Frissíts: openai>=1.55.0")


vector_stores = vs_api(client)
assistants = assistants_api(client)
threads = threads_api(client)


# ---------------- LOGGING ----------------
def log_event(data: dict):
    # egyszerű stdout log (Render logba megy)
    try:
        print(json.dumps({"ts": int(time.time()), **data}, ensure_ascii=False))
    except Exception:
        print({"ts": int(time.time()), **data})


# ---------------- ADMIN AUTH ----------------
def require_admin(secret: str):
    if not ADMIN_SECRET:
        raise HTTPException(500, "ADMIN_SECRET nincs beállítva a szerveren.")
    if not secret or secret != ADMIN_SECRET:
        raise HTTPException(401, "Unauthorized (bad admin secret).")


# ---------------- UTILS ----------------
def safe_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w.\-() ]+", "_", name, flags=re.U)
    return name[:180] if name else "file"


def pdf_to_text_bytes(upload: UploadFile) -> bytes:
    # PDF -> text (PyMuPDF)
    data = upload.file.read()
    upload.file.seek(0)
    doc = fitz.open(stream=data, filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    content = "\n".join(text).strip()
    if not content:
        content = "[PDF: nem találtam kinyerhető szöveget]"
    return content.encode("utf-8")


def file_to_text_bytes(upload: UploadFile) -> bytes:
    # Egyszerű: ha PDF, textet szedünk ki; ha nem, akkor raw bytes megy fel
    fn = (upload.filename or "").lower()
    if fn.endswith(".pdf"):
        return pdf_to_text_bytes(upload)
    return upload.file.read()


# ---------------- VECTOR STORE HELPERS ----------------
def ensure_vector_store() -> str:
    global OPENAI_VECTOR_STORE_ID
    if OPENAI_VECTOR_STORE_ID:
        return OPENAI_VECTOR_STORE_ID

    vs = vector_stores.create(name="Videmark Knowledge Base")
    OPENAI_VECTOR_STORE_ID = vs.id
    log_event({"type": "vector_store_created", "vector_store_id": OPENAI_VECTOR_STORE_ID})
    return OPENAI_VECTOR_STORE_ID


def list_vector_store_files_paged(vs_id: str, limit: int, offset: int) -> dict:
    """
    OpenAI vector store file listing: limit<=10 és after-cursor.
    Mi itt offsettel szimuláljuk, de belül after-rel lapozunk.
    """
    limit = max(1, min(MAX_ADMIN_LIST_LIMIT, int(limit)))
    offset = max(0, int(offset))

    # végiglapozunk offset+limit elemet (kicsi limit miatt OK)
    need = offset + limit
    collected = []
    after = None

    while len(collected) < need:
        page = vector_stores.files.list(vector_store_id=vs_id, limit=limit, after=after)
        items = getattr(page, "data", []) or []
        for it in items:
            collected.append(it)
            if len(collected) >= need:
                break

        after = getattr(page, "last_id", None)
        has_more = getattr(page, "has_more", None)
        if has_more is False:
            break
        if not after:
            break

    sliced = collected[offset:offset + limit]
    return {
        "items": [
            {
                "file_id": getattr(x, "id", None),
                "status": getattr(x, "status", None),
                "created_at": getattr(x, "created_at", None),
                "vector_store_id": vs_id
            }
            for x in sliced
        ],
        "limit": limit,
        "offset": offset,
        "returned": len(sliced)
    }


# ---------------- SOURCE/CITATION STRIPPER ----------------
_SOURCE_PATTERNS = [
    r"\[\d+(?::\d+)?†source\]",   # [4†source] vagy [4:0†source]
    r"【\d+†source】",             # 
    r"\[\s*source\s*\]",          # [source]
]

def strip_sources(text: str) -> str:
    """
    Kiveszi a válaszok végéről/közepéről a 'source' jelöléseket, pl:
    [4:0†source], [1†source],  stb.
    """
    if not text:
        return ""
    out = text
    for pat in _SOURCE_PATTERNS:
        out = re.sub(pat, "", out, flags=re.I)

    # duplaszóközök, fura szóköz írásjelek előtt
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([.,!?:;])", r"\1", out)
    return out.strip()


# ---------------- HTML FORMATTER ----------------
def format_to_html(text: str) -> str:
    """
    Egyszerű, biztonságosabb HTML formázó:
    - listák: - vagy • kezdetű sorok -> <ul><li>
    - üres sor -> <br>
    - **félkövér** -> <strong>félkövér</strong>
    """
    if not text:
        return ""

    text = re.sub(r"\r\n|\r", "\n", text)

    lines = text.split("\n")
    html_lines = []
    in_list = False

    def apply_bold(escaped: str) -> str:
        # **...** -> <strong>...</strong>
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)

    for line in lines:
        raw = line.strip()

        if re.match(r"^[-•]\s+", raw):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            item = re.sub(r"^[-•]\s+", "", raw)
            item_esc = html.escape(item)
            item_esc = apply_bold(item_esc)
            html_lines.append(f"<li>{item_esc}</li>")
            continue

        if in_list:
            html_lines.append("</ul>")
            in_list = False

        if raw == "":
            html_lines.append("<br>")
        else:
            safe = html.escape(raw)
            safe = apply_bold(safe)
            html_lines.append(safe)

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


# ---------------- EMAIL / LEAD ----------------
def send_email_notification(lead_data: dict):
    if not SMTP_USER or not SMTP_PASSWORD:
        print("⚠️ Nincs SMTP beállítva.")
        return

    subject = f"🔥 ÚJ LEAD: {lead_data.get('name', 'Ismeretlen')}"
    body = f"""
Új érdeklődő érkezett!

Név: {lead_data.get('name')}
Email: {lead_data.get('email')}
Telefon: {lead_data.get('phone')}
Leírás: {lead_data.get('description')}

Session: {lead_data.get('session_id')}
Forrás: {lead_data.get('source')}
"""

    to_email = EMAIL_TO if EMAIL_TO else SMTP_USER

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = to_email

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [to_email], msg.as_string())
        server.quit()
        print("✅ Email elküldve.")
    except Exception as e:
        print(f"❌ Email hiba: {e}")


def send_lead_to_wp(lead_data: dict):
    """
    Elküldi a leadet a WordPress plugin webhookjára:
    POST {WP_LEAD_WEBHOOK_URL}
    Header: X-Admin-Secret: <ADMIN_SECRET>
    Body: JSON
    """
    if not WP_LEAD_WEBHOOK_URL:
        print("⚠️ WP_LEAD_WEBHOOK_URL nincs beállítva, WP lead küldés kihagyva.")
        return

    if not ADMIN_SECRET:
        print("⚠️ ADMIN_SECRET nincs beállítva, WP lead küldés kihagyva.")
        return

    try:
        headers = {
            "Content-Type": "application/json",
            "X-Admin-Secret": ADMIN_SECRET,
        }
        payload = {
            "source": lead_data.get("source", "chatbot"),
            "session_id": lead_data.get("session_id", ""),
            "name": lead_data.get("name", ""),
            "email": lead_data.get("email", ""),
            "phone": lead_data.get("phone", ""),
            "message": lead_data.get("description", lead_data.get("message", "")),
        }
        r = requests.post(WP_LEAD_WEBHOOK_URL, json=payload, headers=headers, timeout=20)
        if 200 <= r.status_code < 300:
            print("✅ Lead elküldve WordPressnek.")
            return
        print(f"❌ WP lead küldés hiba: HTTP {r.status_code} | {r.text[:500]}")
    except Exception as e:
        print(f"❌ WP lead küldés kivétel: {e}")


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")


def extract_lead_from_text(text: str) -> Optional[dict]:
    if not text:
        return None

    m = EMAIL_RE.search(text)
    if not m:
        return None
    email = m.group(0).strip()

    phone = ""
    pm = PHONE_RE.search(text)
    if pm:
        phone = re.sub(r"\s+", " ", pm.group(0)).strip()

    name = ""
    nm = re.search(r"(?:^|\n)\s*(?:név|nev)\s*:\s*(.+)", text, re.I)
    if nm:
        name = nm.group(1).strip().split("\n")[0].strip()

    if not name:
        first_line = (text.strip().splitlines()[0] if text.strip() else "").strip()
        if first_line and not EMAIL_RE.search(first_line) and not PHONE_RE.search(first_line):
            parts = first_line.split()
            if len(parts) <= 4:
                name = first_line

    description = text.strip()
    return {"name": name, "email": email, "phone": phone, "description": description}


# ---------------- ASSISTANT ----------------
def get_or_create_assistant():
    global OPENAI_ASSISTANT_ID
    if OPENAI_ASSISTANT_ID:
        return OPENAI_ASSISTANT_ID

    tools = [
        {"type": "file_search"},
        {
            "type": "function",
            "function": {
                "name": "save_lead",
                "description": "Mentse el az érdeklődő adatait, és küldjön értesítést a Videmarknak.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["email", "description"],
                },
            },
        },
    ]

    vs_id = ensure_vector_store()

    a = assistants.create(
        name="Videmark Asszisztens",
        instructions=(
            "Te a Videmark Asszisztens vagy. "
            "Mindig a tudásbázis alapján válaszolj. "
            "Ha árat kérnek és nincs a tudásbázisban, ne találj ki. "
            "Ha a felhasználó érdeklődő és megadja a nevét/email/telefont és rövid leírást, "
            "kötelező meghívni a save_lead funkciót.\n\n"
            "A lead adatai:\n"
            "- Név\n"
            "- Email cím\n"
            "- Telefonszám\n"
            "- Rövid leírás\n\n"
            "Formázás: ha felsorolás van, használj kötőjeleket és sortörést. "
            "Ne írj a válasz végére semmilyen forrásjelölést (pl. [1†source], [4:0†source], )."
        ),
        model=OPENAI_MODEL,
        tools=tools,
        tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
    )

    OPENAI_ASSISTANT_ID = a.id
    log_event({"type": "assistant_created", "assistant_id": OPENAI_ASSISTANT_ID, "vector_store_id": vs_id})
    return OPENAI_ASSISTANT_ID


def get_or_create_thread(session_id: str) -> str:
    if session_id in _thread_map:
        return _thread_map[session_id]
    t = threads.create()
    _thread_map[session_id] = t.id
    log_event({"type": "thread_created", "session_id": session_id, "thread_id": t.id})
    return t.id


# ---------------- REQUEST/RESPONSE MODELS ----------------
class ChatReq(BaseModel):
    session_id: str
    message: str


class ChatResp(BaseModel):
    reply: str


# ---------------- CHAT ENDPOINT ----------------
@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, x_chatbot_secret: Optional[str] = Header(default=None)):
    if CHATBOT_SECRET and x_chatbot_secret != CHATBOT_SECRET:
        raise HTTPException(401, "Unauthorized (bad chatbot secret).")

    assistant_id = get_or_create_assistant()
    thread_id = get_or_create_thread(req.session_id)

    # user message
    threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=req.message,
    )

    # lead duplikáció védelem
    if req.session_id not in _lead_pending:
        _lead_pending[req.session_id] = False

    run = threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)

    # poll
    while True:
        run_status = threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

        if run_status.status == "completed":
            break

        if run_status.status == "requires_action":
            tool_outputs = []
            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                if tool_call.function.name == "save_lead":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        # kiegészítés: session és forrás
                        args.setdefault('session_id', req.session_id)
                        args.setdefault('source', 'chatbot')

                        # 1) email
                        send_email_notification(args)
                        # 2) WP webhook
                        send_lead_to_wp(args)

                        output_str = '{"success": true, "message": "Lead rögzítve és értesítés elküldve."}'
                        _lead_pending[req.session_id] = False
                        log_event({"type": "lead", "session_id": req.session_id, "lead": args})
                    except Exception as e:
                        output_str = '{"success": false}'
                        log_event({"type": "lead_error", "session_id": req.session_id, "error": str(e)})
                    tool_outputs.append({"tool_call_id": tool_call.id, "output": output_str})

            if tool_outputs:
                threads.runs.submit_tool_outputs(thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs)
            continue

        if run_status.status in ["failed", "cancelled", "expired"]:
            reply = "Hiba történt. Próbáld újra."
            log_event({"type": "assistant", "session_id": req.session_id, "text": reply, "error": run_status.status})
            return ChatResp(reply=format_to_html(reply))

        time.sleep(0.4)

    # get last assistant message
    msgs = threads.messages.list(thread_id=thread_id, limit=10)
    reply_text = ""
    for m in getattr(msgs, "data", []) or []:
        if getattr(m, "role", "") == "assistant":
            parts = getattr(m, "content", []) or []
            if parts and hasattr(parts[0], "text") and hasattr(parts[0].text, "value"):
                reply_text = parts[0].text.value
            break

    if not reply_text:
        reply_text = "Nem kaptam választ. Próbáld újra."

    # ✅ ITT TŰNIK EL A [4:0†source] stb.
    reply_text = strip_sources(reply_text)

    log_event({"type": "assistant", "session_id": req.session_id, "text": reply_text})
    return ChatResp(reply=format_to_html(reply_text))


# ---------------- ADMIN: UPLOAD ----------------
@app.post("/admin/upload")
async def admin_upload(
    file: UploadFile = File(...),
    x_admin_secret: Optional[str] = Header(default=None)
):
    require_admin(x_admin_secret)

    vs_id = ensure_vector_store()
    filename = safe_filename(file.filename)

    # fájl bytes
    content_bytes = file_to_text_bytes(file)
    if not content_bytes:
        raise HTTPException(400, "Empty file")

    # OpenAI upload (assistants file)
    uploaded = client.files.create(
        file=(filename, content_bytes),
        purpose="assistants"
    )

    # attach to vector store
    vector_stores.files.create(
        vector_store_id=vs_id,
        file_id=uploaded.id
    )

    log_event({"type": "admin_upload", "filename": filename, "file_id": uploaded.id, "vector_store_id": vs_id})

    return {"ok": True, "file_id": uploaded.id, "filename": filename, "vector_store_id": vs_id}


# ---------------- ADMIN: LIST FILES ----------------
@app.get("/admin/files")
def admin_files(
    limit: int = Query(default=10, ge=1, le=10),
    offset: int = Query(default=0, ge=0),
    x_admin_secret: Optional[str] = Header(default=None)
):
    require_admin(x_admin_secret)
    vs_id = ensure_vector_store()

    data = list_vector_store_files_paged(vs_id, limit=limit, offset=offset)
    log_event({"type": "admin_files", "limit": limit, "offset": offset, "returned": data.get("returned")})
    return data


# ---------------- ADMIN: DELETE FILE ----------------
class DeleteReq(BaseModel):
    file_id: str


@app.post("/admin/delete")
def admin_delete(req: DeleteReq, x_admin_secret: Optional[str] = Header(default=None)):
    require_admin(x_admin_secret)
    vs_id = ensure_vector_store()

    # remove from vector store
    try:
        vector_stores.files.delete(vector_store_id=vs_id, file_id=req.file_id)
    except Exception:
        # ha már nincs, próbáljuk tovább
        pass

    # delete file from OpenAI storage
    try:
        client.files.delete(req.file_id)
    except Exception:
        pass

    log_event({"type": "admin_delete", "file_id": req.file_id, "vector_store_id": vs_id})
    return {"ok": True, "file_id": req.file_id}
