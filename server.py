import os
import time
import json
import re
import smtplib
import base64
from email.mime.text import MIMEText
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# OCR/vision modell (külön választható)
OPENAI_OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini").strip()

OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID", "").strip()
CHATBOT_SECRET = os.getenv("CHATBOT_SECRET", "").strip()
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "").strip()

# Email beállítások
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "info@videmark.hu")

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://videmark.hu,https://www.videmark.hu").split(",") if o.strip()
]

# --- ITT TANÍTJUK AZ AI-T (PROMPT) ---
SYSTEM_PROMPT = """
Te a Videmark weboldal profi értékesítő asszisztense vagy.

TUDÁSBÁZIS: Használd a feltöltött fájlokat a válaszadáshoz.

FONTOS VISELKEDÉSI SZABÁLYOK:
1. PONTOSÍTÁS (Nagyon fontos!):
   - Ha a felhasználó csak annyit kérdez: "Mennyibe kerül egy videó?" vagy "Milyen árak vannak?", NE sorold fel azonnal az összes árat!
   - Ehelyett kérdezz vissza: "Szívesen segítek! Milyen típusú videóra gondoltál? (pl. Drón felvétel, Reklámvideó, Rendezvény videózás vagy Social Média tartalom?)"
   - Csak akkor mondj konkrét árat, ha tudod, mit akar.

2. LEAD GYŰJTÉS:
   - Ha az ügyfél konkrét árajánlatot kér vagy komolyan érdeklődik, kérd el az adatait: Név, Email, Telefonszám, Rövid leírás.
   - Ha megkaptad, hívd meg a 'save_lead' funkciót.

3. FORMÁZÁS (Hogy szép legyen):
   - A fontos szavakat, árakat mindig emeld ki így: **ár**.
   - Felsorolásnál használj kötőjelet:
     - Tétel 1
     - Tétel 2
   - Használj címsorokat: ### Címsor

Stílus: Magyar, közvetlen, segítőkész, rövid és lényegretörő.
""".strip()

client = OpenAI(api_key=OPENAI_API_KEY)
_thread_map: Dict[str, str] = {}

app = FastAPI(title="Videmark Chatbot API v4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatReq(BaseModel):
    message: str
    session_id: str
    chatbot_secret: Optional[str] = None

class ChatResp(BaseModel):
    reply: str

# ---------------- SAJÁT HTML FORMÁZÓ (NEM KELL KÜLSŐ FÁJL) ----------------
def format_to_html(text: str) -> str:
    """Átalakítja a Markdown jeleket szép HTML kódra a szerveren."""
    if not text:
        return ""

    # 1. Hivatkozások tisztítása
    text = re.sub(r'【.*?】', '', text)

    lines = text.split('\n')
    html_lines = []
    in_list = False

    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue

        if line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html_lines.append('<ul style="margin: 5px 0 10px 20px; padding: 0;">')
                in_list = True
            content = line[2:]
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
            html_lines.append(f'<li style="margin-bottom: 5px; list-style: disc;">{content}</li>')

        elif line.startswith("###"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            content = line.replace("###", "").strip()
            html_lines.append(
                f'<h3 style="margin: 15px 0 5px 0; font-size: 16px; border-bottom: 1px solid rgba(255,255,255,0.2);">{content}</h3>'
            )

        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            html_lines.append(f'<p style="margin: 0 0 8px 0;">{line}</p>')

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)

# ---------------- FUNKCIÓK ----------------
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

Dátum: {time.strftime('%Y-%m-%d %H:%M:%S')}
""".strip()

    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = NOTIFY_EMAIL

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email elküldve.")
    except Exception as e:
        print(f"❌ Email hiba: {e}")

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
                "description": "Mentse el az érdeklődő adatait.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "email"]
                }
            }
        }
    ]

    tool_resources = {}
    if OPENAI_VECTOR_STORE_ID:
        tool_resources = {"file_search": {"vector_store_ids": [OPENAI_VECTOR_STORE_ID]}}

    asst = client.beta.assistants.create(
        name="Videmark Assistant V4",
        instructions=SYSTEM_PROMPT,
        model=OPENAI_MODEL,
        tools=tools,
        tool_resources=tool_resources
    )
    OPENAI_ASSISTANT_ID = asst.id
    return OPENAI_ASSISTANT_ID

def _require_admin(x_admin_secret: str):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET:
        raise HTTPException(401, "Unauthorized")
    if not OPENAI_VECTOR_STORE_ID:
        raise HTTPException(400, "Nincs Vector Store ID (OPENAI_VECTOR_STORE_ID).")

def _upload_bytes_to_vector_store(filename: str, data: bytes):
    """Feltölt bytes-t OpenAI Files-ba, majd csatolja a Vector Store-hoz."""
    f = client.files.create(file=(filename, data), purpose="assistants")
    client.beta.vector_stores.files.create(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=f.id)
    return f.id

def _ocr_image_to_text_bytes(filename: str, image_bytes: bytes) -> bytes:
    """Képből szöveg kinyerése OpenAI vision segítségével, majd txt bytes."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        "Olvasd ki a képen található ÖSSZES szöveget pontosan. "
        "Tartsd meg a bekezdéseket és felsorolásokat, ha vannak. "
        "Ne magyarázz, csak a kinyert szöveget add vissza."
    )

    # Responses API (vision)
    resp = client.responses.create(
        model=OPENAI_OCR_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_base64": b64},
            ],
        }],
    )

    # Összefűzzük a text outputot
    out_text_parts = []
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    out_text_parts.append(c.text)
    text = ("\n".join(out_text_parts)).strip()

    if not text:
        text = "(Nem sikerült szöveget kinyerni a képből.)"

    # Adjunk egy kis fejléces kontextust
    final = f"FORRÁS KÉP: {filename}\n\n{text}\n"
    return final.encode("utf-8")

def _extract_pdf_text_bytes(filename: str, pdf_bytes: bytes) -> Optional[bytes]:
    """
    PDF-ből szöveg kinyerése (ha van beágyazott text).
    Szkennelt PDF esetén ez gyakran üres – olyankor None-t adunk.
    """
    try:
        from pypdf import PdfReader  # pip: pypdf
    except Exception:
        return None

    try:
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            t = t.strip()
            if t:
                parts.append(f"--- OLDAL {i+1} ---\n{t}\n")
        text = "\n".join(parts).strip()
        if len(text) < 200:
            return None
        final = f"FORRÁS PDF: {filename}\n\n{text}\n"
        return final.encode("utf-8")
    except Exception:
        return None

# ---------------- ENDPOINTS ----------------
@app.get("/")
def root():
    return {"status": "ok", "mode": "HTML server-side rendering v4.1"}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, x_chatbot_secret: str = Header(default="")):
    token = req.chatbot_secret or x_chatbot_secret
    if CHATBOT_SECRET and token != CHATBOT_SECRET:
        raise HTTPException(401, "Unauthorized")

    assistant_id = get_or_create_assistant()

    thread_id = _thread_map.get(req.session_id)
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id
        _thread_map[req.session_id] = thread_id

    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=req.message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant_id
    )

    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run_status.status == 'completed':
            break
        elif run_status.status == 'requires_action':
            tool_outputs = []
            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                if tool_call.function.name == "save_lead":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        send_email_notification(args)
                        output_str = '{"success": true, "message": "Email elküldve."}'
                    except Exception:
                        output_str = '{"success": false}'

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": output_str
                    })
            if tool_outputs:
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
                )
            continue
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            return ChatResp(reply="Hiba történt. Próbáld újra.")
        time.sleep(0.5)

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_msg = messages.data[0]

    reply_text = ""
    if last_msg.role == "assistant":
        raw_parts = []
        for content in last_msg.content:
            if content.type == 'text':
                raw_parts.append(content.text.value)

        raw_text = "\n".join(raw_parts)
        reply_text = format_to_html(raw_text)

    return ChatResp(reply=reply_text)

# ----------- ADMIN: upload (bővített: képek OCR + PDF text companion) -----------
@app.post("/admin/upload")
def admin_upload(file: UploadFile = File(...), x_admin_secret: str = Header(default="")):
    _require_admin(x_admin_secret)

    filename = file.filename or "upload.bin"
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Üres fájl.")

    content_type = (file.content_type or "").lower()

    # 1) Kép: OCR -> txt feltöltés
    if content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        txt_bytes = _ocr_image_to_text_bytes(filename, data)
        txt_name = re.sub(r'\.[a-z0-9]+$', '', filename, flags=re.I) + ".txt"
        new_id = _upload_bytes_to_vector_store(txt_name, txt_bytes)
        return {"status": "ok", "mode": "image_ocr_to_txt", "file_id": new_id, "stored_as": txt_name}

    # 2) PDF: feltöltjük a PDF-et + ha van kinyerhető szöveg, feltöltünk PDFname.txt-t is
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        pdf_id = _upload_bytes_to_vector_store(filename, data)

        pdf_txt = _extract_pdf_text_bytes(filename, data)
        txt_id = None
        stored_txt = None
        if pdf_txt:
            stored_txt = re.sub(r'\.pdf$', '', filename, flags=re.I) + ".txt"
            txt_id = _upload_bytes_to_vector_store(stored_txt, pdf_txt)

        return {
            "status": "ok",
            "mode": "pdf_plus_optional_txt",
            "pdf_file_id": pdf_id,
            "txt_file_id": txt_id,
            "txt_stored_as": stored_txt
        }

    # 3) Egyéb (doc/docx/txt/md/html…): sima feltöltés
    new_id = _upload_bytes_to_vector_store(filename, data)
    return {"status": "ok", "mode": "raw_upload", "file_id": new_id, "stored_as": filename}

# ----------- ADMIN: list files -----------
@app.get("/admin/files")
def admin_files(x_admin_secret: str = Header(default=""), limit: int = 100):
    _require_admin(x_admin_secret)

    # Vector store file entries
    vs_files = client.beta.vector_stores.files.list(vector_store_id=OPENAI_VECTOR_STORE_ID, limit=limit)

    items = []
    for it in vs_files.data:
        file_id = it.id  # vector store file id OR file id? (OpenAI lib: vs file item id is file_id)
        # Biztos ami biztos: próbáljuk fájl meta lekéréssel
        try:
            fmeta = client.files.retrieve(file_id)
            items.append({
                "file_id": file_id,
                "filename": getattr(fmeta, "filename", None),
                "bytes": getattr(fmeta, "bytes", None),
                "created_at": getattr(fmeta, "created_at", None),
            })
        except Exception:
            items.append({
                "file_id": file_id,
                "filename": None,
                "bytes": None,
                "created_at": None,
            })

    # Legfrissebb elöl
    items.sort(key=lambda x: (x["created_at"] or 0), reverse=True)
    return {"status": "ok", "count": len(items), "items": items}

# ----------- ADMIN: delete file -----------
@app.post("/admin/delete")
def admin_delete(file_id: str = Query(...), x_admin_secret: str = Header(default="")):
    _require_admin(x_admin_secret)

    # 1) leválasztás a vector store-ról (ha támogatott)
    try:
        client.beta.vector_stores.files.delete(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=file_id)
    except Exception:
        # Ha a SDK itt máshogy kezeli, attól még próbáljuk a file delete-et.
        pass

    # 2) törlés OpenAI Files-ból
    try:
        client.files.delete(file_id)
    except Exception as e:
        raise HTTPException(400, f"Nem sikerült törölni: {e}")

    return {"status": "ok", "deleted_file_id": file_id}

@app.post("/admin/create_vector_store")
def create_vs(name: str = "Store", x_admin_secret: str = Header(default="")):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET:
        raise HTTPException(401)
    vs = client.beta.vector_stores.create(name=name)
    return {"id": vs.id}
