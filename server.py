import os
import time
import json
import re
import smtplib
import base64
from email.mime.text import MIMEText
from typing import Optional, Dict, List, Any

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
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
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "https://videmark.hu,https://www.videmark.hu").split(",")
    if o.strip()
]

# --- ITT TANÍTJUK AZ AI-T (PROMPT) ---
SYSTEM_PROMPT = """
Te a Videmark weboldal profi értékesítő asszisztense vagy.

TUDÁSBÁZIS: A válaszaidhoz elsődlegesen a feltöltött fájlok tartalmát használd (file_search).
Ha a tudásbázisban nem találsz információt, mondd meg őszintén és kérj pontosítást.

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

app = FastAPI(title="Videmark Chatbot API v4.4 (KB + OCR + Admin files, robust list)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- OPENAI COMPAT LAYER ----------------
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
    raise HTTPException(500, "OpenAI beta assistants nem elérhető ebben a csomag verzióban. Frissíts: openai>=1.55.0")

def threads_api(_client: OpenAI):
    if hasattr(_client, "beta") and hasattr(_client.beta, "threads"):
        return _client.beta.threads
    raise HTTPException(500, "OpenAI beta threads nem elérhető ebben a csomag verzióban. Frissíts: openai>=1.55.0")

# ---------------- HELPERS ----------------
def obj_to_dict(x: Any) -> dict:
    """OpenAI SDK objektum -> dict (biztonságosan)."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    # pydantic v2
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            pass
    # pydantic v1
    if hasattr(x, "dict"):
        try:
            return x.dict()
        except Exception:
            pass
    # fallback
    try:
        return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"_raw": str(x)}

def pick(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

# ---------------- MODELS ----------------
class ChatReq(BaseModel):
    message: str
    session_id: str
    chatbot_secret: Optional[str] = None

class ChatResp(BaseModel):
    reply: str

# ---------------- SAJÁT HTML FORMÁZÓ ----------------
def format_to_html(text: str) -> str:
    if not text:
        return ""
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
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line[2:])
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
"""

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

    asst = assistants_api(client).create(
        name="Videmark Assistant V4.4",
        instructions=SYSTEM_PROMPT,
        model=OPENAI_MODEL,
        tools=tools,
        tool_resources=tool_resources
    )
    OPENAI_ASSISTANT_ID = asst.id
    return OPENAI_ASSISTANT_ID

def require_admin(x_admin_secret: str):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET:
        raise HTTPException(401, "Unauthorized (admin)")

def require_vs():
    if not OPENAI_VECTOR_STORE_ID:
        raise HTTPException(400, "Nincs Vector Store ID (OPENAI_VECTOR_STORE_ID)")

def ocr_image_with_openai(image_bytes: bytes, mime: str = "image/png") -> str:
    data_url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Feladat: OCR. Add vissza kizárólag a képen látható szöveget. Ne magyarázz. Ne adj hozzá semmit."},
            {"role": "user", "content": [
                {"type": "text", "text": "Írd ki pontosan a képen lévő szöveget (sorokkal együtt, ha van)."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        temperature=0
    )
    return (resp.choices[0].message.content or "").strip()

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for i in range(len(doc)):
        t = doc[i].get_text("text") or ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts).strip()

def ocr_pdf_with_openai(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        page_text = ocr_image_with_openai(img_bytes, mime="image/png")
        if page_text.strip():
            out.append(f"--- PAGE {i+1} ---\n{page_text}")
    return "\n\n".join(out).strip()

def upload_text_as_file_to_vector_store(text: str, filename: str) -> dict:
    require_vs()
    if not text.strip():
        raise HTTPException(400, "Nem sikerült szöveget kinyerni (üres OCR).")
    content = text.encode("utf-8", errors="ignore")
    f = client.files.create(file=(filename, content), purpose="assistants")
    vsf = vs_api(client).files.create(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=f.id)
    return {"file_id": f.id, "vector_store_file_id": vsf.id, "filename": filename}

# ---------------- ENDPOINTS ----------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, x_chatbot_secret: str = Header(default="")):
    token = req.chatbot_secret or x_chatbot_secret
    if CHATBOT_SECRET and token != CHATBOT_SECRET:
        raise HTTPException(401, "Unauthorized")

    assistant_id = get_or_create_assistant()
    threads = threads_api(client)

    thread_id = _thread_map.get(req.session_id)
    if not thread_id:
        thread = threads.create()
        thread_id = thread.id
        _thread_map[req.session_id] = thread_id

    threads.messages.create(thread_id=thread_id, role="user", content=req.message)
    run = threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)

    while True:
        run_status = threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
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
                    tool_outputs.append({"tool_call_id": tool_call.id, "output": output_str})
            if tool_outputs:
                threads.runs.submit_tool_outputs(thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs)
            continue
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            return ChatResp(reply="Hiba történt. Próbáld újra.")
        time.sleep(0.5)

    messages = threads.messages.list(thread_id=thread_id)
    last_msg = messages.data[0]

    reply_text = ""
    if last_msg.role == "assistant":
        raw_parts = []
        for content in last_msg.content:
            if content.type == 'text':
                raw_parts.append(content.text.value)
        reply_text = format_to_html("\n".join(raw_parts))

    return ChatResp(reply=reply_text)

# ---------------- ADMIN: UPLOAD ----------------
@app.post("/admin/upload")
def admin_upload(
    files: List[UploadFile] = File(default=[]),
    files_arr: List[UploadFile] = File(default=[], alias="files[]"),
    x_admin_secret: str = Header(default=""),
    ocr: bool = Query(default=True),
):
    require_admin(x_admin_secret)
    require_vs()

    all_files: List[UploadFile] = []
    all_files.extend(files or [])
    all_files.extend(files_arr or [])
    if not all_files:
        raise HTTPException(422, "No files provided. Expected 'files' (or legacy 'files[]').")

    results = []
    for file in all_files:
        raw = file.file.read()
        filename = file.filename or "upload.bin"
        content_type = (file.content_type or "").lower()

        if ocr and content_type.startswith("image/"):
            text = ocr_image_with_openai(raw, mime=content_type or "image/png")
            txt_name = re.sub(r'\.[a-z0-9]+$', '', filename, flags=re.I) + "_OCR.txt"
            results.append({"source": filename, "mode": "image_ocr", "uploaded": upload_text_as_file_to_vector_store(text, txt_name)})
            continue

        is_pdf = (content_type in ("application/pdf", "application/x-pdf")) or filename.lower().endswith(".pdf")
        if ocr and is_pdf:
            extracted = extract_text_from_pdf(raw)
            if len(extracted.strip()) >= 80:
                txt_name = re.sub(r'\.pdf$', '', filename, flags=re.I) + "_TEXT.txt"
                results.append({"source": filename, "mode": "pdf_text_extract", "uploaded": upload_text_as_file_to_vector_store(extracted, txt_name)})
            else:
                ocred = ocr_pdf_with_openai(raw)
                txt_name = re.sub(r'\.pdf$', '', filename, flags=re.I) + "_OCR.txt"
                results.append({"source": filename, "mode": "pdf_ocr", "uploaded": upload_text_as_file_to_vector_store(ocred, txt_name)})
            continue

        f = client.files.create(file=(filename, raw), purpose="assistants")
        vsf = vs_api(client).files.create(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=f.id)
        results.append({"source": filename, "mode": "direct", "file_id": f.id, "vector_store_file_id": vsf.id})

    return {"status": "ok", "results": results}

# ---------------- ADMIN: FILES LIST ----------------
@app.get("/admin/files")
def admin_files(x_admin_secret: str = Header(default="")):
    require_admin(x_admin_secret)
    require_vs()

    items = vs_api(client).files.list(vector_store_id=OPENAI_VECTOR_STORE_ID, limit=100)

    out = []
    for it in items.data:
        # it.id = vector_store_file_id (pl. file-...)
        vs_file_id = getattr(it, "id", None) or obj_to_dict(it).get("id")

        # Itt jön a lényeg: részletek lekérése
        file_id = None
        status = getattr(it, "status", "") or obj_to_dict(it).get("status", "")

        try:
            details = vs_api(client).files.retrieve(
                vector_store_id=OPENAI_VECTOR_STORE_ID,
                file_id=vs_file_id
            )
            dd = obj_to_dict(details)

            # több lehetséges helyről próbáljuk kinyerni
            file_id = dd.get("file_id") or dd.get("file", {}).get("id") or dd.get("file", {}).get("file_id")
            status = dd.get("status") or status

        except Exception:
            dd = {}

        fname = ""
        created_at = None
        if file_id:
            try:
                fmeta = client.files.retrieve(file_id)
                fname = getattr(fmeta, "filename", "") or ""
                created_at = getattr(fmeta, "created_at", None)
            except Exception:
                pass

        out.append({
            "vector_store_file_id": vs_file_id,
            "file_id": file_id,
            "status": status,
            "filename": fname,
            "created_at": created_at
        })

    return {"status": "ok", "files": out}
# ---------------- ADMIN: DELETE ----------------
@app.delete("/admin/files/{vector_store_file_id}")
def admin_delete_file(
    vector_store_file_id: str,
    x_admin_secret: str = Header(default=""),
    delete_underlying_file: bool = Query(default=False),
):
    require_admin(x_admin_secret)
    require_vs()

    vs_item = None
    try:
        vs_item = vs_api(client).files.retrieve(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=vector_store_file_id)
    except Exception:
        pass

    vs_api(client).files.delete(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=vector_store_file_id)

    if delete_underlying_file and vs_item is not None:
        d = obj_to_dict(vs_item)
        file_id = pick(d, ["file_id"], None)
        if not file_id:
            file_obj = pick(d, ["file"], None)
            if isinstance(file_obj, dict):
                file_id = file_obj.get("id") or file_obj.get("file_id")
            else:
                try:
                    file_id = getattr(file_obj, "id", None) or getattr(file_obj, "file_id", None)
                except Exception:
                    file_id = None

        if file_id:
            try:
                client.files.delete(file_id)
            except Exception:
                pass

    return {"status": "ok", "deleted_vector_store_file_id": vector_store_file_id}

@app.post("/admin/create_vector_store")
def create_vs(name: str = "Store", x_admin_secret: str = Header(default="")):
    require_admin(x_admin_secret)
    vs = vs_api(client).create(name=name)
    return {"id": vs.id}
@app.get("/admin/files_raw")
def admin_files_raw(x_admin_secret: str = Header(default="")):
    require_admin(x_admin_secret)
    require_vs()

    items = vs_api(client).files.list(vector_store_id=OPENAI_VECTOR_STORE_ID, limit=20)

    # nyers dump, hogy lássuk a mezőket (file_id hol van)
    raw = []
    for it in items.data:
        raw.append(obj_to_dict(it))

    return {"status": "ok", "raw": raw}
