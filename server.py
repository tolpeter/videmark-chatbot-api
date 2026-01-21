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

# (opcionális) beszélgetés napló fájl (Renderen csak akkor marad meg, ha van persistent disk)
LOG_ENABLED = os.getenv("LOG_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
LOG_PATH = os.getenv("LOG_PATH", "./chat_logs.jsonl").strip()  # javaslat: /var/data/chat_logs.jsonl ha van disk

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "https://videmark.hu,https://www.videmark.hu").split(",")
    if o.strip()
]

# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """
Te a Videmark weboldal profi értékesítő asszisztense vagy.

TUDÁSBÁZIS: A válaszaidhoz elsődlegesen a feltöltött fájlok tartalmát használd (file_search).
Ha a kérdésre van konkrét válasz a tudásbázisban (pl. szolgáltatás + ár), akkor AZONNAL azt add meg.

FONTOS VISELKEDÉSI SZABÁLYOK:

0. TUDÁSBÁZIS ELSŐDLEGESSÉGE:
- A válaszaidat mindig a tudásbázis alapján add.
- NE írj magyarázó mondatot arról, hogy “a feltöltött fájlok alapján” válaszolsz – csak válaszolj.

1. PONTOSÍTÁS:
- Ha a felhasználó általánosan kérdez (pl. „Mennyibe kerül egy videó?” / „Milyen árak vannak?”),
  akkor NE sorold fel az összes árat automatikusan.
- Kérdezz vissza röviden, hogy milyen típus érdekli (csak olyan példákat adj, amik a tudásbázisban szerepelnek).

2. NINCS TALÁLGATÁS (Kritikus!):
- SOHA ne említs olyan szolgáltatást / fotózási típust / videós típust vagy árat, ami NEM szerepel a tudásbázisban.
- TILOS példaként felsorolni olyan opciókat (pl. esküvői fotózás), amelyek nem találhatók meg a tudásbázisban.
- TILOS becsült, “kb.”, “általában ennyi”, “tól-ig” jellegű árat adni.

3. HA HIÁNYZIK AZ INFORMÁCIÓ → LEAD (Kötelező):
- Ha a kérdésre nincs konkrét válasz a tudásbázisban, NE találj ki árat vagy szolgáltatást.
- Ilyenkor tereld LEAD irányba, pontosan így:
  "A megadott szolgáltatás nem szerepel a rendelkezésre álló anyagokban.
   A pontos árral kapcsolatban kérlek, add meg az alábbi adatokat,
   és felvesszük veled a kapcsolatot:"
  - Név
  - Email cím
  - Telefonszám
  - Rövid leírás a projektről

4. LEAD MENTÉS + ÉRTESÍTÉS:
- Ha a felhasználó megadja az adatokat, KÖTELEZŐ meghívni a `save_lead` funkciót.

5. FORMÁZÁS:
- A fontos szavakat, árakat mindig emeld ki így: **ár**.
- Felsorolásnál használj kötőjelet:
  - Tétel 1
  - Tétel 2
- Használj címsorokat: ### Címsor

Stílus: Magyar, közvetlen, segítőkész, rövid és lényegretörő.
""".strip()

client = OpenAI(api_key=OPENAI_API_KEY)

# session -> thread
_thread_map: Dict[str, str] = {}

# ha leadet kértünk, jelöljük
_lead_pending: Dict[str, bool] = {}

app = FastAPI(title="Videmark Chatbot API v4.7 (limit<=10 fixed + paged listing + logs)")

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

# ---------------- HELPERS ----------------
def obj_to_dict(x: Any) -> dict:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return x.dict()
        except Exception:
            pass
    try:
        return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"_raw": str(x)}

def safe_get_file_meta(file_id: str) -> dict:
    if not file_id:
        return {"filename": "", "created_at": None}
    try:
        fmeta = client.files.retrieve(file_id)
        return {
            "filename": getattr(fmeta, "filename", "") or "",
            "created_at": getattr(fmeta, "created_at", None),
        }
    except Exception:
        return {"filename": "", "created_at": None}

def require_admin(x_admin_secret: str):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET:
        raise HTTPException(401, "Unauthorized (admin)")

def require_vs():
    if not OPENAI_VECTOR_STORE_ID:
        raise HTTPException(400, "Nincs Vector Store ID (OPENAI_VECTOR_STORE_ID)")

def is_single_keyword(msg: str) -> bool:
    s = (msg or "").strip()
    if not s:
        return False
    cleaned = re.sub(r"[^\w\sáéíóöőúüűÁÉÍÓÖŐÚÜŰ-]", " ", s).strip()
    words = [w for w in cleaned.split() if w]
    return 1 <= len(words) <= 2

def log_event(event: dict):
    if not LOG_ENABLED:
        return
    try:
        event["ts"] = int(time.time())
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass

# --- FIX: OpenAI vector store list limit MAX 10 -> lapozunk ---
VS_LIST_PAGE_LIMIT = 10  # az error alapján <=10

def _vs_files_list_paged(max_items: int = 200) -> List[Any]:
    """
    Biztonságos listázás: a list() limitje max 10, ezért 'after' paraméterrel lapozunk.
    """
    require_vs()

    collected: List[Any] = []
    after: Optional[str] = None

    # hard safety
    if max_items < 1:
        return []

    while len(collected) < max_items:
        remaining = max_items - len(collected)
        page_limit = min(VS_LIST_PAGE_LIMIT, remaining)

        kwargs = {"vector_store_id": OPENAI_VECTOR_STORE_ID, "limit": page_limit}
        if after:
            kwargs["after"] = after  # OpenAI list pagination

        page = vs_api(client).files.list(**kwargs)

        data = getattr(page, "data", None) or []
        if not data:
            break

        collected.extend(data)
        # next cursor: last item's id
        last_id = getattr(data[-1], "id", None) or obj_to_dict(data[-1]).get("id")
        after = last_id

        # ha nincs több:
        has_more = getattr(page, "has_more", None)
        if has_more is False:
            break

        # extra guard: ha nincs last_id, nem tudunk tovább lapozni
        if not after:
            break

    return collected

# ---------------- HTML FORMATTER ----------------
def format_to_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"【.*?】", "", text)

    lines = text.split("\n")
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
            content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line[2:])
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
            line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            html_lines.append(f'<p style="margin: 0 0 8px 0;">{line}</p>')

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

Dátum: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = NOTIFY_EMAIL

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email elküldve.")
    except Exception as e:
        print(f"❌ Email hiba: {e}")

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
            if 1 <= len(parts) <= 4:
                name = first_line

    desc = ""
    dm = re.search(r"(?:^|\n)\s*(?:leírás|leiras|projekt|rövid leírás|rovid leiras)\s*:\s*(.+)", text, re.I)
    if dm:
        desc = dm.group(1).strip()
    else:
        desc = text.strip()
        if len(desc) > 800:
            desc = desc[:800] + "..."

    return {
        "name": name or "Ismeretlen",
        "email": email,
        "phone": phone,
        "description": desc,
    }

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
                    "required": ["name", "email"],
                },
            },
        },
    ]

    tool_resources = {}
    if OPENAI_VECTOR_STORE_ID:
        tool_resources = {"file_search": {"vector_store_ids": [OPENAI_VECTOR_STORE_ID]}}

    asst = assistants_api(client).create(
        name="Videmark Assistant V4.7",
        instructions=SYSTEM_PROMPT,
        model=OPENAI_MODEL,
        tools=tools,
        tool_resources=tool_resources,
    )
    OPENAI_ASSISTANT_ID = asst.id
    return OPENAI_ASSISTANT_ID

# ---------------- OCR ----------------
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
        temperature=0,
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
    return {"file_id": f.id, "vector_store_file_id": getattr(vsf, "id", None), "filename": filename}

# ---------------- MODELS ----------------
class ChatReq(BaseModel):
    message: str
    session_id: str
    chatbot_secret: Optional[str] = None

class ChatResp(BaseModel):
    reply: str

# ---------------- ENDPOINTS ----------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, x_chatbot_secret: str = Header(default="")):
    token = req.chatbot_secret or x_chatbot_secret
    if CHATBOT_SECRET and token != CHATBOT_SECRET:
        raise HTTPException(401, "Unauthorized")

    user_msg = (req.message or "").strip()
    log_event({"type": "user", "session_id": req.session_id, "text": user_msg})

    # --- EASTER EGGS (csak pontos egyezés) ---
    if user_msg == "Csákó Edina":
        reply = "SZeretlek Drágám!"
        log_event({"type": "assistant", "session_id": req.session_id, "text": reply})
        return ChatResp(reply=format_to_html(reply))

    if user_msg == "Dani vagyok":
        reply = "Szia Webmester Mekmester"
        log_event({"type": "assistant", "session_id": req.session_id, "text": reply})
        return ChatResp(reply=format_to_html(reply))

    # --- LEAD fallback: ha előzőleg LEAD-et kértünk, próbáljuk kiszedni a user üzenetből ---
    if _lead_pending.get(req.session_id):
        lead = extract_lead_from_text(user_msg)
        if lead:
            send_email_notification(lead)
            _lead_pending[req.session_id] = False
            reply = "Köszi! Megkaptuk az adataid, hamarosan felvesszük veled a kapcsolatot. ✅"
            log_event({"type": "assistant", "session_id": req.session_id, "text": reply, "lead": lead})
            return ChatResp(reply=format_to_html(reply))

    # --- 1-2 szavas üzenet guard ---
    if is_single_keyword(user_msg):
        reply = (
            "Kérlek pontosíts egy kicsit: melyik szolgáltatás érdekel pontosan (videó, drón, fotózás, vágás stb.) "
            "és milyen jellegű projektről van szó?"
        )
        log_event({"type": "assistant", "session_id": req.session_id, "text": reply, "guard": "single_keyword"})
        return ChatResp(reply=format_to_html(reply))

    assistant_id = get_or_create_assistant()
    threads = threads_api(client)

    thread_id = _thread_map.get(req.session_id)
    if not thread_id:
        thread = threads.create()
        thread_id = thread.id
        _thread_map[req.session_id] = thread_id

    threads.messages.create(thread_id=thread_id, role="user", content=user_msg)
    run = threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)

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
                        send_email_notification(args)
                        output_str = '{"success": true, "message": "Email elküldve."}'
                        _lead_pending[req.session_id] = False
                        log_event({"type": "lead", "session_id": req.session_id, "lead": args})
                    except Exception:
                        output_str = '{"success": false}'
                    tool_outputs.append({"tool_call_id": tool_call.id, "output": output_str})

            if tool_outputs:
                threads.runs.submit_tool_outputs(thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs)
            continue

        if run_status.status in ["failed", "cancelled", "expired"]:
            reply = "Hiba történt. Próbáld újra."
            log_event({"type": "assistant", "session_id": req.session_id, "text": reply, "error": run_status.status})
            return ChatResp(reply=format_to_html(reply))

        time.sleep(0.5)

    messages = threads.messages.list(thread_id=thread_id)
    last_msg = messages.data[0]

    reply_text = ""
    if last_msg.role == "assistant":
        raw_parts = []
        for content in last_msg.content:
            if content.type == "text":
                raw_parts.append(content.text.value)
        raw_text = "\n".join(raw_parts).strip()

        if "add meg az alábbi adatokat" in raw_text.lower() or "felvesszük veled a kapcsolatot" in raw_text.lower():
            _lead_pending[req.session_id] = True

        reply_text = format_to_html(raw_text)
        log_event({"type": "assistant", "session_id": req.session_id, "text": raw_text})

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
            txt_name = re.sub(r"\.[a-z0-9]+$", "", filename, flags=re.I) + "_OCR.txt"
            results.append({"source": filename, "mode": "image_ocr", "uploaded": upload_text_as_file_to_vector_store(text, txt_name)})
            continue

        is_pdf = (content_type in ("application/pdf", "application/x-pdf")) or filename.lower().endswith(".pdf")
        if ocr and is_pdf:
            extracted = extract_text_from_pdf(raw)
            if len(extracted.strip()) >= 80:
                txt_name = re.sub(r"\.pdf$", "", filename, flags=re.I) + "_TEXT.txt"
                results.append({"source": filename, "mode": "pdf_text_extract", "uploaded": upload_text_as_file_to_vector_store(extracted, txt_name)})
            else:
                ocred = ocr_pdf_with_openai(raw)
                txt_name = re.sub(r"\.pdf$", "", filename, flags=re.I) + "_OCR.txt"
                results.append({"source": filename, "mode": "pdf_ocr", "uploaded": upload_text_as_file_to_vector_store(ocred, txt_name)})
            continue

        f = client.files.create(file=(filename, raw), purpose="assistants")
        vsf = vs_api(client).files.create(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=f.id)
        results.append({"source": filename, "mode": "direct", "file_id": f.id, "vector_store_file_id": getattr(vsf, "id", None)})

    return {"status": "ok", "results": results}

# ---------------- ADMIN: FILES LIST (FIXED LIMIT<=10 + paged) ----------------
@app.get("/admin/files")
def admin_files(
    x_admin_secret: str = Header(default=""),
    limit: int = Query(default=50, ge=1, le=200),
):
    require_admin(x_admin_secret)

    if not OPENAI_VECTOR_STORE_ID:
        return {"status": "ok", "files": [], "warning": "Nincs OPENAI_VECTOR_STORE_ID a Render ENV-ben."}

    # OpenAI oldali limit max 10, mi pedig lapozva gyűjtjük össze a kért mennyiséget
    try:
        items = _vs_files_list_paged(max_items=min(limit, 200))
    except Exception as e:
        return {"status": "ok", "files": [], "warning": f"OpenAI list hiba: {str(e)}"}

    out = []
    for it in items:
        d_it = obj_to_dict(it)
        vs_file_id = getattr(it, "id", None) or d_it.get("id") or ""
        status = getattr(it, "status", "") or d_it.get("status", "")

        file_id = None
        try:
            details = vs_api(client).files.retrieve(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=vs_file_id)
            dd = obj_to_dict(details)
            file_id = (
                dd.get("file_id")
                or (dd.get("file") or {}).get("id")
                or (dd.get("file") or {}).get("file_id")
                or dd.get("openai_file_id")
                or dd.get("source_file_id")
            )
            status = dd.get("status") or status
        except Exception:
            dd = {}

        if not file_id and isinstance(vs_file_id, str) and vs_file_id.startswith("file-"):
            file_id = vs_file_id

        meta = safe_get_file_meta(file_id) if file_id else {"filename": "", "created_at": None}

        out.append({
            "vector_store_file_id": vs_file_id,
            "file_id": file_id,
            "status": status,
            "filename": meta.get("filename", "") or "",
            "created_at": meta.get("created_at", None),
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

    underlying_file_id = None
    try:
        vs_item = vs_api(client).files.retrieve(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=vector_store_file_id)
        dd = obj_to_dict(vs_item)
        underlying_file_id = (
            dd.get("file_id")
            or (dd.get("file") or {}).get("id")
            or (dd.get("file") or {}).get("file_id")
            or dd.get("openai_file_id")
            or dd.get("source_file_id")
        )
    except Exception:
        underlying_file_id = None

    vs_api(client).files.delete(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=vector_store_file_id)

    if delete_underlying_file:
        if not underlying_file_id and vector_store_file_id.startswith("file-"):
            underlying_file_id = vector_store_file_id
        if underlying_file_id:
            try:
                client.files.delete(underlying_file_id)
            except Exception:
                pass

    return {
        "status": "ok",
        "deleted_vector_store_file_id": vector_store_file_id,
        "deleted_underlying_file_id": underlying_file_id,
    }

@app.post("/admin/create_vector_store")
def create_vs(name: str = "Store", x_admin_secret: str = Header(default="")):
    require_admin(x_admin_secret)
    vs = vs_api(client).create(name=name)
    return {"id": vs.id}

# Debug (ha kell)
@app.get("/admin/files_raw")
def admin_files_raw(
    x_admin_secret: str = Header(default=""),
    limit: int = Query(default=20, ge=1, le=200),
):
    require_admin(x_admin_secret)
    if not OPENAI_VECTOR_STORE_ID:
        return {"status": "ok", "raw": [], "warning": "Nincs OPENAI_VECTOR_STORE_ID."}

    try:
        items = _vs_files_list_paged(max_items=min(limit, 200))
        raw = [obj_to_dict(it) for it in items]
        return {"status": "ok", "raw": raw}
    except Exception as e:
        return {"status": "ok", "raw": [], "warning": str(e)}

# ---------------- ADMIN: LOGS (optional) ----------------
def read_logs(limit: int = 200, session_id: Optional[str] = None) -> List[dict]:
    if not LOG_ENABLED:
        return []
    if not os.path.exists(LOG_PATH):
        return []
    rows: List[dict] = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if session_id and obj.get("session_id") != session_id:
                    continue
                rows.append(obj)
            except Exception:
                continue
    return rows[-limit:]

def summarize_sessions(limit: int = 50) -> List[dict]:
    logs = read_logs(limit=5000, session_id=None)
    last_by_session: Dict[str, dict] = {}
    counts: Dict[str, int] = {}
    for e in logs:
        sid = e.get("session_id") or ""
        if not sid:
            continue
        counts[sid] = counts.get(sid, 0) + 1
        if sid not in last_by_session or int(e.get("ts", 0)) > int(last_by_session[sid].get("ts", 0)):
            last_by_session[sid] = e
    sessions = sorted(last_by_session.items(), key=lambda kv: int(kv[1].get("ts", 0)), reverse=True)
    out = []
    for sid, last in sessions[:limit]:
        out.append({
            "session_id": sid,
            "last_ts": int(last.get("ts", 0)),
            "count": counts.get(sid, 0),
            "last_type": last.get("type", ""),
            "last_text": (last.get("text", "") or "")[:160],
        })
    return out

@app.get("/admin/logs")
def admin_logs(
    x_admin_secret: str = Header(default=""),
    session_id: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
):
    require_admin(x_admin_secret)
    try:
        data = read_logs(limit=limit, session_id=session_id)
        return {"status": "ok", "logs_enabled": LOG_ENABLED, "log_path": LOG_PATH, "logs": data}
    except Exception as e:
        return {"status": "ok", "logs_enabled": LOG_ENABLED, "log_path": LOG_PATH, "logs": [], "warning": str(e)}

@app.get("/admin/log_sessions")
def admin_log_sessions(
    x_admin_secret: str = Header(default=""),
    limit: int = Query(default=50, ge=1, le=200),
):
    require_admin(x_admin_secret)
    try:
        return {"status": "ok", "sessions": summarize_sessions(limit=limit)}
    except Exception as e:
        return {"status": "ok", "sessions": [], "warning": str(e)}
