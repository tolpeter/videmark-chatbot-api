import os
import time
import json
import re
import smtplib
import base64
from email.mime.text import MIMEText
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from bs4 import BeautifulSoup
import requests

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID", "").strip()

CHATBOT_SECRET = os.getenv("CHATBOT_SECRET", "").strip()
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "").strip()

# JWT settings (simple token-based auth)
JWT_SECRET = os.getenv("JWT_SECRET", "videmark-super-secret-change-me").strip()
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

# Email beállítások
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "info@videmark.hu")

# Persistent data paths (use /var/data on Render for persistence)
DATA_DIR = os.getenv("DATA_DIR", "/var/data").strip()
os.makedirs(DATA_DIR, exist_ok=True)

PROMPTS_FILE = os.path.join(DATA_DIR, "prompts.json")
NOTES_FILE = os.path.join(DATA_DIR, "notes.json")
LOG_PATH = os.path.join(DATA_DIR, "chat_logs.jsonl")

LOG_ENABLED = os.getenv("LOG_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "https://videmark.hu,https://www.videmark.hu").split(",")
    if o.strip()
]

# ---------------- DEFAULT PROMPT ----------------
DEFAULT_SYSTEM_PROMPT = """
Te a Videmark weboldal profi értékesítő asszisztense vagy.

TUDÁSBÁZIS: A válaszaidhoz elsődlegesen a feltöltött fájlok tartalmát használd (file_search).
Ha a kérdésre van konkrét válasz a tudásbázisban (pl. szolgáltatás + ár), akkor AZONNAL azt add meg.

FONTOS VISELKEDÉSI SZABÁLYOK:

0. TUDÁSBÁZIS ELSŐDLEGESSÉGE:
- A válaszaidat mindig a tudásbázis alapján add.
- NE írj magyarázó mondatot arról, hogy "a feltöltött fájlok alapján" válaszolsz – csak válaszolj.

1. PONTOSÍTÁS:
- Ha a felhasználó általánosan kérdez (pl. „Mennyibe kerül egy videó?" / „Milyen árak vannak?"),
  akkor NE sorold fel az összes árat automatikusan.
- Kérdezz vissza röviden, hogy milyen típus érdekli (csak olyan példákat adj, amik a tudásbázisban szerepelnek).

2. NINCS TALÁLGATÁS (Kritikus!):
- SOHA ne említs olyan szolgáltatást / fotózási típust / videós típust vagy árat, ami NEM szerepel a tudásbázisban.
- TILOS példaként felsorolni olyan opciókat (pl. esküvői fotózás), amelyek nem találhatók meg a tudásbázisban.
- TILOS becsült, "kb.", "általában ennyi", "tól-ig" jellegű árat adni.

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

app = FastAPI(title="Videmark Chatbot API v5.0 (Admin Panel + Knowledge Base)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATA MODELS ----------------
class ChatReq(BaseModel):
    message: str
    session_id: str = ""
    chatbot_secret: Optional[str] = None

class ChatResp(BaseModel):
    reply: str

class LoginReq(BaseModel):
    password: str

class LoginResp(BaseModel):
    token: str
    expires_at: str

class PromptReq(BaseModel):
    name: str
    content: str
    active: bool = False

class NoteReq(BaseModel):
    title: str
    content: str
    tags: List[str] = []

class ScrapeReq(BaseModel):
    url: str
    selector: Optional[str] = None

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

def vs_api(_client: OpenAI):
    if hasattr(_client, "beta") and hasattr(_client.beta, "vector_stores"):
        return _client.beta.vector_stores
    if hasattr(_client, "vector_stores"):
        return _client.vector_stores
    raise HTTPException(500, "OpenAI python csomag túl régi")

def assistants_api(_client: OpenAI):
    if hasattr(_client, "beta") and hasattr(_client.beta, "assistants"):
        return _client.beta.assistants
    raise HTTPException(500, "OpenAI beta assistants nem elérhető")

def threads_api(_client: OpenAI):
    if hasattr(_client, "beta") and hasattr(_client.beta, "threads"):
        return _client.beta.threads
    raise HTTPException(500, "OpenAI beta threads nem elérhető")

VS_LIST_PAGE_LIMIT = 10

def _vs_files_list_paged(max_items: int = 200) -> List[Any]:
    require_vs()
    collected: List[Any] = []
    after: Optional[str] = None

    if max_items < 1:
        return []

    while len(collected) < max_items:
        remaining = max_items - len(collected)
        page_limit = min(VS_LIST_PAGE_LIMIT, remaining)

        kwargs = {"vector_store_id": OPENAI_VECTOR_STORE_ID, "limit": page_limit}
        if after:
            kwargs["after"] = after

        page = vs_api(client).files.list(**kwargs)

        data = getattr(page, "data", None) or []
        if not data:
            break

        collected.extend(data)
        last_id = getattr(data[-1], "id", None) or obj_to_dict(data[-1]).get("id")
        after = last_id

        has_more = getattr(page, "has_more", None)
        if has_more is False:
            break

        if not after:
            break

    return collected

# ---------------- JWT AUTH ----------------
def create_token() -> dict:
    expires_at = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)
    token_data = {
        "exp": int(expires_at.timestamp()),
        "iat": int(datetime.utcnow().timestamp()),
    }
    # Simple token (base64 encoded JSON + signature)
    token_json = json.dumps(token_data)
    token_b64 = base64.b64encode(token_json.encode()).decode()
    # Signature: hash of token + secret
    import hashlib
    signature = hashlib.sha256((token_b64 + JWT_SECRET).encode()).hexdigest()
    token = f"{token_b64}.{signature}"
    
    return {
        "token": token,
        "expires_at": expires_at.isoformat()
    }

def verify_token(token: str) -> bool:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return False
        
        token_b64, signature = parts
        
        # Verify signature
        import hashlib
        expected_sig = hashlib.sha256((token_b64 + JWT_SECRET).encode()).hexdigest()
        if signature != expected_sig:
            return False
        
        # Verify expiry
        token_json = base64.b64decode(token_b64).decode()
        token_data = json.loads(token_json)
        
        if token_data["exp"] < int(datetime.utcnow().timestamp()):
            return False
        
        return True
    except Exception:
        return False

def require_auth(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized - No token provided")
    
    token = authorization.replace("Bearer ", "")
    if not verify_token(token):
        raise HTTPException(401, "Unauthorized - Invalid or expired token")

# ---------------- PROMPTS MANAGEMENT ----------------
def load_prompts() -> List[dict]:
    if not os.path.exists(PROMPTS_FILE):
        # Create default prompt
        default = {
            "id": "default",
            "name": "Default System Prompt",
            "content": DEFAULT_SYSTEM_PROMPT,
            "active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        save_prompts([default])
        return [default]
    
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_prompts(prompts: List[dict]):
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

def get_active_prompt() -> str:
    prompts = load_prompts()
    for p in prompts:
        if p.get("active"):
            return p.get("content", DEFAULT_SYSTEM_PROMPT)
    return DEFAULT_SYSTEM_PROMPT

# ---------------- NOTES MANAGEMENT ----------------
def load_notes() -> List[dict]:
    if not os.path.exists(NOTES_FILE):
        return []
    
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_notes(notes: List[dict]):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

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
    
    # Get active prompt
    system_prompt = get_active_prompt()
    
    if OPENAI_ASSISTANT_ID:
        # Update existing assistant with current prompt
        try:
            assistants_api(client).update(
                assistant_id=OPENAI_ASSISTANT_ID,
                instructions=system_prompt
            )
            return OPENAI_ASSISTANT_ID
        except Exception:
            pass

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
        name="Videmark Assistant V5.0",
        instructions=system_prompt,
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
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Írd ki a képen látható szöveget pontos formázással."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=4000,
    )
    return resp.choices[0].message.content or ""

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(text_parts)

def ocr_pdf_with_openai(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_parts = []
    for page_num in range(min(5, len(doc))):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        extracted_parts.append(ocr_image_with_openai(img_bytes, "image/png"))
    doc.close()
    return "\n\n".join(extracted_parts)

def upload_text_as_file_to_vector_store(text: str, filename: str) -> dict:
    require_vs()
    from io import BytesIO

    txt_bytes = text.encode("utf-8")
    f = client.files.create(file=(filename, BytesIO(txt_bytes)), purpose="assistants")
    vsf = vs_api(client).files.create(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=f.id)

    return {
        "file_id": f.id,
        "vector_store_file_id": getattr(vsf, "id", None),
        "filename": filename,
    }

# ---------------- WEB SCRAPING ----------------
def scrape_website(url: str, selector: Optional[str] = None) -> str:
    """
    Scrape text content from a website
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        
        # Get text
        if selector:
            elements = soup.select(selector)
            text = "\n\n".join([elem.get_text(strip=True) for elem in elements])
        else:
            text = soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        
        return text
    except Exception as e:
        raise HTTPException(400, f"Scraping hiba: {str(e)}")

# ---------------- ENDPOINTS ----------------
@app.get("/")
def root():
    return {"status": "ok", "version": "5.0", "features": ["admin_panel", "knowledge_base", "web_scraping"]}

# ---------------- AUTH ENDPOINTS ----------------
@app.post("/admin/login", response_model=LoginResp)
def admin_login(req: LoginReq):
    if req.password != ADMIN_SECRET:
        raise HTTPException(401, "Helytelen jelszó")
    
    token_data = create_token()
    return LoginResp(**token_data)

@app.get("/admin/verify")
def admin_verify(authorization: str = Header(default="")):
    require_auth(authorization)
    return {"status": "ok", "authenticated": True}

# ---------------- PROMPTS ENDPOINTS ----------------
@app.get("/admin/prompts")
def get_prompts(authorization: str = Header(default="")):
    require_auth(authorization)
    prompts = load_prompts()
    return {"status": "ok", "prompts": prompts}

@app.post("/admin/prompts")
def create_prompt(req: PromptReq, authorization: str = Header(default="")):
    require_auth(authorization)
    
    prompts = load_prompts()
    
    # If setting as active, deactivate others
    if req.active:
        for p in prompts:
            p["active"] = False
    
    new_prompt = {
        "id": f"prompt_{int(time.time())}",
        "name": req.name,
        "content": req.content,
        "active": req.active,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    prompts.append(new_prompt)
    save_prompts(prompts)
    
    # Update assistant if this is active
    if req.active:
        get_or_create_assistant()
    
    return {"status": "ok", "prompt": new_prompt}

@app.put("/admin/prompts/{prompt_id}")
def update_prompt(prompt_id: str, req: PromptReq, authorization: str = Header(default="")):
    require_auth(authorization)
    
    prompts = load_prompts()
    found = False
    
    for p in prompts:
        if p["id"] == prompt_id:
            # If setting as active, deactivate others
            if req.active:
                for other in prompts:
                    other["active"] = False
            
            p["name"] = req.name
            p["content"] = req.content
            p["active"] = req.active
            p["updated_at"] = datetime.utcnow().isoformat()
            found = True
            break
    
    if not found:
        raise HTTPException(404, "Prompt nem található")
    
    save_prompts(prompts)
    
    # Update assistant if this is active
    if req.active:
        get_or_create_assistant()
    
    return {"status": "ok"}

@app.delete("/admin/prompts/{prompt_id}")
def delete_prompt(prompt_id: str, authorization: str = Header(default="")):
    require_auth(authorization)
    
    prompts = load_prompts()
    prompts = [p for p in prompts if p["id"] != prompt_id]
    
    # Make sure at least one is active
    if prompts and not any(p.get("active") for p in prompts):
        prompts[0]["active"] = True
        get_or_create_assistant()
    
    save_prompts(prompts)
    
    return {"status": "ok"}

@app.post("/admin/prompts/{prompt_id}/activate")
def activate_prompt(prompt_id: str, authorization: str = Header(default="")):
    require_auth(authorization)
    
    prompts = load_prompts()
    
    for p in prompts:
        p["active"] = (p["id"] == prompt_id)
    
    save_prompts(prompts)
    get_or_create_assistant()
    
    return {"status": "ok"}

# ---------------- NOTES ENDPOINTS ----------------
@app.get("/admin/notes")
def get_notes(authorization: str = Header(default="")):
    require_auth(authorization)
    notes = load_notes()
    return {"status": "ok", "notes": notes}

@app.post("/admin/notes")
def create_note(req: NoteReq, authorization: str = Header(default="")):
    require_auth(authorization)
    
    notes = load_notes()
    
    new_note = {
        "id": f"note_{int(time.time())}",
        "title": req.title,
        "content": req.content,
        "tags": req.tags,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    notes.append(new_note)
    save_notes(notes)
    
    # Upload to vector store
    try:
        filename = f"{req.title.replace(' ', '_')}.txt"
        result = upload_text_as_file_to_vector_store(req.content, filename)
        new_note["vector_file_id"] = result.get("file_id")
        save_notes(notes)
    except Exception as e:
        print(f"Warning: Could not upload note to vector store: {e}")
    
    return {"status": "ok", "note": new_note}

@app.put("/admin/notes/{note_id}")
def update_note(note_id: str, req: NoteReq, authorization: str = Header(default="")):
    require_auth(authorization)
    
    notes = load_notes()
    found = False
    
    for n in notes:
        if n["id"] == note_id:
            n["title"] = req.title
            n["content"] = req.content
            n["tags"] = req.tags
            n["updated_at"] = datetime.utcnow().isoformat()
            found = True
            break
    
    if not found:
        raise HTTPException(404, "Jegyzet nem található")
    
    save_notes(notes)
    
    return {"status": "ok"}

@app.delete("/admin/notes/{note_id}")
def delete_note(note_id: str, authorization: str = Header(default="")):
    require_auth(authorization)
    
    notes = load_notes()
    notes = [n for n in notes if n["id"] != note_id]
    save_notes(notes)
    
    return {"status": "ok"}

# ---------------- WEB SCRAPING ENDPOINT ----------------
@app.post("/admin/scrape")
def scrape_and_upload(req: ScrapeReq, authorization: str = Header(default="")):
    require_auth(authorization)
    require_vs()
    
    # Scrape the website
    text = scrape_website(req.url, req.selector)
    
    if not text or len(text) < 50:
        raise HTTPException(400, "Túl kevés tartalom lett kinyerve az oldalról")
    
    # Create filename from URL
    from urllib.parse import urlparse
    parsed = urlparse(req.url)
    filename = f"{parsed.netloc.replace('.', '_')}_{int(time.time())}.txt"
    
    # Upload to vector store
    result = upload_text_as_file_to_vector_store(text, filename)
    
    return {
        "status": "ok",
        "url": req.url,
        "characters_extracted": len(text),
        "file_id": result.get("file_id"),
        "filename": filename
    }

# ---------------- CHAT ENDPOINT (unchanged) ----------------
@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, x_chatbot_secret: str = Header(default="")):
    token = req.chatbot_secret or x_chatbot_secret
    if CHATBOT_SECRET and token != CHATBOT_SECRET:
        raise HTTPException(401, "Unauthorized")

    user_msg = (req.message or "").strip()
    log_event({"type": "user", "session_id": req.session_id, "text": user_msg})

    # Easter eggs
    if user_msg == "Csákó Edina":
        reply = "SZeretlek Drágám!"
        log_event({"type": "assistant", "session_id": req.session_id, "text": reply})
        return ChatResp(reply=format_to_html(reply))

    if user_msg == "Dani vagyok":
        reply = "Szia Webmester Mekmester"
        log_event({"type": "assistant", "session_id": req.session_id, "text": reply})
        return ChatResp(reply=format_to_html(reply))

    # LEAD fallback
    if _lead_pending.get(req.session_id):
        lead = extract_lead_from_text(user_msg)
        if lead:
            send_email_notification(lead)
            _lead_pending[req.session_id] = False
            reply = "Köszi! Megkaptuk az adataid, hamarosan felvesszük veled a kapcsolatot. ✅"
            log_event({"type": "assistant", "session_id": req.session_id, "text": reply, "lead": lead})
            return ChatResp(reply=format_to_html(reply))

    # Single keyword guard
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

# ---------------- ADMIN: UPLOAD (unchanged) ----------------
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
        raise HTTPException(422, "No files provided")

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

# ---------------- ADMIN: FILES LIST ----------------
@app.get("/admin/files")
def admin_files(
    x_admin_secret: str = Header(default=""),
    limit: int = Query(default=50, ge=1, le=200),
):
    require_admin(x_admin_secret)

    if not OPENAI_VECTOR_STORE_ID:
        return {"status": "ok", "files": [], "warning": "Nincs OPENAI_VECTOR_STORE_ID"}

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

# ---------------- ADMIN: ANALYTICS ----------------
@app.get("/admin/analytics")
def admin_analytics(authorization: str = Header(default="")):
    require_auth(authorization)
    
    if not LOG_ENABLED or not os.path.exists(LOG_PATH):
        return {
            "status": "ok",
            "total_messages": 0,
            "total_sessions": 0,
            "total_leads": 0,
            "recent_activity": []
        }
    
    try:
        logs = []
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except Exception:
                        continue
        
        # Calculate stats
        total_messages = len([l for l in logs if l.get("type") == "user"])
        sessions = set([l.get("session_id") for l in logs if l.get("session_id")])
        total_sessions = len(sessions)
        total_leads = len([l for l in logs if l.get("type") == "lead"])
        
        # Recent activity (last 10)
        recent = logs[-10:] if len(logs) > 10 else logs
        
        return {
            "status": "ok",
            "total_messages": total_messages,
            "total_sessions": total_sessions,
            "total_leads": total_leads,
            "recent_activity": recent
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
