import os
import time
import json
import re
import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, List, Any

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
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

# Naplózás helye (Renderen /var/data ajánlott perzisztens diskhez)
LOG_PATH = os.getenv("LOG_PATH", "/var/data/chat_logs.jsonl")
if not os.path.isabs(LOG_PATH): LOG_PATH = "./chat_logs.jsonl" # Fallback

LEAD_BACKUP_PATH = os.getenv("LEAD_BACKUP_PATH", "/var/data/leads_backup.jsonl")
if not os.path.isabs(LEAD_BACKUP_PATH): LEAD_BACKUP_PATH = "./leads_backup.jsonl" # Fallback

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "https://videmark.hu,https://www.videmark.hu").split(",")
    if o.strip()
]

# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """
Te a Videmark weboldal profi értékesítő asszisztense vagy.

TUDÁSBÁZIS: A válaszaidhoz elsődlegesen a feltöltött fájlok tartalmát használd (file_search).
Ha a kérdésre van konkrét válasz a tudásbázisban, akkor AZONNAL azt add meg.

SZABÁLYOK:
1. PONTOSÍTÁS: Ha a felhasználó általánosan kérdez (pl. "Mennyibe kerül egy videó?"), kérdezz vissza a típusra. Ne listázd ki az összeset.
2. NINCS TALÁLGATÁS: SOHA ne mondj olyan árat vagy szolgáltatást, ami nincs a fájlokban.
3. LEAD GYŰJTÉS: Ha nincs válasz a fájlokban, kérj elérhetőséget (Név, Email, Telefon, Leírás) és mondd, hogy felvesszük vele a kapcsolatot.
4. LEAD MENTÉS: Ha megadja az adatokat, hívd meg a `save_lead` funkciót!
5. FORMÁZÁS: Használj Markdown-t (**félkövér**, listák).

Stílus: Magyar, közvetlen, segítőkész.
"""

client = OpenAI(api_key=OPENAI_API_KEY)

# Memória cache
_thread_map: Dict[str, str] = {}
_lead_pending: Dict[str, bool] = {}

app = FastAPI(title="Videmark Chatbot API v5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HELPERS ----------------
def require_admin(secret: str):
    if ADMIN_SECRET and secret != ADMIN_SECRET:
        raise HTTPException(401, "Unauthorized (admin)")

def log_event(event: dict):
    """Esemény naplózása fájlba."""
    try:
        event["ts"] = int(time.time())
        event["dt"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Log hiba: {e}")

def save_lead_backup(lead_data: dict):
    """Biztonsági mentés, ha az email nem menne el."""
    try:
        lead_data["ts"] = datetime.datetime.now().isoformat()
        with open(LEAD_BACKUP_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(lead_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Lead backup hiba: {e}")

def send_email_notification(lead_data: dict):
    """Email küldése SMTP-vel."""
    print(f"📧 Email küldés: {lead_data.get('email')}")
    
    if not SMTP_USER or not SMTP_PASSWORD:
        print("⚠️ Nincs SMTP beállítva! Csak backup mentés.")
        save_lead_backup(lead_data)
        return False

    subject = f"🔥 ÚJ LEAD: {lead_data.get('name', 'Ismeretlen')}"
    body_text = f"""
Új érdeklődő érkezett a Chatbotból!

Név: {lead_data.get('name')}
Email: {lead_data.get('email')}
Telefon: {lead_data.get('phone')}
Leírás: {lead_data.get('description')}

Dátum: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = NOTIFY_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email elküldve.")
        return True
    except Exception as e:
        print(f"❌ Email hiba: {e}")
        save_lead_backup(lead_data) # Ha hiba van, mentsük fájlba
        return False

# Regex fallback
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

def extract_lead_from_text(text: str) -> Optional[dict]:
    if not text: return None
    m = EMAIL_RE.search(text)
    if not m: return None
    return {
        "name": "Chat Felhasználó",
        "email": m.group(0).strip(),
        "phone": (PHONE_RE.search(text) or type('obj', (object,), {'group': lambda x: ""})).group(0).strip(),
        "description": text
    }

# ---------------- ASSISTANT ----------------
def get_or_create_assistant():
    global OPENAI_ASSISTANT_ID
    if OPENAI_ASSISTANT_ID: return OPENAI_ASSISTANT_ID

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
                        "description": {"type": "string"},
                    },
                    "required": ["name", "email"],
                },
            },
        },
    ]
    
    tool_res = {}
    if OPENAI_VECTOR_STORE_ID:
        tool_res = {"file_search": {"vector_store_ids": [OPENAI_VECTOR_STORE_ID]}}

    asst = client.beta.assistants.create(
        name="Videmark Assistant v5",
        instructions=SYSTEM_PROMPT,
        model=OPENAI_MODEL,
        tools=tools,
        tool_resources=tool_res,
    )
    OPENAI_ASSISTANT_ID = asst.id
    return OPENAI_ASSISTANT_ID

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
    return {"status": "ok", "time": datetime.datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, x_chatbot_secret: str = Header(default="")):
    token = req.chatbot_secret or x_chatbot_secret
    if CHATBOT_SECRET and token != CHATBOT_SECRET:
        raise HTTPException(401, "Unauthorized")

    user_msg = (req.message or "").strip()
    log_event({"type": "user", "session_id": req.session_id, "text": user_msg})

    # --- EASTER EGGS (Visszatettem!) ---
    if user_msg == "Csákó Edina":
        reply = "Szeretlek Drágám! ❤️"
        log_event({"type": "assistant", "session_id": req.session_id, "text": reply})
        return ChatResp(reply=reply)

    if user_msg == "Dani vagyok":
        reply = "Szia Webmester Mekmester! 🛠️"
        log_event({"type": "assistant", "session_id": req.session_id, "text": reply})
        return ChatResp(reply=reply)

    # Lead Fallback (ha a tool nem hívódott meg, de kértük)
    if _lead_pending.get(req.session_id):
        lead = extract_lead_from_text(user_msg)
        if lead:
            send_email_notification(lead)
            _lead_pending[req.session_id] = False
            reply = "Köszi! Megkaptuk az adataid, hamarosan keresünk. ✅"
            log_event({"type": "assistant", "session_id": req.session_id, "text": reply, "lead": lead})
            return ChatResp(reply=reply)

    assistant_id = get_or_create_assistant()
    
    # Thread (szál) kezelése
    thread_id = _thread_map.get(req.session_id)
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id
        _thread_map[req.session_id] = thread_id

    # Idő kontextus hozzáadása
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
    context_msg = f"[System Info: Jelenlegi idő: {current_time_str}]\n\n{user_msg}"

    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=context_msg)
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)

    # Várakozás a válaszra (Polling)
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        
        if run_status.status == "completed":
            break
        
        if run_status.status == "requires_action":
            tool_outputs = []
            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                if tool_call.function.name == "save_lead":
                    args = json.loads(tool_call.function.arguments)
                    success = send_email_notification(args)
                    log_event({"type": "lead", "session_id": req.session_id, "lead": args})
                    output_str = '{"success": true}' if success else '{"success": false}'
                    tool_outputs.append({"tool_call_id": tool_call.id, "output": output_str})
                    _lead_pending[req.session_id] = False
            
            if tool_outputs:
                client.beta.threads.runs.submit_tool_outputs(thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs)
            continue
            
        if run_status.status in ["failed", "cancelled", "expired"]:
            return ChatResp(reply="Hiba történt a feldolgozás során.")
        
        time.sleep(0.5)

    # Utolsó válasz kinyerése
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    reply_text = ""
    if messages.data and messages.data[0].role == "assistant":
        for content in messages.data[0].content:
            if content.type == "text":
                reply_text += content.text.value
    
    # Ha adatot kér az AI, jegyezzük fel
    if "add meg az alábbi adatokat" in reply_text.lower():
        _lead_pending[req.session_id] = True

    log_event({"type": "assistant", "session_id": req.session_id, "text": reply_text})
    return ChatResp(reply=reply_text)

# ---------------- ADMIN: LOG VIEWER ----------------
@app.get("/admin/view_logs", response_class=HTMLResponse)
def view_logs_html(secret: str = Query(..., alias="secret")):
    require_admin(secret)
    
    if not os.path.exists(LOG_PATH):
        return "<h1>Még nincs log fájl.</h1>"

    logs = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): logs.append(json.loads(line))
    except Exception as e:
        return f"Hiba: {e}"

    sessions = {}
    for entry in logs:
        sid = entry.get("session_id", "unknown")
        if sid not in sessions: sessions[sid] = []
        sessions[sid].append(entry)

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Videmark Napló</title>
        <style>
            body{font-family:sans-serif;background:#f3f4f6;padding:20px;}
            .card{background:#fff;padding:15px;margin-bottom:15px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}
            .head{font-weight:bold;color:#4F46E5;border-bottom:1px solid #eee;margin-bottom:10px;padding-bottom:5px;}
            .msg{padding:8px;margin:5px 0;border-radius:6px;font-size:14px;}
            .user{background:#e0e7ff;color:#1e1b4b;text-align:right;margin-left:20%;}
            .assistant{background:#f1f5f9;color:#334155;margin-right:20%;}
            .lead{background:#dcfce7;border:1px solid #22c55e;font-weight:bold;}
            .ts{font-size:11px;color:#999;display:block;}
        </style>
    </head>
    <body><h1>💬 Chat Napló</h1>
    """
    
    sorted_sids = sorted(sessions.keys(), key=lambda k: sessions[k][-1].get("ts", 0), reverse=True)

    for sid in sorted_sids:
        msgs = sessions[sid]
        last = msgs[-1].get("dt", "N/A")
        html += f'<div class="card"><div class="head">Session: {sid} | Utolsó: {last}</div>'
        for m in msgs:
            role = m.get("type","")
            txt = m.get("text","")
            ts = m.get("dt","")
            if role == "lead":
                html += f'<div class="msg lead"><span class="ts">{ts}</span>LEAD: {json.dumps(m.get("lead"), ensure_ascii=False)}</div>'
            else:
                html += f'<div class="msg {role}"><span class="ts">{ts} ({role})</span>{txt}</div>'
        html += '</div>'
    
    return html + "</body></html>"

# ---------------- ADMIN: UPLOAD & FILES ----------------
@app.post("/admin/upload")
def admin_upload(files: List[UploadFile] = File(default=[]), files_arr: List[UploadFile] = File(default=[], alias="files[]"), x_admin_secret: str = Header(default=""), ocr: bool = Query(default=True)):
    require_admin(x_admin_secret)
    all_files = files + files_arr
    if not all_files: raise HTTPException(422, "Nincs fájl kiválasztva")
    
    results = []
    for file in all_files:
        try:
            raw = file.file.read()
            # Feltöltés OpenAI-nak (automatikusan kezeli a tartalmat)
            f = client.files.create(file=(file.filename, raw), purpose="assistants")
            client.beta.vector_stores.files.create(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=f.id)
            results.append({"filename": file.filename, "status": "OK", "id": f.id})
        except Exception as e:
            results.append({"filename": file.filename, "status": "ERROR", "error": str(e)})
            
    return {"results": results}

@app.get("/admin/files")
def admin_files(x_admin_secret: str = Header(default="")):
    require_admin(x_admin_secret)
    if not OPENAI_VECTOR_STORE_ID: return {"files": []}
    files = client.beta.vector_stores.files.list(vector_store_id=OPENAI_VECTOR_STORE_ID, limit=100)
    
    out = []
    for f in files.data:
        # Fájl részletek lekérése a nevéért
        try:
            meta = client.files.retrieve(f.id)
            fname = meta.filename
        except: fname = "Unknown"
        out.append({"id": f.id, "filename": fname})
        
    return {"files": out}

@app.delete("/admin/files/{file_id}")
def admin_del(file_id: str, x_admin_secret: str = Header(default="")):
    require_admin(x_admin_secret)
    # Törlés Vector Store-ból és File Storage-ból is
    try:
        client.beta.vector_stores.files.delete(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=file_id)
        client.files.delete(file_id)
    except: pass
    return {"status": "deleted"}

@app.post("/admin/create_vector_store")
def create_vs(name: str = "VidemarkStore", x_admin_secret: str = Header(default="")):
    require_admin(x_admin_secret)
    vs = client.beta.vector_stores.create(name=name)
    return {"id": vs.id}
