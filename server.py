import os
import time
import json
import re  # Szövegtisztításhoz
import smtplib
from email.mime.text import MIMEText
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ---------------- CONFIG ----------------
# API Kulcsok
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Vector Store ID (Adatok helye)
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
# Assistant ID (Automatikus generáljuk, ha nincs)
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID", "").strip()

# Titkosítók
CHATBOT_SECRET = os.getenv("CHATBOT_SECRET", "").strip()
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "").strip()

# Email beállítások (Lead értesítéshez)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "info@videmark.hu")

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://videmark.hu,https://www.videmark.hu").split(",") if o.strip()
]

# Prompt
SYSTEM_PROMPT = """
Te a Videmark weboldal hivatalos, barátságos asszisztense vagy.

Szolgáltatások: Drón videó/fotó, reklámvideó, short tartalom (TikTok/Reels), fotózás.

Feladatod:
1. Válaszolj kérdésekre a tudásbázis (fájlok) alapján. Ha nincs infó, kérdezz vissza.
2. LEAD GYŰJTÉS: Ha az ügyfél érdeklődik, kérd el ezeket: Név, Email, Telefonszám, Projekt leírása.
3. HA megkaptad az adatokat, hívd meg a 'save_lead' funkciót!

Stílus: Magyar, tegező, segítőkész, rövid (max 3 mondat). Formázd a választ félkövér szöveggel a fontos részeknél.
""".strip()

# Globális kliens
client = OpenAI(api_key=OPENAI_API_KEY)

# Memória cache: Session ID -> Thread ID párosítás
_thread_map: Dict[str, str] = {}

app = FastAPI(title="Videmark Chatbot API v2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELLEK ----------------
class ChatReq(BaseModel):
    message: str
    session_id: str
    chatbot_secret: Optional[str] = None

class ChatResp(BaseModel):
    reply: str

# ---------------- SEGÉDFÜGGVÉNYEK ----------------

def send_email_notification(lead_data: dict):
    """Emailt küld neked, ha bejött egy lead."""
    if not SMTP_USER or not SMTP_PASSWORD:
        print("⚠️ Nincs beállítva SMTP, nem tudok emailt küldeni.")
        return

    subject = f"🔥 ÚJ LEAD: {lead_data.get('name', 'Ismeretlen')}"
    body = f"""
    Új érdeklődő érkezett a chatboton keresztül!
    
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
        print(f"✅ Email elküldve: {subject}")
    except Exception as e:
        print(f"❌ Email hiba: {e}")

def get_or_create_assistant():
    """Létrehozza vagy frissíti az Assistant-t."""
    global OPENAI_ASSISTANT_ID
    
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

    if OPENAI_ASSISTANT_ID:
        return OPENAI_ASSISTANT_ID
    
    tool_resources = {}
    if OPENAI_VECTOR_STORE_ID:
        tool_resources = {"file_search": {"vector_store_ids": [OPENAI_VECTOR_STORE_ID]}}

    print("⏳ Assistant létrehozása...")
    assistant = client.beta.assistants.create(
        name="Videmark Assistant",
        instructions=SYSTEM_PROMPT,
        model=OPENAI_MODEL,
        tools=tools,
        tool_resources=tool_resources
    )
    OPENAI_ASSISTANT_ID = assistant.id
    print(f"✅ Assistant létrehozva: {OPENAI_ASSISTANT_ID}")
    return OPENAI_ASSISTANT_ID

# ---------------- ENDPOINTS ----------------

@app.get("/")
def root():
    return {
        "service": "Videmark Chatbot V2.1",
        "model": OPENAI_MODEL,
        "assistant_id": OPENAI_ASSISTANT_ID,
        "vector_store": OPENAI_VECTOR_STORE_ID
    }

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq, x_chatbot_secret: str = Header(default="")):
    token = req.chatbot_secret or x_chatbot_secret
    if CHATBOT_SECRET and token != CHATBOT_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    assistant_id = get_or_create_assistant()
    
    thread_id = _thread_map.get(req.session_id)
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id
        _thread_map[req.session_id] = thread_id
    
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=req.message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        
        if run_status.status == 'completed':
            break
        elif run_status.status == 'requires_action':
            tool_outputs = []
            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                if tool_call.function.name == "save_lead":
                    args = json.loads(tool_call.function.arguments)
                    send_email_notification(args)
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": '{"success": true, "message": "Lead saved via email."}'
                    })
            if tool_outputs:
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
                )
            continue
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            return ChatResp(reply="Sajnos technikai hiba történt. Próbáld újra később.")
        
        time.sleep(0.5)

    # VÁLASZ TISZTÍTÁSA ÉS KINYERÉSE
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_msg = messages.data[0]
    
    reply_text = "..."
    if last_msg.role == "assistant":
        parts = []
        for content in last_msg.content:
            if content.type == 'text':
                val = content.text.value
                # REGEX: Annotációk törlése
                val = re.sub(r'【.*?】', '', val)
                parts.append(val)
        reply_text = "\n".join(parts)

    return ChatResp(reply=reply_text)

# --- ADMIN FELTÖLTÉS (Javítva az új klienshez) ---
@app.post("/admin/upload")
def admin_upload(file: UploadFile = File(...), x_admin_secret: str = Header(default="")):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET:
        raise HTTPException(401, "Admin secret needed")
    if not OPENAI_VECTOR_STORE_ID:
         raise HTTPException(400, "Nincs OPENAI_VECTOR_STORE_ID!")

    try:
        openai_file = client.files.create(
            file=(file.filename, file.file.read()),
            purpose="assistants"
        )
        client.beta.vector_stores.files.create(
            vector_store_id=OPENAI_VECTOR_STORE_ID,
            file_id=openai_file.id
        )
    except Exception as e:
        raise HTTPException(500, f"Hiba: {str(e)}")

    return {"status": "ok", "filename": file.filename}

@app.post("/admin/create_vector_store")
def create_vs(name: str = "VidemarkStore", x_admin_secret: str = Header(default="")):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET: raise HTTPException(401)
    vs = client.beta.vector_stores.create(name=name)
    return {"id": vs.id, "note": "Add Render ENV-hez: OPENAI_VECTOR_STORE_ID"}
