import os
import time
import json
import re  # <--- FONTOS: Ez kell a tisztításhoz!
import smtplib
from email.mime.text import MIMEText
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, UploadFile, File, Header, HTTPException, Query
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

# Email config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "info@videmark.hu")

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://videmark.hu,https://www.videmark.hu").split(",") if o.strip()
]

SYSTEM_PROMPT = """
Te a Videmark weboldal hivatalos, barátságos asszisztense vagy.
Szolgáltatások: Drón videó/fotó, reklámvideó, short tartalom (TikTok/Reels), fotózás.

Feladatod:
1. Válaszolj kérdésekre a tudásbázis (fájlok) alapján.
2. Formázd a válaszaidat szépen: használj felsorolásokat (bullet points), új sorokat és emeld ki félkövérrel a lényeget (árakat).
3. LEAD GYŰJTÉS: Ha az ügyfél érdeklődik, kérd el: Név, Email, Telefonszám, Projekt leírása. Ha megkaptad, hívd a 'save_lead' funkciót.

Stílus: Magyar, tegező, segítőkész.
""".strip()

client = OpenAI(api_key=OPENAI_API_KEY)
_thread_map: Dict[str, str] = {}

app = FastAPI(title="Videmark Chatbot API v2.2")

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

# ---------------- SEGÉDFÜGGVÉNYEK ----------------

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
        print(f"✅ Email elküldve.")
    except Exception as e:
        print(f"❌ Email hiba: {e}")

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
        name="Videmark Assistant",
        instructions=SYSTEM_PROMPT,
        model=OPENAI_MODEL,
        tools=tools,
        tool_resources=tool_resources
    )
    OPENAI_ASSISTANT_ID = asst.id
    return OPENAI_ASSISTANT_ID

# ---------------- ENDPOINTS ----------------

@app.get("/")
def root():
    return {"status": "ok", "model": OPENAI_MODEL}

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
                    args = json.loads(tool_call.function.arguments)
                    send_email_notification(args)
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": '{"success": true}'
                    })
            if tool_outputs:
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
                )
            continue
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            return ChatResp(reply="Hiba történt. Próbáld újra.")
        time.sleep(0.5)

    # VÁLASZ TISZTÍTÁSA (REGEX)
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_msg = messages.data[0]
    
    reply_text = ""
    if last_msg.role == "assistant":
        parts = []
        for content in last_msg.content:
            if content.type == 'text':
                val = content.text.value
                # 1. Annotációk törlése (【4:0†source】)
                val = re.sub(r'【.*?】', '', val)
                parts.append(val)
        reply_text = "\n".join(parts)

    return ChatResp(reply=reply_text)

@app.post("/admin/upload")
def admin_upload(file: UploadFile = File(...), x_admin_secret: str = Header(default="")):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET: raise HTTPException(401)
    if not OPENAI_VECTOR_STORE_ID: raise HTTPException(400, "Nincs Vector Store ID")

    f = client.files.create(file=(file.filename, file.file.read()), purpose="assistants")
    client.beta.vector_stores.files.create(vector_store_id=OPENAI_VECTOR_STORE_ID, file_id=f.id)
    return {"status": "ok"}

@app.post("/admin/create_vector_store")
def create_vs(name: str = "Store", x_admin_secret: str = Header(default="")):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET: raise HTTPException(401)
    vs = client.beta.vector_stores.create(name=name)
    return {"id": vs.id}
