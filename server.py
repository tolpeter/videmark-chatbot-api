import os
import time
import json
import re
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

app = FastAPI(title="Videmark Chatbot API v4.0")

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
    if not text: return ""

    # 1. Hivatkozások tisztítása
    text = re.sub(r'【.*?】', '', text)

    # 2. Címsorok (### Cím) -> <h3>Cím</h3>
    # A szöveg közepén lévő ###-ket is kezeli
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

        # Lista kezelés (- elem)
        if line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html_lines.append('<ul style="margin: 5px 0 10px 20px; padding: 0;">')
                in_list = True
            content = line[2:]
            # Félkövér a listán belül
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
            html_lines.append(f'<li style="margin-bottom: 5px; list-style: disc;">{content}</li>')
        
        # Címsor kezelés (###)
        elif line.startswith("###"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            content = line.replace("###", "").strip()
            html_lines.append(f'<h3 style="margin: 15px 0 5px 0; font-size: 16px; border-bottom: 1px solid rgba(255,255,255,0.2);">{content}</h3>')
        
        # Sima szöveg
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            # Félkövér
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
        print(f"✅ Email elküldve.")
    except Exception as e:
        print(f"❌ Email hiba: {e}")

def get_or_create_assistant():
    global OPENAI_ASSISTANT_ID
    # Ha már van ID, használjuk (gyorsabb)
    if OPENAI_ASSISTANT_ID: return OPENAI_ASSISTANT_ID
    
    # Eszközök
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

    # Assistant létrehozása (minden induláskor frissíti a promptot!)
    # Megjegyzés: Élesben érdemes lehet update-elni a meglévőt, de így a legegyszerűbb, hogy érvényesüljön az új prompt.
    asst = client.beta.assistants.create(
        name="Videmark Assistant V4",
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
    return {"status": "ok", "mode": "HTML server-side rendering v4"}

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
                    except:
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

    # VÁLASZ LEKÉRÉSE ÉS FORMÁZÁSA
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_msg = messages.data[0]
    
    reply_text = ""
    if last_msg.role == "assistant":
        raw_parts = []
        for content in last_msg.content:
            if content.type == 'text':
                raw_parts.append(content.text.value)
        
        raw_text = "\n".join(raw_parts)
        
        # ITT A LÉNYEG: A szerver alakítja át HTML-lé!
        reply_text = format_to_html(raw_text)

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
