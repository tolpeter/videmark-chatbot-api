import os
from typing import Optional, Dict, Any, List
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------- CONFIG ---------
PROVIDER = os.getenv("AI_PROVIDER", "openai")  # "gemini" or "openai"

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# OpenAI (ha később kell)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# CORS (csak a te oldalaid)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://videmark.hu,https://www.videmark.hu").split(",")

SYSTEM_PROMPT = """
Te a Videmark weboldal hivatalos asszisztense vagy.
Feladatod:
- Ügyféltájékoztatás: drón videó/fotó, reklámvideó, short tartalom (TikTok/Reels/Shorts), fotózás, utómunka, social média támogatás, AI megoldások.
- Leadgyűjtés: ha a felhasználó ajánlatot kér vagy érdeklődik, kérd be: Név, Email, Telefonszám (opcionális), Helyszín, Határidő, Mire kell (videó/fotó/drón/short), mennyiség.
Stílus:
- magyarul, tömören, érthetően
- max 2 kérdés egyszerre
- javasolj következő lépést (pl. "kérsz gyors árajánlatot?")
"""

# --------- APP ---------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
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
    return {"ok": True, "service": "videmark-chatbot-api"}

def call_gemini(messages: List[Dict[str, str]]) -> str:
    if not GEMINI_API_KEY:
        return "Hiányzik a GEMINI_API_KEY a szerveren."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 500},
    }

    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        return f"Hiba a Gemini hívásnál: {r.status_code} – {r.text[:200]}"

    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "Nem tudtam választ generálni (Gemini válasz formátum)."

def call_openai(messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        return "Hiányzik az OPENAI_API_KEY a szerveren."

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    input_msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    for m in messages:
        input_msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})

    payload = {"model": OPENAI_MODEL, "input": input_msgs, "temperature": 0.4, "max_output_tokens": 500}

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        return f"Hiba az OpenAI hívásnál: {r.status_code} – {r.text[:200]}"

    data = r.json()
    try:
        if "output_text" in data and data["output_text"]:
            return data["output_text"]
        return data["output"][0]["content"][0]["text"]
    except Exception:
        return "Nem tudtam választ generálni (OpenAI válasz formátum)."

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    history = req.context or []
    messages = []
    for m in history[-10:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": str(m["content"])})

    messages.append({"role": "user", "content": req.message})

    if PROVIDER == "openai":
        reply = call_openai(messages)
    else:
        reply = call_gemini(messages)

    return ChatResp(reply=reply)
