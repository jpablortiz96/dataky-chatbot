import os
import joblib
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.metrics.pairwise import cosine_similarity

from pydantic import BaseModel
from google import genai

import time
import re

# =========================================================
# 0) Cargar variables de entorno (.env)
# =========================================================
load_dotenv()

# -------------------------
# Branding / CTA (opcional pero recomendado)
# -------------------------
SITE_URL = os.getenv("SITE_URL", "https://dataky.net").strip()
BRAND_NAME = os.getenv("BRAND_NAME", "Dataky").strip()
CTA_URL = os.getenv("CTA_URL", f"{SITE_URL}/contact").strip()
HUMAN_WHATSAPP_URL = os.getenv("HUMAN_WHATSAPP_URL", "").strip()

# -------------------------
# Gemini
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_CLIENT = None



# Carpeta donde está el índice (por defecto: data/)
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.joblib")

# Cache global en RAM (cargamos index.joblib 1 vez)
INDEX: Dict[str, Any] | None = None

# =========================================================
# Memoria en RAM por sesión
# =========================================================
SESSIONS: Dict[str, Dict[str, Any]] = {}

MAX_HISTORY = 8          # máximo mensajes guardados por sesión (user+bot)
SESSION_TTL_SEC = 60*60  # 1 hora sin actividad -> se limpia


app = FastAPI(title="Dataky asistente")

# =========================================================
# 1) Cargar índice desde disco a memoria (RAM)
# =========================================================
def load_index() -> Dict[str, Any]:
    """
    Carga el índice TF-IDF (joblib) a memoria.
    Esto hace que el retrieval sea MUY rápido y evita reentrenar cada vez.
    """
    global INDEX

    if not os.path.exists(INDEX_PATH):
        INDEX = None
        return {
            "loaded": False,
            "error": f"No existe {INDEX_PATH}. Ejecuta 04_build_tfidf_index.py primero.",
        }

    INDEX = joblib.load(INDEX_PATH)

    docs_count = len(INDEX.get("docs", []))
    return {"loaded": True, "docs_count": docs_count, "index_path": INDEX_PATH}

def init_gemini():
    """
    Inicializa el cliente de Gemini para poder llamar al modelo.
    """
    global GEMINI_CLIENT

    if not GEMINI_API_KEY:
        GEMINI_CLIENT = None
        return {"gemini_ready": False, "error": "Falta GEMINI_API_KEY en .env"}

    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        return {"gemini_ready": True}
    except Exception as e:
        GEMINI_CLIENT = None
        return {"gemini_ready": False, "error": str(e)}


# =========================================================
# 2) Retrieval Top-K (búsqueda)
# =========================================================
def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Devuelve los top-k documentos más relevantes para una query:

    Pasos:
    - vectorizer.transform(query)  -> vector TF-IDF de la pregunta
    - cosine_similarity vs X       -> ranking de similitud
    - seleccionamos top-K docs
    """
    # Si no está cargado, intenta cargarlo
    if INDEX is None:
        info = load_index()
        if not info.get("loaded"):
            return []
        

    vectorizer = INDEX["vectorizer"]
    X = INDEX["X"]
    docs = INDEX["docs"]


    # Convertimos la pregunta a TF-IDF
    q_vec = vectorizer.transform([query])  # shape: (1, n_features)

    # Similaridad coseno contra todos los docs
    sims = cosine_similarity(q_vec, X).flatten()  # shape: (n_docs,)

    # Tomamos top-k
    top_idx = sims.argsort()[::-1][:k]

    results: List[Dict[str, Any]] = []
    for idx in top_idx:
        d = docs[idx]
        results.append(
            {
                "score": float(sims[idx]),
                "url": d.get("url", ""),
                "title": d.get("title", ""),
                "text_preview": d.get("text", ""),
            }
        )

    return results

def _now() -> float:
    return time.time()

def get_or_create_session(session_id: str | None) -> str:
    """
    Retorna un session_id válido y asegura que exista en memoria.
    """
    if not session_id:
        # Si el front no envía session_id, generamos uno simple
        session_id = str(int(_now() * 1000))

    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "history": [],             # lista de {"role": "user"/"assistant", "content": "..."}
            "last_product_url": None,  # último producto detectado
            "last_sources": [],        # últimas fuentes (urls)
            "last_seen": _now(),
        }

    SESSIONS[session_id]["last_seen"] = _now()
    return session_id

def add_to_history(session_id: str, role: str, content: str):
    """
    Guarda mensajes y mantiene el historial corto.
    """
    h = SESSIONS[session_id]["history"]
    h.append({"role": role, "content": content})

    # Recortamos historial para no crecer infinito
    if len(h) > MAX_HISTORY:
        SESSIONS[session_id]["history"] = h[-MAX_HISTORY:]


def history_block(session_id: str) -> str:
    """
    Convierte historial reciente en un bloque que se inyecta al prompt.
    """
    h = SESSIONS[session_id]["history"]
    if not h:
        return ""

    lines = []
    for m in h[-MAX_HISTORY:]:
        role = "Usuario" if m["role"] == "user" else "Asistente"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def cleanup_sessions():
    """
    Limpia sesiones viejas (TTL). Se puede llamar en cada request (barato).
    """
    now = _now()
    to_delete = []
    for sid, data in SESSIONS.items():
        if now - data.get("last_seen", now) > SESSION_TTL_SEC:
            to_delete.append(sid)
    for sid in to_delete:
        del SESSIONS[sid]

def detect_purchase_intent(text: str) -> bool:
    """
    Detecta intención de compra del usuario (follow-up).
    """
    t = text.lower()
    keywords = [
        "link de compra", "enlace de compra", "comprar", "compra",
        "checkout", "pagar", "carrito", "precio", "quiero el link"
    ]
    return any(k in t for k in keywords)


def pick_product_url(sources: List[str]) -> str | None:
    """
    Elige el mejor link de compra desde fuentes.
    Preferimos URLs con /product/ o /producto/
    """
    for u in sources:
        uu = u.lower()
        if "/product/" in uu or "/producto/" in uu:
            return u
    return sources[0] if sources else None


def build_context_block(hits: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    """
    Convierte el top-K retrieval en un bloque de texto.
    Esto es lo que se inyecta al LLM como 'fuentes'.
    """
    blocks = []
    for i, h in enumerate(hits, start=1):
        text = (h.get("text", "") or "")[:max_chars]
        blocks.append(
            f"[Fuente {i}]\n"
            f"Título: {h.get('title','')}\n"
            f"URL: {h.get('url','')}\n"
            f"Contenido: {text}\n"
        )
    return "\n\n".join(blocks).strip()


def build_prompt(question: str, context_block: str, history: str = "") -> str:
    """
    Prompt con reglas anti-alucinación.
    La clave es obligar al modelo a usar SOLO 'FUENTES'.
    """
    return f"""
Eres el asistente de {BRAND_NAME} enfocado en VENTAS y SOPORTE.

REGLAS:
1) Responde SOLO con la información contenida en FUENTES.
2) NO inventes precios, enlaces, productos ni características.
3) Si no hay suficiente info en FUENTES, di: "No lo veo en el sitio aún" y pide 1 dato extra.
4) Responde en español, claro, directo y útil.

HISTORIAL RECIENTE (para mantener coherencia):
{history if history else "No hay historial"}

FUENTES:
{context_block}

Pregunta del usuario:
{question}

Termina SIEMPRE con:
👉 Contacto: {CTA_URL}
{"💬 WhatsApp: " + HUMAN_WHATSAPP_URL if HUMAN_WHATSAPP_URL else ""}
""".strip()


def ask_gemini(prompt: str) -> str:
    """
    Llama al modelo Gemini y devuelve el texto.
    """
    if GEMINI_CLIENT is None:
        init_gemini()

    if GEMINI_CLIENT is None:
        return "Gemini no está configurado. Revisa GEMINI_API_KEY en tu .env."

    resp = GEMINI_CLIENT.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return (resp.text or "").strip() or "No pude generar una respuesta en este momento."


# =========================================================
# 3) Startup: cargar el índice apenas inicia FastAPI
# =========================================================
@app.on_event("startup")
def on_startup():
    load_index()
    init_gemini()


# =========================================================
# 4) Endpoints
# =========================================================
@app.get("/", response_class=HTMLResponse)
def home():
    """
    Página simple para confirmar que el backend está vivo.
    """
    loaded = INDEX is not None
    return f"""
    <html>
      <body style="font-family: Arial; padding: 24px;">
        <h2>✅ Backend Chatbot IA </h2>
        <p><b>Index cargado:</b> {"SI ✅" if loaded else "NO ❌"}</p>
        <p><b>Ruta del índice:</b> {INDEX_PATH}</p>
        <p>Prueba retrieval aquí:</p>
        <code>/admin/test_retrieve?q=pack+tiktok&k=5</code>
      </body>
    </html>
    """

@app.get("/health")
def health():
    """
    Health check (muy útil para Render).
    """
    return {
        "ok": True, 
        "index_loaded": INDEX is not None, 
        "index_path": INDEX_PATH,
        "gemini_ready": GEMINI_CLIENT is not None,
        "sessions_in_ram": len(SESSIONS)
        }


class ChatIn(BaseModel):
    message: str
    k: int = 4
    session_id: str | None = None


@app.post("/api/chat")
def chat(payload: ChatIn):
    """
    Flujo RAG con memoria por sesión:
    - guarda historial
    - si pide link de compra y ya hay producto -> responde directo
    - si no -> retrieval + gemini
    """
    cleanup_sessions()

    question = payload.message.strip()
    k = payload.k

    # 1) asegurar sesión
    sid = get_or_create_session(payload.session_id)

    # 2) guardar mensaje del usuario
    add_to_history(sid, "user", question)

    # 3) si intención es compra y ya tenemos producto -> respuesta directa
    if detect_purchase_intent(question):
        last_product = SESSIONS[sid].get("last_product_url")
        if last_product:
            answer = (
                f"¡Claro! ✅ Aquí tienes el link de compra:\n\n"
                f"{last_product}\n\n"
                f"👉 Contacto: {CTA_URL}\n" +
                (f"💬 WhatsApp: {HUMAN_WHATSAPP_URL}" if HUMAN_WHATSAPP_URL else "")
            )
            add_to_history(sid, "assistant", answer)
            return {"answer": answer, "sources": [last_product]}

    # 4) retrieval normal
    hits = retrieve(question, k=k)

    if not hits:
        answer = (
            "No encontré fuentes suficientes en el índice todavía 😅\n"
            "Revisa que exista data/index.joblib o vuelve a indexar.\n\n"
            f"👉 Contacto: {CTA_URL}\n" +
            (f"💬 WhatsApp: {HUMAN_WHATSAPP_URL}" if HUMAN_WHATSAPP_URL else "")
        )
        add_to_history(sid, "assistant", answer)
        return {"answer": answer, "sources": []}

    # 5) fuentes del retrieval
    sources = [h.get("url", "") for h in hits if h.get("url", "").startswith("http")]

    # 6) guardar last_sources y detectar producto
    SESSIONS[sid]["last_sources"] = sources
    product_url = pick_product_url(sources)
    if product_url:
        SESSIONS[sid]["last_product_url"] = product_url

    # 7) armar prompt con historial
    context_block = build_context_block(hits)
    hist = history_block(sid)

    prompt = build_prompt(question, context_block, history=hist)
    answer = ask_gemini(prompt)

    # 8) guardar respuesta del bot en historial
    add_to_history(sid, "assistant", answer)

    return {"answer": answer, "sources": sources, "session_id": sid}




@app.get("/admin/test_retrieve")
def test_retrieve(q: str, k: int = 4):
    """
    Prueba de retrieval SIN IA:
      /admin/test_retrieve?q=pack%20tiktok&k=5
    """
    results = retrieve(q, k=k)
    return {"query": q, "k": k, "results": results}


@app.post("/admin/reload_index")
def reload_index():
    """
    Recarga el índice desde disco sin reiniciar el servidor.
    Útil si vuelves a generar data/index.joblib.
    """
    info = load_index()
    return {"ok": True, **info}


@app.get("/admin/test_chat")
def test_chat(q: str):
    hits = retrieve(q, k=4)
    context_block = build_context_block(hits)
    prompt = build_prompt(q, context_block)
    answer = ask_gemini(prompt)
    return {"question": q, "answer": answer}


@app.get("/widget", response_class=HTMLResponse)
def widget():
    """
    Widget embebible con iframe para WordPress.
    """
    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{BRAND_NAME} Chat</title>
  <style>
    :root {{
      --bg: #0b1220;
      --card: rgba(255,255,255,0.06);
      --border: rgba(255,255,255,0.12);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --shadow: 0 24px 80px rgba(0,0,0,0.55);
      --radius: 18px;
      --btn: linear-gradient(135deg, rgba(29,135,234,1), rgba(103,190,217,1));
      --user: rgba(254,223,64,0.18);     /* amarillo suave */
      --bot: rgba(103,190,217,0.18);     /* azul suave */
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      height: 100vh;
      background: radial-gradient(1200px 500px at 20% 10%, rgba(29,135,234,0.25), transparent 60%),
                  radial-gradient(900px 500px at 80% 30%, rgba(254,223,64,0.18), transparent 55%),
                  radial-gradient(700px 500px at 60% 90%, rgba(103,190,217,0.18), transparent 55%),
                  var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }}

    .wrap {{
      width: 100%;
      max-width: 420px;
      height: 620px;
      border-radius: var(--radius);
      background: var(--card);
      border: 1px solid var(--border);
      backdrop-filter: blur(14px);
      box-shadow: var(--shadow);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
    }}

    .header {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: rgba(254,223,64,0.95);
      box-shadow: 0 0 0 6px rgba(254,223,64,0.12);
    }}

    .title {{
      display: flex;
      flex-direction: column;
      line-height: 1.15;
    }}
    .title b {{
      font-size: 14px;
      letter-spacing: 0.2px;
    }}
    .title span {{
      font-size: 12px;
      color: var(--muted);
    }}

    .chat {{
      flex: 1;
      padding: 14px 14px 6px;
      overflow-y: auto;
    }}

    .msg {{
      max-width: 86%;
      padding: 10px 12px;
      border-radius: 14px;
      margin: 8px 0;
      border: 1px solid var(--border);
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 13.5px;
      line-height: 1.35;
    }}

    .bot {{
      background: var(--bot);
      border-top-left-radius: 6px;
    }}
    .user {{
      margin-left: auto;
      background: var(--user);
      border-top-right-radius: 6px;
    }}

    .sources {{
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    .sources a {{
      color: rgba(103,190,217,0.95);
      text-decoration: none;
    }}
    .sources a:hover {{
      text-decoration: underline;
    }}

    .chips {{
      display: flex;
      gap: 8px;
      padding: 10px 12px;
      flex-wrap: wrap;
      border-top: 1px solid rgba(255,255,255,0.06);
      background: rgba(0,0,0,0.10);
    }}
    .chip {{
      font-size: 12px;
      color: var(--text);
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      cursor: pointer;
      user-select: none;
    }}
    .chip:hover {{
      background: rgba(255,255,255,0.08);
    }}

    .inputbar {{
      border-top: 1px solid var(--border);
      padding: 10px 12px;
      display: flex;
      gap: 10px;
      align-items: center;
      background: rgba(0,0,0,0.18);
    }}

    textarea {{
      flex: 1;
      resize: none;
      height: 42px;
      max-height: 120px;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.05);
      color: var(--text);
      outline: none;
      font-size: 13px;
    }}
    textarea::placeholder {{
      color: rgba(255,255,255,0.45);
    }}

    button {{
      width: 44px;
      height: 44px;
      border-radius: 14px;
      border: 0;
      cursor: pointer;
      background: var(--btn);
      color: #06111f;
      font-weight: 800;
      box-shadow: 0 12px 30px rgba(29,135,234,0.20);
    }}

    .typing {{
      font-size: 12px;
      color: var(--muted);
      padding: 0 14px 10px;
    }}

    @media (max-width: 480px) {{
      .wrap {{
        height: 78vh;
        max-width: 96vw;
      }}
    }}
  </style>
</head>

<body>
  <div class="wrap">
    <div class="header">
      <div class="dot"></div>
      <div class="title">
        <b>{BRAND_NAME} Assistant</b>
        <span>Respuestas rápidas • Ventas • Soporte</span>
      </div>
    </div>

    <div id="chat" class="chat"></div>

    <div class="chips">
      <div class="chip" onclick="sendChip('¿Qué servicios ofrecen?')">Servicios</div>
      <div class="chip" onclick="sendChip('¿Tienen packs o productos?')">Packs</div>
      <div class="chip" onclick="sendChip('Quiero el link de compra')">Comprar</div>
      <div class="chip" onclick="sendChip('¿Cómo los contacto?')">Contacto</div>
    </div>

    <div id="typing" class="typing" style="display:none;">Escribiendo…</div>

    <div class="inputbar">
      <textarea id="msg" placeholder="Escribe tu pregunta..."></textarea>
      <button id="btn" title="Enviar">➤</button>
    </div>
  </div>

<script>
  // ----------------------------
  // Session ID: para coherencia
  // ----------------------------
  const SESSION_KEY = "dataky_session_id";
  let sessionId = localStorage.getItem(SESSION_KEY);

  if (!sessionId) {{
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }}

  const chat = document.getElementById("chat");
  const msg = document.getElementById("msg");
  const btn = document.getElementById("btn");
  const typing = document.getElementById("typing");

  function addMessage(text, who="bot", sources=[]) {{
    const bubble = document.createElement("div");
    bubble.className = "msg " + who;
    bubble.textContent = text;
    chat.appendChild(bubble);

    if (who === "bot" && sources && sources.length > 0) {{
      const s = document.createElement("div");
      s.className = "sources";
      s.innerHTML = "Fuentes: " + sources.slice(0,2).map(u => `<a href="${{u}}" target="_blank">${{u}}</a>`).join(" • ");
      chat.appendChild(s);
    }}

    chat.scrollTop = chat.scrollHeight;
  }}

  async function sendMessage(text) {{
    if (!text || !text.trim()) return;

    addMessage(text, "user");
    msg.value = "";

    typing.style.display = "block";
    btn.disabled = true;

    try {{
      const res = await fetch("/api/chat", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          message: text,
          k: 4,
          session_id: sessionId
        }})
      }});

      const data = await res.json();
      typing.style.display = "none";
      btn.disabled = false;

      addMessage(data.answer || "No pude responder en este momento.", "bot", data.sources || []);
    }} catch (err) {{
      typing.style.display = "none";
      btn.disabled = false;
      addMessage("Ups… hubo un error conectando con el servidor 😅", "bot");
    }}
  }}

  function sendChip(text) {{
    sendMessage(text);
  }}

  btn.addEventListener("click", () => sendMessage(msg.value));

  msg.addEventListener("keydown", (e) => {{
    if (e.key === "Enter" && !e.shiftKey) {{
      e.preventDefault();
      sendMessage(msg.value);
    }}
  }});

  // Mensaje inicial
  addMessage("Hola 👋 Soy el asistente de {BRAND_NAME}. ¿En qué te puedo ayudar hoy?");
</script>

</body>
</html>
""".strip()


