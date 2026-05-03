"""Archiq AI — AI-помощник по строительным нормам РК."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sqlite3
import os
import json
import requests
import base64
import io
import hashlib
from datetime import datetime

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_OCR_MODEL = os.getenv("HF_OCR_MODEL", "microsoft/trocr-base-printed")
SDXL_API_URL = os.getenv("SDXL_API_URL", "")
DB_PATH = os.getenv("DB_PATH", "norms.db")

# --- Gemini client wrapper ---
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def gemini_chat(prompt: str, system_instruction: Optional[str] = None) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=500, detail="google-generativeai not installed")
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

# --- HuggingFace OCR wrapper ---
def hf_ocr(image_bytes: bytes) -> str:
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HF_API_KEY not configured")
    api_url = f"https://api-inference.huggingface.co/models/{HF_OCR_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(api_url, headers=headers, data=image_bytes, timeout=120)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HF OCR error: {response.text}")
    result = response.json()
    if isinstance(result, list) and result:
        return result[0].get("generated_text", str(result))
    return str(result)

# --- AI Image Generation ---
def generate_ai_image(prompt: str, style: str = "photorealistic", seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate an image using available AI service."""
    if GEMINI_API_KEY and GEMINI_AVAILABLE:
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            enhanced_prompt = f"Create a {style} architectural visualization of: {prompt}. High quality, professional architectural photography, ultra-detailed."
            response = model.generate_content([enhanced_prompt])
            return {"status": "success", "method": "gemini", "prompt": prompt, "description": response.text, "image_url": None}
        except Exception:
            pass
    if SDXL_API_URL and SDXL_API_URL != "not_configured":
        try:
            payload = {"prompt": prompt, "negative_prompt": "blurry, low quality, distorted, ugly", "width": 1024, "height": 1024, "steps": 30}
            if seed is not None:
                payload["seed"] = seed
            headers = {"Content-Type": "application/json"}
            if HF_API_KEY:
                headers["Authorization"] = f"Bearer {HF_API_KEY}"
            response = requests.post(SDXL_API_URL, json=payload, headers=headers, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return {"status": "success", "method": "stable_diffusion", "prompt": prompt, "image_url": result.get("url"), "images": result.get("images", [])}
        except Exception:
            pass
    system = "You are an expert architectural visualization specialist. Create a vivid, highly detailed visual description of the architectural scene."
    visual_prompt = f"Create a {style} architectural visualization for: {prompt}\nDescribe in extreme detail: lighting conditions, material textures, atmospheric effects, camera angle, lens choice, time of day, weather conditions, and overall visual impact."
    if GEMINI_API_KEY and GEMINI_AVAILABLE:
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=system)
            response = model.generate_content(visual_prompt)
            return {"status": "success", "method": "gemini_text", "prompt": prompt, "description": response.text}
        except Exception:
            pass
    return {"status": "success", "method": "none", "prompt": prompt, "description": f"[{style}] {prompt}"}

# --- Database ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS norms (
        id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, title TEXT,
        category TEXT, url TEXT, content TEXT
    )""")
    cursor = conn.execute("SELECT COUNT(*) FROM norms")
    if cursor.fetchone()[0] == 0:
        samples = [
            ("SNiP 2.01-2005", "Климатические зоны РК", "KLI MAT", "", "Определение климатических зон для проектирования."),
            ("SNiP 2.01-2005 §3.1", "Температура наружного воздуха (зима)", "KLI MAT", "", "Расчетные температуры для отопительного периода. От -40 до -15°C."),
            ("SNiP 2.01-2005 §4.1", "Ветровая нагрузка", "KLI MAT", "", "Ветровое давление 0.35-0.85 кН/м²."),
            ("SNiP 3.02-2014 §4.1", "Высота перила лестницы", "BEZOPASNOST_TRUDA", "", "Минимальная высота перил 0.9-1.2 м."),
            ("SNiP 3.02-2014 §4.2", "Ширина лестничного марша жилое", "BEZOPASNOST_TRUDA", "", "Минимальная ширина 0.8-1.2 м."),
            ("SNiP 3.02-2014 §4.3", "Ширина лестничного марша нежилое", "BEZOPASNOST_TRUDA", "", "Минимальная ширина 1.2-2.0 м."),
            ("SNiP 4.01-2014 §2.1", "Высота потолка жилое", "JILIE_DOMA", "", "Минимальная высота 2.5 м."),
            ("SNiP 4.01-2014 §3.1", "Площадь жилой комнаты", "JILIE_DOMA", "", "Минимальная площадь 6-8 м²."),
            ("SNiP 5.01-2014 §1.1", "Пожарный проезд", "POZH_BEZOPASNOST", "", "Ширина проезда не менее 3.5 м."),
            ("SNiP 5.01-2014 §2.1", "Расстояние до соседнего здания", "POZH_BEZOPASNOST", "", "Минимальное расстояние 6-15 м."),
        ]
        conn.executemany("INSERT INTO norms (code, title, category, url, content) VALUES (?,?,?,?,?)", samples)
        conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Models ---
class AskRequest(BaseModel):
    question: str
    use_norms: Optional[bool] = True

class AskResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None

class NormItem(BaseModel):
    id: int
    code: str
    title: str
    category: str
    url: Optional[str] = None

class VisualizationRequest(BaseModel):
    prompt: str
    style: Optional[str] = "photorealistic"
    seed: Optional[int] = None

class Visualization3DRequest(BaseModel):
    prompt: str
    view_angle: Optional[str] = "front"

# --- Landing Page ---
LANDING_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Archiq AI</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem}
.container{max-width:700px;width:100%;text-align:center}
.logo{font-size:3rem;font-weight:800;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}
.subtitle{font-size:1.1rem;color:#94a3b8;margin-bottom:2rem}
.status{display:inline-flex;align-items:center;gap:.5rem;background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);padding:.5rem 1rem;border-radius:999px;margin-bottom:2rem}
.dot{width:8px;height:8px;border-radius:50%;background:#22c55e;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.endpoints{text-align:left;background:rgba(30,41,59,.8);border:1px solid #334155;border-radius:12px;padding:1.5rem}
.endpoints h2{font-size:1rem;color:#94a3b8;margin-bottom:1rem;text-transform:uppercase;letter-spacing:.05em}
.ep{display:flex;gap:1rem;padding:.75rem 0;border-bottom:1px solid #1e293b;align-items:baseline}
.ep:last-child{border-bottom:none}
.method{font-family:monospace;font-size:.85rem;font-weight:700;min-width:65px;padding:.2rem .5rem;border-radius:4px;text-align:center}
.get{background:rgba(56,189,248,.15);color:#38bdf8}
.post{background:rgba(168,85,247,.15);color:#a855f7}
.path{font-family:monospace;font-size:.9rem;color:#e2e8f0}
.desc{color:#64748b;font-size:.85rem;margin-left:auto}
.footer{margin-top:2rem;color:#475569;font-size:.8rem}
.footer a{color:#818cf8;text-decoration:none}
</style></head>
<body><div class="container">
<div class="logo">Archiq AI</div>
<div class="subtitle">AI-помощник по строительным нормам Республики Казахстан</div>
<div class="status"><span class="dot"></span> Сервис работает</div>
<div class="endpoints"><h2>API Endpoints</h2>
<div class="ep"><span class="method get">GET</span><span class="path">/health</span><span class="desc">Статус</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/norms?search=...</span><span class="desc">Поиск норм</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/ask</span><span class="desc">AI-ответ</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/ocr</span><span class="desc">OCR</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/analyze-document</span><span class="desc">OCR+AI</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/documents</span><span class="desc">Документы</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/generate-visualization</span><span class="desc">Визуализация</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/generate-3d-view</span><span class="desc">3D вид</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/visualization-styles</span><span class="desc">Стили</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/visualizations</span><span class="desc">Список</span></div>
</div>
<div class="footer"><p>GitHub: <a href="https://github.com/K09-0/ARCHIQ-AI" target="_blank">K09-0/ARCHIQ-AI</a></p></div>
</div></body></html>"""

# --- App ---
app = FastAPI(title="Archiq AI", description="AI-помощник по СНиП/ПБ РК")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def root():
    return LANDING_PAGE

@app.get("/health")
def health_check():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY), "hf_configured": bool(HF_API_KEY), "sdsd_available": bool(SDXL_API_URL)}

@app.get("/norms", response_model=List[NormItem])
def list_norms(category: Optional[str] = None, search: Optional[str] = None):
    conn = get_db()
    query = "SELECT id, code, title, category, url FROM norms WHERE 1=1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if search:
        query += " AND (code LIKE ? OR title LIKE ? OR content LIKE ?)"
        like = f"%{search}%"
        params.extend([like, like, like])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.post("/ask", response_model=AskResponse)
def ask_ai(req: AskRequest):
    system = "Ты — эксперт по строительным нормам и правилам Республики Казахстан. Отвечай на русском, четко и по делу. Приводи ссылки на нормативы."
    prompt = req.question
    sources = None
    if req.use_norms:
        conn = get_db()
        like = f"%{req.question[:40]}%"
        rows = conn.execute("SELECT code, title, content FROM norms WHERE title LIKE ? OR content LIKE ? LIMIT 5", (like, like)).fetchall()
        conn.close()
        if rows:
            sources = [dict(r) for r in rows]
            context = "\n".join([f"- {r['code']}: {r['title']}" for r in rows])
            prompt = f"Контекст из нормативной базы РК:\n{context}\n\nВопрос: {req.question}"
    answer = gemini_chat(prompt, system)
    return AskResponse(answer=answer, sources=sources)

@app.post("/ocr")
def ocr_endpoint(file: UploadFile = File(...)):
    content = file.file.read()
    text = hf_ocr(content)
    return {"filename": file.filename, "text": text}

@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...), question: str = Form("")):
    content = file.file.read()
    text = hf_ocr(content)
    if not question:
        question = "Проанализируй этот документ и выдели ключевую информацию."
    prompt = f"Документ (OCR):\n{text}\n\nЗадача: {question}"
    answer = gemini_chat(prompt)
    return {"extracted_text": text, "analysis": answer}

@app.get("/documents")
def list_documents():
    conn = get_db()
    rows = conn.execute("SELECT id, code, title FROM norms WHERE code IS NOT NULL").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/generate-visualization")
def generate_visualization(req: VisualizationRequest):
    result = generate_ai_image(req.prompt, req.style, req.seed)
    return result

@app.post("/generate-3d-view")
def generate_3d(req: Visualization3DRequest):
    prompt = f"3D architectural view, {req.view_angle} angle: {req.prompt}"
    result = generate_ai_image(prompt, "photorealistic")
    return result

@app.get("/visualization-styles")
def get_styles():
    return {
        "photorealistic": {"name": "Фотореализм", "category": "architectural_style"},
        "watercolor": {"name": "Акварель", "category": "artistic"},
        "sketch": {"name": "Эскиз", "category": "artistic"},
        "minimalism": {"name": "Минимализм", "category": "architectural_style"},
        "high-tech": {"name": "Хай-тек", "category": "architectural_style"},
        "industrial": {"name": "Индустриал", "category": "architectural_style"},
        "neoclassic": {"name": "Неоклассика", "category": "architectural_style"},
        "scandinavian": {"name": "Скандинавский", "category": "architectural_style"},
    }

@app.get("/visualizations")
def list_visualizations(limit: int = 20):
    return {"visualizations": [], "message": "No saved visualizations yet"}
