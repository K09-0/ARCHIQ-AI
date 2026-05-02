from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
import google.generativeai as genai

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def gemini_chat(prompt: str, system_instruction: Optional[str] = None) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
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
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", str(result))
    if isinstance(result, dict):
        return result.get("generated_text", str(result))
    return str(result)

# --- AI Image Generation ---
def generate_ai_image(prompt: str, style: str = "photorealistic", seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate an image using available AI service (Gemini or Stable Diffusion)."""
    # Try using Gemini for image generation if available
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
            )
            enhanced_prompt = f"Create a {style} architectural visualization of: {prompt}. High quality, professional architectural photography, ultra-detailed."
            response = model.generate_content([enhanced_prompt])
            # Note: Gemini may not support image generation in all regions, fallback to text description
            return {
                "status": "success",
                "method": "gemini",
                "prompt": prompt,
                "description": response.text,
                "image_url": None,
            }
        except Exception as e:
            # Fallback: use Gemini to generate detailed description
            pass
    
    # Try Stable Diffusion API if configured
    if SDXL_API_URL and SDXL_API_URL != "not_configured":
        try:
            payload = {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted, ugly",
                "width": 1024,
                "height": 1024,
                "steps": 30,
            }
            if seed is not None:
                payload["seed"] = seed
            headers = {"Content-Type": "application/json"}
            if HF_API_KEY:
                headers["Authorization"] = f"Bearer {HF_API_KEY}"
            response = requests.post(
                SDXL_API_URL,
                json=payload,
                headers=headers,
                timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "method": "stable_diffusion",
                    "prompt": prompt,
                    "image_url": result.get("url"),
                    "images": result.get("images", []),
                }
        except Exception as e:
            pass
    
    # Use Gemini to generate a detailed visual description
    system = (
        "You are an expert architectural visualization specialist. "
        "Create a vivid, highly detailed visual description of the architectural scene. "
        "Focus on lighting, materials, textures, atmosphere, and photographic details. "
        "Describe as if you are directing a professional architectural photography shoot."
    )
    visual_prompt = (
        f"Create a {style} architectural visualization for: {prompt}\n"
        f"Describe in extreme detail: lighting conditions, material textures, "
        f"atmospheric effects, camera angle, lens choice, time of day, weather conditions, "
        f"and overall visual impact. Make it vivid enough for an artist to paint."
    )
    description = gemini_chat(visual_prompt, system_instruction=system)
    
    return {
        "status": "description_only",
        "method": "gemini_description",
        "prompt": prompt,
        "description": description,
        "image_url": None,
        "note": "Configure SDXL_API_URL or enable Gemini image generation for actual image output",
    }

# --- Database setup ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS norms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT,
            content TEXT,
            url TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            ocr_text TEXT,
            analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visualizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT,
            style TEXT,
            time_of_day TEXT,
            weather TEXT,
            lighting TEXT,
            materials TEXT,
            viewpoint TEXT,
            architectural_style TEXT,
            building_type TEXT,
            region TEXT,
            aspect_ratio TEXT,
            resolution TEXT,
            num_variants INTEGER,
            base_plan TEXT,
            prompt TEXT,
            seed INTEGER,
            generation_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visualization_3d (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_data TEXT,
            viewpoint TEXT,
            render_type TEXT,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Seed demo norms
    seed_data = [
        ("СНиП РК 2.02-2005", "Сейсмостойкое строительство", "Безопасность", "Требования к сейсмостойкости зданий и сооружений в Республике Казахстан.", "https://law.kz"),
        ("СНиП РК 2.03-2005", "Бетонные и железобетонные конструкции", "Конструкции", "Нормы проектирования бетонных и железобетонных конструкций.", "https://law.kz"),
        ("СНиП РК 2.04-2014", "Водоснабжение. Наружные сети и сооружения", "Инженерия", "Нормы проектирования наружных сетей водоснабжения.", "https://law.kz"),
        ("СНиП РК 3.01-2014", "Организация строительного производства", "Организация", "Требования к организации строительного производства.", "https://law.kz"),
        ("СНиП РК 3.02-2014", "Техника безопасности в строительстве", "Безопасность", "Требования безопасности при выполнении строительно-монтажных работ.", "https://law.kz"),
        ("СТ РК 1365-2013", "Здания жилые многоквартирные. Общие технические требования", "Жилье", "Общие технические требования к жилым зданиям.", "https://law.kz"),
        ("СНиП РК 2.01-2005", "Строительная климатология и геофизика", "Климат", "Нормы климатологических и геофизических параметров для строительства.", "https://law.kz"),
        ("СНиП РК 4.02-2014", "Тепловая защита зданий", "Энергия", "Требования к тепловой защите зданий и помещений.", "https://law.kz"),
    ]
    cursor.execute("SELECT COUNT(*) FROM norms")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO norms (code, title, category, content, url) VALUES (?,?,?,?,?)", seed_data)
    conn.commit()
    conn.close()

# --- Pydantic Models ---
class AskRequest(BaseModel):
    question: str
    context: Optional[str] = None
    use_norms: bool = True

class AskResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None

class NormItem(BaseModel):
    id: int
    code: str
    title: str
    category: Optional[str]
    url: Optional[str]

class VisualizationRequest(BaseModel):
    description: str
    style: str = "фотореализм"
    time_of_day: str = "день"
    weather: str = "ясно"
    lighting: str = "естественное"
    materials: List[str] = None
    viewpoint: str = "фасад"
    aspect_ratio: str = "16:9"
    resolution: str = "1024x1024"
    num_variants: int = 4
    base_plan: Optional[str] = None
    building_type: str = "жилое"
    architectural_style: str = "современный"
    region: str = "Казахстан"

class Visualization3DRequest(BaseModel):
    plan_data: str
    viewpoints: List[str] = None

class VisualizationStyleResponse(BaseModel):
    id: str
    name: str
    category: str
    description: str

# --- FastAPI App ---
app = FastAPI(title="ARCHIQ AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Style definitions ---
STYLES = {
    "photorealism": {"name": "Фотореализм", "category": "visualization", "description": "Гиперреалистичная визуализация"},
    "3d-vis": {"name": "3D-визуализация", "category": "visualization", "description": "CGI 3D рендер"},
    "arch-render": {"name": "Архитектурный рендер", "category": "visualization", "description": "Профессиональный архитектурный рендер"},
    "concept-art": {"name": "Концепт-арт", "category": "visualization", "description": "Архитектурный концепт-арт"},
    "modern": {"name": "Современный", "category": "architectural_style", "description": "Современная архитектура"},
    "minimalism": {"name": "Минимализм", "category": "architectural_style", "description": "Минималистичный стиль"},
    "high-tech": {"name": "Хай-тек", "category": "architectural_style", "description": "Футуристический хай-тек"},
    "industrial": {"name": "Индустриал", "category": "architectural_style", "description": "Индустриальный стиль"},
    "neoclassic": {"name": "Неоклассика", "category": "architectural_style", "description": "Неоклассический стиль"},
    "scandinavian": {"name": "Скандинавский", "category": "architectural_style", "description": "Скандинавский стиль"},
}

# --- Routes ---
@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/health")
def health_check():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY), "hf_configured": bool(HF_API_KEY), "sdsd_available": bool(SDXL_API_URL)}

@app.get("/norms", response_model=List[NormItem])
def list_norms(category: Optional[str] = None, search: Optional[str] = None):
    conn = get_db()
    cursor = conn.cursor()
    query = "SELECT id, code, title, category, url FROM norms WHERE 1=1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if search:
        query += " AND (code LIKE ? OR title LIKE ? OR content LIKE ?)"
        like = f"%{search}%"
        params.extend([like, like, like])
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.post("/ask", response_model=AskResponse)
def ask_ai(req: AskRequest):
    system = (
        "Ты — эксперт по строительным нормам и правилам Республики Казахстан. "
        "Отвечай на русском языке, четко и по делу. Приводи ссылки на нормативы, если они есть."
    )
    prompt = req.question
    sources = None
    if req.use_norms:
        conn = get_db()
        cursor = conn.cursor()
        like = f"%{req.question[:40]}%"
        cursor.execute(
            "SELECT code, title, content FROM norms WHERE title LIKE ? OR content LIKE ? LIMIT 5",
            (like, like)
        )
        norms = cursor.fetchall()
        conn.close()
        if norms:
            context_lines = []
            sources = []
            for row in norms:
                context_lines.append(f"{row['code']} — {row['title']}: {row['content']}")
                sources.append({"code": row["code"], "title": row["title"]})
            prompt = (
                f"Контекст из нормативной базы РК:\n"
                f"{chr(10).join(context_lines)}\n\n"
                f"Вопрос: {req.question}"
            )
    if req.context:
        prompt = f"Дополнительный контекст: {req.context}\n\n{prompt}"
    answer = gemini_chat(prompt, system_instruction=system)
    return {"answer": answer, "sources": sources}

@app.post("/ocr")
def ocr_document(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    text = hf_ocr(image_bytes)
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (filename, ocr_text) VALUES (?, ?)", (file.filename, text))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"document_id": doc_id, "filename": file.filename, "text": text}

@app.post("/analyze-document")
def analyze_document(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    try:
        extracted_text = hf_ocr(image_bytes)
    except Exception as e:
        extracted_text = ""
    system = (
        "Ты — эксперт по строительным нормам РК. Проанализируй документ, укажи нарушения, "
        "соответствие нормам и рекомендации. Отвечай на русском."
    )
    prompt = f"Текст из строительного документа (получен через OCR):\n{extracted_text}\n\nПроанализируй содержание."
    try:
        analysis = gemini_chat(prompt, system_instruction=system)
    except Exception as e:
        analysis = f"Ошибка анализа: {e}"
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (filename, ocr_text, analysis) VALUES (?, ?, ?)",
        (file.filename, extracted_text, analysis)
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"document_id": doc_id, "filename": file.filename, "text": extracted_text, "analysis": analysis}

@app.get("/documents")
def list_documents():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, created_at FROM documents ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- AI Visualization Endpoints ---

@app.post("/generate-visualization")
def generate_visualization(req: VisualizationRequest):
    """Генерация визуализации фасада/интерьера на основе текстового описания и плана."""
    
    # Нормализуем материалы
    materials = req.materials if req.materials else ["glass", "concrete"]
    materials_str = ", ".join(materials)
    
    # Формируем промпт
    prompt_parts = [
        req.description,
        f"Style: {req.style}, architectural style: {req.architectural_style}",
        f"Viewpoint: {req.viewpoint}",
        f"Time of day: {req.time_of_day}, weather: {req.weather}",
        f"Lighting: {req.lighting}",
        f"Materials: {materials_str}",
        f"Building type: {req.building_type}",
        f"Region: {req.region}",
    ]
    full_prompt = ". ".join(prompt_parts) + ". Professional architectural visualization."
    
    results = []
    for i in range(req.num_variants):
        seed = hash(int(datetime.now().timestamp())) + i
        result = generate_ai_image(
            prompt=full_prompt,
            style=req.style,
            seed=seed,
        )
        result["variant_id"] = i + 1
        result["seed"] = seed
        results.append(result)
    
    # Сохраняем в БД
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO visualizations (
            description, style, time_of_day, weather, lighting, materials,
            viewpoint, architectural_style, building_type, region,
            aspect_ratio, resolution, num_variants, base_plan, prompt, generation_result
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        req.description, req.style, req.time_of_day, req.weather, req.lighting,
        json.dumps(materials), req.viewpoint, req.architectural_style,
        req.building_type, req.region, req.aspect_ratio, req.resolution,
        req.num_variants, req.base_plan, full_prompt, json.dumps(results)
    ))
    viz_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "visualization_id": viz_id,
        "description": req.description,
        "style": req.style,
        "num_variants": req.num_variants,
        "results": results,
    }

@app.post("/generate-3d-view")
def generate_3d_view(req: Visualization3DRequest):
    """Генерация 3D-рендеров по чертежу здания."""
    
    viewpoints = req.viewpoints if req.viewpoints else ["north", "south", "east", "west", "perspective"]
    
    results = []
    conn = get_db()
    cursor = conn.cursor()
    
    for viewpoint in viewpoints:
        prompt = (
            f"Create a 3D architectural render from this building plan. "
            f"Viewpoint: {viewpoint}. {req.plan_data}. "
            f"Professional architectural visualization, realistic materials."
        )
        
        viz_result = generate_ai_image(prompt, style="3d vis")
        viz_result["viewpoint"] = viewpoint
        results.append(viz_result)
        
        cursor.execute("""
            INSERT INTO visualization_3d (plan_data, viewpoint, render_type, result)
            VALUES (?, ?, ?, ?)
        """, (req.plan_data, viewpoint, "3d_render", json.dumps(viz_result)))
    
    viz_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "visualization_id": viz_id,
        "viewpoints": viewpoints,
        "results": results,
    }

@app.get("/visualization-styles")
def get_visualization_styles():
    """Получение доступных стилей генерации."""
    style_list = []
    for key, style in STYLES.items():
        style_list.append({
            "id": key,
            "name": style["name"],
            "category": style["category"],
            "description": style["description"],
        })
    return {"styles": style_list}

@app.get("/visualizations")
def list_visualizations(limit: int = 20):
    """Список всех сгенерированных визуализаций."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, description, style, architectural_style, num_variants, 
               created_at, result_status
        FROM visualizations 
        ORDER BY created_at DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))