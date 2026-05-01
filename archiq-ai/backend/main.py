from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import os
import json
import requests
import base64
import io

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_OCR_MODEL = os.getenv("HF_OCR_MODEL", "microsoft/trocr-base-printed")
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
    # Seed demo norms (Kazakhstan construction norms)
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

# --- FastAPI App ---
app = FastAPI(title="ARCHIQ AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- Routes ---
@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/health")
def health_check():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY), "hf_configured": bool(HF_API_KEY)}

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
    # Save to DB
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
    # 1. OCR via HF
    try:
        extracted_text = hf_ocr(image_bytes)
    except Exception as e:
        extracted_text = ""
    # 2. Analyze via Gemini
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
