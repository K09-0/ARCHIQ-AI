"""Archiq AI — Генератор профессиональных архитектурных планов."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sqlite3
import os
import json
import io
import math
import base64
import re
from datetime import datetime
from pathlib import Path

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_OCR_MODEL = os.getenv("HF_OCR_MODEL", "microsoft/trocr-base-printed")
DB_PATH = os.getenv("DB_PATH", "norms.db")

# --- Lazy Gemini import ---
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- Output directory ---
OUTPUT_DIR = Path("/tmp/archiq-output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== AI ARCHITECT ====================

def ai_architect_prompt(project: dict) -> str:
    return f"""Ты — профессиональный архитектор. Сгенерируй архитектурный план здания.

Параметры проекта:
- Тип здания: {project.get('building_type', 'жилой дом')}
- Площадь: {project.get('area', 100)} м²
- Этажность: {project.get('floors', 1)}
- Количество комнат: {project.get('rooms', 3)}
- Участок: {project.get('site_width', 20)}м x {project.get('site_depth', 30)}м
- Дополнительные требования: {project.get('requirements', 'стандартная планировка')}

Ответь ТОЛЬКО JSON в этом формате (без markdown, без пояснений):
{{
  "building": {{
    "width": 12,
    "depth": 10,
    "floors": 1,
    "rooms": [
      {{"name": "Гостиная", "width": 5, "depth": 4, "x": 0, "y": 0}},
      {{"name": "Кухня", "width": 3, "depth": 4, "x": 5, "y": 0}},
      {{"name": "Спальня 1", "width": 3.5, "depth": 4, "x": 0, "y": 4}},
      {{"name": "Спальня 2", "width": 3.5, "depth": 4, "x": 3.5, "y": 4}},
      {{"name": "Ванная", "width": 2.5, "depth": 2, "x": 7, "y": 4}},
      {{"name": "Прихожая", "width": 5, "depth": 2, "x": 7, "y": 6}}
    ],
    "entrance": {{"x": 6, "y": 0}},
    "windows": [{{"room": "Гостиная", "x": 2.5, "y": 0, "width": 2}}, {{"room": "Спальня 1", "x": 1.75, "y": 8, "width": 1.5}}],
    "doors": [{{"from": "Прихожая", "to": "Гостиная", "x": 3, "y": 0}}, {{"from": "Прихожая", "to": "Кухня", "x": 6.5, "y": 0}}]
  }},
  "site": {{
    "width": {project.get('site_width', 20)},
    "depth": {project.get('site_depth', 30)},
    "building_x": 4,
    "building_y": 10,
    "parking": true,
    "garden": true,
    "driveway": true
  }},
  "compliance": [
    "СНиП 2.08.01-89: площадь жилых комнат не менее 8 м²",
    "СНиП 2.08.01-89: высота потолков не менее 2.5 м",
    "СНиП 21-01-97: ширина коридоров не менее 1.2 м"
  ],
  "description": "Краткое описание проекта"
}}"""

def generate_plan(project: dict) -> dict:
    """Generate architectural plan using AI."""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        # Fallback: generate a simple plan without AI
        return fallback_plan(project)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(ai_architect_prompt(project))
        text = response.text
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
        return fallback_plan(project)
    except Exception as e:
        print(f"AI error: {e}")
        return fallback_plan(project)

def fallback_plan(project: dict) -> dict:
    """Generate a simple plan without AI."""
    area = project.get('area', 100)
    rooms = project.get('rooms', 3)
    floors = project.get('floors', 1)
    floor_area = area / floors
    
    # Default room layout
    room_types = ["Гостиная", "Кухня", "Спальня 1"]
    if rooms > 3:
        room_types.extend([f"Спальня {i}" for i in range(2, rooms)])
    if rooms > 2:
        room_types.append("Ванная")
        room_types.append("Прихожая")
    if rooms > 4:
        room_types.append("Кабинет")
    
    # Simple grid layout
    building_width = math.sqrt(floor_area) * 1.2
    building_depth = floor_area / building_width
    rooms_list = []
    x, y = 0, 0
    row_height = building_depth / 2
    
    for i, room in enumerate(room_types[:rooms + 2]):
        room_width = building_width / (rooms // 2 + 1)
        rooms_list.append({
            "name": room,
            "width": round(room_width, 1),
            "depth": round(row_height, 1),
            "x": round(x, 1),
            "y": round(y, 1)
        })
        x += room_width
        if x >= building_width:
            x = 0
            y += row_height
    
    return {
        "building": {
            "width": round(building_width, 1),
            "depth": round(building_depth, 1),
            "floors": floors,
            "rooms": rooms_list,
            "entrance": {"x": round(building_width / 2, 1), "y": 0},
            "windows": [],
            "doors": []
        },
        "site": {
            "width": project.get('site_width', 20),
            "depth": project.get('site_depth', 30),
            "building_x": 3,
            "building_y": 5,
            "parking": True,
            "garden": True,
            "driveway": True
        },
        "compliance": [
            "СНиП 2.08.01-89: площадь жилых комнат не менее 8 м²",
            "СНиП 2.08.01-89: высота потолков не менее 2.5 м"
        ],
        "description": f"Проект {project.get('building_type', 'жилого дома')} площадью {area} м², {floors} этаж(ей), {rooms} комнат"
    }

# ==================== SVG GENERATOR ====================

def generate_site_plan_svg(plan: dict) -> str:
    """Generate SVG site plan."""
    site = plan.get("site", {})
    building = plan.get("building", {})
    rooms = building.get("rooms", [])
    
    site_w = site.get("width", 20)
    site_d = site.get("depth", 30)
    bldg_w = building.get("width", 10)
    bldg_d = building.get("depth", 8)
    bldg_x = site.get("building_x", 4)
    bldg_y = site.get("building_y", 8)
    
    scale = 20  # pixels per meter
    svg_w = site_w * scale + 100
    svg_h = site_d * scale + 100
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">
  <defs>
    <style>
      .site {{ fill: #f0f7f0; stroke: #2d5016; stroke-width: 2; }}
      .building {{ fill: #e8e8e8; stroke: #333; stroke-width: 2; }}
      .room {{ fill: #fff; stroke: #666; stroke-width: 1; }}
      .room-label {{ font-family: Arial, sans-serif; font-size: 11px; fill: #333; text-anchor: middle; }}
      .room-size {{ font-family: Arial, sans-serif; font-size: 9px; fill: #888; text-anchor: middle; }}
      .dimension {{ font-family: Arial, sans-serif; font-size: 10px; fill: #1a73e8; }}
      .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #333; }}
      .parking {{ fill: #ddd; stroke: #999; stroke-width: 1; }}
      .garden {{ fill: #e8f5e9; stroke: #4caf50; stroke-width: 1; }}
      .driveway {{ fill: #e0e0e0; stroke: #999; stroke-width: 1; stroke-dasharray: 4,2; }}
      .entrance {{ fill: #ff9800; stroke: #e65100; stroke-width: 1; }}
    </style>
  </defs>
  
  <!-- Title -->
  <text x="50" y="30" class="title">План участка</text>
  <text x="50" y="48" style="font-family:Arial;font-size:11px;fill:#666;">{plan.get("description", "Архитектурный план")}</text>
  
  <!-- Site boundary -->
  <rect x="50" y="60" width="{site_w * scale}" height="{site_d * scale}" class="site" rx="2"/>
  
  <!-- Garden area -->'''
    
    if site.get("garden"):
        svg += f'''
  <rect x="50" y="60" width="{site_w * scale}" height="{site_d * scale}" class="garden" rx="2" opacity="0.3"/>'''
    
    # Driveway
    if site.get("driveway"):
        svg += f'''
  <rect x="{50 + bldg_x * scale - 3}" y="60" width="6" height="{bldg_y * scale}" class="driveway"/>'''
    
    # Building
    svg += f'''
  <!-- Building -->
  <rect x="{50 + bldg_x * scale}" y="{60 + bldg_y * scale}" width="{bldg_w * scale}" height="{bldg_d * scale}" class="building"/>
  
  <!-- Rooms -->'''
    
    for room in rooms:
        rx = 50 + (bldg_x + room["x"]) * scale
        ry = 60 + (bldg_y + room["y"]) * scale
        rw = room["width"] * scale
        rd = room["depth"] * scale
        svg += f'''
  <rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" class="room"/>
  <text x="{rx + rw/2}" y="{ry + rd/2 - 5}" class="room-label">{room["name"]}</text>
  <text x="{rx + rw/2}" y="{ry + rd/2 + 10}" class="room-size">{room["width"]}×{room["depth"]}м</text>'''
    
    # Entrance
    entrance = building.get("entrance", {})
    if entrance:
        ex = 50 + (bldg_x + entrance.get("x", bldg_w/2)) * scale - 8
        ey = 60 + (bldg_y + entrance.get("y", 0)) * scale - 3
        svg += f'''
  <rect x="{ex}" y="{ey}" width="16" height="6" class="entrance" rx="1"/>
  <text x="{ex + 8}" y="{ey - 4}" style="font-size:9px;fill:#e65100;text-anchor:middle;">Вход</text>'''
    
    # Parking
    if site.get("parking"):
        px = 50 + (site_w - 6) * scale
        py = 60 + (site_d - 8) * scale
        svg += f'''
  <rect x="{px}" y="{py}" width="{5 * scale}" height="{6 * scale}" class="parking"/>
  <text x="{px + 2.5 * scale}" y="{py + 3 * scale}" style="font-family:Arial;font-size:11px;fill:#666;text-anchor:middle;">Парковка</text>'''
    
    # Dimensions
    svg += f'''
  <!-- Dimensions -->
  <text x="{50 + site_w * scale / 2}" y="{60 + site_d * scale + 25}" class="dimension" text-anchor="middle">{site_w}м</text>
  <text x="{30}" y="{60 + site_d * scale / 2}" class="dimension" text-anchor="middle" transform="rotate(-90,{30},{60 + site_d * scale / 2})">{site_d}м</text>
  <text x="{50 + bldg_x * scale + bldg_w * scale / 2}" y="{60 + bldg_y * scale - 8}" class="dimension" text-anchor="middle">{bldg_w}м × {bldg_d}м</text>
  
  <!-- Legend -->
  <rect x="{svg_w - 150}" y="{svg_h - 100}" width="130" height="80" fill="#1e293b" rx="5" opacity="0.9"/>
  <text x="{svg_w - 140}" y="{svg_h - 80}" style="font-family:Arial;font-size:11px;fill:#38bdf8;font-weight:bold;">Легенда</text>
  <rect x="{svg_w - 140}" y="{svg_h - 70}" width="15" height="10" fill="#e8e8e8" stroke="#333"/>
  <text x="{svg_w - 120}" y="{svg_h - 61}" style="font-family:Arial;font-size:10px;fill:#ccc;">Здание</text>
  <rect x="{svg_w - 140}" y="{svg_h - 50}" width="15" height="10" fill="#ff9800"/>
  <text x="{svg_w - 120}" y="{svg_h - 41}" style="font-family:Arial;font-size:10px;fill:#ccc;">Вход</text>
  <rect x="{svg_w - 140}" y="{svg_h - 30}" width="15" height="10" fill="#ddd" stroke="#999"/>
  <text x="{svg_w - 120}" y="{svg_h - 21}" style="font-family:Arial;font-size:10px;fill:#ccc;">Парковка</text>
</svg>'''
    
    return svg

def generate_floor_plan_svg(plan: dict) -> str:
    """Generate detailed floor plan SVG."""
    building = plan.get("building", {})
    rooms = building.get("rooms", [])
    bldg_w = building.get("width", 10)
    bldg_d = building.get("depth", 8)
    floors = building.get("floors", 1)
    
    scale = 40
    margin = 60
    svg_w = bldg_w * scale + margin * 2
    svg_h = bldg_d * scale + margin * 2 + 30
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">
  <defs>
    <style>
      .wall {{ stroke: #333; stroke-width: 3; fill: none; }}
      .wall-thin {{ stroke: #666; stroke-width: 1.5; fill: none; }}
      .room {{ fill: #fafafa; }}
      .room-label {{ font-family: Arial, sans-serif; font-size: 13px; fill: #333; text-anchor: middle; font-weight: bold; }}
      .room-area {{ font-family: Arial, sans-serif; font-size: 10px; fill: #888; text-anchor: middle; }}
      .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #333; }}
      .dim {{ font-family: Arial, sans-serif; font-size: 11px; fill: #1a73e8; }}
      .entrance {{ fill: #ff9800; stroke: #e65100; stroke-width: 2; }}
    </style>
  </defs>
  
  <text x="{margin}" y="25" class="title">План {floors} этажа</text>'''
    
    for room in rooms:
        rx = margin + room["x"] * scale
        ry = margin + 20 + room["y"] * scale
        rw = room["width"] * scale
        rd = room["depth"] * scale
        area = room["width"] * room["depth"]
        svg += f'''
  <rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" class="room" stroke="#333" stroke-width="2"/>
  <text x="{rx + rw/2}" y="{ry + rd/2 - 5}" class="room-label">{room["name"]}</text>
  <text x="{rx + rw/2}" y="{ry + rd/2 + 12}" class="room-area">{area:.1f} м²</text>'''
    
    # Entrance
    entrance = building.get("entrance", {})
    if entrance:
        ex = margin + entrance.get("x", bldg_w/2) * scale - 10
        ey = margin + 20 + entrance.get("y", 0) * scale - 5
        svg += f'''
  <rect x="{ex}" y="{ey}" width="20" height="10" class="entrance" rx="2"/>'''
    
    # Dimensions
    svg += f'''
  <text x="{margin + bldg_w * scale / 2}" y="{margin + 20 + bldg_d * scale + 20}" class="dim" text-anchor="middle">{bldg_w} м</text>
  <text x="{margin - 10}" y="{margin + 20 + bldg_d * scale / 2}" class="dim" text-anchor="middle" transform="rotate(-90,{margin - 10},{margin + 20 + bldg_d * scale / 2})">{bldg_d} м</text>
</svg>'''
    
    return svg

# ==================== SPECS GENERATOR ====================

def generate_specs(plan: dict) -> dict:
    """Generate project specifications."""
    building = plan.get("building", {})
    site = plan.get("site", {})
    rooms = building.get("rooms", [])
    
    total_area = building.get("width", 0) * building.get("depth", 0)
    living_area = sum(r.get("width", 0) * r.get("depth", 0) for r in rooms if "Спальня" in r.get("name", "") or "Гостиная" in r.get("name", ""))
    
    specs = {
        "total_area": round(total_area, 1),
        "living_area": round(living_area, 1),
        "floors": building.get("floors", 1),
        "rooms_count": len(rooms),
        "building_dimensions": f"{building.get('width', 0)} x {building.get('depth', 0)} м",
        "site_dimensions": f"{site.get('width', 0)} x {site.get('depth', 0)} м",
        "building_footprint": round(total_area, 1),
        "site_coverage": round(total_area / (site.get('width', 1) * site.get('depth', 1)) * 100, 1),
        "compliance": plan.get("compliance", []),
        "description": plan.get("description", "")
    }
    
    return specs

# ==================== APP ====================

app = FastAPI(title="Archiq AI", description="Генератор архитектурных планов")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Landing Page ---
LANDING_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Archiq AI — Генератор архитектурных планов</title>
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
<div class="logo">🏗️ Archiq AI</div>
<div class="subtitle">Генератор профессиональных архитектурных планов</div>
<div class="status"><span class="dot"></span> Сервис работает</div>
<div class="endpoints"><h2>API Endpoints</h2>
<div class="ep"><span class="method get">GET</span><span class="path">/health</span><span class="desc">Статус</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/generate-plan</span><span class="desc">Сгенерировать план</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/analyze-site-plan</span><span class="desc">Анализ плана участка</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/site-plan-svg?project_id=...</span><span class="desc">SVG план участка</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/floor-plan-svg?project_id=...</span><span class="desc">SVG поэтажный план</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/specs?project_id=...</span><span class="desc">Спецификация</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/projects</span><span class="desc">Список проектов</span></div>
</div>
<div class="footer"><p>GitHub: <a href="https://github.com/K09-0/ARCHIQ-AI" target="_blank">K09-0/ARCHIQ-AI</a></p></div>
</div></body></html>"""

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def root():
    return LANDING_PAGE

@app.get("/health")
def health_check():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY), "hf_configured": bool(HF_API_KEY), "service": "Archiq AI — Генератор архитектурных планов"}

# --- Database ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT,
        building_type TEXT,
        area REAL,
        floors INTEGER,
        rooms INTEGER,
        site_width REAL,
        site_depth REAL,
        requirements TEXT,
        plan_json TEXT,
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Project generation ---
@app.post("/generate-plan")
def generate_project(
    name: str = Form("Новый проект"),
    building_type: str = Form("жилой дом"),
    area: float = Form(100),
    floors: int = Form(1),
    rooms: int = Form(3),
    site_width: float = Form(20),
    site_depth: float = Form(30),
    requirements: str = Form(""),
    site_plan: UploadFile = File(None)
):
    project = {
        "name": name,
        "building_type": building_type,
        "area": area,
        "floors": floors,
        "rooms": rooms,
        "site_width": site_width,
        "site_depth": site_depth,
        "requirements": requirements
    }
    
    # If site plan uploaded, try to extract data
    if site_plan:
        project["site_plan_uploaded"] = True
    
    plan = generate_plan(project)
    specs = generate_specs(plan)
    project_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save plan SVGs
    site_svg = generate_site_plan_svg(plan)
    floor_svg = generate_floor_plan_svg(plan)
    (OUTPUT_DIR / f"{project_id}_site.svg").write_text(site_svg)
    (OUTPUT_DIR / f"{project_id}_floor.svg").write_text(floor_svg)
    
    # Save to DB
    conn = get_db()
    conn.execute(
        "INSERT INTO projects (id, name, building_type, area, floors, rooms, site_width, site_depth, requirements, plan_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (project_id, name, building_type, area, floors, rooms, site_width, site_depth, requirements, json.dumps(plan), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    
    return {
        "project_id": project_id,
        "plan": plan,
        "specs": specs,
        "site_plan_url": f"/site-plan-svg?project_id={project_id}",
        "floor_plan_url": f"/floor-plan-svg?project_id={project_id}"
    }

@app.get("/site-plan-svg")
def get_site_plan_svg(project_id: str):
    path = OUTPUT_DIR / f"{project_id}_site.svg"
    if path.exists():
        return HTMLResponse(content=path.read_text())
    return JSONResponse({"error": "Plan not found"}, status_code=404)

@app.get("/floor-plan-svg")
def get_floor_plan_svg(project_id: str):
    path = OUTPUT_DIR / f"{project_id}_floor.svg"
    if path.exists():
        return HTMLResponse(content=path.read_text())
    return JSONResponse({"error": "Plan not found"}, status_code=404)

@app.get("/specs")
def get_specs_endpoint(project_id: str):
    conn = get_db()
    row = conn.execute("SELECT plan_json FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    if row:
        plan = json.loads(row["plan_json"])
        return generate_specs(plan)
    return JSONResponse({"error": "Project not found"}, status_code=404)

@app.get("/projects")
def list_projects():
    conn = get_db()
    rows = conn.execute("SELECT id, name, building_type, area, floors, rooms, created_at FROM projects ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/analyze-site-plan")
async def analyze_site_plan(file: UploadFile = File(...)):
    """Analyze uploaded site plan image."""
    content = await file.read()
    if not GEMINI_API_KEY or not GEMINI_AVAILABLE:
        return {"error": "Gemini API key not configured. Site plan analysis requires Gemini.", "gemini_configured": False}
    
    try:
        import base64
        img_base64 = base64.b64encode(content).decode()
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Ты — архитектор-эксперт. Проанализируй план участка и определи: размеры участка, расположение зданий, подъездные пути, зоны озеленения. Ответь JSON: {\"site_width\": X, \"site_depth\": Y, \"buildings\": [{\"x\": X, \"y\": Y, \"width\": W, \"depth\": D}], \"features\": []}. Если не можешь определить точные размеры — дай оценку.",
            {"mime_type": "image/jpeg", "data": img_base64}
        ])
        
        text = response.text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
        return {"analysis": text}
    except Exception as e:
        return {"error": str(e)}
