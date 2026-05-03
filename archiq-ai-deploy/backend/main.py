#!/usr/bin/env python3
"""Archiq AI v4 — Профессиональный генератор архитектурных планов."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from typing import Optional, List, Dict, Any
import sqlite3, os, json, math, random, base64, re, io
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from site_analyzer import analyze_site_plan, analysis_to_plan_params
from ai_architect import (
    ai_architect_v2, check_snip, generate_project_description,
    ai_architect_generate_description, SnipCheckResult
)

# ==================== CONFIG ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
DB_PATH = os.getenv("DB_PATH", "norms.db")
OUTPUT = Path("/tmp/archiq-output")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ==================== ROOM DEFINITIONS ====================
@dataclass
class Room:
    name: str
    min_area: float; max_area: float
    min_w: float; min_d: float
    wall: str = "any"  # N=bottom(y=0), S=top(y=bd), E=right(x=bw), W=left(x=0)
    is_wet: bool = False; has_window: bool = True
    adj_to: List[str] = field(default_factory=list)
    w: float = 0; d: float = 0; x: float = 0; y: float = 0

def catalog(btype: str, area: float, n: int) -> List[Room]:
    r = []
    if btype in ("жилой дом", "дача"):
        r.append(Room("Гостиная", max(18,area*0.22), 35, 4.0, 3.5, "S", adj_to=["Прихожая","Кухня"]))
        r.append(Room("Кухня", max(9,area*0.12), 18, 2.8, 2.5, "N", True, adj_to=["Гостиная","Прихожая"]))
        r.append(Room("Прихожая", 5, 12, 2.0, 1.8, "any", False, False, adj_to=["Гостиная","Коридор"]))
        r.append(Room("Ванная", 3.5, 7, 1.8, 1.8, "N", True, True, adj_to=["Коридор"]))
        r.append(Room("Коридор", 4, 10, 1.2, 2.0, "any", False, False))
        beds = max(1, n - 2)
        for i in range(beds):
            nm = "Спальня" if i==0 else (f"Детская" if i==beds-1 and n>4 else f"Спальня {i+1}")
            r.append(Room(nm, 10, 18, 3.0, 2.8, "S" if "Спальня" in nm else "E", adj_to=["Коридор"]))
        if area > 120:
            r.append(Room("Кабинет", 8, 14, 2.5, 2.5, "E", adj_to=["Коридор"]))
        if area > 150:
            r.append(Room("Гардеробная", 4, 8, 1.8, 1.8, "any", False, False))
            r.append(Room("Кладовая", 2, 5, 1.2, 1.2, "any", False, False))
        if area > 100:
            r.append(Room("Туалет", 1.5, 3, 1.2, 1.2, "any", True, False, adj_to=["Коридор"]))
    elif btype == "баня":
        r = [Room("Парная",5,9,2.2,2.0,"N",False,False), Room("Моечная",5,9,2.2,2.0,"any",True,True),
             Room("Предбанник",7,14,2.5,2.5,"S",adj_to=["Комната отдыха"]),
             Room("Комната отдыха",12,22,3.5,3.0,"S")]
    elif btype == "гараж":
        r = [Room("Гараж",20,40,4.0,4.0,"any",False), Room("Котельная",6,10,2.2,2.2,"N")]
    else:
        r = [Room("Основное",area*0.6,area*0.8,5.0,4.0,"S"), Room("Приёмная",7,14,2.8,2.5,"S"),
             Room("Санузел",3,6,1.8,1.6,"any",True), Room("Коридор",4,8,1.2,2.0,"any",False)]
    return r

# ==================== LAYOUT ENGINE ====================

def layout(rooms: List[Room], bw: float, bd: float) -> List[Room]:
    placed = []; occ = []
    order = {"Прихожая":0,"Гостиная":1,"Кухня":2,"Коридор":3,"Ванная":4,"Туалет":5,"Предбанник":6}
    rooms.sort(key=lambda r: order.get(r.name, 10))
    
    for room in rooms:
        area = (room.min_area + room.max_area) / 2
        best = None; best_score = -1
        for _ in range(500):
            rw = random.uniform(room.min_w, min(bw*0.65, math.sqrt(area)*1.4))
            rw = max(room.min_w, rw); rd = area / rw
            if rd < room.min_d or rd > bd*0.65: continue
            if rw * rd < room.min_area * 0.85 or rw * rd > room.max_area: continue
            x = random.uniform(0, bw - rw); y = random.uniform(0, bd - rd)
            x = round(x, 1); y = round(y, 1)
            overlaps = any(not(x+rw<=ox+0.15 or ox+ow<=x+0.15 or y+rd<=oy+0.15 or oy+od<=y+0.15) for ox,oy,ow,od in occ)
            if overlaps: continue
            score = 0
            if room.has_window:
                if y < 0.5: score += 15
                if y+rd > bd-0.5: score += 12
                if x < 0.5: score += 8
                if x+rw > bw-0.5: score += 8
            for adj_name in room.adj_to:
                for p in placed:
                    if p.name == adj_name:
                        dist = math.sqrt((x+rw/2 - p.x-p.w/2)**2 + (y+rd/2 - p.y-p.d/2)**2)
                        score += max(0, 25 - dist*4)
            score -= (x + y) * 0.3
            if score > best_score:
                best_score = score; best = (x, y, round(rw, 1), round(rd, 1))
        if best:
            room.x, room.y, room.w, room.d = best
            placed.append(room); occ.append((room.x, room.y, room.w, room.d))
    return placed

def snip_check(rooms: List[Dict]) -> List[str]:
    r = []
    for rm in rooms:
        a = rm.get("w",0)*rm.get("d",0); n = rm.get("name","")
        checks = [
            ("Спальня" in n or "Детская" in n, a >= 8, f"{n}: {a:.1f}м² {'≥' if a>=8 else '<'} 8м² (СНиП 2.08.01-89)"),
            (n == "Гостиная", a >= 16, f"{n}: {a:.1f}м² {'≥' if a>=16 else '<'} 16м²"),
            (n == "Кухня", a >= 8, f"{n}: {a:.1f}м² {'≥' if a>=8 else '<'} 8м²"),
            (n == "Ванная", a >= 3.5, f"{n}: {a:.1f}м² {'≥' if a>=3.5 else '<'} 3.5м²"),
        ]
        for cond, ok, msg in checks:
            if cond: r.append(("✅" if ok else "⚠️") + " " + msg); break
    return r

# ==================== AI ARCHITECT ====================

def ai_architect(proj: dict) -> Optional[dict]:
    if not GEMINI_API_KEY: return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""Архитектор: спроектируй планировку. Параметры: тип={proj['building_type']}, площадь={proj['area']}м², этажей={proj['floors']}, комнат={proj['rooms']}, участок={proj['site_width']}x{proj['site_depth']}м, требования: {proj.get('requirements','стандарт')}.

Ответь ТОЛЬКО JSON (без markdown, без пояснений):
{{"building":{{"width":X.X,"depth":X.X,"floors":X,"rooms":[{{"name":"Имя","w":X.X,"d":X.X,"x":X.X,"y":X.X}}]}},"description":"описание"}}

Правила:
- Гостиная ≥16м², спальни ≥8м², кухня ≥8м²
- Все комнаты в пределах building.width × building.depth
- x ≥ 0, y ≥ 0, x+w ≤ width, y+d ≤ depth
- Комнаты НЕ перекрываются"""

        resp = model.generate_content(prompt)
        text = resp.text; j = re.search(r'\{[\s\S]*\}', text)
        if j:
            plan = json.loads(j.group())
            bldg = plan.get("building", {}); rooms = bldg.get("rooms", [])
            if rooms and bldg.get("width") and bldg.get("depth"):
                for rm in rooms:
                    rm.setdefault("is_wet", False)
                    rm.setdefault("has_window", True)
                return plan
    except Exception as e:
        print(f"AI error: {e}")
    return None

# ==================== PLAN GENERATOR ====================

def gen_plan(proj: dict) -> dict:
    area=proj["area"]; fl=proj["floors"]; rc=proj["rooms"]
    sw=proj["site_width"]; sd=proj["site_depth"]; bt=proj["building_type"]
    fa=area/fl
    
    # Try AI first
    ai_plan = ai_architect(proj)
    if ai_plan:
        bldg = ai_plan["building"]
        bw=bldg["width"]; bd=bldg["depth"]
        ai_plan["site"] = {"width":sw,"depth":sd,"building_x":round((sw-bw)/3,1),"building_y":round((sd-bd)/3,1),"parking":True,"garden":True,"driveway":True}
        ai_plan["description"] = ai_plan.get("description", f"{bt}, {area}м², {fl}эт.")
        return ai_plan
    
    # Fallback: algorithmic
    bw=round(math.sqrt(fa)*1.25,1); bd=round(fa/bw,1)
    rooms=catalog(bt,fa,rc); placed=layout(rooms,bw,bd)
    rd=[{"name":r.name,"w":r.w,"d":r.d,"x":round(r.x,1),"y":round(r.y,1),"is_wet":r.is_wet,"has_window":r.has_window} for r in placed]
    return {"building":{"width":bw,"depth":bd,"floors":fl,"rooms":rd,"entrance":{"x":round(bw/2,1),"y":0}},
            "site":{"width":sw,"depth":sd,"building_x":round((sw-bw)/3,1),"building_y":round((sd-bd)/3,1),"parking":True,"garden":True,"driveway":True},
            "description":f"{bt}, {area}м², {fl}эт., {len(placed)}пом."}

# ==================== SVG GENERATORS ====================

def svg_site(plan: dict) -> str:
    site=plan["site"]; bldg=plan["building"]; rooms=bldg["rooms"]
    sw=site["width"]; sd=site["depth"]; bw=bldg["width"]; bd=bldg["depth"]
    bx=site["building_x"]; by=site["building_y"]; fl=bldg["floors"]
    S=16; M=80; TH=70; SH=50
    W=sw*S+M*2+160; H=sd*S+M*2+TH+SH
    X0=M+bx*S; Y0=M+TH+(sd-by-bd)*S; WW=bw*S; HH=bd*S
    
    s=f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs><marker id="a" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#1a73e8"/></marker>
<pattern id="g" patternUnits="userSpaceOnUse" width="20" height="20"><rect width="20" height="20" fill="#e8f5e9"/><circle cx="5" cy="5" r="1" fill="#a5d6a7"/><circle cx="15" cy="15" r="1" fill="#a5d6a7"/></pattern>
<pattern id="pk" patternUnits="userSpaceOnUse" width="30" height="20"><rect width="30" height="20" fill="#e0e0e0"/><line x1="15" y1="0" x2="15" y2="20" stroke="#bbb"/></pattern></defs>
<text x="{M}" y="30" font-family="Arial" font-size="18" font-weight="bold" fill="#333">ПЛАН УЧАСТКА</text>
<text x="{M}" y="50" font-family="Arial" font-size="12" fill="#666">{plan.get("description","")}</text>
<text x="{M}" y="64" font-family="Arial" font-size="10" fill="#999">М 1:100 | {datetime.now().strftime("%d.%m.%Y")}</text>
<rect x="{M}" y="{M+TH}" width="{sw*S}" height="{sd*S}" fill="url(#g)" stroke="#2d5016" stroke-width="2"/>
<rect x="{X0}" y="{Y0}" width="{WW}" height="{HH}" fill="#fff" stroke="#333" stroke-width="3"/>'''
    
    for rm in rooms:
        rx=X0+rm["x"]*S; ry=Y0+(bd-rm["y"]-rm["d"])*S; rw=rm["w"]*S; rd=rm["d"]*S; ar=rm["w"]*rm["d"]
        s+=f'''
<rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" fill="none" stroke="#555" stroke-width="1.5"/>
<text x="{rx+rw/2}" y="{ry+rd/2-5}" font-family="Arial" font-size="11" font-weight="bold" fill="#333" text-anchor="middle">{rm["name"]}</text>
<text x="{rx+rw/2}" y="{ry+rd/2+9}" font-family="Arial" font-size="9" fill="#888" text-anchor="middle">{ar:.1f}м²</text>'''
    
    s+=f'''
<rect x="{X0+WW/2-10}" y="{Y0+HH-3}" width="20" height="6" fill="#ff9800" stroke="#e65100" rx="1"/>
<text x="{X0+WW/2}" y="{Y0+HH+16}" font-family="Arial" font-size="9" fill="#e65100" text-anchor="middle">ВХОД</text>
<polygon points="{M+bx*S},{M+TH+sd*S} {M+(bx+3)*S},{M+TH+sd*S} {M+(bx+3)*S},{Y0+HH} {M+bx*S},{Y0+HH}" fill="#d7ccc8" stroke="#a1887f"/>
<rect x="{M+(sw-6)*S}" y="{M+TH+(sd-8)*S}" width="{5*S}" height="{6*S}" fill="url(#pk)" stroke="#999"/>
<text x="{M+(sw-3.5)*S}" y="{M+TH+(sd-3.5)*S}" font-family="Arial" font-size="10" fill="#666" text-anchor="middle">Парковка</text>'''
    
    # Trees
    for i in range(3):
        tx = M + random.randint(int(bx*S + WW + 2*S), int(sw*S - 3*S))
        ty = M + TH + random.randint(2*S, int(sd*S - 3*S))
        s+=f'<circle cx="{tx}" cy="{ty}" r="8" fill="#4caf50" opacity="0.7"/><circle cx="{tx}" cy="{ty}" r="4" fill="#388e3c"/>'
    
    # Dimensions
    s+=f'''
<line x1="{M}" y1="{M+TH+sd*S+15}" x2="{M+sw*S}" y2="{M+TH+sd*S+15}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M+sw*S/2}" y="{M+TH+sd*S+30}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle">{sw} м</text>
<line x1="{M+sw*S+15}" y1="{M+TH}" x2="{M+sw*S+15}" y2="{M+TH+sd*S}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M+sw*S+30}" y="{M+TH+sd*S/2}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle" transform="rotate(90,{M+sw*S+30},{M+TH+sd*S/2})">{sd} м</text>
<line x1="{X0}" y1="{Y0-10}" x2="{X0+WW}" y2="{Y0-10}" stroke="#e65100" stroke-width="0.8"/>
<text x="{X0+WW/2}" y="{Y0-14}" font-family="Arial" font-size="10" fill="#e65100" text-anchor="middle">{bw} м</text>
<line x1="{W-100}" y1="{M+TH+50}" x2="{W-100}" y2="{M+TH+10}" stroke="#333" stroke-width="2"/>
<polygon points="{W-100},{M+TH+5} {W-106},{M+TH+20} {W-94},{M+TH+20}" fill="#e53935"/>
<text x="{W-100}" y="{M+TH}" font-family="Arial" font-size="12" font-weight="bold" fill="#e53935" text-anchor="middle">N</text>'''
    
    # Stamp
    s+=f'''
<rect x="{M}" y="{H-SH}" width="{sw*S}" height="{SH}" fill="none" stroke="#333" stroke-width="1.5"/>
<line x1="{M}" y1="{H-SH+20}" x2="{M+sw*S}" y2="{H-SH+20}" stroke="#333"/>
<line x1="{M+sw*S*0.6}" y1="{H-SH}" x2="{M+sw*S*0.6}" y2="{H}" stroke="#333"/>
<text x="{M+5}" y="{H-SH+14}" font-family="Arial" font-size="9" fill="#333">План участка</text>
<text x="{M+sw*S*0.6+5}" y="{H-SH+14}" font-family="Arial" font-size="9" fill="#333">М 1:100</text>
<text x="{M+5}" y="{H-SH+38}" font-family="Arial" font-size="8" fill="#666">{datetime.now().strftime("%d.%m.%Y")}</text>
<text x="{M+sw*S*0.6+5}" y="{H-SH+38}" font-family="Arial" font-size="8" fill="#666">Archiq AI</text>
</svg>'''
    return s

def svg_floor(plan: dict) -> str:
    bldg=plan["building"]; rooms=bldg["rooms"]; bw=bldg["width"]; bd=bldg["depth"]; fl=bldg["floors"]
    S=50; M=80; TH=70; SH=50
    W=bw*S+M*2; H=bd*S+M*2+TH+SH+120
    WALL_T=3
    
    s=f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs>
<marker id="a" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#1a73e8"/></marker>
<marker id="arrow" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto"><polygon points="0 0,6 2,0 4" fill="#999"/></marker>
</defs>
<text x="{M}" y="30" font-family="Arial" font-size="18" font-weight="bold" fill="#333">ПЛАН {fl}-го ЭТАЖА</text>
<text x="{M}" y="50" font-family="Arial" font-size="12" fill="#666">{plan.get("description","")}</text>
<text x="{M}" y="64" font-family="Arial" font-size="10" fill="#999">Масштаб 1:100</text>
<rect x="{M}" y="{M+TH}" width="{bw*S}" height="{bd*S}" fill="#fafafa" stroke="#333" stroke-width="{WALL_T}"/>'''
    
    # Room divisions (walls)
    for rm in rooms:
        rx=M+rm["x"]*S; ry=M+TH+(bd-rm["y"]-rm["d"])*S; rw=rm["w"]*S; rd=rm["d"]*S; ar=rm["w"]*rm["d"]
        s+=f'''
<rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" fill="none" stroke="#444" stroke-width="2"/>
<!-- Windows -->'''
        # Windows on exterior walls
        if rm.get("has_window", True):
            # Bottom wall (y=0)
            if rm["y"] < 0.3:
                wx=rx+rw/2-S; s+=f'<rect x="{wx}" y="{ry-2}" width="{S*2}" height="4" fill="#81d4fa" stroke="#0288d1" stroke-width="1" rx="1"/>'
            # Top wall (y=bd)
            if rm["y"]+rm["d"] > bd-0.3:
                wx=rx+rw/2-S; s+=f'<rect x="{wx}" y="{ry+rd-2}" width="{S*2}" height="4" fill="#81d4fa" stroke="#0288d1" stroke-width="1" rx="1"/>'
            # Left wall (x=0)
            if rm["x"] < 0.3:
                wy=ry+rd/2-S; s+=f'<rect x="{rx-2}" y="{wy}" width="4" height="{S*2}" fill="#81d4fa" stroke="#0288d1" stroke-width="1" rx="1"/>'
            # Right wall (x=bw)
            if rm["x"]+rm["w"] > bw-0.3:
                wy=ry+rd/2-S; s+=f'<rect x="{rx+rw-2}" y="{wy}" width="4" height="{S*2}" fill="#81d4fa" stroke="#0288d1" stroke-width="1" rx="1"/>'
        
        # Door indicator (entrance for прихожая or first room)
        if rm["name"] == "Прихожая" or (rm["x"] < 0.5 and rm["y"] < 0.5):
            dx = rx + rw/2 - S
            s+=f'''
<rect x="{dx}" y="{ry+rd-3}" width="{S*2}" height="6" fill="#ffcc80" stroke="#e65100" stroke-width="1" rx="1"/>
<path d="M {dx} {ry+rd} A {S*2} {S*2} 0 0 1 {dx+S*2} {ry+rd}" fill="none" stroke="#e65100" stroke-width="0.5" stroke-dasharray="3,2"/>'''
        
        # Room label
        s+=f'''
<text x="{rx+rw/2}" y="{ry+rd/2-6}" font-family="Arial" font-size="12" font-weight="bold" fill="#333" text-anchor="middle">{rm["name"]}</text>
<text x="{rx+rw/2}" y="{ry+rd/2+10}" font-family="Arial" font-size="10" fill="#888" text-anchor="middle">{ar:.1f}м²</text>'''
        
        # Furniture hints for key rooms
        if rm["name"] == "Гостиная":
            # Sofa
            fx=rx+rw*0.2; fy=ry+rd*0.2; fw=rm["w"]*S*0.4; fd=rm["d"]*S*0.15
            s+=f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fd}" fill="#e0e0e0" stroke="#999" rx="2" opacity="0.5"/>'
            # TV
            s+=f'<rect x="{rx+rw*0.35}" y="{ry+rd*0.02}" width="{rm["w"]*S*0.3}" height="{S*0.1}" fill="#333" rx="1" opacity="0.3"/>'
        elif rm["name"] == "Кухня":
            # Counter
            s+=f'<rect x="{rx+2}" y="{ry+2}" width="{S*0.6}" height="{rd-4}" fill="#bdbdbd" stroke="#999" rx="1" opacity="0.5"/>'
            # Table
            tx=rx+rw*0.5; ty=ry+rd*0.5
            s+=f'<circle cx="{tx}" cy="{ty}" r="{S*0.3}" fill="#e0e0e0" stroke="#999" opacity="0.5"/>'
        elif "Спальня" in rm["name"] or rm["name"] == "Детская":
            # Bed
            bx2=rx+rw*0.1; by2=ry+rd*0.1; bww=rm["w"]*S*0.4; bdd=rm["d"]*S*0.5
            s+=f'<rect x="{bx2}" y="{by2}" width="{bww}" height="{bdd}" fill="#e0e0e0" stroke="#999" rx="3" opacity="0.5"/>'
            s+=f'<rect x="{bx2+2}" y="{by2+2}" width="{bww-4}" height="{bdd*0.3}" fill="#fff" stroke="#bbb" rx="2" opacity="0.5"/>'
        elif rm["name"] == "Ванная":
            # Bathtub
            s+=f'<rect x="{rx+2}" y="{ry+2}" width="{S*0.7}" height="{S*0.35}" fill="#e1f5fe" stroke="#4fc3f7" rx="3" opacity="0.5"/>'
    
    # Entrance
    s+=f'''
<rect x="{M+bw*S/2-12}" y="{M+TH+bd*S-4}" width="24" height="8" fill="#ff9800" stroke="#e65100" stroke-width="1.5" rx="1"/>
<text x="{M+bw*S/2}" y="{M+TH+bd*S+18}" font-family="Arial" font-size="10" font-weight="bold" fill="#e65100" text-anchor="middle">ВХОД</text>
<line x1="{M}" y1="{M+TH+bd*S+15}" x2="{M+bw*S}" y2="{M+TH+bd*S+15}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M+bw*S/2}" y="{M+TH+bd*S+30}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle">{bw} м</text>
<line x1="{M-20}" y1="{M+TH}" x2="{M-20}" y2="{M+TH+bd*S}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M-30}" y="{M+TH+bd*S/2}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle" transform="rotate(-90,{M-30},{M+TH+bd*S/2})">{bd} м</text>'''
    
    # Room schedule
    ty=M+TH+bd*S+55; ta=sum(r["w"]*r["d"] for r in rooms)
    s+=f'''
<text x="{M}" y="{ty}" font-family="Arial" font-size="13" font-weight="bold" fill="#333">ЭКСПЛИКАЦИЯ ПОМЕЩЕНИЙ</text>
<rect x="{M}" y="{ty+5}" width="{bw*S}" height="{20+len(rooms)*18+20}" fill="none" stroke="#333"/>
<rect x="{M}" y="{ty+5}" width="{bw*S}" height="18" fill="#e0e0e0"/>
<text x="{M+8}" y="{ty+17}" font-family="Arial" font-size="9" font-weight="bold">№</text>
<text x="{M+30}" y="{ty+17}" font-family="Arial" font-size="9" font-weight="bold">Наименование</text>
<text x="{M+bw*S-70}" y="{ty+17}" font-family="Arial" font-size="9" font-weight="bold">Площадь, м²</text>'''
    for i,rm in enumerate(rooms):
        y=ty+25+i*18; ar=rm["w"]*rm["d"]; bg="#fff" if i%2==0 else "#f5f5f5"
        s+=f'''
<rect x="{M}" y="{y}" width="{bw*S}" height="17" fill="{bg}"/>
<text x="{M+12}" y="{y+12}" font-family="Arial" font-size="9">{i+1}</text>
<text x="{M+30}" y="{y+12}" font-family="Arial" font-size="9">{rm["name"]}</text>
<text x="{M+bw*S-70}" y="{y+12}" font-family="Arial" font-size="9">{ar:.1f}</text>'''
    s+=f'''
<rect x="{M}" y="{ty+25+len(rooms)*18}" width="{bw*S}" height="18" fill="#e0e0e0"/>
<text x="{M+30}" y="{ty+25+len(rooms)*18+12}" font-family="Arial" font-size="9" font-weight="bold">ИТОГО</text>
<text x="{M+bw*S-70}" y="{ty+25+len(rooms)*18+12}" font-family="Arial" font-size="9" font-weight="bold">{ta:.1f}</text>'''
    
    # Stamp
    s+=f'''
<rect x="{M}" y="{H-SH}" width="{bw*S}" height="{SH}" fill="none" stroke="#333" stroke-width="1.5"/>
<line x1="{M}" y1="{H-SH+20}" x2="{M+bw*S}" y2="{H-SH+20}" stroke="#333"/>
<line x1="{M+bw*S*0.6}" y1="{H-SH}" x2="{M+bw*S*0.6}" y2="{H}" stroke="#333"/>
<text x="{M+5}" y="{H-SH+14}" font-family="Arial" font-size="9" fill="#333">План {fl}-го этажа</text>
<text x="{M+bw*S*0.6+5}" y="{H-SH+14}" font-family="Arial" font-size="9" fill="#333">М 1:100</text>
<text x="{M+5}" y="{H-SH+38}" font-family="Arial" font-size="8" fill="#666">{datetime.now().strftime("%d.%m.%Y")}</text>
<text x="{M+bw*S*0.6+5}" y="{H-SH+38}" font-family="Arial" font-size="8" fill="#666">Archiq AI</text>
</svg>'''
    return s

# ==================== PDF EXPORT ====================

def gen_pdf(plan: dict) -> bytes:
    from reportlab.lib.pagesizes import A3
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A3)
    W, H = A3
    
    bldg = plan["building"]; site = plan["site"]; rooms = bldg["rooms"]
    
    # Title page
    c.setFillColor(colors.HexColor("#0f172a"))
    c.rect(0, 0, W, H, fill=True, stroke=False)
    c.setFillColor(colors.HexColor("#38bdf8"))
    c.setFont("Helvetica-Bold", 36)
    c.drawString(50*mm, H - 120*mm, "Archiq AI")
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 20)
    c.drawString(50*mm, H - 140*mm, plan.get("description", "Архитектурный план"))
    c.setFont("Helvetica", 14)
    c.drawString(50*mm, H - 160*mm, f"Дата: {datetime.now().strftime('%d.%m.%Y')}")
    c.drawString(50*mm, H - 180*mm, f"Тип: {bldg.get('floors', 1)} этаж, {len(rooms)} помещений")
    
    # Specs page
    c.showPage()
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(20*mm, H - 20*mm, "Спецификация проекта")
    
    ta = sum(r["w"]*r["d"] for r in rooms)
    la = sum(r["w"]*r["d"] for r in rooms if r["name"] in ("Гостиная","Спальня","Детская","Кабинет"))
    sa = site["width"] * site["depth"]
    
    data = [["Параметр", "Значение"], ["Общая площадь", f"{ta:.1f} м²"], ["Жилая площадь", f"{la:.1f} м²"],
            ["Этажность", str(bldg["floors"])], ["Количество помещений", str(len(rooms))],
            ["Размеры здания", f"{bldg['width']} x {bldg['depth']} м"],
            ["Размеры участка", f"{site['width']} x {site['depth']} м"],
            ["Коэффициент застройки", f"{ta/sa*100:.1f}%"],
            ["Периметр", f"{2*(bldg['width']+bldg['depth']):.1f} м"]]
    t = Table(data, colWidths=[80*mm, 80*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8), ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#334155")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    t.wrapOn(c, W, H); t.drawOn(c, 20*mm, H - 100*mm)
    
    # Room schedule
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20*mm, H - 180*mm, "Экспликация помещений")
    room_data = [["№", "Помещение", "Площадь, м²"]]
    for i, rm in enumerate(rooms):
        room_data.append([str(i+1), rm["name"], f'{rm["w"]*rm["d"]:.1f}'])
    room_data.append(["", "ИТОГО", f"{ta:.1f}"])
    rt = Table(room_data, colWidths=[20*mm, 100*mm, 40*mm])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#334155")),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    rt.wrapOn(c, W, H); rt.drawOn(c, 20*mm, H - 280*mm)
    
    # SNiP
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20*mm, H - 310*mm, "Проверка СНиП")
    checks = snip_check(rooms)
    y = H - 330*mm; c.setFont("Helvetica", 12)
    for ch in checks:
        color = colors.HexColor("#22c55e") if ch.startswith("✅") else colors.red
        c.setFillColor(color); c.drawString(25*mm, y, ch); y -= 18*mm
    
    c.save()
    return buf.getvalue()

# ==================== DXF EXPORT ====================

def export_dxf(plan: dict) -> str:
    b=plan["building"]; rooms=b["rooms"]; bw=b["width"]; bd=b["depth"]; S=1000
    dxf=f'''  0
SECTION
  2
ENTITIES
'''
    # Exterior walls
    pts = [(0,0),(bw*S,0),(bw*S,bd*S),(0,bd*S)]
    for i in range(len(pts)):
        x1,y1 = pts[i]; x2,y2 = pts[(i+1)%len(pts)]
        dxf += f'''  0
LINE
  8
WALLS-EXT
 10
{x1}
 20
{y1}
 30
0
 11
{x2}
 21
{y2}
 31
0
'''
    # Room divisions
    for rm in rooms:
        x=rm["x"]*S; y=rm["y"]*S; w=rm["w"]*S; d=rm["d"]*S
        for (ax,ay,bx2,by2) in [(x,y,x+w,y),(x+w,y,x+w,y+d),(x+w,y+d,x,y+d),(x,y+d,x,y)]:
            dxf += f'''  0
LINE
  8
ROOM-{rm["name"].upper().replace(" ","_")}
 10
{ax}
 20
{ay}
 30
0
 11
{bx2}
 21
{by2}
 31
0
'''
        # Label
        cx=(x+w/2); cy=(y+d/2)
        dxf += f'''  0
TEXT
  8
TEXT
 10
{cx}
 20
{cy}
 30
0
 40
400
  1
{rm["name"]} {rm["w"]}x{rm["d"]}m
'''
    dxf += '''  0
ENDSEC
  0
EOF
'''
    return dxf

# ==================== SPECS ====================

def specs(plan: dict) -> dict:
    b=plan["building"]; s=plan["site"]; rooms=b["rooms"]
    ta=sum(r["w"]*r["d"] for r in rooms)
    la=sum(r["w"]*r["d"] for r in rooms if r["name"] in ("Гостиная","Спальня","Детская","Кабинет"))
    sa=s["width"]*s["depth"]
    return {"total_area":round(ta,1),"living_area":round(la,1),"floors":b["floors"],
            "rooms":len(rooms),"bldg_dim":f"{b['width']}×{b['depth']}м",
            "site_dim":f"{s['width']}×{s['depth']}м","footprint":round(ta,1),
            "site_area":round(sa,1),"coverage":round(ta/sa*100,1),
            "perimeter":round(2*(b["width"]+b["depth"]),1),
            "compliance":snip_check(rooms),"description":plan.get("description","")}

# ==================== APP ====================

app = FastAPI(title="Archiq AI v4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LANDING = """<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Archiq AI v6</title><style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem}.c{max-width:600px;text-align:center}.l{font-size:3rem;font-weight:800;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}.s{color:#94a3b8;margin-bottom:2rem}.st{display:inline-flex;align-items:center;gap:.5rem;background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);padding:.5rem 1rem;border-radius:999px;margin-bottom:2rem}.d{width:8px;height:8px;border-radius:50%;background:#22c55e;animation:p 2s infinite}@keyframes p{0%,100%{opacity:1}50%{opacity:.4}}.e{text-align:left;background:rgba(30,41,59,.8);border:1px solid #334155;border-radius:12px;padding:1.5rem}.e h2{font-size:1rem;color:#94a3b8;margin-bottom:1rem;text-transform:uppercase}.r{display:flex;gap:1rem;padding:.6rem 0;border-bottom:1px solid #1e293b}.r:last-child{border-bottom:none}.m{font-family:monospace;font-size:.8rem;font-weight:700;min-width:55px;padding:.15rem .4rem;border-radius:4px;text-align:center}.g{background:rgba(56,189,248,.15);color:#38bdf8}.p{background:rgba(168,85,247,.15);color:#a855f7}.o{background:rgba(251,146,60,.15);color:#fb923c}.t{font-family:monospace;font-size:.85rem;color:#e2e8f0}.x{color:#64748b;font-size:.8rem;margin-left:auto}.f{margin-top:2rem;color:#475569;font-size:.8rem}.f a{color:#818cf8;text-decoration:none}</style></head>
<body><div class="c"><div class="l">🏗️ Archiq AI v6</div><div class="s">AI-архитектор + СНиП валидация + анализ участка + чертежи + PDF/DXF</div><div class="st"><span class="d"></span> Работает</div>
<div class="e"><h2>API</h2><div class="r"><span class="m g">GET</span><span class="t">/health</span><span class="x">Статус</span></div><div class="r"><span class="m p">POST</span><span class="t">/generate</span><span class="x">Генерация по параметрам</span></div><div class="r"><span class="m p">POST</span><span class="t">/analyze-site</span><span class="x">AI-анализ плана участка</span></div><div class="r"><span class="m p">POST</span><span class="t">/ai-architect</span><span class="x">🆕 AI-архитектор</span></div><div class="r"><span class="m o">POST</span><span class="t">/generate-ai</span><span class="x">🆕 AI генерация + файлы</span></div><div class="r"><span class="m p">POST</span><span class="t">/snip-check</span><span class="x">Проверка СНиП</span></div><div class="r"><span class="m p">POST</span><span class="t">/generate-from-site</span><span class="x">План → планировка</span></div><div class="r"><span class="m g">GET</span><span class="t">/site-svg?id=...</span><span class="x">SVG участок</span></div><div class="r"><span class="m g">GET</span><span class="t">/floor-svg?id=...</span><span class="x">SVG этаж</span></div><div class="r"><span class="m g">GET</span><span class="t">/pdf?id=...</span><span class="x">PDF</span></div><div class="r"><span class="m g">GET</span><span class="t">/dxf?id=...</span><span class="x">DXF</span></div></div>
<div class="f"><p>GitHub: <a href="https://github.com/K09-0/ARCHIQ-AI" target="_blank">K09-0/ARCHIQ-AI</a></p></div></div></body></html>"""

def init_db():
    conn=sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS p(id TEXT PRIMARY KEY,name TEXT,bt TEXT,area REAL,fl INT,rooms INT,sw REAL,sd REAL,req TEXT,plan TEXT,ts TEXT)")
    conn.commit(); conn.close()

@app.on_event("startup")
def startup(): init_db()

@app.get("/",response_class=HTMLResponse)
def root(): return LANDING

@app.get("/health")
def health(): return {"status":"ok","gemini":bool(GEMINI_API_KEY),"hf":bool(HF_API_KEY)}

@app.post("/generate")
def generate(name:str=Form("Проект"),bt:str=Form("жилой дом"),area:float=Form(100),fl:int=Form(1),rooms:int=Form(3),sw:float=Form(20),sd:float=Form(30),req:str=Form("")):
    proj={"name":name,"building_type":bt,"area":area,"floors":fl,"rooms":rooms,"site_width":sw,"site_depth":sd,"requirements":req}
    pl=gen_plan(proj); sp=specs(pl); pid=datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUTPUT/f"{pid}_site.svg").write_text(svg_site(pl))
    (OUTPUT/f"{pid}_floor.svg").write_text(svg_floor(pl))
    (OUTPUT/f"{pid}.dxf").write_text(export_dxf(pl))
    (OUTPUT/f"{pid}.pdf").write_bytes(gen_pdf(pl))
    conn=sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO p VALUES(?,?,?,?,?,?,?,?,?,?,?)",(pid,name,bt,area,fl,rooms,sw,sd,req,json.dumps(pl),datetime.now().isoformat()))
    conn.commit(); conn.close()
    return {"id":pid,"plan":pl,"specs":sp}

@app.get("/site-svg")
def site_svg(id:str):
    p=OUTPUT/f"{id}_site.svg"
    return HTMLResponse(content=p.read_text()) if p.exists() else JSONResponse({"error":"Not found"},404)

@app.get("/floor-svg")
def floor_svg(id:str):
    p=OUTPUT/f"{id}_floor.svg"
    return HTMLResponse(content=p.read_text()) if p.exists() else JSONResponse({"error":"Not found"},404)

@app.get("/dxf")
def dxf(id:str):
    p=OUTPUT/f"{id}.dxf"
    if p.exists():
        return HTMLResponse(content=p.read_text(), headers={"Content-Disposition": f"attachment; filename={id}.dxf"})
    return JSONResponse({"error":"Not found"},404)

@app.get("/pdf")
def pdf(id:str):
    p=OUTPUT/f"{id}.pdf"
    if p.exists(): return FileResponse(str(p), media_type="application/pdf", filename=f"{id}.pdf")
    return JSONResponse({"error":"Not found"},404)

@app.get("/specs")
def get_specs(id:str):
    conn=sqlite3.connect(DB_PATH); r=conn.execute("SELECT plan FROM p WHERE id=?",(id,)).fetchone(); conn.close()
    return specs(json.loads(r[0])) if r else JSONResponse({"error":"Not found"},404)

@app.get("/projects")
def list_p():
    conn=sqlite3.connect(DB_PATH); conn.row_factory=sqlite3.Row
    rows=conn.execute("SELECT id,name,bt,area,fl,rooms,ts FROM p ORDER BY ts DESC").fetchall(); conn.close()
    return [dict(r) for r in rows]

@app.post("/analyze-site")
async def analyze_site(file:UploadFile=File(...), model:str=Form("gemini-2.0-flash")):
    """Этап 3: AI-анализ плана участка через Gemini Vision.
    
    Загрузите изображение плана участка — получите:
    - Размеры участка (ширина, глубина, площадь)
    - Границы участка
    - Существующие постройки
    - Красные линии
    - Авто-калибровка масштаба
    - Северная ориентация
    - Подъездные пути, озеленение
    """
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "GEMINI_API_KEY not configured"}, 400)
    
    content = await file.read()
    if not content:
        return JSONResponse({"error": "Empty file"}, 400)
    
    mime = file.content_type or "image/jpeg"
    result = analyze_site_plan(content, mime, GEMINI_API_KEY, model)
    
    if not result.success:
        return JSONResponse({"error": result.error}, 500)
    
    return result.to_dict()


@app.post("/generate-from-site")
async def generate_from_site(
    file: UploadFile = File(...),
    name: str = Form("Проект"),
    bt: str = Form("жилой дом"),
    area: float = Form(100),
    fl: int = Form(1),
    rooms: int = Form(3),
    req: str = Form(""),
    model: str = Form("gemini-2.0-flash"),
):
    """Этап 3+4: Загрузить план участка → AI анализ → генерация планировки.
    
    1. Анализирует план участка через Gemini Vision
    2. Извлекает размеры и параметры
    3. Генерирует оптимальную планировку
    4. Создаёт SVG + PDF + DXF
    """
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "GEMINI_API_KEY not configured"}, 400)
    
    content = await file.read()
    if not content:
        return JSONResponse({"error": "Empty file"}, 400)
    
    mime = file.content_type or "image/jpeg"
    
    # Step 1: Analyze site plan
    analysis = analyze_site_plan(content, mime, GEMINI_API_KEY, model)
    
    if not analysis.success:
        return JSONResponse({"error": f"Site analysis failed: {analysis.error}"}, 500)
    
    # Step 2: Convert analysis to plan parameters
    site_params = analysis_to_plan_params(analysis)
    
    # Step 3: Generate plan using detected site dimensions
    proj = {
        "name": name,
        "building_type": bt,
        "area": area,
        "floors": fl,
        "rooms": rooms,
        "site_width": site_params["site_width"],
        "site_depth": site_params["site_depth"],
        "requirements": req,
    }
    
    pl = gen_plan(proj)
    sp = specs(pl)
    pid = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 4: Generate all outputs
    (OUTPUT / f"{pid}_site.svg").write_text(svg_site(pl))
    (OUTPUT / f"{pid}_floor.svg").write_text(svg_floor(pl))
    (OUTPUT / f"{pid}.dxf").write_text(export_dxf(pl))
    (OUTPUT / f"{pid}.pdf").write_bytes(gen_pdf(pl))
    
    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO p VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (pid, name, bt, area, fl, rooms,
         site_params["site_width"], site_params["site_depth"],
         req, json.dumps(pl), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    
    return {
        "id": pid,
        "site_analysis": analysis.to_dict(),
        "site_params": site_params,
        "plan": pl,
        "specs": sp,
    }


# ==================== ЭТАП 4: AI-АРХИТЕКТОР ====================

@app.post("/ai-architect")
async def ai_architect_endpoint(
    bt: str = Form("жилой дом"),
    area: float = Form(100),
    fl: int = Form(1),
    rooms: int = Form(3),
    sw: float = Form(20),
    sd: float = Form(30),
    req: str = Form(""),
    model: str = Form("gemini-2.0-flash"),
):
    """Этап 4: AI-архитектор — Gemini генерирует оптимальную планировку.
    
    - Генерация планировки с учётом СНиП
    - Автоматическая проверка СНиП
    - Текстовое описание проекта
    - Учёт участка и дополнительных требований
    """
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "GEMINI_API_KEY not configured"}, status_code=400)
    
    try:
        result = ai_architect_v2(
            building_type=bt,
            area=area,
            floors=fl,
            rooms_count=rooms,
            site_width=sw,
            site_depth=sd,
            requirements=req,
            gemini_api_key=GEMINI_API_KEY,
            model_name=model,
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "traceback": tb[:1000]}, status_code=500)
    
    if not result.success:
        return JSONResponse({"error": result.error}, status_code=500)
    
    return result.to_dict()


@app.post("/snip-check")
async def snip_check_endpoint(
    building: str = Form(...),
    rooms: str = Form(...),
):
    """Проверить планировку на соответствие СНиП.
    
    Передайте JSON здания и список комнат.
    """
    try:
        bldg = json.loads(building)
        rm_list = json.loads(rooms)
        result = check_snip(bldg, rm_list)
        return result.to_dict()
    except json.JSONDecodeError as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)


@app.post("/generate-ai")
async def generate_ai(
    name: str = Form("Проект"),
    bt: str = Form("жилой дом"),
    area: float = Form(100),
    fl: int = Form(1),
    rooms: int = Form(3),
    sw: float = Form(20),
    sd: float = Form(30),
    req: str = Form(""),
    model: str = Form("gemini-2.0-flash"),
):
    """Этап 4: AI-генерация + все выходные файлы.
    
    1. Gemini генерирует планировку
    2. Проверка СНиП
    3. Генерация SVG + PDF + DXF
    4. Текстовое описание проекта
    """
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "GEMINI_API_KEY not configured"}, 400)
    
    # Step 1: AI architect
    result = ai_architect_v2(
        building_type=bt,
        area=area,
        floors=fl,
        rooms_count=rooms,
        site_width=sw,
        site_depth=sd,
        requirements=req,
        gemini_api_key=GEMINI_API_KEY,
        model_name=model,
    )
    
    if not result.success:
        return JSONResponse({"error": result.error}, 500)
    
    # Step 2: Convert to plan format
    building = result.building
    plan = {
        "building": building,
        "site": {
            "width": sw,
            "depth": sd,
            "building_x": round((sw - building["width"]) / 3, 1),
            "building_y": round((sd - building["depth"]) / 3, 1),
            "parking": True,
            "garden": True,
            "driveway": True,
        },
        "description": result.description,
    }
    
    # Step 3: Generate outputs
    sp = specs(plan)
    pid = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    (OUTPUT / f"{pid}_site.svg").write_text(svg_site(plan))
    (OUTPUT / f"{pid}_floor.svg").write_text(svg_floor(plan))
    (OUTPUT / f"{pid}.dxf").write_text(export_dxf(plan))
    (OUTPUT / f"{pid}.pdf").write_bytes(gen_pdf(plan))
    
    # Step 4: Generate text description
    text_desc = generate_project_description(
        building, plan["site"], result.snip_check
    )
    
    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO p VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (pid, name, bt, area, fl, rooms, sw, sd,
         req + " | AI-архитектор", json.dumps(plan), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    
    return {
        "id": pid,
        "plan": plan,
        "specs": sp,
        "snip_check": result.snip_check.to_dict(),
        "description": text_desc,
        "reasoning": result.reasoning,
    }
