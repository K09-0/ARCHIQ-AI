"""Archiq AI — Генератор архитектурных планов (v2)."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sqlite3, os, json, math, random, re, base64
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
DB_PATH = os.getenv("DB_PATH", "norms.db")
OUTPUT_DIR = Path("/tmp/archiq-output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== ROOM LAYOUT ENGINE ====================

@dataclass
class RoomDef:
    name: str
    min_area: float
    max_area: float
    min_dim: float
    preferred: str = "any"
    is_wet: bool = False
    has_window: bool = True
    has_door: bool = True
    width: float = 0
    depth: float = 0
    x: float = 0
    y: float = 0

def get_rooms(btype: str, area: float, nrooms: int) -> List[RoomDef]:
    rooms = []
    if btype in ("жилой дом", "дача"):
        rooms = [
            RoomDef("Прихожая", 4, 10, 1.5),
            RoomDef("Гостиная", max(16, area*0.2), max(30, area*0.3), 3.5, "sun"),
            RoomDef("Кухня", max(8, area*0.1), max(16, area*0.15), 2.5, is_wet=True),
            RoomDef("Ванная", 3, 6, 1.6, is_wet=True),
            RoomDef("Коридор", 3, 8, 1.2, has_window=False),
        ]
        bed = max(1, nrooms - 2)
        for i in range(bed):
            nm = "Спальня" if i==0 else (f"Детская" if i==bed-1 and nrooms>4 else f"Спальня {i+1}")
            rooms.append(RoomDef(nm, 10, 18, 3.0, "quiet"))
        if area > 120:
            rooms.append(RoomDef("Кабинет", 8, 12, 2.5, "quiet"))
        if area > 150:
            rooms.append(RoomDef("Гардеробная", 4, 8, 1.5, has_window=False))
            rooms.append(RoomDef("Кладовая", 2, 5, 1.2, has_window=False))
        if area > 100:
            rooms.append(RoomDef("Туалет", 1.5, 3, 1.2, is_wet=True))
    elif btype == "гараж":
        rooms = [RoomDef("Гараж", 18, 36, 3.5, has_window=False), RoomDef("Кладовая", 3, 8, 1.5, has_window=False)]
    elif btype == "баня":
        rooms = [RoomDef("Парная", 4, 8, 1.8, has_window=False), RoomDef("Моечная", 4, 8, 1.8, is_wet=True),
                 RoomDef("Предбанник", 6, 12, 2.0), RoomDef("Комната отдыха", 10, 20, 3.0)]
    else:
        rooms = [RoomDef("Основное помещение", area*0.6, area*0.8, 4.0),
                 RoomDef("Приёмная", 6, 12, 2.5), RoomDef("Санузел", 3, 6, 1.6, is_wet=True),
                 RoomDef("Коридор", 3, 8, 1.2, has_window=False)]
    return rooms

def pack_rooms(rooms: List[RoomDef], bw: float, bd: float) -> List[RoomDef]:
    target = bw * bd
    total = sum(r.min_area for r in rooms)
    corr = target * 0.15
    usable = target - corr
    if total > usable:
        sc = usable / total
        for r in rooms: r.min_area *= sc; r.max_area *= sc
    rooms.sort(key=lambda r: r.min_area, reverse=True)
    placed = []
    occ = []
    for room in rooms:
        a = room.min_area
        asp = random.choice([(1.2,0.83),(1.5,0.67),(1.0,1.0),(0.8,1.25)])
        rw = max(room.min_dim, min(math.sqrt(a*asp[0]), bw*0.7))
        rd = max(room.min_dim, min(a/rw, bd*0.7))
        best = None; best_sc = float('inf')
        step = 0.5
        for gx in [0]+[round(s*step,1) for s in range(1,int(bw/step))]:
            for gy in [0]+[round(s*step,1) for s in range(1,int(bd/step))]:
                if gx+rw>bw or gy+rd>bd: continue
                ov = any(not(gx+rw<=ox+0.01 or ox+ow<=gx+0.01 or gy+rd<=oy+0.01 or oy+od<=gy+0.01) for ox,oy,ow,od in occ)
                if ov: continue
                sc = gx+gy
                if gx==0: sc-=5
                if gy==0: sc-=3
                if sc<best_sc: best_sc=sc; best=(gx,gy)
        if best:
            room.x=best[0]; room.y=best[1]; room.width=round(rw,1); room.depth=round(rd,1)
            placed.append(room); occ.append((room.x,room.y,room.width,room.depth))
    return placed

def check_snip(rooms: List[Dict]) -> List[str]:
    r = []
    for rm in rooms:
        a = rm.get("width",0)*rm.get("depth",0); n = rm.get("name","")
        if ("Спальня" in n or "Детская" in n) and a<8: r.append(f"⚠️ {n}: {a:.1f}м² < 8м²")
        elif n=="Гостиная" and a<16: r.append(f"⚠️ {n}: {a:.1f}м² < 16м²")
        elif n=="Кухня" and a<6: r.append(f"⚠️ {n}: {a:.1f}м² < 6м²")
        else: r.append(f"✅ {n}: {a:.1f}м²")
    return r

# ==================== SVG GENERATORS ====================

def gen_site_svg(plan: dict) -> str:
    site=plan.get("site",{}); bldg=plan.get("building",{}); rooms=bldg.get("rooms",[])
    sw=site.get("width",20); sd=site.get("depth",30)
    bw=bldg.get("width",10); bd=bldg.get("depth",8)
    bx=site.get("building_x",5); by=site.get("building_y",10)
    fl=bldg.get("floors",1); S=16; M=80; TH=70; SH=50
    svg_w=sw*S+M*2+160; svg_h=sd*S+M*2+TH+SH
    bx0=M+bx*S; by0=M+TH+(sd-by-bd)*S; bww=bw*S; bdd=bd*S
    s=f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">
<defs><marker id="a" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#333"/></marker>
<pattern id="g" patternUnits="userSpaceOnUse" width="20" height="20"><rect width="20" height="20" fill="#e8f5e9"/><circle cx="5" cy="5" r="1" fill="#a5d6a7"/><circle cx="15" cy="15" r="1" fill="#a5d6a7"/></pattern></defs>
<text x="{M}" y="30" font-family="Arial" font-size="18" font-weight="bold" fill="#333">ПЛАН УЧАСТКА</text>
<text x="{M}" y="50" font-family="Arial" font-size="12" fill="#666">{plan.get("description","")}</text>
<text x="{M}" y="64" font-family="Arial" font-size="10" fill="#999">Масштаб 1:100</text>
<rect x="{M}" y="{M+TH}" width="{sw*S}" height="{sd*S}" fill="url(#g)" stroke="#2d5016" stroke-width="2" rx="1"/>
<polygon points="{M+bx*S},{M+TH+sd*S} {M+(bx+3)*S},{M+TH+sd*S} {M+(bx+3)*S},{by0+bdd} {M+bx*S},{by0+bdd}" fill="#d7ccc8" stroke="#a1887f" stroke-width="1"/>
<rect x="{bx0}" y="{by0}" width="{bww}" height="{bdd}" fill="#fff" stroke="#333" stroke-width="2.5"/>
<rect x="{bx0}" y="{by0}" width="{bww}" height="{bdd}" fill="#f5f5f5"/>'''
    for rm in rooms:
        rx=bx0+rm["x"]*S; ry=by0+(bd-rm["y"]-rm["depth"])*S; rw=rm["width"]*S; rd=rm["depth"]*S; ar=rm["width"]*rm["depth"]
        s+=f'''
<rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" fill="none" stroke="#666" stroke-width="1.5"/>
<text x="{rx+rw/2}" y="{ry+rd/2-6}" font-family="Arial" font-size="11" font-weight="bold" fill="#333" text-anchor="middle">{rm["name"]}</text>
<text x="{rx+rw/2}" y="{ry+rd/2+8}" font-family="Arial" font-size="9" fill="#888" text-anchor="middle">{ar:.1f} м²</text>'''
    s+=f'''
<rect x="{bx0+bww/2-10}" y="{by0+bdd-3}" width="20" height="6" fill="#ff9800" stroke="#e65100" stroke-width="1" rx="1"/>
<text x="{bx0+bww/2}" y="{by0+bdd+18}" font-family="Arial" font-size="9" fill="#e65100" text-anchor="middle">ВХОД</text>
<line x1="{M}" y1="{M+TH+sd*S+15}" x2="{M+sw*S}" y2="{M+TH+sd*S+15}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M+sw*S/2}" y="{M+TH+sd*S+30}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle">{sw} м</text>
<line x1="{M-15}" y1="{M+TH}" x2="{M-15}" y2="{M+TH+sd*S}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M-25}" y="{M+TH+sd*S/2}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle" transform="rotate(-90,{M-25},{M+TH+sd*S/2})">{sd} м</text>
<line x1="{bx0}" y1="{by0-8}" x2="{bx0+bww}" y2="{by0-8}" stroke="#e65100" stroke-width="0.8"/>
<text x="{bx0+bww/2}" y="{by0-12}" font-family="Arial" font-size="10" fill="#e65100" text-anchor="middle">{bw} м</text>
<line x1="{bx0+bww+8}" y1="{by0}" x2="{bx0+bww+8}" y2="{by0+bdd}" stroke="#e65100" stroke-width="0.8"/>
<text x="{bx0+bww+16}" y="{by0+bdd/2}" font-family="Arial" font-size="10" fill="#e65100" text-anchor="middle" transform="rotate(90,{bx0+bww+16},{by0+bdd/2})">{bd} м</text>
<line x1="{svg_w-80}" y1="{M+TH+60}" x2="{svg_w-80}" y2="{M+TH+20}" stroke="#333" stroke-width="2"/>
<polygon points="{svg_w-80},{M+TH+15} {svg_w-86},{M+TH+30} {svg_w-74},{M+TH+30}" fill="#e53935"/>
<text x="{svg_w-80}" y="{M+TH+10}" font-family="Arial" font-size="12" font-weight="bold" fill="#e53935" text-anchor="middle">N</text>
<rect x="{M}" y="{svg_h-SH}" width="{sw*S}" height="{SH}" fill="none" stroke="#333" stroke-width="1.5"/>
<line x1="{M}" y1="{svg_h-SH+20}" x2="{M+sw*S}" y2="{svg_h-SH+20}" stroke="#333" stroke-width="1"/>
<line x1="{M+sw*S*0.6}" y1="{svg_h-SH}" x2="{M+sw*S*0.6}" y2="{svg_h}" stroke="#333" stroke-width="1"/>
<text x="{M+5}" y="{svg_h-SH+14}" font-family="Arial" font-size="9" fill="#333">План участка</text>
<text x="{M+sw*S*0.6+5}" y="{svg_h-SH+14}" font-family="Arial" font-size="9" fill="#333">Масштаб 1:100</text>
<text x="{M+5}" y="{svg_h-SH+38}" font-family="Arial" font-size="8" fill="#666">{datetime.now().strftime("%d.%m.%Y")}</text>
<text x="{M+sw*S*0.6+5}" y="{svg_h-SH+38}" font-family="Arial" font-size="8" fill="#666">Archiq AI</text>
</svg>'''
    return s

def gen_floor_svg(plan: dict) -> str:
    bldg=plan.get("building",{}); rooms=bldg.get("rooms",[])
    bw=bldg.get("width",10); bd=bldg.get("depth",8); fl=bldg.get("floors",1)
    S=45; M=80; TH=70; SH=50; TY=30
    svg_w=bw*S+M*2; svg_h=bd*S+M*2+TH+SH+TY
    s=f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">
<defs><marker id="a" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#333"/></marker></defs>
<text x="{M}" y="30" font-family="Arial" font-size="18" font-weight="bold" fill="#333">ПЛАН {fl}-го ЭТАЖА</text>
<text x="{M}" y="50" font-family="Arial" font-size="12" fill="#666">{plan.get("description","")}</text>
<text x="{M}" y="64" font-family="Arial" font-size="10" fill="#999">Масштаб 1:100</text>
<line x1="{M}" y1="{M+TH-15}" x2="{M}" y2="{M+TH+bd*S+15}" stroke="#e53935" stroke-width="0.8" stroke-dasharray="8,4"/>
<line x1="{M+bw*S}" y1="{M+TH-15}" x2="{M+bw*S}" y2="{M+TH+bd*S+15}" stroke="#e53935" stroke-width="0.8" stroke-dasharray="8,4"/>
<line x1="{M-15}" y1="{M+TH}" x2="{M+bw*S+15}" y2="{M+TH}" stroke="#e53935" stroke-width="0.8" stroke-dasharray="8,4"/>
<line x1="{M-15}" y1="{M+TH+bd*S}" x2="{M+bw*S+15}" y2="{M+TH+bd*S}" stroke="#e53935" stroke-width="0.8" stroke-dasharray="8,4"/>
<circle cx="{M}" cy="{M+TH-20}" r="8" fill="none" stroke="#e53935" stroke-width="1"/><text x="{M}" y="{M+TH-16}" font-family="Arial" font-size="8" fill="#e53935" text-anchor="middle">А</text>
<circle cx="{M+bw*S}" cy="{M+TH-20}" r="8" fill="none" stroke="#e53935" stroke-width="1"/><text x="{M+bw*S}" y="{M+TH-16}" font-family="Arial" font-size="8" fill="#e53935" text-anchor="middle">Б</text>'''
    ta=0
    for rm in rooms:
        rx=M+rm["x"]*S; ry=M+TH+(bd-rm["y"]-rm["depth"])*S; rw=rm["width"]*S; rd=rm["depth"]*S; ar=rm["width"]*rm["depth"]; ta+=ar
        s+=f'''
<rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" fill="#fafafa" stroke="#333" stroke-width="2"/>
<text x="{rx+rw/2}" y="{ry+rd/2-8}" font-family="Arial" font-size="12" font-weight="bold" fill="#333" text-anchor="middle">{rm["name"]}</text>
<text x="{rx+rw/2}" y="{ry+rd/2+8}" font-family="Arial" font-size="10" fill="#888" text-anchor="middle">{ar:.1f} м²</text>'''
    s+=f'''
<rect x="{M+bw*S/2-12}" y="{M+TH+bd*S-4}" width="24" height="8" fill="#ff9800" stroke="#e65100" stroke-width="1.5" rx="1"/>
<text x="{M+bw*S/2}" y="{M+TH+bd*S+18}" font-family="Arial" font-size="10" font-weight="bold" fill="#e65100" text-anchor="middle">ВХОД</text>
<line x1="{M}" y1="{M+TH+bd*S+30}" x2="{M+bw*S}" y2="{M+TH+bd*S+30}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M+bw*S/2}" y="{M+TH+bd*S+44}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle">{bw} м</text>
<line x1="{M-30}" y1="{M+TH}" x2="{M-30}" y2="{M+TH+bd*S}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M-40}" y="{M+TH+bd*S/2}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle" transform="rotate(-90,{M-40},{M+TH+bd*S/2})">{bd} м</text>
<text x="{M}" y="{M+TH+bd*S+70}" font-family="Arial" font-size="13" font-weight="bold" fill="#333">ЭКСПЛИКАЦИЯ ПОМЕЩЕНИЙ</text>
<rect x="{M}" y="{M+TH+bd*S+75}" width="{bw*S}" height="{25+len(rooms)*18}" fill="none" stroke="#333" stroke-width="1"/>
<rect x="{M}" y="{M+TH+bd*S+75}" width="{bw*S}" height="18" fill="#e0e0e0"/>
<text x="{M+10}" y="{M+TH+bd*S+87}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">№</text>
<text x="{M+30}" y="{M+TH+bd*S+87}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">Наименование</text>
<text x="{M+bw*S-80}" y="{M+TH+bd*S+87}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">Площадь, м²</text>'''
    for i,rm in enumerate(rooms):
        y=M+TH+bd*S+75+18+i*18; ar=rm["width"]*rm["depth"]; bg="#fff" if i%2==0 else "#f5f5f5"
        s+=f'''
<rect x="{M}" y="{y}" width="{bw*S}" height="17" fill="{bg}"/>
<text x="{M+15}" y="{y+12}" font-family="Arial" font-size="9" fill="#333">{i+1}</text>
<text x="{M+30}" y="{y+12}" font-family="Arial" font-size="9" fill="#333">{rm["name"]}</text>
<text x="{M+bw*S-80}" y="{y+12}" font-family="Arial" font-size="9" fill="#333">{ar:.1f}</text>'''
    s+=f'''
<rect x="{M}" y="{M+TH+bd*S+75+18+len(rooms)*18}" width="{bw*S}" height="18" fill="#e0e0e0"/>
<text x="{M+30}" y="{M+TH+bd*S+75+18+len(rooms)*18+12}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">ИТОГО</text>
<text x="{M+bw*S-80}" y="{M+TH+bd*S+75+18+len(rooms)*18+12}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">{ta:.1f}</text>
<rect x="{M}" y="{svg_h-SH}" width="{bw*S}" height="{SH}" fill="none" stroke="#333" stroke-width="1.5"/>
<line x1="{M}" y1="{svg_h-SH+20}" x2="{M+bw*S}" y2="{svg_h-SH+20}" stroke="#333" stroke-width="1"/>
<line x1="{M+bw*S*0.6}" y1="{svg_h-SH}" x2="{M+bw*S*0.6}" y2="{svg_h}" stroke="#333" stroke-width="1"/>
<text x="{M+5}" y="{svg_h-SH+14}" font-family="Arial" font-size="9" fill="#333">План {fl}-го этажа</text>
<text x="{M+bw*S*0.6+5}" y="{svg_h-SH+14}" font-family="Arial" font-size="9" fill="#333">Масштаб 1:100</text>
<text x="{M+5}" y="{svg_h-SH+38}" font-family="Arial" font-size="8" fill="#666">{datetime.now().strftime("%d.%m.%Y")}</text>
<text x="{M+bw*S*0.6+5}" y="{svg_h-SH+38}" font-family="Arial" font-size="8" fill="#666">Archiq AI</text>
</svg>'''
    return s

def gen_specs(plan: dict) -> dict:
    bldg=plan.get("building",{}); site=plan.get("site",{}); rooms=bldg.get("rooms",[])
    ta=sum(r.get("width",0)*r.get("depth",0) for r in rooms)
    la=sum(r.get("width",0)*r.get("depth",0) for r in rooms if r.get("name") in ("Гостиная","Спальня","Детская","Кабинет"))
    sa=site.get("width",1)*site.get("depth",1)
    return {"total_area":round(ta,1),"living_area":round(la,1),"floors":bldg.get("floors",1),
            "rooms_count":len(rooms),"building_dimensions":f"{bldg.get('width',0)}x{bldg.get('depth',0)}м",
            "site_dimensions":f"{site.get('width',0)}x{site.get('depth',0)}м",
            "building_footprint":round(ta,1),"site_area":round(sa,1),
            "site_coverage":round(ta/sa*100,1),"perimeter":round(2*(bldg.get('width',0)+bldg.get('depth',0)),1),
            "compliance":check_snip(rooms),"description":plan.get("description","")}

def gen_plan(project: dict) -> dict:
    area=project.get('area',100); fl=project.get('floors',1); rc=project.get('rooms',3)
    sw=project.get('site_width',20); sd=project.get('site_depth',30); bt=project.get('building_type','жилой дом')
    fa=area/fl; bw=round(math.sqrt(fa)*1.3,1); bd=round(fa/(bw if bw else 10),1)
    rooms=get_rooms(bt,fa,rc); placed=pack_rooms(rooms,bw,bd)
    rd=[{"name":r.name,"width":r.width,"depth":r.depth,"x":round(r.x,1),"y":round(r.y,1),"is_wet":r.is_wet,"has_window":r.has_window} for r in placed]
    return {"building":{"width":bw,"depth":bd,"floors":fl,"rooms":rd,"entrance":{"x":round(bw/2,1),"y":0}},
            "site":{"width":sw,"depth":sd,"building_x":round((sw-bw)/3,1),"building_y":round((sd-bd)/3,1),"parking":True,"garden":True,"driveway":True},
            "description":f"{bt}, {area}м², {fl}эт., {len(placed)}пом."}

# ==================== APP ====================

app = FastAPI(title="Archiq AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LANDING="""<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Archiq AI</title><style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem}.c{max-width:600px;text-align:center}.l{font-size:3rem;font-weight:800;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}.s{color:#94a3b8;margin-bottom:2rem}.st{display:inline-flex;align-items:center;gap:.5rem;background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);padding:.5rem 1rem;border-radius:999px;margin-bottom:2rem}.d{width:8px;height:8px;border-radius:50%;background:#22c55e;animation:p 2s infinite}@keyframes p{0%,100%{opacity:1}50%{opacity:.4}}.e{text-align:left;background:rgba(30,41,59,.8);border:1px solid #334155;border-radius:12px;padding:1.5rem}.e h2{font-size:1rem;color:#94a3b8;margin-bottom:1rem;text-transform:uppercase}.r{display:flex;gap:1rem;padding:.6rem 0;border-bottom:1px solid #1e293b}.r:last-child{border-bottom:none}.m{font-family:monospace;font-size:.8rem;font-weight:700;min-width:55px;padding:.15rem .4rem;border-radius:4px;text-align:center}.g{background:rgba(56,189,248,.15);color:#38bdf8}.p{background:rgba(168,85,247,.15);color:#a855f7}.t{font-family:monospace;font-size:.85rem;color:#e2e8f0}.x{color:#64748b;font-size:.8rem;margin-left:auto}.f{margin-top:2rem;color:#475569;font-size:.8rem}.f a{color:#818cf8;text-decoration:none}</style></head>
<body><div class="c"><div class="l">🏗️ Archiq AI</div><div class="s">Генератор профессиональных архитектурных планов</div><div class="st"><span class="d"></span> Сервис работает</div>
<div class="e"><h2>API</h2><div class="r"><span class="m g">GET</span><span class="t">/health</span><span class="x">Статус</span></div><div class="r"><span class="m p">POST</span><span class="t">/generate-plan</span><span class="x">Генерация плана</span></div><div class="r"><span class="m p">POST</span><span class="t">/analyze-site-plan</span><span class="x">Анализ участка</span></div><div class="r"><span class="m g">GET</span><span class="t">/site-plan-svg?id=...</span><span class="x">SVG план</span></div><div class="r"><span class="m g">GET</span><span class="t">/floor-plan-svg?id=...</span><span class="x">SVG этаж</span></div><div class="r"><span class="m g">GET</span><span class="t">/specs?id=...</span><span class="x">Спецификация</span></div></div>
<div class="f"><p>GitHub: <a href="https://github.com/K09-0/ARCHIQ-AI" target="_blank">K09-0/ARCHIQ-AI</a></p></div></div></body></html>"""

def init_db():
    conn=sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,name TEXT,building_type TEXT,area REAL,floors INTEGER,rooms INTEGER,site_width REAL,site_depth REAL,requirements TEXT,plan_json TEXT,created_at TEXT)")
    conn.commit(); conn.close()

@app.on_event("startup")
def startup(): init_db()

@app.get("/",response_class=HTMLResponse)
def root(): return LANDING

@app.get("/health")
def health(): return {"status":"ok","gemini":bool(GEMINI_API_KEY),"hf":bool(HF_API_KEY)}

@app.post("/generate-plan")
def plan(name:str=Form("Проект"),building_type:str=Form("жилой дом"),area:float=Form(100),floors:int=Form(1),rooms:int=Form(3),site_width:float=Form(20),site_depth:float=Form(30),requirements:str=Form("")):
    proj={"name":name,"building_type":building_type,"area":area,"floors":floors,"rooms":rooms,"site_width":site_width,"site_depth":site_depth,"requirements":requirements}
    pl=gen_plan(proj); sp=gen_specs(pl); pid=datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUTPUT_DIR/f"{pid}_site.svg").write_text(gen_site_svg(pl))
    (OUTPUT_DIR/f"{pid}_floor.svg").write_text(gen_floor_svg(pl))
    conn=sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO projects VALUES(?,?,?,?,?,?,?,?,?,?,?)",(pid,name,building_type,area,floors,rooms,site_width,site_depth,requirements,json.dumps(pl),datetime.now().isoformat()))
    conn.commit(); conn.close()
    return {"project_id":pid,"plan":pl,"specs":sp}

@app.get("/site-plan-svg")
def site_svg(project_id:str):
    p=OUTPUT_DIR/f"{project_id}_site.svg"
    return HTMLResponse(content=p.read_text()) if p.exists() else JSONResponse({"error":"Not found"},status_code=404)

@app.get("/floor-plan-svg")
def floor_svg(project_id:str):
    p=OUTPUT_DIR/f"{project_id}_floor.svg"
    return HTMLResponse(content=p.read_text()) if p.exists() else JSONResponse({"error":"Not found"},status_code=404)

@app.get("/specs")
def specs(project_id:str):
    conn=sqlite3.connect(DB_PATH)
    r=conn.execute("SELECT plan_json FROM projects WHERE id=?",(project_id,)).fetchone(); conn.close()
    return gen_specs(json.loads(r[0])) if r else JSONResponse({"error":"Not found"},status_code=404)

@app.get("/projects")
def list_proj():
    conn=sqlite3.connect(DB_PATH); conn.row_factory=sqlite3.Row
    rows=conn.execute("SELECT id,name,building_type,area,floors,rooms,created_at FROM projects ORDER BY created_at DESC").fetchall(); conn.close()
    return [dict(r) for r in rows]

@app.post("/analyze-site-plan")
async def analyze(file:UploadFile=File(...)):
    if not GEMINI_API_KEY: return {"error":"GEMINI_API_KEY not set"}
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        content=await file.read()
        model=genai.GenerativeModel("gemini-1.5-flash")
        resp=model.generate_content(["Архитектор: проанализируй план участка. JSON: {\"site_width\":X,\"site_depth\":Y}", {"mime_type":"image/jpeg","data":base64.b64encode(content).decode()}])
        m=re.search(r'\{[\s\S]*\}',resp.text)
        return json.loads(m.group()) if m else {"text":resp.text}
    except ImportError: return {"error":"Install google-generativeai"}
    except Exception as e: return {"error":str(e)}
