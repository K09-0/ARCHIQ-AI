#!/usr/bin/env python3
"""Archiq AI v2 — Профессиональный генератор архитектурных планов."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional, List, Dict, Any
import sqlite3, os, json, math, random, base64, re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

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
    min_area: float
    max_area: float
    min_w: float
    min_d: float
    preferred_wall: str = "any"  # N, S, E, W, any
    is_wet: bool = False
    has_window: bool = True
    adj_to: List[str] = field(default_factory=list)
    w: float = 0; d: float = 0; x: float = 0; y: float = 0

def catalog(btype: str, area: float, n: int) -> List[Room]:
    """Generate room catalog based on building type."""
    r = []
    if btype in ("жилой дом", "дача"):
        r.append(Room("Гостиная", max(18,area*0.22), 35, 4.0, 3.5, "S", adj_to=["Прихожая","Кухня"]))
        r.append(Room("Кухня", max(9,area*0.12), 18, 2.8, 2.5, "N", is_wet=True, adj_to=["Гостиная","Прихожая"]))
        r.append(Room("Прихожая", 5, 12, 2.0, 1.8, "any", has_window=False, adj_to=["Гостиная","Коридор"]))
        r.append(Room("Ванная", 3.5, 7, 1.8, 1.8, "N", is_wet=True, adj_to=["Коридор"]))
        r.append(Room("Коридор", 4, 10, 1.2, 2.0, "any", has_window=False))
        
        beds = max(1, n - 2)
        for i in range(beds):
            nm = "Спальня" if i==0 else (f"Детская" if i==beds-1 and n>4 else f"Спальня {i+1}")
            pref = "S" if "Спальня" in nm else "E"
            r.append(Room(nm, 10, 18, 3.0, 2.8, pref, adj_to=["Коридор"]))
        if area > 120:
            r.append(Room("Кабинет", 8, 14, 2.5, 2.5, "E", adj_to=["Коридор"]))
        if area > 150:
            r.append(Room("Гардеробная", 4, 8, 1.8, 1.8, "any", has_window=False))
            r.append(Room("Кладовая", 2, 5, 1.2, 1.2, "any", has_window=False))
        if area > 100:
            r.append(Room("Туалет", 1.5, 3, 1.2, 1.2, "any", is_wet=True, has_window=False, adj_to=["Коридор"]))
    elif btype == "баня":
        r = [Room("Парная",5,9,2.2,2.0,"N",False), Room("Моечная",5,9,2.2,2.0,"any",True),
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
    """Smart room placement with adjacency optimization."""
    placed = []
    occ = []  # (x,y,w,d)
    
    # Sort: entrance first, then public rooms, then private
    order = {"Прихожая":0, "Гостиная":1, "Кухня":2, "Коридор":3, "Ванная":4, "Туалет":5, "Предбанник":6}
    rooms.sort(key=lambda r: order.get(r.name, 10))
    
    for room in rooms:
        # Calculate dimensions
        area = (room.min_area + room.max_area) / 2
        best = None; best_score = -1
        
        for _ in range(200):  # Try random positions
            # Random aspect ratio within constraints
            rw = random.uniform(room.min_w, min(bw*0.7, math.sqrt(area)*1.5))
            rw = max(room.min_w, rw)
            rd = area / rw
            if rd < room.min_d or rd > bd*0.7: continue
            if rw * rd < room.min_area * 0.9 or rw * rd > room.max_area: continue
            
            # Random position
            x = random.uniform(0, bw - rw)
            y = random.uniform(0, bd - rd)
            x = round(x, 1); y = round(y, 1)
            
            # Check overlap
            overlaps = any(not(x+rw<=ox+0.1 or ox+ow<=x+0.1 or y+rd<=oy+0.1 or oy+od<=y+0.1) for ox,oy,ow,od in occ)
            if overlaps: continue
            
            # Score: prefer edges for windows, adjacency for connected rooms
            score = 0
            # Prefer walls for windows
            if room.has_window:
                if y < 0.5: score += 10  # bottom wall (S)
                if y+rd > bd-0.5: score += 8  # top wall (N)
                if x < 0.5: score += 5  # left wall (W)
                if x+rw > bw-0.5: score += 5  # right wall (E)
            # Prefer adjacency
            for adj_name in room.adj_to:
                for p in placed:
                    if p.name == adj_name:
                        dist = math.sqrt((x+rw/2 - p.x-p.w/2)**2 + (y+rd/2 - p.y-p.d/2)**2)
                        score += max(0, 20 - dist*3)
            # Prefer compact layout
            score -= (x + y) * 0.5
            
            if score > best_score:
                best_score = score
                best = (x, y, round(rw, 1), round(rd, 1))
        
        if best:
            room.x, room.y, room.w, room.d = best
            placed.append(room)
            occ.append((room.x, room.y, room.w, room.d))
    
    return placed

def snip_check(rooms: List[Dict]) -> List[str]:
    """SNiP compliance check."""
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
            if cond:
                r.append(("✅" if ok else "⚠️") + " " + msg)
                break
    return r

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
<defs>
<marker id="a" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#1a73e8"/></marker>
<pattern id="g" patternUnits="userSpaceOnUse" width="20" height="20"><rect width="20" height="20" fill="#e8f5e9"/><circle cx="5" cy="5" r="1" fill="#a5d6a7"/><circle cx="15" cy="15" r="1" fill="#a5d6a7"/></pattern>
<pattern id="p" patternUnits="userSpaceOnUse" width="30" height="20"><rect width="30" height="20" fill="#e0e0e0"/><line x1="15" y1="0" x2="15" y2="20" stroke="#bbb"/></pattern>
</defs>
<text x="{M}" y="30" font-family="Arial" font-size="18" font-weight="bold" fill="#333">ПЛАН УЧАСТКА</text>
<text x="{M}" y="50" font-family="Arial" font-size="12" fill="#666">{plan.get("description","")}</text>
<text x="{M}" y="64" font-family="Arial" font-size="10" fill="#999">Масштаб 1:100 | Дата: {datetime.now().strftime("%d.%m.%Y")}</text>
<rect x="{M}" y="{M+TH}" width="{sw*S}" height="{sd*S}" fill="url(#g)" stroke="#2d5016" stroke-width="2"/>
<rect x="{X0}" y="{Y0}" width="{WW}" height="{HH}" fill="#fff" stroke="#333" stroke-width="3"/>'''
    
    # Rooms
    for rm in rooms:
        rx=X0+rm["x"]*S; ry=Y0+(bd-rm["y"]-rm["d"])*S; rw=rm["w"]*S; rd=rm["d"]*S; ar=rm["w"]*rm["d"]
        s+=f'''
<rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" fill="none" stroke="#555" stroke-width="1.5"/>
<text x="{rx+rw/2}" y="{ry+rd/2-5}" font-family="Arial" font-size="11" font-weight="bold" fill="#333" text-anchor="middle">{rm["name"]}</text>
<text x="{rx+rw/2}" y="{ry+rd/2+9}" font-family="Arial" font-size="9" fill="#888" text-anchor="middle">{ar:.1f}м²</text>'''
    
    # Entrance, driveway, parking
    s+=f'''
<rect x="{X0+WW/2-10}" y="{Y0+HH-3}" width="20" height="6" fill="#ff9800" stroke="#e65100" rx="1"/>
<text x="{X0+WW/2}" y="{Y0+HH+16}" font-family="Arial" font-size="9" fill="#e65100" text-anchor="middle">ВХОД</text>
<polygon points="{M+bx*S},{M+TH+sd*S} {M+(bx+3)*S},{M+TH+sd*S} {M+(bx+3)*S},{Y0+HH} {M+bx*S},{Y0+HH}" fill="#d7ccc8" stroke="#a1887f"/>
<rect x="{M+(sw-6)*S}" y="{M+TH+(sd-8)*S}" width="{5*S}" height="{6*S}" fill="url(#p)" stroke="#999"/>
<text x="{M+(sw-3.5)*S}" y="{M+TH+(sd-4)*S}" font-family="Arial" font-size="10" fill="#666" text-anchor="middle">Парковка</text>
<text x="{M+(sw-3.5)*S}" y="{M+TH+(sd-3)*S}" font-family="Arial" font-size="9" fill="#888" text-anchor="middle">5×6м</text>'''
    
    # Dimensions
    s+=f'''
<line x1="{M}" y1="{M+TH+sd*S+15}" x2="{M+sw*S}" y2="{M+TH+sd*S+15}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M+sw*S/2}" y="{M+TH+sd*S+30}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle">{sw} м</text>
<line x1="{M+sw*S+15}" y1="{M+TH}" x2="{M+sw*S+15}" y2="{M+TH+sd*S}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M+sw*S+30}" y="{M+TH+sd*S/2}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle" transform="rotate(90,{M+sw*S+30},{M+TH+sd*S/2})">{sd} м</text>
<line x1="{X0}" y1="{Y0-10}" x2="{X0+WW}" y2="{Y0-10}" stroke="#e65100" stroke-width="0.8"/>
<text x="{X0+WW/2}" y="{Y0-14}" font-family="Arial" font-size="10" fill="#e65100" text-anchor="middle">{bw} м</text>'''
    
    # North arrow
    s+=f'''
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
    
    s=f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs><marker id="a" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#1a73e8"/></marker></defs>
<text x="{M}" y="30" font-family="Arial" font-size="18" font-weight="bold" fill="#333">ПЛАН {fl}-го ЭТАЖА</text>
<text x="{M}" y="50" font-family="Arial" font-size="12" fill="#666">{plan.get("description","")}</text>
<text x="{M}" y="64" font-family="Arial" font-size="10" fill="#999">Масштаб 1:100</text>
<rect x="{M}" y="{M+TH}" width="{bw*S}" height="{bd*S}" fill="#fafafa" stroke="#333" stroke-width="3"/>'''
    
    ta=0
    for rm in rooms:
        rx=M+rm["x"]*S; ry=M+TH+(bd-rm["y"]-rm["d"])*S; rw=rm["w"]*S; rd=rm["d"]*S; ar=rm["w"]*rm["d"]; ta+=ar
        s+=f'''
<rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" fill="none" stroke="#444" stroke-width="2"/>
<text x="{rx+rw/2}" y="{ry+rd/2-6}" font-family="Arial" font-size="12" font-weight="bold" fill="#333" text-anchor="middle">{rm["name"]}</text>
<text x="{rx+rw/2}" y="{ry+rd/2+10}" font-family="Arial" font-size="10" fill="#888" text-anchor="middle">{ar:.1f}м²</text>'''
    
    # Entrance
    s+=f'''
<rect x="{M+bw*S/2-12}" y="{M+TH+bd*S-4}" width="24" height="8" fill="#ff9800" stroke="#e65100" stroke-width="1.5" rx="1"/>
<text x="{M+bw*S/2}" y="{M+TH+bd*S+18}" font-family="Arial" font-size="10" font-weight="bold" fill="#e65100" text-anchor="middle">ВХОД</text>'''
    
    # Dimensions
    s+=f'''
<line x1="{M}" y1="{M+TH+bd*S+15}" x2="{M+bw*S}" y2="{M+TH+bd*S+15}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M+bw*S/2}" y="{M+TH+bd*S+30}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle">{bw} м</text>
<line x1="{M-20}" y1="{M+TH}" x2="{M-20}" y2="{M+TH+bd*S}" stroke="#1a73e8" stroke-width="1" marker-start="url(#a)" marker-end="url(#a)"/>
<text x="{M-30}" y="{M+TH+bd*S/2}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle" transform="rotate(-90,{M-30},{M+TH+bd*S/2})">{bd} м</text>'''
    
    # Room schedule
    ty=M+TH+bd*S+55
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

# ==================== PLAN GENERATOR ====================

def gen_plan(proj: dict) -> dict:
    area=proj["area"]; fl=proj["floors"]; rc=proj["rooms"]
    sw=proj["site_width"]; sd=proj["site_depth"]; bt=proj["building_type"]
    fa=area/fl; bw=round(math.sqrt(fa)*1.25,1); bd=round(fa/bw,1)
    rooms=catalog(bt,fa,rc); placed=layout(rooms,bw,bd)
    rd=[{"name":r.name,"w":r.w,"d":r.d,"x":round(r.x,1),"y":round(r.y,1),
         "is_wet":r.is_wet,"has_window":r.has_window} for r in placed]
    return {"building":{"width":bw,"depth":bd,"floors":fl,"rooms":rd,"entrance":{"x":round(bw/2,1),"y":0}},
            "site":{"width":sw,"depth":sd,"building_x":round((sw-bw)/3,1),"building_y":round((sd-bd)/3,1),
                    "parking":True,"garden":True,"driveway":True},
            "description":f"{bt}, {area}м², {fl}эт., {len(placed)}пом."}

# ==================== DXF EXPORT ====================

def export_dxf(plan: dict) -> str:
    """Generate DXF file content."""
    b=plan["building"]; rooms=b["rooms"]; bw=b["width"]; bd=b["depth"]
    SCALE=1000  # 1m = 1000 DXF units
    
    dxf = f'''  0
SECTION
  2
ENTITIES
'''
    # Walls (exterior)
    pts = [(0,0), (bw*SCALE,0), (bw*SCALE,bd*SCALE), (0,bd*SCALE), (0,0)]
    for i in range(len(pts)-1):
        dxf += f'''  0
POLYLINE
  8
WALLS
 66
1
  0
VERTEX
  8
WALLS
 10
{pts[i][0]}
 20
{pts[i][1]}
 30
0
  0
SEQEND
'''
    
    # Room divisions
    for rm in rooms:
        x=rm["x"]*SCALE; y=rm["y"]*SCALE; w=rm["w"]*SCALE; d=rm["d"]*SCALE
        dxf += f'''  0
POLYLINE
  8
ROOM-{rm["name"].upper().replace(" ","_")}
 66
1
  0
VERTEX
  8
ROOM-{rm["name"].upper().replace(" ","_")}
 10
{x}
 20
{y}
 30
0
  0
VERTEX
  8
ROOM-{rm["name"].upper().replace(" ","_")}
 10
{x+w}
 20
{y}
 30
0
  0
VERTEX
  8
ROOM-{rm["name"].upper().replace(" ","_")}
 10
{x+w}
 20
{y+d}
 30
0
  0
VERTEX
  8
ROOM-{rm["name"].upper().replace(" ","_")}
 10
{x}
 20
{y+d}
 30
0
  0
SEQEND
'''
    
    # Text labels
    for rm in rooms:
        x=(rm["x"]+rm["w"]/2)*SCALE; y=(rm["y"]+rm["d"]/2)*SCALE
        dxf += f'''  0
TEXT
  8
TEXT
 10
{x}
 20
{y}
 30
0
 40
500
  1
{rm["name"]} {rm["w"]}x{rm["d"]}m
'''
    
    dxf += '''  0
ENDSEC
  0
EOF
'''
    return dxf

# ==================== APP ====================

app = FastAPI(title="Archiq AI v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LANDING = """<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Archiq AI</title><style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem}.c{max-width:600px;text-align:center}.l{font-size:3rem;font-weight:800;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}.s{color:#94a3b8;margin-bottom:2rem}.st{display:inline-flex;align-items:center;gap:.5rem;background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);padding:.5rem 1rem;border-radius:999px;margin-bottom:2rem}.d{width:8px;height:8px;border-radius:50%;background:#22c55e;animation:p 2s infinite}@keyframes p{0%,100%{opacity:1}50%{opacity:.4}}.e{text-align:left;background:rgba(30,41,59,.8);border:1px solid #334155;border-radius:12px;padding:1.5rem}.e h2{font-size:1rem;color:#94a3b8;margin-bottom:1rem;text-transform:uppercase}.r{display:flex;gap:1rem;padding:.6rem 0;border-bottom:1px solid #1e293b}.r:last-child{border-bottom:none}.m{font-family:monospace;font-size:.8rem;font-weight:700;min-width:55px;padding:.15rem .4rem;border-radius:4px;text-align:center}.g{background:rgba(56,189,248,.15);color:#38bdf8}.p{background:rgba(168,85,247,.15);color:#a855f7}.t{font-family:monospace;font-size:.85rem;color:#e2e8f0}.x{color:#64748b;font-size:.8rem;margin-left:auto}.f{margin-top:2rem;color:#475569;font-size:.8rem}.f a{color:#818cf8;text-decoration:none}</style></head>
<body><div class="c"><div class="l">🏗️ Archiq AI v2</div><div class="s">Профессиональный генератор архитектурных планов</div><div class="st"><span class="d"></span> Работает</div>
<div class="e"><h2>API</h2><div class="r"><span class="m g">GET</span><span class="t">/health</span><span class="x">Статус</span></div><div class="r"><span class="m p">POST</span><span class="t">/generate</span><span class="x">Генерация плана</span></div><div class="r"><span class="m g">GET</span><span class="t">/site-svg?id=...</span><span class="x">SVG план участка</span></div><div class="r"><span class="m g">GET</span><span class="t">/floor-svg?id=...</span><span class="x">SVG поэтажный план</span></div><div class="r"><span class="m g">GET</span><span class="t">/dxf?id=...</span><span class="x">DXF (AutoCAD)</span></div><div class="r"><span class="m g">GET</span><span class="t">/specs?id=...</span><span class="x">Спецификация</span></div></div>
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
async def analyze(file:UploadFile=File(...)):
    if not GEMINI_API_KEY: return {"error":"GEMINI_API_KEY not set"}
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        c=await file.read()
        m=genai.GenerativeModel("gemini-1.5-flash")
        r=m.generate_content(["Архитектор: проанализируй план участка. JSON: {\"site_width\":X,\"site_depth\":Y}", {"mime_type":"image/jpeg","data":base64.b64encode(c).decode()}])
        t=r.text; j=re.search(r'\{[\s\S]*\}',t)
        return json.loads(j.group()) if j else {"text":t}
    except ImportError: return {"error":"Install google-generativeai"}
    except Exception as e: return {"error":str(e)}
