"""Archiq AI — Генератор профессиональных архитектурных планов."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import sqlite3, os, json, io, math, random, base64, re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

OUTPUT_DIR = Path("/tmp/archiq-output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== CORE: ROOM LAYOUT ENGINE ====================

@dataclass
class Room:
    name: str
    min_area: float
    max_area: float
    min_dim: float  # minimum dimension
    preferred_pos: str  # "any", "sun", "north", "quiet"
    has_window: bool = True
    has_door: bool = True
    is_wet: bool = False  # bathroom/kitchen need plumbing
    width: float = 0
    depth: float = 0
    x: float = 0
    y: float = 0

@dataclass
class Wall:
    x1: float; y1: float; x2: float; y2: float; thickness: float = 0.3
    is_external: bool = True

@dataclass
class Window:
    wall_idx: int; pos: float; width: float; height: float = 1.5

@dataclass
class Door:
    wall_idx: int; pos: float; width: float = 0.9

# СНиП-требования для типовых помещений
ROOM_CATALOG = {
    "гостиная":      Room("Гостиная",      18, 35, 3.5, "sun", True, True),
    "кухня":         Room("Кухня",          8, 18, 2.5, "any", True, True, is_wet=True),
    "спальня":       Room("Спальня",       10, 20, 3.0, "quiet", True, True),
    "детская":       Room("Детская",       10, 18, 3.0, "sun", True, True),
    "кабинет":       Room("Кабинет",        8, 15, 2.5, "quiet", True, True),
    "ванная":        Room("Ванная",         3,  8, 1.6, "any", True, True, is_wet=True),
    "туалет":        Room("Туалет",         1.2,3,  1.2, "any", False, True, is_wet=True),
    "прихожая":      Room("Прихожая",       4, 10, 1.5, "any", False, True),
    "коридор":       Room("Коридор",        3,  8, 1.2, "any", False, True),
    "кладовая":      Room("Кладовая",       2,  6, 1.2, "any", False, False),
    "гардеробная":   Room("Гардеробная",    4,  8, 1.5, "any", False, False),
    "котельная":     Room("Котельная",      6, 12, 2.0, "any", True, True),
    "терраса":       Room("Терраса",        6, 20, 2.5, "sun", False, True),
    "гараж":         Room("Гараж",         18, 36, 3.5, "any", False, True),
}

def determine_rooms(building_type: str, area: float, rooms_count: int) -> List[Room]:
    """Determine room types based on building parameters."""
    result = []
    
    if building_type in ("жилой дом", "дача"):
        # Always needed
        result.append(Room("Прихожая", 4, 10, 1.5, "any", False, True))
        result.append(Room("Гостиная", max(16, area*0.2), max(30, area*0.3), 3.5, "sun", True, True))
        result.append(Room("Кухня", max(8, area*0.1), max(16, area*0.15), 2.5, "any", True, True, is_wet=True))
        result.append(Room("Ванная", 3, 6, 1.6, "any", True, True, is_wet=True))
        result.append(Room("Коридор", 3, 8, 1.2, "any", False, True))
        
        # Bedrooms
        bedroom_count = max(1, rooms_count - 2)  # minus living + kitchen
        for i in range(bedroom_count):
            name = "Спальня" if i == 0 else f"Спальня {i+1}"
            if i == bedroom_count - 1 and rooms_count > 4:
                name = "Детская"
            result.append(Room(name, 10, 18, 3.0, "quiet", True, True))
        
        # Optional rooms
        if area > 120:
            result.append(Room("Кабинет", 8, 12, 2.5, "quiet", True, True))
        if area > 150:
            result.append(Room("Гардеробная", 4, 8, 1.5, "any", False, False))
            result.append(Room("Кладовая", 2, 5, 1.2, "any", False, False))
        if area > 100:
            result.append(Room("Туалет", 1.5, 3, 1.2, "any", False, True, is_wet=True))
    
    elif building_type == "гараж":
        result.append(Room("Гараж", 18, 36, 3.5, "any", False, True))
        result.append(Room("Кладовая", 3, 8, 1.5, "any", False, False))
    
    elif building_type == "баня":
        result.append(Room("Парная", 4, 8, 1.8, "any", False, False))
        result.append(Room("Моечная", 4, 8, 1.8, "any", False, True, is_wet=True))
        result.append(Room("Предбанник", 6, 12, 2.0, "any", True, True))
        result.append(Room("Комната отдыха", 10, 20, 3.0, "any", True, True))
    
    elif building_type in ("офис", "магазин", "склад"):
        result.append(Room("Основное помещение", area*0.6, area*0.8, 4.0, "any", True, True))
        result.append(Room("Приёмная", 6, 12, 2.5, "any", True, True))
        result.append(Room("Санузел", 3, 6, 1.6, "any", True, True, is_wet=True))
        result.append(Room("Коридор", 3, 8, 1.2, "any", False, True))
    
    return result

def pack_rooms(rooms: List[Room], building_width: float, building_depth: float) -> List[Room]:
    """Pack rooms into building footprint using bin-packing heuristic."""
    target_area = building_width * building_depth
    
    # Calculate total room area and scale
    total_room_area = sum(r.min_area for r in rooms)
    corridor_area = target_area * 0.15  # 15% for corridors
    usable_area = target_area - corridor_area
    
    if total_room_area > usable_area:
        scale = usable_area / total_room_area
        for r in rooms:
            r.min_area *= scale
            r.max_area *= scale
    
    # Sort by area descending (larger first)
    rooms.sort(key=lambda r: r.min_area, reverse=True)
    
    # Grid-based placement
    grid_w = building_width
    grid_d = building_depth
    placed = []
    occupied = []  # list of (x, y, w, d)
    
    for room in rooms:
        area = room.min_area
        aspect = random.choice([(1.2, 0.83), (1.5, 0.67), (1.0, 1.0), (0.8, 1.25)])
        rw = math.sqrt(area * aspect[0])
        rd = area / rw
        
        # Clamp dimensions
        rw = max(room.min_dim, min(rw, grid_w * 0.7))
        rd = max(room.min_dim, min(rd, grid_d * 0.7))
        rd = area / rw  # recalc to maintain area
        
        # Find position using best-fit
        best_pos = None
        best_score = float('inf')
        
        # Try grid positions
        step = 0.5
        for gx in [0] + [round(s * step, 1) for s in range(1, int(grid_w/step))]:
            for gy in [0] + [round(s * step, 1) for s in range(1, int(grid_d/step))]:
                if gx + rw > grid_w or gy + rd > grid_d:
                    continue
                
                # Check overlap
                overlap = False
                for (ox, oy, ow, od) in occupied:
                    if not (gx + rw <= ox + 0.01 or ox + ow <= gx + 0.01 or
                            gy + rd <= oy + 0.01 or oy + od <= gy + 0.01):
                        overlap = True
                        break
                if overlap:
                    continue
                
                # Score: prefer corners and edges
                score = gx + gy  # prefer top-left
                if gx == 0: score -= 5  # prefer left wall (sun)
                if gy == 0: score -= 3  # prefer front
                if gx + rw >= grid_w - 0.1: score -= 2  # right wall OK
                
                if score < best_score:
                    best_score = score
                    best_pos = (gx, gy)
        
        if best_pos:
            room.x = best_pos[0]
            room.y = best_pos[1]
            room.width = round(rw, 1)
            room.depth = round(rd, 1)
            placed.append(room)
            occupied.append((room.x, room.y, room.width, room.depth))
    
    return placed

def generate_walls(rooms: List[Room], building_width: float, building_depth: float) -> List[Wall]:
    """Generate walls between rooms and exterior."""
    walls = []
    
    # Exterior walls
    walls.append(Wall(0, 0, building_width, 0, 0.3, True))        # bottom
    walls.append(Wall(0, building_depth, building_width, building_depth, 0.3, True))  # top
    walls.append(Wall(0, 0, 0, building_depth, 0.3, True))         # left
    walls.append(Wall(building_width, 0, building_width, building_depth, 0.3, True))  # right
    
    # Interior walls between adjacent rooms
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            # Check if rooms share a vertical edge
            if abs(r1.x + r1.width - r2.x) < 0.15 or abs(r2.x + r2.width - r1.x) < 0.15:
                y_start = max(r1.y, r2.y)
                y_end = min(r1.y + r1.depth, r2.y + r2.depth)
                if y_end > y_start + 0.5:
                    wx = r1.x + r1.width if abs(r1.x + r1.width - r2.x) < 0.15 else r2.x + r2.width
                    walls.append(Wall(wx, y_start, wx, y_end, 0.15, False))
            
            # Check if rooms share a horizontal edge
            if abs(r1.y + r1.depth - r2.y) < 0.15 or abs(r2.y + r2.depth - r1.y) < 0.15:
                x_start = max(r1.x, r2.x)
                x_end = min(r1.x + r1.width, r2.x + r2.width)
                if x_end > x_start + 0.5:
                    wy = r1.y + r1.depth if abs(r1.y + r1.depth - r2.y) < 0.15 else r2.y + r2.depth
                    walls.append(Wall(x_start, wy, x_end, wy, 0.15, False))
    
    return walls

def generate_windows_doors(rooms: List[Room], walls: List[Wall], building_width: float, building_depth: float):
    """Generate windows and doors."""
    windows = []
    doors = []
    
    for i, room in enumerate(rooms):
        if room.has_window:
            # Window on exterior wall
            cx = room.x + room.width / 2
            cy = room.y + room.depth / 2
            
            # Find nearest exterior wall
            for wi, w in enumerate(walls):
                if not w.is_external:
                    continue
                if w.y1 == 0 and w.y2 == 0:  # bottom wall
                    if room.y < 0.5 and w.x1 <= cx <= w.x2:
                        windows.append(Window(wi, (cx - w.x1) / (w.x2 - w.x1), 1.5))
                elif w.y1 == building_depth:  # top wall
                    if room.y + room.depth > building_depth - 0.5 and w.x1 <= cx <= w.x2:
                        windows.append(Window(wi, (cx - w.x1) / (w.x2 - w.x1), 1.5))
                elif w.x1 == 0 and w.x2 == 0:  # left wall
                    if room.x < 0.5 and w.y1 <= cy <= w.y2:
                        pass  # handle differently
                elif w.x1 == building_width:  # right wall
                    if room.x + room.width > building_width - 0.5:
                        pass
    
    # Main entrance door
    doors.append(Door(0, 0.5, 0.9))  # bottom wall, center-ish
    
    return windows, doors

# ==================== SNiP COMPLIANCE ====================

def check_snip(rooms: List[Room], floors: int) -> List[str]:
    """Check SNiP compliance."""
    issues = []
    ok = []
    
    for r in rooms:
        area = r.width * r.depth
        if r.name in ("Спальня", "Детская") and area < 8:
            issues.append(f"⚠️ {r.name}: площадь {area:.1f} м² < 8 м² (СНиП 2.08.01-89)")
        elif r.name == "Гостиная" and area < 16:
            issues.append(f"⚠️ {r.name}: площадь {area:.1f} м² < 16 м²")
        elif r.name == "Кухня" and area < 6:
            issues.append(f"⚠️ {r.name}: площадь {area:.1f} м² < 6 м²")
        elif r.width < r.min_dim:
            issues.append(f"⚠️ {r.name}: ширина {r.width:.1f} м < {r.min_dim} м")
        else:
            ok.append(f"✅ {r.name}: {area:.1f} м² — соответствует СНиП")
    
    return ok + issues

# ==================== SVG GENERATORS ====================

def generate_professional_site_svg(plan: dict) -> str:
    """Generate professional site plan SVG with rose, dimensions, legend."""
    site = plan.get("site", {})
    building = plan.get("building", {})
    rooms = building.get("rooms", [])
    
    site_w = site.get("width", 20)
    site_d = site.get("depth", 30)
    bldg_w = building.get("width", 10)
    bldg_d = building.get("depth", 8)
    bldg_x = site.get("building_x", 5)
    bldg_y = site.get("building_y", 10)
    floors = building.get("floors", 1)
    
    SCALE = 16
    MARGIN = 80
    TITLE_H = 70
    STAMP_H = 50
    LEGEND_W = 160
    
    svg_w = site_w * SCALE + MARGIN * 2 + LEGEND_W
    svg_h = site_d * SCALE + MARGIN * 2 + TITLE_H + STAMP_H
    
    bx = MARGIN + bldg_x * SCALE
    by = MARGIN + TITLE_H + (site_d - bldg_y - bldg_d) * SCALE
    bw = bldg_w * SCALE
    bd = bldg_d * SCALE
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#333"/></marker>
    <pattern id="grass" patternUnits="userSpaceOnUse" width="20" height="20"><rect width="20" height="20" fill="#e8f5e9"/><circle cx="5" cy="5" r="1" fill="#a5d6a7"/><circle cx="15" cy="15" r="1" fill="#a5d6a7"/></pattern>
    <pattern id="parking" patternUnits="userSpaceOnUse" width="30" height="20"><rect width="30" height="20" fill="#e0e0e0"/><line x1="15" y1="0" x2="15" y2="20" stroke="#bbb" stroke-width="1"/></pattern>
  </defs>
  
  <!-- Title block -->
  <text x="{MARGIN}" y="30" font-family="Arial" font-size="18" font-weight="bold" fill="#333">ПЛАН УЧАСТКА</text>
  <text x="{MARGIN}" y="50" font-family="Arial" font-size="12" fill="#666">{plan.get("description", "Архитектурный план")}</text>
  <text x="{MARGIN}" y="64" font-family="Arial" font-size="10" fill="#999">Масштаб 1:{int(site_w*100/svg_w*10)}</text>
  
  <!-- Site boundary -->
  <rect x="{MARGIN}" y="{MARGIN + TITLE_H}" width="{site_w * SCALE}" height="{site_d * SCALE}" fill="url(#grass)" stroke="#2d5016" stroke-width="2" rx="1"/>
  
  <!-- Driveway -->
  <polygon points="{MARGIN + bldg_x * SCALE},{MARGIN + TITLE_H + site_d * SCALE} {MARGIN + (bldg_x + 3) * SCALE},{MARGIN + TITLE_H + site_d * SCALE} {MARGIN + (bldg_x + 3) * SCALE},{by + bd} {MARGIN + bldg_x * SCALE},{by + bd}" fill="#d7ccc8" stroke="#a1887f" stroke-width="1"/>
  
  <!-- Building -->
  <rect x="{bx}" y="{by}" width="{bw}" height="{bd}" fill="#fff" stroke="#333" stroke-width="2.5"/>
  <!-- Building hatch -->
  <rect x="{bx}" y="{by}" width="{bw}" height="{bd}" fill="#f5f5f5"/>'''
    
    # Rooms
    for room in rooms:
        rx = bx + room["x"] * SCALE
        ry = by + (bldg_d - room["y"] - room["depth"]) * SCALE
        rw = room["width"] * SCALE
        rd = room["depth"] * SCALE
        area = room["width"] * room["depth"]
        svg += f'''
  <rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" fill="none" stroke="#666" stroke-width="1.5"/>
  <text x="{rx + rw/2}" y="{ry + rd/2 - 6}" font-family="Arial" font-size="11" font-weight="bold" fill="#333" text-anchor="middle">{room["name"]}</text>
  <text x="{rx + rw/2}" y="{ry + rd/2 + 8}" font-family="Arial" font-size="9" fill="#888" text-anchor="middle">{area:.1f} м²</text>'''
    
    # Entrance
    svg += f'''
  <rect x="{bx + bw/2 - 10}" y="{by + bd - 3}" width="20" height="6" fill="#ff9800" stroke="#e65100" stroke-width="1" rx="1"/>
  <text x="{bx + bw/2}" y="{by + bd + 18}" font-family="Arial" font-size="9" fill="#e65100" text-anchor="middle">ВХОД</text>'''
    
    # Dimensions
    svg += f'''
  <!-- Site dimensions -->
  <line x1="{MARGIN}" y1="{MARGIN + TITLE_H + site_d * SCALE + 15}" x2="{MARGIN + site_w * SCALE}" y2="{MARGIN + TITLE_H + site_d * SCALE + 15}" stroke="#1a73e8" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
  <text x="{MARGIN + site_w * SCALE / 2}" y="{MARGIN + TITLE_H + site_d * SCALE + 30}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle">{site_w} м</text>
  
  <line x1="{MARGIN - 15}" y1="{MARGIN + TITLE_H}" x2="{MARGIN - 15}" y2="{MARGIN + TITLE_H + site_d * SCALE}" stroke="#1a73e8" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
  <text x="{MARGIN - 25}" y="{MARGIN + TITLE_H + site_d * SCALE / 2}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle" transform="rotate(-90,{MARGIN - 25},{MARGIN + TITLE_H + site_d * SCALE / 2})">{site_d} м</text>
  
  <!-- Building dimensions -->
  <line x1="{bx}" y1="{by - 8}" x2="{bx + bw}" y2="{by - 8}" stroke="#e65100" stroke-width="0.8"/>
  <text x="{bx + bw/2}" y="{by - 12}" font-family="Arial" font-size="10" fill="#e65100" text-anchor="middle">{bldg_w} м</text>
  <line x1="{bx + bw + 8}" y1="{by}" x2="{bx + bw + 8}" y2="{by + bd}" stroke="#e65100" stroke-width="0.8"/>
  <text x="{bx + bw + 16}" y="{by + bd/2}" font-family="Arial" font-size="10" fill="#e65100" text-anchor="middle" transform="rotate(90,{bx + bw + 16},{by + bd/2})">{bldg_d} м</text>'''
    
    # North arrow
    na_cx = svg_w - LEGEND_W/2
    na_cy = MARGIN + TITLE_H + 40
    svg += f'''
  <!-- North arrow -->
  <line x1="{na_cx}" y1="{na_cy + 20}" x2="{na_cx}" y2="{na_cy - 20}" stroke="#333" stroke-width="2"/>
  <polygon points="{na_cx},{na_cy - 25} {na_cx - 6},{na_cy - 10} {na_cx + 6},{na_cy - 10}" fill="#e53935"/>
  <text x="{na_cx}" y="{na_cy - 30}" font-family="Arial" font-size="12" font-weight="bold" fill="#e53935" text-anchor="middle">N</text>'''
    
    # Legend
    lx = svg_w - LEGEND_W + 10
    ly = svg_h - STAMP_H - 10
    svg += f'''
  <!-- Legend -->
  <rect x="{lx - 10}" y="{ly - 80}" width="{LEGEND_W - 5}" height="90" fill="#1e293b" rx="5" opacity="0.95"/>
  <text x="{lx}" y="{ly - 60}" font-family="Arial" font-size="11" font-weight="bold" fill="#38bdf8">Условные обозначения</text>
  <rect x="{lx}" y="{ly - 48}" width="20" height="12" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <text x="{lx + 28}" y="{ly - 38}" font-family="Arial" font-size="10" fill="#ccc">Здание ({floors} эт.)</text>
  <rect x="{lx}" y="{ly - 28}" width="20" height="6" fill="#ff9800" stroke="#e65100"/>
  <text x="{lx + 28}" y="{ly - 21}" font-family="Arial" font-size="10" fill="#ccc">Вход</text>
  <polygon points="{lx},{ly - 8} {lx + 10},{ly - 14} {lx + 10},{ly - 2}" fill="#e53935"/>
  <text x="{lx + 28}" y="{ly - 6}" font-family="Arial" font-size="10" fill="#ccc">Север (N)</text>'''
    
    # Stamp (ГОСТ основная надпись)
    stamp_y = svg_h - STAMP_H
    svg += f'''
  <!-- Stamp -->
  <rect x="{MARGIN}" y="{stamp_y}" width="{site_w * SCALE}" height="{STAMP_H}" fill="none" stroke="#333" stroke-width="1.5"/>
  <line x1="{MARGIN}" y1="{stamp_y + 20}" x2="{MARGIN + site_w * SCALE}" y2="{stamp_y + 20}" stroke="#333" stroke-width="1"/>
  <line x1="{MARGIN + site_w * SCALE * 0.6}" y1="{stamp_y}" x2="{MARGIN + site_w * SCALE * 0.6}" y2="{stamp_y + STAMP_H}" stroke="#333" stroke-width="1"/>
  <text x="{MARGIN + 5}" y="{stamp_y + 14}" font-family="Arial" font-size="9" fill="#333">План участка</text>
  <text x="{MARGIN + site_w * SCALE * 0.6 + 5}" y="{stamp_y + 14}" font-family="Arial" font-size="9" fill="#333">Масштаб 1:100</text>
  <text x="{MARGIN + 5}" y="{stamp_y + 38}" font-family="Arial" font-size="8" fill="#666">{datetime.now().strftime("%d.%m.%Y")}</text>
  <text x="{MARGIN + site_w * SCALE * 0.6 + 5}" y="{stamp_y + 38}" font-family="Arial" font-size="8" fill="#666">Archiq AI</text>
</svg>'''
    
    return svg

def generate_professional_floor_svg(plan: dict) -> str:
    """Generate professional floor plan SVG."""
    building = plan.get("building", {})
    rooms = building.get("rooms", [])
    bldg_w = building.get("width", 10)
    bldg_d = building.get("depth", 8)
    floors = building.get("floors", 1)
    
    SCALE = 45
    MARGIN = 80
    TITLE_H = 70
    STAMP_H = 50
    TABLE_Y = 30  # room table height
    
    svg_w = bldg_w * SCALE + MARGIN * 2
    svg_h = bldg_d * SCALE + MARGIN * 2 + TITLE_H + STAMP_H + TABLE_Y
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#333"/></marker>
  </defs>
  
  <!-- Title -->
  <text x="{MARGIN}" y="30" font-family="Arial" font-size="18" font-weight="bold" fill="#333">ПЛАН {floors}-го ЭТАЖА</text>
  <text x="{MARGIN}" y="50" font-family="Arial" font-size="12" fill="#666">{plan.get("description", "Архитектурный план")}</text>
  <text x="{MARGIN}" y="64" font-family="Arial" font-size="10" fill="#999">Масштаб 1:100</text>
  
  <!-- Axes -->
  <line x1="{MARGIN}" y1="{MARGIN + TITLE_H - 15}" x2="{MARGIN}" y2="{MARGIN + TITLE_H + bldg_d * SCALE + 15}" stroke="#e53935" stroke-width="0.8" stroke-dasharray="8,4"/>
  <line x1="{MARGIN + bldg_w * SCALE}" y1="{MARGIN + TITLE_H - 15}" x2="{MARGIN + bldg_w * SCALE}" y2="{MARGIN + TITLE_H + bldg_d * SCALE + 15}" stroke="#e53935" stroke-width="0.8" stroke-dasharray="8,4"/>
  <line x1="{MARGIN - 15}" y1="{MARGIN + TITLE_H}" x2="{MARGIN + bldg_w * SCALE + 15}" y2="{MARGIN + TITLE_H}" stroke="#e53935" stroke-width="0.8" stroke-dasharray="8,4"/>
  <line x1="{MARGIN - 15}" y1="{MARGIN + TITLE_H + bldg_d * SCALE}" x2="{MARGIN + bldg_w * SCALE + 15}" y2="{MARGIN + TITLE_H + bldg_d * SCALE}" stroke="#e53935" stroke-width="0.8" stroke-dasharray="8,4"/>
  
  <!-- Axis labels -->
  <circle cx="{MARGIN}" cy="{MARGIN + TITLE_H - 20}" r="8" fill="none" stroke="#e53935" stroke-width="1"/><text x="{MARGIN}" y="{MARGIN + TITLE_H - 16}" font-family="Arial" font-size="8" fill="#e53935" text-anchor="middle">А</text>
  <circle cx="{MARGIN + bldg_w * SCALE}" cy="{MARGIN + TITLE_H - 20}" r="8" fill="none" stroke="#e53935" stroke-width="1"/><text x="{MARGIN + bldg_w * SCALE}" y="{MARGIN + TITLE_H - 16}" font-family="Arial" font-size="8" fill="#e53935" text-anchor="middle">Б</text>
  <circle cx="{MARGIN - 22}" cy="{MARGIN + TITLE_H}" r="8" fill="none" stroke="#e53935" stroke-width="1"/><text x="{MARGIN - 22}" y="{MARGIN + TITLE_H + 3}" font-family="Arial" font-size="8" fill="#e53935" text-anchor="middle">1</text>
  <circle cx="{MARGIN - 22}" cy="{MARGIN + TITLE_H + bldg_d * SCALE}" r="8" fill="none" stroke="#e53935" stroke-width="1"/><text x="{MARGIN - 22}" y="{MARGIN + TITLE_H + bldg_d * SCALE + 3}" font-family="Arial" font-size="8" fill="#e53935" text-anchor="middle">2</text>'''
    
    # Rooms
    total_area = 0
    for room in rooms:
        rx = MARGIN + room["x"] * SCALE
        ry = MARGIN + TITLE_H + (bldg_d - room["y"] - room["depth"]) * SCALE
        rw = room["width"] * SCALE
        rd = room["depth"] * SCALE
        area = room["width"] * room["depth"]
        total_area += area
        svg += f'''
  <rect x="{rx}" y="{ry}" width="{rw}" height="{rd}" fill="#fafafa" stroke="#333" stroke-width="2"/>
  <text x="{rx + rw/2}" y="{ry + rd/2 - 8}" font-family="Arial" font-size="12" font-weight="bold" fill="#333" text-anchor="middle">{room["name"]}</text>
  <text x="{rx + rw/2}" y="{ry + rd/2 + 8}" font-family="Arial" font-size="10" fill="#888" text-anchor="middle">{area:.1f} м²</text>'''
    
    # Entrance
    svg += f'''
  <rect x="{MARGIN + bldg_w * SCALE / 2 - 12}" y="{MARGIN + TITLE_H + bldg_d * SCALE - 4}" width="24" height="8" fill="#ff9800" stroke="#e65100" stroke-width="1.5" rx="1"/>
  <text x="{MARGIN + bldg_w * SCALE / 2}" y="{MARGIN + TITLE_H + bldg_d * SCALE + 18}" font-family="Arial" font-size="10" font-weight="bold" fill="#e65100" text-anchor="middle">ВХОД</text>'''
    
    # Dimensions
    svg += f'''
  <line x1="{MARGIN}" y1="{MARGIN + TITLE_H + bldg_d * SCALE + 30}" x2="{MARGIN + bldg_w * SCALE}" y2="{MARGIN + TITLE_H + bldg_d * SCALE + 30}" stroke="#1a73e8" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
  <text x="{MARGIN + bldg_w * SCALE / 2}" y="{MARGIN + TITLE_H + bldg_d * SCALE + 44}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle">{bldg_w} м</text>
  <line x1="{MARGIN - 30}" y1="{MARGIN + TITLE_H}" x2="{MARGIN - 30}" y2="{MARGIN + TITLE_H + bldg_d * SCALE}" stroke="#1a73e8" stroke-width="1" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
  <text x="{MARGIN - 40}" y="{MARGIN + TITLE_H + bldg_d * SCALE / 2}" font-family="Arial" font-size="12" fill="#1a73e8" text-anchor="middle" transform="rotate(-90,{MARGIN - 40},{MARGIN + TITLE_H + bldg_d * SCALE / 2})">{bldg_d} м</text>'''
    
    # Room schedule (экспликация)
    table_y = MARGIN + TITLE_H + bldg_d * SCALE + 60
    svg += f'''
  <!-- Экспликация помещений -->
  <text x="{MARGIN}" y="{table_y}" font-family="Arial" font-size="13" font-weight="bold" fill="#333">ЭКСПЛИКАЦИЯ ПОМЕЩЕНИЙ</text>
  <rect x="{MARGIN}" y="{table_y + 5}" width="{bldg_w * SCALE}" height="{25 + len(rooms) * 18}" fill="none" stroke="#333" stroke-width="1"/>
  <rect x="{MARGIN}" y="{table_y + 5}" width="{bldg_w * SCALE}" height="18" fill="#e0e0e0"/>
  <text x="{MARGIN + 10}" y="{table_y + 17}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">№</text>
  <text x="{MARGIN + 30}" y="{table_y + 17}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">Наименование</text>
  <text x="{MARGIN + bldg_w * SCALE - 80}" y="{table_y + 17}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">Площадь, м²</text>'''
    
    for i, room in enumerate(rooms):
        y = table_y + 25 + i * 18
        area = room["width"] * room["depth"]
        bg = "#fff" if i % 2 == 0 else "#f5f5f5"
        svg += f'''
  <rect x="{MARGIN}" y="{y}" width="{bldg_w * SCALE}" height="17" fill="{bg}"/>
  <text x="{MARGIN + 15}" y="{y + 12}" font-family="Arial" font-size="9" fill="#333">{i+1}</text>
  <text x="{MARGIN + 30}" y="{y + 12}" font-family="Arial" font-size="9" fill="#333">{room["name"]}</text>
  <text x="{MARGIN + bldg_w * SCALE - 80}" y="{y + 12}" font-family="Arial" font-size="9" fill="#333">{area:.1f}</text>'''
    
    # Total
    svg += f'''
  <rect x="{MARGIN}" y="{table_y + 25 + len(rooms) * 18}" width="{bldg_w * SCALE}" height="18" fill="#e0e0e0"/>
  <text x="{MARGIN + 30}" y="{table_y + 25 + len(rooms) * 18 + 12}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">ИТОГО</text>
  <text x="{MARGIN + bldg_w * SCALE - 80}" y="{table_y + 25 + len(rooms) * 18 + 12}" font-family="Arial" font-size="9" font-weight="bold" fill="#333">{total_area:.1f}</text>'''
    
    # Stamp
    stamp_y = svg_h - STAMP_H
    svg += f'''
  <!-- Stamp -->
  <rect x="{MARGIN}" y="{stamp_y}" width="{bldg_w * SCALE}" height="{STAMP_H}" fill="none" stroke="#333" stroke-width="1.5"/>
  <line x1="{MARGIN}" y1="{stamp_y + 20}" x2="{MARGIN + bldg_w * SCALE}" y2="{stamp_y + 20}" stroke="#333" stroke-width="1"/>
  <line x1="{MARGIN + bldg_w * SCALE * 0.6}" y1="{stamp_y}" x2="{MARGIN + bldg_w * SCALE * 0.6}" y2="{stamp_y + STAMP_H}" stroke="#333" stroke-width="1"/>
  <text x="{MARGIN + 5}" y="{stamp_y + 14}" font-family="Arial" font-size="9" fill="#333">План {floors}-го этажа</text>
  <text x="{MARGIN + bldg_w * SCALE * 0.6 + 5}" y="{stamp_y + 14}" font-family="Arial" font-size="9" fill="#333">Масштаб 1:100</text>
  <text x="{MARGIN + 5}" y="{stamp_y + 38}" font-family="Arial" font-size="8" fill="#666">{datetime.now().strftime("%d.%m.%Y")}</text>
  <text x="{MARGIN + bldg_w * SCALE * 0.6 + 5}" y="{stamp_y + 38}" font-family="Arial" font-size="8" fill="#666">Archiq AI</text>
</svg>'''
    
    return svg

# ==================== SPECIFICATIONS ====================

def generate_specs(plan: dict) -> dict:
    building = plan.get("building", {})
    site = plan.get("site", {})
    rooms = building.get("rooms", [])
    
    total_area = sum(r.get("width", 0) * r.get("depth", 0) for r in rooms)
    living_area = sum(r.get("width", 0) * r.get("depth", 0) for r in rooms if r.get("name") in ("Гостиная", "Спальня", "Детская", "Кабинет"))
    wet_area = sum(r.get("width", 0) * r.get("depth", 0) for r in rooms if r.get("is_wet", False))
    site_area = site.get("width", 1) * site.get("depth", 1)
    
    return {
        "total_area": round(total_area, 1),
        "living_area": round(living_area, 1),
        "wet_area": round(wet_area, 1),
        "floors": building.get("floors", 1),
        "rooms_count": len(rooms),
        "building_dimensions": f"{building.get('width', 0)} x {building.get('depth', 0)} м",
        "site_dimensions": f"{site.get('width', 0)} x {site.get('depth', 0)} м",
        "building_footprint": round(total_area, 1),
        "site_area": round(site_area, 1),
        "site_coverage": round(total_area / site_area * 100, 1),
        "perimeter": round(2 * (building.get('width', 0) + building.get('depth', 0)), 1),
        "compliance": check_snip_rooms(rooms),
        "description": plan.get("description", "")
    }

def check_snip_rooms(rooms: List[Dict]) -> List[str]:
    results = []
    for r in rooms:
        area = r.get("width", 0) * r.get("depth", 0)
        name = r.get("name", "")
        if "Спальня" in name or "Детская" in name:
            if area >= 8:
                results.append(f"✅ {name}: {area:.1f} м² ≥ 8 м² (СНиП 2.08.01-89)")
            else:
                results.append(f"⚠️ {name}: {area:.1f} м² < 8 м²")
        elif name == "Гостиная":
            if area >= 16:
                results.append(f"✅ {name}: {area:.1f} м² ≥ 16 м²")
            else:
                results.append(f"⚠️ {name}: {area:.1f} м² < 16 м²")
        elif name == "Кухня":
            if area >= 6:
                results.append(f"✅ {name}: {area:.1f} м² ≥ 6 м²")
            else:
                results.append(f"⚠️ {name}: {area:.1f} м² < 6 м²")
        else:
            results.append(f"✅ {name}: {area:.1f} м²")
    return results

# ==================== AI GENERATION ====================

def generate_plan_ai(project: dict) -> dict:
    """Generate plan using Gemini AI if available, else fallback."""
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""Ты — профессиональный архитектор. Сгенерируй план здания.

Параметры: тип={project.get('building_type','дом')}, площадь={project.get('area',100)}м², этажей={project.get('floors',1)}, комнат={project.get('rooms',3)}, участок={project.get('site_width',20)}x{project.get('site_depth',30)}м, требования: {project.get('requirements','стандарт')}

Ответь ТОЛЬКО JSON:
{{"building":{{"width":X,"depth":Y,"floors":Z,"rooms":[{{"name":"Имя","width":W,"depth":D,"x":X,"y":Y,"is_wet":true/false}}],"entrance":{{"x":X,"y":Y}}}},"site":{{"width":W,"depth":D,"building_x":X,"building_y":Y,"parking":true,"garden":true}},"description":"описание"}}"""
            response = model.generate_content(prompt)
            text = response.text
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                plan = json.loads(json_match.group())
                # Ensure required fields
                for room in plan.get("building", {}).get("rooms", []):
                    room.setdefault("is_wet", False)
                return plan
        except Exception as e:
            print(f"AI error: {e}")
    
    return generate_plan_fallback(project)

def generate_plan_fallback(project: dict) -> dict:
    """Generate plan algorithmically."""
    area = project.get('area', 100)
    floors = project.get('floors', 1)
    rooms_count = project.get('rooms', 3)
    site_w = project.get('site_width', 20)
    site_d = project.get('site_depth', 30)
    bldg_type = project.get('building_type', 'жилой дом')
    
    floor_area = area / floors
    bldg_w = math.sqrt(floor_area) * 1.3
    bldg_d = floor_area / bldg_w
    bldg_w = round(bldg_w, 1)
    bldg_d = round(bldg_d, 1)
    
    rooms = determine_rooms(bldg_type, floor_area, rooms_count)
    placed = pack_rooms(rooms, bldg_w, bldg_d)
    
    # Convert to dicts
    rooms_dict = []
    for r in placed:
        rooms_dict.append({
            "name": r.name,
            "width": r.width,
            "depth": r.depth,
            "x": round(r.x, 1),
            "y": round(r.y, 1),
            "is_wet": r.is_wet,
            "has_window": r.has_window
        })
    
    return {
        "building": {
            "width": bldg_w,
            "depth": bldg_d,
            "floors": floors,
            "rooms": rooms_dict,
            "entrance": {"x": round(bldg_w / 2, 1), "y": 0}
        },
        "site": {
            "width": site_w,
            "depth": site_d,
            "building_x": round((site_w - bldg_w) / 3, 1),
            "building_y": round((site_d - bldg_d) / 3, 1),
            "parking": True,
            "garden": True,
            "driveway": True
        },
        "description": f"{bldg_type}, {area} м², {floors} эт., {len(placed)} пом."
    }

# ==================== APP ====================

app = FastAPI(title="Archiq AI", description="Генератор архитектурных планов")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LANDING = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Archiq AI — Генератор архитектурных планов</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem}
.container{max-width:600px;text-align:center}
.logo{font-size:3rem;font-weight:800;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}
.sub{color:#94a3b8;margin-bottom:2rem}
.status{display:inline-flex;align-items:center;gap:.5rem;background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);padding:.5rem 1rem;border-radius:999px;margin-bottom:2rem}
.dot{width:8px;height:8px;border-radius:50%;background:#22c55e;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.endpoints{text-align:left;background:rgba(30,41,59,.8);border:1px solid #334155;border-radius:12px;padding:1.5rem}
.endpoints h2{font-size:1rem;color:#94a3b8;margin-bottom:1rem;text-transform:uppercase;letter-spacing:.05em}
.ep{display:flex;gap:1rem;padding:.6rem 0;border-bottom:1px solid #1e293b}
.ep:last-child{border-bottom:none}
.method{font-family:monospace;font-size:.8rem;font-weight:700;min-width:55px;padding:.15rem .4rem;border-radius:4px;text-align:center}
.get{background:rgba(56,189,248,.15);color:#38bdf8}
.post{background:rgba(168,85,247,.15);color:#a855f7}
.path{font-family:monospace;font-size:.85rem;color:#e2e8f0}
.desc{color:#64748b;font-size:.8rem;margin-left:auto}
.ft{margin-top:2rem;color:#475569;font-size:.8rem}
.ft a{color:#818cf8;text-decoration:none}
</style></head>
<body><div class="container">
<div class="logo">🏗️ Archiq AI</div>
<div class="sub">Генератор профессиональных архитектурных планов</div>
<div class="status"><span class="dot"></span> Сервис работает</div>
<div class="endpoints"><h2>API</h2>
<div class="ep"><span class="method get">GET</span><span class="path">/health</span><span class="desc">Статус</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/generate-plan</span><span class="desc">Генерация плана</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/analyze-site-plan</span><span class="desc">Анализ плана участка</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/site-plan-svg?id=...</span><span class="desc">SVG план участка</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/floor-plan-svg?id=...</span><span class="desc">SVG поэтажный план</span></div>
<div class="ep"><span class="method get">GET</span><span class="path">/specs?id=...</span><span class="desc">Спецификация</span></div>
</div>
<div class="ft"><p>GitHub: <a href="https://github.com/K09-0/ARCHIQ-AI" target="_blank">K09-0/ARCHIQ-AI</a></p></div>
</div></body></html>"""

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY, name TEXT, building_type TEXT, area REAL,
        floors INTEGER, rooms INTEGER, site_width REAL, site_depth REAL,
        requirements TEXT, plan_json TEXT, created_at TEXT)""")
    conn.commit()
    conn.close()

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def root():
    return LANDING

@app.get("/health")
def health():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY), "hf_configured": bool(HF_API_KEY), "service": "Archiq AI — Генератор архитектурных планов"}

@app.post("/generate-plan")
def generate(name: str = Form("Новый проект"), building_type: str = Form("жилой дом"),
             area: float = Form(100), floors: int = Form(1), rooms: int = Form(3),
             site_width: float = Form(20), site_depth: float = Form(30),
             requirements: str = Form("")):
    project = {"name": name, "building_type": building_type, "area": area, "floors": floors,
               "rooms": rooms, "site_width": site_width, "site_depth": site_depth, "requirements": requirements}
    
    plan = generate_plan_ai(project)
    specs = generate_specs(plan)
    project_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    (OUTPUT_DIR / f"{project_id}_site.svg").write_text(generate_professional_site_svg(plan))
    (OUTPUT_DIR / f"{project_id}_floor.svg").write_text(generate_professional_floor_svg(plan))
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (project_id, name, building_type, area, floors, rooms, site_width, site_depth,
         requirements, json.dumps(plan), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return {"project_id": project_id, "plan": plan, "specs": specs}

@app.get("/site-plan-svg")
def get_site(project_id: str):
    path = OUTPUT_DIR / f"{project_id}_site.svg"
    if path.exists():
        return HTMLResponse(content=path.read_text())
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/floor-plan-svg")
def get_floor(project_id: str):
    path = OUTPUT_DIR / f"{project_id}_floor.svg"
    if path.exists():
        return HTMLResponse(content=path.read_text())
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/specs")
def get_specs(project_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT plan_json FROM projects WHERE id=?", (project_id,)).fetchone()
    conn.close()
    if row:
        return generate_specs(json.loads(row[0]))
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/projects")
def list_projects():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id,name,building_type,area,floors,rooms,created_at FROM projects ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/analyze-site-plan")
async def analyze_site(file: UploadFile = File(...)):
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return {"error": "Gemini not configured"}
    try:
        content = await file.read()
        img_b64 = base64.b64encode(content).decode()
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content([
            "Архитектор: проанализируй план участка. Ответь JSON: {\"site_width\":X,\"site_depth\":Y,\"features\":[]}",
            {"mime_type": "image/jpeg", "data": img_b64}
        ])
        m = re.search(r'\{[\s\S]*\}', resp.text)
        return json.loads(m.group()) if m else {"analysis": resp.text}
    except Exception as e:
        return {"error": str(e)}
