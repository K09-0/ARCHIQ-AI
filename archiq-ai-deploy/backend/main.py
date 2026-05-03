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

#!/usr/bin/env python3
"""Этап 3: AI-анализ плана участка через Gemini Vision.

Анализирует загруженное изображение плана участка и извлекает:
- Размеры участка (ширина, глубина)
- Границы участка
- Существующие постройки
- Красные линии
- Авто-калибровка масштаба
- Северная ориентация
- Подъездные пути, озеленение
"""

import base64
import re
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Объект, обнаруженный на плане участка."""
    type: str  # building, tree, road, parking, fence, water, etc.
    label: Optional[str] = None
    x: float = 0.0  # normalized 0-1
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    confidence: float = 0.0
    area_sqm: Optional[float] = None
    description: str = ""


@dataclass
class SiteAnalysisResult:
    """Полный результат анализа плана участка."""
    # Размеры участка
    site_width: Optional[float] = None
    site_depth: Optional[float] = None
    site_area: Optional[float] = None
    
    # Масштаб
    scale_detected: Optional[str] = None  # e.g. "1:500"
    pixels_per_meter: Optional[float] = None
    
    # Границы
    boundary_type: str = "unknown"  # rectangular, irregular
    boundary_points: List[Dict[str, float]] = field(default_factory=list)
    
    # Существующие объекты
    existing_buildings: List[DetectedObject] = field(default_factory=list)
    trees: List[DetectedObject] = field(default_factory=list)
    roads: List[DetectedObject] = field(default_factory=list)
    fences: List[DetectedObject] = field(default_factory=list)
    water_features: List[DetectedObject] = field(default_factory=list)
    other_objects: List[DetectedObject] = field(default_factory=list)
    
    # Красные линии
    red_lines_detected: bool = False
    red_lines_description: str = ""
    
    # Ориентация
    north_detected: bool = False
    north_direction: Optional[str] = None  # "top", "right", "bottom", "left"
    
    # Подъезд
    driveway_detected: bool = False
    driveway_description: str = ""
    
    # OCR текст
    ocr_text: str = ""
    
    # Gemini сырой ответ
    raw_description: str = ""
    
    # Статус
    success: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {}
        for k, v in asdict(self).items():
            if isinstance(v, DetectedObject):
                result[k] = asdict(v)
            elif isinstance(v, list) and v and isinstance(v[0], DetectedObject):
                result[k] = [asdict(obj) for obj in v]
            else:
                result[k] = v
        return result


def analyze_site_plan(
    image_bytes: bytes,
    mime_type: str,
    gemini_api_key: str,
    model_name: str = "gemini-2.0-flash",
) -> SiteAnalysisResult:
    """Анализ плана участка через Gemini Vision.
    
    Args:
        image_bytes: Raw image bytes
        mime_type: MIME type (image/jpeg, image/png, etc.)
        gemini_api_key: Gemini API key
        model_name: Gemini model to use
    
    Returns:
        SiteAnalysisResult with extracted data
    """
    result = SiteAnalysisResult()
    
    if not gemini_api_key:
        result.error = "GEMINI_API_KEY not configured"
        return result
    
    try:
        import google.generativeai as genai
    except ImportError:
        result.error = "google-generativeai not installed"
        return result
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)
        
        b64_image = base64.b64encode(image_bytes).decode()
        
        prompt = _build_analysis_prompt()
        
        response = model.generate_content([
            prompt,
            {"mime_type": mime_type, "data": b64_image}
        ])
        
        text = response.text
        result.raw_description = text
        
        # Parse JSON from response
        parsed = _parse_gemini_response(text)
        if parsed:
            result = _populate_result(result, parsed)
            result.success = True
        else:
            # Fallback: still return text
            result.success = True
            result.ocr_text = text[:2000]
    
    except Exception as e:
        logger.error(f"Site analysis error: {e}")
        result.error = str(e)
    
    return result


def _build_analysis_prompt() -> str:
    """Build the Gemini Vision prompt for site plan analysis."""
    return """Ты — профессиональный архитектор-аналитик. Проанализируй изображение плана участка.

Ответь ТОЛЬКО валидным JSON (без markdown-обёрток, без пояснений до или после JSON).

Формат ответа:
{
  "site": {
    "width_meters": число или null,
    "depth_meters": число или null,
    "area_sqm": число или null,
    "boundary_type": "rectangular" | "irregular" | "unknown",
    "scale": "1:500" или null,
    "pixels_per_meter": число или null
  },
  "north": {
    "detected": true/false,
    "direction": "top" | "right" | "bottom" | "left" | null
  },
  "existing_buildings": [
    {"label": "название или null", "x": 0.0-1.0, "y": 0.0-1.0, "width": 0.0-1.0, "height": 0.0-1.0, "area_sqm": число или null, "description": "описание"}
  ],
  "trees": [
    {"label": null, "x": 0.0-1.0, "y": 0.0-1.0, "width": 0.0-1.0, "height": 0.0-1.0, "count": число, "description": "описание"}
  ],
  "roads": [
    {"label": "название улицы или null", "x": 0.0-1.0, "y": 0.0-1.0, "width": 0.0-1.0, "height": 0.0-1.0, "description": "описание"}
  ],
  "red_lines": {
    "detected": true/false,
    "description": "описание красных линий"
  },
  "driveway": {
    "detected": true/false,
    "description": "описание подъезда"
  },
  "water_features": [
    {"label": null, "x": 0.0-1.0, "y": 0.0-1.0, "width": 0.0-1.0, "height": 0.0-1.0, "description": "описание"}
  ],
  "other_objects": [
    {"type": "тип", "label": "название", "x": 0.0-1.0, "y": 0.0-1.0, "width": 0.0-1.0, "height": 0.0-1.0, "description": "описание"}
  ],
  "ocr_text": "весь распознанный текст с плана",
  "description": "общее описание плана участка на русском"
}

ПРАВИЛА:
- Координаты x, y — это позиция центра объекта (0.0 = левый/верхний край, 1.0 = правый/нижний)
- width, height — размеры объекта в долях от изображения (0.0-1.0)
- Если не можешь определить числовое значение — используй null
- Если объект не обнаружен — используй пустой массив []
- Описывай всё на русском языке
- Постарайся определить масштаб плана (обычно указан как 1:N)
- Ищи красные линии (границы зон регулирования застройки)
- Определи где север (обычно стрелка с буквой N)
- Если размеры не указаны явно, оцени визуально пропорции"""


def _parse_gemini_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from Gemini response text."""
    # Try direct JSON parse
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code blocks
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
    
    # Try to find JSON object
    j = re.search(r'\{[\s\S]*\}', text)
    if j:
        try:
            return json.loads(j.group())
        except json.JSONDecodeError:
            # Try to fix common issues
            try:
                fixed = _fix_json(j.group())
                return json.loads(fixed)
            except Exception:
                pass
    return None


def _fix_json(text: str) -> str:
    """Attempt to fix common JSON issues in LLM output."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Remove comments
    text = re.sub(r'//[^\n]*\n', '\n', text)
    return text


def _populate_result(result: SiteAnalysisResult, data: Dict[str, Any]) -> SiteAnalysisResult:
    """Populate SiteAnalysisResult from parsed JSON data."""
    site = data.get("site", {})
    result.site_width = site.get("width_meters")
    result.site_depth = site.get("depth_meters")
    result.site_area = site.get("area_sqm")
    result.scale_detected = site.get("scale")
    result.pixels_per_meter = site.get("pixels_per_meter")
    
    boundary = data.get("boundary_type", "unknown")
    result.boundary_type = boundary if boundary in ("rectangular", "irregular", "unknown") else "unknown"
    
    north = data.get("north", {})
    result.north_detected = north.get("detected", False)
    result.north_direction = north.get("direction")
    
    red_lines = data.get("red_lines", {})
    result.red_lines_detected = red_lines.get("detected", False)
    result.red_lines_description = red_lines.get("description", "")
    
    driveway = data.get("driveway", {})
    result.driveway_detected = driveway.get("detected", False)
    result.driveway_description = driveway.get("description", "")
    
    result.ocr_text = data.get("ocr_text", "")
    result.raw_description = data.get("description", "")
    
    # Parse objects
    for building in data.get("existing_buildings", []):
        result.existing_buildings.append(DetectedObject(
            type="building", **_clean_obj(building)
        ))
    
    for tree in data.get("trees", []):
        result.trees.append(DetectedObject(
            type="tree", **_clean_obj(tree)
        ))
    
    for road in data.get("roads", []):
        result.roads.append(DetectedObject(
            type="road", **_clean_obj(road)
        ))
    
    for fence in data.get("fences", []):
        result.fences.append(DetectedObject(
            type="fence", **_clean_obj(fence)
        ))
    
    for water in data.get("water_features", []):
        result.water_features.append(DetectedObject(
            type="water", **_clean_obj(water)
        ))
    
    for other in data.get("other_objects", []):
        result.other_objects.append(DetectedObject(
            type=other.get("type", "unknown"), **_clean_obj(other)
        ))
    
    return result


def _clean_obj(data: Dict) -> Dict:
    """Clean object data, removing extra keys and setting defaults."""
    allowed = {"label", "x", "y", "width", "height", "confidence", "area_sqm", "description", "count", "type"}
    return {k: v for k, v in data.items() if k in allowed and v is not None}


def suggest_site_dimensions(analysis: SiteAnalysisResult) -> Dict[str, Any]:
    """Generate suggested site dimensions for plan generation based on analysis.
    
    If Gemini couldn't determine exact dimensions, provides reasonable defaults
    based on detected features.
    """
    width = analysis.site_width
    depth = analysis.site_depth
    area = analysis.site_area
    
    # If we got dimensions from analysis, use them
    if width and depth:
        return {"width": width, "depth": depth, "source": "detected"}
    
    # If we got area, estimate dimensions (assume ~1.5 ratio)
    if area:
        w = round((area * 1.5) ** 0.5, 1)
        d = round(area / w, 1)
        return {"width": w, "depth": d, "source": "estimated_from_area"}
    
    # If buildings detected, estimate from their relative sizes
    if analysis.existing_buildings:
        # Typical small residential plot
        return {"width": 20, "depth": 30, "source": "default_residential"}
    
    # Fallback
    return {"width": 15, "depth": 25, "source": "default_small"}


def analysis_to_plan_params(analysis: SiteAnalysisResult) -> Dict[str, Any]:
    """Convert analysis result into parameters suitable for gen_plan()."""
    dims = suggest_site_dimensions(analysis)
    
    params = {
        "site_width": dims["width"],
        "site_depth": dims["depth"],
        "site_analysis": {
            "source": dims["source"],
            "scale": analysis.scale_detected,
            "north": analysis.north_direction,
            "existing_buildings_count": len(analysis.existing_buildings),
            "trees_detected": len(analysis.trees),
            "red_lines": analysis.red_lines_detected,
            "description": analysis.raw_description[:500] if analysis.raw_description else "",
        }
    }
    
    return params


#!/usr/bin/env python3
"""Этап 4: AI-архитектор — Gemini как архитектор.

- Генерация оптимальной планировки с учётом СНиП
- Автоматическая проверка СНиП
- Текстовое описание проекта
- Учёт участка, ориентации, соседних построек
"""

import re
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ==================== СНиП RULES ====================

SNIP_RULES = {
    "СНиП 2.08.01-89": {
        "Спальня": {"min_area": 8, "min_width": 2.5, "note": "спальня ≥ 8 м²"},
        "Детская": {"min_area": 8, "min_width": 2.5, "note": "детская ≥ 8 м²"},
        "Спальня 2": {"min_area": 8, "min_width": 2.5, "note": "спальня ≥ 8 м²"},
        "Спальня 3": {"min_area": 8, "min_width": 2.5, "note": "спальня ≥ 8 м²"},
        "Гостиная": {"min_area": 16, "min_width": 3.0, "note": "гостиная ≥ 16 м²"},
        "Кухня": {"min_area": 8, "min_width": 2.0, "note": "кухня ≥ 8 м²"},
        "Ванная": {"min_area": 3.5, "min_width": 1.5, "note": "ванная ≥ 3.5 м²"},
        "Туалет": {"min_area": 1.2, "min_width": 0.9, "note": "туалет ≥ 1.2 м²"},
        "Прихожая": {"min_area": 5, "min_width": 1.4, "note": "прихожая ≥ 5 м²"},
        "Коридор": {"min_area": 4, "min_width": 1.2, "note": "коридор ≥ 1.2 м"},
        "Кабинет": {"min_area": 8, "min_width": 2.5, "note": "кабинет ≥ 8 м²"},
    },
    "СП 55.13330.2016": {
        "Парная": {"min_area": 5, "min_width": 2.0, "note": "парная ≥ 5 м²"},
        "Моечная": {"min_area": 5, "min_width": 2.0, "note": "моечная ≥ 5 м²"},
        "Предбанник": {"min_area": 7, "min_width": 2.0, "note": "предбанник ≥ 7 м²"},
        "Комната отдыха": {"min_area": 12, "min_width": 3.0, "note": "комната отдыха ≥ 12 м²"},
    },
    "СП 113.13330.2012": {
        "Гараж": {"min_area": 18, "min_width": 3.5, "note": "гараж ≥ 18 м²"},
        "Котельная": {"min_area": 6, "min_width": 2.0, "note": "котельная ≥ 6 м², высота ≥ 2.5 м"},
    },
}

# Общие правила
GENERAL_SNIP_RULES = [
    {"rule": "Высота жилых помещений ≥ 2.5 м", "type": "height"},
    {"rule": "Высота кухни ≥ 2.5 м (1-этаж), ≥ 2.3 м (мансарда)", "type": "height"},
    {"rule": "Ширина коридора ≥ 1.2 м", "type": "corridor"},
    {"rule": "Ширина марша лестницы ≥ 0.9 м", "type": "stairs"},
    {"rule": "Уклон лестницы ≤ 1:1.75 (≈ 30°)", "type": "stairs"},
    {"rule": "Инсоляция жилых комнат ≥ 2.5 ч/день", "type": "insolation"},
    {"rule": "Влажные зоны НЕ над жилыми (многоэтаж)", "type": "wet_above"},
]


@dataclass
class SnipViolation:
    """Нарушение СНиП."""
    rule: str
    room: str
    expected: str
    actual: str
    severity: str  # "critical", "warning"
    snip_ref: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SnipCheckResult:
    """Результат проверки СНиП."""
    passed: bool
    total_checks: int = 0
    violations: List[SnipViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "total_checks": self.total_checks,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "summary": self.summary,
        }


@dataclass
class ArchitectResponse:
    """Ответ AI-архитектора."""
    building: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    snip_check: SnipCheckResult = field(default_factory=SnipCheckResult)
    reasoning: str = ""
    site_analysis: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None
    model_used: str = ""

    def to_dict(self) -> Dict:
        return {
            "building": self.building,
            "description": self.description,
            "snip_check": self.snip_check.to_dict(),
            "reasoning": self.reasoning,
            "site_analysis": self.site_analysis,
            "success": self.success,
            "error": self.error,
            "model_used": self.model_used,
        }


def check_snip(building: Dict[str, Any], rooms: List[Dict]) -> SnipCheckResult:
    """Проверка планировки на соответствие СНиП.
    
    Args:
        building: building dict с width, depth, floors
        rooms: list of room dicts с name, w, d, x, y
    
    Returns:
        SnipCheckResult
    """
    result = SnipCheckResult(passed=True)
    checks_done = 0
    
    for rm in rooms:
        name = rm.get("name", "")
        area = rm.get("w", 0) * rm.get("d", 0)
        w = rm.get("w", 0)
        d = rm.get("d", 0)
        
        # Determine room category
        category = _get_room_category(name)
        
        # Find SNiP rule for this room type
        snip_rule = _find_snip_rule(category)
        
        if snip_rule:
            checks_done += 1
            
            # Check minimum area
            min_area = snip_rule.get("min_area", 0)
            if area < min_area:
                result.violations.append(SnipViolation(
                    rule=snip_rule["note"],
                    room=name,
                    expected=f"≥ {min_area} м²",
                    actual=f"{area:.1f} м²",
                    severity="critical",
                    snip_ref=_find_snip_ref(category),
                ))
                result.passed = False
            
            # Check minimum width
            min_w = snip_rule.get("min_width", 0)
            if w < min_w:
                result.violations.append(SnipViolation(
                    rule=f"Мин. ширина {category} ≥ {min_w} м",
                    room=name,
                    expected=f"≥ {min_w} м",
                    actual=f"{w:.1f} м",
                    severity="critical" if w < min_w * 0.8 else "warning",
                    snip_ref=_find_snip_ref(category),
                ))
                if w < min_w * 0.8:
                    result.passed = False
        
        # Check room is within building bounds
        bw = building.get("width", 999)
        bd = building.get("depth", 999)
        x = rm.get("x", 0)
        y = rm.get("y", 0)
        
        checks_done += 1
        if x < -0.1 or y < -0.1 or x + w > bw + 0.1 or y + d > bd + 0.1:
            result.violations.append(SnipViolation(
                rule="Комната выходит за пределы здания",
                room=name,
                expected=f"0 ≤ x+w ≤ {bw}, 0 ≤ y+d ≤ {bd}",
                actual=f"x={x}, y={y}, w={w}, d={d}",
                severity="critical",
                snip_ref="геометрия",
            ))
            result.passed = False
        
        # Check for overlaps
        checks_done += 1
        for other in rooms:
            if other is rm:
                continue
            if _overlaps(rm, other):
                result.violations.append(SnipViolation(
                    rule="Комнаты перекрываются",
                    room=f"{name} ↔ {other['name']}",
                    expected="без перекрытий",
                    actual=f"перекрытие {name}/{other['name']}",
                    severity="critical",
                    snip_ref="геометрия",
                ))
                result.passed = False
                break  # one violation per room pair
        
        # Wet room check (bathroom/toilet shouldn't be far from plumbing)
        if rm.get("is_wet") or "Ванная" in name or "Туалет" in name:
            checks_done += 1
            # Check if adjacent to corridor or hallway
            if not _is_adjacent_to(rm, rooms, ["Коридор", "Прихожая", "Холл"]):
                result.warnings.append(
                    f"⚠️ {name} не примыкает к коридору — рекомендуется для мокрой зоны"
                )
    
    # Building proportion check
    bw = building.get("width", 0)
    bd = building.get("depth", 0)
    if bw > 0 and bd > 0:
        ratio = max(bw, bd) / min(bw, bd)
        if ratio > 3:
            result.warnings.append(
                f"⚠️ Соотношение сторон здания {ratio:.1f}:1 — рекомендуется ≤ 2.5:1"
            )
    
    result.total_checks = checks_done
    if not result.violations and not result.warnings:
        result.summary = "✅ Все проверки СНиП пройдены"
    elif not result.violations:
        result.summary = f"⚠️ {len(result.warnings)} предупреждений"
    else:
        crit = sum(1 for v in result.violations if v.severity == "critical")
        result.summary = f"❌ {crit} критических нарушений, {len(result.warnings)} предупреждений"
    
    return result


def ai_architect_v2(
    building_type: str = "жилой дом",
    area: float = 100,
    floors: int = 1,
    rooms_count: int = 3,
    site_width: float = 20,
    site_depth: float = 30,
    requirements: str = "",
    site_analysis: Optional[Dict] = None,
    gemini_api_key: str = "",
    model_name: str = "gemini-2.0-flash",
) -> ArchitectResponse:
    """AI-архитектор: Gemini генерирует оптимальную планировку.
    
    Args:
        building_type: тип здания
        area: общая площадь
        floors: этажность
        rooms_count: количество комнат
        site_width: ширина участка
        site_depth: глубина участка
        requirements: дополнительные требования
        site_analysis: результат анализа участка (из Этапа 3)
        gemini_api_key: API ключ
        model_name: модель Gemini
    
    Returns:
        ArchitectResponse с планировкой, описанием и проверкой СНиП
    """
    resp = ArchitectResponse(model_used=model_name)
    
    if not gemini_api_key:
        resp.error = "GEMINI_API_KEY not configured"
        return resp
    
    try:
        import google.generativeai as genai
    except ImportError:
        resp.error = "google-generativeai not installed"
        return resp
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = _build_architect_prompt(
            building_type, area, floors, rooms_count,
            site_width, site_depth, requirements, site_analysis
        )
        
        response = model.generate_content(prompt)
        text = response.text
        
        # Parse JSON
        parsed = _parse_json_response(text)
        
        if not parsed:
            resp.error = f"Gemini вернул невалидный JSON: {text[:300]}"
            return resp
        
        # Extract building data
        building = parsed.get("building", {})
        rooms = building.get("rooms", [])
        description = parsed.get("description", "")
        reasoning = parsed.get("reasoning", "")
        
        if not rooms or not building.get("width") or not building.get("depth"):
            resp.error = "Gemini вернул пустую планировку"
            return resp
        
        # Normalize rooms
        rooms = _normalize_rooms(rooms)
        building["rooms"] = rooms
        
        # SNiP check
        snip_result = check_snip(building, rooms)
        
        # If SNiP violations, try to auto-fix and re-check
        if not snip_result.passed:
            rooms = _auto_fix_snip(rooms, building)
            snip_result = check_snip(building, rooms)
            building["rooms"] = rooms
        
        resp.building = building
        resp.description = description
        resp.reasoning = reasoning
        resp.snip_check = snip_result
        resp.site_analysis = site_analysis or {}
        resp.success = True
        
        return resp
    
    except Exception as e:
        logger.error(f"AI architect error: {e}")
        resp.error = str(e)
        return resp


def generate_project_description(building: Dict, site: Dict, snip: SnipCheckResult) -> str:
    """Генерация текстового описания проекта."""
    rooms = building.get("rooms", [])
    bw = building.get("width", 0)
    bd = building.get("depth", 0)
    fl = building.get("floors", 1)
    floors_word = {1: "одноэтажный", 2: "двухэтажный", 3: "трёхэтажный"}.get(fl, f"{fl}-этажный")
    
    total_living = sum(r["w"] * r["d"] for r in rooms if _is_living_room(r.get("name", "")))
    total_area = sum(r["w"] * r["d"] for r in rooms)
    
    # Room list
    room_list = []
    for rm in rooms:
        a = rm["w"] * rm["d"]
        room_list.append(f"{rm['name']} — {a:.1f} м² ({rm['w']:.1f}×{rm['d']:.1f} м)")
    
    site_area = site.get("width", 0) * site.get("depth", 0)
    coverage = (total_area / site_area * 100) if site_area > 0 else 0
    
    desc = []
    desc.append(f"**{floors_word.capitalize()} жилой дом**, общая площадь {total_area:.0f} м², жилая {total_living:.0f} м².")
    desc.append(f"Размеры здания: {bw:.1f} × {bd:.1f} м.")
    desc.append("")
    desc.append(f"**Участок:** {site.get('width', 0):.0f} × {site.get('depth', 0):.0f} м ({site_area:.0f} м²)")
    desc.append(f"Коэффициент застройки: {coverage:.1f}%")
    desc.append("")
    desc.append("**Помещения:**")
    for rl in room_list:
        desc.append(f"  • {rl}")
    desc.append("")
    
    if snip.passed:
        desc.append("✅ **Проверка СНиП:** все нормы соблюдены")
    else:
        desc.append(f"⚠️ **Проверка СНиП:** {snip.summary}")
        for v in snip.violations[:3]:
            desc.append(f"  • {v.snip_ref}: {v.rule} — {v.room}: {v.actual} (ожидалось {v.expected})")
    
    return "\n".join(desc)


# ==================== INTERNAL FUNCTIONS ====================

def _get_room_category(name: str) -> str:
    """Map room name to SNiP category."""
    if name in ("Спальня", "Детская", "Спальня 2", "Спальня 3"):
        return name
    if "Спальня" in name:
        return "Спальня"
    return name


def _find_snip_rule(category: str) -> Optional[Dict]:
    """Find SNiP rule for a room category."""
    for ruleset in SNIP_RULES.values():
        if category in ruleset:
            return ruleset[category]
    return None


def _find_snip_ref(category: str) -> str:
    """Find SNiP reference for a room category."""
    for ref, ruleset in SNIP_RULES.items():
        if category in ruleset:
            return ref
    return "общие нормы"


def _overlaps(r1: Dict, r2: Dict) -> bool:
    """Check if two rooms overlap."""
    return not (
        r1.get("x", 0) + r1.get("w", 0) <= r2.get("x", 0) + 0.1 or
        r2.get("x", 0) + r2.get("w", 0) <= r1.get("x", 0) + 0.1 or
        r1.get("y", 0) + r1.get("d", 0) <= r2.get("y", 0) + 0.1 or
        r2.get("y", 0) + r2.get("d", 0) <= r1.get("y", 0) + 0.1
    )


def _is_adjacent_to(room: Dict, all_rooms: List[Dict], target_names: List[str]) -> bool:
    """Check if a room is adjacent to any of the target room types."""
    threshold = 1.5  # meters
    for other in all_rooms:
        if other.get("name") in target_names:
            # Check if rooms share a wall or are very close
            dx = abs((room.get("x", 0) + room.get("w", 0) / 2) - 
                     (other.get("x", 0) + other.get("w", 0) / 2))
            dy = abs((room.get("y", 0) + room.get("d", 0) / 2) - 
                     (other.get("y", 0) + other.get("d", 0) / 2))
            if dx < room.get("w", 0) / 2 + other.get("w", 0) / 2 + threshold and \
               dy < room.get("d", 0) / 2 + other.get("d", 0) / 2 + threshold:
                return True
    return False


def _is_living_room(name: str) -> bool:
    """Check if room is a living space."""
    living = ["Спальня", "Детская", "Гостиная", "Кабинет", "Комната отдыха"]
    return any(l in name for l in living)


def _normalize_rooms(rooms: List[Dict]) -> List[Dict]:
    """Normalize room data, add defaults."""
    for rm in rooms:
        rm.setdefault("is_wet", False)
        rm.setdefault("has_window", True)
        rm.setdefault("x", 0)
        rm.setdefault("y", 0)
        
        # Mark wet rooms
        wet_names = ["Ванная", "Туалет", "Санузел", "Душевая", "Моечная", "Парная", "Котельная"]
        if any(w in rm.get("name", "") for w in wet_names):
            rm["is_wet"] = True
    
    return rooms


def _auto_fix_snip(rooms: List[Dict], building: Dict) -> List[Dict]:
    """Attempt to auto-fix SNiP violations by scaling up undersized rooms."""
    bw = building.get("width", 0)
    bd = building.get("depth", 0)
    
    for rm in rooms:
        name = rm.get("name", "")
        category = _get_room_category(name)
        snip_rule = _find_snip_rule(category)
        
        if snip_rule:
            min_area = snip_rule.get("min_area", 0)
            min_w = snip_rule.get("min_width", 0)
            current_area = rm.get("w", 0) * rm.get("d", 0)
            
            # Scale up if too small
            if current_area < min_area:
                scale = (min_area * 1.05) / current_area  # 5% margin
                new_w = rm["w"] * scale
                new_d = rm["d"] * scale
                
                # Ensure minimum width
                if new_w < min_w:
                    new_w = min_w * 1.05
                    new_d = (min_area * 1.05) / new_w
                
                rm["w"] = round(new_w, 1)
                rm["d"] = round(new_d, 1)
    
    return rooms


def _parse_json_response(text: str) -> Optional[Dict]:
    """Parse JSON from Gemini response."""
    text = text.strip()
    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
    
    j = re.search(r'\{[\s\S]*\}', text)
    if j:
        try:
            return json.loads(j.group())
        except json.JSONDecodeError:
            try:
                # Fix trailing commas
                fixed = re.sub(r',\s*([}\]])', r'\1', j.group())
                return json.loads(fixed)
            except Exception:
                pass
    return None


def _build_architect_prompt(
    building_type: str, area: float, floors: int, rooms_count: int,
    site_width: float, site_depth: float, requirements: str,
    site_analysis: Optional[Dict]
) -> str:
    """Build the Gemini architect prompt."""
    
    site_context = ""
    if site_analysis:
        sa = site_analysis
        site_context = f"""
КОНТЕКСТ УЧАСТКА (из анализа плана):
- Размеры: {sa.get('site_width', site_width)}×{sa.get('site_depth', site_depth)} м
- Север: {sa.get('north_direction', 'не определён')}
- Существующие постройки: {len(sa.get('existing_buildings', []))} шт
- Деревья: {len(sa.get('trees', []))} шт
- Красные линии: {'обнаружены' if sa.get('red_lines_detected') else 'не обнаружены'}
- Подъезд: {'обнаружен' if sa.get('driveway_detected') else 'стандартный'}
- Описание: {sa.get('raw_description', '')[:200]}
"""
    
    return f"""Ты — профессиональный архитектор с 20-летним опытом. Спроектируй оптимальную планировку.

ПАРАМЕТРЫ ПРОЕКТА:
- Тип здания: {building_type}
- Общая площадь: {area} м²
- Этажность: {floors}
- Количество комнат: {rooms_count}
- Участок: {site_width} × {site_depth} м
{site_context}
Дополнительные требования: {requirements or 'стандартные'}

СНиП ТРЕБОВАНИЯ (ОБЯЗАТЕЛЬНО СОБЛЮДАТЬ):
- Гостиная ≥ 16 м², ширина ≥ 3.0 м
- Спальни ≥ 8 м² каждая, ширина ≥ 2.5 м
- Кухня ≥ 8 м², ширина ≥ 2.0 м
- Ванная ≥ 3.5 м², ширина ≥ 1.5 м
- Прихожая ≥ 5 м²
- Коридор ≥ 1.2 м шириной
- Высота потолков ≥ 2.5 м (учитывай в описании)
- Спальни — на юг/восток для инсоляции
- Мокрые зоны — рядом с коридором, НЕ над жилыми комнатами (многоэтаж)
- Кухня рядом с прихожей и гостиной
- Все комнаты должны помещаться в building.width × building.depth
- x ≥ 0, y ≥ 0, x+w ≤ width, y+d ≤ depth
- Комнаты НЕ перекрываются

ЗОНАЛЬНОЕ РАСПОЛОЖЕНИЕ:
- Прихожая — у входа (y=0 или x=0)
- Гостиная — южная сторона, центр
- Кухня — рядом с прихожей и гостиной
- Спальни — южная/восточная сторона, тихая зона
- Ванная/туалет — рядом с коридором, северная сторона
- Кабинет — тихая зона, может быть на 2 этаже

ОТВЕТЬ ТОЛЬКО JSON (без markdown-обёрток):
{{
  "building": {{
    "width": X.X,
    "depth": X.X,
    "floors": X,
    "rooms": [
      {{
        "name": "Название",
        "w": X.X,
        "d": X.X,
        "x": X.X,
        "y": X.X,
        "is_wet": false,
        "has_window": true
      }}
    ]
  }},
  "description": "краткое описание проекта на русском (2-3 предложения)",
  "reasoning": "объяснение архитектурных решений на русском (почему такое расположение, как учтены СНиП и участок)"
}}

ВАЖНО:
- Все размеры в метрах
- Координаты (x, y) — левый нижний угол комнаты
- Площадь комнаты = w × d
- building.width × building.depth ≈ {area * 0.7:.0f}-{area * 0.9:.0f} м² (футпринт)
- Для {floors} этажей: общая площадь {area} м² → площадь этажа ≈ {area // floors} м²
- Комнаты НЕ должны перекрываться!"""


def ai_architect_generate_description(
    building: Dict, rooms: List[Dict], site: Dict,
    gemini_api_key: str = "", model_name: str = "gemini-2.0-flash"
) -> str:
    """Generate a rich architectural description using Gemini."""
    if not gemini_api_key:
        return generate_project_description(building, site, SnipCheckResult(passed=True))
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)
        
        room_info = "\n".join(
            f"  - {r['name']}: {r['w']:.1f}×{r['d']:.1f} м ({r['w']*r['d']:.1f} м²)"
            for r in rooms
        )
        
        prompt = f"""Опиши архитектурный проект профессиональным языком (3-4 абзаца на русском):

Здание: {building.get('width', 0):.1f} × {building.get('depth', 0):.1f} м, {building.get('floors', 1)} этаж(а)
Участок: {site.get('width', 0):.0f} × {site.get('depth', 0):.0f} м

Помещения:
{room_info}

Опиши:
1. Общую концепцию и стиль
2. Зонирование (дневная/ночная зоны)
3. Особенности планировки
4. Рекомендации по благоустройству участка"""

        resp = model.generate_content(prompt)
        return resp.text[:2000]
    except Exception as e:
        logger.error(f"Description generation error: {e}")
        return generate_project_description(building, site, SnipCheckResult(passed=True))


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

@app.get("/ai-test2")
def ai_test2():
    return {"version": "v6-ai-test2", "status": "ok"}

@app.get("/ai-test")
def ai_test():
    """Test endpoint to verify ai_architect module loads."""
    try:
        from ai_architect import (
            ai_architect_v2, check_snip, generate_project_description,
            ai_architect_generate_description, SnipCheckResult, ArchitectResponse
        )
        # Test SNiP check without Gemini
        bldg = {"width": 10, "depth": 8, "floors": 1}
        rooms = [
            {"name": "Спальня", "w": 3.5, "d": 3.0, "x": 0, "y": 0},
            {"name": "Гостиная", "w": 5.0, "d": 4.0, "x": 0, "y": 3},
            {"name": "Кухня", "w": 3.0, "d": 3.0, "x": 0, "y": 7},
        ]
        snip = check_snip(bldg, rooms)
        return {"modules": "OK", "snip_check": snip.to_dict()}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()[:1000]}

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
    """Этап 4: AI-архитектор — Gemini генерирует оптимальную планировку."""
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "GEMINI_API_KEY not configured"}, status_code=400)
    
    try:
        import sys
        print(f"DEBUG: Calling ai_architect_v2 with bt={bt}, area={area}, model={model}", flush=True)
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
        print(f"DEBUG: result.success={result.success}, error={result.error}", flush=True)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"EXCEPTION: {e}\n{tb}", flush=True)
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
