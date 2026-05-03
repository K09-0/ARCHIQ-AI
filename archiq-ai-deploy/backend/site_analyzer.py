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
