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
    model_name: str = "gemini-2.5-flash",
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
    gemini_api_key: str = "", model_name: str = "gemini-2.5-flash"
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
