"""AI Visualization Generator for Archiq AI.

Модуль генерации визуализаций фасадов и интерьеров на основе текстовых описаний
и архитектурных планов. Интеграция с нейросетями (Stable Diffusion, DALL-E 3).
"""

import os
import json
import base64
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Style(str, Enum):
    """Доступные стили генерации визуализаций."""
    PHOTOREALISM = "фотореализм"
    RENDER_3D = "3d-визуализация"
    ARCH_RENDER = "архитектурный-рендер"
    CONCEPT_ART = "концепт-арт"
    MODERN = "современный"
    MINIMALISM = "минимализм"
    HIGH_TECH = "хай-тек"
    INDUSTRIAL = "индустриал"
    NEOCLASSIC = "неоклассика"
    SCANDINAVIAN = "скандинавский"


class TimeOfDay(str, Enum):
    """Время суток для визуализации."""
    DAY = "день"
    SUNSET = "закат"
    NIGHT = "ночь"
    DAWN = "рассвет"
    TWILIGHT = "сумерки"


class Weather(str, Enum):
    """Погодные условия."""
    CLEAR = "ясно"
    CLOUDY = "пасмурно"
    RAIN = "дождь"
    SNOW = "снег"
    FOG = "туман"


class Lighting(str, Enum):
    """Тип освещения."""
    NATURAL = "естественное"
    ARTIFICIAL = "искусственное"
    MIXED = "смешанное"
    DRAMATIC = "драматичное"
    SOFT = "мягкое"


class MaterialFinish(str, Enum):
    """Материальные отделки."""
    GLASS = "стекло"
    CONCRETE = "бетон"
    WOOD = "дерево"
    METAL = "металл"
    BRICK = "кирпич"
    STONE = "камень"
    PLASTER = "штукатурка"
    STEEL = "сталь"


@dataclass
class GenerationParams:
    """Параметры генерации визуализации."""
    description: str
    style: Style = Style.PHOTOREALISM
    time_of_day: TimeOfDay = TimeOfDay.DAY
    weather: Weather = Weather.CLEAR
    lighting: Lighting = Lighting.NATURAL
    materials: List[MaterialFinish] = None
    viewpoint: str = "фасад"
    aspect_ratio: str = "16:9"
    resolution: str = "1024x1024"
    num_variants: int = 4
    base_plan: Optional[str] = None
    building_type: str = "жилое"
    architectural_style: str = "современный"
    region: str = "Казахстан"
    
    def __post_init__(self):
        if self.materials is None:
            self.materials = [MaterialFinish.GLASS, MaterialFinish.CONCRETE]


class PromptGenerator:
    """Генератор промптов для нейросетей на основе архитектурных параметров."""
    
    # Шаблоны промптов для разных стилей
    STYLE_PROMPTS = {
        Style.PHOTOREALISM: (
            "photorealistic architectural photography, highly detailed, "
            "professional architectural render, 8k resolution, "
            "cinematic lighting, realistic textures"
        ),
        Style.RENDER_3D: (
            "3D architectural visualization, CG render, "
            "digital art, detailed modeling, "
            "professional 3D graphics"
        ),
        Style.ARCH_RENDER: (
            "architectural render, architectural visualization, "
            "professional architectural photography, "
            "architectural digest style, ultra-detailed"
        ),
        Style.CONCEPT_ART: (
            "concept art, architectural concept design, "
            "digital painting, artistic interpretation, "
            "creative architectural visualization"
        ),
    }
    
    STYLE_PROMPTS.update({
        Style.MODERN: "modern architecture, contemporary design, clean lines, minimalist aesthetic",
        Style.MINIMALISM: "minimalist architecture, simple forms, geometric purity, zen aesthetic",
        Style.HIGH_TECH: "high-tech architecture, futuristic design, technical elements, advanced materials",
        Style.INDUSTRIAL: "industrial style, raw materials, exposed structures, loft aesthetic",
        Style.NEOCLASSIC: "neoclassical architecture, symmetrical design, classical elements",
        Style.SCANDINAVIAN: "scandinavian architecture, light colors, natural materials, cozy aesthetic",
    })
    
    TIME_PROMPTS = {
        TimeOfDay.DAY: "daytime, bright sunlight, clear sky, natural daylight",
        TimeOfDay.SUNSET: "sunset, golden hour, warm lighting, dramatic sky",
        TimeOfDay.NIGHT: "night time, illuminated building, artificial lighting, urban night",
        TimeOfDay.DAWN: "dawn, early morning, soft light, clear sky",
        TimeOfDay.TWILIGHT: "twilight, blue hour, moody lighting, transitional time",
    }
    
    WEATHER_PROMPTS = {
        Weather.CLEAR: "clear weather, blue sky, sunny day",
        Weather.CLOUDY: "overcast, cloudy sky, diffused lighting",
        Weather.RAIN: "rainy weather, wet surfaces, rain effects, moody atmosphere",
        Weather.SNOW: "snowy weather, white snow, winter scene, cold atmosphere",
        Weather.FOG: "foggy weather, misty, atmospheric fog, mysterious ambiance",
    }
    
    LIGHTING_PROMPTS = {
        Lighting.NATURAL: "natural lighting, sunlight, realistic shadows",
        Lighting.ARTIFICIAL: "artificial lighting, architectural lighting, LED, spotlights",
        Lighting.MIXED: "mixed lighting, natural and artificial, balanced illumination",
        Lighting.DRAMATIC: "dramatic lighting, strong shadows, high contrast",
        Lighting.SOFT: "soft lighting, gentle shadows, diffused illumination",
    }
    
    MATERIAL_PROMPTS = {
        MaterialFinish.GLASS: "glass facade, reflective surfaces, transparent elements",
        MaterialFinish.CONCRETE: "concrete surfaces, brutalist elements, raw concrete",
        MaterialFinish.WOOD: "wooden elements, timber facade, natural wood textures",
        MaterialFinish.METAL: "metal cladding, steel elements, metallic surfaces",
        MaterialFinish.BRICK: "brick facade, red brick, traditional masonry",
        MaterialFinish.STONE: "stone cladding, natural stone, solid materials",
        MaterialFinish.PLASTER: "plaster finish, smooth facade, painted surfaces",
        MaterialFinish.STEEL: "steel structure, exposed steel, industrial metal",
    }

    @classmethod
    def generate_prompt(cls, params: GenerationParams) -> str:
        """Генерация полного промпта для нейросети."""
        parts = []
        
        # Базовое описание
        parts.append(params.description)
        
        # Архитектурный стиль
        if params.architectural_style:
            parts.append(params.architectural_style)
        
        # Тип здания
        parts.append(f"{params.building_type} building")
        
        # Ракурс
        parts.append(params.viewpoint)
        
        # Стиль визуализации
        style_prompt = cls.STYLE_PROMPTS.get(params.style, "")
        if style_prompt:
            parts.append(style_prompt)
        
        # Время суток
        time_prompt = cls.TIME_PROMPTS.get(params.time_of_day, "")
        if time_prompt:
            parts.append(time_prompt)
        
        # Погода
        weather_prompt = cls.WEATHER_PROMPTS.get(params.weather, "")
        if weather_prompt:
            parts.append(weather_prompt)
        
        # Освещение
        lighting_prompt = cls.LIGHTING_PROMPTS.get(params.lighting, "")
        if lighting_prompt:
            parts.append(lighting_prompt)
        
        # Материалы
        if params.materials:
            material_prompts = [
                cls.MATERIAL_PROMPTS.get(m, "") 
                for m in params.materials 
                if m in cls.MATERIAL_PROMPTS
            ]
            if material_prompts:
                parts.append(", ".join(material_prompts))
        
        # Региональные особенности
        parts.append(f"suitable for {params.region} climate")
        
        # Качество
        parts.append("ultra-detailed, professional architectural photography")
        
        prompt = ", ".join(parts)
        return prompt.strip(", ")

    @classmethod
    def generate_prompts_batch(cls, params: GenerationParams) -> List[Dict[str, Any]]:
        """Генерация пакета промптов для пакетной генерации."""
        viewpoints = {
            "северный": "north elevation",
            "южный": "south elevation", 
            "восточный": "east elevation",
            "западный": "west elevation",
            "фасад": "main facade",
            "задний фасад": "rear facade",
            "панорама": "panoramic view",
            "угол": "corner view",
        }
        
        prompts = []
        base_viewpoint = viewpoints.get(params.viewpoint.lower(), params.viewpoint)
        
        for i in range(params.num_variants):
            variant_params = GenerationParams(
                description=params.description,
                style=params.style,
                time_of_day=params.time_of_day,
                weather=params.weather,
                lighting=params.lighting,
                materials=params.materials,
                viewpoint=base_viewpoint if i == 0 else f"{base_viewpoint}, variant {i+1}",
                aspect_ratio=params.aspect_ratio,
                resolution=params.resolution,
                num_variants=1,
                base_plan=params.base_plan,
                building_type=params.building_type,
                architectural_style=params.architectural_style,
                region=params.region,
            )
            
            prompt = cls.generate_prompt(variant_params)
            
            # Генерация seed для воспроизводимости
            seed_str = f"{prompt}{i}{params.description}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            
            prompts.append({
                "prompt": prompt,
                "seed": seed,
                "params": asdict(variant_params),
                "variant_id": i + 1,
            })
        
        return prompts


class VisualizationEngine:
    """Механизм генерации визуализаций."""
    
    def __init__(self):
        self.supported_apis = self._detect_apis()
    
    def _detect_apis(self) -> List[str]:
        """Определение доступных API для генерации."""
        apis = []
        
        # Проверка доступности различных API
        try:
            import openai
            apis.append("openai")
        except ImportError:
            pass
        
        try:
            import stability_sdk
            apis.append("stability")
        except ImportError:
            pass
        
        try:
            import google.generativeai as genai
            apis.append("gemini")
        except ImportError:
            pass
        
        return apis
    
    def generate_image(self, prompt: str, style: Style, seed: Optional[int] = None) -> Dict[str, Any]:
        """Генерация изображения через доступное API."""
        # Возвращает структуру для дальнейшего использования
        return {
            "prompt": prompt,
            "style": style,
            "seed": seed,
            "status": "ready_for_generation",
            "api_available": self.supported_apis,
        }
    
    def generate_3d_view(self, plan_data: str, viewpoints: List[str]) -> List[Dict[str, Any]]:
        """Генерация 3D-рендеров по плану здания."""
        results = []
        for viewpoint in viewpoints:
            results.append({
                "plan_data": plan_data,
                "viewpoint": viewpoint,
                "type": "3d_render",
                "status": "ready_for_generation",
            })
        return results


def get_available_styles() -> List[Dict[str, str]]:
    """Получение списка доступных стилей генерации."""
    styles = []
    
    # Основные стили визуализации
    for style in [Style.PHOTOREALISM, Style.RENDER_3D, Style.ARCH_RENDER, Style.CONCEPT_ART]:
        styles.append({
            "id": style,
            "name": style.value,
            "category": "visualization_style",
            "description": f"{style.value} визуализация",
        })
    
    # Архитектурные стили
    for style in [Style.MODERN, Style.MINIMALISM, Style.HIGH_TECH, Style.INDUSTRIAL, 
                  Style.NEOCLASSIC, Style.SCANDINAVIAN]:
        styles.append({
            "id": style,
            "name": style.value,
            "category": "architectural_style",
            "description": f"{style.value} стиль",
        })
    
    return styles


def get_available_times() -> List[Dict[str, str]]:
    """Получение списка доступных времен суток."""
    return [{
        "id": time.value,
        "name": time.value,
        "category": "time_of_day",
    } for time in TimeOfDay]


def get_available_weather() -> List[Dict[str, str]]:
    """Получение списка доступных погодных условий."""
    return [{
        "id": weather.value,
        "name": weather.value,
        "category": "weather",
    } for weather in Weather]


def get_available_materials() -> List[Dict[str, str]]:
    """Получение списка доступных материалов."""
    return [{
        "id": material.value,
        "name": material.value,
        "category": "material_finish",
    } for material in MaterialFinish]


if __name__ == "__main__":
    # Тест генерации промпта
    params = GenerationParams(
        description="Многоэтажный жилой дом",
        style=Style.PHOTOREALISM,
        time_of_day=TimeOfDay.SUNSET,
        weather=Weather.CLEAR,
        materials=[MaterialFinish.GLASS, MaterialFinish.CONCRETE],
        architectural_style="современный",
        num_variants=4,
    )
    
    generator = PromptGenerator()
    prompts = generator.generate_prompts_batch(params)
    
    print("Сгенерированные промпты:")
    for i, p in enumerate(prompts):
        print(f"\n{i+1}. {p['prompt']}")
        print(f"   Seed: {p['seed']}")
