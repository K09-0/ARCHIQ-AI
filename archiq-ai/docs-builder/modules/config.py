"""Конфигурация для генератора документации."""

import os
from enum import Enum
from pathlib import Path


class Language(str, Enum):
    """Поддерживаемые языки документации."""
    RUSSIAN = "ru"
    KAZAKH = "kk"


class DocumentType(str, Enum):
    """Типы документов."""
    TECHNICAL_PLAN = "technical_plan"      # Рабочие чертежи
    VISUALIZATION = "visualization"        # 3D-визуализации
    SPECIFICATION = "specification"        # Спецификации
    ESTIMATE = "estimate"                  # Сметы
    EXPLANATORY_NOTE = "explanatory_note"  # Пояснительная записка
    COMPLIANCE_DECLARATION = "compliance_declaration"  # Декларация соответствия СНиП
    APPENDICES = "appendices"              # Приложения


class DocumentSize(str, Enum):
    """Форматы документов по ГОСТ."""
    A4 = "A4"      # 210 x 297 мм
    A3 = "A3"      # 297 x 420 мм
    A2 = "A2"      # 420 x 594 мм
    A1 = "A1"      # 594 x 841 мм
    A0 = "A0"      # 841 x 1189 мм


class StampType(str, Enum):
    """Типы штампов по ГОСТ."""
    MAIN = "main"          # Основной штамп
    APPROVAL = "approval"  # Штамп согласования
    QUALITY = "quality"    # Штамп качества


# Пути
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
FONTS_DIR = BASE_DIR / "fonts"

# Создаем директории, если их нет
for dir_path in [TEMPLATES_DIR, STATIC_DIR, OUTPUT_DIR, DATA_DIR, FONTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Настройки документации
DEFAULT_LANGUAGE = Language.RUSSIAN
DEFAULT_PAGE_SIZE = DocumentSize.A4
WATERMARK_OPACITY = 0.1
QR_CODE_SIZE = 40  # мм
STAMP_OPACITY = 0.8

# Стандарты РК (ГОСТ, СНиП)
KZ_STANDARDS = {
    "GOST": {
        "title_ru": "ГОСТ",
        "title_kk": "ҒЫЛ",
        "margins": {"top": 10, "bottom": 10, "left": 15, "right": 15},  # мм
        "font_size": {"title": 16, "heading": 14, "body": 10, "footnote": 8},
    },
    "SNiP_RK": {
        "title_ru": "СНиП РК",
        "title_kk": "ҚР СНиП",
        "code": "Construction Standards RK",
    },
}

# Цвета
COLORS = {
    "primary": "#1a5276",      # Синий (главный)
    "secondary": "#2e86c1",   # Светло-синий
    "accent": "#f39c12",      # Оранжевый (акцент)
    "text": "#2c3e50",        # Темно-синий (текст)
    "text_light": "#7f8c8d",  # Серый (второстепенный текст)
    "border": "#bdc3c7",      # Серый (границы)
    "watermark": "#95a5a6",   # Серый полупрозрачный (водяной знак)
}

# Настройки генерации
GENERATION_CONFIG = {
    "dpi": 300,                    # Разрешение для изображений
    "image_quality": 95,           # Качество сжатия
    "max_image_width": 1800,       # Максимальная ширина изображения
    "enable_watermark": True,      # Включить водяные знаки
    "enable_qr_codes": True,       # Включить QR-коды
    "enable_stamps": True,         # Включить штампы
    "auto_toc": True,              # Автоматическое оглавление
    "auto_numbering": True,        # Автоматическая нумерация
    "cross_references": True,      # Перекрестные ссылки
}

# Шаблоны имен файлов
FILE_NAME_TEMPLATES = {
    "project": "{project_id}_{project_name}",
    "document": "{doc_type}_{doc_id}_{title}",
    "package": "{project_id}_documentation_package",
    "archive": "{project_id}_docs_{timestamp}.zip",
}
