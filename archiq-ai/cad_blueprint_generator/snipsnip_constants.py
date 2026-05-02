"""Константы и настройки проекта."""

from enum import Enum

class BuildingType(Enum):
    AUTO_SERVICE = "auto_service"
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"

class LayerType(Enum):
    STRUCTURAL = "structural"          # Несущие конструкции
    MEP = "mep"                        # Инженерия (Mechanical, Electrical, Plumbing)
    PARKING = "parking"                # Парковка
    SERVICE_ZONES = "service_zones"    # Зоны обслуживания
    EVACUATION = "evacuation"          # Эвакуационные пути
    ANNOTATIONS = "annotations"        # Размеры и текст
    SECTIONS = "sections"              # Разрезы и фасады

class MaterialType(Enum):
    CONCRETE = "concrete"
    BRICK = "brick"
    REINFORCED_CONCRETE = "reinforced_concrete"
    WOOD = "wood"
    METAL = "metal"
    GLASS = "glass"

# Цвета для различных слоев (RGB)
LAYER_COLORS = {
    LayerType.STRUCTURAL: (255, 0, 0),        # Красный
    LayerType.MEP: (0, 0, 255),               # Синий
    LayerType.PARKING: (255, 255, 0),         # Желтый
    LayerType.SERVICE_ZONES: (0, 255, 0),     # Зеленый
    LayerType.EVACUATION: (255, 165, 0),      # Оранжевый
    LayerType.ANNOTATIONS: (0, 0, 0),         # Черный
    LayerType.SECTIONS: (128, 0, 128),        # Фиолетовый
}

# Стили линий
LINE_STYLES = {
    "wall": "CONTINUOUS",
    "hidden": "DASHED",
    "center": "CENTER",
    "dashed": "DASHED",
    "dot": "DOT",
}

# Минимальные размеры по СНиП (в метрах)
MIN_DIMENSIONS = {
    "room_width": 2.0,           # Минимальная ширина помещения
    "room_height": 2.5,          # Минимальная высота помещения
    "hallway_width": 0.9,        # Ширина коридора
    "stair_width": 1.2,          # Ширина лестницы
    "door_width": 0.9,           # Ширина двери
    "door_corner": 0.4,          # Расстояние от угла до двери
    "window_sill": 0.9,          # Высота подоконника
    "evacuation_width": 1.1,     # Ширина эвакуационного выхода
    "parking_space": (2.3, 4.8), # Размер машиноместа (ширина, длина)
    "parking_aisle": 5.0,        # Ширина подъездной дорожки
}

# Толщины стен по умолчанию (в метрах)
WALL_THICKNESS = {
    "exterior": 0.5,             # Наружная стена
    "interior": 0.2,             # Внутренняя перегородка
    "partition": 0.1,            # Легкая перегородка
}

# Высоты
DEFAULT_FLOOR_HEIGHT = 3.0       # Высота этажа
MEP_CLEARANCE = 0.1              # Отступ для инженерных сетей от потолка