"""Модели данных для чертежей и метаданных."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import uuid


class Point2D:
    """2D точка (x, y) в метрах."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def shift(self, dx: float, dy: float):
        return Point2D(self.x + dx, self.y + dy)


@dataclass
class Rect:
    """Прямоугольник (x, y, width, height)."""
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Point2D:
        return Point2D(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


@dataclass
class Metadata:
    """Метаданные чертежа."""
    building_type: str
    total_area: float = 0.0
    perimeter: float = 0.0
    width: float = 0.0
    depth: float = 0.0
    height: float = 0.0
    num_floors: int = 1
    floor_height: float = 3.0
    wall_thickness: Dict[str, float] = field(default_factory=lambda: {
        "exterior": 0.5,
        "interior": 0.2,
    })
    materials: Dict[str, str] = field(default_factory=dict)
    rooms: List[Dict] = field(default_factory=list)
    building_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def add_room(self, name: str, area: float, room_type: str):
        self.rooms.append({
            "name": name,
            "area": area,
            "type": room_type
        })


@dataclass
class Dimension:
    """Размерная линия."""
    start: Point2D
    end: Point2D
    dimension_line_offset: float = 0.15
    text: str = ""
    text_height: float = 0.035
