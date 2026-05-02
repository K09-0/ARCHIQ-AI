"""Валидатор размеров по СНиП."""

from typing import List, Dict, Tuple
from .snipsnip_data_models import Rect, Metadata
from .snipsnip_snip_database import SNiPDatabase


class DimensionValidator:
    """Проверка чертежа на соответствие нормам СНиП."""

    def __init__(self):
        self.snip_db = SNiPDatabase()
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_building(self, metadata: Metadata, rooms: List[Dict]) -> Dict:
        """Полная валидация здания."""
        self.errors.clear()
        self.warnings.clear()

        # Проверка габаритов здания
        self._validate_overall_dimensions(metadata)

        # Проверка каждого помещения
        for room in rooms:
            self._validate_room(room)

        # Проверка эвакуации
        self._validate_evacuation(rooms, metadata)

        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }

    def _validate_overall_dimensions(self, metadata: Metadata):
        """Проверка габаритов всего здания."""
        if metadata.width < 5.0:
            self.warnings.append(f"Общая ширина здания {metadata.width}м крайне мала")
        if metadata.depth < 5.0:
            self.warnings.append(f"Общая глубина здания {metadata.depth}м крайне мала")
        if metadata.height < 2.5:
            self.errors.append(f"Высота здания {metadata.height}м < 2.5м")
        if metadata.floor_height < 2.5:
            self.errors.append(f"Высота этажа {metadata.floor_height}м < 2.5м")

    def _validate_room(self, room: Dict):
        """Проверка отдельного помещения."""
        name = room.get("name", "unknown")
        room_type = room.get("type", "living_room")
        bounds = room.get("bounds", None)

        if bounds is None:
            self.errors.append(f"{name}: отсутствуют координаты")
            return

        x, y, w, h = bounds
        area = w * h

        # Проверка размеров по СНиП
        if room_type != "corridor":
            errors = self.snip_db.validate_room(room_type, w, h, area)
            self.errors.extend([f"{name}: {e}" for e in errors])

            # Проверка соотношения сторон (предупреждения)
            aspect_ratio = w / h if h > 0 else 999
            if aspect_ratio > 3 or aspect_ratio < 1/3:
                self.warnings.append(
                    f"{name}: нерациональное соотношение сторон {aspect_ratio:.2f}"
                )

        # Проверка толщины стен
        wall_type = room.get("wall_type", "interior")
        wall_thickness = room.get("wall_thickness", 0.2)
        min_thickness = self.snip_db.min_distances.get(f"{wall_type}_wall_thickness", 0.1)
        if wall_thickness < min_thickness:
            self.errors.append(
                f"{name}: толщина стены {wall_thickness}м < минимальной {min_thickness}м"
            )

        # Проверка высоты потолка
        ceiling_height = room.get("ceiling_height", 3.0)
        min_height = self.snip_db.heights["room_min"]
        if ceiling_height < min_height:
            self.errors.append(
                f"{name}: высота потолка {ceiling_height}м < {min_height}м"
            )

        # Проверка расстояния до угла для окон и дверей
        door_positions = room.get("door_positions", [])
        window_positions = room.get("window_positions", [])
        min_corner_distance = self.snip_db.min_distances["door_to_corner"]

        for door in door_positions:
            dx, dy = door
            # Дистанция до ближайшего угла
            corners = [
                (x, y), (x + w, y),
                (x, y + h), (x + w, y + h)
            ]
            for cx, cy in corners:
                dist = ((dx - cx) ** 2 + (dy - cy) ** 2) ** 0.5
                if dist < min_corner_distance:
                    self.errors.append(
                        f"{name}: дверь слишком близко ({dist:.2f}м) к углу,"
                        f" минимум {min_corner_distance}м"
                    )

        for window in window_positions:
            wx, wy = window
            # Проверка высоты подоконника
            if wy < self.snip_db.heights.get("window_sill_min", 0.9):
                self.warnings.append(
                    f"{name}: высота окна может быть слишком низкой"
                )

        # Проверка дверных проемов между комнатами
        if room_type in ["bedroom", "living_room"]:
            if w < 2.4 and h < 2.4:
                self.warnings.append(
                    f"{name}: маленькое жилое помещение {w}x{h}м"
                )

    def _validate_evacuation(self, rooms: List[Dict], metadata: Metadata):
        """Проверка эвакуационных путей."""
        # Ищем коридоры
        corridors = [r for r in rooms if r.get("type") == "corridor"]
        exits = [r for r in rooms if r.get("has_exit", False)]

        if corridors:
            max_corridor_width = max(
                (r.get("bounds", (0, 0, 0.8, 0))[2] for r in corridors), default=0
            )
            min_corridor_width = min(
                (r.get("bounds", (0, 0, 999, 0))[2] for r in corridors), default=999
            )

            evacuation_width = self.snip_db.evacuation["corridor_width_min"]
            if min_corridor_width < evacuation_width:
                self.errors.append(
                    f"Эвакуация: ширина коридора {min_corridor_width}м < "
                    f"минимальной {evacuation_width}м"
                )

        if len(exits) < 2 and metadata.building_type != "residential":
            self.warnings.append(
                "Эвакуация: рекомендуется 2 и более выхода для безопасности"
            )

        # Проверяем расстояния до выхода
        for room in rooms:
            if not room.get("has_exit", False):
                rx, ry, rw, rh = room.get("bounds", (0, 0, 1, 1))
                room_center = (rx + rw / 2, ry + rh / 2)
                min_dist = 999
                for exit_room in exits:
                    ex, ey, ew, eh = exit_room.get("bounds", (0, 0, 1, 1))
                    exit_center = (ex + ew / 2, ey + eh / 2)
                    dist = ((room_center[0] - exit_center[0]) ** 2 +
                           (room_center[1] - exit_center[1]) ** 2) ** 0.5
                    min_dist = min(min_dist, dist)

                max_allowed = self.snip_db.evacuation["max_distance_to_exit"]
                if min_dist > max_allowed:
                    self.warnings.append(
                        f"{room.get('name', 'unknown')}: расстояние до выхода "
                        f"{min_dist:.1f}м > рекомендованного {max_allowed}м"
                    )