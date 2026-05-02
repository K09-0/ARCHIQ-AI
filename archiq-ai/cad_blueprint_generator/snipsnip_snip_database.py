"""База СНиП - нормы и правила для валидации чертежей."""

from typing import Dict, List, Tuple, Optional


class SNiPDatabase:
    """База данных норм СНиП для валидации чертежей."""

    def __init__(self):
        # Минимальные габариты помещений (в метрах)
        self.min_room_dimensions = {
            "living_room": (3.0, 2.5),
            "bedroom": (2.5, 2.5),
            "kitchen": (2.4, 2.0),
            "bathroom": (1.5, 1.5),
            "hallway": (1.5, 0.9),
            "garage": (2.3, 4.8),
            "office": (2.5, 2.5),
            "conference": (3.0, 4.0),
            "corridor": (None, 0.9),  # ширина
        }

        # Минимальные расстояния (в метрах)
        self.min_distances = {
            "wall_to_window": 0.5,        # До угла окна
            "door_between_rooms": 0.4,    # Между дверями
            "door_to_corner": 0.4,        # От угла до двери
            "stair_to_wall": 0.2,         # Лестница от стены
            "partition_thickness": 0.1,   # Толщина перегородки
            "exterior_wall_thickness": 0.5,
        }

        # Высоты (в метрах)
        self.heights = {
            "room_min": 2.5,
            "floor_standard": 3.0,
            "garage_min": 2.5,
            "basement_min": 2.2,
            "window_sill_min": 0.9,
            "window_sill_living": 1.0,
            "ceiling_slope_min": 2.5,    # При уклоне крыши
        }

        # Площади (в кв. метрах)
        self.areas = {
            "living_room_min": 16.0,
            "bedroom_min": 10.0,
            "kitchen_min": 8.0,
            "bathroom_min": 4.0,
            "wc_min": 2.0,
            "hallway_min": 3.0,
            "garage_one_car": 10.0,
            "garage_two_car": 20.0,
            "office_workstation": 8.0,
        }

        # Эвакуация (в метрах)
        self.evacuation = {
            "corridor_width_min": 1.1,   # Ширина коридора для эвакуации
            "exit_width_min": 1.1,       # Ширина выхода
            "exit_door_width": 0.9,      # Ширина двери эвакуации
            "max_distance_to_exit": 30.0,# Макс. расстояние до выхода
            "exit_doors_count_min": 2,   # Мин. кол-во выходов
            "stair_width_min": 1.2,      # Ширина лестницы
        }

        # Инженерные сети
        self.mep = {
            "pipe_clearance": 0.05,      # Отступ для труб
            "electrical_box_height": 1.5,# Высота розеток
            "switch_height": 1.4,        # Высота выключателей
            "ventilation_clearance": 0.3,# Отступ для вентиляции
            "plumbing_stack_width": 0.15,# Ширина стояка слива
            "electrical_panel_clearance": 1.0, # Ограничение перед щитком
        }

        # Парковка
        self.parking = {
            "space_width": 2.3,           # Ширина машиноместа
            "space_length": 4.8,          # Длина машиноместа
            "aisle_width_one_way": 3.0,   # Односторонняя подъездная
            "aisle_width_two_way": 5.0,   # Двусторонняя подъездная
            "ramp_slope_max": 0.15,       # Макс. уклон пандуса (15%)
        }

    def validate_room(self, room_type: str, width: float, depth: float, area: float) -> List[str]:
        """Валидация размеров помещения по СНиП. Возвращает список ошибок."""
        errors = []

        if room_type in self.min_room_dimensions:
            min_w, min_d = self.min_room_dimensions[room_type]
            if min_w and width < min_w:
                errors.append(f"{room_type}: ширина {width}м < минимальной {min_w}м")
            if min_d and depth < min_d:
                errors.append(f"{room_type}: глубина {depth}м < минимальной {min_d}м")

        room_key = f"{room_type.lower()}_min"
        if room_key in self.areas and area < self.areas[room_key]:
            errors.append(f"{room_type}: площадь {area}м² < минимальной {self.areas[room_key]}м²")

        return errors

    def validate_evacuation(self, corridor_width: float, exit_width: float,
                          max_distance: float) -> List[str]:
        """Валидация эвакуационных путей."""
        errors = []
        if corridor_width < self.evacuation["corridor_width_min"]:
            errors.append(f"Коридор: ширина {corridor_width}м < минимальной {self.evacuation['corridor_width_min']}м")
        if exit_width < self.evacuation["exit_width_min"]:
            errors.append(f"Выход: ширина {exit_width}м < минимальной {self.evacuation['exit_width_min']}м")
        if max_distance > self.evacuation["max_distance_to_exit"]:
            errors.append(f"Расстояние до выхода {max_distance}м > максимального {self.evacuation['max_distance_to_exit']}м")
        return errors

    def get_standard_dimension(self, key: str, default=None):
        """Получить стандартный размер по ключу."""
        return self.min_distances.get(key, default)
