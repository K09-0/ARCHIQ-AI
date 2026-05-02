"""Трассировка инженерных сетей: водопровод, канализация, электрика."""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .snipsnip_data_models import Point2D
from .snipsnip_constants import LayerType


class NetworkType(Enum):
    WATER_SUPPLY = "water_supply"
    SEWER = "sewer"
    ELECTRICAL = "electrical"
    VENTILATION = "ventilation"
    HEATING = "heating"


@dataclass
class NetworkNode:
    """Узел сети (розетка, вентиль, выключатель)."""
    position: Point2D
    node_type: str
    floor_level: int = 0
    height: float = 0.0  # Высота от уровня пола (для MEP)


@dataclass
class NetworkPipe:
    """Труба/канал сети."""
    start: Point2D
    end: Point2D
    network_type: NetworkType
    diameter: float = 0.05  # Диаметр трубы в метрах
    floor_level: int = 0


@dataclass
class Circuit:
    """Электрическая цепь."""
    name: str
    outlets: List[NetworkNode]
    switches: List[NetworkNode]
    breaker_amperage: int = 16
    cable_type: str = "copper_2.5mm2"


class NetworkTracer:
    """Трассировка и проектирование инженерных сетей здания."""

    def __init__(self):
        self.nodes: List[NetworkNode] = []
        self.pipes: List[NetworkPipe] = []
        self.circuits: List[Circuit] = []
        self.min_clearance = 0.05  # Минимальный отступ между сетями

    def add_node(self, x: float, y: float, node_type: str,
                 floor: int = 0, height: float = 0.0) -> NetworkNode:
        """Добавить узел сети."""
        node = NetworkNode(
            position=Point2D(x, y),
            node_type=node_type,
            floor_level=floor,
            height=height
        )
        self.nodes.append(node)
        return node

    def trace_pipe_straight(self, start: Point2D, end: Point2D,
                          network_type: NetworkType, floor: int = 0,
                          diameter: float = 0.05) -> NetworkPipe:
        """Протянуть трубу прямой линией."""
        pipe = NetworkPipe(
            start=start,
            end=end,
            network_type=network_type,
            diameter=diameter,
            floor_level=floor
        )
        self.pipes.append(pipe)
        return pipe

    def trace_pipe_l_shaped(self, start: Point2D, corner: Point2D,
                          end: Point2D, network_type: NetworkType,
                          floor: int = 0, diameter: float = 0.05) -> List[NetworkPipe]:
        """Протянуть L-образную трубу (с одним изгибом)."""
        pipes = []
        pipe1 = self.trace_pipe_straight(start, corner, network_type, floor, diameter)
        pipe2 = self.trace_pipe_straight(corner, end, network_type, floor, diameter)
        return [pipe1, pipe2]

    def trace_pipe_u_shaped(self, start: Point2D, corner1: Point2D,
                          corner2: Point2D, end: Point2D,
                          network_type: NetworkType, floor: int = 0,
                          diameter: float = 0.05) -> List[NetworkPipe]:
        """Протянуть U-образную трубу (с двумя изгибами)."""
        pipes = []
        pipe1 = self.trace_pipe_straight(start, corner1, network_type, floor, diameter)
        pipe2 = self.trace_pipe_straight(corner1, corner2, network_type, floor, diameter)
        pipe3 = self.trace_pipe_straight(corner2, end, network_type, floor, diameter)
        return [pipe1, pipe2, pipe3]

    def add_electrical_circuit(self, num_outlets: int, room_bounds: Tuple[float, float, float, float],
                              circuit_name: str = "", floor: int = 0) -> Circuit:
        """Сгенерировать электрическую цепь для помещения.
        Расставляет розетки вдоль стен и выключатели у входа.
        """
        x, y, w, h = room_bounds
        circuit_name = circuit_name or f"Circuit_{len(self.circuits) + 1}"

        outlets = []
        switches = []

        # Выключатель у входа (левый нижний угол)
        switch_height = 1.4  # Стандартная высота выключателя
        switch = self.add_node(
            x + 0.1, y + 0.1, "switch",
            floor=floor, height=switch_height
        )
        switches.append(switch)

        # Розетки вдоль стен (по периметру)
        outlets_per_wall = max(1, num_outlets // 4)
        spacing_x = w / (outlets_per_wall + 1) if outlets_per_wall > 0 else 0
        spacing_y = h / (outlets_per_wall + 1) if outlets_per_wall > 0 else 0

        outlet_height = 0.3  # Высота розетки от пола

        # Нижняя стена
        for i in range(outlets_per_wall):
            node = self.add_node(
                x + spacing_x * (i + 1), y + 0.05,
                "outlet", floor=floor, height=outlet_height
            )
            outlets.append(node)

        # Правая стена
        for i in range(outlets_per_wall):
            node = self.add_node(
                x + w - 0.05, y + spacing_y * (i + 1),
                "outlet", floor=floor, height=outlet_height
            )
            outlets.append(node)

        # Верхняя стена
        for i in range(outlets_per_wall):
            node = self.add_node(
                x + w - spacing_x * (i + 1), y + h - 0.05,
                "outlet", floor=floor, height=outlet_height
            )
            outlets.append(node)

        # Левая стена
        for i in range(outlets_per_wall):
            node = self.add_node(
                x + 0.05, y + h - spacing_y * (i + 1),
                "outlet", floor=floor, height=outlet_height
            )
            outlets.append(node)

        circuit = Circuit(
            name=circuit_name,
            outlets=outlets,
            switches=switches,
            breaker_amperage=16
        )
        self.circuits.append(circuit)
        return circuit

    def generate_supply_water(self, entry_point: Point2D,
                            rooms: List[Dict], floor: int = 0) -> List[NetworkPipe]:
        """Сгенерировать разводку холодной воды к помещениям."""
        pipes = []
        current = entry_point
        pipe_diameter = 0.025  # 25 мм для квартиры

        for i, room in enumerate(rooms):
            rx, ry, rw, rh = room["bounds"]
            target = Point2D(rx + 0.1, ry + rh / 2)  # Точка вмеша воды

            if i == 0:
                # Прямая труба к первому помещению
                pipe = self.trace_pipe_straight(current, target,
                                               NetworkType.WATER_SUPPLY, floor, pipe_diameter)
                pipes.append(pipe)
            else:
                # T-образный отвод от предыдущей трубы
                prev_pipe = pipes[-1]
                tee_point = prev_pipe.end
                # Горизонтальный отвод до нужного x
                if abs(tee_point.x - target.x) > 0.01:
                    corner = Point2D(target.x, tee_point.y)
                    pipe1, pipe2 = self.trace_pipe_l_shaped(
                        tee_point, corner, target,
                        NetworkType.WATER_SUPPLY, floor, pipe_diameter
                    )
                    pipes.extend([pipe1, pipe2])
                else:
                    pipe = self.trace_pipe_straight(
                        tee_point, target,
                        NetworkType.WATER_SUPPLY, floor, pipe_diameter
                    )
                    pipes.append(pipe)

            # Внутренняя разводка в помещение (минимальная)
            internal_target = Point2D(target.x, target.y + 0.3)
            internal_pipe = self.trace_pipe_straight(
                target, internal_target,
                NetworkType.WATER_SUPPLY, floor, 0.02
            )
            pipes.append(internal_pipe)

        return pipes

    def generate_sewer(self, rooms: List[Dict],
                      main_stack: Point2D, floor: int = 0) -> List[NetworkPipe]:
        """Сгенерировать канализационные трубы к стояку."""
        pipes = []
        pipe_diameter = 0.075  # 75 мм

        for room in rooms:
            if room.get("type") in ["bathroom", "kitchen"]:
                rx, ry, rw, rh = room["bounds"]
                # Точка вмеша канализации (низ помещения, ближе к стояку)
                target_x = rx + rw / 2
                target_y = ry + 0.15
                target = Point2D(target_x, target_y)

                # Уклон к стояку (всегда вниз по оси Y к основной трубе)
                if floor == 0:
                    # На первом этаже - горизонтально к стояку
                    if abs(target_x - main_stack.x) > 0.01:
                        corner = Point2D(main_stack.x, target_y)
                        pipe1, pipe2 = self.trace_pipe_l_shaped(
                            target, corner, main_stack,
                            NetworkType.SEWER, floor, pipe_diameter
                        )
                        pipes.extend([pipe1, pipe2])
                    else:
                        pipe = self.trace_pipe_straight(
                            target, main_stack, NetworkType.SEWER, floor, pipe_diameter
                        )
                        pipes.append(pipe)
                else:
                    # На верхних этажах - сначала вверх, затем вниз к стояку
                    up_point = Point2D(target_x, target_y - 0.3)  # Вверх от пола
                    pipe1, pipe2 = self.trace_pipe_l_shaped(
                        target, up_point, Point2D(main_stack.x, up_point.y),
                        NetworkType.SEWER, floor, pipe_diameter
                    )
                    pipes.extend([pipe1, pipe2])

        return pipes

    def get_network_elements(self, network_type: NetworkType) -> Dict:
        """Получить все элементы указанного типа сети."""
        nodes = [n for n in self.nodes if any(
            p.network_type == network_type and (
                abs(p.start.x - n.position.x) < 0.001 and abs(p.start.y - n.position.y) < 0.001 or
                abs(p.end.x - n.position.x) < 0.001 and abs(p.end.y - n.position.y) < 0.001
            ) for p in self.pipes
        )]
        pipes = [p for p in self.pipes if p.network_type == network_type]
        return {"nodes": nodes, "pipes": pipes}

    def clear(self):
        """Очистить все сети."""
        self.nodes.clear()
        self.pipes.clear()
        self.circuits.clear()
