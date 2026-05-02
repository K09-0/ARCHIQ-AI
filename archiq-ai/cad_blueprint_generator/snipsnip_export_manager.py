"""Экспорт чертежей в форматы DXF, SVG, PDF."""

import os
import io
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import ezdxf
import svgwrite
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib import colors

try:
    from .snipsnip_constants import LayerType, LAYER_COLORS, LINE_STYLES
    from .snipsnip_data_models import Point2D, Rect, Metadata, Dimension
except ImportError:
    from snipsnip_constants import LayerType, LAYER_COLORS, LINE_STYLES
    from snipsnip_data_models import Point2D, Rect, Metadata, Dimension


class ExportManager:
    """Управление экспортом чертежей в различные форматы."""

    # Единицы: 1 метр = 1000 unit в DXF (convenient scale)
    SCALE = 1000.0

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_dxf(self, filename: str,
                   walls: List[Tuple[Point2D, Point2D, Dict]],
                   doors: List[Dict],
                   windows: List[Dict],
                   dimensions: List[Dimension],
                   metadata: Metadata,
                   layers: Dict[str, List],
                   network_elements: Optional[Dict] = None) -> str:
        """Экспорт в формат DXF для AutoCAD.

        Args:
            filename: Имя выходного файла (без расширения)
            walls: Список стен (start, end, атрибуты)
            doors: Список дверей
            windows: Список окон
            dimensions: Размерные линии
            metadata: Метаданные чертежа
            layers: Дополнительные слои
            network_elements: Инженерные сети

        Returns:
            Путь к сохраненному файлу
        """
        doc = ezdxf.new('R2013', setup=True)
        msp = doc.modelspace()

        # Настройка единиц (метры)
        doc.header['$INSUNITS'] = 1  # Unitless

        # Создаем стили линий
        self._create_line_types(doc)

        # Создаем слои
        self._create_dxf_layers(doc)

        # Рисуем несущие конструкции
        for wall_start, wall_end, attrs in walls:
            self._draw_wall(msp, wall_start, wall_end, attrs)

        # Рисуем двери
        for door in doors:
            self._draw_door(msp, door)

        # Рисуем окна
        for window in windows:
            self._draw_window(msp, window)

        # Рисуем размерные линии
        for dim in dimensions:
            self._draw_dimension(msp, dim)

        # Рисуем инженерные сети
        if network_elements:
            self._draw_networks(msp, network_elements)

        # Рисуем дополнительные слои (парковка, зоны и т.д.)
        for layer_name, elements in layers.items():
            self._draw_layer_elements(msp, layer_name, elements)

        # Добавляем метаданные (XData)
        self._add_metadata(doc, metadata)

        # Сохраняем файл
        filepath = os.path.join(self.output_dir, f"{filename}.dxf")
        doc.saveas(filepath)
        return filepath

    def _create_line_types(self, doc):
        """Создать пользовательские типы линий."""
        ltypes = doc.linetypes
        if "DASHED" not in ltypes:
            ltypes.new("DASHED", dxfattribs={
                'pattern': [10.0, 5.0],
                'description': "Dashed line"
            })
        if "HIDDEN" not in ltypes:
            ltypes.new("HIDDEN", dxfattribs={
                'pattern': [5.0, 5.0],
                'description': "Hidden line"
            })
        if "CENTER" not in ltypes:
            ltypes.new("CENTER", dxfattribs={
                'pattern': [10.0, 2.0, 2.0, 2.0],
                'description': "Center line"
            })

    def _create_dxf_layers(self, doc):
        """Создать слои в DXF файле."""
        layers = doc.layers
        for layer_type in LayerType:
            color_num = self._get_closest_aco_color(layer_type)
            layers.new(
                name=layer_type.value.upper(),
                dxfattribs={
                    'color': color_num,
                    'linetype': 'CONTINUOUS'
                }
            )

        # Специальные слои для сетей
        net_layers = {
            "NETWORK_WATER": 6,  # Cyan
            "NETWORK_SEWER": 2,  # Yellow
            "NETWORK_ELECTRICAL": 5,  # Blue
            "NETWORK_DIMENSIONS": 7,  # White
        }
        for name, color in net_layers.items():
            layers.new(name=name, dxfattribs={'color': color})

    def _get_closest_aco_color(self, layer_type: LayerType) -> int:
        """Получить ближайший ACO цвет AutoCAD по RGB."""
        rgb = LAYER_COLORS.get(layer_type, (0, 0, 0))
        # AutoCAD ACO цвета (ограниченная палитра)
        aco_palette = {
            1: (255, 0, 0),    # Red
            2: (255, 255, 0),  # Yellow
            3: (0, 255, 0),    # Green
            4: (0, 255, 255),  # Cyan
            5: (0, 0, 255),    # Blue
            6: (255, 0, 255),  # Magenta
            7: (255, 255, 255), # White
            8: (0, 0, 0),      # Black
        }
        # Находим ближайший цвет
        best_color = 7
        best_dist = float('inf')
        for aco, color in aco_palette.items():
            dist = sum((a - b) ** 2 for a, b in zip(rgb, color))
            if dist < best_dist:
                best_dist = dist
                best_color = aco
        return best_color

    def _draw_wall(self, msp, start: Point2D, end: Point2D, attrs: Dict):
        """Нарисовать стену."""
        layer = attrs.get("layer", LayerType.STRUCTURAL.value.upper())
        thickness = attrs.get("thickness", 0.2)

        # Основная линия стены
        msp.add_line(
            (start.x * self.SCALE, start.y * self.SCALE),
            (end.x * self.SCALE, end.y * self.SCALE),
            dxfattribs={
                'layer': layer,
                'color': self._get_closest_aco_color(LayerType.STRUCTURAL),
                'linetype': 'CONTINUOUS',
            }
        )

        # Двойная линия для толщины (если стена толстая)
        if thickness > 0.15:
            # Смещаем параллельную линию
            dx = end.x - start.x
            dy = end.y - start.y
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                nx = -dy / length * thickness
                ny = dx / length * thickness
                msp.add_line(
                    ((start.x + nx) * self.SCALE, (start.y + ny) * self.SCALE),
                    ((end.x + nx) * self.SCALE, (end.y + ny) * self.SCALE),
                    dxfattribs={'layer': layer}
                )

    def _draw_door(self, msp, door: Dict):
        """Нарисовать дверь."""
        x = door['x'] * self.SCALE
        y = door['y'] * self.SCALE
        width = door.get('width', 0.9) * self.SCALE
        arc_radius = door.get('arc_radius', 0.15) * self.SCALE

        # Дверная линия (дуга)
        msp.add_arc(
            center=(x + width/2, y),
            radius=arc_radius,
            start_param=0,
            end_param=180,
            dxfattribs={
                'layer': LayerType.ANNOTATIONS.value.upper(),
                'color': 3,
            }
        )

    def _draw_window(self, msp, window: Dict):
        """Нарисовать окно."""
        x = window['x'] * self.SCALE
        y = window['y'] * self.SCALE
        width = window.get('width', 1.0) * self.SCALE
        height = window.get('height', 0.1) * self.SCALE

        msp.add_line(
            (x, y), (x + width, y),
            dxfattribs={
                'layer': LayerType.ANNOTATIONS.value.upper(),
                'color': 4,
                'linetype': 'HIDDEN'
            }
        )

    def _draw_dimension(self, msp, dim: Dimension):
        """Нарисовать размерную линию."""
        msp.add_linear_dim(
            base=(dim.start.x * self.SCALE, dim.start.y * self.SCALE),
            p1=(dim.start.x * self.SCALE, dim.start.y * self.SCALE),
            p2=(dim.end.x * self.SCALE, dim.end.y * self.SCALE),
            dimlinepoint=((dim.start.x + dim.end.x) / 2 * self.SCALE,
                         (dim.start.y + dim.end.y) / 2 * self.SCALE + dim.dimension_line_offset * self.SCALE),
            dxfattribs={
                'layer': LayerType.ANNOTATIONS.value.upper(),
                'color': 7,
            }
        )
        # Текст размера
        if dim.text:
            # Здесь можно добавить текст размера
            pass

    def _draw_networks(self, msp, network_elements: Dict):
        """Нарисовать инженерные сети."""
        # Трубы водоснабжения
        water_layer = "NETWORK_WATER"
        for pipe in network_elements.get("water_pipes", []):
            msp.add_line(
                (pipe['start'].x * self.SCALE, pipe['start'].y * self.SCALE),
                (pipe['end'].x * self.SCALE, pipe['end'].y * self.SCALE),
                dxfattribs={
                    'layer': water_layer,
                    'color': 6,  # Cyan
                    'lineweight': 15,
                }
            )

        # Канализация
        sewer_layer = "NETWORK_SEWER"
        for pipe in network_elements.get("sewer_pipes", []):
            msp.add_line(
                (pipe['start'].x * self.SCALE, pipe['start'].y * self.SCALE),
                (pipe['end'].x * self.SCALE, pipe['end'].y * self.SCALE),
                dxfattribs={
                    'layer': sewer_layer,
                    'color': 2,  # Yellow
                    'lineweight': 20,
                }
            )

        # Электричество
        elec_layer = "NETWORK_ELECTRICAL"
        for pipe in network_elements.get("electrical_pipes", []):
            msp.add_line(
                (pipe['start'].x * self.SCALE, pipe['start'].y * self.SCALE),
                (pipe['end'].x * self.SCALE, pipe['end'].y * self.SCALE),
                dxfattribs={
                    'layer': elec_layer,
                    'color': 5,  # Blue
                    'lineweight': 10,
                }
            )

    def _draw_layer_elements(self, msp, layer_name: str, elements: List):
        """Нарисовать элементы дополнительного слоя."""
        for elem in elements:
            if elem.get('type') == 'rectangle':
                msp.add_lwpolyline(
                    [
                        (elem['x'] * self.SCALE, elem['y'] * self.SCALE),
                        ((elem['x'] + elem['width']) * self.SCALE, elem['y'] * self.SCALE),
                        ((elem['x'] + elem['width']) * self.SCALE, (elem['y'] + elem['height']) * self.SCALE),
                        (elem['x'] * self.SCALE, (elem['y'] + elem['height']) * self.SCALE),
                        (elem['x'] * self.SCALE, elem['y'] * self.SCALE),
                    ],
                    dxfattribs={
                        'layer': layer_name.upper(),
                        'color': 3,
                    }
                )

    def _add_metadata(self, doc, metadata: Metadata):
        """Добавить метаданные чертежа."""
        # Сохраняем в заголовок DXF
        doc.header['$ACADVER'] = 'AC1027'  # AutoCAD 2013
        doc.header['$EXTMIN'] = (0, 0, 0)
        doc.header['$EXTMAX'] = (
            metadata.width * self.SCALE,
            metadata.depth * self.SCALE,
            metadata.height * self.SCALE
        )

    def export_svg(self, filename: str,
                   walls: List[Tuple[Point2D, Point2D, Dict]],
                   doors: List[Dict],
                   windows: List[Dict],
                   dimensions: List[Dimension],
                   metadata: Metadata,
                   layers: Dict[str, List],
                   width: int = 1200,
                   height: int = 800) -> str:
        """Экспорт в SVG формат.

        Args:
            filename: Имя выходного файла
            walls, doors, windows, dimensions, metadata, layers: см. export_dxf
            width, height: Размеры SVG в пикселях

        Returns:
            Путь к сохраненному файлу
        """
        # Вычисляем масштаб
        padding = 50
        scale_x = (width - 2 * padding) / max(metadata.width, 0.1)
        scale_y = (height - 2 * padding) / max(metadata.depth, 0.1)
        scale = min(scale_x, scale_y)

        dwg = svgwrite.Drawing(
            os.path.join(self.output_dir, f"{filename}.svg"),
            size=(f"{width}px", f"{height}px")
        )

        # Фон
        dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))

        # Смещение для центровки
        offset_x = (width - metadata.width * scale) / 2
        offset_y = (height - metadata.depth * scale) / 2

        def transform(p: Point2D):
            return (
                p.x * scale + offset_x,
                p.y * scale + offset_y
            )

        # Рисуем стены
        for wall_start, wall_end, attrs in walls:
            ts = transform(wall_start)
            te = transform(wall_end)
            dwg.add(dwg.line(
                start=ts, end=te,
                stroke='red',
                stroke_width=max(1, attrs.get('thickness', 0.2) * scale * 0.3),
                stroke_linecap='round'
            ))

        # Рисуем двери
        for door in doors:
            x, y = transform(Point2D(door['x'], door['y']))
            width_door = door.get('width', 0.9) * scale
            dwg.add(dwg.arc(
                center=(x + width_door / 2, y),
                r=0.15 * scale,
                start=(0, 180),
                stroke='brown',
                stroke_width=2,
                fill='none'
            ))

        # Рисуем окна
        for window in windows:
            x, y = transform(Point2D(window['x'], window['y']))
            width_win = window.get('width', 1.0) * scale
            dwg.add(dwg.line(
                start=(x, y), end=(x + width_win, y),
                stroke='blue',
                stroke_width=2,
                stroke_dasharray='5,5'
            ))

        # Рисуем размерные линии
        for dim in dimensions:
            ts = transform(dim.start)
            te = transform(dim.end)
            dwg.add(dwg.line(
                start=ts, end=te,
                stroke='black',
                stroke_width=1,
                stroke_linecap='butt',
                marker_mid=dwg.marker(insert=(6, 2), size=(10, 10))
            ))
            # Размерный текст
            mid_x = (ts[0] + te[0]) / 2
            mid_y = (ts[1] + te[1]) / 2 - 15
            dwg.add(dwg.text(
                dim.text or f"{((dim.end.x-dim.start.x)**2 + (dim.end.y-dim.start.y)**2)**0.5:.2f}м",
                insert=(mid_x, mid_y),
                font_size='12px',
                fill='black',
                text_anchor='middle'
            ))

        # Рисуем сетевые элементы
        if layers:
            for layer_name, elements in layers.items():
                color = {
                    'parking': 'yellow',
                    'service_zones': 'lightgreen',
                    'evacuation': 'orange'
                }.get(layer_name, 'gray')
                for elem in elements:
                    if elem.get('type') == 'rectangle':
                        x, y = transform(Point2D(elem['x'], elem['y']))
                        rect_width = elem['width'] * scale
                        rect_height = elem['height'] * scale
                        dwg.add(dwg.rect(
                            insert=(x, y),
                            size=(rect_width, rect_height),
                            fill=color,
                            fill_opacity=0.3,
                            stroke=color,
                            stroke_width=1
                        ))

        dwg.save()
        return os.path.join(self.output_dir, f"{filename}.svg")

    def export_pdf(self, filename: str,
                   walls: List[Tuple[Point2D, Point2D, Dict]],
                   doors: List[Dict],
                   windows: List[Dict],
                   dimensions: List[Dimension],
                   metadata: Metadata,
                   layers: Dict[str, List]) -> str:
        """Экспорт в PDF формате.

        Args:
            filename: Имя выходного файла
            walls, doors, windows, dimensions, metadata, layers: см. export_dxf

        Returns:
            Путь к сохраненному файлу
        """
        filepath = os.path.join(self.output_dir, f"{filename}.pdf")
        c = canvas.Canvas(filepath, pagesize=A4)
        page_width, page_height = A4

        # Поля
        margin = 20 * mm
        draw_width = page_width - 2 * margin
        draw_height = page_height - 2 * margin - 30 * mm  # Оставляем место для текста

        # Масштаб
        scale = min(draw_width / max(metadata.width, 0.1),
                   draw_height / max(metadata.depth, 0.1))

        def transform(p: Point2D):
            return (
                margin + p.x * scale,
                margin + draw_height - p.y * scale  # Инвертируем Y для PDF
            )

        # Заголовок
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(page_width/2, page_height - 20*mm,
                           f"Архитектурный план: {metadata.building_type}")

        c.setFont("Helvetica", 10)
        info_text = f"ID: {metadata.building_id} | Площадь: {metadata.total_area:.1f}м² | " \
                   f"Размеры: {metadata.width:.1f}×{metadata.depth:.1f}м"
        c.drawCentredString(page_width/2, page_height - 30*mm, info_text)

        # Рисуем стены
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(2)
        for wall_start, wall_end, attrs in walls:
            ts = transform(wall_start)
            te = transform(wall_end)
            c.line(ts[0], ts[1], te[0], te[1])

        # Двери
        c.setStrokeColorRGB(0.6, 0.4, 0.2)
        c.setLineWidth(1)
        for door in doors:
            x, y = transform(Point2D(door['x'], door['y']))
            width_door = door.get('width', 0.9) * scale
            c.arc(x, y - 0.15*scale, x + width_door, y + 0.15*scale, 0, 180)

        # Окна
        c.setStrokeColorRGB(0, 0, 1)
        c.setLineWidth(1)
        for window in windows:
            x, y = transform(Point2D(window['x'], window['y']))
            width_win = window.get('width', 1.0) * scale
            c.line(x, y, x + width_win, y)

        # Размеры
        c.setStrokeColorRGB(0.5, 0.5, 0.5)
        c.setLineWidth(0.5)
        c.setFont("Helvetica", 8)
        for dim in dimensions:
            ts = transform(dim.start)
            te = transform(dim.end)
            c.line(ts[0], ts[1], te[0], te[1])
            mid_x = (ts[0] + te[0]) / 2
            mid_y = (ts[1] + te[1]) / 2 - 10
            c.drawCentredString(mid_x, mid_y,
                               dim.text or f"{((dim.end.x-dim.start.x)**2 + (dim.end.y-dim.start.y)**2)**0.5:.2f}м")

        # Легенда
        c.setFont("Helvetica-Bold", 10)
        legend_y = margin + draw_height + 10 * mm
        c.drawString(margin, legend_y, "Легенда:")
        c.setFont("Helvetica", 8)
        c.setStrokeColorRGB(0, 0, 0)
        c.line(margin + 40, legend_y + 3, margin + 60, legend_y + 3)
        c.drawString(margin + 65, legend_y, "Несущие стены")

        c.setStrokeColorRGB(0.6, 0.4, 0.2)
        c.arc(margin + 50, legend_y + 13, margin + 60, legend_y + 23, 0, 180)