:"""Vector export functionality (DXF, SVG)."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import xml.etree.ElementTree as ET

from ..models.site_plan import SitePlanData, GeometryResult, SiteObject, ObjectType
from ..models.objects import SlopeAnalysis


try:
    import ezdxf
    DXF_AVAILABLE = True
except ImportError:
    DXF_AVAILABLE = False


class Vectorizer:
    """Export site plans to vector formats."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize vectorizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_format = config.get('output_format', 'dxf')
        self.dpi = config.get('dpi', 300)
    
    def to_dxf(self, site_plan: SitePlanData, output_path: Path) -> bool:
        """Export site plan to DXF format.
        
        Args:
            site_plan: Site plan data
            output_path: Output file path
            
        Returns:
            True if successful
        """
        if not DXF_AVAILABLE:
            print("Warning: ezdxf not available, DXF export disabled")
            return False
        
        try:
            import ezdxf
            
            # Create new DXF document
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            # Add site boundary
            boundary = site_plan.geometry.site_boundary
            if len(boundary.points) >= 3:
                # Close the polygon
                points = boundary.points + [boundary.points[0]]
                msp.add_lwpolyline(points, dxfattribs={'layer': 'boundary'})
            
            # Add objects
            for obj in site_plan.geometry.objects:
                self._add_object_to_dxf(msp, obj)
            
            # Add metadata
            doc.set_variable('$INSUNITS', 1)  # Meters
            
            # Save
            doc.saveas(str(output_path))
            
            return True
        except Exception as e:
            print(f"DXF export failed: {e}")
            return False
    
    def to_svg(self, site_plan: SitePlanData, output_path: Path) -> bool:
        """Export site plan to SVG format.
        
        Args:
            site_plan: Site plan data
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            # Calculate bounds
            bounds = self._calculate_bounds(site_plan)
            
            # Create SVG document
            width = bounds['width']
            height = bounds['height']
            
            svg = ET.Element(
                'svg',
                xmlns="http://www.w3.org/2000/svg",
                version="1.1",
                width=f"{width}",
                height=f"{height}",
                viewBox=f"0 0 {width} {height}"
            )
            
            # Add defs for styles
            defs = ET.SubElement(svg, 'defs')
            
            # Site boundary style
            boundary_style = ET.SubElement(defs, 'style')
            boundary_style.text = """
                .boundary { fill: none; stroke: #000000; stroke-width: 2; }
                .building { fill: #cccccc; stroke: #666666; stroke-width: 1; }
                .tree { fill: #228B22; stroke: #006400; stroke-width: 1; }
                .road { fill: #808080; stroke: #404040; stroke-width: 1; }
                .path { fill: #DEB887; stroke: #8B7355; stroke-width: 1; }
                .water { fill: #4682B4; stroke: #191970; stroke-width: 1; }
                .unknown { fill: #ff0000; stroke: #8b0000; stroke-width: 1; }
            """
            
            # Add site boundary
            boundary = site_plan.geometry.site_boundary
            if len(boundary.points) >= 3:
                points_str = self._points_to_svg_string(boundary.points, bounds)
                ET.SubElement(
                    svg, 'polygon',
                    points=points_str,
                    class_="boundary"
                )
            
            # Add objects
            for obj in site_plan.geometry.objects:
                self._add_object_to_svg(svg, obj, bounds)
            
            # Write file
            tree = ET.ElementTree(svg)
            ET.indent(tree, space='  ')
            tree.write(output_path, encoding='unicode', xml_declaration=True)
            
            return True
        except Exception as e:
            print(f"SVG export failed: {e}")
            return False
    
    def to_geojson(self, site_plan: SitePlanData) -> Dict[str, Any]:
        """Convert to GeoJSON format.
        
        Args:
            site_plan: Site plan data
            
        Returns:
            GeoJSON dictionary
        """
        features = []
        
        # Site boundary
        boundary = site_plan.geometry.site_boundary
        if len(boundary.points) >= 3:
            boundary_feature = {
                "type": "Feature",
                "properties": {
                    "type": "site_boundary",
                    "area": site_plan.area_sqm,
                    "perimeter": site_plan.perimeter_m
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [boundary.points + [boundary.points[0]]]
                }
            }
            features.append(boundary_feature)
        
        # Objects
        for obj in site_plan.geometry.objects:
            if len(obj.contour.points) >= 3:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "type": obj.object_type.value,
                        "area": obj.contour.area,
                        "perimeter": obj.contour.perimeter,
                        "confidence": obj.confidence,
                        "label": obj.label or ""
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [obj.contour.points + [obj.contour.points[0]]]
                    }
                }
                features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "site_id": "site",
                "area_sqm": site_plan.area_sqm,
                "perimeter_m": site_plan.perimeter_m,
                "utilization_ratio": site_plan.utilization_ratio,
                "processed_at": site_plan.processed_at.isoformat(),
                "has_calibration": site_plan.calibration is not None
            }
        }
    
    def _add_object_to_dxf(self, msp, obj: SiteObject):
        """Add object to DXF modelspace.
        
        Args:
            msp: DXF modelspace
            obj: Site object
        """
        if len(obj.contour.points) < 3:
            return
        
        # Determine layer based on object type
        layer_map = {
            ObjectType.BUILDING: 'buildings',
            ObjectType.TREE: 'trees',
            ObjectType.ROAD: 'roads',
            ObjectType.PATH: 'paths',
            ObjectType.WATER: 'water',
            ObjectType.FENCE: 'fences',
            ObjectType.UNKNOWN: 'unknown'
        }
        
        layer = layer_map.get(obj.object_type, 'unknown')
        
        # Add polygon
        points = obj.contour.points + [obj.contour.points[0]]
        msp.add_lwpolyline(points, dxfattribs={'layer': layer})
        
        # Add centroid marker
        cx, cy = obj.contour.centroid
        msp.add_point((cx, cy), dxfattribs={'layer': f'{layer}_centroid'})
        
        # Add label if available
        if obj.label:
            msp.add_text(
                obj.label,
                dxfattribs={
                    'layer': f'{layer}_label',
                    'height': 0.5
                }
            ).set_pos((cx, cy))
    
    def _add_object_to_svg(self, svg, obj: SiteObject, bounds: Dict[str, float]):
        """Add object to SVG element.
        
        Args:
            svg: SVG element
            obj: Site object
            bounds: Bounds dictionary
        """
        if len(obj.contour.points) < 3:
            return
        
        # Determine class based on object type
        class_map = {
            ObjectType.BUILDING: 'building',
            ObjectType.TREE: 'tree',
            ObjectType.ROAD: 'road',
            ObjectType.PATH: 'path',
            ObjectType.WATER: 'water',
            ObjectType.FENCE: 'fence',
            ObjectType.UNKNOWN: 'unknown'
        }
        
        class_name = class_map.get(obj.object_type, 'unknown')
        points_str = self._points_to_svg_string(obj.contour.points, bounds)
        
        ET.SubElement(
            svg, 'polygon',
            points=points_str,
            class_=class_name
        )
    
    def _calculate_bounds(self, site_plan: SitePlanData) -> Dict[str, float]:
        """Calculate bounding box for SVG.
        
        Args:
            site_plan: Site plan data
            
        Returns:
            Dictionary with bounds information
        """
        boundary = site_plan.geometry.site_boundary
        
        if len(boundary.points) == 0:
            return {'min_x': 0, 'min_y': 0, 'width': 1000, 'height': 1000}
        
        x_coords, y_coords = zip(*boundary.points)
        
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Add padding
        padding = 0.1
        width *= (1 + padding)
        height *= (1 + padding)
        
        return {
            'min_x': min_x,
            'min_y': min_y,
            'width': width,
            'height': height
        }
    
    def _points_to_svg_string(self, points: List[Tuple[float, float]],
                             bounds: Dict[str, float]) -> str:
        """Convert points to SVG polygon string.
        
        Args:
            points: List of (x, y) coordinates
            bounds: Bounds dictionary
            
        Returns:
            SVG points string
        """
        min_x = bounds['min_x']
        min_y = bounds['min_y']
        
        svg_points = []
        for x, y in points:
            svg_x = x - min_x
            svg_y = y - min_y
            svg_points.append(f"{svg_x:.2f},{svg_y:.2f}")
        
        return ' '.join(svg_points)