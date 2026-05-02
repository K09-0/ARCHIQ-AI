"""Geometry extraction from site plans."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from skimage import measure
from scipy.spatial import ConvexHull

from ..models.site_plan import ContourData, GeometryResult


class GeometryExtractor:
    """Extract geometric features from site plans."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize geometry extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_area = config.get('min_contour_area', 100)
        self.canny_low = config.get('canny_low', 50)
        self.canny_high = config.get('canny_high', 150)
    
    def extract(self, image: np.ndarray,
               calibration: Optional[Any] = None) -> GeometryResult:
        """Extract geometry from image.
        
        Args:
            image: Preprocessed binary image
            calibration: Optional calibration data
            
        Returns:
            GeometryResult with extracted features
        """
        # Find contours
        contours, hierarchy = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            # Try with different retrieval mode
            contours, _ = cv2.findContours(
                image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
        
        # Filter contours by area
        valid_contours = [
            c for c in contours 
            if cv2.contourArea(c) >= self.min_area
        ]
        
        if not valid_contours:
            raise ValueError("No valid contours found")
        
        # Find main boundary (largest contour)
        main_contour = max(valid_contours, key=cv2.contourArea)
        
        # Simplify contour (Douglas-Peucker algorithm)
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Extract points
        points = self._contour_to_points(approx)
        
        # Calculate convex hull for reference
        hull_points = self._get_convex_hull(main_contour)
        
        # Detect corners
        corners = self._detect_corners(points)
        
        # Calculate side lengths and angles
        side_lengths, angles = self._calculate_polygon_metrics(corners)
        
        # Create contour data
        contour_data = self._create_contour_data(points)
        
        # Create geometry result
        geometry = GeometryResult(
            site_boundary=contour_data,
            objects=[],
            corner_points=corners,
            side_lengths=side_lengths,
            angles=angles
        )
        
        return geometry
    
    def _contour_to_points(self, contour: np.ndarray) -> List[Tuple[float, float]]:
        """Convert OpenCV contour to list of points."""
        return [(float(p[0][0]), float(p[0][1])) for p in contour]
    
    def _get_convex_hull(self, contour: np.ndarray) -> List[Tuple[float, float]]:
        """Calculate convex hull of contour."""
        hull = cv2.convexHull(contour)
        return self._contour_to_points(hull)
    
    def _detect_corners(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Detect corners using angle thresholding.
        
        Args:
            points: List of contour points
            
        Returns:
            List of corner coordinates
        """
        if len(points) < 3:
            return points
        
        corners = []
        min_angle = 45  # Minimum angle to be considered a corner (degrees)
        
        for i in range(len(points)):
            prev_idx = (i - 1) % len(points)
            next_idx = (i + 1) % len(points)
            
            # Calculate vectors
            v1 = (points[prev_idx][0] - points[i][0],
                  points[prev_idx][1] - points[i][1])
            v2 = (points[next_idx][0] - points[i][0],
                  points[next_idx][1] - points[i][1])
            
            # Calculate angle
            angle = self._angle_between_vectors(v1, v2)
            
            if angle <= min_angle:
                corners.append(points[i])
        
        return corners if corners else [points[0], points[len(points)//2]]
    
    def _angle_between_vectors(self, v1: Tuple[float, float], 
                              v2: Tuple[float, float]) -> float:
        """Calculate angle between two vectors in degrees."""
        import math
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        det = v1[0] * v2[1] - v1[1] * v2[0]
        angle = math.degrees(math.atan2(det, dot))
        return abs(angle)
    
    def _calculate_polygon_metrics(self, 
                                 corners: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
        """Calculate side lengths and angles for polygon.
        
        Args:
            corners: List of corner coordinates
            
        Returns:
            Tuple of (side_lengths, angles)
        """
        if len(corners) < 2:
            return [0], [0]
        
        side_lengths = []
        angles = []
        
        for i in range(len(corners)):
            next_idx = (i + 1) % len(corners)
            
            # Calculate side length
            dx = corners[next_idx][0] - corners[i][0]
            dy = corners[next_idx][1] - corners[i][1]
            length = (dx ** 2 + dy ** 2) ** 0.5
            side_lengths.append(length)
            
            # Calculate angle at corner
            if len(corners) >= 3:
                prev_idx = (i - 1) % len(corners)
                v1 = (corners[prev_idx][0] - corners[i][0],
                      corners[prev_idx][1] - corners[i][1])
                v2 = (corners[next_idx][0] - corners[i][0],
                      corners[next_idx][1] - corners[i][1])
                angle = self._angle_between_vectors(v1, v2)
                angles.append(angle)
        
        return side_lengths, angles
    
    def _create_contour_data(self, points: List[Tuple[float, float]]) -> ContourData:
        """Create ContourData from points.
        
        Args:
            points: List of (x, y) coordinates
            
        Returns:
            ContourData object
        """
        if len(points) < 3:
            raise ValueError("At least 3 points required for contour")
        
        polygon = Polygon(points)
        
        # Calculate bounding box
        x_coords, y_coords = zip(*points)
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        # Calculate centroid
        centroid = (polygon.centroid.x, polygon.centroid.y)
        
        return ContourData(
            points=points,
            area=polygon.area,
            perimeter=polygon.length,
            centroid=centroid,
            bbox=bbox
        )
    
    def detect_scale_bar(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect scale bar in image using template matching.
        
        Args:
            image: Input image
            
        Returns:
            Scale bar information or None
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect lines
        lines = cv2.HoughLinesP(
            gray, 1, np.pi/180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        
        if lines is None:
            return None
        
        # Find longest horizontal/vertical lines
        longest_horizontal = None
        longest_vertical = None
        max_h_length = 0
        max_v_length = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 15 or angle > 165:  # Horizontal
                if length > max_h_length:
                    max_h_length = length
                    longest_horizontal = line[0]
            elif 75 < angle < 105:  # Vertical
                if length > max_v_length:
                    max_v_length = length
                    longest_vertical = line[0]
        
        if longest_horizontal is not None:
            return {
                'type': 'horizontal',
                'length_px': max_h_length,
                'coordinates': longest_horizontal
            }
        elif longest_vertical is not None:
            return {
                'type': 'vertical',
                'length_px': max_v_length,
                'coordinates': longest_vertical
            }
        
        return None
    
    def compute_hu_moments(self, contour: np.ndarray) -> np.ndarray:
        """Compute Hu moments for shape description.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            Hu moments (7 values)
        """
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Log transform for scale invariance
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments