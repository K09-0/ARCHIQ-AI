:"""Scale calibration for site plans."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from shapely.geometry import LineString, Point

from ..models.site_plan import CalibrationData


class ScaleCalibrator:
    """Calibrate scale for site plan measurements."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize calibrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def calibrate(self, reference_points: List[Tuple[float, float]],
                 reference_length: float,
                 image_shape: Tuple[int, int]) -> CalibrationData:
        """Calibrate scale using reference measurement.
        
        Args:
            reference_points: Two points defining reference line
            reference_length: Known length in meters
            image_shape: Image dimensions (height, width)
            
        Returns:
            CalibrationData with scale factor
        """
        if len(reference_points) != 2:
            raise ValueError("Exactly 2 reference points required")
        
        # Calculate pixel distance between reference points
        pixel_distance = self._calculate_distance(
            reference_points[0], reference_points[1]
        )
        
        # Calculate scale factor (pixels per meter)
        if reference_length > 0:
            scale_factor = pixel_distance / reference_length
        else:
            raise ValueError("Reference length must be positive")
        
        # Calculate orientation
        orientation = self._calculate_orientation(
            reference_points[0], reference_points[1]
        )
        
        # Detect north direction (simplified - assumes North is up or marked)
        north_direction = self._estimate_north_direction(image_shape)
        
        return CalibrationData(
            scale_factor=scale_factor,
            reference_length=reference_length,
            reference_points=reference_points,
            calibrated_at=datetime.now(),
            orientation=orientation,
            north_direction=north_direction
        )
    
    def calibrate_from_scale_bar(self, image: np.ndarray,
                                scale_bar_length_m: float) -> CalibrationData:
        """Calibrate using detected scale bar in image.
        
        Args:
            image: Input image
            scale_bar_length_m: Known length of scale bar in meters
            
        Returns:
            CalibrationData
        """
        # Detect scale bar lines
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect lines
        lines = cv2.HoughLinesP(
            gray, 1, np.pi/180, threshold=100,
            minLineLength=50, maxLineGap=5
        )
        
        if lines is None:
            raise ValueError("No lines detected")
        
        # Find longest horizontal or vertical line (likely scale bar)
        longest_line = None
        max_length = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            
            if length > max_length:
                max_length = length
                longest_line = line[0]
        
        if longest_line is None:
            raise ValueError("Could not detect scale bar")
        
        x1, y1, x2, y2 = longest_line
        reference_points = [(x1, y1), (x2, y2)]
        
        return self.calibrate(reference_points, scale_bar_length_m, image.shape[:2])
    
    def calibrate_from_known_dimension(self, image: np.ndarray,
                                      dimension_text: str,
                                      dimension_length_m: float,
                                      text_position: Tuple[int, int]) -> CalibrationData:
        """Calibrate using a dimension line found near text.
        
        Args:
            image: Input image
            dimension_text: OCR-detected dimension text
            dimension_length_m: Known dimension in meters
            text_position: Approximate position of text
            
        Returns:
            CalibrationData
        """
        # Find dimension line near text
        x, y = text_position
        search_region = 50
        
        # Extract region around text
        h, w = image.shape[:2]
        x1 = max(0, x - search_region)
        y1 = max(0, y - search_region)
        x2 = min(w, x + search_region)
        y2 = min(h, y + search_region)
        
        region = image[y1:y2, x1:x2]
        
        # Detect lines in region
        if len(region.shape) == 3:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = region
        
        lines = cv2.HoughLinesP(
            gray_region, 1, np.pi/180, threshold=50,
            minLineLength=20, maxLineGap=5
        )
        
        if lines is not None:
            # Find longest line (likely dimension line)
            longest_line = None
            max_length = 0
            
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                length = ((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2) ** 0.5
                
                if length > max_length:
                    max_length = length
                    longest_line = line[0]
            
            if longest_line is not None:
                lx1, ly1, lx2, ly2 = longest_line
                # Convert to image coordinates
                reference_points = [
                    (lx1 + x1, ly1 + y1),
                    (lx2 + x1, ly2 + y1)
                ]
                
                return self.calibrate(reference_points, dimension_length_m, image.shape[:2])
        
        raise ValueError("Could not find dimension line")
    
    def auto_detect_scale(self, image: np.ndarray) -> Optional[CalibrationData]:
        """Automatically detect scale from image features.
        
        Args:
            image: Input image
            
        Returns:
            CalibrationData or None
        """
        # Try to detect scale bar
        try:
            # Look for typical scale bar lengths (1m, 5m, 10m, 50m, 100m)
            possible_lengths = [1, 5, 10, 50, 100, 200, 500]
            
            for length in possible_lengths:
                try:
                    return self.calibrate_from_scale_bar(image, length)
                except ValueError:
                    continue
        except Exception:
            pass
        
        return None
    
    def _calculate_distance(self, p1: Tuple[float, float], 
                           p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    
    def _calculate_orientation(self, p1: Tuple[float, float],
                              p2: Tuple[float, float]) -> float:
        """Calculate orientation angle in degrees.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Orientation angle in degrees (-180 to 180)
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def _estimate_north_direction(self, image_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Estimate north direction (simplified - assumes up is north).
        
        Args:
            image_shape: Image dimensions (height, width)
            
        Returns:
            North direction vector or None
        """
        # Simplified: assume up is north
        # In production, would detect north arrow marker
        return (0, -1)
    
    def apply_calibration(self, points: List[Tuple[float, float]],
                         scale_factor: float) -> List[Tuple[float, float]]:
        """Apply scale calibration to points.
        
        Args:
            points: List of (x, y) coordinates in pixels
            scale_factor: Scale factor (pixels per meter)
            
        Returns:
            List of calibrated coordinates in meters
        """
        return [(x / scale_factor, y / scale_factor) for x, y in points]
    
    def reverse_calibration(self, points: List[Tuple[float, float]],
                           scale_factor: float) -> List[Tuple[float, float]]:
        """Convert from meters back to pixels.
        
        Args:
            points: List of (x, y) coordinates in meters
            scale_factor: Scale factor (pixels per meter)
            
        Returns:
            List of pixel coordinates
        """
        return [(x * scale_factor, y * scale_factor) for x, y in points]}