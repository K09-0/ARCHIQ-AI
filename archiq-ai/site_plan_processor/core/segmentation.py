"""Object segmentation for site plans."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

from ..models.site_plan import SiteObject, ObjectType, ContourData
from ..models.objects import SegmentationResult


class ObjectSegmenter:
    """Segment objects in site plan images."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize segmenter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_area = config.get('min_contour_area', 100)
        self.max_area = config.get('max_contour_area', 1000000)
    
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Perform instance segmentation on image.
        
        Args:
            image: Input binary image
            
        Returns:
            Segmentation results
        """
        import time
        start_time = time.time()
        
        # Store original shape
        image_shape = image.shape[:2]
        
        # Watershed segmentation
        markers = self._watershed_segmentation(image)
        
        # Extract objects from markers
        objects = self._extract_objects(markers, image)
        
        # Classify objects
        objects = self._classify_objects(objects, image)
        
        elapsed = time.time() - start_time
        
        return SegmentationResult(
            objects=objects,
            mask=None,
            image_shape=image_shape,
            processing_time=elapsed
        )
    
    def _watershed_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Perform watershed segmentation.
        
        Args:
            image: Binary image
            
        Returns:
            Label image with markers
        """
        # Distance transform
        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        
        # Find sure foreground area
        _, sure_fg = cv2.threshold(
            dist_transform, 0.3 * dist_transform.max(), 255, 0
        )
        sure_fg = np.uint8(sure_fg)
        
        # Find sure background area
        sure_bg = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=3)
        
        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        # Note: watershed expects 3-channel image
        if len(image.shape) == 2:
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            color_image = image.copy()
        
        markers = cv2.watershed(color_image, markers)
        
        return markers
    
    def _extract_objects(self, markers: np.ndarray,
                        image: np.ndarray) -> List[SiteObject]:
        """Extract objects from watershed markers.
        
        Args:
            markers: Watershed label image
            image: Original image
            
        Returns:
            List of detected objects
        """
        objects = []
        
        # Get unique labels (excluding background 0 and watershed boundaries -1)
        labels = np.unique(markers)
        labels = labels[(labels > 1) & (labels != -1)]
        
        for label in labels:
            # Create mask for this label
            mask = (markers == label).astype(np.uint8) * 255
            
            # Find contours in mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Approximate contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calculate properties
            points = [(float(p[0][0]), float(p[0][1])) for p in approx]
            
            if len(points) < 3:
                continue
            
            # Create polygon
            from shapely.geometry import Polygon
            polygon = Polygon(points)
            
            # Calculate bounding box
            x_coords, y_coords = zip(*points)
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
            # Calculate centroid
            centroid = (polygon.centroid.x, polygon.centroid.y)
            
            contour_data = ContourData(
                points=points,
                area=polygon.area,
                perimeter=polygon.length,
                centroid=centroid,
                bbox=bbox
            )
            
            obj = SiteObject(
                object_id=f"obj_{len(objects):04d}",
                object_type=ObjectType.UNKNOWN,
                contour=contour_data,
                confidence=0.5,
                label=None,
                metadata={}
            )
            objects.append(obj)
        
        return objects
    
    def _classify_objects(self, objects: List[SiteObject],
                         image: np.ndarray) -> List[SiteObject]:
        """Classify objects based on shape and texture features.
        
        Args:
            objects: List of objects to classify
            image: Original image
            
        Returns:
            List of classified objects
        """
        for obj in objects:
            obj_type, confidence = self._classify_shape(obj.contour)
            obj.object_type = obj_type
            obj.confidence = confidence
        
        return objects
    
    def _classify_shape(self, contour: ContourData) -> Tuple[ObjectType, float]:
        """Classify object based on shape features.
        
        Args:
            contour: Contour data
            
        Returns:
            Tuple of (object_type, confidence)
        """
        area = contour.area
        perimeter = contour.perimeter
        
        # Calculate compactness (circularity)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Calculate aspect ratio
        bbox = contour.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if height > 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = 1
        
        # Number of vertices (approximation from points)
        num_vertices = len(contour.points)
        
        # Classification rules (simplified)
        if circularity > 0.7 and 0.8 < aspect_ratio < 1.2:
            # Circle-like shape - could be tree
            return ObjectType.TREE, 0.7
        elif aspect_ratio > 3 or aspect_ratio < 0.33:
            # Elongated shape - could be road
            return ObjectType.ROAD, 0.6
        elif num_vertices >= 4 and num_vertices <= 6:
            # Polygon - could be building
            if area > 1000:
                return ObjectType.BUILDING, 0.8
        elif aspect_ratio > 1.5:
            return ObjectType.PATH, 0.6
        
        return ObjectType.UNKNOWN, 0.5
    
    def segment_by_color(self, image: np.ndarray, 
                        color_ranges: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """Segment image by color ranges (HSV).
        
        Args:
            image: Input BGR image
            color_ranges: List of (lower, upper) HSV bounds
            
        Returns:
            List of binary masks
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = []
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
        
        return masks
    
    def detect_buildings_template(self, image: np.ndarray,
                                 building_height_range: Tuple[int, int] = (20, 100)) -> List[Dict[str, Any]]:
        """Detect buildings using geometric properties.
        
        Args:
            image: Input image
            building_height_range: Expected building height range in pixels
            
        Returns:
            List of building detections
        """
        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        buildings = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 500:
                continue
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Building-like shapes have 4-6 vertices
            if 4 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(contour)
                
                if building_height_range[0] <= h <= building_height_range[1]:
                    rect = cv2.minAreaRect(contour)
                    angle = rect[2]
                    
                    buildings.append({
                        'contour': [(float(p[0][0]), float(p[0][1])) for p in approx],
                        'area': area,
                        'bbox': (x, y, x + w, y + h),
                        'angle': angle,
                        'confidence': 0.7
                    })
        
        return buildings