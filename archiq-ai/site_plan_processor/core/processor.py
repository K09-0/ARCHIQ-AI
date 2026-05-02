"""Main site plan processor."""

import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from datetime import datetime

from ..models.site_plan import (
    SitePlan, SitePlanData, GeometryResult, CalibrationData, 
    ContourData, SiteObject, ObjectType
)
from ..models.objects import SegmentationResult, OCRResult, SlopeAnalysis
from .geometry import GeometryExtractor
from .segmentation import ObjectSegmenter
from .ocr import OCRProcessor
from .calibration import ScaleCalibrator
from .vectorizer import Vectorizer


logger = logging.getLogger(__name__)


class SitePlanProcessor:
    """Main processor for site plan analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.geometry_extractor = GeometryExtractor(self.config)
        self.segmenter = ObjectSegmenter(self.config)
        self.ocr_processor = OCRProcessor(self.config)
        self.calibrator = ScaleCalibrator(self.config)
        self.vectorizer = Vectorizer(self.config)
        self.current_plan: Optional[SitePlanData] = None
        self.current_image: Optional[np.ndarray] = None
        self.calibration: Optional[CalibrationData] = None
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'min_contour_area': 100,
            'max_contour_area': 1000000,
            'canny_low': 50,
            'canny_high': 150,
            'blur_kernel': 5,
            'scale_detection_method': 'template',
            'segmentation_method': 'watershed',
            'ocr_lang': 'eng',
            'output_format': 'dxf',
            'dpi': 300,
        }
    
    def process_image(self, image: np.ndarray, 
                     site_id: str,
                     calibration_points: Optional[List] = None,
                     known_scale: Optional[float] = None) -> SitePlan:
        """Process a site plan image.
        
        Args:
            image: Input image as numpy array
            site_id: Unique site identifier
            calibration_points: Optional reference points for scale
            known_scale: Known scale factor if available
            
        Returns:
            Complete site plan analysis
        """
        self.current_image = image
        
        # Preprocess image
        processed = self._preprocess(image)
        
        # Extract geometry
        geometry = self.geometry_extractor.extract(
            processed, 
            calibration=self.calibration
        )
        
        # Calibrate scale if reference provided
        if calibration_points and known_scale:
            self.calibration = self.calibrator.calibrate(
                calibration_points, 
                known_scale,
                image.shape
            )
        elif known_scale:
            # Use default calibration
            h, w = image.shape[:2]
            self.calibration = CalibrationData(
                scale_factor=known_scale,
                reference_length=1.0,
                reference_points=[(0, 0), (w, 0)],
                calibrated_at=datetime.now(),
                orientation=0.0
            )
        
        # Extract objects
        segmentation = self.segmenter.segment(processed)
        
        # Extract text
        ocr_result = self.ocr_processor.extract(image)
        
        # Apply scale to geometries
        if self.calibration:
            geometry = self._apply_scale(geometry, self.calibration.scale_factor)
            segmentation = self._apply_scale_to_objects(
                segmentation, 
                self.calibration.scale_factor
            )
        
        # Calculate slopes
        slopes = self._analyze_slopes(image, geometry)
        
        # Calculate areas and utilization
        total_area = geometry.site_boundary.area
        building_area = sum(
            obj.contour.area 
            for obj in segmentation.objects 
            if obj.object_type == ObjectType.BUILDING
        )
        utilization = building_area / total_area if total_area > 0 else 0
        
        # Generate GeoJSON
        geojson = self._generate_geojson(geometry, segmentation)
        
        # Compile results
        plan_data = SitePlanData(
            site_id=site_id,
            geometry=geometry,
            calibration=self.calibration,
            area_sqm=total_area * (self.calibration.scale_factor ** 2) if self.calibration else total_area,
            perimeter_m=geometry.site_boundary.perimeter * (self.calibration.scale_factor if self.calibration else 1),
            utilization_ratio=utilization,
            slopes=slopes,
            elevation_data=None,
            geojson=geojson,
            metadata={
                'source_format': 'image',
                'processing_date': datetime.now().isoformat(),
                'ocr_text': ocr_result.text,
                'num_objects': len(segmentation.objects)
            }
        )
        
        self.current_plan = plan_data
        
        return SitePlan(
            filename=f"site_{site_id}",
            file_format="image",
            file_size=image.nbytes,
            data=plan_data,
            status="completed",
            error=None
        )
    
    def process_pdf(self, pdf_path: Path, 
                   site_id: str,
                   page: int = 0) -> SitePlan:
        """Process a PDF site plan.
        
        Args:
            pdf_path: Path to PDF file
            site_id: Unique site identifier
            page: Page number to process (0-based)
            
        Returns:
            Complete site plan analysis
        """
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        page_obj = doc[page]
        pix = page_obj.get_pixmap(dpi=self.config['dpi'])
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        doc.close()
        
        return self.process_image(image, site_id)
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Adaptive threshold for line detection
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return binary
    
    def _apply_scale(self, geometry: GeometryResult, 
                    scale_factor: float) -> GeometryResult:
        """Apply scale factor to geometry."""
        # Scale contour points
        scaled_points = [
            (x * scale_factor, y * scale_factor)
            for x, y in geometry.site_boundary.points
        ]
        
        scaled_boundary = ContourData(
            points=scaled_points,
            area=geometry.site_boundary.area * (scale_factor ** 2),
            perimeter=geometry.site_boundary.perimeter * scale_factor,
            centroid=(
                geometry.site_boundary.centroid[0] * scale_factor,
                geometry.site_boundary.centroid[1] * scale_factor
            ),
            bbox=(
                geometry.site_boundary.bbox[0] * scale_factor,
                geometry.site_boundary.bbox[1] * scale_factor,
                geometry.site_boundary.bbox[2] * scale_factor,
                geometry.site_boundary.bbox[3] * scale_factor
            )
        )
        
        # Scale side lengths
        scaled_sides = [s * scale_factor for s in geometry.side_lengths]
        
        return GeometryResult(
            site_boundary=scaled_boundary,
            objects=geometry.objects,
            corner_points=scaled_boundary.points,
            side_lengths=scaled_sides,
            angles=geometry.angles
        )
    
    def _apply_scale_to_objects(self, segmentation: SegmentationResult,
                               scale_factor: float) -> SegmentationResult:
        """Apply scale to segmented objects."""
        scaled_objects = []
        for obj in segmentation.objects:
            scaled_points = [
                (x * scale_factor, y * scale_factor)
                for x, y in obj.contour.points
            ]
            scaled_contour = ContourData(
                points=scaled_points,
                area=obj.contour.area * (scale_factor ** 2),
                perimeter=obj.contour.perimeter * scale_factor,
                centroid=(
                    obj.contour.centroid[0] * scale_factor,
                    obj.contour.centroid[1] * scale_factor
                ),
                bbox=(
                    obj.contour.bbox[0] * scale_factor,
                    obj.contour.bbox[1] * scale_factor,
                    obj.contour.bbox[2] * scale_factor,
                    obj.contour.bbox[3] * scale_factor
                )
            )
            scaled_objects.append(SiteObject(
                object_id=obj.object_id,
                object_type=obj.object_type,
                contour=scaled_contour,
                confidence=obj.confidence,
                label=obj.label,
                metadata=obj.metadata
            ))
        
        return SegmentationResult(
            objects=scaled_objects,
            mask=segmentation.mask,
            image_shape=segmentation.image_shape,
            processing_time=segmentation.processing_time
        )
    
    def _analyze_slopes(self, image: np.ndarray, 
                       geometry: GeometryResult) -> List[Dict[str, Any]]:
        """Analyze terrain slopes."""
        # Simple slope analysis based on elevation changes
        # In production, would use DEM/raster elevation data
        slopes = []
        return slopes
    
    def _generate_geojson(self, geometry: GeometryResult,
                         segmentation: SegmentationResult) -> Dict[str, Any]:
        """Generate GeoJSON representation."""
        features = []
        
        # Site boundary
        boundary_feature = {
            "type": "Feature",
            "properties": {
                "type": "site_boundary",
                "area": geometry.site_boundary.area,
                "perimeter": geometry.site_boundary.perimeter
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [geometry.site_boundary.points + 
                               [geometry.site_boundary.points[0]]]
            }
        }
        features.append(boundary_feature)
        
        # Objects
        for obj in segmentation.objects:
            feature = {
                "type": "Feature",
                "properties": {
                    "type": obj.object_type.value,
                    "area": obj.contour.area,
                    "confidence": obj.confidence,
                    "label": obj.label or ""
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [obj.contour.points + 
                                   [obj.contour.points[0]]]
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def export_dxf(self, output_path: Path) -> bool:
        """Export current plan to DXF format.
        
        Args:
            output_path: Path for DXF output file
            
        Returns:
            True if successful
        """
        if not self.current_plan:
            return False
        
        return self.vectorizer.to_dxf(
            self.current_plan,
            output_path
        )
    
    def export_svg(self, output_path: Path) -> bool:
        """Export current plan to SVG format.
        
        Args:
            output_path: Path for SVG output file
            
        Returns:
            True if successful
        """
        if not self.current_plan:
            return False
        
        return self.vectorizer.to_svg(
            self.current_plan,
            output_path
        )
