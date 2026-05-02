"""Additional object models."""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any
from .site_plan import ObjectType, ContourData


class SegmentationResult(BaseModel):
    """Result of instance segmentation."""
    objects: List[SiteObject] = Field(..., description="Segmented objects")
    mask: Optional[List[List[float]]] = Field(None, description="Segmentation mask as 2D array")
    image_shape: Tuple[int, int] = Field(..., description="Original image dimensions")
    processing_time: float = Field(..., description="Processing time in seconds")


class OCRResult(BaseModel):
    """OCR extraction result."""
    text: str = Field(..., description="Full extracted text")
    dimensions: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted dimensions")
    labels: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted labels with positions")
    confidence: float = Field(..., description="Overall OCR confidence")


class SlopeAnalysis(BaseModel):
    """Slope/grade analysis result."""
    max_slope: float = Field(..., description="Maximum slope in degrees")
    avg_slope: float = Field(..., description="Average slope in degrees")
    slope_zones: List[Dict[str, Any]] = Field(..., description="Areas with different slope ranges")
    drainage_direction: Optional[Tuple[float, float]] = Field(None, description="Primary drainage direction vector")
