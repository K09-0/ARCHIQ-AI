"""Data models for site plan processing."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum


class ObjectType(str, Enum):
    BUILDING = "building"
    TREE = "tree"
    ROAD = "road"
    PATH = "path"
    WATER = "water"
    FENCE = "fence"
    UNKNOWN = "unknown"


class ContourData(BaseModel):
    """Contour/shape data for a site object."""
    points: List[Tuple[float, float]] = Field(..., description="List of (x, y) coordinates")
    area: float = Field(..., description="Area in square units")
    perimeter: float = Field(..., description="Perimeter in units")
    centroid: Tuple[float, float] = Field(..., description="Center point (x, y)")
    bbox: Tuple[float, float, float, float] = Field(..., description="Bounding box (xmin, ymin, xmax, ymax)")


class SiteObject(BaseModel):
    """Represents an object detected on the site plan."""
    object_id: str = Field(..., description="Unique identifier")
    object_type: ObjectType = Field(..., description="Type of object")
    contour: ContourData = Field(..., description="Geometric contour data")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence score")
    label: Optional[str] = Field(None, description="OCR-detected label")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GeometryResult(BaseModel):
    """Result of geometry extraction."""
    site_boundary: ContourData = Field(..., description="Site boundary contour")
    objects: List[SiteObject] = Field(default_factory=list, description="Detected objects")
    corner_points: List[Tuple[float, float]] = Field(..., description="Corner coordinates")
    side_lengths: List[float] = Field(..., description="Length of each side")
    angles: List[float] = Field(..., description="Angles at corners in degrees")


class CalibrationData(BaseModel):
    """Scale calibration data."""
    scale_factor: float = Field(..., description="Pixels per meter")
    reference_length: float = Field(..., description="Known reference length in meters")
    reference_points: List[Tuple[float, float]] = Field(..., description="Reference line endpoints")
    calibrated_at: datetime = Field(default_factory=datetime.now)
    orientation: float = Field(default=0.0, description="Rotation angle in degrees")
    north_direction: Optional[Tuple[float, float]] = Field(None, description="North direction vector")


class SitePlanData(BaseModel):
    """Complete processed site plan data."""
    site_id: str = Field(..., description="Unique site identifier")
    geometry: GeometryResult = Field(..., description="Geometric analysis results")
    calibration: Optional[CalibrationData] = Field(None, description="Scale calibration")
    area_sqm: float = Field(..., description="Total area in square meters")
    perimeter_m: float = Field(..., description="Perimeter in meters")
    utilization_ratio: float = Field(..., description="Building coverage ratio")
    slopes: List[Dict[str, Any]] = Field(default_factory=list, description="Slope/grade data")
    elevation_data: Optional[Dict[str, Any]] = Field(None, description="Elevation information")
    geojson: Dict[str, Any] = Field(..., description="GeoJSON representation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processed_at: datetime = Field(default_factory=datetime.now)


class SitePlan(BaseModel):
    """Site plan with file information."""
    filename: str = Field(..., description="Original filename")
    file_format: str = Field(..., description="Source file format")
    file_size: int = Field(..., description="File size in bytes")
    data: SitePlanData = Field(..., description="Processed site plan data")
    status: str = Field(..., description="Processing status")
    error: Optional[str] = Field(None, description="Error message if failed")
