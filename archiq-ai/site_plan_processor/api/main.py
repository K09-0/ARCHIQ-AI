"""FastAPI routes for site plan processing."""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List, Dict, Any
import io
import numpy as np
from pathlib import Path
import tempfile
import json
import fitz  # PyMuPDF
import cv2

from ..core.processor import SitePlanProcessor
from ..models.site_plan import SitePlan, SitePlanData
from ..models.objects import CalibrationData


router = APIRouter()

# Global processor instance
processor = SitePlanProcessor()

# Store processed plans in memory (in production, use database)
processed_plans: Dict[str, SitePlan] = {}


@router.post("/upload-site-plan", response_model=SitePlan)
async def upload_site_plan(
    file: UploadFile = File(...),
    site_id: Optional[str] = None,
    calibration_points: Optional[str] = None,
    known_scale: Optional[float] = None,
    use_pdf_page: Optional[int] = 0
):
    """Upload and process a site plan.
    
    Accepts:
    - Images (PNG, JPG, JPEG, TIFF)
    - PDF files
    
    Args:
        file: Site plan file
        site_id: Unique identifier for the site
        calibration_points: JSON string of calibration points
        known_scale: Known scale in meters (for calibration)
        use_pdf_page: PDF page to process (0-based)
        
    Returns:
        Processed site plan data
    """
    
    # Generate site ID if not provided
    if not site_id:
        import uuid
        site_id = str(uuid.uuid4())[:8]
    
    # Read file
    contents = await file.read()
    
    # Process based on file type
    file_ext = Path(file.filename).suffix.lower()
    
    try:
        if file_ext in ['.pdf']:
            # Process PDF
            with tempfile.NamedTemporaryFile(
                delete=False, suffix='.pdf'
            ) as tmp:
                tmp.write(contents)
                tmp_path = Path(tmp.name)
            
            try:
                plan = processor.process_pdf(
                    tmp_path,
                    site_id=site_id,
                    page=use_pdf_page
                )
            finally:
                tmp_path.unlink()
                
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            # Process image
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail="Could not decode image file"
                )
            
            # Parse calibration points if provided
            cal_points = None
            if calibration_points:
                try:
                    cal_data = json.loads(calibration_points)
                    if isinstance(cal_data, list) and len(cal_data) == 2:
                        cal_points = [
                            tuple(cal_data[0]),
                            tuple(cal_data[1])
                        ]
                except json.JSONDecodeError:
                    pass
            
            plan = processor.process_image(
                image,
                site_id=site_id,
                calibration_points=cal_points,
                known_scale=known_scale
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}"
            )
        
        # Store processed plan
        processed_plans[site_id] = plan
        
        return plan
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@router.get("/site-parameters")
async def get_site_parameters(site_id: str):
    """Get calculated parameters for a processed site.
    
    Args:
        site_id: Site identifier
        
    Returns:
        Site parameters and measurements
    """
    if site_id not in processed_plans:
        raise HTTPException(
            status_code=404,
            detail=f"Site {site_id} not found. Please upload first."
        )
    
    plan = processed_plans[site_id]
    
    if plan.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Processing not completed: {plan.error}"
        )
    
    data = plan.data
    
    parameters = {
        "site_id": data.site_id,
        "area": {
            "square_meters": round(data.area_sqm, 2),
            "hectares": round(data.area_sqm / 10000, 4),
            "acres": round(data.area_sqm / 4046.86, 4)
        },
        "perimeter": {
            "meters": round(data.perimeter_m, 2),
            "kilometers": round(data.perimeter_m / 1000, 4)
        },
        "geometry": {
            "corner_points": data.geometry.corner_points,
            "side_lengths": [
                round(l, 2) for l in data.geometry.side_lengths
            ],
            "angles": [
                round(a, 2) for a in data.geometry.angles
            ]
        },
        "utilization": {
            "building_coverage_ratio": round(data.utilization_ratio, 4),
            "building_coverage_percent": round(
                data.utilization_ratio * 100, 2
            )
        },
        "objects": {
            "total": len(data.geometry.objects),
            "by_type": {}
        },
        "slopes": data.slopes,
        "has_elevation_data": data.elevation_data is not None,
        "calibrated": data.calibration is not None,
        "processed_at": data.processed_at.isoformat()
    }
    
    # Count objects by type
    for obj in data.geometry.objects:
        obj_type = obj.object_type.value
        if obj_type not in parameters["objects"]["by_type"]:
            parameters["objects"]["by_type"][obj_type] = 0
        parameters["objects"]["by_type"][obj_type] += 1
    
    return parameters


@router.post("/extract-geometry")
async def extract_geometry(
    file: UploadFile = File(...),
    site_id: Optional[str] = None,
    extract_objects: bool = True,
    extract_text: bool = True
):
    """Extract geometry and features from site plan.
    
    Args:
        file: Site plan file
        site_id: Site identifier
        extract_objects: Whether to detect objects
        extract_text: Whether to extract text
        
    Returns:
        Extracted geometry and features
    """
    
    # Generate site ID if not provided
    if not site_id:
        import uuid
        site_id = str(uuid.uuid4())[:8]
    
    # Read file
    contents = await file.read()
    file_ext = Path(file.filename).suffix.lower()
    
    try:
        if file_ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported format for geometry extraction"
            )
        
        # Process image
        plan = processor.process_image(image, site_id=site_id)
        
        result = {
            "site_id": site_id,
            "boundary": {
                "points": plan.data.geometry.site_boundary.points,
                "area": plan.data.geometry.site_boundary.area,
                "perimeter": plan.data.geometry.site_boundary.perimeter
            },
            "corners": plan.data.geometry.corner_points,
            "sides": plan.data.geometry.side_lengths,
            "angles": plan.data.geometry.angles
        }
        
        if extract_objects:
            result["objects"] = [
                {
                    "id": obj.object_id,
                    "type": obj.object_type.value,
                    "area": obj.contour.area,
                    "perimeter": obj.contour.perimeter,
                    "centroid": obj.contour.centroid,
                    "bbox": obj.contour.bbox,
                    "points": obj.contour.points,
                    "confidence": obj.confidence
                }
                for obj in plan.data.geometry.objects
            ]
        
        # Store plan
        processed_plans[site_id] = plan
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Geometry extraction failed: {str(e)}"
        )


@router.post("/calibrate-scale")
async def calibrate_scale(
    file: UploadFile = File(...),
    known_length: float = None,
    reference_points: Optional[str] = None,
    auto_detect: bool = False
):
    """Calibrate scale for a site plan.
    
    Args:
        file: Site plan file
        known_length: Known reference length in meters
        reference_points: JSON string of reference line endpoints
        auto_detect: Attempt automatic scale detection
        
    Returns:
        Calibration data
    """
    
    # Read file
    contents = await file.read()
    file_ext = Path(file.filename).suffix.lower()
    
    try:
        if file_ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported format"
            )
        
        calibrator = processor.calibrator
        
        # Determine calibration method
        if auto_detect:
            calibration = calibrator.auto_detect_scale(image)
            if not calibration:
                raise HTTPException(
                    status_code=400,
                    detail="Could not auto-detect scale"
                )
        elif reference_points and known_length:
            ref_data = json.loads(reference_points)
            if len(ref_data) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Need exactly 2 reference points"
                )
            calibration = calibrator.calibrate(
                [tuple(ref_data[0]), tuple(ref_data[1])],
                known_length,
                image.shape[:2]
            )
        elif known_length:
            # Try to detect scale bar
            calibration = calibrator.calibrate_from_scale_bar(
                image, known_length
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Insufficient calibration parameters"
            )
        
        return {
            "scale_factor": calibration.scale_factor,
            "reference_length": calibration.reference_length,
            "reference_points": calibration.reference_points,
            "calibrated_at": calibration.calibrated_at.isoformat(),
            "orientation": calibration.orientation,
            "pixels_per_meter": calibration.scale_factor,
            "meters_per_pixel": 1.0 / calibration.scale_factor if calibration.scale_factor > 0 else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Calibration failed: {str(e)}"
        )


@router.get("/export-dxf")
async def export_dxf(site_id: str):
    """Export site plan as DXF file.
    
    Args:
        site_id: Site identifier
        
    Returns:
        DXF file
    """
    if site_id not in processed_plans:
        raise HTTPException(
            status_code=404,
            detail=f"Site {site_id} not found"
        )
    
    plan = processed_plans[site_id]
    
    if plan.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Processing not completed: {plan.error}"
        )
    
    try:
        # Export to temporary file
        output_path = Path(f"/tmp/site_{site_id}.dxf")
        success = processor.export_dxf(output_path)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="DXF export failed"
            )
        
        return FileResponse(
            output_path,
            media_type="application/dxf",
            filename=f"site_{site_id}.dxf"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DXF export failed: {str(e)}"
        )


@router.get("/export-svg")
async def export_svg(site_id: str):
    """Export site plan as SVG file.
    
    Args:
        site_id: Site identifier
        
    Returns:
        SVG file
    """
    if site_id not in processed_plans:
        raise HTTPException(
            status_code=404,
            detail=f"Site {site_id} not found"
        )
    
    plan = processed_plans[site_id]
    
    if plan.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Processing not completed: {plan.error}"
        )
    
    try:
        output_path = Path(f"/tmp/site_{site_id}.svg")
        success = processor.export_svg(output_path)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="SVG export failed"
            )
        
        return FileResponse(
            output_path,
            media_type="image/svg+xml",
            filename=f"site_{site_id}.svg"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SVG export failed: {str(e)}"
        )


@router.get("/export-geojson")
async def export_geojson(site_id: str):
    """Export site plan as GeoJSON.
    
    Args:
        site_id: Site identifier
        
    Returns:
        GeoJSON data
    """
    if site_id not in processed_plans:
        raise HTTPException(
            status_code=404,
            detail=f"Site {site_id} not found"
        )
    
    plan = processed_plans[site_id]
    
    if plan.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Processing not completed: {plan.error}"
        )
    
    try:
        geojson = processor.vectorizer.to_geojson(plan.data)
        return JSONResponse(content=geojson)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"GeoJSON export failed: {str(e)}"
        )