from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from pathlib import Path
import shutil
from datetime import datetime
import json
from app.services.segmentation_service import get_segmentation_service
from app.services.yolo_service import get_yolo_service
from PIL import Image
import numpy as np

router = APIRouter()

class DetectObjectsRequest(BaseModel):
    image_path: str
    confidence_threshold: float = 0.25

# Ensure uploads directory exists
UPLOADS_DIR = Path("app/static/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_files(
    image: UploadFile = File(...),
    mask: UploadFile = File(None),
    points: UploadFile = File(None),
    mode: str = Form(None)
):
    """
    Upload image and mask/points for object removal.
    Modes:
    - brush: Uses the provided mask directly
    - click (SAM): Uses SAM to generate mask from points
    - yolo: Uses YOLO to detect and segment object at click point
    """
    try:
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image
        image_filename = f"image_{timestamp}.png"
        image_path = UPLOADS_DIR / image_filename
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        result = {
            "status": "success",
            "image_path": f"uploads/{image_filename}",
            "timestamp": timestamp,
            "mode": None
        }
        
        # Handle brush mode - mask provided directly
        if mask:
            mask_filename = f"mask_{timestamp}.png"
            mask_path = UPLOADS_DIR / mask_filename
            with mask_path.open("wb") as buffer:
                shutil.copyfileobj(mask.file, buffer)
            result["mask_path"] = f"uploads/{mask_filename}"
            result["mode"] = "brush"
        
        # Handle click/YOLO mode - generate mask from points
        elif points:
            points_filename = f"points_{timestamp}.json"
            points_path = UPLOADS_DIR / points_filename
            points_content = await points.read()
            with points_path.open("wb") as buffer:
                buffer.write(points_content)
            result["points_path"] = str(points_path)
            
            # Parse points
            try:
                points_data = json.loads(points_content)
                result["points"] = points_data
                
                if points_data and len(points_data) > 0:
                    mask_filename = f"mask_{timestamp}.png"
                    mask_path = UPLOADS_DIR / mask_filename
                    
                    # Use YOLO mode if specified
                    if mode == "yolo":
                        yolo_service = get_yolo_service()
                        
                        # Use first point for single object, or multiple for multiple objects
                        if len(points_data) == 1:
                            _, mask, detection_info = yolo_service.segment_from_click(
                                image_path=str(image_path),
                                click_point=points_data[0],
                                output_mask_path=str(mask_path)
                            )
                            result["detection"] = detection_info
                        else:
                            _, mask, detection_infos = yolo_service.segment_multiple_clicks(
                                image_path=str(image_path),
                                click_points=points_data,
                                output_mask_path=str(mask_path)
                            )
                            result["detections"] = detection_infos
                        
                        result["mask_path"] = f"uploads/{mask_filename}"
                        result["mode"] = "yolo"
                        result["segmentation"] = "yolo_generated"
                    
                    # Use SAM mode (default for click)
                    else:
                        segmentation_service = get_segmentation_service()
                        
                        _, mask = segmentation_service.segment_from_points(
                            image_path=str(image_path),
                            points=points_data,
                            output_mask_path=str(mask_path)
                        )
                        
                        result["mask_path"] = f"uploads/{mask_filename}"
                        result["mode"] = "sam"
                        result["segmentation"] = "sam_generated"
                else:
                    result["warning"] = "No points provided for segmentation"
                    
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid points JSON: {str(e)}")
            except Exception as e:
                # If segmentation fails, still save the data but note the error
                result["segmentation_error"] = str(e)
                result["mode"] = mode or "click"
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/detect-objects")
async def detect_objects(request: DetectObjectsRequest):
    """
    Detect all objects in an image using YOLO.
    Returns list of detected objects with their masks.
    """
    try:
        yolo_service = get_yolo_service()
        
        # Convert URL path to filesystem path if needed
        image_path = request.image_path
        if image_path.startswith('uploads/'):
            image_path = 'app/static/' + image_path
        
        # Detect all objects
        detection_result = yolo_service.detect_all_objects(
            image_path=image_path,
            confidence_threshold=request.confidence_threshold
        )
        
        # Save individual masks for each detected object
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_masks = []
        
        for obj in detection_result.get("detected_objects", []):
            mask_filename = f"mask_{timestamp}_obj{obj['id']}_{obj['class_name']}.png"
            mask_path = UPLOADS_DIR / mask_filename
            
            # Convert mask_data back to numpy array and save
            mask_array = np.array(obj['mask_data'], dtype=np.uint8)
            mask_image = Image.fromarray(mask_array)
            mask_image.save(mask_path)
            
            # Add mask URL to object info
            obj['mask_url'] = f"uploads/{mask_filename}"
            # Remove the large mask_data array from response
            del obj['mask_data']
            saved_masks.append(f"uploads/{mask_filename}")
        
        detection_result['saved_masks'] = saved_masks
        
        return {
            "status": "success",
            **detection_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")
