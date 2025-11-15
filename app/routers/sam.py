from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.sam_service import get_sam_service

router = APIRouter()


class ImageUploadRequest(BaseModel):
    image: str
    session_id: str = "default"


class AddPointRequest(BaseModel):
    session_id: str = "default"
    x: int
    y: int
    is_positive: bool = True


class SessionRequest(BaseModel):
    session_id: str = "default"


@router.post("/sam/upload")
async def upload_image(request: ImageUploadRequest):
    """
    Upload image for SAM interactive segmentation
    """
    try:
        sam_service = get_sam_service()
        result = sam_service.upload_image(request.image, request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sam/add_point")
async def add_point(request: AddPointRequest):
    """
    Add a point to the current segmentation
    """
    try:
        sam_service = get_sam_service()
        result = sam_service.add_point(
            request.session_id,
            request.x,
            request.y,
            request.is_positive
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sam/clear_clicks")
async def clear_clicks(request: SessionRequest):
    """
    Clear all clicks and return to original image
    """
    try:
        sam_service = get_sam_service()
        result = sam_service.clear_clicks(request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sam/add_mask")
async def add_mask(request: SessionRequest):
    """
    Add current mask to the mask collection
    """
    try:
        sam_service = get_sam_service()
        result = sam_service.add_mask(request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sam/delete_masks")
async def delete_masks(request: SessionRequest):
    """
    Delete all masks and return to original image
    """
    try:
        sam_service = get_sam_service()
        result = sam_service.delete_masks(request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class ProcessRequest(BaseModel):
    session_id: str = "default"
    num_inference_steps: int = 20
    guidance_scale: float = 2.5
    seed: int = 300000


@router.post("/sam/process")
async def process_and_remove(request: ProcessRequest):
    """
    Process SAM masks and remove objects using ObjectClear
    """
    try:
        from app.services.inpainting_service import get_inpainting_service
        from pathlib import Path
        from datetime import datetime
        import numpy as np
        from PIL import Image
        
        sam_service = get_sam_service()
        
        if request.session_id not in sam_service.sessions:
            raise HTTPException(status_code=400, detail='No image uploaded for this session')
        
        state = sam_service.sessions[request.session_id]
        
        # Get combined mask
        combined_mask = sam_service.get_combined_mask(request.session_id)
        if combined_mask is None or not np.any(combined_mask):
            raise HTTPException(status_code=400, detail='No masks created. Please add at least one mask.')
        
        # Save image and mask temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uploads_dir = Path("app/static/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = uploads_dir / f"sam_image_{timestamp}.png"
        mask_path = uploads_dir / f"sam_mask_{timestamp}.png"
        
        # Save original image
        origin_image = Image.fromarray(state['origin_image'].astype('uint8'))
        origin_image.save(image_path)
        
        # Save combined mask (convert to binary mask)
        mask_binary = (combined_mask > 0).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_binary)
        mask_image.save(mask_path)
        
        # Call inpainting service
        inpainting_service = get_inpainting_service()
        output_path, result_info = inpainting_service.remove_object(
            image_path=str(image_path),
            mask_path=str(mask_path),
            object_name="object",
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        
        # Return result with URL
        output_url = "/" + str(Path(output_path).relative_to("app/static"))
        
        return {
            "success": True,
            "output_url": output_url,
            "output_path": output_path,
            "inference_time": result_info.get("inference_time", 0),
            "num_masks": len(state['masks']) if state['masks'] else 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
