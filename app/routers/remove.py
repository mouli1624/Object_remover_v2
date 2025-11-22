from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from app.services.inpainting_service import get_inpainting_service

router = APIRouter()

class RemoveObjectRequest(BaseModel):
    image_path: str
    mask_path: str
    object_name: str = "object"

class RemoveMultipleObjectsRequest(BaseModel):
    image_path: str
    mask_path: str
    object_names: list[str]

class ReplaceObjectRequest(BaseModel):
    image_path: str
    mask_path: str
    prompt: str
    object_name: str = "object"
    guidance: float = 4.5

@router.post("/remove")
async def remove_object(request: RemoveObjectRequest):
    """
    Remove a single object from the image using Flux Kontext.
    """
    try:
        inpainting_service = get_inpainting_service()
        
        # Convert URL paths to filesystem paths if needed
        image_path = request.image_path
        mask_path = request.mask_path
        
        # If paths start with 'uploads/', convert to 'app/static/uploads/'
        if image_path.startswith('uploads/'):
            image_path = 'app/static/' + image_path
        if mask_path.startswith('uploads/'):
            mask_path = 'app/static/' + mask_path
        
        # Perform inpainting
        output_path, result_info = inpainting_service.remove_object(
            image_path=image_path,
            mask_path=mask_path,
            object_name=request.object_name
        )
        
        # Convert path to URL
        # output_path is like: app/static/uploads/result_image_xxx.png
        # We need: /uploads/result_image_xxx.png
        if "app/static/" in output_path:
            output_url = "/" + output_path.replace("app/static/", "")
        else:
            # Fallback: assume it's already in uploads folder
            output_url = "/uploads/" + Path(output_path).name
        
        print(f"[REMOVE] Output path: {output_path}")
        print(f"[REMOVE] Output URL: {output_url}")
        
        return {
            "status": "success",
            "output_url": output_url,
            "output_path": output_path,
            **result_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object removal failed: {str(e)}")

@router.post("/remove-multiple")
async def remove_multiple_objects(request: RemoveMultipleObjectsRequest):
    """
    Remove multiple objects from the image using Flux Kontext.
    """
    try:
        inpainting_service = get_inpainting_service()
        
        # Convert URL paths to filesystem paths if needed
        image_path = request.image_path
        mask_path = request.mask_path
        
        # If paths start with 'uploads/', convert to 'app/static/uploads/'
        if image_path.startswith('uploads/'):
            image_path = 'app/static/' + image_path
        if mask_path.startswith('uploads/'):
            mask_path = 'app/static/' + mask_path
        
        # Perform inpainting
        output_path, result_info = inpainting_service.remove_multiple_objects(
            image_path=image_path,
            mask_path=mask_path,
            object_names=request.object_names
        )
        
        # Convert path to URL
        output_url = "/" + output_path.replace("app/static/", "")
        
        return {
            "status": "success",
            "output_url": output_url,
            "output_path": output_path,
            **result_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object removal failed: {str(e)}")

@router.post("/remove-kontext")
async def remove_object_kontext(request: RemoveObjectRequest):
    """
    Remove a single object from the image using Flux Kontext Pro.
    This method draws the mask directly on the image.
    """
    try:
        inpainting_service = get_inpainting_service()
        
        # Convert URL paths to filesystem paths if needed
        image_path = request.image_path
        mask_path = request.mask_path
        
        # If paths start with 'uploads/', convert to 'app/static/uploads/'
        if image_path.startswith('uploads/'):
            image_path = 'app/static/' + image_path
        if mask_path.startswith('uploads/'):
            mask_path = 'app/static/' + mask_path
        
        # Perform inpainting with Kontext
        output_path, result_info = inpainting_service.remove_object_kontext(
            image_path=image_path,
            mask_path=mask_path,
            object_name=request.object_name
        )
        
        # Convert path to URL
        if "app/static/" in output_path:
            output_url = "/" + output_path.replace("app/static/", "")
        else:
            output_url = "/uploads/" + Path(output_path).name
        
        print(f"[REMOVE-KONTEXT] Output path: {output_path}")
        print(f"[REMOVE-KONTEXT] Output URL: {output_url}")
        
        return {
            "status": "success",
            "output_url": output_url,
            "output_path": output_path,
            **result_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kontext removal failed: {str(e)}")

@router.post("/remove-hybrid")
async def remove_object_hybrid(request: RemoveObjectRequest):
    """
    Remove a single object using hybrid mode: LaMa + Flux Kontext.
    First runs LaMa for initial inpainting, then Flux Kontext for refinement.
    """
    try:
        inpainting_service = get_inpainting_service()
        
        # Convert URL paths to filesystem paths if needed
        image_path = request.image_path
        mask_path = request.mask_path
        
        # If paths start with 'uploads/', convert to 'app/static/uploads/'
        if image_path.startswith('uploads/'):
            image_path = 'app/static/' + image_path
        if mask_path.startswith('uploads/'):
            mask_path = 'app/static/' + mask_path
        
        # Perform hybrid inpainting
        output_path, result_info = inpainting_service.remove_object_hybrid(
            image_path=image_path,
            mask_path=mask_path,
            object_name=request.object_name
        )
        
        # Convert path to URL
        if "app/static/" in output_path:
            output_url = "/" + output_path.replace("app/static/", "")
        else:
            output_url = "/uploads/" + Path(output_path).name
        
        print(f"[REMOVE-HYBRID] Output path: {output_path}")
        print(f"[REMOVE-HYBRID] Output URL: {output_url}")
        
        return {
            "status": "success",
            "output_url": output_url,
            "output_path": output_path,
            **result_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid removal failed: {str(e)}")

@router.post("/replace")
async def replace_object(request: ReplaceObjectRequest):
    """
    Replace an object in the image with AI-generated content using Flux Kontext Pro.
    The mask is overlaid on the image as white pixels, then the prompt guides what to generate.
    """
    try:
        inpainting_service = get_inpainting_service()
        
        # Convert URL paths to filesystem paths if needed
        image_path = request.image_path
        mask_path = request.mask_path
        
        # If paths start with 'uploads/', convert to 'app/static/uploads/'
        if image_path.startswith('uploads/'):
            image_path = 'app/static/' + image_path
        if mask_path.startswith('uploads/'):
            mask_path = 'app/static/' + mask_path
        
        # Perform object replacement
        output_path, result_info = inpainting_service.replace_object(
            image_path=image_path,
            mask_path=mask_path,
            prompt=request.prompt,
            object_name=request.object_name,
            guidance=6
        )
        
        # Convert path to URL
        if "app/static/" in output_path:
            output_url = "/" + output_path.replace("app/static/", "")
        else:
            output_url = "/uploads/" + Path(output_path).name
        
        print(f"[REPLACE] Output path: {output_path}")
        print(f"[REPLACE] Output URL: {output_url}")
        print(f"[REPLACE] Prompt: {request.prompt}")
        
        return {
            "status": "success",
            "output_url": output_url,
            "output_path": output_path,
            **result_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object replacement failed: {str(e)}")
