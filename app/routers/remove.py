from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from app.services.inpainting_service import get_inpainting_service
import os

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

class ReplaceFurnitureRequest(BaseModel):
    image_path: str
    mask_path: str
    furniture_image: str  # Base64 encoded image OR path to predefined furniture
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

@router.post("/replace-furniture")
async def replace_with_furniture(request: ReplaceFurnitureRequest):
    """
    Replace an object with custom furniture image.
    Enlarges mask by 100px on each side, places furniture at center, then uses Flux Kontext.
    Supports both base64 encoded images and predefined furniture paths.
    """
    try:
        from datetime import datetime
        import base64
        
        inpainting_service = get_inpainting_service()
        
        # Convert URL paths to filesystem paths if needed
        image_path = request.image_path
        mask_path = request.mask_path
        
        if image_path.startswith('uploads/'):
            image_path = 'app/static/' + image_path
        if mask_path.startswith('uploads/'):
            mask_path = 'app/static/' + mask_path
        
        # Check if furniture_image is a path or base64
        furniture_image = request.furniture_image
        
        if furniture_image.startswith('app/static/furnitures/') or furniture_image.startswith('app\\static\\furnitures\\'):
            # It's a predefined furniture path
            furniture_path = Path(furniture_image)
            if not furniture_path.exists():
                raise HTTPException(status_code=404, detail=f"Furniture image not found: {furniture_image}")
            print(f"[FURNITURE] Using predefined furniture: {furniture_path}")
        else:
            # It's base64 encoded - decode and save temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            furniture_filename = f"furniture_{timestamp}.png"
            furniture_path = Path("app/static/uploads") / furniture_filename
            
            # Remove data URI prefix if present
            if ',' in furniture_image:
                furniture_image = furniture_image.split(',')[1]
            
            furniture_data = base64.b64decode(furniture_image)
            with open(furniture_path, 'wb') as f:
                f.write(furniture_data)
            
            print(f"[FURNITURE] Saved uploaded furniture image to: {furniture_path}")
        
        # Perform furniture replacement
        output_path, result_info = inpainting_service.replace_with_custom_furniture(
            image_path=image_path,
            mask_path=mask_path,
            furniture_path=str(furniture_path),
            object_name=request.object_name,
            guidance=request.guidance
        )
        
        # Convert path to URL
        if "app/static/" in output_path:
            output_url = "/" + output_path.replace("app/static/", "")
        else:
            output_url = "/uploads/" + Path(output_path).name
        
        print(f"[FURNITURE] Output path: {output_path}")
        print(f"[FURNITURE] Output URL: {output_url}")
        
        return {
            "status": "success",
            "output_url": output_url,
            "output_path": output_path,
            **result_info
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Furniture replacement failed: {str(e)}")

@router.post("/replace-furniture-v2")
async def replace_with_furniture_v2(request: ReplaceFurnitureRequest):
    """
    Replace an object with custom furniture image using Flux 2 Dev.
    Sends mask-overlayed image and furniture image separately.
    Supports both base64 encoded images and predefined furniture paths.
    """
    try:
        from datetime import datetime
        import base64
        
        inpainting_service = get_inpainting_service()
        
        # Convert URL paths to filesystem paths if needed
        image_path = request.image_path
        mask_path = request.mask_path
        
        if image_path.startswith('uploads/'):
            image_path = 'app/static/' + image_path
        if mask_path.startswith('uploads/'):
            mask_path = 'app/static/' + mask_path
        
        # Check if furniture_image is a path or base64
        furniture_image = request.furniture_image
        
        if furniture_image.startswith('app/static/furnitures/') or furniture_image.startswith('app\\static\\furnitures\\'):
            # It's a predefined furniture path
            furniture_path = Path(furniture_image)
            if not furniture_path.exists():
                raise HTTPException(status_code=404, detail=f"Furniture image not found: {furniture_image}")
            print(f"[FURNITURE V2] Using predefined furniture: {furniture_path}")
        else:
            # It's base64 encoded - decode and save temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            furniture_filename = f"furniture_v2_{timestamp}.png"
            furniture_path = Path("app/static/uploads") / furniture_filename
            
            # Remove data URI prefix if present
            if ',' in furniture_image:
                furniture_image = furniture_image.split(',')[1]
            
            furniture_data = base64.b64decode(furniture_image)
            with open(furniture_path, 'wb') as f:
                f.write(furniture_data)
            
            print(f"[FURNITURE V2] Saved uploaded furniture image to: {furniture_path}")
        
        # Perform furniture replacement using v2 (Flux 2 Dev)
        output_path, result_info = inpainting_service.replace_with_custom_furniture_v2(
            image_path=image_path,
            mask_path=mask_path,
            furniture_path=str(furniture_path),
            object_name=request.object_name
        )
        
        # Convert path to URL
        if "app/static/" in output_path:
            output_url = "/" + output_path.replace("app/static/", "")
        else:
            output_url = "/uploads/" + Path(output_path).name
        
        print(f"[FURNITURE V2] Output path: {output_path}")
        print(f"[FURNITURE V2] Output URL: {output_url}")
        
        return {
            "status": "success",
            "output_url": output_url,
            "output_path": output_path,
            **result_info
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Furniture replacement v2 failed: {str(e)}")

@router.get("/list-furnitures")
async def list_furnitures():
    """
    List all available predefined furniture images organized by category.
    """
    try:
        furnitures_dir = Path("app/static/furnitures")
        
        if not furnitures_dir.exists():
            return {"categories": []}
        
        categories = []
        
        # Iterate through category folders
        for category_folder in furnitures_dir.iterdir():
            if not category_folder.is_dir():
                continue
            
            # Clean up category name
            category_name = category_folder.name.replace("-20251218T165450Z-1-001", "").replace("-20251218T165458Z-1-001", "").replace("-20251218T165027Z-1-001", "").replace("-20251218T165044Z-1-001", "").replace("-20251218T165439Z-1-001", "").replace("-20251218T165447Z-1-001", "").strip()
            
            furniture_items = []
            
            # Recursively find all image files in this category
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                for img_path in category_folder.rglob(ext):
                    # Create relative URL path
                    relative_path = img_path.relative_to(Path("app/static"))
                    url_path = "/" + str(relative_path).replace("\\", "/")
                    
                    # Extract furniture name from filename
                    furniture_name = img_path.stem.replace("luxury ", "").replace("coffee ", "coffee ").title()
                    
                    furniture_items.append({
                        "name": furniture_name,
                        "url": url_path,
                        "path": str(img_path)
                    })
            
            if furniture_items:
                categories.append({
                    "category": category_name,
                    "items": furniture_items
                })
        
        return {"categories": categories}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list furnitures: {str(e)}")
