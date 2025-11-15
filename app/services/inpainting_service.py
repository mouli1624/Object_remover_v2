import os
import base64
import replicate
import requests
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file
class InpaintingService:
    """
    Service for object removal using LaMa via Replicate
    """
    
    def __init__(self):
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            print("=" * 60)
            print("WARNING: REPLICATE_API_TOKEN not found in environment variables!")
            print("Set it with: export REPLICATE_API_TOKEN='your-token-here'") 
            print("Get your token from: https://replicate.com/account/api-tokens")
            print("=" * 60)
        else:
            print("=" * 60)
            print("✅ Replicate API token found!")
            print("Using LaMa model for inpainting")
            print("=" * 60)
    
    def dilate_mask(self, mask_path: str, dilation_pixels: int = 30) -> str:
        """
        Dilate (expand) the mask by the specified number of pixels.
        
        Args:
            mask_path: Path to the mask image
            dilation_pixels: Number of pixels to expand the mask (default: 30)
            
        Returns:
            Path to the dilated mask
        """
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Create a circular kernel for dilation
        kernel_size = dilation_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate the mask
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Save dilated mask
        mask_path_obj = Path(mask_path)
        dilated_mask_path = str(mask_path_obj.parent / f"dilated_{mask_path_obj.name}")
        cv2.imwrite(dilated_mask_path, dilated_mask)
        
        print(f"Mask dilated by {dilation_pixels} pixels")
        
        return dilated_mask_path
    
    def remove_object(
        self,
        image_path: str,
        mask_path: str,
        object_name: str = "object",
        output_path: Optional[str] = None,
        **kwargs  # Ignore extra parameters for compatibility
    ) -> Tuple[str, dict]:
        """
        Remove object from image using LaMa via Replicate.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the mask image
            object_name: Name of the object to remove (for logging)
            output_path: Optional path to save the result
        
        Returns:
            Tuple of (output_image_path, result_info)
        """
        if not self.api_token:
            raise Exception("REPLICATE_API_TOKEN not set. Please set your Replicate API token.")
        
        try:
            import time
            start_time = time.time()
            
            print("=" * 60)
            print(f"Removing '{object_name}' from image using LaMa...")
            print("=" * 60)
            
            # Dilate the mask to expand it by 30 pixels
            dilated_mask_path = self.dilate_mask(mask_path, dilation_pixels=90)
            
            # Load image and dilated mask
            with open(image_path, 'rb') as f:
                image_data = f.read()
            with open(dilated_mask_path, 'rb') as f:
                mask_data = f.read()
            
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mask_base64 = base64.b64encode(mask_data).decode('utf-8')
            
            # Create data URIs
            image_uri = f"data:image/png;base64,{image_base64}"
            mask_uri = f"data:image/png;base64,{mask_base64}"
            
            print("Sending request to Replicate LaMa model...")
            
            # Run Flux Fill model on Replicate
            output = replicate.run(
                # "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72",
                "black-forest-labs/flux-fill-dev",
                input={
                    "prompt": "empty room",
                    "guidance":0.7,
                    "image": image_uri,
                    "mask": mask_uri
                }
            )
            
            print(f"Replicate output type: {type(output)}")
            print(f"Replicate output: {output}")
            
            # Flux Fill returns a list of URLs
            if isinstance(output, list) and len(output) > 0:
                result_url = output[0]
                print(f"Downloading result from: {result_url}")
                
                # Download the result from URL
                response = requests.get(result_url)
                response.raise_for_status()
                result_data = response.content
            else:
                # Fallback for other models that return file-like objects
                result_data = output.read()
            
            # Save output
            if output_path is None:
                image_path_obj = Path(image_path)
                output_path = str(image_path_obj.parent / f"result_{image_path_obj.name}")
            
            with open(output_path, 'wb') as f:
                f.write(result_data)
            
            inference_time = time.time() - start_time
            
            print(f"✅ Inpainting completed in {inference_time:.2f}s")
            print(f"Result saved to: {output_path}")
            print("=" * 60)
            
            result_info = {
                "success": True,
                "object_removed": object_name,
                "output_path": output_path,
                "inference_time": inference_time
            }
            
            return output_path, result_info
                
        except Exception as e:
            print("=" * 60)
            print(f"ERROR during inpainting: {e}")
            print("=" * 60)
            raise
    
    def remove_multiple_objects(
        self,
        image_path: str,
        mask_path: str,
        object_names: list,
        output_path: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Remove multiple objects from image.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the combined mask image
            object_names: List of object names to remove
            output_path: Optional path to save the result
        
        Returns:
            Tuple of (output_image_path, result_info)
        """
        # Create combined prompt
        if len(object_names) == 1:
            object_desc = object_names[0]
        elif len(object_names) == 2:
            object_desc = f"{object_names[0]} and {object_names[1]}"
        else:
            object_desc = ", ".join(object_names[:-1]) + f", and {object_names[-1]}"
        
        return self.remove_object(image_path, mask_path, object_desc, output_path)
    
    def remove_object_kontext(
        self,
        image_path: str,
        mask_path: str,
        object_name: str = "object",
        output_path: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Remove object using Flux Kontext Pro by drawing mask directly on image.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the mask image
            object_name: Name of the object to remove (for logging)
            output_path: Optional path to save the result
        
        Returns:
            Tuple of (output_image_path, result_info)
        """
        if not self.api_token:
            raise Exception("REPLICATE_API_TOKEN not set. Please set your Replicate API token.")
        
        try:
            start_time = time.time()
            
            print("=" * 60)
            print(f"Removing '{object_name}' from image using Flux Kontext Pro...")
            print("=" * 60)
            
            # Dilate the mask to expand it by 100 pixels
            dilated_mask_path = self.dilate_mask(mask_path, dilation_pixels=90)
            
            # Load image and dilated mask
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(dilated_mask_path, cv2.IMREAD_GRAYSCALE)
            
            print(f"Original image shape: {image_rgb.shape}")
            print(f"Original mask shape: {mask.shape}")
            
            # Remove extra dimension if present (e.g., (H, W, 1) -> (H, W))
            if len(mask.shape) == 3 and mask.shape[2] == 1:
                mask = mask.squeeze(axis=2)
                print(f"Squeezed mask to shape: {mask.shape}")
            
            # Ensure mask and image have same dimensions
            if mask.shape[:2] != image_rgb.shape[:2]:
                print(f"Resizing mask from {mask.shape} to match image {image_rgb.shape[:2]}")
                mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
            
            print(f"Final mask shape: {mask.shape}, image shape: {image_rgb.shape}")
            
            # Create a copy of the image to draw the mask on
            masked_image = image_rgb.copy()
            
            # Simple approach: iterate through channels and apply mask
            for i in range(3):  # RGB channels
                masked_image[:, :, i] = np.where(mask > 127, 255, image_rgb[:, :, i])
            
            masked_image = masked_image.astype(np.uint8)
            
            # Convert to PIL Image
            masked_pil = Image.fromarray(masked_image)
            
            # Save the masked input image for debugging
            image_path_obj = Path(image_path)
            input_save_path = str(image_path_obj.parent / f"input_kontext_{image_path_obj.name}")
            masked_pil.save(input_save_path)
            print(f"Saved input image to: {input_save_path}")
            
            # Convert to base64
            buffered = io.BytesIO()
            masked_pil.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Create data URI
            image_uri = f"data:image/png;base64,{image_base64}"
            
            print("Sending request to Replicate Flux Kontext Pro...")
            
            # Run Flux Kontext Pro model on Replicate
            output = replicate.run(
                "black-forest-labs/flux-kontext-pro",
                input={
                    "prompt": "remove blur from the image",
                    "input_image": image_uri,
                    "guidance":"4.5",
                    "output_format": "png"
                }
            )
            
            print(f"Replicate output type: {type(output)}")
            print(f"Replicate output: {output}")
            
            # Flux Kontext Pro returns a FileOutput object
            # The FileOutput object itself is the URL when converted to string
            if hasattr(output, 'read'):
                # Direct file-like object with read method
                result_data = output.read()
            elif isinstance(output, str):
                # Direct URL string
                result_url = output
                print(f"Downloading result from: {result_url}")
                response = requests.get(result_url)
                response.raise_for_status()
                result_data = response.content
            elif isinstance(output, list) and len(output) > 0:
                # List of URLs
                result_url = output[0]
                print(f"Downloading result from: {result_url}")
                response = requests.get(result_url)
                response.raise_for_status()
                result_data = response.content
            else:
                # FileOutput object - convert to string to get URL
                result_url = str(output)
                print(f"Downloading result from: {result_url}")
                response = requests.get(result_url)
                response.raise_for_status()
                result_data = response.content
            
            # Save output
            if output_path is None:
                image_path_obj = Path(image_path)
                output_path = str(image_path_obj.parent / f"result_kontext_{image_path_obj.name}")
            
            with open(output_path, 'wb') as f:
                f.write(result_data)
            
            inference_time = time.time() - start_time
            
            print(f"✅ Kontext inpainting completed in {inference_time:.2f}s")
            print(f"Result saved to: {output_path}")
            print("=" * 60)
            
            result_info = {
                "success": True,
                "object_removed": object_name,
                "output_path": output_path,
                "inference_time": inference_time,
                "model": "flux-kontext-pro"
            }
            
            return output_path, result_info
                
        except Exception as e:
            print("=" * 60)
            print(f"ERROR during Kontext inpainting: {e}")
            print("=" * 60)
            raise
    
    def remove_object_hybrid(
        self,
        image_path: str,
        mask_path: str,
        object_name: str = "object",
        output_path: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Hybrid mode: First uses LaMa for initial inpainting, then Flux Kontext for refinement.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the mask image
            object_name: Name of the object to remove (for logging)
            output_path: Optional path to save the final result
        
        Returns:
            Tuple of (output_image_path, result_info)
        """
        if not self.api_token:
            raise Exception("REPLICATE_API_TOKEN not set. Please set your Replicate API token.")
        
        try:
            start_time = time.time()
            
            print("=" * 60)
            print(f"🔄 HYBRID MODE: Removing '{object_name}' using LaMa + Flux Kontext")
            print("=" * 60)
            
            # Step 1: Run LaMa inpainting
            print("\n📍 Step 1/2: Running LaMa inpainting...")
            image_path_obj = Path(image_path)
            lama_output_path = str(image_path_obj.parent / f"lama_intermediate_{image_path_obj.name}")
            
            # Dilate mask for LaMa
            dilated_mask_path = self.dilate_mask(mask_path, dilation_pixels=90)
            
            # Load image and dilated mask
            with open(image_path, 'rb') as f:
                image_data = f.read()
            with open(dilated_mask_path, 'rb') as f:
                mask_data = f.read()
            
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mask_base64 = base64.b64encode(mask_data).decode('utf-8')
            
            # Create data URIs
            image_uri = f"data:image/png;base64,{image_base64}"
            mask_uri = f"data:image/png;base64,{mask_base64}"
            
            print("Sending request to LaMa model...")
            
            # Run LaMa model
            lama_output = replicate.run(
                "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72",
                input={
                    "image": image_uri,
                    "mask": mask_uri
                }
            )
            
            # Handle LaMa output
            if hasattr(lama_output, 'read'):
                lama_result_data = lama_output.read()
            elif isinstance(lama_output, str):
                response = requests.get(lama_output)
                response.raise_for_status()
                lama_result_data = response.content
            elif isinstance(lama_output, list) and len(lama_output) > 0:
                response = requests.get(lama_output[0])
                response.raise_for_status()
                lama_result_data = response.content
            else:
                result_url = str(lama_output)
                response = requests.get(result_url)
                response.raise_for_status()
                lama_result_data = response.content
            
            # Save LaMa intermediate result
            with open(lama_output_path, 'wb') as f:
                f.write(lama_result_data)
            
            lama_time = time.time() - start_time
            print(f"✅ LaMa completed in {lama_time:.2f}s")
            print(f"Intermediate result saved to: {lama_output_path}")
            
            # Step 2: Run Flux Kontext on LaMa output
            print("\n📍 Step 2/2: Running Flux Kontext refinement...")
            kontext_start = time.time()
            
            # Load LaMa result directly (no mask overlay)
            with open(lama_output_path, 'rb') as f:
                lama_result_data = f.read()
            
            # Convert LaMa output to base64 for Kontext
            kontext_image_base64 = base64.b64encode(lama_result_data).decode('utf-8')
            kontext_image_uri = f"data:image/png;base64,{kontext_image_base64}"
            
            print("Sending request to Flux Kontext Pro...")
            
            # Run Flux Kontext Pro
            kontext_output = replicate.run(
                "black-forest-labs/flux-kontext-pro",
                input={
                    "prompt": "remove blur from the image",
                    "input_image": kontext_image_uri,
                    "guidance": "4.5",
                    "output_format": "png"
                }
            )
            
            # Handle Kontext output
            if hasattr(kontext_output, 'read'):
                result_data = kontext_output.read()
            elif isinstance(kontext_output, str):
                response = requests.get(kontext_output)
                response.raise_for_status()
                result_data = response.content
            elif isinstance(kontext_output, list) and len(kontext_output) > 0:
                response = requests.get(kontext_output[0])
                response.raise_for_status()
                result_data = response.content
            else:
                result_url = str(kontext_output)
                response = requests.get(result_url)
                response.raise_for_status()
                result_data = response.content
            
            # Save final output
            if output_path is None:
                output_path = str(image_path_obj.parent / f"result_hybrid_{image_path_obj.name}")
            
            with open(output_path, 'wb') as f:
                f.write(result_data)
            
            kontext_time = time.time() - kontext_start
            total_time = time.time() - start_time
            
            print(f"✅ Flux Kontext completed in {kontext_time:.2f}s")
            print(f"🎉 HYBRID MODE completed in {total_time:.2f}s total")
            print(f"Final result saved to: {output_path}")
            print("=" * 60)
            
            result_info = {
                "success": True,
                "object_removed": object_name,
                "output_path": output_path,
                "lama_time": lama_time,
                "kontext_time": kontext_time,
                "total_time": total_time,
                "model": "hybrid_lama_kontext",
                "intermediate_path": lama_output_path
            }
            
            return output_path, result_info
                
        except Exception as e:
            print("=" * 60)
            print(f"ERROR during hybrid inpainting: {e}")
            print("=" * 60)
            raise


# Singleton instance
_inpainting_service = None

def get_inpainting_service() -> InpaintingService:
    """
    Get or create the inpainting service singleton.
    """
    global _inpainting_service
    if _inpainting_service is None:
        _inpainting_service = InpaintingService()
    return _inpainting_service
