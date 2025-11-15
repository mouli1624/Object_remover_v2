from pathlib import Path
import numpy as np
from PIL import Image
import torch
from typing import List, Tuple, Optional
import cv2

class SegmentationService:
    """
    Service for object segmentation using SAM (Segment Anything Model)
    """
    
    def __init__(self):
        self.model = None
        self.predictor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize SAM model. Uses vit_h for best quality.
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            # Use vit_h for best quality
            model_type = "vit_h"
            
            # Try multiple possible paths for the checkpoint
            possible_paths = [
                "../models/sam_vit_h_4b8939.pth",  # From backend/app/services/
                "models/sam_vit_h_4b8939.pth",      # From backend/
                "../../models/sam_vit_h_4b8939.pth" # From backend/app/
            ]
            
            checkpoint_path = None
            for path in possible_paths:
                if Path(path).exists():
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                print("=" * 60)
                print("WARNING: SAM checkpoint not found!")
                print("Tried paths:")
                for path in possible_paths:
                    print(f"  - {path}")
                print("Click mode will use simple circular masks as fallback.")
                print("=" * 60)
                return
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("=" * 60)
            print(f"Loading SAM model from: {checkpoint_path}")
            print(f"Device: {device}")
            print("This may take a few seconds...")
            
            self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.model.to(device=device)
            self.predictor = SamPredictor(self.model)
            
            print(f"✅ SAM model loaded successfully!")
            print("=" * 60)
            
        except ImportError as e:
            print("=" * 60)
            print("WARNING: segment-anything package not installed!")
            print("Install with: pip3 install git+https://github.com/facebookresearch/segment-anything.git")
            print(f"Error: {e}")
            print("=" * 60)
        except Exception as e:
            print("=" * 60)
            print(f"ERROR initializing SAM model: {e}")
            print("Click mode will use simple circular masks as fallback.")
            print("=" * 60)
    
    def segment_from_points(
        self,
        image_path: str,
        points: List[dict],
        output_mask_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment objects from click points using SAM.
        
        Args:
            image_path: Path to the input image
            points: List of dicts with 'x' and 'y' coordinates
            output_mask_path: Optional path to save the mask
        
        Returns:
            Tuple of (original_image, mask)
        """
        if self.predictor is None:
            # Fallback: create simple circular masks around points
            return self._create_simple_mask(image_path, points, output_mask_path)
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for predictor
        self.predictor.set_image(image)
        
        # Convert points to numpy array
        input_points = np.array([[p['x'], p['y']] for p in points])
        input_labels = np.ones(len(points))  # 1 = foreground point
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Select the best mask (highest score)
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
        # Convert mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Save mask if output path provided
        if output_mask_path:
            mask_image = Image.fromarray(mask_uint8)
            mask_image.save(output_mask_path)
        
        return image, mask_uint8
    
    def _create_simple_mask(
        self,
        image_path: str,
        points: List[dict],
        output_mask_path: Optional[str] = None,
        radius: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback method: Create simple circular masks around click points.
        Used when SAM is not available.
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create empty mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw circles around each point
        for point in points:
            x, y = int(point['x']), int(point['y'])
            cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Save mask if output path provided
        if output_mask_path:
            mask_image = Image.fromarray(mask)
            mask_image.save(output_mask_path)
        
        return image, mask
    
    def segment_from_mask(
        self,
        image_path: str,
        mask_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and mask from paths.
        Used for brush mode where mask is already drawn.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the mask image
        
        Returns:
            Tuple of (original_image, mask)
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        return image, mask
    
    def combine_masks(
        self,
        masks: List[np.ndarray]
    ) -> np.ndarray:
        """
        Combine multiple masks into one.
        
        Args:
            masks: List of mask arrays
        
        Returns:
            Combined mask
        """
        if not masks:
            return None
        
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = np.maximum(combined, mask)
        
        return combined


# Singleton instance
_segmentation_service = None

def get_segmentation_service() -> SegmentationService:
    """
    Get or create the segmentation service singleton.
    """
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = SegmentationService()
    return _segmentation_service
