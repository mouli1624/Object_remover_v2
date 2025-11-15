from pathlib import Path
import numpy as np
from PIL import Image
import torch
from typing import Dict, Optional
import base64
import io
from .sam_tools.interact_tools import SamControler


class SAMService:
    """
    Service for interactive SAM-based segmentation with session management
    """
    
    def __init__(self):
        self.model = None
        self.sessions: Dict[str, dict] = {}
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize SAM model. Uses vit_h for best quality.
        """
        try:
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
                print("SAM interactive mode will not be available.")
                print("=" * 60)
                return
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("=" * 60)
            print(f"Loading SAM model for interactive mode from: {checkpoint_path}")
            print(f"Device: {device}")
            print("This may take a few seconds...")
            
            model_type = "vit_h"
            self.model = SamControler(checkpoint_path, model_type, device)
            
            print(f"✅ SAM interactive model loaded successfully!")
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
            print("SAM interactive mode will not be available.")
            print("=" * 60)
    
    def upload_image(self, image_data: str, session_id: str = "default") -> dict:
        """
        Upload and initialize image for SAM segmentation
        
        Args:
            image_data: Base64 encoded image data
            session_id: Session identifier
            
        Returns:
            Dictionary with success status and image size
        """
        if self.model is None:
            raise Exception("SAM model not initialized")
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        image_np = np.array(image)
        image_size = (image_np.shape[0], image_np.shape[1])
        
        # Initialize session
        self.sessions[session_id] = {
            'origin_image': image_np,
            'painted_image': image_np.copy(),
            'mask': np.zeros((image_size[0], image_size[1]), np.uint8),
            'masks': [],
            'mask_names': [],
            'click_state': [[], []]
        }
        
        # Reset SAM
        self.model.sam_controler.reset_image()
        self.model.sam_controler.set_image(image_np)
        
        return {
            'success': True,
            'image_size': image_size,
            'message': 'Image uploaded successfully'
        }
    
    def add_point(self, session_id: str, x: int, y: int, is_positive: bool = True) -> dict:
        """
        Add a point to the current segmentation
        
        Args:
            session_id: Session identifier
            x: X coordinate
            y: Y coordinate
            is_positive: True for foreground, False for background
            
        Returns:
            Dictionary with painted image
        """
        try:
            if self.model is None:
                raise Exception("SAM model not initialized")
            
            if session_id not in self.sessions:
                raise Exception(f'No image uploaded for session: {session_id}. Please upload an image first.')
            
            state = self.sessions[session_id]
            state['click_state'][0].append([x, y])
            state['click_state'][1].append(1 if is_positive else 0)
            
            print(f"Adding point: ({x}, {y}), positive={is_positive}")
            print(f"Total points: {len(state['click_state'][0])}")
            
            # Generate mask with SAM
            self.model.sam_controler.reset_image()
            self.model.sam_controler.set_image(state['origin_image'])
            
            mask, logit, painted_image = self.model.first_frame_click(
                image=state['origin_image'],
                points=np.array(state['click_state'][0]),
                labels=np.array(state['click_state'][1]),
                multimask=True
            )
            
            state['mask'] = mask
            state['logit'] = logit
            state['painted_image'] = np.array(painted_image) if isinstance(painted_image, Image.Image) else painted_image
            
            # Convert painted image to base64
            buffered = io.BytesIO()
            painted_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                'success': True,
                'painted_image': f'data:image/png;base64,{img_str}'
            }
        except Exception as e:
            print(f"Error in add_point: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def clear_clicks(self, session_id: str) -> dict:
        """
        Clear all clicks and return to original image
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with original image
        """
        if session_id not in self.sessions:
            raise Exception('No image uploaded for this session')
        
        state = self.sessions[session_id]
        state['click_state'] = [[], []]
        state['mask'] = np.zeros_like(state['mask'])
        
        # Return original image
        origin_pil = Image.fromarray(state['origin_image'].astype('uint8'))
        buffered = io.BytesIO()
        origin_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'success': True,
            'image': f'data:image/png;base64,{img_str}'
        }
    
    def add_mask(self, session_id: str) -> dict:
        """
        Add current mask to the mask collection
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with updated image showing all masks
        """
        if session_id not in self.sessions:
            raise Exception('No image uploaded for this session')
        
        state = self.sessions[session_id]
        mask = state['mask']
        state['masks'].append(mask)
        state['mask_names'].append(f"mask_{len(state['masks']):03d}")
        
        # Show all masks
        from .sam_tools.painter import mask_painter
        select_frame = state['origin_image'].copy()
        for i, mask in enumerate(state['masks']):
            select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=i+2)
        
        # Clear click state
        state['click_state'] = [[], []]
        
        # Convert to base64
        frame_pil = Image.fromarray(select_frame.astype('uint8'))
        buffered = io.BytesIO()
        frame_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'success': True,
            'image': f'data:image/png;base64,{img_str}',
            'mask_names': state['mask_names']
        }
    
    def delete_masks(self, session_id: str) -> dict:
        """
        Delete all masks and return to original image
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with original image
        """
        if session_id not in self.sessions:
            raise Exception('No image uploaded for this session')
        
        state = self.sessions[session_id]
        state['masks'] = []
        state['mask_names'] = []
        state['click_state'] = [[], []]
        state['mask'] = np.zeros_like(state['mask'])
        
        # Return original image
        origin_pil = Image.fromarray(state['origin_image'].astype('uint8'))
        buffered = io.BytesIO()
        origin_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'success': True,
            'image': f'data:image/png;base64,{img_str}',
            'mask_names': []
        }
    
    def get_combined_mask(self, session_id: str) -> Optional[np.ndarray]:
        """
        Get the combined mask for the session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Combined mask as numpy array or None
        """
        if session_id not in self.sessions:
            return None
        
        state = self.sessions[session_id]
        
        # Combine masks - only use saved masks, not current mask
        if state['masks']:
            template_mask = state['masks'][0] * 1
            for i in range(1, len(state['masks'])):
                template_mask = np.clip(template_mask + state['masks'][i] * (i + 1), 0, i + 1)
            return template_mask
        else:
            # No saved masks - return None instead of current mask
            return None


# Singleton instance
_sam_service = None

def get_sam_service() -> SAMService:
    """
    Get or create the SAM service singleton.
    """
    global _sam_service
    if _sam_service is None:
        _sam_service = SAMService()
    return _sam_service
