from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional
import torch

class YOLOService:
    """
    Service for object detection and segmentation using YOLO
    """
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize YOLOv8 segmentation model.
        """
        try:
            from ultralytics import YOLO
            
            # Use YOLOv8 segmentation model
            # Options: yolov8n-seg.pt (nano), yolov8s-seg.pt (small), yolov8m-seg.pt (medium)
            model_name = "yolov8n-seg.pt"  # Fastest, good for real-time
            
            print("=" * 60)
            print(f"Loading YOLO segmentation model: {model_name}")
            print("This will download the model on first run (~6MB)...")
            
            self.model = YOLO(model_name)
            
            print(f"✅ YOLO model loaded successfully!")
            print("=" * 60)
            
        except ImportError as e:
            print("=" * 60)
            print("WARNING: ultralytics package not installed!")
            print("Install with: pip3 install ultralytics")
            print(f"Error: {e}")
            print("=" * 60)
        except Exception as e:
            print("=" * 60)
            print(f"ERROR initializing YOLO model: {e}")
            print("=" * 60)
    
    def segment_from_click(
        self,
        image_path: str,
        click_point: dict,
        output_mask_path: Optional[str] = None,
        confidence_threshold: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Detect and segment object at click point using YOLO.
        
        Args:
            image_path: Path to the input image
            click_point: Dict with 'x' and 'y' coordinates
            output_mask_path: Optional path to save the mask
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            Tuple of (original_image, mask, detection_info)
        """
        if self.model is None:
            # Fallback: create simple circular mask
            return self._create_simple_mask(image_path, click_point, output_mask_path)
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Run YOLO segmentation
        results = self.model(image_path, conf=confidence_threshold, verbose=False)
        
        # Get click coordinates
        click_x, click_y = int(click_point['x']), int(click_point['y'])
        
        # Find the object that contains the click point
        selected_mask = None
        selected_box = None
        selected_class = None
        selected_conf = 0
        
        # Check if model has segmentation masks
        has_masks = results[0].masks is not None
        
        if has_masks:
            # Use segmentation masks (YOLOv5-seg, YOLOv8-seg)
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confidences)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Check if click point is inside this mask
                if mask_resized[click_y, click_x] > 0.5:
                    # Found the object! Use this mask
                    selected_mask = mask_resized
                    selected_box = box
                    selected_class = int(cls)
                    selected_conf = float(conf)
                    break
        else:
            # Fallback for detection-only models (YOLOv5, YOLOv8 detection)
            # Create masks from bounding boxes
            print("⚠️  Detection-only model detected. Creating masks from bounding boxes.")
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Find box containing the click point
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Check if click is inside this box
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        # Create mask from bounding box
                        selected_mask = np.zeros((h, w), dtype=np.float32)
                        selected_mask[y1:y2, x1:x2] = 1.0
                        selected_box = box
                        selected_class = int(cls)
                        selected_conf = float(conf)
                        break
        
        # If no mask found at click point, find nearest object
        if selected_mask is None and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            min_distance = float('inf')
            nearest_idx = -1
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                # Calculate distance from click to box center
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                distance = np.sqrt((click_x - box_center_x)**2 + (click_y - box_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = i
            
            if nearest_idx >= 0:
                if has_masks:
                    # Use segmentation mask
                    mask = results[0].masks.data[nearest_idx].cpu().numpy()
                    selected_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    # Create mask from bounding box
                    box = boxes[nearest_idx]
                    x1, y1, x2, y2 = map(int, box)
                    selected_mask = np.zeros((h, w), dtype=np.float32)
                    selected_mask[y1:y2, x1:x2] = 1.0
                
                selected_box = boxes[nearest_idx]
                selected_class = int(classes[nearest_idx])
                selected_conf = float(confidences[nearest_idx])
        
        # Create final mask
        if selected_mask is not None:
            mask_uint8 = (selected_mask * 255).astype(np.uint8)
        else:
            # No objects detected, create empty mask
            mask_uint8 = np.zeros((h, w), dtype=np.uint8)
        
        # Save mask if output path provided
        if output_mask_path:
            mask_image = Image.fromarray(mask_uint8)
            mask_image.save(output_mask_path)
        
        # Prepare detection info
        detection_info = {
            "detected": selected_mask is not None,
            "class_id": selected_class if selected_class is not None else -1,
            "class_name": self.model.names[selected_class] if selected_class is not None else "none",
            "confidence": selected_conf,
            "bbox": selected_box.tolist() if selected_box is not None else None
        }
        
        return image_rgb, mask_uint8, detection_info
    
    def segment_multiple_clicks(
        self,
        image_path: str,
        click_points: List[dict],
        output_mask_path: Optional[str] = None,
        confidence_threshold: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Detect and segment multiple objects from multiple click points.
        
        Args:
            image_path: Path to the input image
            click_points: List of dicts with 'x' and 'y' coordinates
            output_mask_path: Optional path to save the combined mask
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            Tuple of (original_image, combined_mask, detection_infos)
        """
        if self.model is None or not click_points:
            return self._create_simple_mask(image_path, click_points[0] if click_points else {"x": 0, "y": 0}, output_mask_path)
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Initialize combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        detection_infos = []
        
        # Process each click point
        for click_point in click_points:
            _, mask, info = self.segment_from_click(
                image_path, 
                click_point, 
                output_mask_path=None,
                confidence_threshold=confidence_threshold
            )
            
            # Combine masks (union)
            combined_mask = np.maximum(combined_mask, mask)
            detection_infos.append(info)
        
        # Save combined mask
        if output_mask_path:
            mask_image = Image.fromarray(combined_mask)
            mask_image.save(output_mask_path)
        
        return image_rgb, combined_mask, detection_infos
    
    def _create_simple_mask(
        self,
        image_path: str,
        click_point: dict,
        output_mask_path: Optional[str] = None,
        radius: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Fallback: Create simple circular mask around click point.
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y = int(click_point['x']), int(click_point['y'])
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        if output_mask_path:
            mask_image = Image.fromarray(mask)
            mask_image.save(output_mask_path)
        
        detection_info = {
            "detected": False,
            "class_id": -1,
            "class_name": "fallback_circle",
            "confidence": 0.0,
            "bbox": None
        }
        
        return image_rgb, mask, detection_info
    
    def detect_all_objects(
        self,
        image_path: str,
        confidence_threshold: float = 0.25
    ) -> dict:
        """
        Detect all objects in the image and return their information.
        
        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            Dict with detected objects and their masks
        """
        if self.model is None:
            return {
                "detected_objects": [],
                "total_count": 0,
                "error": "YOLO model not initialized"
            }
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Run YOLO detection/segmentation
        results = self.model(image_path, conf=confidence_threshold, verbose=False)
        
        detected_objects = []
        has_masks = results[0].masks is not None
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            if has_masks:
                masks = results[0].masks.data.cpu().numpy()
            
            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                
                # Get or create mask for this object
                if has_masks and i < len(masks):
                    # Use segmentation mask
                    mask = masks[i]
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_data = (mask_resized * 255).astype(np.uint8)
                else:
                    # Create mask from bounding box
                    mask_data = np.zeros((h, w), dtype=np.uint8)
                    mask_data[y1:y2, x1:x2] = 255
                
                obj_info = {
                    "id": i,
                    "class_id": int(cls),
                    "class_name": self.model.names[int(cls)],
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                    "mask_data": mask_data.tolist(),  # Convert to list for JSON serialization
                    "has_segmentation_mask": has_masks
                }
                
                detected_objects.append(obj_info)
        
        return {
            "detected_objects": detected_objects,
            "total_count": len(detected_objects),
            "image_size": {"width": w, "height": h},
            "has_segmentation": has_masks
        }


# Singleton instance
_yolo_service = None

def get_yolo_service() -> YOLOService:
    """
    Get or create the YOLO service singleton.
    """
    global _yolo_service
    if _yolo_service is None:
        _yolo_service = YOLOService()
    return _yolo_service
