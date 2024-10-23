from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from ultralytics import YOLO
import torch
from tqdm import tqdm

@dataclass
class WindowConfig:
    """Configuration for sliding window detection."""
    height: int
    width: int
    overlap_height_ratio: float
    overlap_width_ratio: float
    confidence_threshold: float

class DetectionEngine:
    """
    Handles YOLO-based detection operations for geospatial imagery.
    
    Properties:
        model: YOLO model instance
        window_config: Sliding window configuration
        _processing_stats: Dictionary tracking detection statistics
    """
    
    def __init__(self, model_path: Union[str, Path], window_config: WindowConfig):
        """
        Initialize DetectionEngine with model and configuration.
        
        Args:
            model_path: Path to YOLO model weights
            window_config: WindowConfig instance with detection parameters
        """
        self.model = YOLO(model_path)
        self.window_config = window_config
        self._processing_stats = {}
        self._start_time = None
        self._end_time = None
    
    def detect(self, image_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Perform object detection on a geospatial image.
        
        Args:
            image_path: Path to georeferenced image file
            
        Returns:
            GeoDataFrame containing detected objects with geometries
        """
        self._start_time = time.time()
        self._processing_stats = {}
        
        # Read image and get geospatial information
        with rasterio.open(image_path) as dataset:
            img_data = dataset.read()  # Shape: (bands, height, width)
            transform = dataset.transform
            crs = dataset.crs
            img_height, img_width = img_data.shape[1], img_data.shape[2]
        
        # Generate sliding windows
        windows = self._generate_windows(img_height, img_width)
        self._processing_stats['total_windows'] = len(windows)
        
        # Process windows and collect detections
        detections = []
        for window in tqdm(windows, desc="Processing detection windows"):
            window_detections = self._process_window(
                img_data, window, transform
            )
            detections.extend(window_detections)
        
        # Merge overlapping detections
        final_detections = self._merge_detections(detections)
        
        # Create GeoDataFrame
        if final_detections:
            gdf = gpd.GeoDataFrame(
                final_detections,
                crs=crs
            )
        else:
            gdf = gpd.GeoDataFrame(
                [],
                columns=['geometry', 'class', 'confidence'],
                crs=crs
            )
        
        self._end_time = time.time()
        self._update_statistics(gdf)
        
        return gdf
    
    def _generate_windows(self, img_height: int, img_width: int) -> List[Dict]:
        """
        Generate sliding window coordinates.
        
        Args:
            img_height: Image height in pixels
            img_width: Image width in pixels
            
        Returns:
            List of window dictionaries with coordinates
        """
        stride_height = int(self.window_config.height * (1 - self.window_config.overlap_height_ratio))
        stride_width = int(self.window_config.width * (1 - self.window_config.overlap_width_ratio))
        
        windows = []
        
        # Generate y coordinates
        y_positions = list(range(0, img_height - self.window_config.height + 1, stride_height))
        if y_positions and y_positions[-1] + self.window_config.height < img_height:
            y_positions.append(img_height - self.window_config.height)
            
        # Generate x coordinates
        x_positions = list(range(0, img_width - self.window_config.width + 1, stride_width))
        if x_positions and x_positions[-1] + self.window_config.width < img_width:
            x_positions.append(img_width - self.window_config.width)
        
        # Create windows
        for y in y_positions:
            for x in x_positions:
                windows.append({
                    'x': x,
                    'y': y,
                    'width': self.window_config.width,
                    'height': self.window_config.height
                })
        
        return windows
    
    def _process_window(self, 
                       img_data: np.ndarray, 
                       window: Dict, 
                       transform: rasterio.Affine) -> List[Dict]:
        """
        Process a single detection window.
        
        Args:
            img_data: Full image data
            window: Window coordinates dictionary
            transform: Geospatial transform
            
        Returns:
            List of detection dictionaries
        """
        # Extract window data
        window_data = img_data[:, 
                             window['y']:window['y'] + window['height'],
                             window['x']:window['x'] + window['width']]
        
        # Transpose to HWC format for YOLO
        window_data = np.transpose(window_data, (1, 2, 0))
        
        # Ensure uint8 format
        if window_data.dtype != np.uint8:
            window_data = (255 * (window_data - window_data.min()) / 
                         (window_data.ptp() + 1e-8)).astype(np.uint8)
        
        # Run detection
        results = self.model(
            window_data,
            augment=True,
            conf=self.window_config.confidence_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                class_names = [self.model.names[int(cls_id)] for cls_id in class_ids]
                
                # Process each detection
                for box, confidence, class_name in zip(boxes, confidences, class_names):
                    x1, y1, x2, y2 = box
                    
                    # Adjust coordinates to original image space
                    x1 += window['x']
                    x2 += window['x']
                    y1 += window['y']
                    y2 += window['y']
                    
                    # Convert to geographic coordinates
                    top_left = rasterio.transform.xy(transform, y1, x1, offset='center')
                    bottom_right = rasterio.transform.xy(transform, y2, x2, offset='center')
                    
                    # Create polygon coordinates
                    polygon_coords = [
                        [top_left[0], top_left[1]],
                        [top_left[0], bottom_right[1]],
                        [bottom_right[0], bottom_right[1]],
                        [bottom_right[0], top_left[1]],
                        [top_left[0], top_left[1]]
                    ]
                    
                    detections.append({
                        'geometry': Polygon(polygon_coords),
                        'class': class_name,
                        'confidence': float(confidence)
                    })
        
        return detections
    
    def _merge_detections(self, 
                         detections: List[Dict],
                         iou_threshold: float = 0.5) -> List[Dict]:
        """
        Merge overlapping detections using NMS-like approach.
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for merging
            
        Returns:
            List of merged detection dictionaries
        """
        if not detections:
            return []
        
        # Group detections by class
        class_detections = {}
        for detection in detections:
            class_name = detection['class']
            if class_name not in class_detections:
                class_detections[class_name] = []
            class_detections[class_name].append(detection)
        
        merged_detections = []
        
        # Process each class separately
        for class_name, class_dets in class_detections.items():
            # Sort by confidence
            class_dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            while class_dets:
                best_det = class_dets.pop(0)
                best_geom = best_det['geometry']
                
                # Find overlapping detections
                i = 0
                while i < len(class_dets):
                    curr_det = class_dets[i]
                    curr_geom = curr_det['geometry']
                    
                    if best_geom.intersects(curr_geom):
                        intersection = best_geom.intersection(curr_geom)
                        union = best_geom.union(curr_geom)
                        iou = intersection.area / union.area
                        
                        if iou >= iou_threshold:
                            # Merge detections
                            best_det['geometry'] = union
                            best_det['confidence'] = max(
                                best_det['confidence'],
                                curr_det['confidence']
                            )
                            class_dets.pop(i)
                            continue
                    i += 1
                
                merged_detections.append(best_det)
        
        return merged_detections
    
    def _update_statistics(self, final_gdf: gpd.GeoDataFrame) -> None:
        """Update processing statistics after detection."""
        self._processing_stats.update({
            'processing_time': self._end_time - self._start_time,
            'total_detections': len(final_gdf),
            'detections_per_second': len(final_gdf) / (self._end_time - self._start_time),
            'class_distribution': final_gdf['class'].value_counts().to_dict() if len(final_gdf) > 0 else {},
            'mean_confidence': float(final_gdf['confidence'].mean()) if len(final_gdf) > 0 else 0.0
        })
    
    def get_statistics(self) -> Dict:
        """Get detection processing statistics."""
        return self._processing_stats
    
    def cleanup(self) -> None:
        """Clean up resources and GPU memory."""
        self.model = None
        torch.cuda.empty_cache()
