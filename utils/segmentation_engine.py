from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
import rasterio
from rasterio.windows import Window
from ultralytics import SAM
import torch
from tqdm import tqdm
import time

@dataclass
class SegmentationConfig:
    window_size: int
    center_size: int
    stride: int

class SegmentationEngine:
    def __init__(self, model_path: Union[str, Path], config: SegmentationConfig):
        self.model = SAM(model_path)
        self.config = config
        self.current_prompts = None
        self._processing_stats = {}
        self._start_time = None
        self._end_time = None
    
    def segment(self, image_path: Union[str, Path], prompts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        self.current_prompts = prompts
        self._start_time = time.time()
        self._processing_stats = {}
        
        with rasterio.open(image_path) as dataset:
            transform = dataset.transform
            crs = dataset.crs
            width = dataset.width
            height = dataset.height
        
        windows = self._generate_windows(width, height)
        self._processing_stats['total_windows'] = len(windows)
        
        segmented_polygons = []
        for window in tqdm(windows, desc="Processing segmentation windows"):
            window_segments = self._process_window(image_path, window, transform)
            segmented_polygons.extend(window_segments)
        
        if segmented_polygons:
            gdf = gpd.GeoDataFrame(segmented_polygons, crs=crs)
        else:
            gdf = gpd.GeoDataFrame([], columns=['geometry', 'class', 'segment_id', 'window_x', 'window_y'], crs=crs)
        
        self._end_time = time.time()
        self._update_statistics(gdf)
        return gdf
    
    def _generate_windows(self, width: int, height: int) -> List[Window]:
        windows = []
        x_steps = list(range(0, width - self.config.window_size + 1, self.config.stride))
        y_steps = list(range(0, height - self.config.window_size + 1, self.config.stride))
        
        if not x_steps or x_steps[-1] + self.config.window_size < width:
            x_steps.append(width - self.config.window_size)
        if not y_steps or y_steps[-1] + self.config.window_size < height:
            y_steps.append(height - self.config.window_size)
        
        for y in y_steps:
            for x in x_steps:
                window = Window(col_off=x, row_off=y, width=self.config.window_size, height=self.config.window_size)
                windows.append(window)
        
        return windows

    def _process_window(self, image_path: Union[str, Path], window: Window, transform: rasterio.Affine) -> List[Dict]:
        """
        Process a single window for segmentation.

        Args:
            image_path: Path to the source image
            window: Rasterio Window object defining the region to process
            transform: Geospatial transform for the image

        Returns:
            List of dictionaries containing segmented polygons and metadata
        """
        # Get image dimensions and window transform
        with rasterio.open(image_path) as dataset:
            img_width = dataset.width
            img_height = dataset.height
            # Calculate transform for this specific window
            window_transform = rasterio.windows.transform(window, transform)

        offset_x = (self.config.window_size - self.config.center_size) // 2
        offset_y = (self.config.window_size - self.config.center_size) // 2

        # Check if window is at image edges
        is_right_edge = window.col_off + window.width >= img_width
        is_bottom_edge = window.row_off + window.height >= img_height

        if is_right_edge or is_bottom_edge:
            center_size_x = window.width if is_right_edge else self.config.center_size
            center_size_y = window.height if is_bottom_edge else self.config.center_size
            offset_x = (window.width - center_size_x) // 2
            offset_y = (window.height - center_size_y) // 2
        else:
            center_size_x = center_size_y = self.config.center_size

        center_window = Window(
            window.col_off + offset_x,
            window.row_off + offset_y,
            center_size_x,
            center_size_y
        )

        center_bounds = rasterio.windows.bounds(center_window, transform)
        center_polygon = box(*center_bounds)

        if self.current_prompts is None or self.current_prompts.empty:
            return []

        window_prompts = self.current_prompts[self.current_prompts.geometry.intersects(center_polygon)]
        if window_prompts.empty:
            return []

        with rasterio.open(image_path) as dataset:
            window_data = dataset.read(window=window)

        image = np.transpose(window_data, (1, 2, 0))
        if image.dtype != np.uint8:
            image = (255 * (image - image.min()) / (image.ptp() + 1e-8)).astype(np.uint8)

        bounding_boxes = []
        for _, prompt in window_prompts.iterrows():
            bbox = prompt.geometry.bounds
            # Convert geographic coordinates to pixel coordinates relative to window
            window_bbox = [
                rasterio.transform.rowcol(window_transform, bbox[0], bbox[3]),  # Upper left
                rasterio.transform.rowcol(window_transform, bbox[2], bbox[1])  # Lower right
            ]

            bbox_relative = [
                min(window_bbox[0][1], window_bbox[1][1]),  # xmin
                min(window_bbox[0][0], window_bbox[1][0]),  # ymin
                max(window_bbox[0][1], window_bbox[1][1]),  # xmax
                max(window_bbox[0][0], window_bbox[1][0])  # ymax
            ]

            # Clip to ensure coordinates are within window bounds
            bbox_relative = [
                max(0, min(window.width, bbox_relative[0])),
                max(0, min(window.height, bbox_relative[1])),
                max(0, min(window.width, bbox_relative[2])),
                max(0, min(window.height, bbox_relative[3]))
            ]

            bounding_boxes.append(bbox_relative)

        results = self.model.predict(
            source=image,
            imgsz=self.config.window_size,
            bboxes=bounding_boxes,
            show_boxes=False
        )

        segmented_polygons = []
        try:
            masks = results[0].masks.xy
            for i, mask_polygon in enumerate(masks):
                try:
                    # Convert mask coordinates to geographic coordinates
                    polygon_pixel = Polygon(mask_polygon)
                    geographic_coords = []
                    for x, y in polygon_pixel.exterior.coords:
                        # Convert pixel coordinates to geographic coordinates using window transform
                        geo_x, geo_y = rasterio.transform.xy(window_transform, y, x)
                        geographic_coords.append((geo_x, geo_y))

                    polygon_geo = Polygon(geographic_coords)
                    polygon_geo = polygon_geo.intersection(center_polygon)

                    if not polygon_geo.is_empty and polygon_geo.is_valid and polygon_geo.geom_type == 'Polygon':
                        segmented_polygons.append({
                            'geometry': polygon_geo,
                            'window_x': window.col_off,
                            'window_y': window.row_off,
                            'segment_id': f'segment_{window.col_off}_{window.row_off}_{i}',
                            'class': window_prompts.iloc[i]['class'] if i < len(window_prompts) else 'unknown'
                        })
                except Exception:
                    continue
        except Exception:
            pass

        return segmented_polygons
    def _update_statistics(self, final_gdf: gpd.GeoDataFrame) -> None:
        self._processing_stats.update({
            'processing_time': self._end_time - self._start_time,
            'total_segments': len(final_gdf),
            'segments_per_second': len(final_gdf) / (self._end_time - self._start_time),
            'class_distribution': final_gdf['class'].value_counts().to_dict() if len(final_gdf) > 0 else {}
        })
    
    def get_statistics(self) -> Dict:
        return self._processing_stats
    
    def cleanup(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
