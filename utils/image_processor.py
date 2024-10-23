from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import geopandas as gpd
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import json
from shapely.geometry import Polygon

from detection_engine import DetectionEngine, WindowConfig
from segmentation_engine import SegmentationEngine, SegmentationConfig
from geometry_processor import GeometryProcessor, CleaningConfig
from config_manager import ConfigManager


class ProcessingState(Enum):
    """Enum defining possible states of the processing pipeline."""
    INITIALIZED = "initialized"
    DETECTING = "detecting"
    DETECTION_COMPLETE = "detection_complete"
    SEGMENTING = "segmenting"
    SEGMENTATION_COMPLETE = "segmentation_complete"
    CLEANING = "cleaning"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProcessingStateManager:
    """Manages processing state and associated data."""
    state: ProcessingState = ProcessingState.INITIALIZED
    detections: Optional[gpd.GeoDataFrame] = None
    segments: Optional[gpd.GeoDataFrame] = None
    current_stage: Optional[str] = None
    error_count: int = 0
    last_successful_stage: Optional[ProcessingState] = None

    def update_state(self, new_state: ProcessingState) -> None:
        """Update the processing state."""
        self.current_stage = new_state.value
        self.state = new_state

        if new_state != ProcessingState.ERROR:
            self.last_successful_stage = new_state

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.state = ProcessingState.ERROR


@dataclass
class ProcessingMetrics:
    """Container for processing metrics across all stages."""
    start_time: datetime
    end_time: Optional[datetime] = None
    detection_time: Optional[float] = None
    segmentation_time: Optional[float] = None
    cleaning_time: Optional[float] = None
    detection_stats: Dict = None
    segmentation_stats: Dict = None
    geometry_stats: Dict = None
    quality_metrics: Dict = None

    def get_total_time(self) -> float:
        """Calculate total processing time in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format."""
        return {
            'total_time': self.get_total_time(),
            'detection_time': self.detection_time,
            'segmentation_time': self.segmentation_time,
            'cleaning_time': self.cleaning_time,
            'detection_stats': self.detection_stats,
            'segmentation_stats': self.segmentation_stats,
            'geometry_stats': self.geometry_stats,
            'quality_metrics': self.quality_metrics
        }


class QualityMetrics:
    """Calculates quality metrics for each processing stage."""
    
    @staticmethod
    def calculate_detection_quality(detections: gpd.GeoDataFrame) -> Dict:
        if detections.empty:
            return {
                'total_detections': 0,
                'classes_detected': {},
                'mean_confidence': 0.0,
                'confidence_std': 0.0,
                'confidence_quartiles': [],
                'area_stats': {}
            }
        
        detections['area'] = detections.geometry.area
        confidence_stats = detections['confidence'].describe()
        
        area_stats = {}
        for class_name in detections['class'].unique():
            class_data = detections[detections['class'] == class_name]
            area_stats[class_name] = {
                'mean_area': float(class_data['area'].mean()),
                'std_area': float(class_data['area'].std()),
                'min_area': float(class_data['area'].min()),
                'max_area': float(class_data['area'].max())
            }
        
        return {
            'total_detections': len(detections),
            'classes_detected': detections['class'].value_counts().to_dict(),
            'mean_confidence': float(confidence_stats['mean']),
            'confidence_std': float(confidence_stats['std']),
            'confidence_quartiles': [
                float(detections['confidence'].quantile(q))
                for q in [0.25, 0.5, 0.75]
            ],
            'area_stats': area_stats
        }

    @staticmethod
    def calculate_segment_quality(segments: gpd.GeoDataFrame) -> Dict:
        if segments.empty:
            return {
                'total_segments': 0,
                'segments_per_class': {},
                'geometry_stats': {},
                'shape_complexity': {}
            }
        
        segments['area'] = segments.geometry.area
        segments['perimeter'] = segments.geometry.length
        segments['compactness'] = (4 * np.pi * segments['area']) / (segments['perimeter'] ** 2)
        
        shape_metrics = {}
        for class_name in segments['class'].unique():
            class_data = segments[segments['class'] == class_name]
            shape_metrics[class_name] = {
                'mean_compactness': float(class_data['compactness'].mean()),
                'std_compactness': float(class_data['compactness'].std()),
                'mean_area': float(class_data['area'].mean()),
                'std_area': float(class_data['area'].std())
            }
        
        return {
            'total_segments': len(segments),
            'segments_per_class': segments['class'].value_counts().to_dict(),
            'geometry_stats': shape_metrics
        }


class ImageProcessor:
    """
    Processes geospatial imagery through object detection, segmentation, and cleaning pipeline.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize processor with configuration."""
        self.config = ConfigManager(config_path).load_config()
        self.image_path = None
        self.output_path = None
        self.state_manager = ProcessingStateManager()
        self.metrics = None
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the processing pipeline."""
        log_path = Path(self.config.paths.output_folder) / "processing.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def detect_objects(self, image_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Perform object detection on an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            GeoDataFrame containing detected objects
        """
        self.image_path = Path(image_path)
        detection_start = datetime.now()
        self.logger.info(f"Starting detection on {self.image_path}")
        
        try:
            # Initialize detection engine
            detection_engine = DetectionEngine(
                model_path=self.config.paths.yolo_model,
                window_config=WindowConfig(
                    height=self.config.yolo.window.height,
                    width=self.config.yolo.window.width,
                    overlap_height_ratio=self.config.yolo.overlap.height_ratio,
                    overlap_width_ratio=self.config.yolo.overlap.width_ratio,
                    confidence_threshold=self.config.yolo.confidence_threshold
                )
            )
            
            # Perform detection
            detections = detection_engine.detect(self.image_path)
            
            # Calculate metrics
            detection_time = (datetime.now() - detection_start).total_seconds()
            detection_stats = detection_engine.get_statistics()
            quality_metrics = QualityMetrics.calculate_detection_quality(detections)
            
            self.logger.info(f"Detection completed: {len(detections)} objects found")
            
            # Clean up
            detection_engine.cleanup()
            
            return detections, detection_time, detection_stats, quality_metrics
            
        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}", exc_info=True)
            raise

    def segment_detections(self, 
                         image_path: Union[str, Path],
                         detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Perform segmentation on detected objects.
        
        Args:
            image_path: Path to input image
            detections: GeoDataFrame of detected objects
            
        Returns:
            GeoDataFrame containing segmented objects
        """
        if detections.empty:
            self.logger.info("No detections to segment")
            return detections, 0, {}, {}
        
        segmentation_start = datetime.now()
        self.logger.info("Starting segmentation")
        
        try:
            # Initialize segmentation engine
            segmentation_engine = SegmentationEngine(
                model_path=self.config.paths.sam_model,
                config=SegmentationConfig(
                    window_size=self.config.sam.window_size,
                    center_size=self.config.sam.center_size,
                    stride=self.config.sam.stride
                )
            )
            
            # Perform segmentation
            segments = segmentation_engine.segment(image_path, detections)
            
            # Calculate metrics
            segmentation_time = (datetime.now() - segmentation_start).total_seconds()
            segmentation_stats = segmentation_engine.get_statistics()
            quality_metrics = QualityMetrics.calculate_segment_quality(segments)
            
            self.logger.info(f"Segmentation completed: {len(segments)} segments created")
            
            # Clean up
            segmentation_engine.cleanup()
            
            return segments, segmentation_time, segmentation_stats, quality_metrics
            
        except Exception as e:
            self.logger.error(f"Segmentation error: {str(e)}", exc_info=True)
            raise

    def clean_segments(self, segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Clean and process segmented objects.
        
        Args:
            segments: GeoDataFrame of segmented objects
            
        Returns:
            GeoDataFrame containing cleaned segments
        """
        if segments.empty:
            return segments, 0, {}
        
        cleaning_start = datetime.now()
        self.logger.info("Starting geometry cleaning")
        
        try:
            cleaned_segments = []
            print(self.config.cleaning)
            cleaning_configs = self.config.cleaning
            cleaning_configs_dict = {}
            for clean_config in cleaning_configs:
                cleaning_configs_dict[clean_config] = cleaning_configs[clean_config]

            for clean_config in cleaning_configs_dict:
                class_segments = segments[segments['class'] == clean_config]
                if not class_segments.empty:
                    processor = GeometryProcessor(cleaning_configs_dict[clean_config])
                    processor.load_geometries(class_segments)
                    cleaned = processor.clean_segments()
                    if not cleaned.empty:
                        cleaned_segments.append(cleaned)
            
            # Combine cleaned segments
            if cleaned_segments:
                final_results = gpd.GeoDataFrame(
                    pd.concat(cleaned_segments, ignore_index=True)
                )
            else:
                final_results = gpd.GeoDataFrame(
                    columns=['geometry', 'class'],
                    crs=segments.crs
                )
            
            cleaning_time = (datetime.now() - cleaning_start).total_seconds()
            
            return final_results, cleaning_time, processor.get_processing_stats()
            
        except Exception as e:
            self.logger.error(f"Cleaning error: {str(e)}", exc_info=True)
            raise

    def process_image(self, 
                     image_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None) -> gpd.GeoDataFrame:
        """
        Process an image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            output_path: Optional path for output results
            
        Returns:
            GeoDataFrame containing final processed objects
        """
        self.metrics = ProcessingMetrics(start_time=datetime.now())
        self.output_path = Path(output_path) if output_path else \
            Path(self.config.paths.output_folder) / f"{Path(image_path).stem}_processed.geojson"
        
        try:
            # Detection phase
            detections, det_time, det_stats, det_quality = self.detect_objects(image_path)
            self.metrics.detection_time = det_time
            self.metrics.detection_stats = det_stats
            
            # Segmentation phase
            segments, seg_time, seg_stats, seg_quality = self.segment_detections(image_path, detections)
            self.metrics.segmentation_time = seg_time
            self.metrics.segmentation_stats = seg_stats
            
            # Cleaning phase
            final_results, clean_time, clean_stats = self.clean_segments(segments)
            self.metrics.cleaning_time = clean_time
            self.metrics.geometry_stats = clean_stats
            
            # Compile quality metrics
            self.metrics.quality_metrics = {
                'detection': det_quality,
                'segmentation': seg_quality
            }
            
            # Save results
            if not final_results.empty:
                final_results.to_file(self.output_path, driver='GeoJSON')
            
            self.metrics.end_time = datetime.now()
            self._log_metrics()
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}", exc_info=True)
            raise

    def _log_metrics(self) -> None:
        """Log detailed processing metrics with ASCII-compatible formatting."""
        metrics_dict = self.metrics.to_dict()

        self.logger.info("\nProcessing Summary:")
        self.logger.info(f"Total Time: {metrics_dict['total_time']:.2f} seconds")
        self.logger.info(f"|- Detection: {metrics_dict['detection_time']:.2f} seconds")
        self.logger.info(f"|- Segmentation: {metrics_dict['segmentation_time']:.2f} seconds")
        self.logger.info(f"`- Cleaning: {metrics_dict['cleaning_time']:.2f} seconds")

        if self.metrics.quality_metrics:
            self.logger.info("\nQuality Metrics:")
            det_metrics = self.metrics.quality_metrics['detection']
            self.logger.info("Detection:")
            self.logger.info(f"|- Objects Found: {det_metrics['total_detections']}")
            self.logger.info(f"|- Mean Confidence: {det_metrics['mean_confidence']:.3f}")
            self.logger.info(f"`- Classes: {det_metrics['classes_detected']}")

            if 'segmentation' in self.metrics.quality_metrics:
                seg_metrics = self.metrics.quality_metrics['segmentation']
                self.logger.info("\nSegmentation:")
                self.logger.info(f"|- Total Segments: {seg_metrics['total_segments']}")
                for cls, stats in seg_metrics['geometry_stats'].items():
                    self.logger.info(f"`- {cls}:")
                    self.logger.info(f"   |- Mean Area: {stats['mean_area']:.1f}")
                    self.logger.info(f"   `- Mean Compactness: {stats['mean_compactness']:.3f}")

    def get_state(self) -> Dict:
        """
        Get current processing state and metrics.

        Returns:
            Dictionary containing current state and metrics
        """
        return {
            'image_path': str(self.image_path) if self.image_path else None,
            'output_path': str(self.output_path) if self.output_path else None,
            'metrics': self.metrics.to_dict() if self.metrics else None
        }