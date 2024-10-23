import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import json
import time
import numpy as np

@dataclass
class CleaningConfig:
    """Configuration for geometry cleaning operations."""
    simplification_tolerance: float
    overlap_threshold: float
    min_area: float
    max_area: float
    convexity_threshold: float
    final_convexity_threshold: float

class GeometryProcessor:
    """
    Handles geometric operations and cleaning for detected objects in geospatial data.
    Provides enhanced monitoring and metrics without parallel processing complexity.
    """
    
    def __init__(self, clean_config: CleaningConfig):
        self.clean_config = clean_config
        self.current_geometries = None
        self._processing_stats = {}
        self._start_time = None
        self._end_time = None
    
    def load_geometries(self, geometries: Union[gpd.GeoDataFrame, str, List[str]]) -> None:
        """
        Load geometries from a GeoDataFrame or file(s).
        
        Args:
            geometries: GeoDataFrame or path(s) to GeoJSON file(s)
        """
        if isinstance(geometries, gpd.GeoDataFrame):
            self.current_geometries = geometries
        elif isinstance(geometries, (str, list)):
            files = [geometries] if isinstance(geometries, str) else geometries
            gdfs = []
            for file in files:
                gdf = gpd.read_file(file)
                gdfs.append(gdf)
            self.current_geometries = pd.concat(gdfs, ignore_index=True)
            
        self._processing_stats = {
            'input_count': len(self.current_geometries),
            'input_classes': self.current_geometries['class'].value_counts().to_dict() 
                           if 'class' in self.current_geometries.columns else {}
        }
    
    def clean_segments(self, class_name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Clean and process geometric segments with enhanced monitoring.
        
        Args:
            class_name: Optional class name to filter geometries
            
        Returns:
            GeoDataFrame with cleaned geometries
        """
        self._start_time = time.time()
        
        if self.current_geometries is None:
            raise ValueError("No geometries loaded. Call load_geometries() first.")
        
        gdf = self.current_geometries.copy()
        initial_count = len(gdf)
        
        # Filter by class if specified
        if class_name:
            gdf = gdf[gdf['class'] == class_name]
        
        # Track statistics through the cleaning process
        stats = {'initial_count': initial_count}
        
        # Fix invalid geometries
        gdf['geometry'] = gdf['geometry'].buffer(0)
        gdf['geometry'] = gdf['geometry'].simplify(
            self.clean_config.simplification_tolerance, 
            preserve_topology=True
        )
        
        # Remove invalid geometries
        gdf = gdf[gdf.geometry.is_valid]
        stats['after_cleaning'] = len(gdf)
        
        # Size filtering
        gdf['area'] = gdf.geometry.area
        gdf = gdf[
            (gdf['area'] >= self.clean_config.min_area) & 
            (gdf['area'] <= self.clean_config.max_area)
        ]
        stats['after_size_filter'] = len(gdf)
        
        # Convexity filtering
        gdf['convexity'] = gdf.geometry.apply(self._calculate_convexity)
        gdf = gdf[gdf['convexity'] <= self.clean_config.convexity_threshold]
        stats['after_convexity'] = len(gdf)
        
        # Overlap reduction
        gdf = self._reduce_overlaps(gdf)
        stats['final_count'] = len(gdf)
        
        self._end_time = time.time()
        self._processing_stats.update(stats)
        self._processing_stats['processing_time'] = self._end_time - self._start_time
        
        return gdf

    def _reduce_overlaps(self, gdf: gpd.GeoDataFrame, max_iterations: int = 3) -> gpd.GeoDataFrame:
        """
        Iteratively reduce overlapping geometries with proper index handling.

        Args:
            gdf: GeoDataFrame containing geometries to process
            max_iterations: Maximum number of iteration cycles

        Returns:
            GeoDataFrame with reduced overlaps and proper indexing
        """
        if len(gdf) == 0:
            return gdf

        iteration = 0
        total_merges = 0

        while iteration < max_iterations:
            # Create fresh spatial index for current geometries
            gdf = gdf.reset_index(drop=True)  # Reset index to ensure alignment
            sindex = gdf.sindex
            to_remove = set()
            merges_this_iteration = 0

            # Process each geometry
            for idx, row in tqdm(gdf.iterrows(), desc=f"Reducing overlaps - Iteration {iteration + 1}"):
                if idx in to_remove:
                    continue

                current_geom = row.geometry
                current_bounds = current_geom.bounds

                # Query spatial index for potential matches
                possible_matches_idx = list(sindex.intersection(current_bounds))

                # Filter out self and already processed geometries
                possible_matches_idx = [
                    match_idx for match_idx in possible_matches_idx
                    if match_idx != idx and match_idx not in to_remove
                ]

                # Sort matches by overlap quality
                match_overlaps = []
                for match_idx in possible_matches_idx:
                    if match_idx >= len(gdf):  # Safety check
                        continue

                    geom2 = gdf.iloc[match_idx].geometry
                    if current_geom.intersects(geom2):
                        intersection = current_geom.intersection(geom2)
                        if not intersection.is_empty:
                            overlap_area = intersection.area
                            min_area = min(current_geom.area, geom2.area)
                            overlap_ratio = overlap_area / min_area

                            if overlap_ratio >= self.clean_config.overlap_threshold:
                                # Calculate merge priority
                                merged = current_geom.union(geom2)
                                priority = self._calculate_merge_priority(
                                    current_geom,
                                    geom2,
                                    merged
                                )
                                match_overlaps.append((match_idx, priority))

                # Sort by priority score
                match_overlaps.sort(key=lambda x: x[1], reverse=True)

                # Merge geometries with best matches first
                if match_overlaps:
                    merged_geom = current_geom
                    for match_idx, _ in match_overlaps:
                        if match_idx in to_remove:
                            continue

                        geom2 = gdf.iloc[match_idx].geometry

                        # Recheck overlap after previous merges
                        intersection = merged_geom.intersection(geom2)
                        if not intersection.is_empty:
                            overlap_area = intersection.area
                            min_area = min(merged_geom.area, geom2.area)
                            overlap_ratio = overlap_area / min_area

                            if overlap_ratio >= self.clean_config.overlap_threshold:
                                merged = merged_geom.union(geom2)
                                if self._validate_merged_geometry(merged):
                                    merged_geom = merged
                                    to_remove.add(match_idx)
                                    merges_this_iteration += 1

                    if merged_geom != current_geom:
                        gdf.loc[idx, 'geometry'] = merged_geom

            # Remove merged geometries
            if to_remove:
                gdf = gdf.drop(index=list(to_remove)).reset_index(drop=True)

            # Update statistics
            total_merges += merges_this_iteration

            # Break if no merges occurred
            if merges_this_iteration == 0:
                break

            iteration += 1

        # Final index reset and cleanup
        gdf = gdf.reset_index(drop=True)

        # Update processing statistics
        self._processing_stats.update({
            'total_merges': total_merges,
            'merge_iterations': iteration,
            'final_geometry_count': len(gdf)
        })

        return gdf

    def _calculate_merge_priority(self, geom1: Polygon, geom2: Polygon, merged: Polygon) -> float:
        """
        Calculate priority score for merging two geometries.

        Args:
            geom1: First geometry
            geom2: Second geometry
            merged: Result of merging geom1 and geom2

        Returns:
            Float priority score (higher is better)
        """
        # Safety checks
        if not merged.is_valid or merged.is_empty:
            return 0.0

        # Calculate overlap ratio
        intersection = geom1.intersection(geom2)
        overlap_ratio = intersection.area / min(geom1.area, geom2.area)

        # Calculate compactness change
        orig_perimeters = geom1.length + geom2.length
        merged_perimeter = merged.length
        perimeter_ratio = merged_perimeter / orig_perimeters

        # Calculate convexity
        convexity = merged.area / merged.convex_hull.area

        # Combine metrics (weights can be adjusted)
        priority = (
                overlap_ratio * 0.4 +
                (1 - perimeter_ratio) * 0.3 +
                convexity * 0.3
        )

        return priority

    def _validate_merged_geometry(self, geometry: Polygon) -> bool:
        """
        Validate a merged geometry meets quality criteria.

        Args:
            geometry: Geometry to validate

        Returns:
            Boolean indicating if geometry is valid
        """
        if not geometry.is_valid or geometry.is_empty:
            return False

        # Check convexity
        if geometry.area / geometry.convex_hull.area < 0.5:  # Too concave
            return False

        # Check for thin sections using buffer operations
        buffer_size = np.sqrt(geometry.area) * 0.05  # 5% of sqrt area
        cleaned = geometry.buffer(-buffer_size).buffer(buffer_size)

        if cleaned.is_empty:
            return False

        if cleaned.area / geometry.area < 0.9:  # More than 10% area in thin sections
            return False

        return True
    @staticmethod
    def _calculate_convexity(polygon: Polygon) -> float:
        """Calculate the convexity ratio of a polygon."""
        if polygon.is_empty or not polygon.is_valid:
            return float('inf')
        
        hull_area = polygon.convex_hull.area
        if polygon.area == 0:
            return float('inf')
        
        return float(hull_area / polygon.area)
    
    def get_processing_stats(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Get detailed statistics about the cleaning process.
        
        Returns:
            Dictionary containing processing statistics and metrics
        """
        stats = self._processing_stats.copy()
        
        if self._start_time and self._end_time:
            stats.update({
                'processing_rate': stats['initial_count'] / max(1, self._end_time - self._start_time),
                'reduction_ratio': stats['final_count'] / max(1, stats['initial_count'])
            })
            
        if 'after_size_filter' in stats:
            stats['size_rejection_rate'] = 1 - (stats['after_size_filter'] / max(1, stats['after_cleaning']))
            
        if 'after_convexity' in stats:
            stats['convexity_rejection_rate'] = 1 - (stats['after_convexity'] / max(1, stats['after_size_filter']))
            
        return stats
    
    def export_metrics(self, output_path: Optional[str] = None) -> Dict:
        """
        Export detailed metrics about the geometries and cleaning process.
        
        Args:
            output_path: Optional path to save metrics JSON
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'processing_stats': self.get_processing_stats(),
            'geometry_metrics': self.calculate_metrics(),
            'config_summary': {
                'min_area': self.clean_config.min_area,
                'max_area': self.clean_config.max_area,
                'overlap_threshold': self.clean_config.overlap_threshold,
                'convexity_threshold': self.clean_config.convexity_threshold
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return metrics
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate geometric metrics for current geometries."""
        if self.current_geometries is None or len(self.current_geometries) == 0:
            return {}
        
        metrics = {
            'total_objects': len(self.current_geometries),
            'total_area': float(self.current_geometries.geometry.area.sum()),
            'mean_area': float(self.current_geometries.geometry.area.mean()),
            'mean_convexity': float(self.current_geometries['convexity'].mean() 
                                  if 'convexity' in self.current_geometries.columns 
                                  else float('nan'))
        }
        
        if 'class' in self.current_geometries.columns:
            for class_name in self.current_geometries['class'].unique():
                class_data = self.current_geometries[self.current_geometries['class'] == class_name]
                metrics[f'{class_name}_count'] = len(class_data)
                metrics[f'{class_name}_total_area'] = float(class_data.geometry.area.sum())
        
        return metrics

