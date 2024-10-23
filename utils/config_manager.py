from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import yaml
from schema import Schema, And, Use, Optional as SchemaOptional

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Any, Optional


@dataclass
class PathsConfig:
    tif_folder: str
    output_folder: str
    detections_out_name: str
    final_out_name: str
    yolo_model: str
    sam_model: str


@dataclass
class WindowConfig:
    height: int
    width: int


@dataclass
class OverlapConfig:
    height_ratio: float
    width_ratio: float


@dataclass
class YoloConfig:
    window: WindowConfig
    overlap: OverlapConfig
    confidence_threshold: float


@dataclass
class SamConfig:
    window_size: int
    center_size: int
    stride: int


@dataclass
class DetectionPassConfig:
    pass_id: int
    class_name: str
    eps: float
    min_samples: int


@dataclass
class DetectionConfig:
    passes: List[DetectionPassConfig]


@dataclass
class CleaningClassConfig:
    simplification_tolerance: float
    overlap_threshold: float
    min_area: float
    max_area: float
    convexity_threshold: float
    final_convexity_threshold: float


@dataclass
class ProcessingConfig:
    segmentation_cycles: int
    debug_mode: bool


@dataclass
class Config:
    paths: PathsConfig
    yolo: YoloConfig
    sam: SamConfig
    detection: DetectionConfig
    cleaning: Dict[str, CleaningClassConfig]
    processing: ProcessingConfig

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'Config':
        """
        Create Config instance from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config instance with loaded configuration
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert paths section
        paths = PathsConfig(
            tif_folder=config_dict['paths']['tif_folder'],
            output_folder=config_dict['paths']['output_folder'],
            detections_out_name=config_dict['paths']['detections_out_name'],
            final_out_name=config_dict['paths']['final_out_name'],
            yolo_model=config_dict['paths']['yolo_model'],
            sam_model=config_dict['paths']['sam_model']
        )

        # Convert YOLO section
        yolo = YoloConfig(
            window=WindowConfig(
                height=config_dict['yolo']['window']['height'],
                width=config_dict['yolo']['window']['width']
            ),
            overlap=OverlapConfig(
                height_ratio=config_dict['yolo']['overlap']['height_ratio'],
                width_ratio=config_dict['yolo']['overlap']['width_ratio']
            ),
            confidence_threshold=config_dict['yolo']['confidence_threshold']
        )

        # Convert SAM section
        sam = SamConfig(
            window_size=config_dict['sam']['window_size'],
            center_size=config_dict['sam']['center_size'],
            stride=config_dict['sam']['stride']
        )

        # Convert detection passes
        detection_passes = []
        for pass_config in config_dict['detection']['passes']:
            detection_passes.append(
                DetectionPassConfig(
                    pass_id=pass_config['pass'],
                    class_name=pass_config['class'],
                    eps=pass_config['eps'],
                    min_samples=pass_config['min_samples']
                )
            )
        detection = DetectionConfig(passes=detection_passes)

        # Convert cleaning configs
        cleaning = {}
        for class_name, clean_config in config_dict['cleaning'].items():
            cleaning[class_name] = CleaningClassConfig(
                simplification_tolerance=clean_config['simplification_tolerance'],
                overlap_threshold=clean_config['overlap_threshold'],
                min_area=clean_config['min_area'],
                max_area=clean_config['max_area'],
                convexity_threshold=clean_config['convexity_threshold'],
                final_convexity_threshold=clean_config['final_convexity_threshold']
            )

        # Convert processing section
        processing = ProcessingConfig(
            segmentation_cycles=config_dict['processing']['segmentation_cycles'],
            debug_mode=config_dict['processing']['debug_mode']
        )

        return cls(
            paths=paths,
            yolo=yolo,
            sam=sam,
            detection=detection,
            cleaning=cleaning,
            processing=processing
        )

    def to_dict(self) -> Dict:
        """Convert Config instance to dictionary."""
        return {
            'paths': asdict(self.paths),
            'yolo': {
                'window': asdict(self.yolo.window),
                'overlap': asdict(self.yolo.overlap),
                'confidence_threshold': self.yolo.confidence_threshold
            },
            'sam': asdict(self.sam),
            'detection': {
                'passes': [asdict(pass_config) for pass_config in self.detection.passes]
            },
            'cleaning': {
                class_name: asdict(config)
                for class_name, config in self.cleaning.items()
            },
            'processing': asdict(self.processing)
        }
class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigManager:
    """
    Manages configuration loading, validation, and updates for geospatial detection pipeline.

    Properties:
        config_path (Path): Path to the configuration file
        current_config (Config): Currently loaded configuration
    """

    DEFAULT_CONFIG = {
        'paths': {
            'tif_folder': './input',
            'output_folder': './output',
            'detections_out_name': '_detections.geojson',
            'final_out_name': '_objects.geojson',
            'yolo_model': './models/best.pt',
            'sam_model': './models/sam2_l.pt'
        },
        'yolo': {
            'window': {
                'height': 640,
                'width': 640
            },
            'overlap': {
                'height_ratio': 0.1,
                'width_ratio': 0.1
            },
            'confidence_threshold': 0.4
        },
        'sam': {
            'window_size': 1024,
            'center_size': 1024,
            'stride': 600
        },
        'detection': {
            'passes': [
                {
                    'pass': 0,
                    'class': 'object',
                    'eps': 0.25,
                    'min_samples': 2
                },
                {
                    'pass': 1,
                    'class': 'object',
                    'eps': 0.15,
                    'min_samples': 1
                }
            ]
        },
        'cleaning': {
            'object': {
                'simplification_tolerance': 0.15,
                'overlap_threshold': 0.25,
                'min_area': 1,
                'max_area': 32,
                'convexity_threshold': 1.15,
                'final_convexity_threshold': 40
            }
        },
        'processing': {
            'segmentation_cycles': 3,
            'debug_mode': False
        }
    }

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.current_config = None
        self._schema = self._create_validation_schema()

    def _create_validation_schema(self) -> Schema:
        """Creates the schema for configuration validation."""
        return Schema({
            'paths': {
                'tif_folder': And(str, len),
                'output_folder': And(str, len),
                'detections_out_name': And(str, len),
                'final_out_name': And(str, len),
                'yolo_model': And(str, len),
                'sam_model': And(str, len)
            },
            'yolo': {
                'window': {
                    'height': And(int, lambda x: x > 0),
                    'width': And(int, lambda x: x > 0)
                },
                'overlap': {
                    'height_ratio': And(float, lambda x: 0 <= x <= 1),
                    'width_ratio': And(float, lambda x: 0 <= x <= 1)
                },
                'confidence_threshold': And(float, lambda x: 0 <= x <= 1)
            },
            'sam': {
                'window_size': And(int, lambda x: x > 0),
                'center_size': And(int, lambda x: x > 0),
                'stride': And(int, lambda x: x > 0)
            },
            'detection': {
                'passes': [
                    {
                        'pass': And(int, lambda x: x >= 0),
                        'class': str,
                        'eps': And(float, lambda x: x > 0),
                        'min_samples': And(int, lambda x: x > 0)
                    }
                ]
            },
            'cleaning': {
                str: {
                    'simplification_tolerance': And(float, lambda x: x > 0),
                    'overlap_threshold': And(float, lambda x: 0 <= x <= 1),
                    'min_area': And(float, lambda x: x >= 0),
                    'max_area': And(float, lambda x: x > 0),
                    'convexity_threshold': And(float, lambda x: x > 1),
                    'final_convexity_threshold': And(float, lambda x: x > 1)
                }
            },
            'processing': {
                'segmentation_cycles': And(int, lambda x: x > 0),
                'debug_mode': bool
            }
        })

    def load_config(self) -> 'Config':
        """
        Loads and validates configuration from the YAML file.

        Returns:
            Config: Validated configuration object

        Raises:
            ConfigValidationError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            # Validate configuration
            self.validate_config(config_dict)

            # Convert to Config object
            self.current_config = Config.from_yaml(self.config_path)
            return self.current_config

        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Error parsing YAML file: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Error loading configuration: {e}")

    def validate_config(self, config_dict: Dict) -> bool:
        """
        Validates configuration dictionary against schema.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            bool: True if valid

        Raises:
            ConfigValidationError: If configuration is invalid
        """
        try:
            self._schema.validate(config_dict)
            return True
        except Exception as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Updates current configuration with new values.

        Args:
            updates: Dictionary of configuration updates

        Raises:
            ConfigValidationError: If updates would make config invalid
        """
        if self.current_config is None:
            raise ConfigValidationError("No configuration loaded to update")

        # Convert current config to dict
        current_dict = asdict(self.current_config)

        # Apply updates
        self._deep_update(current_dict, updates)

        # Validate updated config
        self.validate_config(current_dict)

        # Save updates to file
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(current_dict, f)

        # Reload config
        self.current_config = self.load_config()

    def _deep_update(self, base_dict: Dict, updates: Dict) -> None:
        """Recursively updates nested dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    @property
    def model_configs(self) -> Dict:
        """Returns dictionary of current model configurations."""
        if self.current_config is None:
            raise ConfigValidationError("No configuration loaded")
        return {
            'yolo': self.current_config.yolo,
            'sam': self.current_config.sam
        }

    @property
    def processing_configs(self) -> Dict:
        """Returns dictionary of current processing configurations."""
        if self.current_config is None:
            raise ConfigValidationError("No configuration loaded")
        return {
            'detection': self.current_config.detection,
            'cleaning': self.current_config.cleaning,
            'processing': self.current_config.processing
        }

    def get_class_configs(self, class_name: str) -> Dict:
        """
        Returns configuration for specific class.

        Args:
            class_name: Name of class to get configuration for

        Returns:
            Dictionary of class-specific configurations
        """
        if self.current_config is None:
            raise ConfigValidationError("No configuration loaded")

        class_configs = {}

        # Get detection passes for class
        class_passes = [
            pass_config for pass_configs in self.current_config.detection.passes.values()
            for pass_config in pass_configs
            if pass_config.class_name == class_name
        ]

        # Get cleaning config for class
        cleaning_config = next(
            (config for config in self.current_config.cleaning
             if config.class_name == class_name),
            None
        )

        if class_passes or cleaning_config:
            class_configs['detection_passes'] = class_passes
            class_configs['cleaning'] = cleaning_config

        return class_configs

    @classmethod
    def create_default_config(cls, output_path: Union[str, Path],
                              class_name: str = None,
                              model_path: str = None,
                              data_path: str = None) -> Path:
        """
        Creates a new configuration file with default values.

        Args:
            output_path: Path where the config file should be created
            class_name: Optional name for the object class to detect (default: 'object')
            model_path: Optional path to the YOLO model file
            data_path: Optional path to the input data directory

        Returns:
            Path: Path to the created configuration file

        Raises:
            ConfigValidationError: If the configuration can't be created
        """
        output_path = Path(output_path)
        config = cls.DEFAULT_CONFIG.copy()

        # Update with provided values
        if class_name:
            # Update class name in detection passes
            for pass_config in config['detection']['passes']:
                pass_config['class'] = class_name

            # Update cleaning config
            config['cleaning'] = {
                class_name: config['cleaning']['object']
            }

            # Update output filename
            config['paths']['final_out_name'] = f"_{class_name}s.geojson"

        if model_path:
            config['paths']['yolo_model'] = str(Path(model_path))

        if data_path:
            data_path = Path(data_path)
            config['paths']['tif_folder'] = str(data_path)
            config['paths']['output_folder'] = str(data_path / 'output')

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save configuration
        try:
            with open(output_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
            return output_path
        except Exception as e:
            raise ConfigValidationError(f"Failed to create configuration file: {e}")

    @staticmethod
    def create_project_structure(base_path: Union[str, Path], class_name: str) -> Dict[str, Path]:
        """
        Creates a complete project structure with default configuration.

        Args:
            base_path: Base directory for the project
            class_name: Name of the object class to detect

        Returns:
            Dictionary containing paths to key project directories and files

        Example structure:
        project_root/
        ├── config/
        │   └── config.yaml
        ├── input/
        ├── output/
        ├── models/
        └── logs/
        """
        base_path = Path(base_path)

        # Define project structure
        structure = {
            'config': base_path / 'config',
            'input': base_path / 'input',
            'output': base_path / 'output',
            'models': base_path / 'models',
            'logs': base_path / 'logs'
        }

        # Create directories
        for path in structure.values():
            path.mkdir(parents=True, exist_ok=True)

        # Create configuration file
        config_path = structure['config'] / 'config.yaml'
        ConfigManager.create_default_config(
            config_path,
            class_name=class_name,
            data_path=structure['input'],
            model_path=str(structure['models'] / 'best.pt')
        )

        structure['config_file'] = config_path
        return structure

    def get_minimal_config(self, class_name: str) -> Dict:
        """
        Returns a minimal working configuration for a single class.
        Useful for testing or simple deployments.

        Args:
            class_name: Name of the object class to detect

        Returns:
            Dictionary containing minimal required configuration
        """
        return {
            'paths': {
                'tif_folder': './input',
                'output_folder': './output',
                'yolo_model': './models/best.pt',
                'sam_model': './models/sam2_l.pt'
            },
            'detection': {
                'passes': [{
                    'pass': 0,
                    'class': class_name,
                    'eps': 0.25,
                    'min_samples': 2
                }]
            },
            'cleaning': {
                class_name: {
                    'simplification_tolerance': 0.15,
                    'overlap_threshold': 0.25,
                    'min_area': 1,
                    'max_area': 32,
                    'convexity_threshold': 1.15
                }
            }
        }