from pathlib import Path
import logging
from typing import Union

from config_manager import ConfigManager
from image_processor import ImageProcessor

def process_yaml_path(yaml_path: Union[str, Path]) -> None:
    """
    Process images using configuration from a YAML file path.
    
    Args:
        yaml_path: Path to YAML configuration file
    """
    config_path = Path(yaml_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Initialize and run processor
    processor = ImageProcessor(config_path)
    
    # Get input/output paths from config
    config = ConfigManager(config_path).load_config()
    input_dir = Path(config.paths.tif_folder)
    output_dir = Path(config.paths.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all images
    image_files = list(input_dir.glob("*.tif"))
    for img_path in image_files:
        output_path = output_dir / f"{img_path.stem}_objects.geojson"
        processor.process_image(img_path, output_path)

# Example usage
if __name__ == "__main__":
    # Can be called directly with a path parameter
    process_yaml_path(r"C:\Users\red\PycharmProjects\ultra_geo\config.yaml")
