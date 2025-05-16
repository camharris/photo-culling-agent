#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Image processing functionality for the Photo Culling Agent."""

import os
from typing import Dict, List, Optional, Any, Tuple, Union, BinaryIO
from PIL import Image
import base64
from io import BytesIO


class ImageProcessor:
    """Handles loading, validation, and preparation of images for analysis."""

    VALID_EXTENSIONS = (".jpg", ".jpeg")

    def __init__(self):
        """Initialize the ImageProcessor."""
        pass

    def validate_image(self, file_path: str) -> bool:
        """Validate if the file is a supported image format.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if the image is valid, False otherwise
        """
        if not os.path.exists(file_path):
            return False
        
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.VALID_EXTENSIONS

    def load_image(self, file_path: str) -> Optional[Image.Image]:
        """Load an image from the file system.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            PIL.Image.Image or None: The loaded image or None if loading fails
        """
        if not self.validate_image(file_path):
            return None
        
        try:
            return Image.open(file_path)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None

    def extract_basic_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic metadata from an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict: Dictionary containing basic image metadata
        """
        metadata = {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
        }
        
        # Extract EXIF data if available
        exif_data = {}
        if hasattr(image, "_getexif") and image._getexif():
            exif = image._getexif()
            if exif:
                # Add relevant EXIF data
                exif_data = {"exif": exif}
        
        metadata.update(exif_data)
        return metadata

    def encode_image_base64(self, image: Image.Image) -> str:
        """Encode image as base64 string for API submission.
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Base64 encoded image string
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def prepare_image_for_analysis(self, file_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Prepare an image for GPT-4o analysis.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple: (base64_encoded_image, metadata) or (None, None) if preparation fails
        """
        image = self.load_image(file_path)
        if image is None:
            return None, None
        
        metadata = self.extract_basic_metadata(image)
        base64_image = self.encode_image_base64(image)
        
        return base64_image, metadata 