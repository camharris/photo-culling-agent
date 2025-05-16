#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unit tests for ImageProcessor class."""

import os
import base64
from io import BytesIO
from typing import Any, Dict, Optional
import pytest
from PIL import Image
from pytest_mock import MockerFixture

from src.photo_culling_agent.image_processor import ImageProcessor


class TestImageProcessor:
    """Unit tests for the ImageProcessor class."""

    @pytest.fixture
    def image_processor(self) -> ImageProcessor:
        """Create and return an ImageProcessor instance.
        
        Returns:
            ImageProcessor: An instance of the ImageProcessor class
        """
        return ImageProcessor()
    
    @pytest.fixture
    def sample_image(self, tmp_path: Any) -> str:
        """Create a simple test image.
        
        Args:
            tmp_path: pytest fixture providing a temporary directory path
            
        Returns:
            str: Path to the test image
        """
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        file_path = os.path.join(tmp_path, "test_image.jpg")
        img.save(file_path)
        return file_path
    
    @pytest.fixture
    def invalid_image_path(self, tmp_path: Any) -> str:
        """Return a path to a non-existent image.
        
        Args:
            tmp_path: pytest fixture providing a temporary directory path
            
        Returns:
            str: Path to a non-existent image
        """
        return os.path.join(tmp_path, "nonexistent.jpg")
    
    @pytest.fixture
    def invalid_format_path(self, tmp_path: Any) -> str:
        """Create a file with invalid format.
        
        Args:
            tmp_path: pytest fixture providing a temporary directory path
            
        Returns:
            str: Path to a file with invalid format
        """
        file_path = os.path.join(tmp_path, "test.txt")
        with open(file_path, 'w') as f:
            f.write("This is not an image")
        return file_path

    def test_validate_image_valid(self, image_processor: ImageProcessor, sample_image: str) -> None:
        """Test validate_image with a valid image.
        
        Args:
            image_processor: ImageProcessor instance
            sample_image: Path to a valid test image
        """
        assert image_processor.validate_image(sample_image) is True

    def test_validate_image_nonexistent(self, image_processor: ImageProcessor, invalid_image_path: str) -> None:
        """Test validate_image with a non-existent file.
        
        Args:
            image_processor: ImageProcessor instance
            invalid_image_path: Path to a non-existent image
        """
        assert image_processor.validate_image(invalid_image_path) is False

    def test_validate_image_invalid_format(self, image_processor: ImageProcessor, invalid_format_path: str) -> None:
        """Test validate_image with an invalid file format.
        
        Args:
            image_processor: ImageProcessor instance
            invalid_format_path: Path to a file with invalid format
        """
        assert image_processor.validate_image(invalid_format_path) is False

    def test_load_image_valid(self, image_processor: ImageProcessor, sample_image: str) -> None:
        """Test load_image with a valid image.
        
        Args:
            image_processor: ImageProcessor instance
            sample_image: Path to a valid test image
        """
        image = image_processor.load_image(sample_image)
        assert image is not None
        assert isinstance(image, Image.Image)

    def test_load_image_invalid(self, image_processor: ImageProcessor, invalid_image_path: str) -> None:
        """Test load_image with an invalid image.
        
        Args:
            image_processor: ImageProcessor instance
            invalid_image_path: Path to a non-existent image
        """
        image = image_processor.load_image(invalid_image_path)
        assert image is None

    def test_extract_basic_metadata(self, image_processor: ImageProcessor, sample_image: str) -> None:
        """Test extract_basic_metadata.
        
        Args:
            image_processor: ImageProcessor instance
            sample_image: Path to a valid test image
        """
        image = image_processor.load_image(sample_image)
        assert image is not None
        
        metadata = image_processor.extract_basic_metadata(image)
        assert isinstance(metadata, dict)
        assert "format" in metadata
        assert "size" in metadata
        assert "mode" in metadata

    def test_encode_image_base64(self, image_processor: ImageProcessor, sample_image: str) -> None:
        """Test encode_image_base64.
        
        Args:
            image_processor: ImageProcessor instance
            sample_image: Path to a valid test image
        """
        image = image_processor.load_image(sample_image)
        assert image is not None
        
        base64_str = image_processor.encode_image_base64(image)
        assert isinstance(base64_str, str)
        
        # Verify it's valid base64 by decoding it
        try:
            decoded = base64.b64decode(base64_str)
            assert isinstance(decoded, bytes)
        except Exception:
            pytest.fail("Failed to decode base64 string")

    def test_prepare_image_for_analysis_valid(self, image_processor: ImageProcessor, sample_image: str) -> None:
        """Test prepare_image_for_analysis with a valid image.
        
        Args:
            image_processor: ImageProcessor instance
            sample_image: Path to a valid test image
        """
        base64_image, metadata = image_processor.prepare_image_for_analysis(sample_image)
        
        assert base64_image is not None
        assert metadata is not None
        assert isinstance(base64_image, str)
        assert isinstance(metadata, dict)

    def test_prepare_image_for_analysis_invalid(self, image_processor: ImageProcessor, invalid_image_path: str) -> None:
        """Test prepare_image_for_analysis with an invalid image.
        
        Args:
            image_processor: ImageProcessor instance
            invalid_image_path: Path to a non-existent image
        """
        base64_image, metadata = image_processor.prepare_image_for_analysis(invalid_image_path)
        
        assert base64_image is None
        assert metadata is None 