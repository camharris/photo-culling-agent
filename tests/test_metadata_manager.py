#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unit tests for MetadataManager class."""

import os
import json
import tempfile
from typing import Any, Dict, List, Optional
import pytest
from pytest_mock import MockerFixture

from src.photo_culling_agent.metadata_manager import MetadataManager


class TestMetadataManager:
    """Unit tests for the MetadataManager class."""

    @pytest.fixture
    def metadata_manager(self) -> MetadataManager:
        """Create and return a MetadataManager instance.
        
        Returns:
            MetadataManager: An instance of the MetadataManager class
        """
        return MetadataManager()

    @pytest.fixture
    def sample_metadata(self) -> Dict[str, Any]:
        """Create sample metadata for testing.
        
        Returns:
            Dict: Sample metadata
        """
        return {
            "filename": "test_image.jpg",
            "verdict": "keep",
            "score": 87.4,
            "rating": "4.5 stars",
            "post_processed": False,
            "tags": ["strong composition", "layered depth", "vibrant sky"],
            "location": "Alabama Hills, California (approximate)",
            "analysis": {
                "composition": 90,
                "exposure": 85,
                "subject": 86,
                "layering": 89,
                "notes": "Natural framing from arch, distant snowy peaks, vibrant contrast"
            },
            "user_verdict_override": None,
            "user_feedback": None,
            "learning_signal": None,
            "relative_rank": None
        }

    def test_add_metadata(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test adding metadata.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        metadata_manager.add_metadata(sample_metadata)
        
        # Verify the metadata was added to the store
        assert "test_image.jpg" in metadata_manager.metadata_store
        assert metadata_manager.metadata_store["test_image.jpg"] == sample_metadata
        
        # Verify the image was categorized correctly
        assert "test_image.jpg" in metadata_manager.keep_images
        assert "test_image.jpg" not in metadata_manager.toss_images
        assert "test_image.jpg" not in metadata_manager.error_images

    def test_add_metadata_without_filename(self, metadata_manager: MetadataManager) -> None:
        """Test adding metadata without a filename raises an error.
        
        Args:
            metadata_manager: MetadataManager instance
        """
        invalid_metadata = {"verdict": "keep", "score": 85}
        
        with pytest.raises(ValueError, match="Metadata must contain a filename"):
            metadata_manager.add_metadata(invalid_metadata)

    def test_update_categorization(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test updating categorization based on verdict.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        # Add a "keep" metadata
        metadata_manager.add_metadata(sample_metadata)
        assert "test_image.jpg" in metadata_manager.keep_images
        
        # Change verdict to "toss" and update
        toss_metadata = sample_metadata.copy()
        toss_metadata["verdict"] = "toss"
        metadata_manager.add_metadata(toss_metadata)
        
        # Verify categorization changed
        assert "test_image.jpg" not in metadata_manager.keep_images
        assert "test_image.jpg" in metadata_manager.toss_images
        
        # Add an invalid verdict
        error_metadata = sample_metadata.copy()
        error_metadata["verdict"] = "invalid"
        metadata_manager.add_metadata(error_metadata)
        
        # Verify it's in the error category
        assert "test_image.jpg" not in metadata_manager.keep_images
        assert "test_image.jpg" not in metadata_manager.toss_images
        assert "test_image.jpg" in metadata_manager.error_images

    def test_get_metadata(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test getting metadata for a specific image.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        metadata_manager.add_metadata(sample_metadata)
        
        # Get existing metadata
        retrieved = metadata_manager.get_metadata("test_image.jpg")
        assert retrieved == sample_metadata
        
        # Get non-existent metadata
        nonexistent = metadata_manager.get_metadata("nonexistent.jpg")
        assert nonexistent is None

    def test_get_all_metadata(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test getting all metadata.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        # Add multiple metadata entries
        metadata_manager.add_metadata(sample_metadata)
        
        second_metadata = sample_metadata.copy()
        second_metadata["filename"] = "second_image.jpg"
        metadata_manager.add_metadata(second_metadata)
        
        # Get all metadata
        all_metadata = metadata_manager.get_all_metadata()
        assert len(all_metadata) == 2
        assert "test_image.jpg" in all_metadata
        assert "second_image.jpg" in all_metadata

    def test_get_categorized_images(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test getting lists of categorized images.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        # Add a "keep" image
        metadata_manager.add_metadata(sample_metadata)
        
        # Add a "toss" image
        toss_metadata = sample_metadata.copy()
        toss_metadata["filename"] = "toss_image.jpg"
        toss_metadata["verdict"] = "toss"
        metadata_manager.add_metadata(toss_metadata)
        
        # Add an "error" image
        error_metadata = sample_metadata.copy()
        error_metadata["filename"] = "error_image.jpg"
        error_metadata["verdict"] = "error"
        metadata_manager.add_metadata(error_metadata)
        
        # Get categorized lists
        keep_images = metadata_manager.get_keep_images()
        toss_images = metadata_manager.get_toss_images()
        error_images = metadata_manager.get_error_images()
        
        assert "test_image.jpg" in keep_images
        assert "toss_image.jpg" in toss_images
        assert "error_image.jpg" in error_images

    def test_update_user_verdict(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test updating user verdict override.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        # Add a "keep" image
        metadata_manager.add_metadata(sample_metadata)
        assert "test_image.jpg" in metadata_manager.keep_images
        
        # Override to "toss"
        metadata_manager.update_user_verdict("test_image.jpg", "toss")
        
        # Verify override was applied
        updated_metadata = metadata_manager.get_metadata("test_image.jpg")
        assert updated_metadata["user_verdict_override"] == "toss"
        
        # Verify categorization changed
        assert "test_image.jpg" not in metadata_manager.keep_images
        assert "test_image.jpg" in metadata_manager.toss_images

    def test_update_user_verdict_invalid_image(self, metadata_manager: MetadataManager) -> None:
        """Test updating user verdict for non-existent image.
        
        Args:
            metadata_manager: MetadataManager instance
        """
        with pytest.raises(ValueError, match="No metadata for nonexistent.jpg"):
            metadata_manager.update_user_verdict("nonexistent.jpg", "keep")

    def test_update_user_verdict_invalid_verdict(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test updating user verdict with invalid verdict value.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        metadata_manager.add_metadata(sample_metadata)
        
        with pytest.raises(ValueError, match="Invalid verdict: invalid"):
            metadata_manager.update_user_verdict("test_image.jpg", "invalid")

    def test_add_user_feedback(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test adding user feedback.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        metadata_manager.add_metadata(sample_metadata)
        
        feedback = "This is a great landscape photo!"
        metadata_manager.add_user_feedback("test_image.jpg", feedback)
        
        updated_metadata = metadata_manager.get_metadata("test_image.jpg")
        assert updated_metadata["user_feedback"] == feedback

    def test_add_user_feedback_invalid_image(self, metadata_manager: MetadataManager) -> None:
        """Test adding user feedback for non-existent image.
        
        Args:
            metadata_manager: MetadataManager instance
        """
        with pytest.raises(ValueError, match="No metadata for nonexistent.jpg"):
            metadata_manager.add_user_feedback("nonexistent.jpg", "feedback")

    def test_export_metadata_to_json_single(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test exporting metadata for a single image.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        metadata_manager.add_metadata(sample_metadata)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = metadata_manager.export_metadata_to_json(temp_dir, "test_image.jpg")
            
            # Verify the file exists
            assert os.path.exists(output_file)
            
            # Verify the content
            with open(output_file, 'r') as f:
                exported_data = json.load(f)
            
            assert exported_data == sample_metadata

    def test_export_metadata_to_json_all(self, metadata_manager: MetadataManager, sample_metadata: Dict[str, Any]) -> None:
        """Test exporting metadata for all images.
        
        Args:
            metadata_manager: MetadataManager instance
            sample_metadata: Sample metadata
        """
        metadata_manager.add_metadata(sample_metadata)
        
        second_metadata = sample_metadata.copy()
        second_metadata["filename"] = "second_image.jpg"
        metadata_manager.add_metadata(second_metadata)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = metadata_manager.export_metadata_to_json(temp_dir)
            
            # Verify the file exists
            assert os.path.exists(output_file)
            
            # Verify the content
            with open(output_file, 'r') as f:
                exported_data = json.load(f)
            
            assert "test_image.jpg" in exported_data
            assert "second_image.jpg" in exported_data
            assert exported_data["test_image.jpg"] == sample_metadata
            assert exported_data["second_image.jpg"] == second_metadata

    def test_export_metadata_invalid_image(self, metadata_manager: MetadataManager) -> None:
        """Test exporting metadata for non-existent image.
        
        Args:
            metadata_manager: MetadataManager instance
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="No metadata for nonexistent.jpg"):
                metadata_manager.export_metadata_to_json(temp_dir, "nonexistent.jpg") 