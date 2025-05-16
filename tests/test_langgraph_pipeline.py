#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unit tests for PhotoCullingGraph LangGraph pipeline."""

import os
import json
from typing import Any, Dict, List, Optional
import pytest
from unittest.mock import MagicMock, patch
from pytest_mock import MockerFixture
import tempfile
from PIL import Image

from src.photo_culling_agent.image_processor import ImageProcessor
from src.photo_culling_agent.gpt_analyzer import GPTAnalyzer
from src.photo_culling_agent.metadata_manager import MetadataManager
from src.photo_culling_agent.langgraph_pipeline import PhotoCullingGraph, PhotoCullingState
from src.photo_culling_agent.langgraph_pipeline.langgraph_pipeline import (
    process_image,
    analyze_image,
    decide_verdict,
    comparative_analysis,
    update_metadata,
    should_end_workflow,
    DEFAULT_WEIGHTS,
    DECISION_THRESHOLDS,
    ConfidenceLevel
)


class TestPhotoCullingGraph:
    """Unit tests for the PhotoCullingGraph class."""

    @pytest.fixture
    def image_processor(self) -> MagicMock:
        """Create a mock ImageProcessor.
        
        Returns:
            MagicMock: Mocked ImageProcessor instance
        """
        mock_processor = MagicMock(spec=ImageProcessor)
        mock_processor.validate_image.return_value = True
        mock_processor.prepare_image_for_analysis.return_value = ("mock_base64", {"size": (100, 100), "format": "JPEG"})
        return mock_processor

    @pytest.fixture
    def gpt_analyzer(self) -> MagicMock:
        """Create a mock GPTAnalyzer.
        
        Returns:
            MagicMock: Mocked GPTAnalyzer instance
        """
        mock_analyzer = MagicMock(spec=GPTAnalyzer)
        mock_analyzer.analyze_image.return_value = {
            "filename": "test.jpg",
            "verdict": "keep",
            "score": 85.5,
            "rating": "4 stars",
            "tags": ["test"],
            "analysis": {
                "composition": 80,
                "exposure": 85,
                "subject": 90,
                "layering": 87,
                "notes": "Test notes"
            }
        }
        mock_analyzer.validate_analysis_result.return_value = True
        return mock_analyzer

    @pytest.fixture
    def metadata_manager(self) -> MagicMock:
        """Create a mock MetadataManager.
        
        Returns:
            MagicMock: Mocked MetadataManager instance
        """
        mock_manager = MagicMock(spec=MetadataManager)
        mock_manager.get_keep_images.return_value = ["test.jpg"]
        mock_manager.get_toss_images.return_value = []
        mock_manager.get_metadata.return_value = {
            "filename": "test.jpg",
            "verdict": "keep"
        }
        return mock_manager

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
        file_path = os.path.join(tmp_path, "test.jpg")
        img.save(file_path)
        return file_path

    @pytest.fixture
    def culling_graph(
        self,
        image_processor: MagicMock,
        gpt_analyzer: MagicMock,
        metadata_manager: MagicMock
    ) -> PhotoCullingGraph:
        """Create a PhotoCullingGraph with mock components.
        
        Args:
            image_processor: Mock ImageProcessor
            gpt_analyzer: Mock GPTAnalyzer
            metadata_manager: Mock MetadataManager
            
        Returns:
            PhotoCullingGraph: Graph instance with mock components
        """
        return PhotoCullingGraph(
            image_processor=image_processor,
            gpt_analyzer=gpt_analyzer,
            metadata_manager=metadata_manager
        )

    @pytest.fixture
    def analyzed_state(self) -> PhotoCullingState:
        """Create a state with analysis results for testing.
        
        Returns:
            PhotoCullingState: State with analysis results
        """
        return {
            "image_path": "test.jpg",
            "base64_image": "mock_base64",
            "image_metadata": {"size": (100, 100), "format": "JPEG"},
            "analysis_result": {
                "filename": "test.jpg",
                "verdict": "keep",
                "score": 85.5,
                "rating": "4 stars",
                "tags": ["test"],
                "analysis": {
                    "composition": 80,
                    "exposure": 85,
                    "subject": 90,
                    "layering": 87,
                    "notes": "Test notes"
                }
            },
            "verdict": "keep",
            "confidence": None,
            "confidence_level": None,
            "decision_rationale": None,
            "user_feedback": None,
            "user_verdict_override": None,
            "similar_images": None,
            "relative_ranking": None,
            "error": None,
            "completed": False
        }

    @pytest.fixture
    def low_quality_analyzed_state(self) -> PhotoCullingState:
        """Create a state with low quality analysis results for testing.
        
        Returns:
            PhotoCullingState: State with low quality analysis results
        """
        return {
            "image_path": "test_low.jpg",
            "base64_image": "mock_base64",
            "image_metadata": {"size": (100, 100), "format": "JPEG"},
            "analysis_result": {
                "filename": "test_low.jpg",
                "verdict": "toss",
                "score": 45.5,
                "rating": "2 stars",
                "tags": ["test", "low_quality"],
                "analysis": {
                    "composition": 40,
                    "exposure": 45,
                    "subject": 50,
                    "layering": 47,
                    "notes": "Low quality image"
                }
            },
            "verdict": "toss",
            "confidence": None,
            "confidence_level": None,
            "decision_rationale": None,
            "user_feedback": None,
            "user_verdict_override": None,
            "similar_images": None,
            "relative_ranking": None,
            "error": None,
            "completed": False
        }

    def test_process_image_node(self, image_processor: MagicMock) -> None:
        """Test the process_image node function.
        
        Args:
            image_processor: Mock ImageProcessor
        """
        # Create initial state
        state: PhotoCullingState = {
            "image_path": "test.jpg",
            "base64_image": None,
            "image_metadata": None,
            "analysis_result": None,
            "verdict": None,
            "confidence": None,
            "confidence_level": None,
            "decision_rationale": None,
            "user_feedback": None,
            "user_verdict_override": None,
            "similar_images": None,
            "relative_ranking": None,
            "error": None,
            "completed": False
        }
        
        # Run the process_image node
        result = process_image(state, image_processor)
        
        # Verify the image was processed
        image_processor.validate_image.assert_called_once_with("test.jpg")
        image_processor.prepare_image_for_analysis.assert_called_once_with("test.jpg")
        
        # Check that state was updated correctly
        assert result["base64_image"] == "mock_base64"
        assert result["image_metadata"] == {"size": (100, 100), "format": "JPEG"}
        assert result["error"] is None

    def test_analyze_image_node(self, gpt_analyzer: MagicMock) -> None:
        """Test the analyze_image node function.
        
        Args:
            gpt_analyzer: Mock GPTAnalyzer
        """
        # Create state after image processing
        state: PhotoCullingState = {
            "image_path": "test.jpg",
            "base64_image": "mock_base64",
            "image_metadata": {"size": (100, 100), "format": "JPEG"},
            "analysis_result": None,
            "verdict": None,
            "confidence": None,
            "confidence_level": None,
            "decision_rationale": None,
            "user_feedback": None,
            "user_verdict_override": None,
            "similar_images": None,
            "relative_ranking": None,
            "error": None,
            "completed": False
        }
        
        # Run the analyze_image node
        result = analyze_image(state, gpt_analyzer)
        
        # Verify the image was analyzed
        gpt_analyzer.analyze_image.assert_called_once()
        gpt_analyzer.validate_analysis_result.assert_called_once()
        
        # Check that state was updated correctly
        assert result["analysis_result"] is not None
        assert result["verdict"] == "keep"
        assert result["error"] is None

    def test_decide_verdict_node_high_quality(self, analyzed_state: PhotoCullingState) -> None:
        """Test the decide_verdict node function with high quality image.
        
        Args:
            analyzed_state: State with analysis results
        """
        # Run the decide_verdict node
        result = decide_verdict(analyzed_state)
        
        # Check that state was updated correctly
        assert result["verdict"] == "keep"
        assert result["confidence"] is not None
        assert result["confidence"] > 0.5  # High confidence for keep verdict
        assert result["confidence_level"] in [ConfidenceLevel.DEFINITE_KEEP.name, ConfidenceLevel.LIKELY_KEEP.name]
        assert result["decision_rationale"] is not None
        assert result["decision_rationale"]["weighted_score"] > DECISION_THRESHOLDS["keep"]
        assert result["error"] is None

    def test_decide_verdict_node_low_quality(self, low_quality_analyzed_state: PhotoCullingState) -> None:
        """Test the decide_verdict node function with low quality image.
        
        Args:
            low_quality_analyzed_state: State with low quality analysis results
        """
        # Run the decide_verdict node
        result = decide_verdict(low_quality_analyzed_state)
        
        # Check that state was updated correctly
        assert result["verdict"] == "toss"
        assert result["confidence"] is not None
        assert result["confidence"] < 0.5  # Lower confidence for toss verdict
        assert result["confidence_level"] in [ConfidenceLevel.LIKELY_TOSS.name, ConfidenceLevel.DEFINITE_TOSS.name]
        assert result["decision_rationale"] is not None
        assert result["decision_rationale"]["weighted_score"] < DECISION_THRESHOLDS["keep"]
        assert result["error"] is None

    def test_decide_verdict_custom_weights(self, analyzed_state: PhotoCullingState) -> None:
        """Test the decide_verdict node function with custom weights.
        
        Args:
            analyzed_state: State with analysis results
        """
        # Define custom weights to emphasize composition
        custom_weights = {
            "composition": 2.0,  # Double weight for composition
            "exposure": 0.5,
            "subject": 0.5,
            "layering": 0.5,
            "base_score": 0.5
        }
        
        # Run the decide_verdict node with custom weights
        result = decide_verdict(analyzed_state, custom_weights)
        
        # Check that custom weights were applied
        assert result["decision_rationale"]["criteria_weights"] == custom_weights
        assert result["error"] is None

    def test_comparative_analysis_node(self, analyzed_state: PhotoCullingState) -> None:
        """Test the comparative_analysis node function (placeholder).
        
        Args:
            analyzed_state: State with analysis results
        """
        # Run the comparative_analysis node (currently just a passthrough)
        result = comparative_analysis(analyzed_state)
        
        # Check that state is unchanged (since it's a placeholder)
        assert result == analyzed_state

    def test_update_metadata_node(self, metadata_manager: MagicMock, analyzed_state: PhotoCullingState) -> None:
        """Test the update_metadata node function.
        
        Args:
            metadata_manager: Mock MetadataManager
            analyzed_state: State with analysis results
        """
        # Add decision data to the state
        decided_state = decide_verdict(analyzed_state)
        
        # Add user feedback
        decided_state["user_feedback"] = "Great photo!"
        
        # Run the update_metadata node
        result = update_metadata(decided_state, metadata_manager)
        
        # Verify metadata was updated
        metadata_manager.add_metadata.assert_called_once()
        
        # Extract the metadata that was passed to add_metadata
        called_metadata = metadata_manager.add_metadata.call_args[0][0]
        
        # Verify the user feedback was included
        assert called_metadata["user_feedback"] == "Great photo!"
        
        # Verify that the enhanced decision data was included
        assert "final_verdict" in called_metadata
        assert "confidence" in called_metadata
        assert "confidence_level" in called_metadata
        assert "decision_rationale" in called_metadata
        
        # Check that state was updated correctly
        assert result["completed"] is True
        assert result["error"] is None

    def test_should_end_workflow(self) -> None:
        """Test the should_end_workflow function."""
        # Test with error
        error_state: PhotoCullingState = {
            "image_path": "test.jpg",
            "base64_image": None,
            "image_metadata": None,
            "analysis_result": None,
            "verdict": None,
            "confidence": None,
            "confidence_level": None,
            "decision_rationale": None,
            "user_feedback": None,
            "user_verdict_override": None,
            "similar_images": None,
            "relative_ranking": None,
            "error": "Test error",
            "completed": False
        }
        assert should_end_workflow(error_state) is True
        
        # Test with completion
        completed_state: PhotoCullingState = {
            "image_path": "test.jpg",
            "base64_image": "mock_base64",
            "image_metadata": {"size": (100, 100), "format": "JPEG"},
            "analysis_result": {"verdict": "keep"},
            "verdict": "keep",
            "confidence": 0.85,
            "confidence_level": ConfidenceLevel.DEFINITE_KEEP.name,
            "decision_rationale": {"score": 85},
            "user_feedback": None,
            "user_verdict_override": None,
            "similar_images": None,
            "relative_ranking": None,
            "error": None,
            "completed": True
        }
        assert should_end_workflow(completed_state) is True
        
        # Test with ongoing process
        ongoing_state: PhotoCullingState = {
            "image_path": "test.jpg",
            "base64_image": "mock_base64",
            "image_metadata": {"size": (100, 100), "format": "JPEG"},
            "analysis_result": {"verdict": "keep"},
            "verdict": "keep",
            "confidence": 0.85,
            "confidence_level": ConfidenceLevel.DEFINITE_KEEP.name,
            "decision_rationale": {"score": 85},
            "user_feedback": None,
            "user_verdict_override": None,
            "similar_images": None,
            "relative_ranking": None,
            "error": None,
            "completed": False
        }
        assert should_end_workflow(ongoing_state) is False

    def test_graph_initialization(
        self,
        image_processor: MagicMock,
        gpt_analyzer: MagicMock,
        metadata_manager: MagicMock
    ) -> None:
        """Test PhotoCullingGraph initialization.
        
        Args:
            image_processor: Mock ImageProcessor
            gpt_analyzer: Mock GPTAnalyzer
            metadata_manager: Mock MetadataManager
        """
        graph = PhotoCullingGraph(
            image_processor=image_processor,
            gpt_analyzer=gpt_analyzer,
            metadata_manager=metadata_manager
        )
        
        assert graph.image_processor is image_processor
        assert graph.gpt_analyzer is gpt_analyzer
        assert graph.metadata_manager is metadata_manager
        assert graph.decision_weights == DEFAULT_WEIGHTS
        assert graph.graph is not None

    def test_graph_initialization_custom_weights(
        self,
        image_processor: MagicMock,
        gpt_analyzer: MagicMock,
        metadata_manager: MagicMock
    ) -> None:
        """Test PhotoCullingGraph initialization with custom weights.
        
        Args:
            image_processor: Mock ImageProcessor
            gpt_analyzer: Mock GPTAnalyzer
            metadata_manager: Mock MetadataManager
        """
        custom_weights = {
            "composition": 2.0,
            "exposure": 0.5,
            "subject": 0.5,
            "layering": 0.5,
            "base_score": 0.5
        }
        
        graph = PhotoCullingGraph(
            image_processor=image_processor,
            gpt_analyzer=gpt_analyzer,
            metadata_manager=metadata_manager,
            decision_weights=custom_weights
        )
        
        assert graph.decision_weights == custom_weights

    def test_process_image_workflow(
        self,
        culling_graph: PhotoCullingGraph,
        sample_image: str
    ) -> None:
        """Test the full image processing workflow.
        
        Args:
            culling_graph: PhotoCullingGraph instance
            sample_image: Path to test image
        """
        # Process the image
        result = culling_graph.process_image(sample_image)
        
        # Verify the result
        assert result["completed"] is True
        assert result["error"] is None
        assert result["verdict"] == "keep"
        assert result["confidence"] is not None
        assert result["confidence_level"] is not None
        assert result["decision_rationale"] is not None

    def test_provide_feedback(
        self,
        culling_graph: PhotoCullingGraph,
        sample_image: str
    ) -> None:
        """Test providing feedback through the workflow.
        
        Args:
            culling_graph: PhotoCullingGraph instance
            sample_image: Path to test image
        """
        # Process with feedback
        result = culling_graph.provide_feedback(
            sample_image,
            "Great landscape photo!",
            "keep"
        )
        
        # Verify the result
        assert result["completed"] is True
        assert result["error"] is None
        assert result["user_feedback"] == "Great landscape photo!"
        assert result["user_verdict_override"] == "keep"
        assert result["confidence"] is not None
        assert result["confidence_level"] is not None
        assert result["decision_rationale"] is not None

    def test_get_keep_images(self, culling_graph: PhotoCullingGraph) -> None:
        """Test getting keep images.
        
        Args:
            culling_graph: PhotoCullingGraph instance
        """
        result = culling_graph.get_keep_images()
        assert result == ["test.jpg"]
        culling_graph.metadata_manager.get_keep_images.assert_called_once()

    def test_get_toss_images(self, culling_graph: PhotoCullingGraph) -> None:
        """Test getting toss images.
        
        Args:
            culling_graph: PhotoCullingGraph instance
        """
        result = culling_graph.get_toss_images()
        assert result == []
        culling_graph.metadata_manager.get_toss_images.assert_called_once()

    def test_get_metadata(self, culling_graph: PhotoCullingGraph) -> None:
        """Test getting image metadata.
        
        Args:
            culling_graph: PhotoCullingGraph instance
        """
        result = culling_graph.get_metadata("test.jpg")
        assert result == {"filename": "test.jpg", "verdict": "keep"}
        culling_graph.metadata_manager.get_metadata.assert_called_once_with("test.jpg") 