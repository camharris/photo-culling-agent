#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for feedback-driven learning integration across components."""

from typing import TYPE_CHECKING, Any, Dict
from unittest.mock import MagicMock

import pytest

from src.photo_culling_agent.langgraph_pipeline import PhotoCullingGraph

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


class TestFeedbackIntegration:
    """End-to-end style tests for feedback learning integration."""

    @pytest.fixture
    def mock_pipeline_with_components(self, mocker: "MockerFixture") -> PhotoCullingGraph:
        """Create a graph with mocked analyzer to observe set/clear calls."""
        mock_image_processor = MagicMock()
        mock_metadata_manager = MagicMock()
        mock_gpt_analyzer = MagicMock()

        graph = PhotoCullingGraph(
            image_processor=mock_image_processor,
            gpt_analyzer=mock_gpt_analyzer,
            metadata_manager=mock_metadata_manager,
        )
        return graph

    def test_incorporate_feedback_sets_feedback_context(
        self, mock_pipeline_with_components: PhotoCullingGraph
    ) -> None:
        """Ensure feedback aggregation triggers analyzer.set_feedback_context with summary."""
        processed_images: Dict[str, Dict[str, Any]] = {
            "/tmp/a.jpg": {
                "analysis_result": {"verdict": "keep", "score": 88, "analysis": {"notes": "Great"}},
                "learning_signal": "Agree",
                "user_feedback": "Nice colors",
                "user_verdict_override": "keep",
            },
            "/tmp/b.jpg": {
                "analysis_result": {"verdict": "toss", "score": 45, "analysis": {"notes": "Blur"}},
                "learning_signal": "Disagree",
                "user_feedback": "Too dark",
                "user_verdict_override": "keep",
            },
        }

        mock_pipeline_with_components.incorporate_feedback_data(processed_images)

        analyzer = mock_pipeline_with_components.gpt_analyzer
        analyzer.set_feedback_context.assert_called_once()
        args, _ = analyzer.set_feedback_context.call_args
        assert isinstance(args[0], str) and "AI Verdict" in args[0]

    def test_clear_learning_context_calls_analyzer(
        self, mock_pipeline_with_components: PhotoCullingGraph
    ) -> None:
        """Ensure clearing context calls analyzer.clear_feedback_context."""
        analyzer = mock_pipeline_with_components.gpt_analyzer
        mock_pipeline_with_components.clear_learning_context()
        analyzer.clear_feedback_context.assert_called_once()
