#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the Gradio learning-related handlers."""

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from src.photo_culling_agent.gradio_interface.gradio_interface import PhotoCullingInterface

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


class TestLearningHandlers:
    """Tests for apply learnings and hard reset handlers."""

    @pytest.fixture
    def interface(self, mocker: "MockerFixture", tmp_path: Any) -> PhotoCullingInterface:
        """Create an interface with a mocked pipeline."""
        mocker.patch("os.makedirs")
        mocker.patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp"))
        mock_pipeline = MagicMock()
        mocker.patch(
            "src.photo_culling_agent.gradio_interface.gradio_interface.PhotoCullingGraph",
            return_value=mock_pipeline,
        )
        return PhotoCullingInterface(output_dir=str(tmp_path / "out"))

    def test_handle_apply_learnings_and_reset_ui(self, interface: PhotoCullingInterface) -> None:
        """Ensure feedback gets incorporated and UI is reset."""
        # Seed processed images to make incorporation occur
        interface.processed_images = {
            "/tmp/a.jpg": {"analysis_result": {"verdict": "keep", "analysis": {}}}
        }

        returned = interface.handle_apply_learnings_and_reset_ui()

        # Pipeline call and state reset
        interface.pipeline.incorporate_feedback_data.assert_called_once()
        assert interface.processed_images == {}
        assert len(interface.uploads_in_progress) == 0

        # Default UI reset values: first element is status message
        assert isinstance(returned, tuple) and isinstance(returned[0], str)

    def test_handle_hard_reset(self, interface: PhotoCullingInterface) -> None:
        """Ensure hard reset clears learning context and UI state."""
        # Seed state
        interface.processed_images = {"/tmp/a.jpg": {}}
        interface.uploads_in_progress = {"/tmp/a.jpg"}

        returned = interface.handle_hard_reset()

        interface.pipeline.clear_learning_context.assert_called_once()
        assert interface.processed_images == {}
        assert len(interface.uploads_in_progress) == 0
        assert isinstance(returned, tuple) and isinstance(returned[0], str)
