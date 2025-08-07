#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for GPTAnalyzer feedback context behavior."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.photo_culling_agent.gpt_analyzer import GPTAnalyzer

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


class TestGPTAnalyzerFeedbackContext:
    """Unit tests for GPTAnalyzer feedback prompt composition and error logging."""

    @pytest.fixture
    def analyzer(self, mocker: "MockerFixture") -> GPTAnalyzer:
        """Create an analyzer with mocked OpenAI client."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
        instance = GPTAnalyzer()
        # Mock the OpenAI client to avoid network calls
        instance.client = MagicMock()
        return instance

    def test_set_feedback_context_truncates_long_text(self, analyzer: GPTAnalyzer) -> None:
        """Ensure long feedback is truncated and wrapped into prompt section."""
        long_text = "x" * 10000
        analyzer.set_feedback_context(long_text)
        assert analyzer.feedback_context_for_prompt is not None
        assert len(analyzer.feedback_context_for_prompt) < 4500
        assert (
            "Important: Please learn from this recent user feedback"
            in analyzer.feedback_context_for_prompt
        )

    def test_clear_feedback_context(self, analyzer: GPTAnalyzer) -> None:
        """Ensure clearing context removes it from analyzer state."""
        analyzer.set_feedback_context("some feedback")
        assert analyzer.feedback_context_for_prompt is not None
        analyzer.clear_feedback_context()
        assert analyzer.feedback_context_for_prompt is None
