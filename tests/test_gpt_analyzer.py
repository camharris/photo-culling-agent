#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unit tests for GPTAnalyzer class."""

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.photo_culling_agent.gpt_analyzer import GPTAnalyzer


class TestGPTAnalyzer:
    """Unit tests for the GPTAnalyzer class."""

    @pytest.fixture
    def mock_env_api_key(self, monkeypatch: Any) -> None:
        """Set a mock API key in the environment.

        Args:
            monkeypatch: pytest fixture for patching
        """
        monkeypatch.setenv("OPENAI_API_KEY", "mock-api-key")

    @pytest.fixture
    def gpt_analyzer(self, mock_env_api_key: None) -> GPTAnalyzer:
        """Create and return a GPTAnalyzer instance with mocked API key.

        Args:
            mock_env_api_key: fixture to set mock API key

        Returns:
            GPTAnalyzer: An instance of the GPTAnalyzer class
        """
        return GPTAnalyzer()

    @pytest.fixture
    def sample_analysis_result(self) -> Dict[str, Any]:
        """Return a sample analysis result.

        Returns:
            Dict: Sample analysis result
        """
        return {
            "verdict": "keep",
            "score": 85.7,
            "rating": "4.5 stars",
            "tags": ["strong composition", "good lighting", "mountains"],
            "location": "Mountain range (generic)",
            "analysis": {
                "composition": 88,
                "exposure": 85,
                "subject": 84,
                "layering": 86,
                "notes": "Good use of leading lines and balanced composition",
            },
        }

    def test_init_with_env_api_key(self, mock_env_api_key: None) -> None:
        """Test initialization with API key from environment.

        Args:
            mock_env_api_key: fixture to set mock API key
        """
        analyzer = GPTAnalyzer()
        assert analyzer.api_key == "mock-api-key"
        assert analyzer.client is not None

    def test_init_with_provided_api_key(self) -> None:
        """Test initialization with provided API key."""
        analyzer = GPTAnalyzer(api_key="provided-api-key")
        assert analyzer.api_key == "provided-api-key"
        assert analyzer.client is not None

    def test_init_without_api_key(self, monkeypatch: Any) -> None:
        """Test initialization without API key raises error.

        Args:
            monkeypatch: pytest fixture for patching
        """
        # Clear the OPENAI_API_KEY environment variable
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            GPTAnalyzer()

    def test_customize_system_prompt(self, gpt_analyzer: GPTAnalyzer) -> None:
        """Test customizing the system prompt.

        Args:
            gpt_analyzer: GPTAnalyzer instance
        """
        custom_prompt = "Custom system prompt for testing"
        gpt_analyzer.customize_system_prompt(custom_prompt)
        assert gpt_analyzer.base_system_prompt == custom_prompt

    @patch("openai.OpenAI")
    def test_analyze_image_success(
        self,
        mock_openai: MagicMock,
        gpt_analyzer: GPTAnalyzer,
        sample_analysis_result: Dict[str, Any],
    ) -> None:
        """Test successful image analysis.

        Args:
            mock_openai: Mocked OpenAI class
            gpt_analyzer: GPTAnalyzer instance
            sample_analysis_result: Sample analysis result
        """
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = json.dumps(sample_analysis_result)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        # Replace the real client with the mock
        gpt_analyzer.client = mock_client

        # Test the analyze_image method
        result = gpt_analyzer.analyze_image(
            base64_image="mock_base64", file_name="test.jpg", post_processed=False
        )

        # Verify the API was called with the right arguments
        mock_client.chat.completions.create.assert_called_once()

        # Check that the result has the expected structure
        assert result["filename"] == "test.jpg"
        assert result["verdict"] == sample_analysis_result["verdict"]
        assert result["score"] == sample_analysis_result["score"]
        assert result["post_processed"] is False
        assert "user_verdict_override" in result
        assert "user_feedback" in result
        assert "learning_signal" in result
        assert "relative_rank" in result

        # Check that the validation returns True for this result
        assert gpt_analyzer.validate_analysis_result(result) is True

    @patch("openai.OpenAI")
    def test_analyze_image_api_error(
        self, mock_openai: MagicMock, gpt_analyzer: GPTAnalyzer
    ) -> None:
        """Test handling API errors during image analysis.

        Args:
            mock_openai: Mocked OpenAI class
            gpt_analyzer: GPTAnalyzer instance
        """
        # Mock the OpenAI client to raise an exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        # Replace the real client with the mock
        gpt_analyzer.client = mock_client

        # Test the analyze_image method with error
        result = gpt_analyzer.analyze_image(base64_image="mock_base64", file_name="test.jpg")

        # Check that an error result is returned
        assert result["filename"] == "test.jpg"
        assert result["verdict"] == "error"
        assert result["score"] == 0
        assert "error" in result
        assert result["error"] == "API error"

        # Check that validation returns False for an error result
        assert gpt_analyzer.validate_analysis_result(result) is False

    def test_validate_analysis_result(
        self, gpt_analyzer: GPTAnalyzer, sample_analysis_result: Dict[str, Any]
    ) -> None:
        """Test validating analysis results.

        Args:
            gpt_analyzer: GPTAnalyzer instance
            sample_analysis_result: Sample analysis result
        """
        # Add required fields to the sample result
        full_result = sample_analysis_result.copy()
        full_result["filename"] = "test.jpg"

        # Test with valid result
        assert gpt_analyzer.validate_analysis_result(full_result) is True

        # Test with missing required fields
        missing_verdict = full_result.copy()
        del missing_verdict["verdict"]
        assert gpt_analyzer.validate_analysis_result(missing_verdict) is False

        missing_score = full_result.copy()
        del missing_score["score"]
        assert gpt_analyzer.validate_analysis_result(missing_score) is False

        # Test with missing analysis fields
        missing_analysis_field = full_result.copy()
        missing_analysis_field["analysis"] = {}  # Empty analysis dict
        assert gpt_analyzer.validate_analysis_result(missing_analysis_field) is False
