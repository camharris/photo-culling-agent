#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""GPT-4o analyzer for image evaluation in the Photo Culling Agent."""

import json
import logging
import os
from typing import Any, Dict, Optional

from openai import OpenAI


class GPTAnalyzer:
    """Manages communication with OpenAI's GPT-4o API for image analysis."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GPTAnalyzer.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=self.api_key)

        # Module logger
        self._logger = logging.getLogger(__name__)

        # Updated system prompt for photo analysis with HITL context
        self.base_system_prompt = """
        You are a professional landscape photographer assisting with photo grading for a human-in-the-loop system. Your job is to evaluate each image for both artistic and technical merit and return structured output to guide further decision-making.

        Your responsibilities:
        1. Evaluate the image holistically for composition, exposure, subject interest, and layering.
        2. Assign a total score from 0–100 based on overall quality and potential for editing.
        3. Provide a verdict: 'keep' or 'toss' (if uncertain, lean toward 'keep' and flag in notes).
        4. Score each of the following aspects individually (0–100):
        - composition
        - exposure
        - subject
        - layering
        5. If possible, detect and include an approximate location (e.g., Yosemite, Zion) or return null.
        6. Generate 3–6 relevant descriptive tags (e.g., "dramatic sky", "flat composition", "leading lines").
        7. Provide concise notes (1–3 sentences) describing the strengths and weaknesses of the image.
        8. Respect the `post_processed` flag if provided (true or false). Images marked as unedited should be judged more leniently on exposure or contrast.

        Output your response as **valid JSON** in the following format:

        {
        "verdict": "keep" or "toss",
        "score": float (0–100),
        "rating": "X stars" (1–5 stars, including half stars),
        "post_processed": boolean,
        "tags": [list of descriptive strings],
        "location": "Approximate location or null",
        "analysis": {
            "composition": int (0–100),
            "exposure": int (0–100),
            "subject": int (0–100),
            "layering": int (0–100),
            "notes": "Short paragraph with strengths and weaknesses"
        },
        "relative_rank": null,
        "user_verdict_override": null,
        "user_feedback": null,
        "learning_signal": null
        }
        """
        self.feedback_context_for_prompt: Optional[str] = None

    def customize_system_prompt(self, custom_prompt: str) -> None:
        """Update the base system prompt used for analysis.

        Args:
            custom_prompt: New system prompt for the GPT-4o analysis
        """
        self.base_system_prompt = custom_prompt
        # Clear any existing feedback context if the base prompt changes significantly
        self.clear_feedback_context()

    def set_feedback_context(self, feedback_summary: Optional[str]) -> None:
        """Set a feedback summary to be included in subsequent analysis prompts.

        Args:
            feedback_summary: A string summarizing user feedback from previous analyses.
                              Set to None to clear feedback.
        """
        if feedback_summary:
            # Cap feedback to avoid excessively long prompts
            max_chars = 4000
            trimmed_feedback = (
                feedback_summary[: max_chars - 3] + "..."
                if len(feedback_summary) > max_chars
                else feedback_summary
            )
            self.feedback_context_for_prompt = (
                "\n\n---\nImportant: Please learn from this recent user feedback to improve your grading:\n"
                f"{trimmed_feedback}\n---\n"
            )
        else:
            self.feedback_context_for_prompt = None

    def clear_feedback_context(self) -> None:
        """Clear any existing feedback context from the prompt."""
        self.feedback_context_for_prompt = None

    def analyze_image(
        self, base64_image: str, file_name: str, post_processed: bool = False
    ) -> Dict[str, Any]:
        """Analyze an image using GPT-4o.

        Args:
            base64_image: Base64-encoded image string
            file_name: Original filename of the image
            post_processed: Flag indicating if the image has been post-processed

        Returns:
            Dict: Analysis results in the specified JSON structure
        """
        # Create the user prompt with the image
        user_prompt = f"Analyze this landscape photograph. Filename: {file_name}."
        if post_processed:
            user_prompt += " Note: This image has been post-processed."

        current_system_prompt = self.base_system_prompt
        if self.feedback_context_for_prompt:
            current_system_prompt = self.feedback_context_for_prompt + self.base_system_prompt

        try:
            # Call the OpenAI API with the image
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": current_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
            )

            # Extract and parse the JSON response
            result = json.loads(response.choices[0].message.content)

            # Add filename to the result
            result["filename"] = file_name
            result["post_processed"] = post_processed

            # Initialize fields for user feedback
            result["user_verdict_override"] = None
            result["user_feedback"] = None
            result["learning_signal"] = None
            result["relative_rank"] = None

            return result

        except Exception as e:
            self._logger.error(f"Error analyzing image: {e}")
            # Return a basic error structure if analysis fails
            return {
                "filename": file_name,
                "verdict": "error",
                "score": 0,
                "error": str(e),
                "post_processed": post_processed,
                "user_verdict_override": None,
                "user_feedback": None,
                "learning_signal": None,
                "relative_rank": None,
            }

    def validate_analysis_result(self, result: Dict[str, Any]) -> bool:
        """Validate that the analysis result has the expected structure.

        Args:
            result: Analysis result dictionary to validate

        Returns:
            bool: True if the result has valid structure, False otherwise
        """
        required_fields = ["verdict", "score", "analysis"]
        if "error" in result:
            return False

        for field in required_fields:
            if field not in result:
                return False

        if "analysis" in result:
            analysis_fields = ["composition", "exposure", "subject", "layering", "notes"]
            for field in analysis_fields:
                if field not in result["analysis"]:
                    return False

        return True
