#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""GPT-4o analyzer for image evaluation in the Photo Culling Agent."""

import os
from typing import Dict, List, Optional, Any, Tuple, Union
import json
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
        
        # Default system prompt for photo analysis
        self.system_prompt = """
        You are a professional landscape photographer analyzing images. Your task is to:
        1. Evaluate the image for artistic and technical quality
        2. Assign a score from 0-100
        3. Provide verdict: 'keep' or 'toss'
        4. Score specific aspects: composition, exposure, subject, layering
        5. Detect and note approximate location if identifiable
        6. Generate descriptive tags
        7. Provide brief notes about strengths and weaknesses
        
        Format your response as valid JSON with the following structure:
        {
          "verdict": "keep" or "toss",
          "score": float from 0-100,
          "rating": "X stars" (1-5 stars, can use half stars),
          "tags": [list of descriptive tags],
          "location": "Approximate location if identifiable or null",
          "analysis": {
            "composition": int from 0-100,
            "exposure": int from 0-100,
            "subject": int from 0-100,
            "layering": int from 0-100,
            "notes": "Brief evaluation notes"
          }
        }
        """
    
    def customize_system_prompt(self, custom_prompt: str) -> None:
        """Update the system prompt used for analysis.
        
        Args:
            custom_prompt: New system prompt for the GPT-4o analysis
        """
        self.system_prompt = custom_prompt
    
    def analyze_image(self, base64_image: str, file_name: str, post_processed: bool = False) -> Dict[str, Any]:
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
        
        try:
            # Call the OpenAI API with the image
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"}
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
            print(f"Error analyzing image: {e}")
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
                "relative_rank": None
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