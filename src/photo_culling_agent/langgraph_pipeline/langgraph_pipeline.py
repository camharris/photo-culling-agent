#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""LangGraph pipeline implementation for the Photo Culling Agent."""

import os
from typing import Dict, List, Optional, Any, Tuple, Union, TypedDict, Annotated, NamedTuple
from enum import Enum, auto
import json
import math

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.photo_culling_agent.image_processor import ImageProcessor
from src.photo_culling_agent.gpt_analyzer import GPTAnalyzer
from src.photo_culling_agent.metadata_manager import MetadataManager


# Define decision confidence levels
class ConfidenceLevel(Enum):
    """Confidence levels for keep/toss decisions."""
    DEFINITE_KEEP = auto()
    LIKELY_KEEP = auto()
    BORDERLINE = auto()
    LIKELY_TOSS = auto()
    DEFINITE_TOSS = auto()


# Define the state structure for the workflow
class PhotoCullingState(TypedDict):
    """State for the Photo Culling workflow.
    
    The state stores information about the current image being processed,
    analysis results, and any user feedback.
    """
    # Input data
    image_path: Optional[str]
    base64_image: Optional[str]
    image_metadata: Optional[Dict[str, Any]]
    
    # Analysis data
    analysis_result: Optional[Dict[str, Any]]
    
    # Decision data
    verdict: Optional[str]  # "keep", "toss", or None if not decided yet
    confidence: Optional[float]  # Confidence score (0.0-1.0)
    confidence_level: Optional[str]  # ConfidenceLevel as string
    decision_rationale: Optional[Dict[str, Any]]  # Reasoning behind verdict
    
    # User feedback
    user_feedback: Optional[str]
    user_verdict_override: Optional[str]  # "keep", "toss", or None if not overridden
    
    # Comparative analysis data
    similar_images: Optional[List[str]]  # List of similar image paths (for future)
    relative_ranking: Optional[int]  # Ranking within similar image group (for future)
    
    # Error handling
    error: Optional[str]
    
    # Execution status
    completed: bool


# Define scoring weights for different criteria
DEFAULT_WEIGHTS = {
    "composition": 1.0,
    "exposure": 0.9,
    "subject": 1.0,
    "layering": 0.8,
    "base_score": 1.2,  # Weight for the overall score
}

# Define decision thresholds
DECISION_THRESHOLDS = {
    "keep": 70.0,  # Base threshold for keep verdict
    "definite_keep": 85.0,
    "likely_keep": 75.0,
    "borderline": 65.0,
    "likely_toss": 50.0,
    "definite_toss": 30.0,
}


# Workflow node implementations
def process_image(state: PhotoCullingState, image_processor: ImageProcessor) -> PhotoCullingState:
    """Process an image and prepare it for analysis.
    
    Args:
        state: Current workflow state
        image_processor: ImageProcessor instance
        
    Returns:
        Updated workflow state
    """
    try:
        # Get the image path from state
        image_path = state.get("image_path")
        if not image_path:
            return {
                **state,
                "error": "No image path provided",
                "completed": False
            }
        
        # Validate and process the image
        if not image_processor.validate_image(image_path):
            return {
                **state,
                "error": f"Invalid image: {image_path}",
                "completed": False
            }
        
        # Prepare the image for analysis
        base64_image, image_metadata = image_processor.prepare_image_for_analysis(image_path)
        if base64_image is None or image_metadata is None:
            return {
                **state,
                "error": f"Failed to prepare image for analysis: {image_path}",
                "completed": False
            }
        
        # Update state with processed image data
        return {
            **state,
            "base64_image": base64_image,
            "image_metadata": image_metadata,
            "error": None,
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error processing image: {str(e)}",
            "completed": False
        }


def analyze_image(state: PhotoCullingState, gpt_analyzer: GPTAnalyzer) -> PhotoCullingState:
    """Analyze an image using GPT-4o.
    
    Args:
        state: Current workflow state
        gpt_analyzer: GPTAnalyzer instance
        
    Returns:
        Updated workflow state
    """
    try:
        # Check for errors from previous steps
        if state.get("error"):
            return state
        
        # Get required data from state
        base64_image = state.get("base64_image")
        image_path = state.get("image_path")
        
        if not base64_image or not image_path:
            return {
                **state,
                "error": "Missing required data for analysis",
                "completed": False
            }
        
        # Get whether the image has been post-processed
        image_metadata = state.get("image_metadata", {})
        post_processed = image_metadata.get("post_processed", False)
        
        # Get the filename from the path
        file_name = os.path.basename(image_path)
        
        # Analyze the image
        analysis_result = gpt_analyzer.analyze_image(
            base64_image=base64_image,
            file_name=file_name,
            post_processed=post_processed
        )
        
        # Validate the analysis result
        if not gpt_analyzer.validate_analysis_result(analysis_result):
            return {
                **state,
                "error": "Invalid analysis result",
                "completed": False
            }
        
        # Update state with analysis result
        return {
            **state,
            "analysis_result": analysis_result,
            # Initial verdict from GPT, will be refined in decide_verdict
            "verdict": analysis_result.get("verdict"),
            "error": None,
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error analyzing image: {str(e)}",
            "completed": False
        }


def decide_verdict(state: PhotoCullingState, weights: Optional[Dict[str, float]] = None) -> PhotoCullingState:
    """Decide the final verdict based on analysis result with weighted scoring.
    
    Args:
        state: Current workflow state
        weights: Optional custom weights for different criteria
        
    Returns:
        Updated workflow state with enhanced decision
    """
    try:
        # Check for errors from previous steps
        if state.get("error"):
            return state
        
        # Get analysis result from state
        analysis_result = state.get("analysis_result")
        if not analysis_result:
            return {
                **state,
                "error": "No analysis result available for decision",
                "completed": False
            }
        
        # Use default weights if not provided
        scoring_weights = weights or DEFAULT_WEIGHTS
        
        # Extract scores from analysis
        score = analysis_result.get("score", 0)
        analysis_details = analysis_result.get("analysis", {})
        
        composition_score = analysis_details.get("composition", 0)
        exposure_score = analysis_details.get("exposure", 0)
        subject_score = analysis_details.get("subject", 0)
        layering_score = analysis_details.get("layering", 0)
        
        # Calculate weighted score
        weighted_score = (
            score * scoring_weights.get("base_score", 1.0) +
            composition_score * scoring_weights.get("composition", 1.0) +
            exposure_score * scoring_weights.get("exposure", 1.0) +
            subject_score * scoring_weights.get("subject", 1.0) +
            layering_score * scoring_weights.get("layering", 1.0)
        )
        
        # Normalize to 0-100 scale
        total_weight = (
            scoring_weights.get("base_score", 1.0) +
            scoring_weights.get("composition", 1.0) +
            scoring_weights.get("exposure", 1.0) +
            scoring_weights.get("subject", 1.0) +
            scoring_weights.get("layering", 1.0)
        )
        
        normalized_score = weighted_score / total_weight
        
        # Determine verdict based on thresholds
        verdict = "keep" if normalized_score >= DECISION_THRESHOLDS["keep"] else "toss"
        
        # Determine confidence level
        confidence_level = None
        if normalized_score >= DECISION_THRESHOLDS["definite_keep"]:
            confidence_level = ConfidenceLevel.DEFINITE_KEEP.name
        elif normalized_score >= DECISION_THRESHOLDS["likely_keep"]:
            confidence_level = ConfidenceLevel.LIKELY_KEEP.name
        elif normalized_score >= DECISION_THRESHOLDS["borderline"]:
            confidence_level = ConfidenceLevel.BORDERLINE.name
        elif normalized_score >= DECISION_THRESHOLDS["likely_toss"]:
            confidence_level = ConfidenceLevel.LIKELY_TOSS.name
        else:
            confidence_level = ConfidenceLevel.DEFINITE_TOSS.name
        
        # Calculate confidence score (0.0-1.0)
        # Distance from borderline threshold normalized to 0.5-1.0 range
        borderline = DECISION_THRESHOLDS["borderline"]
        if normalized_score >= borderline:
            # For keep verdicts: how far above borderline, normalized to 0.5-1.0
            max_keep_score = 100.0
            confidence = 0.5 + 0.5 * ((normalized_score - borderline) / (max_keep_score - borderline))
        else:
            # For toss verdicts: how far below borderline, normalized to 0.0-0.5
            min_score = 0.0
            confidence = 0.5 * (1.0 - ((borderline - normalized_score) / (borderline - min_score)))
        
        # Clamp confidence to 0.0-1.0 range
        confidence = max(0.0, min(1.0, confidence))
        
        # Generate decision rationale
        decision_rationale = {
            "weighted_score": normalized_score,
            "original_verdict": analysis_result.get("verdict"),
            "final_verdict": verdict,
            "criteria_scores": {
                "composition": composition_score,
                "exposure": exposure_score,
                "subject": subject_score,
                "layering": layering_score,
                "base_score": score
            },
            "criteria_weights": scoring_weights,
            "threshold_applied": DECISION_THRESHOLDS["keep"],
            "notes": "",
        }
        
        # Add explanatory notes to rationale
        if verdict != analysis_result.get("verdict"):
            decision_rationale["notes"] = f"Final verdict differs from initial GPT verdict due to weighted scoring."
        
        if confidence_level == ConfidenceLevel.BORDERLINE.name:
            decision_rationale["notes"] += " This is a borderline case that may benefit from human review."
        
        # Update the analysis result with enhanced decision data
        analysis_result["final_verdict"] = verdict
        analysis_result["confidence"] = confidence
        analysis_result["confidence_level"] = confidence_level
        analysis_result["decision_rationale"] = decision_rationale
        
        # Update state with decision data
        return {
            **state,
            "analysis_result": analysis_result,
            "verdict": verdict,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "decision_rationale": decision_rationale,
            "error": None,
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error deciding verdict: {str(e)}",
            "completed": False
        }


def comparative_analysis(state: PhotoCullingState) -> PhotoCullingState:
    """Perform comparative analysis for similar images (placeholder for future implementation).
    
    This will compare the current image with similar images to make relative decisions.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated workflow state
    """
    # Placeholder for future implementation
    # Currently just passes the state through without modification
    
    # For future implementation:
    # 1. Group similar images (e.g., burst shots, same subject)
    # 2. Rank images within each group
    # 3. Apply "keep best N" logic for each group
    # 4. Adjust verdicts based on relative ranking
    
    return state


def update_metadata(state: PhotoCullingState, metadata_manager: MetadataManager) -> PhotoCullingState:
    """Update metadata based on analysis result.
    
    Args:
        state: Current workflow state
        metadata_manager: MetadataManager instance
        
    Returns:
        Updated workflow state
    """
    try:
        # Check for errors from previous steps
        if state.get("error"):
            return state
        
        # Get analysis result from state
        analysis_result = state.get("analysis_result")
        if not analysis_result:
            return {
                **state,
                "error": "No analysis result available",
                "completed": False
            }
        
        # Update user feedback and verdict override if present in state
        user_feedback = state.get("user_feedback")
        user_verdict_override = state.get("user_verdict_override")
        
        if user_feedback is not None:
            analysis_result["user_feedback"] = user_feedback
            
        if user_verdict_override is not None:
            analysis_result["user_verdict_override"] = user_verdict_override
        
        # Add metadata to the store
        metadata_manager.add_metadata(analysis_result)
        
        # Update state to indicate completion
        return {
            **state,
            "completed": True,
            "error": None,
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error updating metadata: {str(e)}",
            "completed": False
        }


def should_end_workflow(state: PhotoCullingState) -> bool:
    """Determine if the workflow should end.
    
    Args:
        state: Current workflow state
        
    Returns:
        bool: True if the workflow should end, False otherwise
    """
    # End if there's an error or if processing is completed
    return state.get("error") is not None or state.get("completed", False)


class PhotoCullingGraph:
    """LangGraph implementation of the Photo Culling workflow."""
    
    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        gpt_analyzer: Optional[GPTAnalyzer] = None,
        metadata_manager: Optional[MetadataManager] = None,
        decision_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the Photo Culling Graph.
        
        Args:
            image_processor: ImageProcessor instance (creates a new one if None)
            gpt_analyzer: GPTAnalyzer instance (creates a new one if None)
            metadata_manager: MetadataManager instance (creates a new one if None)
            decision_weights: Optional custom weights for verdict decisions
        """
        # Initialize components if not provided
        self.image_processor = image_processor or ImageProcessor()
        self.gpt_analyzer = gpt_analyzer or GPTAnalyzer()
        self.metadata_manager = metadata_manager or MetadataManager()
        self.decision_weights = decision_weights or DEFAULT_WEIGHTS
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph.
        
        Returns:
            StateGraph: The constructed graph
        """
        # Create a new state graph
        builder = StateGraph(PhotoCullingState)
        
        # Define the nodes
        builder.add_node("process_image", lambda state: process_image(state, self.image_processor))
        builder.add_node("analyze_image", lambda state: analyze_image(state, self.gpt_analyzer))
        builder.add_node("decide_verdict", lambda state: decide_verdict(state, self.decision_weights))
        builder.add_node("comparative_analysis", comparative_analysis)
        builder.add_node("update_metadata", lambda state: update_metadata(state, self.metadata_manager))
        
        # Define the edges (workflow transitions)
        builder.set_entry_point("process_image")
        builder.add_edge("process_image", "analyze_image")
        builder.add_edge("analyze_image", "decide_verdict")
        builder.add_edge("decide_verdict", "comparative_analysis") 
        builder.add_edge("comparative_analysis", "update_metadata")
        
        # Define conditional edges
        builder.add_conditional_edges(
            "update_metadata",
            should_end_workflow,
            {
                True: END,  # End the workflow if should_end_workflow returns True
                False: "process_image",  # Loop back to process_image if should_end_workflow returns False
            }
        )
        
        # Compile the graph
        return builder.compile()
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image through the workflow.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict: The final workflow state
        """
        # Initialize state
        initial_state: PhotoCullingState = {
            "image_path": image_path,
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
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        return final_state
    
    def provide_feedback(self, image_path: str, feedback: str, verdict_override: Optional[str] = None) -> Dict[str, Any]:
        """Process an image with user feedback.
        
        Args:
            image_path: Path to the image file
            feedback: User feedback text
            verdict_override: Optional user verdict override ("keep" or "toss")
            
        Returns:
            Dict: The final workflow state
        """
        # Initialize state with feedback
        initial_state: PhotoCullingState = {
            "image_path": image_path,
            "base64_image": None,
            "image_metadata": None,
            "analysis_result": None,
            "verdict": None,
            "confidence": None,
            "confidence_level": None,
            "decision_rationale": None,
            "user_feedback": feedback,
            "user_verdict_override": verdict_override,
            "similar_images": None,
            "relative_ranking": None,
            "error": None,
            "completed": False
        }
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        return final_state
    
    def get_keep_images(self) -> List[str]:
        """Get list of images categorized as 'keep'.
        
        Returns:
            List[str]: Filenames of images to keep
        """
        return self.metadata_manager.get_keep_images()
    
    def get_toss_images(self) -> List[str]:
        """Get list of images categorized as 'toss'.
        
        Returns:
            List[str]: Filenames of images to toss
        """
        return self.metadata_manager.get_toss_images()
    
    def get_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific image.
        
        Args:
            filename: Image filename
            
        Returns:
            Dict or None: Image metadata or None if not found
        """
        return self.metadata_manager.get_metadata(filename)
    
    def export_metadata(self, output_dir: str, filename: Optional[str] = None) -> str:
        """Export metadata to JSON file(s).
        
        Args:
            output_dir: Directory to save JSON files
            filename: Specific image filename to export, or None for all
            
        Returns:
            str: Path to the exported JSON file or directory
        """
        return self.metadata_manager.export_metadata_to_json(output_dir, filename) 