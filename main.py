#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Photo Culling Agent - Main Entry Point.

This script serves as the entry point for the Photo Culling Agent application,
which analyzes landscape photographs using GPT-4o via LangGraph and provides
a Gradio interface for human-in-the-loop review.
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from src.photo_culling_agent.image_processor import ImageProcessor
from src.photo_culling_agent.gpt_analyzer import GPTAnalyzer
from src.photo_culling_agent.metadata_manager import MetadataManager
from src.photo_culling_agent.langgraph_pipeline import PhotoCullingGraph

# Load environment variables from .env file
load_dotenv()


def process_single_image(image_path: str, output_dir: str, custom_weights: Optional[Dict[str, float]] = None) -> None:
    """Process a single image through the LangGraph pipeline.
    
    Args:
        image_path: Path to the image file to process
        output_dir: Directory to save the metadata output
        custom_weights: Optional custom weights for decision criteria
    """
    print(f"Processing image: {image_path}")
    
    # Initialize the LangGraph pipeline
    pipeline = PhotoCullingGraph(decision_weights=custom_weights)
    
    # Process the image
    result = pipeline.process_image(image_path)
    
    # Check for errors
    if result.get("error"):
        print(f"Error processing image: {result['error']}")
        return
    
    # Print the verdict and confidence information
    verdict = result.get("verdict")
    confidence_level = result.get("confidence_level")
    confidence = result.get("confidence", 0.0) * 100  # Convert to percentage
    
    print(f"Verdict: {verdict.upper()} ({confidence_level}) - Confidence: {confidence:.1f}%")
    
    # Print decision rationale
    rationale = result.get("decision_rationale", {})
    if rationale:
        print("\nDecision Details:")
        print(f"- Weighted Score: {rationale.get('weighted_score', 0):.1f}/100")
        print(f"- Original GPT Verdict: {rationale.get('original_verdict', 'unknown')}")
        print(f"- Final Verdict: {rationale.get('final_verdict', 'unknown')}")
        
        if rationale.get("notes"):
            print(f"- Notes: {rationale.get('notes')}")
        
        # Print individual criteria scores
        criteria_scores = rationale.get("criteria_scores", {})
        if criteria_scores:
            print("\nCriteria Scores:")
            for criterion, score in criteria_scores.items():
                if criterion != "base_score":
                    print(f"- {criterion.capitalize()}: {score}/100")
                else:
                    print(f"- Overall: {score}/100")
    
    # Get the filename for the metadata
    filename = os.path.basename(image_path)
    
    # Export the metadata
    os.makedirs(output_dir, exist_ok=True)
    output_path = pipeline.export_metadata(output_dir, filename)
    print(f"\nMetadata exported to: {output_path}")


def process_batch(image_dir: str, output_dir: str, custom_weights: Optional[Dict[str, float]] = None) -> None:
    """Process a batch of images through the LangGraph pipeline.
    
    Args:
        image_dir: Directory containing images to process
        output_dir: Directory to save the metadata output
        custom_weights: Optional custom weights for decision criteria
    """
    print(f"Processing images from directory: {image_dir}")
    
    # Initialize the LangGraph pipeline
    pipeline = PhotoCullingGraph(decision_weights=custom_weights)
    
    # Get all image files in the directory
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process")
    
    # Counters for confidence levels
    confidence_counts = {
        "DEFINITE_KEEP": 0,
        "LIKELY_KEEP": 0,
        "BORDERLINE": 0,
        "LIKELY_TOSS": 0,
        "DEFINITE_TOSS": 0
    }
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        # Process the image
        result = pipeline.process_image(image_path)
        
        # Check for errors
        if result.get("error"):
            print(f"  Error: {result['error']}")
            continue
        
        # Get and print verdict information
        verdict = result.get("verdict", "unknown")
        confidence_level = result.get("confidence_level", "unknown")
        confidence = result.get("confidence", 0.0) * 100  # Convert to percentage
        
        print(f"  Verdict: {verdict.upper()} ({confidence_level}) - Confidence: {confidence:.1f}%")
        
        # Update confidence level counts
        if confidence_level in confidence_counts:
            confidence_counts[confidence_level] += 1
    
    # Export all metadata
    os.makedirs(output_dir, exist_ok=True)
    output_path = pipeline.export_metadata(output_dir)
    print(f"All metadata exported to: {output_path}")
    
    # Print summary
    keep_images = pipeline.get_keep_images()
    toss_images = pipeline.get_toss_images()
    print("\nProcessing Summary:")
    print(f"Total images: {len(image_files)}")
    print(f"Keep images: {len(keep_images)}")
    print(f"Toss images: {len(toss_images)}")
    
    # Print confidence level breakdown
    print("\nConfidence Level Breakdown:")
    for level, count in confidence_counts.items():
        if count > 0:
            percentage = (count / len(image_files)) * 100
            print(f"- {level}: {count} images ({percentage:.1f}%)")
    
    # Print borderline cases that might need review
    if confidence_counts["BORDERLINE"] > 0:
        print("\nNote: Borderline cases may benefit from manual review.")


def parse_weights(weights_str: str) -> Dict[str, float]:
    """Parse a weights string into a dictionary.
    
    Format: "composition=1.0,exposure=0.8,subject=1.2,layering=0.9,base_score=1.0"
    
    Args:
        weights_str: String representation of weights
        
    Returns:
        Dict[str, float]: Dictionary of weights
    """
    weights = {}
    pairs = weights_str.split(',')
    
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=')
            try:
                weights[key.strip()] = float(value.strip())
            except ValueError:
                print(f"Warning: Invalid weight value '{value}' for '{key}'. Using default.")
    
    return weights


def main() -> None:
    """Run the Photo Culling Agent application.
    
    This function initializes the application components and processes images.
    """
    # Initialize core components
    print("Photo Culling Agent - Starting application...")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Photo Culling Agent")
    
    # Add arguments
    parser.add_argument("--image", type=str, help="Path to single image to process")
    parser.add_argument("--dir", type=str, help="Directory containing images to process")
    parser.add_argument("--output", type=str, default="./output", help="Output directory for metadata")
    parser.add_argument("--weights", type=str, help="Custom weights for decision criteria (format: composition=1.0,exposure=0.8,subject=1.2,layering=0.9,base_score=1.0)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check arguments
    if not args.image and not args.dir:
        parser.print_help()
        print("\nError: Either --image or --dir must be specified")
        print("\nAlternatively, you can use the Gradio web interface with:")
        print("  python run_gradio.py")
        sys.exit(1)
    
    if args.image and args.dir:
        print("Error: Only one of --image or --dir should be specified")
        sys.exit(1)
    
    # Parse custom weights if provided
    custom_weights = None
    if args.weights:
        custom_weights = parse_weights(args.weights)
        print("Using custom decision weights:")
        for key, value in custom_weights.items():
            print(f"- {key}: {value}")
    
    # Process based on arguments
    try:
        if args.image:
            # Process a single image
            process_single_image(args.image, args.output, custom_weights)
        else:
            # Process a batch of images
            process_batch(args.dir, args.output, custom_weights)
        
        print("Processing complete!")
        print("\nTip: For an interactive web interface with more features, run:")
        print("  python run_gradio.py")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 