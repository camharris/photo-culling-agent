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


def process_single_image(image_path: str, output_dir: str) -> None:
    """Process a single image through the LangGraph pipeline.
    
    Args:
        image_path: Path to the image file to process
        output_dir: Directory to save the metadata output
    """
    print(f"Processing image: {image_path}")
    
    # Initialize the LangGraph pipeline
    pipeline = PhotoCullingGraph()
    
    # Process the image
    result = pipeline.process_image(image_path)
    
    # Check for errors
    if result.get("error"):
        print(f"Error processing image: {result['error']}")
        return
    
    # Print the verdict
    verdict = result.get("verdict")
    print(f"Verdict: {verdict}")
    
    # Get the filename for the metadata
    filename = os.path.basename(image_path)
    
    # Export the metadata
    os.makedirs(output_dir, exist_ok=True)
    output_path = pipeline.export_metadata(output_dir, filename)
    print(f"Metadata exported to: {output_path}")


def process_batch(image_dir: str, output_dir: str) -> None:
    """Process a batch of images through the LangGraph pipeline.
    
    Args:
        image_dir: Directory containing images to process
        output_dir: Directory to save the metadata output
    """
    print(f"Processing images from directory: {image_dir}")
    
    # Initialize the LangGraph pipeline
    pipeline = PhotoCullingGraph()
    
    # Get all image files in the directory
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        # Process the image
        result = pipeline.process_image(image_path)
        
        # Check for errors
        if result.get("error"):
            print(f"  Error: {result['error']}")
            continue
        
        # Print the verdict
        verdict = result.get("verdict")
        print(f"  Verdict: {verdict}")
    
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check arguments
    if not args.image and not args.dir:
        parser.print_help()
        print("\nError: Either --image or --dir must be specified")
        sys.exit(1)
    
    if args.image and args.dir:
        print("Error: Only one of --image or --dir should be specified")
        sys.exit(1)
    
    # Process based on arguments
    try:
        if args.image:
            # Process a single image
            process_single_image(args.image, args.output)
        else:
            # Process a batch of images
            process_batch(args.dir, args.output)
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 