#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Photo Culling Agent - Main Entry Point.

This script serves as the entry point for the Photo Culling Agent application,
which analyzes landscape photographs using GPT-4o via LangGraph and provides
a Gradio interface for human-in-the-loop review.
"""

import os
import sys
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from src.photo_culling_agent.image_processor import ImageProcessor
from src.photo_culling_agent.gpt_analyzer import GPTAnalyzer
from src.photo_culling_agent.metadata_manager import MetadataManager

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    """Run the Photo Culling Agent application.
    
    This function initializes the application components and starts the Gradio interface.
    """
    # Initialize core components
    print("Photo Culling Agent - Starting application...")
    
    # Initialize the core components
    try:
        image_processor = ImageProcessor()
        print("✓ Image processor initialized")
        
        gpt_analyzer = GPTAnalyzer()
        print("✓ GPT analyzer initialized")
        
        metadata_manager = MetadataManager()
        print("✓ Metadata manager initialized")
        
        # TODO: In Phase 2, implement the LangGraph pipeline
        # TODO: In Phase 3, implement the Gradio interface
        
        print("All Phase 1 core components initialized successfully")
        print("Phase 1 implementation complete - Core Analysis Pipeline is ready")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 