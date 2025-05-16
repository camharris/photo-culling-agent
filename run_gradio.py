#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script to launch the Gradio interface for the Photo Culling Agent."""

import argparse
import json
from src.photo_culling_agent.gradio_interface import PhotoCullingInterface
from dotenv import load_dotenv

def main():
    """Main function to parse arguments and launch the interface."""
    load_dotenv()  # Load environment variables from .env

    parser = argparse.ArgumentParser(description="Launch the Photo Culling Agent Gradio Interface.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_gradio",
        help="Directory to save output metadata. Defaults to ./output_gradio.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="JSON string of decision weights (e.g., '{\"composition\": 1.5, \"exposure\": 0.8}')",
        default=None,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio sharing link (creates a public URL).",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="127.0.0.1",
        help="Server name to run the Gradio app on.",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port to run the Gradio app on.",
    )

    args = parser.parse_args()

    decision_weights = None
    if args.weights:
        try:
            decision_weights = json.loads(args.weights)
        except json.JSONDecodeError:
            print("Error: Invalid JSON string for --weights. Using default weights.")

    interface = PhotoCullingInterface(
        output_dir=args.output_dir,
        decision_weights=decision_weights
    )
    print(f"Launching Gradio interface on http://{args.server_name}:{args.server_port}")
    if args.share:
        print("Gradio share link will be generated.")
    
    interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )

if __name__ == "__main__":
    main() 