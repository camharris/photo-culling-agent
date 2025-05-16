#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Gradio interface implementation for the Photo Culling Agent."""

import os
import tempfile
import time
import gradio as gr
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from PIL import Image, ExifTags
import json
import shutil
from pathlib import Path

from src.photo_culling_agent.langgraph_pipeline import PhotoCullingGraph


class PhotoCullingInterface:
    """Gradio interface for the Photo Culling Agent application."""
    
    def __init__(
        self,
        output_dir: str = "./output",
        decision_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the Gradio interface.
        
        Args:
            output_dir: Directory to save the output metadata
            decision_weights: Optional custom weights for decision criteria
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temporary directory for uploads
        self.temp_dir = tempfile.mkdtemp(prefix="photo_culling_")
        
        # Initialize LangGraph pipeline
        self.pipeline = PhotoCullingGraph(decision_weights=decision_weights)
        
        # Track processed images
        self.processed_images: Dict[str, Dict[str, Any]] = {}
        self.uploads_in_progress: Set[str] = set()
        
        # Create the interface
        self.interface = self._build_interface()
    
    def _build_interface(self) -> gr.Blocks:
        """Build the Gradio interface.
        
        Returns:
            gr.Blocks: The Gradio interface
        """
        with gr.Blocks(title="Photo Culling Agent") as interface:
            gr.Markdown("# Photo Culling Agent")
            gr.Markdown("Upload landscape photographs to analyze and categorize them as keep or toss.")
            
            with gr.Tab("Upload & Analyze"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Image upload component
                        file_upload = gr.File(
                            label="Upload Images",
                            file_types=["image"],
                            file_count="multiple",
                            type="filepath"
                        )
                        
                        # Analysis button
                        analyze_btn = gr.Button("Analyze Images", variant="primary")
                        
                        # Progress indicator
                        progress = gr.Textbox(
                            label="Processing Status",
                            placeholder="Upload images and click 'Analyze Images'",
                            interactive=False
                        )
                    
                    with gr.Column(scale=3):
                        # Results display
                        results_gallery = gr.Gallery(
                            label="Analyzed Images",
                            columns=3,
                            object_fit="contain",
                            height=500
                        )
                        
                        # Selected image details
                        with gr.Row():
                            selected_image = gr.Image(
                                label="Selected Image",
                                type="filepath",
                                height=400,
                                interactive=False
                            )
                            
                            with gr.Column():
                                verdict_box = gr.Textbox(label="Verdict")
                                confidence_box = gr.Textbox(label="Confidence")
                                score_box = gr.Textbox(label="Score")
                                notes_box = gr.Textbox(label="Analysis Notes", lines=5)
            
            with gr.Tab("Results Summary"):
                # Summary table and statistics
                with gr.Row():
                    results_table = gr.DataFrame(label="Analysis Results")
                
                with gr.Row():
                    with gr.Column():
                        # Counts by verdict and confidence level
                        verdict_counts = gr.BarPlot(
                            label="Images by Verdict",
                            x="category", 
                            y="count", 
                            title="Images by Verdict",
                            y_title="Number of Images",
                            x_title="Verdict"
                        )
                    
                    with gr.Column():
                        confidence_counts = gr.BarPlot(
                            label="Images by Confidence Level",
                            x="category", 
                            y="count", 
                            title="Images by Confidence Level",
                            y_title="Number of Images",
                            x_title="Confidence Level"
                        )
                
                export_btn = gr.Button("Export All Metadata")
                export_path = gr.Textbox(label="Export Path", interactive=False)
            
            # Event handlers
            file_upload.upload(
                fn=self.handle_upload,
                inputs=[file_upload],
                outputs=[progress, results_gallery]
            )
            
            analyze_btn.click(
                fn=self.analyze_images,
                inputs=[file_upload],
                outputs=[progress, results_gallery, results_table, verdict_counts, confidence_counts]
            )
            
            results_gallery.select(
                fn=self.show_image_details,
                outputs=[selected_image, verdict_box, confidence_box, score_box, notes_box]
            )
            
            export_btn.click(
                fn=self.export_metadata,
                outputs=[export_path]
            )
        
        return interface
    
    def handle_upload(self, files: List[str]) -> Tuple[str, List[Dict[str, str]]]:
        """Handle file uploads.
        
        Args:
            files: List of uploaded file paths
            
        Returns:
            Tuple[str, List[Dict[str, str]]]: Status message and gallery images
        """
        if not files:
            return "No files uploaded", []
        
        # Copy files to temporary directory
        copied_files = []
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.temp_dir, filename)
                shutil.copy2(file_path, dest_path)
                copied_files.append(dest_path)
            except Exception as e:
                return f"Error copying file {file_path}: {str(e)}", []
        
        # Mark all new files as in-progress
        for file_path in copied_files:
            self.uploads_in_progress.add(file_path)
        
        # Create gallery items for uploaded files
        gallery_items = [
            {"image": file_path, "label": os.path.basename(file_path)} 
            for file_path in copied_files
        ]
        
        return f"Uploaded {len(files)} images. Click 'Analyze Images' to process.", gallery_items
    
    def analyze_images(
        self,
        files: List[str]
    ) -> Tuple[str, List[Dict[str, str]], pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """Analyze uploaded images using the LangGraph pipeline.
        
        Args:
            files: List of uploaded file paths
            
        Returns:
            Tuple containing status, gallery items, results table, and chart data
        """
        if not files:
            return "No files to analyze", [], pd.DataFrame(), {"data": []}, {"data": []}
        
        # Process only files that haven't been processed yet
        to_process = [f for f in files if f in self.uploads_in_progress]
        
        if not to_process:
            return "All files already processed", self._get_gallery_items(), self._get_results_table(), self._get_verdict_chart(), self._get_confidence_chart()
        
        # Process files
        status_msg = f"Processing {len(to_process)} images..."
        processed_count = 0
        
        for file_path in to_process:
            try:
                # Update progress
                processed_count += 1
                status_msg = f"Processing image {processed_count}/{len(to_process)}: {os.path.basename(file_path)}"
                yield status_msg, self._get_gallery_items(), self._get_results_table(), self._get_verdict_chart(), self._get_confidence_chart()
                
                # Process the image
                result = self.pipeline.process_image(file_path)
                
                # Check for errors
                if result.get("error"):
                    print(f"Error processing {file_path}: {result['error']}")
                    continue
                
                # Store the result
                self.processed_images[file_path] = result
                
                # Remove from in-progress set
                self.uploads_in_progress.remove(file_path)
                
                # Update gallery
                yield (
                    f"Processed {processed_count}/{len(to_process)}", 
                    self._get_gallery_items(),
                    self._get_results_table(),
                    self._get_verdict_chart(),
                    self._get_confidence_chart()
                )
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                status_msg = f"Error processing {os.path.basename(file_path)}: {str(e)}"
                yield status_msg, self._get_gallery_items(), self._get_results_table(), self._get_verdict_chart(), self._get_confidence_chart()
        
        return (
            f"Processed {processed_count} images. {len(self.uploads_in_progress)} remaining.",
            self._get_gallery_items(),
            self._get_results_table(),
            self._get_verdict_chart(),
            self._get_confidence_chart()
        )
    
    def show_image_details(self, evt: gr.SelectData) -> Tuple[str, str, str, str, str]:
        """Show details for the selected image.
        
        Args:
            evt: Gallery selection event
            
        Returns:
            Tuple containing image details
        """
        # Get selected image data
        gallery_items = self._get_gallery_items()
        if not gallery_items or evt.index >= len(gallery_items):
            return None, "No selection", "N/A", "N/A", "No analysis available"
        
        selected_item = gallery_items[evt.index]
        file_path = selected_item["image"]
        
        # If image hasn't been processed yet
        if file_path not in self.processed_images:
            return file_path, "Not analyzed", "N/A", "N/A", "Image has not been analyzed yet"
        
        # Get the analysis result
        result = self.processed_images[file_path]
        
        # Extract details
        verdict = result.get("verdict", "unknown").upper()
        confidence_level = result.get("confidence_level", "N/A")
        confidence = result.get("confidence", 0) * 100
        
        # Format score
        analysis_result = result.get("analysis_result", {})
        score = analysis_result.get("score", 0)
        
        # Format details for display
        verdict_text = f"{verdict} ({confidence_level})"
        confidence_text = f"{confidence:.1f}%"
        score_text = f"{score}/100"
        
        # Get analysis notes
        rationale = result.get("decision_rationale", {})
        analysis_data = analysis_result.get("analysis", {})
        
        notes = analysis_data.get("notes", "No notes available")
        
        # Add criteria scores to notes
        if rationale:
            criteria_scores = rationale.get("criteria_scores", {})
            if criteria_scores:
                notes += "\n\nCriteria Scores:\n"
                for criterion, score in criteria_scores.items():
                    if criterion != "base_score":
                        notes += f"- {criterion.capitalize()}: {score}/100\n"
        
        return file_path, verdict_text, confidence_text, score_text, notes
    
    def export_metadata(self) -> str:
        """Export all metadata to a JSON file.
        
        Returns:
            str: Path to the exported file
        """
        if not self.processed_images:
            return "No images have been processed yet"
        
        # Export all metadata
        try:
            output_path = self.pipeline.export_metadata(self.output_dir)
            return f"Metadata exported to: {output_path}"
        except Exception as e:
            return f"Error exporting metadata: {str(e)}"
    
    def _get_gallery_items(self) -> List[Dict[str, str]]:
        """Get gallery items with appropriate labels based on analysis results.
        
        Returns:
            List[Dict[str, str]]: List of gallery items
        """
        gallery_items = []
        
        # First add processed images
        for file_path, result in self.processed_images.items():
            verdict = result.get("verdict", "unknown").upper()
            confidence_level = result.get("confidence_level", "")
            confidence = result.get("confidence", 0) * 100
            
            label = f"{os.path.basename(file_path)}\n{verdict} ({confidence_level})\nConfidence: {confidence:.1f}%"
            
            gallery_items.append({
                "image": file_path,
                "label": label
            })
        
        # Then add in-progress images
        for file_path in self.uploads_in_progress:
            gallery_items.append({
                "image": file_path,
                "label": f"{os.path.basename(file_path)}\n(Not analyzed yet)"
            })
        
        return gallery_items
    
    def _get_results_table(self) -> pd.DataFrame:
        """Create a DataFrame with analysis results.
        
        Returns:
            pd.DataFrame: Analysis results
        """
        if not self.processed_images:
            return pd.DataFrame()
        
        # Create list of results
        results = []
        for file_path, result in self.processed_images.items():
            filename = os.path.basename(file_path)
            verdict = result.get("verdict", "unknown").upper()
            confidence_level = result.get("confidence_level", "N/A")
            confidence = result.get("confidence", 0) * 100
            
            analysis_result = result.get("analysis_result", {})
            score = analysis_result.get("score", 0)
            
            # Get individual criteria scores
            criteria_scores = {}
            if "decision_rationale" in result:
                criteria_scores = result["decision_rationale"].get("criteria_scores", {})
            
            composition = criteria_scores.get("composition", 0)
            exposure = criteria_scores.get("exposure", 0)
            subject = criteria_scores.get("subject", 0)
            layering = criteria_scores.get("layering", 0)
            
            results.append({
                "Filename": filename,
                "Verdict": verdict,
                "Confidence Level": confidence_level,
                "Confidence": f"{confidence:.1f}%",
                "Score": score,
                "Composition": composition,
                "Exposure": exposure,
                "Subject": subject,
                "Layering": layering
            })
        
        return pd.DataFrame(results)
    
    def _get_verdict_chart(self) -> Dict[str, Any]:
        """Get verdict distribution chart data.
        
        Returns:
            Dict[str, Any]: Chart data
        """
        if not self.processed_images:
            return {"data": []}
        
        # Count verdicts
        verdicts = {"KEEP": 0, "TOSS": 0}
        
        for _, result in self.processed_images.items():
            verdict = result.get("verdict", "unknown").upper()
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        
        # Create chart data
        data = [
            {"category": category, "count": count}
            for category, count in verdicts.items()
        ]
        
        return {"data": data}
    
    def _get_confidence_chart(self) -> Dict[str, Any]:
        """Get confidence level distribution chart data.
        
        Returns:
            Dict[str, Any]: Chart data
        """
        if not self.processed_images:
            return {"data": []}
        
        # Count confidence levels
        confidence_levels = {}
        
        for _, result in self.processed_images.items():
            level = result.get("confidence_level", "UNKNOWN")
            confidence_levels[level] = confidence_levels.get(level, 0) + 1
        
        # Create chart data
        data = [
            {"category": category, "count": count}
            for category, count in confidence_levels.items()
        ]
        
        return {"data": data}
    
    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface.
        
        Args:
            **kwargs: Additional arguments to pass to gradio.launch()
        """
        self.interface.launch(**kwargs)
    
    def __del__(self):
        """Clean up temporary files on deletion."""
        try:
            # Clean up temp directory
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp directory: {e}") 