#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Metadata management functionality for the Photo Culling Agent."""

import os
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime


class MetadataManager:
    """Manages image metadata, categorization, and storage."""

    def __init__(self):
        """Initialize the MetadataManager."""
        # Store metadata by image filename
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        # Track categorization
        self.keep_images: List[str] = []
        self.toss_images: List[str] = []
        self.error_images: List[str] = []
    
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add or update image metadata.
        
        Args:
            metadata: Image metadata dictionary
        """
        if "filename" not in metadata:
            raise ValueError("Metadata must contain a filename")
        
        filename = metadata["filename"]
        self.metadata_store[filename] = metadata
        
        # Categorize based on verdict
        self._update_categorization(filename)
    
    def _update_categorization(self, filename: str) -> None:
        """Update image categorization based on verdict.
        
        Args:
            filename: Image filename to categorize
        """
        # Remove from all categories first
        if filename in self.keep_images:
            self.keep_images.remove(filename)
        if filename in self.toss_images:
            self.toss_images.remove(filename)
        if filename in self.error_images:
            self.error_images.remove(filename)
        
        # Add to appropriate category
        metadata = self.metadata_store[filename]
        verdict = metadata.get("verdict")
        
        # If user has overridden the verdict, use that instead
        if metadata.get("user_verdict_override"):
            verdict = metadata["user_verdict_override"]
        
        if verdict == "keep":
            self.keep_images.append(filename)
        elif verdict == "toss":
            self.toss_images.append(filename)
        else:  # Error or unknown verdict
            self.error_images.append(filename)
    
    def get_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific image.
        
        Args:
            filename: Image filename
            
        Returns:
            Dict or None: Image metadata or None if not found
        """
        return self.metadata_store.get(filename)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored metadata.
        
        Returns:
            Dict: All image metadata indexed by filename
        """
        return self.metadata_store
    
    def get_keep_images(self) -> List[str]:
        """Get list of filenames categorized as 'keep'.
        
        Returns:
            List[str]: Filenames of images to keep
        """
        return self.keep_images
    
    def get_toss_images(self) -> List[str]:
        """Get list of filenames categorized as 'toss'.
        
        Returns:
            List[str]: Filenames of images to toss
        """
        return self.toss_images
    
    def get_error_images(self) -> List[str]:
        """Get list of filenames that had errors during processing.
        
        Returns:
            List[str]: Filenames of images with errors
        """
        return self.error_images
    
    def update_user_verdict(self, filename: str, verdict: str) -> None:
        """Update the user verdict override for an image.
        
        Args:
            filename: Image filename
            verdict: New verdict ('keep' or 'toss')
        """
        if filename not in self.metadata_store:
            raise ValueError(f"No metadata for {filename}")
        
        if verdict not in ["keep", "toss"]:
            raise ValueError(f"Invalid verdict: {verdict}. Must be 'keep' or 'toss'")
        
        self.metadata_store[filename]["user_verdict_override"] = verdict
        self._update_categorization(filename)
    
    def add_user_feedback(self, filename: str, feedback: str) -> None:
        """Add user feedback for an image.
        
        Args:
            filename: Image filename
            feedback: User feedback text
        """
        if filename not in self.metadata_store:
            raise ValueError(f"No metadata for {filename}")
        
        self.metadata_store[filename]["user_feedback"] = feedback
    
    def export_metadata_to_json(self, output_dir: str, filename: Optional[str] = None) -> str:
        """Export metadata to JSON file(s).
        
        Args:
            output_dir: Directory to save JSON files
            filename: Specific image filename to export, or None for all
            
        Returns:
            str: Path to the exported JSON file or directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename:
            # Export single image metadata
            if filename not in self.metadata_store:
                raise ValueError(f"No metadata for {filename}")
            
            metadata = self.metadata_store[filename]
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{timestamp}.json")
            
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_file
        else:
            # Export all metadata
            output_file = os.path.join(output_dir, f"all_metadata_{timestamp}.json")
            
            with open(output_file, 'w') as f:
                json.dump(self.metadata_store, f, indent=2)
            
            return output_file 