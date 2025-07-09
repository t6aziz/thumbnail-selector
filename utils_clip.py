"""
CLIP-based Frame Scoring Utilities

This module contains functions for scoring video frames using OpenAI's CLIP model
to determine how well they match a given text prompt (e.g., "clickbait thumbnail").
"""

import os
import torch
import clip
import sys
import time
from PIL import Image
from typing import List, Tuple, Optional, Dict


class CLIPScorer:
    """
    A class for scoring images using the CLIP model.
    
    This class loads the CLIP model once and provides methods to score
    individual images or batches of images against a text prompt.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize the CLIP scorer.
        
        Args:
            model_name: Name of the CLIP model to use (default: "ViT-B/32")
            device: Device to run the model on. If None, will auto-select GPU if available
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on device: {self.device}")
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model '{model_name}': {e}")
        
        self.model.eval()  # Set model to evaluation mode
    
    def score_image(self, image_path: str, prompt: str) -> float:
        """
        Score a single image against a text prompt using CLIP.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to compare against
            
        Returns:
            Similarity score between 0 and 1 (higher is better match)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be processed
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text prompt
            text_tensor = clip.tokenize([prompt]).to(self.device)
            
            # Get features from CLIP model
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tensor)
                
                # Normalize features to unit length
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_features @ text_features.T).item()
            
            return float(similarity)
            
        except Exception as e:
            raise ValueError(f"Failed to process image '{image_path}': {e}")
    
    def score_image_pil(self, pil_image: Image.Image, prompt: str) -> float:
        """
        Score a PIL image against a text prompt using CLIP.
        
        Args:
            pil_image: PIL Image object
            prompt: Text prompt to compare against
            
        Returns:
            Similarity score between 0 and 1 (higher is better match)
            
        Raises:
            ValueError: If image cannot be processed
        """
        try:
            # Ensure image is in RGB mode
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Preprocess image
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Tokenize text prompt
            text_tensor = clip.tokenize([prompt]).to(self.device)
            
            # Get features from CLIP model
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tensor)
                
                # Normalize features to unit length
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_features @ text_features.T).item()
            
            return float(similarity)
            
        except Exception as e:
            raise ValueError(f"Failed to process PIL image: {e}")


def score_with_clip(frames_data: List[Dict], prompt: str = "a clickbait YouTube thumbnail") -> Tuple[Dict, float]:
    """
    Score a list of frame images using CLIP and return the best one.
    
    Args:
        frames_data: List of frame dictionaries with PIL images and metadata
        prompt: Text prompt to score against (default: "a clickbait YouTube thumbnail")
        
    Returns:
        Tuple of (best_frame_dict, best_score)
        
    Raises:
        ValueError: If no frames are provided or no valid frames found
    """
    if not frames_data:
        raise ValueError("No frames provided")
    
    # Initialize CLIP scorer
    scorer = CLIPScorer()
    
    print(f"🔍 Scoring {len(frames_data)} frames with CLIP...")
    print(f"📝 Prompt: '{prompt}'")
    print("-" * 50)
    
    scores = []
    valid_frames = 0
    total_frames = len(frames_data)
    
    for i, frame_data in enumerate(frames_data):
        try:
            # Show progress every 25% or every 5 frames
            if (i + 1) % max(1, total_frames // 4) == 0 or (i + 1) % 5 == 0:
                progress = ((i + 1) / total_frames) * 100
                print(f"⏳ Processing frames with CLIP... {progress:.0f}% ({i + 1}/{total_frames})")
            
            # Score the PIL image directly
            score = scorer.score_image_pil(frame_data["frame"], prompt)
            frame_name = f"frame_{frame_data['frame_number']:04d}"
            scores.append((frame_data, score))
            valid_frames += 1
            
        except Exception as e:
            print(f"⚠️  Error processing frame {frame_data['frame_number']}: {e}")
            continue
    
    print(f"✅ Completed processing {valid_frames}/{total_frames} frames")
    
    if not scores:
        raise ValueError("No valid frames could be processed")
    
    # Find the best scoring frame
    best_frame_data, best_score = max(scores, key=lambda x: x[1])
    best_frame_name = f"frame_{best_frame_data['frame_number']:04d}"
    
    print("-" * 50)
    print(f"🏆 Best frame: {best_frame_name}")
    print(f"📊 Score: {best_score:.4f}")
    print(f"🕐 Timestamp: {best_frame_data['timestamp']:.1f}s")
    print(f"✅ Processed {valid_frames}/{len(frames_data)} frames successfully")
    
    return best_frame_data, best_score


def batch_score_with_clip(frame_paths: List[str], prompt: str = "a clickbait YouTube thumbnail") -> List[Tuple[str, float]]:
    """
    Score all frames and return a sorted list of (frame_path, score) tuples.
    
    Args:
        frame_paths: List of paths to frame image files
        prompt: Text prompt to score against
        
    Returns:
        List of (frame_path, score) tuples sorted by score (highest first)
    """
    if not frame_paths:
        return []
    
    scorer = CLIPScorer()
    scores = []
    
    for frame_path in frame_paths:
        try:
            score = scorer.score_image(frame_path, prompt)
            scores.append((frame_path, score))
        except Exception as e:
            print(f"⚠️  Error processing {frame_path}: {e}")
            continue
    
    # Sort by score (highest first)
    return sorted(scores, key=lambda x: x[1], reverse=True)
