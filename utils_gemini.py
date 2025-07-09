"""
Gemini-based Frame Scoring Utilities

This module contains functions for scoring video frames using Google's Gemini model
to determine which frame would make the best thumbnail through visual analysis.
"""

import os
import re
import sys
import time
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image


class GeminiScorer:
    """
    A class for scoring images using Google's Gemini model.
    
    This class handles batched evaluation of images where Gemini analyzes
    multiple frames at once to select the best thumbnail candidate.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        """
        Initialize the Gemini scorer.
        
        Args:
            model_name: Name of the Gemini model to use (default: "gemini-2.5-pro")
            
        Raises:
            RuntimeError: If API key is not found or model initialization fails
        """
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment variables")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model '{model_name}': {e}")
    
    def score_batch_data(self, frames_data: List[Dict], batch_size: int = 16) -> Optional[Dict]:
        """
        Score a batch of frame data and return the best one according to Gemini.
        
        Args:
            frames_data: List of frame dictionaries with PIL images and metadata
            batch_size: Maximum number of frames to process at once
            
        Returns:
            Best frame data dict according to Gemini, or None if no clear winner
        """
        if not frames_data:
            return None
        
        # Limit batch size to prevent overwhelming the model
        batch = frames_data[:batch_size]
        
        try:
            # Create prompt with frame names to reduce hallucination
            frame_names = [f"frame_{frame_data['frame_number']:04d}" for frame_data in batch]
            prompt = (
                "You are a social media expert. Here are some video frames:\n"
                + "\n".join(frame_names)
                + "\n\nFrom ONLY these frames, which ONE would make the best YouTube thumbnail and why? "
                  "Reply with the exact frame name from the list above and your reasoning."
            )
            
            # Get PIL images (already in RGB format)
            images = []
            for frame_data in batch:
                try:
                    image = frame_data["frame"]
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    images.append(image)
                except Exception as e:
                    print(f"⚠️  Error processing frame {frame_data['frame_number']}: {e}")
                    continue
            
            if not images:
                return None
            
            # Show loading indicator while waiting for Gemini response
            print("⏳ Analyzing frames with Gemini AI...")
            start_time = time.time()
            
            # Generate response from Gemini
            response = self.model.generate_content([prompt] + images)
            
            # Show completion time
            elapsed_time = time.time() - start_time
            print(f"✅ Gemini analysis complete ({elapsed_time:.1f}s)")
            
            if not response.text:
                return None
            
            # Try to extract the chosen frame
            for frame_data in batch:
                frame_name = f"frame_{frame_data['frame_number']:04d}"
                if frame_name in response.text:
                    return frame_data
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error in Gemini batch scoring: {e}")
            return None


def score_with_gemini(frames_data: List[Dict], batch_size: int = 16) -> Tuple[Dict, str]:
    """
    Score frames using Gemini and return the best thumbnail candidate.
    
    This function uses a tournament-style approach:
    1. Divide frames into batches
    2. Get the best frame from each batch
    3. If multiple batch winners, run a final round to pick the overall winner
    
    Args:
        frames_data: List of frame dictionaries with PIL images and metadata
        batch_size: Maximum number of frames per batch (default: 16)
        
    Returns:
        Tuple of (best_frame_data, explanation)
        
    Raises:
        ValueError: If no frames are provided or no valid frames found
        RuntimeError: If Gemini API is not properly configured
    """
    if not frames_data:
        raise ValueError("No frames provided")
    
    # Initialize Gemini scorer
    scorer = GeminiScorer()
    
    print(f"🔍 Scoring {len(frames_data)} frames with Gemini...")
    print(f"📦 Using batch size: {batch_size}")
    print("-" * 50)
    
    # Process frames in batches
    batch_winners = []
    
    for i in range(0, len(frames_data), batch_size):
        batch = frames_data[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"\n--- Batch {batch_num} ({len(batch)} frames) ---")
        for frame_data in batch:
            frame_name = f"frame_{frame_data['frame_number']:04d}"
            print(f"  📸 {frame_name}")
        
        # Score the batch
        winner = scorer.score_batch_data(batch, batch_size)
        
        if winner:
            batch_winners.append(winner)
            winner_name = f"frame_{winner['frame_number']:04d}"
            print(f"✅ Batch {batch_num} winner: {winner_name}")
        else:
            print(f"⚠️  No clear winner in batch {batch_num}")
    
    if not batch_winners:
        raise ValueError("No valid frames could be processed by Gemini")
    
    # If only one winner, return it
    if len(batch_winners) == 1:
        winner_data = batch_winners[0]
        winner_name = f"frame_{winner_data['frame_number']:04d}"
        print(f"\n🏆 Overall winner: {winner_name}")
        return winner_data, f"Single batch winner: {winner_name}"
    
    # Final round: compare batch winners
    print(f"\n=== Final Round ({len(batch_winners)} candidates) ===")
    for winner in batch_winners:
        winner_name = f"frame_{winner['frame_number']:04d}"
        print(f"  🎯 {winner_name}")
    
    final_winner = scorer.score_batch_data(batch_winners, len(batch_winners))
    
    if final_winner:
        winner_name = f"frame_{final_winner['frame_number']:04d}"
        print(f"\n🏆 Overall winner: {winner_name}")
        return final_winner, f"Final round winner: {winner_name}"
    else:
        # Fallback to first batch winner if final round fails
        fallback_winner = batch_winners[0]
        winner_name = f"frame_{fallback_winner['frame_number']:04d}"
        print(f"\n🏆 Overall winner (fallback): {winner_name}")
        return fallback_winner, f"Fallback winner: {winner_name}"


def get_gemini_explanation(frame_path: str) -> str:
    """
    Get a detailed explanation from Gemini about why a frame makes a good thumbnail.
    
    Args:
        frame_path: Path to the frame image
        
    Returns:
        Detailed explanation from Gemini
    """
    try:
        scorer = GeminiScorer()
        
        image = Image.open(frame_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        prompt = (
            "You are a social media expert. Analyze this video frame and explain "
            "why it would or wouldn't make a good YouTube thumbnail. Consider "
            "factors like visual appeal, clarity, composition, and clickability."
        )
        
        response = scorer.model.generate_content([prompt, image])
        return response.text if response.text else "No explanation available"
        
    except Exception as e:
        return f"Error getting explanation: {e}"
