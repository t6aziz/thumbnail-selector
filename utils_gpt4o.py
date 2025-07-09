"""
GPT-4o-based Frame Scoring Utilities

This module contains functions for scoring video frames using OpenAI's GPT-4o model
to determine which frame would make the best thumbnail through visual analysis.
"""

import os
import re
import base64
import sys
import time
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
import openai
from PIL import Image


class GPT4oScorer:
    """
    A class for scoring images using OpenAI's GPT-4o model.
    
    This class handles individual frame evaluation where GPT-4o provides
    numerical scores and explanations for each frame.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the GPT-4o scorer.
        
        Args:
            model_name: Name of the OpenAI model to use (default: "gpt-4o")
            
        Raises:
            RuntimeError: If API key is not found or client initialization fails
        """
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment variables")
        
        try:
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = model_name
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert an image file to base64 encoding.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded string of the image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be processed
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to encode image '{image_path}': {e}")
    
    def _pil_image_to_base64(self, pil_image: Image.Image) -> str:
        """
        Convert a PIL Image to base64 encoding.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Base64-encoded string of the image
            
        Raises:
            ValueError: If image cannot be processed
        """
        try:
            import io
            
            # Convert PIL image to bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            
            # Encode to base64
            return base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to encode PIL image: {e}")
    
    def score_image(self, image_path: str, return_explanation: bool = True) -> Tuple[float, str]:
        """
        Score a single image using GPT-4o and return score with explanation.
        
        Args:
            image_path: Path to the image file
            return_explanation: Whether to return detailed explanation
            
        Returns:
            Tuple of (score, explanation) where score is 0-10
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be processed or score cannot be extracted
        """
        base64_image = self._image_to_base64(image_path)
        
        try:
            # Create the prompt for scoring
            prompt = (
                "You're a social media expert. Rate this image from 1 to 10 "
                "for how clickable and viral it would be as a YouTube thumbnail. "
                "Consider factors like visual appeal, clarity, composition, and engagement potential. "
                "Start your response with 'SCORE: X' where X is the numerical score, "
                "then provide a brief explanation of your reasoning."
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.4,  # Lower temperature for more consistent responses
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            if not result:
                raise ValueError("Empty response from GPT-4o")
            
            # Extract numerical score from response
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', result, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                # Normalize score to 0-1 range for consistency with other methods
                score = min(max(score / 10.0, 0.0), 1.0)
            else:
                # Fallback: look for any number at the start
                number_match = re.search(r'(\d+(?:\.\d+)?)', result)
                if number_match:
                    score = float(number_match.group(1))
                    score = min(max(score / 10.0, 0.0), 1.0)
                else:
                    raise ValueError(f"Could not extract score from response: {result}")
            
            explanation = result if return_explanation else ""
            return score, explanation
            
        except Exception as e:
            raise ValueError(f"Failed to score image '{image_path}': {e}")
    
    def score_image_pil(self, pil_image: Image.Image, return_explanation: bool = True) -> Tuple[float, str]:
        """
        Score a PIL Image using GPT-4o and return score with explanation.
        
        Args:
            pil_image: PIL Image object
            return_explanation: Whether to return detailed explanation
            
        Returns:
            Tuple of (score, explanation) where score is 0-1
            
        Raises:
            ValueError: If image cannot be processed or score cannot be extracted
        """
        base64_image = self._pil_image_to_base64(pil_image)
        
        try:
            # Create the prompt for scoring
            prompt = (
                "You're a social media expert. Rate this image from 1 to 10 "
                "for how clickable and viral it would be as a YouTube thumbnail. "
                "Consider factors like visual appeal, clarity, composition, and engagement potential. "
                "Start your response with 'SCORE: X' where X is the numerical score, "
                "then provide a brief explanation of your reasoning."
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.4,  # Lower temperature for more consistent responses
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            if not result:
                raise ValueError("Empty response from GPT-4o")
            
            # Extract numerical score from response
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', result, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                # Normalize score to 0-1 range for consistency with other methods
                score = min(max(score / 10.0, 0.0), 1.0)
            else:
                # Fallback: look for any number at the start
                number_match = re.search(r'(\d+(?:\.\d+)?)', result)
                if number_match:
                    score = float(number_match.group(1))
                    score = min(max(score / 10.0, 0.0), 1.0)
                else:
                    raise ValueError(f"Could not extract score from response: {result}")
            
            explanation = result if return_explanation else ""
            return score, explanation
            
        except Exception as e:
            raise ValueError(f"Failed to score PIL image: {e}")


def score_with_gpt4o(frames_data: List[Dict]) -> Tuple[Dict, float]:
    """
    Score frames using GPT-4o and return the best thumbnail candidate.
    
    Args:
        frames_data: List of frame dictionaries with PIL images and metadata
        
    Returns:
        Tuple of (best_frame_data, best_score)
        
    Raises:
        ValueError: If no frames are provided or no valid frames found
        RuntimeError: If GPT-4o API is not properly configured
    """
    if not frames_data:
        raise ValueError("No frames provided")
    
    # Initialize GPT-4o scorer
    scorer = GPT4oScorer()
    
    print(f"🔍 Scoring {len(frames_data)} frames with GPT-4o...")
    print("-" * 50)
    
    scores = []
    valid_frames = 0
    total_frames = len(frames_data)
    
    for i, frame_data in enumerate(frames_data):
        try:
            # Show progress for each frame (GPT-4o is slower)
            frame_name = f"frame_{frame_data['frame_number']:04d}"
            print(f"⏳ Processing {frame_name} with GPT-4o... ({i + 1}/{total_frames})")
            
            score, explanation = scorer.score_image_pil(frame_data["frame"])
            scores.append((frame_data, score, explanation))
            print(f"✅ {frame_name}: {score:.4f}")
            valid_frames += 1
            
        except Exception as e:
            print(f"⚠️  Error processing frame {frame_data['frame_number']}: {e}")
            continue
    
    print(f"✅ Completed processing {valid_frames}/{total_frames} frames")
    
    if not scores:
        raise ValueError("No valid frames could be processed by GPT-4o")
    
    # Find the best scoring frame
    best_frame_data, best_score, best_explanation = max(scores, key=lambda x: x[1])
    best_frame_name = f"frame_{best_frame_data['frame_number']:04d}"
    
    print("-" * 50)
    print(f"🏆 Best frame: {best_frame_name}")
    print(f"📊 Score: {best_score:.4f}")
    print(f"💭 Explanation: {best_explanation}")
    print(f"✅ Processed {valid_frames}/{len(frames_data)} frames successfully")
    
    return best_frame_data, best_score


def batch_score_with_gpt4o(frame_paths: List[str]) -> List[Tuple[str, float, str]]:
    """
    Score all frames and return a sorted list of (frame_path, score, explanation) tuples.
    
    Args:
        frame_paths: List of paths to frame image files
        
    Returns:
        List of (frame_path, score, explanation) tuples sorted by score (highest first)
    """
    if not frame_paths:
        return []
    
    scorer = GPT4oScorer()
    results = []
    
    for frame_path in frame_paths:
        try:
            score, explanation = scorer.score_image(frame_path)
            results.append((frame_path, score, explanation))
        except Exception as e:
            print(f"⚠️  Error processing {frame_path}: {e}")
            continue
    
    # Sort by score (highest first)
    return sorted(results, key=lambda x: x[1], reverse=True)


def get_detailed_analysis(frame_path: str) -> str:
    """
    Get a detailed analysis from GPT-4o about why a frame makes a good thumbnail.
    
    Args:
        frame_path: Path to the frame image
        
    Returns:
        Detailed analysis from GPT-4o
    """
    try:
        scorer = GPT4oScorer()
        base64_image = scorer._image_to_base64(frame_path)
        
        prompt = (
            "You are a social media expert. Provide a detailed analysis of this video frame "
            "as a potential YouTube thumbnail. Consider:\n"
            "1. Visual composition and clarity\n"
            "2. Color scheme and contrast\n"
            "3. Subject matter and focal points\n"
            "4. Emotional impact and engagement potential\n"
            "5. Clickability and viral potential\n"
            "Provide specific recommendations for improvement if applicable."
        )
        
        response = scorer.client.chat.completions.create(
            model=scorer.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.4,
            max_tokens=1000
        )
        
        return response.choices[0].message.content or "No analysis available"
        
    except Exception as e:
        return f"Error getting analysis: {e}"
