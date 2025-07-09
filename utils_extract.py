"""
Video Frame Extraction Utilities

This module contains functions for extracting frames from video files
at specified intervals for thumbnail analysis.
"""

import cv2
import os
import base64
import requests
from typing import List, Optional, Tuple, Dict
import numpy as np
from PIL import Image
import io


def extract_frames_in_memory(video_path: str, interval: float = 1.0) -> List[Dict]:
    """
    Extract frames from a video file at specified time intervals in memory.
    
    Args:
        video_path: Path to the input video file
        interval: Time interval in seconds between extracted frames (default: 1.0)
        
    Returns:
        List of dictionaries containing frame data and metadata:
        [{"frame": PIL_Image, "timestamp": float, "frame_number": int}, ...]
        
    Raises:
        ValueError: If video file cannot be opened or is invalid
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            raise ValueError(f"Invalid video FPS: {video_fps}")
        
        # Calculate frame interval (number of frames to skip)
        frame_interval = int(video_fps * interval)
        if frame_interval <= 0:
            frame_interval = 1
        
        frame_count = 0
        saved_count = 0
        frames_data = []
        
        # Get total frame count for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Extracting frames from '{video_path}' every {interval}s...")
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Show progress indicator every 10% or every 100 frames
            if total_frames > 0 and (frame_count % max(1, total_frames // 10) == 0 or frame_count % 100 == 0):
                progress = (frame_count / total_frames) * 100
                print(f"⏳ Extracting frames... {progress:.0f}% ({frame_count}/{total_frames})")
            
            # Process frame at the specified interval
            if frame_count % frame_interval == 0:
                # Calculate timestamp
                timestamp = frame_count / video_fps
                
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                frames_data.append({
                    "frame": pil_image,
                    "timestamp": timestamp,
                    "frame_number": saved_count
                })
                saved_count += 1
            
            frame_count += 1
        
        print(f"✅ Extracted {saved_count} frames in memory from {frame_count} total frames")
        return frames_data
        
    finally:
        # Always release the video capture object
        cap.release()


def upload_frame_to_imgur(pil_image: Image.Image, verbose: bool = False) -> Optional[str]:
    """
    Upload a PIL Image to Imgur and return the URL.
    
    Args:
        pil_image: PIL Image to upload
        verbose: Whether to show detailed error information
        
    Returns:
        URL of uploaded image, or None if upload failed
    """
    try:
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        # Encode to base64
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Upload to Imgur (anonymous upload)
        headers = {
            'Authorization': 'Client-ID 546c25a59c58ad7',
            'User-Agent': 'Thumbnail-Selector/1.0'
        }
        data = {'image': img_b64, 'type': 'base64', 'title': 'Video Thumbnail'}
        
        response = requests.post('https://api.imgur.com/3/upload', headers=headers, data=data, timeout=30)
        
        if verbose:
            print(f"📡 Imgur response status: {response.status_code}")
            print(f"📡 Response text: {response.text[:200]}...")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                # Get the image ID from the response
                image_id = result['data']['id']
                # Return the proper Imgur page URL (not direct image link)
                url = f"https://imgur.com/{image_id}"
                
                if verbose:
                    print(f"✅ Image uploaded successfully: {url}")
                    direct_link = result['data']['link']
                    print(f"📸 Direct image link: {direct_link}")
                
                return url
        else:
            if verbose:
                print(f"⚠️  Upload failed with status {response.status_code}: {response.text}")
        
        return None
        
    except requests.exceptions.Timeout:
        print("⚠️  Upload timeout - please try again")
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Network error during upload: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Failed to upload image: {e}")
        return None


def format_timestamp(seconds: float) -> str:
    """
    Format timestamp in seconds to MM:SS or HH:MM:SS format.
    
    Args:
        seconds: Timestamp in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


# Legacy function for backward compatibility
def extract_frames(video_path: str, interval: float = 1.0, output_folder: str = "frames") -> List[str]:
    """
    Legacy function - extracts frames to memory then returns empty list.
    Use extract_frames_in_memory() instead.
    """
    frames_data = extract_frames_in_memory(video_path, interval)
    return []  # Return empty list to maintain compatibility
