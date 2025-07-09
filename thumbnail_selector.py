#!/usr/bin/env python3
"""
Thumbnail Selector - Video Frame Analysis Tool

This script extracts frames from video files and uses AI models to determine
which frame would make the best thumbnail for social media platforms.

Supported Models:
- CLIP: Uses OpenAI's CLIP model for semantic similarity matching
- Gemini: Uses Google's Gemini model for visual analysis
- GPT-4o: Uses OpenAI's GPT-4o model for detailed scoring

Usage:
    python thumbnail_selector.py video.mp4 --model clip --interval 2.0
    python thumbnail_selector.py video.mp4 --model gemini --interval 1.0
    python thumbnail_selector.py video.mp4 --model gpt4o --interval 0.5
"""

import sys
import os
import argparse
from typing import List, Tuple, Optional

# Import utility modules
try:
    from utils_extract import extract_frames_in_memory, upload_frame_to_imgur, format_timestamp
    from utils_clip import score_with_clip
    from utils_gemini import score_with_gemini
    from utils_gpt4o import score_with_gpt4o
except ImportError as e:
    print(f"❌ Error importing utilities: {e}")
    print("Please ensure all utility modules are present in the same directory.")
    sys.exit(1)


def validate_video_file(video_path: str) -> bool:
    """
    Validate that the video file exists and has a supported extension.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return False
    
    # Check file extension
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    file_ext = os.path.splitext(video_path)[1].lower()
    
    if file_ext not in supported_extensions:
        print(f"❌ Error: Unsupported video format: {file_ext}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return False
    
    return True


def print_header():
    """Print a nice header for the application."""
    print("=" * 60)
    print("🎬 THUMBNAIL SELECTOR - AI-Powered Frame Analysis")
    print("=" * 60)



def main():
    """Main function that orchestrates the thumbnail selection process."""
    print_header()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract frames from video and select the best thumbnail using AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4 --model clip
  %(prog)s video.mp4 --model gemini --verbose
  %(prog)s video.mp4 --model gpt4o
  %(prog)s video.mp4 --model clip --interval 0.5

Models:
  clip    - OpenAI CLIP model (fast, good for semantic matching)
  gemini  - Google Gemini model (compares multiple frames at once)
  gpt4o   - OpenAI GPT-4o model (analyzes each frame individually with scores)
        """
    )
    
    parser.add_argument(
        "video_path", 
        help="Path to the video file to analyze"
    )
    
    parser.add_argument(
        "--model", 
        choices=["clip", "gemini", "gpt4o"], 
        required=True,
        help="AI model to use for frame scoring"
    )
    
    parser.add_argument(
        "--interval", 
        type=float, 
        default=1.0,
        help="Seconds between extracted frames (default: 1.0)"
    )
    

    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not validate_video_file(args.video_path):
        sys.exit(1)
    
    if args.interval <= 0:
        print("❌ Error: Interval must be greater than 0")
        sys.exit(1)
    
    # Import time for performance measurement
    import time
    start_time = time.time()
    
    try:
        # Step 1: Extract frames from video
        print(f"🎬 Processing video: {os.path.basename(args.video_path)}")
        print(f"📊 Using model: {args.model.upper()}")
        print(f"⏱️  Frame interval: {args.interval}s")
        print()
        
        frames_data = extract_frames_in_memory(
            video_path=args.video_path,
            interval=args.interval
        )
        
        if not frames_data:
            print("❌ Error: No frames were extracted from the video")
            sys.exit(1)
        
        print(f"✅ Extracted {len(frames_data)} frames")
        print()
        
        # Step 2: Score frames using the selected model
        best_frame_data = None
        best_score = None
        
        if args.model == "clip":
            best_frame_data, best_score = score_with_clip(frames_data)
            
        elif args.model == "gemini":
            best_frame_data, explanation = score_with_gemini(frames_data)
            best_score = explanation  # Gemini returns explanation instead of numerical score
            
        elif args.model == "gpt4o":
            best_frame_data, best_score = score_with_gpt4o(frames_data)
        
        # Step 3: Upload best frame and display results
        processing_time = time.time() - start_time
        
        if best_frame_data:
            # Upload the best frame to Imgur
            print("📤 Uploading best frame to Imgur...")
            import time
            start_time = time.time()
            image_url = upload_frame_to_imgur(best_frame_data["frame"], verbose=args.verbose)
            upload_time = time.time() - start_time
            if image_url:
                print(f"✅ Upload complete ({upload_time:.1f}s)")
            else:
                print(f"⚠️  Upload failed ({upload_time:.1f}s)")
            
            # Display results
            print("\n" + "=" * 60)
            print("🏆 THUMBNAIL ANALYSIS RESULTS")
            print("=" * 60)
            
            frame_name = f"frame_{best_frame_data['frame_number']:04d}"
            timestamp = best_frame_data["timestamp"]
            formatted_time = format_timestamp(timestamp)
            
            print(f"🎥 Video: {os.path.basename(args.video_path)}")
            print(f"🤖 Model: {args.model.upper()}")
            print(f"⏱️  Interval: {args.interval}s")
            print(f"📸 Total Frames: {len(frames_data)}")
            print(f"🏆 Best Frame: {frame_name}")
            print(f"🕐 Timestamp: {formatted_time}")
            
            if isinstance(best_score, (int, float)):
                print(f"📊 Score: {best_score:.4f}")
            else:
                print(f"📊 Analysis: {best_score}")
            
            if image_url:
                print(f"🔗 Image URL: {image_url}")
            else:
                print("⚠️  Image upload failed")
            
            if processing_time:
                print(f"⏱️  Processing Time: {processing_time:.2f}s")
            
            print("=" * 60)
            
        else:
            print("❌ Error: No best frame could be determined")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # No cleanup needed since we work in memory
        pass


if __name__ == "__main__":
    main()
