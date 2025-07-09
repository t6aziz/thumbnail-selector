# Thumbnail Selector - AI-Powered Video Frame Analysis

A Python tool that extracts frames from video files and uses AI models to determine which frame would make the best thumbnail for social media platforms. The tool analyzes frames in memory, uploads the best frame to Imgur, and provides a shareable link with timestamp information.

## Features

- **Multiple AI Models**: Choose from CLIP, Gemini, or GPT-4o for frame analysis
- **In-Memory Processing**: No temporary files created - frames are processed entirely in memory
- **Instant Upload**: Best frame is automatically uploaded to Imgur for easy sharing
- **Timestamp Information**: Shows exactly when the best frame occurs in the video
- **Flexible Frame Extraction**: Configurable time intervals between extracted frames
- **Clean Architecture**: Modular design with separate utility modules for each AI model
- **Comprehensive Output**: Detailed analysis results with shareable links and timestamps
- **Easy to Use**: Simple command-line interface with helpful options

## Supported Models

| Model | Description | Strengths |
|-------|-------------|-----------|
| **CLIP** | OpenAI's CLIP model | Fast, good for semantic similarity matching |
| **Gemini** | Google's Gemini model | **Compares multiple frames at once** - analyzes batches of frames together for relative comparison |
| **GPT-4o** | OpenAI's GPT-4o model | **Analyzes each frame individually** - provides detailed numerical scores (1-10) and explanations |

### Model Analysis Approaches

**🔍 Gemini's Batch Comparison Approach:**
- Analyzes 16 frames at once in each batch
- Makes relative comparisons: "Which of these frames is the best thumbnail?"
- Uses tournament-style elimination for large frame sets
- Better at understanding context and relative quality

**📊 GPT-4o's Individual Scoring Approach:**
- Scores each frame individually on a 1-10 scale
- Provides detailed explanations for each score
- More consistent scoring across different runs
- Better for understanding why specific frames scored well

**⚡ CLIP's Similarity Matching:**
- Uses semantic similarity to match frames against "clickbait thumbnail" concept
- Fastest processing speed
- Good baseline for thumbnail selection

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your API keys in a `.env` file:

```bash
# For Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# For GPT-4o
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Video File Placement

**Simply place your video file in the same directory as the script:**

```bash
# Current directory structure:
# Thumbnail-Selector/
# ├── thumbnail_selector.py
# ├── your_video.mp4      # ← Your video file here
# ├── utils_extract.py
# └── ...
```

### Basic Usage

```bash
python thumbnail_selector.py your_video.mp4 --model clip
```

### Advanced Usage

```bash
python thumbnail_selector.py your_video.mp4 \
  --model gemini \
  --verbose
```

### Get Shareable Link and Timestamp

```bash
# Analyze video and get shareable link with timestamp
python thumbnail_selector.py your_video.mp4 --model clip
# Output will include:
# 🕐 Timestamp: 01:23
# 🔗 Image URL: https://i.imgur.com/abc123.jpg
```

### Custom Frame Extraction Interval (Optional)

```bash
# Extract frames every 0.5 seconds (more frames, finer analysis)
python thumbnail_selector.py your_video.mp4 --model clip --interval 0.5

# Extract frames every 2.0 seconds (fewer frames, faster processing)
python thumbnail_selector.py your_video.mp4 --model gemini --interval 2.0
```

### Command Line Options

- `video_path`: Path to the video file to analyze (required)
- `--model`: AI model to use (`clip`, `gemini`, `gpt4o`) (required)
- `--interval`: Seconds between extracted frames (optional, default: 1.0)
- `--verbose`: Enable verbose output (shows detailed processing information)

## Code Structure

The codebase is organized into clean, modular components:

```
Thumbnail-Selector/
├── thumbnail_selector.py    # Main script with CLI interface
├── utils_extract.py         # Video frame extraction and upload utilities
├── utils_clip.py           # CLIP model scoring utilities
├── utils_gemini.py         # Gemini model scoring utilities
├── utils_gpt4o.py          # GPT-4o model scoring utilities
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Ignores cache files and other artifacts
└── .env                   # API keys (create this file)
```

### Main Components

- **`thumbnail_selector.py`**: Clean, lightweight main script that handles CLI parsing and orchestrates the analysis process
- **`utils_extract.py`**: In-memory frame extraction with Imgur upload functionality and timestamp formatting
- **`utils_clip.py`**: CLIP model implementation with batch processing capabilities  
- **`utils_gemini.py`**: Gemini model implementation with tournament-style frame selection
- **`utils_gpt4o.py`**: GPT-4o model implementation with detailed scoring and explanations

## Examples

### Quick Analysis with CLIP
```bash
python thumbnail_selector.py sample.mp4 --model clip
```

### Detailed Analysis with Gemini
```bash
python thumbnail_selector.py sample.mp4 --model gemini --verbose
```

### High-Quality Analysis with GPT-4o
```bash
python thumbnail_selector.py sample.mp4 --model gpt4o
```

### Custom Interval Example
```bash
# Fine-grained analysis (extract more frames)
python thumbnail_selector.py sample.mp4 --model clip --interval 0.5
```

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- PyTorch and CLIP (for CLIP model)
- Google Generative AI (for Gemini model)
- OpenAI API (for GPT-4o model)
- PIL/Pillow (for image processing)
- python-dotenv (for environment variables)
- requests (for image upload to Imgur)

## API Keys

### Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GEMINI_API_KEY`

### OpenAI API Key
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Add it to your `.env` file as `OPENAI_API_KEY`

## Contributing

The codebase is designed to be easily extensible. To add a new AI model:

1. Create a new utility file (e.g., `utils_newmodel.py`)
2. Implement a `score_with_newmodel()` function that takes frame paths and returns the best frame
3. Add the model choice to the argument parser in `thumbnail_selector.py`
4. Add the model case to the scoring logic in the main function

## License

This project is open source and available under the MIT License. 