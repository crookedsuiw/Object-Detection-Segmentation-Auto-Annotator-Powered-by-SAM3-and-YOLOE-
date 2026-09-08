# Object Detection & Segmentation Auto-Annotator (Powered by SAM3 and YOLOE)

Object Detection & Segmentation Auto-Annotator is a powerful, AI-assisted image annotation tool designed for rapid dataset creation. It features a robust graphical user interface built with Tkinter, integrating Vision-Language Models (Florence), (YOLOE) and Segmentation Models (SAM3) to automate the heavy lifting of drawing bounding boxes and polygons.

# About the VL models used:
This program uses meta's SAM3 (Segment Anything Model), Florence, and YOLOE. 
SAM3 works exceptionally well in our testing and it is much more intuitive (as is YOLOE) to use since it as open-vocabulary model that takes in class names as text prompts. It practically requires a dedicated GPU for inference. We achieved a throughput of ~12,000images/hour for 10 classes (fp16 quantization) on an RTX5060 8GB.

Download SAM3 here:
<a href = "https://huggingface.co/facebook/sam3">SAM3</a>

YOLOE is a lighter, more efficient model that runs well on CPU but more prone to errors in edge cases. 

On the the yoloe-26x-seg variant, we achieved a throughput of ~136,000images/hour for 10 classes (fp32) on an RTX5060 8GB.

Download YOLOE here (make sure to choose "Text / visual prompt" variants, not "Prompt-free"):
<a href = "https://docs.ultralytics.com/models/yoloe"></a>

## Key Features

- **AI Auto-Annotation**: Leverage **SAM3 (Segment Anything 3)** to automatically generate pixel-perfect segmentation masks or bounding boxes for your dataset.
- **Customizable Targets**: Specifically choose which classes the AI should target, avoiding unnecessary clutter in your labels.
- **Dual Export Formats**: Seamlessly save your annotations in standard YOLO Bounding Box format (`class cx cy w h`) or YOLO Polygon format.
- **Advanced Editing Tools**:
  - **Dynamic Resizing**: Press `H` to cycle through resize modes (Width, Height, Both) and use your mouse scroll wheel to precisely scale bounding boxes.
  - **Smart Rotation**: Easily rotate bounding boxes and polygons using the scroll wheel.
  - **Distinct Visuals**: Automatically assigns highly distinguishable, unique colors to different class IDs (avoiding reds to preserve selection clarity).
- **Per-Frame Undo History**: Make a mistake? Press `Z` to undo your last action. The tool securely caches the last 30 actions for up to 100 independent frames to prevent memory leaks.
- **Video Splitter Integration**: Built-in utilities for handling frames natively.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `N` | Next Image |
| `P` | Previous Image |
| `Z` | Undo (Per-frame isolated history) |
| `D` | Delete selected box |
| `A` | Trigger Auto-Annotate |
| `T` | Toggle Tracking Mode |
| `G` | Toggle Segmentation (Polygon) Mode |
| `R` | Toggle Rotate Mode |
| `H` | Cycle Resize Mode (Width, Height, Both, None) |
| `S` | Cycle Class ID |
| `0-9` | Quickly jump to a specific Class ID |
| `Arrows` | Nudge selected boxes |
| `Ctrl + S` | Save Labels |
| `Mouse Scroll`| Resize or Rotate (depending on active mode) |

## Quick Start
1. Run the script: `python AUTO_ANNOTATOR_SAM.py` (or launch the compiled `.exe`).
2. Input your comma-separated class names in the Startup GUI.
3. Select your input image directory, label output directory, and image output directory.
4. (Optional) Check **"Use SAM3"** to open the AI configuration panel and select your `.pt` model weights for bulk auto-annotation.
5. Click **Launch Annotator** to begin!

## Standalone Executable
If you are running the compiled standalone executable (`.exe`), please note that the initial "cold start" may take 1-2 minutes. This is normal, as the application needs to temporarily extract gigabytes of heavy PyTorch and Ultralytics binaries before the interface can appear.
