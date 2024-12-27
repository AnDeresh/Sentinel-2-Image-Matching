# Sentinel-2 Image Matching

## Overview
This project performs keypoint detection, descriptor matching, and visualization for Sentinel-2 satellite images. The project leverages ORB (Oriented FAST and Rotated BRIEF) for keypoint detection and uses BFMatcher for descriptor matching. It provides tools to process images, visualize matches, and analyze random or user-specified pairs of images.

---

## Features
- **Keypoint Detection**: Detects and extracts keypoints and descriptors from satellite images.
- **Keypoint Matching**: Matches descriptors between two images using BFMatcher with Hamming distance.
- **Visualization**: Displays images side-by-side with keypoint matches connected via lines.
- **Modular Design**: Reusable and extensible codebase organized into scripts and modules.
- **Support for Random and User-Specified Pairs**:
  - Random pairs: Visualize matches for a random selection of image pairs.
  - User-defined pairs: Visualize matches for two specific images.

---

## Project Structure
```
Sentinel-2-Image-Matching/
│
├── images/                                     # Folder with sample images
│
├── processed_data/                             # Contains preprocessed satellite images
├── scripts/
│   ├── __init__.py                             # Makes `scripts` a module
│   ├── image_processing.py                     # Contains keypoint detection, matching, and visualization functions
│
├── data_processing.ipynb                       # Notebook for preprocessing and organizing data
├── Demo Notebook.ipynb                         # Demonstration of image matching
├── inference.py                                # Script to visualize matches for user-specified image pairs
├── paths.py                                    # Stores folder paths (specify your paths)
├── requirements.txt                            # Python dependencies for the project
├── README.md                                   # Project documentation (this file)
├── .gitignore                                  # Specifies files/folders to ignore in Git
└── Report with potential improvements.pdf      # Analysis and potential improvements for the project
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Sentinel-2-Image-Matching
```

### 2. Create and Activate a Virtual Environment
- **Windows**:
  ```bash
  python -m venv satellite_env
  satellite_env\Scripts\activate
  ```

- **Linux/Mac**:
  ```bash
  python3 -m venv satellite_env
  source satellite_env/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset

The processed Sentinel-2 images for this project can be downloaded from the Hugging Face Datasets Hub:

- **Download processed data**: [Sentinel-2 Image Matching Dataset](https://huggingface.co/datasets/AnnaDee/Sentinel-2-Image-Matching/blob/main/processed_data.rar)

Please make sure to download the dataset and specify the correct paths in `paths.py` for the `processed_data_folder`, `source_folder`, and `destination_folder` as per your local setup.

---

## Usage

### User-Specified Image Pairs
To visualize matches for a pair of images:
1. Edit `paths.py` and replace the paths for `img1_path` and `img2_path` with the desired images.
2. Run the script:
   ```bash
   python inference.py
   ```

---

## Examples
**Visualization for Random Pairs**:
   Displays matches for three randomly selected image pairs:
   ![Example 1](images\img_1.png.png)
   ![Example 2](images\img_1.png.png)

---

## Requirements
- Python 3.7 or later
- Required Python packages:
  - `opencv-python`
  - `matplotlib`
  - `numpy`
  - `rasterio`

Install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---