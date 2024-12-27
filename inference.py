import os

from paths import *
from scripts.image_processing import *

# Example usage
if __name__ == "__main__":
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        raise FileNotFoundError("One or both image paths are invalid. Please provide valid paths.")

    match_and_visualize_inference(img1_path, img2_path)