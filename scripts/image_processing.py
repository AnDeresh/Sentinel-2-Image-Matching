import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

# Detect keypoints and descriptors using ORB
def detect_keypoints_and_descriptors(image_path):
    """
    Detects keypoints and computes descriptors for an input image using the ORB (Oriented FAST and Rotated BRIEF) algorithm.

    Parameters:
    ----------
    image_path : str
        Path to the input image file.

    Returns:
    -------
    image : numpy.ndarray
        The original image read in color (BGR format).
    keypoints : list of cv2.KeyPoint
        A list of detected keypoints in the image.
    descriptors : numpy.ndarray
        A numpy array of shape (n_keypoints, descriptor_length) containing the descriptors for the detected keypoints.
        If no keypoints are detected, this will be `None`.

    Example:
    --------
    >>> image, keypoints, descriptors = detect_keypoints_and_descriptors("example.jpg")
    >>> print(f"Number of keypoints detected: {len(keypoints)}")
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read the image from path: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return image, keypoints, descriptors

# Custom visualization of matches with adjustable thickness
def draw_matches_custom(img1, kp1, img2, kp2, matches, thickness=2, color=(0, 255, 0)):
    """
    Draws keypoint matches between two images on a combined canvas with adjustable line thickness and color.

    Parameters:
    ----------
    img1 : numpy.ndarray
        The first input image (typically in BGR format).
    kp1 : list of cv2.KeyPoint
        Keypoints detected in the first image.
    img2 : numpy.ndarray
        The second input image (typically in BGR format).
    kp2 : list of cv2.KeyPoint
        Keypoints detected in the second image.
    matches : list of cv2.DMatch
        Matches between keypoints in the two images.
    thickness : int, optional
        Thickness of the lines and circles used to visualize the matches. Default is 2.
    color : tuple of int, optional
        Color of the lines and circles in BGR format. Default is (0, 255, 0) (green).

    Returns:
    -------
    canvas : numpy.ndarray
        A combined image showing both input images side-by-side with lines connecting the matched keypoints.

    Example:
    --------
    >>> canvas = draw_matches_custom(img1, kp1, img2, kp2, matches, thickness=3, color=(255, 0, 0))
    >>> plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    >>> plt.axis("off")
    >>> plt.show()
    """
    # Create a blank canvas combining both images side-by-side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    # Draw the matches
    for match in matches:
        # Coordinates of matched points
        pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
        pt2 = (int(kp2[match.trainIdx].pt[0] + w1), int(kp2[match.trainIdx].pt[1]))

        # Draw connecting line and circles at the points
        cv2.line(canvas, pt1, pt2, color, thickness)
        cv2.circle(canvas, pt1, thickness * 2, color, -1)
        cv2.circle(canvas, pt2, thickness * 2, color, -1)

    return canvas

# Visualize matches for random pairs of images
def visualize_random_pairs(image_paths, num_pairs=3):
    """
    Visualizes matches for random pairs of images using the `match_and_visualize` function.

    Parameters:
    ----------
    image_paths : list of str
        A list of file paths to the images to be matched and visualized.
    num_pairs : int, optional
        The number of random pairs to visualize. Default is 3.

    Returns:
    -------
    None
        Displays the visualization of keypoint matches for the selected random pairs of images.

    Description:
    ------------
    1. Randomly selects `num_pairs` adjacent image pairs from the provided list of image paths.
    2. Calls the `match_and_visualize` function to process and visualize each pair.

    Example:
    --------
    >>> image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    >>> visualize_random_pairs(image_paths, num_pairs=2)
    """
    if len(image_paths) < 2:
        raise ValueError("The list of image paths must contain at least two images to form pairs.")

    if num_pairs > len(image_paths) - 1:
        raise ValueError(f"Cannot select {num_pairs} pairs from a list of {len(image_paths)} images. Reduce `num_pairs`.")

    # Select random adjacent pairs
    pairs = random.sample(list(zip(image_paths[:-1], image_paths[1:])), k=num_pairs)
    
    for img1_path, img2_path in pairs:
        print(f"Visualizing matches between: {os.path.basename(img1_path)} and {os.path.basename(img2_path)}")
        match_and_visualize(img1_path, img2_path)

# Match and visualize keypoints between two images
def match_and_visualize(img1_path, img2_path):
    """
    Matches and visualizes keypoints between two images using the ORB detector and BFMatcher.

    Parameters:
    ----------
    img1_path : str
        Path to the first input image.
    img2_path : str
        Path to the second input image.

    Returns:
    -------
    None
        The function does not return anything but displays a visualization of the keypoint matches.

    Description:
    ------------
    1. Detects keypoints and computes descriptors for two input images using ORB.
    2. Matches the descriptors using the BFMatcher with Hamming distance and cross-check enabled.
    3. Sorts matches by distance (better matches first).
    4. Visualizes the top 50 matches on a side-by-side canvas with customizable line thickness and color.
    5. Displays the resulting image with matplotlib.

    Example:
    --------
    >>> match_and_visualize("image1.jpg", "image2.jpg")

    Dependencies:
    --------------
    Requires the `detect_keypoints_and_descriptors` and `draw_matches_custom` functions to be defined in the same script.
    """
    # Detect keypoints and descriptors
    img1, kp1, des1 = detect_keypoints_and_descriptors(img1_path)
    img2, kp2, des2 = detect_keypoints_and_descriptors(img2_path)

    # Check if descriptors are valid
    if des1 is None or des2 is None:
        raise ValueError("Descriptors could not be computed for one or both images. Please check the input images.")

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches with thicker lines
    match_img = draw_matches_custom(img1, kp1, img2, kp2, matches[:50], thickness=5, color=(0, 255, 0))

    # Convert BGR to RGB for display
    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

    # Display the result
    plt.figure(figsize=(15, 10))
    plt.imshow(match_img)
    plt.axis("off")
    plt.title(f"Keypoint Matches: {os.path.basename(img1_path)} and {os.path.basename(img2_path)}")
    plt.show()

def match_and_visualize_inference(img1_path, img2_path):
    """
    Matches and visualizes keypoints between two user-specified images using ORB and BFMatcher.

    Parameters:
    ----------
    img1_path : str
        Path to the first input image.
    img2_path : str
        Path to the second input image.

    Returns:
    -------
    None
        Displays the visualization of keypoint matches for the selected pair of images.
    """
    # Detect keypoints and descriptors
    img1, kp1, des1 = detect_keypoints_and_descriptors(img1_path)
    img2, kp2, des2 = detect_keypoints_and_descriptors(img2_path)

    # Check if descriptors are valid
    if des1 is None or des2 is None:
        raise ValueError("Descriptors could not be computed for one or both images. Please check the input images.")

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches with thicker lines
    match_img = draw_matches_custom(img1, kp1, img2, kp2, matches[:50], thickness=5, color=(0, 255, 0))

    # Convert BGR to RGB for display
    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

    # Display the result
    plt.figure(figsize=(15, 10))
    plt.imshow(match_img)
    plt.axis("off")
    plt.title(f"Keypoint Matches: {os.path.basename(img1_path)} and {os.path.basename(img2_path)}")
    plt.show()