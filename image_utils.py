"""
Image Utility Functions Module
----------------------------
This module provides utility functions for loading, processing, and analyzing images
in the Falconnet demo application. It includes functionality for:
- Loading sample character images from the assets directory
- Computing image differences and similarity metrics
- Generating visualization heatmaps for attack analysis
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_sample_characters():
    """
    Load sample character images from the assets directory.
    
    Returns:
    --------
    dict
        Dictionary mapping character names to PIL.Image objects
        Format: {"Character 1": image1, "Character 2": image2, ...}
    """
    base_path = "assets"
    char_images = {}
    for i in range(1, 6):
        path = os.path.join(base_path, f"char{i}.png")
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            char_images[f"Character {i}"] = img
    return char_images


def compute_mse(img1, img2):
    """
    Compute Mean Squared Error between two images.
    
    Parameters:
    -----------
    img1, img2 : PIL.Image
        The two images to compare. Must have the same dimensions.
    
    Returns:
    --------
    float
        Mean Squared Error between the two images, normalized to [0, 1] range
    """
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    return np.mean((arr1 - arr2) ** 2)


def compute_difference_heatmap(img1, img2, small=False):
    """
    Generate a heatmap visualization showing pixel-level differences between two images.
    
    Parameters:
    -----------
    img1, img2 : PIL.Image
        The two images to compare. Must have the same dimensions.
    small : bool, optional
        If True, generate a smaller figure (3x3 inches)
        If False, generate a larger figure (5x5 inches)
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the difference heatmap with color scale:
        - Black: No difference
        - Yellow: Moderate difference
        - White: High difference
        - Red: Extreme difference
    """
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    diff = np.abs(arr1 - arr2).mean(axis=2)
    fig, ax = plt.subplots()
    if small:
        fig.set_size_inches(3, 3)  # Reduced size for smaller display
    else:
        fig.set_size_inches(5, 5)  # Default size for larger display
    ax.imshow(diff, cmap="hot")
    ax.axis("off")
    
    # Add color legend explanation
    fig.text(0.5, 0.01, "Color Legend: Black (No Change), Yellow (Moderate Change), White (High Change), Red (Extreme Change)", ha="center", fontsize=8)
    return fig
