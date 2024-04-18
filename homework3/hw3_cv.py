# Srikanth Schelbert Homework3 (MP3)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Open the .bmp file
image = Image.open("homework3/moon.bmp")

# Display basic information about the image
print("Image format:", image.format)
print("Image mode:", image.mode)
print("Image size:", image.size)
image.show("Original Image")


def rgb_to_grayscale(image):
    """
    Convert RGB image to grayscale.
    """
    return image.convert('L')


def histogram_equalization(image):
    """
    Apply histogram equalization to grayscale image.
    """
    # Convert image to NumPy array
    img_array = np.array(image)

    # Calculate histogram
    histogram, bins = np.histogram(
        img_array.flatten(), bins=256, range=(0, 255))

    # Calculate cumulative distribution function
    cdf = histogram.cumsum()
    cdf_normalized = cdf * histogram.max() / cdf.max()

    # Apply histogram equalization
    # equalized_image = np.interp(img_array.flatten(), range(
    #     256), cdf_normalized).reshape(img_array.shape)

    transformation_func = (cdf_normalized - cdf_normalized.min()) / \
        (cdf_normalized.max() - cdf_normalized.min()) * 255

    # Apply the transformation to the image
    equalized_image = np.interp(img_array.flatten(), range(
        256), transformation_func).reshape(img_array.shape)

    # Calculate histogram again
    histogram2, bins2 = np.histogram(
        equalized_image.flatten(), bins=256, range=(0, 255))

    # Convert back to PIL image
    equalized_image = Image.fromarray(equalized_image.astype(np.uint8))

    return equalized_image, histogram, bins, histogram2, bins2


# Convert image to grayscale
grayscale_image = rgb_to_grayscale(image)

# Apply histogram equalization
equalized_image, histogram, bins, histogram2, bins2 = histogram_equalization(
    grayscale_image)

# Save the result
equalized_image.save("output_image.bmp")

# Visualize the histogram
plt.figure(figsize=(8, 6))
plt.bar(bins[:-1], histogram, width=1)
plt.title('Histogram of Grayscale Image (before)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Visualize the histogram
plt.figure(figsize=(8, 6))
plt.bar(bins2[:-1], histogram2, width=1)
plt.title('Histogram of Grayscale Image Without Light Correction (after)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Show the result
equalized_image.show()


def lighting_correction(image):
    """
    Perform lighting correction (histogram stretching) on the image.
    """
    # Convert image to NumPy array
    img_array = np.array(image)

    # Perform histogram stretching
    min_val = img_array.min()
    max_val = img_array.max()
    stretched_image = (img_array - min_val) * (255 / (max_val - min_val))

    histogram, bins = np.histogram(
        stretched_image.flatten(), bins=256, range=(0, 255))

    # Convert back to PIL image
    corrected_image = Image.fromarray(stretched_image.astype(np.uint8))

    return corrected_image, histogram, bins


light_image, histogram3, bins3 = lighting_correction(equalized_image)

# Visualize the histogram
plt.figure(figsize=(8, 6))
plt.bar(bins3[:-1], histogram3, width=1)
plt.title('Histogram of Grayscale Image With Light Correction (after)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

light_image.show()
