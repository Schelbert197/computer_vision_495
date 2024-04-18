import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import colorsys

# Load the image
image = Image.open("homework4/gun1.bmp")

# Convert image to RGB (if not already in RGB)
image_rgb = image.convert("RGB")

# Convert image to numpy array
image_array = np.array(image_rgb)

# # Normalize pixel values to range [0, 1]
# image_array_norm = image_array / 255.0

# # Define skin tone threshold ranges in RGB color space
# lower_bound = np.array([0, 48, 80], dtype=np.uint8)
# upper_bound = np.array([255, 255, 255], dtype=np.uint8)

# # Create a binary mask to identify pixels within the skin tone range
# skin_mask = np.logical_and.reduce((image_array_norm[:, :, 0] > lower_bound[0],
#                                    image_array_norm[:, :, 1] > lower_bound[1],
#                                    image_array_norm[:, :, 2] > lower_bound[2],
#                                    image_array_norm[:, :, 0] < upper_bound[0],
#                                    image_array_norm[:, :, 1] < upper_bound[1],
#                                    image_array_norm[:, :, 2] < upper_bound[2]))

# # Apply the mask to the original image to extract skin tone regions
# skin_regions = np.zeros_like(image_array)
# skin_regions[skin_mask] = image_array[skin_mask]

# # Convert the resulting image back to PIL format for display
# skin_image = Image.fromarray((skin_regions * 255).astype(np.uint8))


# # Convert RGB to HSV color space
# image_hsv = np.array([colorsys.rgb_to_hsv(pixel[0] / 255, pixel[1] / 255, pixel[2] / 255)
#                       for pixel in image_array])

# Convert RGB to HSV color space
image_hsv = np.array([colorsys.rgb_to_hsv(pixel[0] / 255, pixel[1] / 255, pixel[2] / 255)
                      for pixel in image_array.reshape(-1, 3)])

# Reshape the HSV array back to the original image shape
image_hsv = image_hsv.reshape(image_array.shape[:-1] + (3,))

# Define skin tone threshold ranges in HSV color space
lower_hue = 0.0
upper_hue = 0.1
lower_saturation = 0.15
upper_saturation = 1.0
lower_value = 0.2
upper_value = 1.0

# Create a binary mask to identify pixels within the skin tone range
skin_mask = np.logical_and.reduce((
    image_hsv[:, :, 0] >= lower_hue,
    image_hsv[:, :, 0] <= upper_hue,
    image_hsv[:, :, 1] >= lower_saturation,
    image_hsv[:, :, 1] <= upper_saturation,
    image_hsv[:, :, 2] >= lower_value,
    image_hsv[:, :, 2] <= upper_value
))

# Apply the mask to the original image to extract skin tone regions
skin_regions = np.zeros_like(image_array)
skin_regions[skin_mask] = image_array[skin_mask]

# Convert the resulting image back to PIL format for display
skin_image = Image.fromarray(skin_regions)

# Display the original image and skin tone regions
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(skin_image)
plt.title("Skin Tone Regions")
plt.axis("off")

plt.show()
