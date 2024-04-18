import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import colorsys
from matplotlib.widgets import Slider

# Load the image
image = Image.open("homework4/gun1.bmp")

# Convert image to RGB (if not already in RGB)
image_rgb = image.convert("RGB")

# Convert image to numpy array
image_array = np.array(image_rgb)

# Convert RGB to HSV color space
image_hsv = np.array([colorsys.rgb_to_hsv(pixel[0] / 255, pixel[1] / 255, pixel[2] / 255)
                      for pixel in image_array.reshape(-1, 3)])

# Reshape the HSV array back to the original image shape
image_hsv = image_hsv.reshape(image_array.shape[:-1] + (3,))

# Create initial skin tone threshold ranges in HSV color space
init_lower_hue = 0.0
init_upper_hue = 0.1
init_lower_saturation = 0.15
init_upper_saturation = 1.0
init_lower_value = 0.3
init_upper_value = 1.0

# Create figure and subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot original image
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

# Plot skin tone regions (initially empty)
skin_image = ax[1].imshow(np.zeros_like(image_array))
ax[1].set_title("Skin Tone Regions")
ax[1].axis("off")

# Create sliders
ax_lower_hue = plt.axes([0.25, 0.05, 0.65, 0.03],
                        facecolor='lightgoldenrodyellow')
ax_upper_hue = plt.axes([0.25, 0.1, 0.65, 0.03],
                        facecolor='lightgoldenrodyellow')
ax_lower_saturation = plt.axes(
    [0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_upper_saturation = plt.axes(
    [0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_lower_value = plt.axes([0.25, 0.25, 0.65, 0.03],
                          facecolor='lightgoldenrodyellow')
ax_upper_value = plt.axes([0.25, 0.3, 0.65, 0.03],
                          facecolor='lightgoldenrodyellow')

slider_lower_hue = Slider(ax_lower_hue, 'Lower Hue',
                          0.0, 1.0, valinit=init_lower_hue)
slider_upper_hue = Slider(ax_upper_hue, 'Upper Hue',
                          0.0, 1.0, valinit=init_upper_hue)
slider_lower_saturation = Slider(
    ax_lower_saturation, 'Lower Saturation', 0.0, 1.0, valinit=init_lower_saturation)
slider_upper_saturation = Slider(
    ax_upper_saturation, 'Upper Saturation', 0.0, 1.0, valinit=init_upper_saturation)
slider_lower_value = Slider(
    ax_lower_value, 'Lower Value', 0.0, 1.0, valinit=init_lower_value)
slider_upper_value = Slider(
    ax_upper_value, 'Upper Value', 0.0, 1.0, valinit=init_upper_value)

# Update skin tone regions when sliders change


def update(val):
    lower_hue = slider_lower_hue.val
    upper_hue = slider_upper_hue.val
    lower_saturation = slider_lower_saturation.val
    upper_saturation = slider_upper_saturation.val
    lower_value = slider_lower_value.val
    upper_value = slider_upper_value.val

    # Create a binary mask to identify pixels within the updated skin tone range
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

    # Update the displayed skin tone regions
    skin_image.set_data(skin_regions)
    fig.canvas.draw_idle()


# Link sliders to update function
slider_lower_hue.on_changed(update)
slider_upper_hue.on_changed(update)
slider_lower_saturation.on_changed(update)
slider_upper_saturation.on_changed(update)
slider_lower_value.on_changed(update)
slider_upper_value.on_changed(update)

plt.show()
