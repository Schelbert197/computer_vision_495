import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

# Load the image
img_path = "homework4/pointer1.bmp"
image = Image.open(img_path)
dir_path = "homework4/test_images"

# # Convert to HSV color space
test_image_path = cv2.imread(img_path)
test_image = cv2.cvtColor(test_image_path, cv2.COLOR_BGR2HSV)
hue_test, saturation_test, value_test = cv2.split(test_image)

image_array = np.array(test_image)

# iterate through each training image and save it
skin_test_images = []
for filename in os.listdir(dir_path):
    # Check if the file is an image (you can add more formats if needed)
    if filename.endswith(".bmp") or filename.endswith(".jpg") or filename.endswith(".png"):
        # Construct full file path
        filepath = os.path.join(dir_path, filename)
        # Read the image
        image_t = cv2.imread(filepath)
        # Append the image to the list
        skin_test_images.append(image_t)

# Create an empty 2D histogram for hue and saturation values
# Number of bins for hue (0 to 179) and saturation (0 to 255)
# Get the shape of the original image
image_shape = image_array.shape[:2]
histogram_bins = (180, 256)
histogram = np.zeros((180, 256), dtype=float)

# Calculate hue and saturation values for each pixel in all skin test images
for image_hsv_t in skin_test_images:
    # test_image_path = cv2.imread("../images/pointer1.bmp")
    test_image = cv2.cvtColor(image_hsv_t, cv2.COLOR_BGR2HSV)
    hue_test, saturation_test, value_test = cv2.split(test_image)
    hue_values = hue_test.flatten()
    saturation_values = saturation_test.flatten()

    # Update the histogram with the counts of hue and saturation values
    histogram += np.histogram2d(hue_values, saturation_values,
                                bins=histogram_bins, range=[[0, 180], [0, 256]])[0]

# Normalize the histogram to [0, 1]
histogram_normalized = histogram / np.max(histogram)

# Threshold for considering a pixel as skin tone based on histogram value
threshold = 0

# Apply the mask to the original image to extract skin tone regions
skin_regions = np.copy(image)

for i, row in enumerate(image_array):
    for j, pix in enumerate(row):
        # print(pix)
        if histogram[pix[0]][pix[1]] > 0:
            pass
        else:
            skin_regions[i][j] = [0, 0, 0]

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
