# Srikanth Schelbert Homework1 (MP2)

import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import math
from PIL import Image

# Open the .bmp file
image = Image.open("homework2/gun.bmp")
# matplotlib.image.imsave('name.png', my_array)

# Display basic information about the image
print("Image format:", image.format)
print("Image mode:", image.mode)
print("Image size:", image.size)

# Show the image
image.show()


def dilate(image):
    """Dilates an image"""
    new_image = np.zeros([image.size[0], image.size[1]])
    # 1 check each pixel for foreground
    for u in range(image.size[0]):
        for v in range(image.size[1]):
            if image.getpixel((u, v)) == 0:
                pass
            else:
                # 2 add all surrounding pixels to fg if pixel is in fg and in image
                new_image = dilate_new_image(new_image, u, v)
    return new_image


def dilate_new_image(new_image, u, v):
    """modifies the new image 2-D array to add dilation"""
    for i in range(u - 1, u + 2):
        for j in range(v - 1, v + 2):
            if (0 <= i < new_image.shape[0]) and (0 <= j < new_image.shape[1]):
                # make 1
                new_image[i][j] = 1
            else:
                pass
    return new_image


new_img = dilate(image)
new_img2 = new_img.T[::-1]
# Plot the array as an image
plt.imshow(new_img2, cmap='viridis', origin='lower')
plt.colorbar()
plt.show()
