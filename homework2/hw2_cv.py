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


def dilate(image, SE=(1, 2, 1, 2)):
    """Dilates an image"""
    new_image = np.zeros([image.size[0], image.size[1]])
    # 1 check each pixel for foreground
    for u in range(image.size[0]):
        for v in range(image.size[1]):
            if image.getpixel((u, v)) == 0:
                pass
            else:
                # 2 add all surrounding pixels to fg if pixel in fg and image
                new_image = dilate_new_image(new_image, u, v, SE)
    return new_image


def dilate_new_image(new_image, u, v, SE=(1, 2, 1, 2)):
    """modifies the new image 2-D array to add dilation"""
    for i in range(u - SE[0], u + SE[1]):
        for j in range(v - SE[2], v + SE[3]):
            if (0 <= i < new_image.shape[0]) and (0 <= j < new_image.shape[1]):
                # make 1
                new_image[i][j] = 1
            else:
                pass
    return new_image


def erode(image, SE=(1, 2, 1, 2)):
    """Erodes an image"""
    new_image = np.zeros([image.size[0], image.size[1]])
    # 1 check each pixel for foreground
    for u in range(image.size[0]):
        for v in range(image.size[1]):
            if image.getpixel((u, v)) == 0:
                pass
            else:
                # 2 add all surrounding pixels to fg if pixel in fg and image
                new_image = erode_new_image(new_image, u, v, SE)
    return new_image


def erode_new_image(new_image, u, v, SE=(1, 2, 1, 2)):
    """modifies the new image 2-D array to add dilation"""
    for i in range(u - SE[0], u + SE[1]):
        for j in range(v - SE[2], v + SE[3]):
            if (0 <= i < new_image.shape[0]) and (0 <= j < new_image.shape[1]):
                # make 1
                new_image[i][j] = 1
            else:
                pass
    return new_image


new_img = dilate(image, SE=(2, 2, 2, 2))
new_img2 = new_img.T[::-1]
# Plot the array as an image
plt.imshow(new_img2, cmap='viridis', origin='lower')
plt.colorbar()
plt.show()
# def superimpose_arrays(large_array, small_array, index):
#     """
#     Superimpose the values of the small array over the larger array at the specified index.
#     """
#     # Get the dimensions of the small array
#     small_shape = small_array.shape

#     # Get the indices to replace in the larger array
#     start_index = (index[0] - small_shape[0] // 2,
#                    index[1] - small_shape[1] // 2)
#     end_index = (start_index[0] + small_shape[0],
#                  start_index[1] + small_shape[1])

#     # Replace the values in the larger array with the values of the smaller array
#     large_array[start_index[0]:end_index[0],
#                 start_index[1]:end_index[1]] = small_array

#     return large_array


# # Example usage:
# # Create large and small arrays
# large_array = np.zeros((10, 10))
# small_array = np.array([[1, 1, 1],
#                         [1, 0, 1],
#                         [1, 1, 1]])

# # Index at which to superimpose the small array
# index = (9, 5)

# # Superimpose the small array over the larger array
# result_array = superimpose_arrays(large_array, small_array, index)

# print("Result array:")
# print(result_array)
