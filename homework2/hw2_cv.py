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
                new_image = erode_new_image(new_image, u, v, image, SE)
    return new_image


def erode_new_image(new_image, u, v, image, SE=(1, 2, 1, 2)):
    """modifies the new image 2-D array to add dilation"""

    set_one = True
    for i in range(u - SE[0], u + SE[1]):
        for j in range(v - SE[2], v + SE[3]):
            if (0 <= i < new_image.shape[0]) and (0 <= j < new_image.shape[1]):
                # set u,v pixel to 0 if it doesn't match image to SE
                if int(image.getpixel((i, j))) >= 1:
                    pass
                else:
                    # keeps u,v val 0 if not a match
                    set_one = False

            else:
                pass

    # Sets u,v value to 1 if it goes through the whole loop and never breaks
        # no just only set to 1 if loops complete otherwise it is already 0
    if set_one == True:
        new_image[u][v] = 1
    return new_image


def convert_to_image(array):
    """Converts a numpy array to an image"""
    return Image.fromarray(array * 255)


def closing(image, input_SE=(2, 2, 2, 2)):
    "closes holes in an image"
    new_img = dilate(image, SE=input_SE)
    new_img2 = erode(convert_to_image(new_img), SE=input_SE)
    new_img3 = convert_to_image(new_img2)
    new_img3.show("Closed Image")


def opening(image, input_SE=(2, 2, 2, 2)):
    """Opens gaps within an image"""
    new_img = erode(image, SE=input_SE)
    new_img2 = dilate(convert_to_image(new_img), SE=input_SE)
    new_img3 = convert_to_image(new_img2)
    new_img3.show("Opened Image")


def boundary(image, input_SE=(2, 2, 2, 2)):
    """Finds the boundary of the image"""
    # create first closed image
    new_img = dilate(image, SE=input_SE)
    new_img2 = erode(convert_to_image(new_img), SE=input_SE)

    # Create new dilated image with closed array
    new_img6 = dilate(convert_to_image(new_img2), SE=(1, 1, 1, 1))
    print(new_img6.shape)
    new_img7 = new_img6.T[:-1:]

    # List of extra zeros to add
    zeros_list = [0] * new_img6.shape[0]

    # Append the list to the numpy array
    new_arr = np.append(new_img7, [zeros_list], axis=0)

    # subtract images and show
    new_img4 = convert_to_image(new_arr - new_img2)
    new_img4.show()

# new_img = dilate(image, SE=(2, 2, 2, 2))
# # new_img2 = new_img.T[::-1]
# new_img2 = erode(convert_to_image(new_img), SE=(2, 2, 2, 2))
# new_img3 = convert_to_image(new_img2)
# new_img3.show()
# new_img4 = new_img3.T[::-1]
# Plot the array as an imagebreak_outer
# plt.imshow(new_img3, cmap='viridis', origin='lower')
# plt.colorbar()
# plt.show()


if __name__ == "__main__":
    response = input(
        "\nWhat would you like to do? \nErode: e\nDilate: d\nClose: c\nOpen: o\nBoundary: b\n(Type Answer):")
    match response:
        case 'd':
            # Show the image
            image.show("Original Image")
            new_img = dilate(image, SE=(2, 2, 2, 2))
            new_img2 = new_img.T[::-1]
            plt.imshow(new_img2, cmap='viridis', origin='lower')
            plt.colorbar()
            plt.show()
        case 'e':
            # Show the image
            image.show("Original Image")
            new_img = erode(image, SE=(1, 1, 1, 1))
            new_img2 = new_img.T[::-1]
            plt.imshow(new_img2, cmap='viridis', origin='lower')
            plt.colorbar()
            plt.show()
        case 'c':
            # Show the image
            image.show("Original Image")
            closing(image, (3, 3, 3, 3))
        case 'o':
            # Show the image
            image.show("Original Image")
            opening(image, (1, 1, 1, 1))
        case 'b':
            image.show()
            boundary(image)
