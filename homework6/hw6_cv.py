import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from scipy.ndimage import maximum_filter
import cv2
import math

# Open the .bmp file
# image = Image.open("homework6/test.bmp")

# Display basic information about the image
# print("Image format:", image.format)
# print("Image mode:", image.mode)
# print("Image size:", image.size)
# image.show("Original Image")


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=50):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def show_hough_line(img, accumulator, thetas, rhos, save_path=None, thresh=30):

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image w/ red hough lines superposed')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='viridis',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # Calculate lines from the accumulator
    # Non-maximum suppression to only use one line per local maximum
    maxima = maximum_filter(accumulator, size=20)
    accumulator = (accumulator == maxima) & (accumulator > thresh)
    new_rhos, new_thetas = np.where(accumulator)
    for q in range(len(new_rhos)):
        a = np.cos(thetas[new_thetas[q]])
        b = np.sin(thetas[new_thetas[q]])
        x0 = a * rhos[new_rhos[q]]
        y0 = b * rhos[new_rhos[q]]
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        ax[0].plot((x1, x2), (y1, y2), color='red')

    # Saves the image to desired directory if given
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


if __name__ == '__main__':
    imgpath = 'homework6/input.bmp'
    img = imageio.imread(imgpath)
    if img.ndim == 3:
        img = rgb2gray(img)
    edge_img = cv2.Canny(img, 100, 200)

    accumulator, thetas, rhos = hough_line(edge_img)
    show_hough_line(img, accumulator, thetas, rhos,
                    save_path='homework6/output.png')
