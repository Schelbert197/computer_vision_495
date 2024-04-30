import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# from scipy import ndimage
# from scipy import misc
from scipy.ndimage import convolve


# Open the .bmp file
image = Image.open("homework5/pointer1.bmp")

# Display basic information about the image
print("Image format:", image.format)
print("Image mode:", image.mode)
print("Image size:", image.size)
# image.show("Original Image")


def gaussian_kernel(size, sigma=1):
    """Creates a gaussian kernel for the smoothing"""
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def gauss_smoothing(image, kernel_size=5, sigma=1, make_gray=True):
    """Takes a raw image and returns a smoothed image"""
    # Convert the image to grayscale np array if true
    if make_gray == True:
        image_gray = image.convert('L')
    else:
        image_gray = image
    image_array = np.array(image_gray)

    smoothed_image = convolve(image_array, np.array(
        gaussian_kernel(kernel_size, sigma)))
    return smoothed_image


def image_gradient(img, f_type='S'):
    """Applies a filter operator to the image based on selected operator"""
    if f_type == 'S':
        # Sobel Operator
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    elif f_type == 'C':
        # Robert Cross Operator
        Kx = np.array([[1, 0], [0, -1]], np.float32)
        Ky = np.array([[0, -1], [1, 0]], np.float32)
    else:
        # Prewitt operator
        Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

               # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, percentage=0.1):

    # highThreshold = img.max() * 0.15
    # lowThreshold = highThreshold * 0.5

    lowThreshold, highThreshold = find_threshold(img, percentage)

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = highThreshold
    strong = lowThreshold

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, lowThreshold, highThreshold


def find_threshold(mag, percent):
    """Finds the threshold"""
    hist, bins = np.histogram(mag.ravel(), bins=256, range=(0, 255))
    hist = hist / np.sum(hist)
    cum_hist = np.cumsum(hist)
    threshold_high = 0
    for i in range(256):
        if cum_hist[i] >= percent:
            threshold_high = i
            break

    threshold_low = threshold_high * 0.5

    print(threshold_low)
    print(threshold_high)

    return threshold_low, threshold_high


def hysteresis(img, t_low, t_high):

    M, N = img.shape
    weak = t_low
    strong = t_high

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img


def plot_image(image, index, title):
    """Plots the image as a subplot"""
    # Display the smoothed image
    plt.subplot(2, 3, index)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)


plot_image(np.array(image.convert('L')), 1, 'Original Image')
# Apply Gaussian smoothing
smooth_img = gauss_smoothing(image, kernel_size=2, sigma=7.0)
plot_image(smooth_img, 2, 'Smoothed Image')
mag, theta = image_gradient(smooth_img, f_type='P')
plot_image(mag, 3, 'Gradient Image')
thresh, t_low, t_high = threshold(mag, 0.9)
plot_image(thresh, 4, 'Threshold Image')
non_max = non_max_suppression(thresh, theta)
plot_image(non_max, 5, 'Suppressed Image')
final = hysteresis(thresh, t_low, t_high)
plot_image(final, 6, 'Final Image')


# Display the smoothed image
plt.show()
