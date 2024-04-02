# Srikanth Schelbert Homework1 (MP1)

import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image


# Open the .bmp file
image = Image.open("homework1/test.bmp")

# Display basic information about the image
print("Image format:", image.format)
print("Image mode:", image.mode)
print("Image size:", image.size)

equivalences = []

# Show the image
image.show()


def set_equivalence(val1, val2):
    if equivalences:
        for i in range(len(equivalences)):
            if val1 in equivalences[i] and val2 in equivalences[i]:
                pass
            elif val1 in equivalences[i]:
                equivalences[i].add(val2)
            elif val2 in equivalences[i]:
                equivalences[i].add(val1)
            else:
                equivalences.append({val1, val2})
    else:
        equivalences.append({val1, val2})
    print(equivalences)


def correct_labels(label_mat, eq_list):
    for i in range(label_mat.shape[0]):
        for j in range(label_mat.shape[1]):
            for k in range(len(eq_list)):
                if label_mat[i][j] in eq_list[k]:
                    label_mat[i][j] = min(eq_list[k])


def CCL(image):
    labels = np.zeros([image.size[0], image.size[1]])
    highest_label = 1
    for u in range(image.size[0]):
        for v in range(image.size[1]):
            # print(image.getpixel((u, v)))
            # check if value is 0
            # print(highest_label)
            if image.getpixel((u, v)) == 0:
                pass
            elif (0 < u < (image.size[0] - 1)) and (0 < v < (image.size[1] - 1)):
                # check that u and v arent at the edges
                # print(u, v)
                up, left = labels[u][v - 1], labels[u - 1][v]
                if up == 0 and left == 0:
                    labels[u][v] = highest_label
                    highest_label += 1
                elif up > 0 and left > 0:
                    labels[u][v] = min([up, left])
                    set_equivalence(up, left)
                    print("setting")
                elif up > 0 and left == 0:
                    labels[u][v] = up
                elif up == 0 and left > 0:
                    labels[u][v] = left
            elif u == 0 and v > 0:
                # u edge
                if labels[u][v - 1] == 0:
                    labels[u][v] = highest_label
                    highest_label += 1
                else:
                    labels[u][v] = labels[u][v - 1]
            elif u > 0 and v == 0:
                # v edge
                if labels[u - 1][v] == 0:
                    labels[u][v] = highest_label
                    highest_label += 1
                else:
                    labels[u][v] = labels[u][v - 1]
    return labels


CCL(image)
