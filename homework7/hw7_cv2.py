import cv2
import numpy as np
import os

# Function to compute similarity measures


def compute_similarity(template, patch, method="SSD"):
    if method == "SSD":
        return np.sum((template - patch) ** 2)
    elif method == "CC":
        return np.sum(template * patch)
    elif method == "NCC":
        template_norm = (template - np.mean(template)) / np.std(template)
        patch_norm = (patch - np.mean(patch)) / np.std(patch)
        return np.sum(template_norm * patch_norm)
    else:
        raise ValueError("Invalid method")


# Load images from directory
image_dir = 'homework7/image_girl'
image_files = sorted([f for f in os.listdir(image_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg'))])

# Read the first image
frame = cv2.imread(os.path.join(image_dir, image_files[0]))

# Resize the image to be twice as large
frame = cv2.resize(frame, None, fx=2, fy=2)

# Select the initial region to track
x, y, w, h = cv2.selectROI("Select Target", frame, False, False)

# Define the template
template = frame[y:y+h, x:x+w]

# Define the bounding box color
color = (0, 255, 0)

# Define the output video file
output_file = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, 10.0,
                      (frame.shape[1], frame.shape[0]))

# Chosen method of calculation
chosen_method = "SSD"

# Loop through the images
for image_file in image_files:
    # Read the image
    frame = cv2.imread(os.path.join(image_dir, image_file))

    # Resize the image to be twice as large
    frame = cv2.resize(frame, None, fx=2, fy=2)

    # Compute the similarity score for each pixel in the frame
    scores = np.zeros((frame.shape[0] - h, frame.shape[1] - w))
    for i in range(frame.shape[0] - h):
        for j in range(frame.shape[1] - w):
            patch = frame[i:i+h, j:j+w]
            scores[i, j] = compute_similarity(
                template, patch, chosen_method)

    # Find the position of the target
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(scores)

    # argmin for SSD argmax for CC
    if chosen_method == "SSD":
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw the bounding box around the target
    cv2.rectangle(frame, top_left, bottom_right, color, 2)

    # Write frame to output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# Release the video writer and close windows
out.release()
# Close windows
cv2.destroyAllWindows()
