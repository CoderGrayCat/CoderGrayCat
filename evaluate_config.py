import os
import numpy as np

# Note: All coordinates are based on the actual coordinates used for drawing boxes on an image with dimensions [256, 256].

# Path to the directory storing the ground truth results.
# Data format: label_index, cx, cy, w, h
# label_index: The index of the label name in the label array, starting from 0.
# cx: The x-coordinate of the center point of the bounding box, calculated as the original center point x-coordinate divided by the image width.
# cy: The y-coordinate of the center point of the bounding box, calculated as the original center point y-coordinate divided by the image height.
# Note: The coordinates mentioned above are normalized coordinates.

# w: The width of the bounding box, calculated as the original bounding box width divided by the image width.
# h: The height of the bounding box, calculated as the original bounding box height divided by the image height.
label_path = r'your truth label path'


# Path to the directory storing the predicted results in numpy format.
# Data format: xmin, ymin, xmax, ymax, confidence, class
# xmin: The x-coordinate of the top-left corner of the bounding box.
# ymin: The y-coordinate of the top-left corner of the bounding box.
# xmax: The x-coordinate of the bottom-right corner of the bounding box.
# ymax: The y-coordinate of the bottom-right corner of the bounding box.
# confidence: The confidence score associated with the bounding box.
# class: The class label associated with the bounding box.
predict_path = r'your predict numpy path'

# Dimensions of the image.
img_size = [256, 256]

# Minimum overlap required for a predicted bounding box to be considered a true positive.
MINOVERLAP = 0.45

# Class names for the objects of interest.
className = ['ship', 'plane']

# Flag indicating whether to automatically delete files.
AUTO_DEL = True
