# YOLOv5 Evaluation

This readme provides an overview of the evaluation process using YOLOv5 for object detection. The evaluation code is implemented in Python.

## Program Dependencies

This readme provides a list of libraries required to run the program. Make sure you have the following libraries installed:

- `torch`: A deep learning framework used for training and inference.
- `shutil`: A utility module for high-level file operations.
- `glob`: A module for searching files and directories.
- `json`: A module for working with JSON data.
- `os`: A module for interacting with the operating system.
- `operator`: A module providing a set of efficient functions for performing common operations.
- `sys`: A module that provides access to some variables used or maintained by the interpreter and to functions that interact with the interpreter.
- `argparse`: A module for parsing command-line arguments.
- `math`: A module providing mathematical functions.

It is important to ensure that these libraries are installed and accessible in your Python environment before running the program.



## Usage

1. Define the paths to the ground truth and predicted results files in the appropriate formats in the file `evaluate_config.py`.
    - Ground truth file format: label_index, cx, cy, w, h
        - `label_index`: The index of the label name in the label array, starting from 0.
        - `cx`: The x-coordinate of the center point of the bounding box, calculated as the original center point x-coordinate divided by the image width.
        - `cy`: The y-coordinate of the center point of the bounding box, calculated as the original center point y-coordinate divided by the image height.
        - `w`: The width of the bounding box, calculated as the original bounding box width divided by the image width.
        - `h`: The height of the bounding box, calculated as the original bounding box height divided by the image height.
    - Predicted results file format: xmin, ymin, xmax, ymax, confidence, class
        - `xmin`: The x-coordinate of the top-left corner of the bounding box.
        - `ymin`: The y-coordinate of the top-left corner of the bounding box.
        - `xmax`: The x-coordinate of the bottom-right corner of the bounding box.
        - `ymax`: The y-coordinate of the bottom-right corner of the bounding box.
        - `confidence`: The confidence score associated with the bounding box.
        - `class`: The class label associated with the bounding box.

2. Set the dimensions of the image using the `img_size` variable. The dimensions should match the actual size of the image used for drawing the bounding boxes.

3. Adjust the `MINOVERLAP` variable to specify the minimum overlap required for a predicted bounding box to be considered a true positive.

4. Define the class names for the objects of interest in the `className` list.

5. Set the `AUTO_DEL` flag to `True` if you want to automatically delete files.

## Getting Started

To start the evaluation process, run the `evaluate.py` script.

Note: This readme provides an overview of the evaluation process. Please refer to the `evaluate.py` script for the complete implementation details.

We hope this helps you in evaluating YOLOv5 for object detection. Let us know if you have any further questions.