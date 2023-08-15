################################################################################
#                                YOLOv7 Inference                               #
################################################################################

# File: extractLayerOutput.py
# Description: This script performs object detection using the YOLOv7 model.
# The YOLOv7 model is loaded from an ONNX file and used to detect objects in an input image.
# The detected objects are saved as separate output files.
# This script requires the OpenCV, ONNX, and ONNXRuntime libraries to be installed.

# Usage:
# 1. Set the CUDA flag and model path in the configuration section.
#    - Set `cuda` to True if you want to use CUDA for inference (requires a compatible GPU).
#    - Set `model_path` to the path of the YOLOv7 ONNX model file.
# 2. Provide the path of the input image in the `input_image` variable.
# 3. Specify the desired output layers in the `Layers` list.
# 4. Run the script to perform inference and save the output files.

# Configuration:
# - Set `cuda` to True if you want to use CUDA for inference. Set it to False to use CPU.
# - Set `model_path` to the path of the YOLOv7 ONNX model file.
# - Set `input_image` to the path of the image.

# Dependencies:
# - OpenCV: pip install opencv-python
# - ONNX: pip install onnx
# - ONNXRuntime: pip install onnxruntime

# Author: [GrayCat]
# Date: [2023.8.15]
# Contact: Please contact the author via email at [zhangchiyyds@qq.com]

################################################################################

import cv2
import onnx
import onnxruntime as ort
import numpy as np
from collections import OrderedDict


# Set CUDA flag and model path
cuda = False
model_path = "models/yoloV7.onnx"

# Read input image
input_image = cv2.imread('inference/images/1.png')

# Define available execution providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']


# Function to resize and pad the image
def letterbox(image, new_shape=(1024, 1024), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        ratio = min(ratio, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return image, ratio, (dw, dh)
def run_inference(input_image, output_layers):
    # Load ONNX model
    model = onnx.load(model_path)
    # Add outputs to the model graph
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    # Create an ONNX runtime session
    session = ort.InferenceSession(model_path, providers=providers)

    # Convert image color space from BGR to RGB
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Perform letterboxing on the image
    image, ratio, dwdh = letterbox(input_image, auto=False)

    # Transpose and expand dimensions of the image
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    image /= 255

    # Get input and output names from the session
    input_names = [i.name for i in session.get_inputs()]

    # Prepare input data
    input_data = {input_names[0]: image}

    # Run the inference
    ort_session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    ort_outs = ort_session.run(None, input_data)
    outputs = [x.name for x in ort_session.get_outputs()]
    output_dict = OrderedDict(zip(outputs, ort_outs))

    # Save individual outputs
    for layer in output_layers:
        file_path = f'{layer}.npy'
        np.save(file_path, output_dict[layer])
        print(f"Successfully exported data from {layer} layer: {file_path}")

if __name__ == '__main__':
    # The Layers you want to extract
    Layers = ['489', '529', '569']

    # Run the inference with your Layers
    run_inference(input_image, Layers)

    # you can load the node by the next code
    # result=np.load('489.npy')
    # print(result])
