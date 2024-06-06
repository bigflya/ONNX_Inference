import onnxruntime as ort
from model_inference_library.engines.base import InferenceEngine

import cv2
import numpy as np

from model_inference_library.utils.label_processing import class_file

class ONNXRuntimeEngine(InferenceEngine):
    def __init__(self):
        self.session = None
        self.input_width:int
        self.input_height:int
        self.img_width:int
        self.img_height:int


    def load_model(self, model_path):
        # Create an inference session using the ONNX model and specify execution providers
        #self.session = ort.InferenceSession(model_path)
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])



    def load_class_file(self, class_path):
        self.classes = class_file(class_path)
        return self.classes

    def preprocess(self, image):

        # Get the model inputs
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocesses the input image before performing inference.

        # Get the height and width of the input image
        self.img_height, self.img_width = image.shape[:2]

        # Convert the image color space from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        image = cv2.resize(image, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(image) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data


    def infer(self, preprocessed_image):
        # Get the model inputs
        model_inputs = self.session.get_inputs()
        # Run inference using the preprocessed image data
        results = self.session.run(None, {model_inputs[0].name: preprocessed_image})
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(results[0]))
        return outputs

    def postprocess(self, _,outputs, conf, nms, score, detections):

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height


        # Get the number of rows in the outputs array
        rows = outputs.shape[0]
        detections.clear()




        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            maxScore = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if maxScore >= score:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                box = [
                    int((x - w / 2) * x_factor),
                    int((y - h / 2) * y_factor),
                    int(w * x_factor),
                    int(h * y_factor),
                ]
                # Add the class ID, score, and box coordinates to the respective lists
                detections.add_detection(box, maxScore, class_id)

        boxes, scores, class_ids = detections.get_detections()
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf, nms,0.5)

        return result_boxes




















