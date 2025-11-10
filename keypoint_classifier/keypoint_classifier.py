import numpy as np
import tensorflow as tf


class KeyPointClassifier:
    def __init__(self, model_path="keypoint_classifier/keypoint_classifier.tflite", num_threads: int = 1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        idx_in = self.input_details[0]["index"]
        self.interpreter.set_tensor(idx_in, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        idx_out = self.output_details[0]["index"]
        result = self.interpreter.get_tensor(idx_out)
        return int(np.argmax(np.squeeze(result)))
