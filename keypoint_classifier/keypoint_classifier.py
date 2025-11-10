import numpy as np

# Prefer standalone tflite_runtime (lightweight) and fall back to TensorFlow Lite only if available.
Interpreter = None
_import_error = None
try:  # Lightweight path (no full TensorFlow)
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception as e:  # noqa: BLE001
    _import_error = e
    try:
        from tensorflow.lite import Interpreter  # type: ignore
    except Exception as e2:  # noqa: BLE001
        _import_error = e2


class KeyPointClassifier:
    def __init__(self, model_path="keypoint_classifier/keypoint_classifier.tflite", num_threads: int = 1):
        if Interpreter is None:
            raise ImportError(
                "Neither tflite_runtime nor TensorFlow Lite is available. Install 'tflite-runtime' or 'tensorflow'."
            ) from _import_error
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
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
