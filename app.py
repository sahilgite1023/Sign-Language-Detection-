from flask import Flask, render_template, request, jsonify, Response
import os
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import numpy as np
import base64
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import csv
import copy
import itertools
from keypoint_classifier.keypoint_classifier import KeyPointClassifier

counter = 0
alpha = 'A'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# define the flask app
app=Flask(__name__)

# Create required directories
required_dirs = ['uploads', 'images/marked/Y', 'images/skeleton/Y', 'images/marked/test', 'images/skeleton/test']
for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Load the pre-trained model
print("Loading pre-trained model...")
try:
    # Get current directory and model paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'Model')
    
    # Ensure model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found at {model_dir}")
    
    # Define model paths
    letter_model_path = os.path.join(model_dir, 'sign_language_model_improved.h5')
    phrase_model_path = os.path.join(model_dir, 'keras_model.h5')
    labels_path = os.path.join(model_dir, 'labels.txt')
    
    # Check if model files exist
    for path in [letter_model_path, phrase_model_path, labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required model file not found: {path}")
    
    # Load models with custom_objects to handle compatibility
    try:
        # Custom objects to handle DepthwiseConv2D compatibility
        custom_objects = {
            'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
            'Dense': tf.keras.layers.Dense,
            'Input': tf.keras.layers.Input,
            'Model': tf.keras.models.Model,
            'Sequential': tf.keras.models.Sequential
        }
        
        # Load models with custom objects and safe_mode
        letter_model = load_model(letter_model_path, custom_objects=custom_objects, compile=False, safe_mode=True)
        phrase_model = load_model(phrase_model_path, custom_objects=custom_objects, compile=False, safe_mode=True)
        
        # Compile models
        letter_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        phrase_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        raise Exception(f"Error loading models: {e}")
    
    # Load labels
    try:
        with open(labels_path, 'r') as f:
            phrase_labels = [line.strip().split(' ')[1] for line in f.readlines()]
    except Exception as e:
        raise Exception(f"Error loading labels: {e}")
    
    print("Pre-trained models and labels loaded successfully")
except Exception as e:
    print(f"Error during model initialization: {e}")
    raise

# Image size for MobileNetV2
IMG_SIZE = 224

# Initialize MediaPipe Hands with improved parameters
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam and real-time recognition components
camera = None
is_camera_active = False
detector = HandDetector(maxHands=1)
classifier = Classifier(os.path.join(current_dir, 'Model', 'keras_model.h5'), os.path.join(current_dir, 'Model', 'labels.txt'))

# Initialize keypoint classifier for A-Z detection
keypoint_classifier = KeyPointClassifier(
    model_path=os.path.join(current_dir, 'keypoint_classifier', 'keypoint_classifier.tflite'),
    num_threads=1,
)

# Load keypoint classifier labels
with open(os.path.join(current_dir, 'keypoint_classifier', 'keypoint_classifier_label.csv'),
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

# Constants for real-time recognition
offset = 20
imgSize = 300
labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

# Global variable to track recognition mode
recognition_mode = 'letter'  # Default to letter recognition

# Global variable to select classifier type for letter recognition
classifier_type = 'keypoint'  # 'keypoint' for keypoint-based, 'cnn' for image-based

# Global variable to store recognized words for both hands
recognized_words_global = ["", ""]
# Global flag to clear recognized words
clear_flag = False

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global recognition_mode
    recognition_mode = request.json.get('mode', 'letter')
    return jsonify({'status': 'success', 'mode': recognition_mode})

@app.route('/set_classifier', methods=['POST'])
def set_classifier():
    global classifier_type
    classifier_type = request.json.get('type', 'keypoint')
    return jsonify({'status': 'success', 'type': classifier_type})

def generate_frames():
    cap = None
    global recognized_words_global, clear_flag
    recognized_words = ["", ""]  # For left and right hand
    last_predicted = [None, None]   # Track last prediction for each hand
    try:
        # Try multiple times to open the camera with more detailed error handling
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if cap is not None:
                    cap.release()
                # Try to get the list of available cameras
                camera_list = []
                index = 0
                while True:
                    temp_cap = cv2.VideoCapture(index)
                    if not temp_cap.isOpened():
                        break
                    camera_list.append(index)
                    temp_cap.release()
                    index += 1
                
                if not camera_list:
                    raise Exception("No cameras detected on your system. Please connect a camera and try again.")
                
                # Try to open the first available camera
                cap = cv2.VideoCapture(camera_list[0])
                if not cap.isOpened():
                    raise Exception("Camera device not accessible. Please check if camera is connected and not in use by another application.")
                
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test reading a frame to ensure camera is working
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    raise Exception("Unable to capture video frame. Please check camera permissions in your browser and system settings.")
                
                print(f"Camera initialized successfully on attempt {attempt + 1}")
                break
            except Exception as e:
                print(f"Camera access attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    print("Waiting before next attempt...")
                    time.sleep(2)  # Wait before retry
                else:
                    raise Exception("Camera initialization failed after multiple attempts. Please ensure:\n1. Camera is properly connected\n2. Camera permissions are granted in your browser\n3. No other application is using the camera\n4. Try disconnecting and reconnecting your camera")
        
        while True:
            # Check if clear_flag is set
            if clear_flag:
                recognized_words = ["", ""]
                recognized_words_global = ["", ""]
                last_predicted = [None, None]
                clear_flag = False
            success, img = cap.read()
            if not success or img is None or img.size == 0:
                raise Exception("Failed to grab valid frame from camera.")
            
            imgOutput = img.copy()
            try:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgRGB.flags.writeable = False
                results = hands.process(imgRGB)
                imgRGB.flags.writeable = True
                # Track which hands are detected this frame
                hands_detected = [False, False]
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        x_min = y_min = float('inf')
                        x_max = y_max = float('-inf')
                        for landmark in hand_landmarks.landmark:
                            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                            x_min = min(x_min, x)
                            x_max = max(x_max, x)
                            y_min = min(y_min, y)
                            y_max = max(y_max, y)
                        x_min = max(0, x_min - offset)
                        y_min = max(0, y_min - offset)
                        x_max = min(img.shape[1], x_max + offset)
                        y_max = min(img.shape[0], y_max + offset)
                        landmark_list = calc_landmark_list(img, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        predicted_text = "-"
                        confidence = 0.0
                        try:
                            if recognition_mode == 'phrase':
                                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                                imgCrop = img[y_min:y_max, x_min:x_max]
                                aspectRatio = (y_max - y_min) / (x_max - x_min) if (x_max - x_min) > 0 else 1
                                if aspectRatio > 1:
                                    k = imgSize / (y_max - y_min)
                                    wCal = math.ceil(k * (x_max - x_min))
                                    if wCal > 0:
                                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                                        wGap = math.ceil((imgSize-wCal)/2)
                                        imgWhite[:, wGap:wGap+wCal] = imgResize
                                else:
                                    k = imgSize / (x_max - x_min)
                                    hCal = math.ceil(k * (y_max - y_min))
                                    if hCal > 0:
                                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                                        hGap = math.ceil((imgSize-hCal)/2)
                                        imgWhite[hGap:hGap+hCal, :] = imgResize
                                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                                predicted_text = labels[index]
                                confidence = float(np.max(prediction))
                            else:
                                if classifier_type == 'keypoint':
                                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                                    predicted_text = keypoint_classifier_labels[hand_sign_id]
                                    confidence = 1.0  # tflite classifier does not provide confidence
                                else:
                                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                                    imgCrop = img[y_min:y_max, x_min:x_max]
                                    aspectRatio = (y_max - y_min) / (x_max - x_min) if (x_max - x_min) > 0 else 1
                                    if aspectRatio > 1:
                                        k = imgSize / (y_max - y_min)
                                        wCal = math.ceil(k * (x_max - x_min))
                                        if wCal > 0:
                                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                                            wGap = math.ceil((imgSize-wCal)/2)
                                            imgWhite[:, wGap:wGap+wCal] = imgResize
                                    else:
                                        k = imgSize / (x_max - x_min)
                                        hCal = math.ceil(k * (y_max - y_min))
                                        if hCal > 0:
                                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                                            hGap = math.ceil((imgSize-hCal)/2)
                                            imgWhite[hGap:hGap+hCal, :] = imgResize
                                    img_array = cv2.resize(imgWhite, (IMG_SIZE, IMG_SIZE))
                                    img_array = preprocess_input(img_array)
                                    img_array = np.expand_dims(img_array, axis=0)
                                    prediction = letter_model.predict(img_array)
                                    predicted_text = chr(ord('A') + np.argmax(prediction[0]))
                                    confidence = float(np.max(prediction[0]))
                            # Draw prediction/confidence for each hand
                            cv2.rectangle(imgOutput, (x_min, y_min-70), (x_min+400, y_min+60-50), (0, 255, 0), cv2.FILLED)
                            cv2.putText(imgOutput, f"{predicted_text} ({confidence:.2f})", (x_min, y_min-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                            cv2.rectangle(imgOutput, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
                            mp_drawing.draw_landmarks(
                                imgOutput,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                            # Debounce: only append if different from last and not "-"
                            if idx < 2:
                                hands_detected[idx] = True
                                if predicted_text != "-" and predicted_text != last_predicted[idx]:
                                    recognized_words[idx] += predicted_text
                                    last_predicted[idx] = predicted_text
                        except Exception as e:
                            print(f"Prediction error: {e}")
                # If a hand is not detected, reset last_predicted for that hand
                for idx in range(2):
                    if not hands_detected[idx]:
                        last_predicted[idx] = None
            except Exception as e:
                print(f"Frame processing error: {e}")

            # At the end of each frame, update the global recognized words
            recognized_words_global = recognized_words.copy()
            ret, buffer = cv2.imencode('.jpg', imgOutput)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Camera error: {e}")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        error_msg = str(e)
        # Split error message into multiple lines if too long
        words = error_msg.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 40:  # Max chars per line
                lines.append(' '.join(current_line[:-1]))
                current_line = [current_line[-1]]
        if current_line:
            lines.append(' '.join(current_line))
            
        # Draw error message
        cv2.putText(error_frame, "Camera Error", (200, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        y_pos = 200
        for line in lines:
            cv2.putText(error_frame, line, (50, y_pos), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 40
        cv2.putText(error_frame, "Please check camera settings and try again", (50, y_pos + 40), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        if cap is not None:
            cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/team', methods=['GET'])
def team():
    return render_template('team.html')

@app.route('/recognized_text')
def recognized_text():
    global recognized_words_global
    return jsonify({
        'left_hand': recognized_words_global[0],
        'right_hand': recognized_words_global[1]
    })

@app.route('/clear_recognized_text', methods=['POST'])
def clear_recognized_text():
    global recognized_words_global, clear_flag
    recognized_words_global = ["", ""]
    clear_flag = True
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True)
