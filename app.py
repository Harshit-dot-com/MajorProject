from flask import Flask, request, jsonify
from PIL import Image
from torchvision.transforms import ToTensor
from flask_cors import CORS
import numpy as np
import onnxruntime as ort
from nudenet import NudeDetector
import cv2
import math
import tempfile
from Nudenet import NudityDetector
import os

app = Flask(__name__)
CORS(app)

def has_detection_above_threshold(image_data, alcohol_model_path, weapons_model_path, threshold=0.7):
    # Save the uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image_data.save(temp_file.name)
        temp_file_path = temp_file.name

    # Detect nudity
    nudity_detected = NudityDetector(temp_file_path)

    # Detect alcohol and weapons
    im = Image.open(temp_file_path).convert('RGB')
    im = im.resize((640, 640))
    im_data = ToTensor()(im)[None]
    size = np.array([[640, 640]], dtype=np.int64)

    alcohol_sess = ort.InferenceSession(alcohol_model_path)
    alcohol_output = alcohol_sess.run(
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size}
    )
    alcohol_labels, _, alcohol_scores = alcohol_output
    alcohol_detected = any(alcohol_scores[0] > threshold)

    weapons_sess = ort.InferenceSession(weapons_model_path)
    weapons_output = weapons_sess.run(
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size}
    )
    weapons_labels, _, weapons_scores = weapons_output
    weapons_detected = any(weapons_scores[0] > threshold)

    # Combine results
    if alcohol_detected and weapons_detected and nudity_detected:
        result_string = "Nudity, Alcohol, and Weapons Detected"
    elif alcohol_detected and nudity_detected:
        result_string = "Nudity and Alcohol Detected"
    elif weapons_detected and nudity_detected:
        result_string = "Nudity and Weapons Detected"
    elif alcohol_detected and weapons_detected:
        result_string = "Alcohol and Weapons Detected"
    elif alcohol_detected:
        result_string = "Alcohol Detected"
    elif weapons_detected:
        result_string = "Weapons Detected"
    elif nudity_detected:
        result_string = "Nudity Detected"
    else:
        result_string = "No Detection"

    print("Alcohol Scores:", alcohol_scores)
    print("Weapons Scores:", weapons_scores)
    print("Nudity Detection:", nudity_detected)

    return result_string


@app.route('/api/detect', methods=['POST'])
def detect():
    alcohol_model_path = r'C:\Users\harsh\Downloads\alcohol.onnx'
    weapons_model_path = r'C:\Users\harsh\Downloads\model1.onnx'
    threshold = 0.9

    image_file = request.files['image']

    if image_file and (image_file.filename.endswith('.jpg') or image_file.filename.endswith('.jpeg') or image_file.filename.endswith('.png')):
        result = has_detection_above_threshold(image_file, alcohol_model_path, weapons_model_path, threshold)
        print(result)
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
