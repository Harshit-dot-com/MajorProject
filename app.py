from flask import Flask, request, jsonify
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import ToTensor
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

def has_detection_above_threshold(image_data, alcohol_model_path, weapons_model_path, threshold=0.6):
    
    im = Image.open(image_data).convert('RGB')
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

    if alcohol_detected and weapons_detected:
        result_string = "Both Alcohol and Weapons Detected"
    elif alcohol_detected:
        result_string = "Alcohol Detected"
    elif weapons_detected:
        result_string = "Weapons Detected"
    else:
        result_string = "No Detection"

    return result_string

@app.route('/api/detect', methods=['POST'])
def detect():
    alcohol_model_path = r'C:\Users\harsh\Downloads\alcohol.onnx'
    weapons_model_path = r'C:\Users\harsh\Downloads\model1.onnx'
    threshold = 0.8

    image_file = request.files['image']

    if image_file and (image_file.filename.endswith('.jpg') or image_file.filename.endswith('.jpeg') or image_file.filename.endswith('.png')):
        result = has_detection_above_threshold(image_file, alcohol_model_path, weapons_model_path, threshold)
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
