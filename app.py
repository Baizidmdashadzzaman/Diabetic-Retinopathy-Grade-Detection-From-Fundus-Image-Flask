import os
from flask import Flask, request, render_template, jsonify, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import disk, opening, closing, black_tophat

app = Flask(__name__)

TARGET_IMAGE_SIZE = (640, 640)
NUM_CLASSES = 5
MODEL_PATH = 'model/diabetic_retinopathy_attention_model.keras'

CLASS_NAMES = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR'
}

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        modulating_factor = tf.keras.backend.pow(1. - p_t, gamma)
        loss = alpha * modulating_factor * cross_entropy
        return tf.keras.backend.sum(loss, axis=-1)
    return focal_loss_fixed

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'focal_loss_fixed': focal_loss()}
    )
    model.summary()
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def suppress_vessels(image, kernel_length):
    if kernel_length % 2 == 0:
        kernel_length += 1
    angles = np.arange(0, 180, 15)
    vessel_response = np.zeros_like(image, dtype=np.float32)
    for angle in angles:
        kernel = cv2.getGaborKernel((kernel_length, kernel_length), sigma=kernel_length / 4.0,
                                    theta=np.deg2rad(angle), lambd=kernel_length / 2.0,
                                    gamma=0.5, psi=0)
        kernel -= kernel.mean()
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
        vessel_response = np.maximum(vessel_response, filtered)
    if vessel_response.max() > 0:
        vessel_response = (vessel_response - vessel_response.min()) / (vessel_response.max() - vessel_response.min())
    return vessel_response

def decompose_lesions(image):
    image_for_cv = ((image + 1.0) * 127.5).astype(np.uint8)
    green_channel = image_for_cv[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    green_proc = clahe.apply(green_channel)
    vessel_map = suppress_vessels(green_proc, 15)
    vessel_scaled = vessel_map * 0.8
    lesion_input = np.clip(green_proc * (1 - vessel_scaled), 0, 255)
    se_bright = disk(7)
    opened = opening(lesion_input, se_bright)
    bright_map = np.maximum(0, lesion_input - opened).astype(np.float32)
    bright_map = bright_map / bright_map.max() if bright_map.max() > 0 else bright_map
    bright_map[bright_map < 0.015] = 0
    dark_maps = []
    for radius in [3, 5, 10, 15, 20]:
        se_dark = disk(radius)
        top_hat = black_tophat(lesion_input, se_dark).astype(np.float32)
        if top_hat.max() > 0:
            top_hat /= top_hat.max()
        dark_maps.append(top_hat)
    dark_map = np.maximum.reduce(dark_maps)
    dark_map[dark_map < 0.005] = 0
    closed = closing((dark_map > 0).astype(np.uint8) * 255, disk(5))
    dark_map = dark_map * (closed / 255.0)
    bright_map_norm = (bright_map * 2.0) - 1.0
    dark_map_norm = (dark_map * 2.0) - 1.0
    return bright_map_norm, dark_map_norm

def preprocess_image_for_prediction(image_path):
    try:
        img_pil = Image.open(image_path).convert("RGB")
        resized_img_pil = img_pil.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
        img_array = np.array(resized_img_pil, dtype=np.float32)
        original_img_norm = (img_array / 127.5) - 1.0
        bright_map, dark_map = decompose_lesions(original_img_norm)
        bright_map_3ch = np.stack([bright_map, bright_map, bright_map], axis=-1)
        dark_map_3ch = np.stack([dark_map, dark_map, dark_map], axis=-1)
        input_original_batch = np.expand_dims(original_img_norm, axis=0)
        input_bright_map_batch = np.expand_dims(bright_map_3ch, axis=0)
        input_dark_map_batch = np.expand_dims(dark_map_3ch, axis=0)
        return input_original_batch, input_bright_map_batch, input_dark_map_batch
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        temp_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, file.filename)
        file.save(filepath)
        original_img, bright_map, dark_map = preprocess_image_for_prediction(filepath)
        os.remove(filepath)
        if original_img is None:
            return jsonify({'error': 'Failed to preprocess image. Please ensure it\'s a valid image file.'}), 400
        try:
            predictions = model.predict([original_img, bright_map, dark_map])
            predicted_class_index = np.argmax(predictions[0])
            predicted_probability = float(np.max(predictions[0]))
            predicted_class_name = CLASS_NAMES.get(predicted_class_index, f'Unknown Class {predicted_class_index}')
            all_probabilities = {CLASS_NAMES.get(i, f'Class {i}'): float(prob) for i, prob in enumerate(predictions[0])}
            return jsonify({
                'prediction': {
                    'class_index': int(predicted_class_index),
                    'class_name': predicted_class_name,
                    'probability': predicted_probability
                },
                'all_probabilities': all_probabilities
            })
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {e}'}), 500

@app.route('/eye_detection')
def eye_detection_page():
    return render_template('eye_detection.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
