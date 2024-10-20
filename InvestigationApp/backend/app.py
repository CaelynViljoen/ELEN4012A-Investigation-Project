# InvestigationApp\backend\app.py

# app.py

from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
from utils import resize_and_pad_image, predict_diabetes_from_image, generate_saliency_map, exudates_marked_image, check_guess
import matplotlib.pyplot as plt
import cv2
import numpy as np

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)

UPLOAD_FOLDER = 'uploads/'
SAL_MAP_FOLDER = 'saliency_maps/'
EXUDATES_FOLDER = 'exudates/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAL_MAP_FOLDER'] = SAL_MAP_FOLDER
app.config['EXUDATES_FOLDER'] = EXUDATES_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(SAL_MAP_FOLDER):
    os.makedirs(SAL_MAP_FOLDER)
if not os.path.exists(EXUDATES_FOLDER):
    os.makedirs(EXUDATES_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/resize-image', methods=['POST'])
def resize_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        image_filename = f"{unique_id}_{filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        file.save(image_path)

        image = Image.open(image_path)
        resized_image = resize_and_pad_image(image)

        resized_filename = f"{unique_id}_resized.png"
        resized_path = os.path.join(app.config['UPLOAD_FOLDER'], resized_filename)
        resized_image.save(resized_path)

        return jsonify({
            'message': 'Image resized successfully',
            'resized_image_url': f"http://127.0.0.1:5000/uploads/{resized_filename}"
        }), 200
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/run-model', methods=['POST'])
def run_model():
    # Check if file part is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        # Generate a unique identifier for the file
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        image_filename = f"{unique_id}_{filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        file.save(image_path)

        # Load the image to process with the model
        image = Image.open(image_path)

        # Run the model and get the results
        classification, confidence = predict_diabetes_from_image(image)

        # Set URLs to "N/A" if classification is non-diabetic
        if classification.lower() == "non-diabetic":
            saliency_url = "N/A"
            exudates_url = "N/A"
        else:
            # Generate the saliency map and get the resized image
            saliency_map, resized_image = generate_saliency_map(image)
            image_cv = np.array(image.convert('RGB'))
            image_cv = image_cv[:, :, ::-1]
            exudates_image = exudates_marked_image(image_cv)

            # Save saliency map
            saliency_filename = f"{unique_id}_saliency.png"
            saliency_path = os.path.join(app.config['SAL_MAP_FOLDER'], saliency_filename)
            plt.imsave(saliency_path, saliency_map, cmap='hot')
            saliency_url = f"http://127.0.0.1:5000/saliency_maps/{saliency_filename}"

            # Save exudates image
            exudates_filename = f"{unique_id}_exudates.png"
            exudates_path = os.path.join(app.config['EXUDATES_FOLDER'], exudates_filename)
            exudates_image_rgb = exudates_image
            cv2.imwrite(exudates_path, exudates_image_rgb)
            exudates_url = f"http://127.0.0.1:5000/exudates/{exudates_filename}"

        # Prepare results
        results = {
            'classification': classification,
            'confidence': confidence
        }

        response = {
            'message': 'File uploaded and model run successfully',
            'image_url': f"http://127.0.0.1:5000/uploads/{filename}",  
            'saliency_url': saliency_url,
            'exudates_url': exudates_url,
            'results': results
        }
        return jsonify(response), 200

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/saliency_maps/<filename>')
def saliency_file(filename):
    return send_from_directory(app.config['SAL_MAP_FOLDER'], filename)

@app.route('/exudates/<filename>')
def exudates_file(filename):
    return send_from_directory(app.config['EXUDATES_FOLDER'], filename)

@app.route('/check-guess', methods=['POST'])
def check_guess_route():
    data = request.get_json()
    user_guess = data.get('guess')
    classification = data.get('classification')

    if user_guess is None or classification is None:
        return jsonify({'error': 'Guess or classification missing'}), 400

    # Debugging: Print guess and classification values
    print(f"User Guess: {user_guess}, Model Classification: {classification}")

    # Use the check_guess function to verify the guess
    is_correct = check_guess(classification, user_guess)
    return jsonify({'correct': is_correct}), 200

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)