from flask import Flask, request, jsonify, send_from_directory
import os
import uuid  # Import UUID for generating unique identifiers
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image  # Import for opening the uploaded image
from utils import predict_diabetes_from_image, generate_saliency_map  # Import utility functions
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
SAL_MAP_FOLDER = 'saliency_maps/'  # Folder for saliency maps
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAL_MAP_FOLDER'] = SAL_MAP_FOLDER

# Ensure the upload and saliency map folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(SAL_MAP_FOLDER):
    os.makedirs(SAL_MAP_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

        # Generate the saliency map and get the resized image
        saliency_map, resized_image = generate_saliency_map(image)

        # Save the resized image and saliency map
        resized_filename = f"{unique_id}_resized.png"
        resized_path = os.path.join(app.config['UPLOAD_FOLDER'], resized_filename)
        resized_image.save(resized_path)

        saliency_filename = f"{unique_id}_saliency.png"
        saliency_path = os.path.join(app.config['SAL_MAP_FOLDER'], saliency_filename)
        plt.imsave(saliency_path, saliency_map, cmap='hot')

        # Prepare results
        results = {
            'classification': classification,
            'confidence': confidence
        }

        response = {
            'message': 'File uploaded and model run successfully',
            'image_url': f"http://127.0.0.1:5000/uploads/{resized_filename}",  # Use the resized image URL
            'saliency_url': f"http://127.0.0.1:5000/saliency_maps/{saliency_filename}",
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

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
