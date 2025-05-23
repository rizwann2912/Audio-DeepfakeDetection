from flask import Flask, request, jsonify
import os
import torch
import numpy as np
import librosa
import torchaudio
from werkzeug.utils import secure_filename
from flask_cors import CORS
import sys
import importlib.util

# Add project directory to path to import modules
sys.path.append('.')

# Import modules from the project
from models.cnn import ShallowCNN
from preprocess.lfcc import LFCC

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
MODEL_PATH = './saved/ShallowCNN_lfcc_I/best.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    model = ShallowCNN(in_features=1, out_dim= 1)  # LFCC features have 80 dimensions
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def extract_features(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz
    
    # Initialize LFCC feature extractor
    feature_extractor = LFCC(sr=16000, n_lfcc=80, n_mels=128)
    
    # Extract features
    features = feature_extractor(y)
    
    # Convert to tensor
    features = torch.tensor(features).float().unsqueeze(0)  # Add batch dimension
    
    return features

def predict(audio_path, model):
    # Extract features
    features = extract_features(audio_path)
    
    # Move to device
    features = features.to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

# Load model
model = None
try:
    model = load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Routes
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if model is None:
                # Fall back to random prediction if model couldn't be loaded
                import random
                prediction = random.choice([0, 1])
                confidence = random.uniform(0.7, 0.95)
                
                # Notify about the fallback
                return jsonify({
                    'prediction': prediction,
                    'confidence': confidence,
                    'note': 'Model not loaded. Using random prediction for demo purposes.'
                })
            
            prediction, confidence = predict(filepath, model)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'prediction': prediction,
                'confidence': confidence
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    print("Starting server...")
    print(f"Using device: {DEVICE}")
    app.run(host='0.0.0.0', port=8000, debug=True)