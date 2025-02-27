import sys
import os
import librosa
import numpy as np
import joblib
from train import extract_features

categories = ['Normal', 'Vox senilis', 'Laryngozele']

# Load the pre-trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(script_dir, 'voice_classifier_model.joblib')
clf = joblib.load(model_filename)

# Function to classify a new audio file
def classify_audio(file_path, clf):
    """
    Classifies a new audio file using the pre-trained classifier.
    
    Args:
        file_path (str): Path to the new WAV file.
        clf: Pre-trained classifier model.
    
    Returns:
        str: Predicted category ('normal', 'vox_senilis', or 'laryngocele').
    """
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    prediction = clf.predict(features)[0]
    return categories[prediction]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python evaluate.py path/to/audio.wav')
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path) or not file_path.endswith('.wav'):
        print('Error: File does not exist or is not a WAV file.')
        sys.exit(1)
    
    prediction = classify_audio(file_path, clf)
    print(f'The predicted category for the audio is: {prediction}')