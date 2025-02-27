import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

destination_folder = 'health_detection/patient-vocal-dataset'

categories = ['Normal', 'Vox_senilis', 'Laryngozele']

def extract_features(file_path):
    """
    Extracts audio features from a WAV file using librosa.
    
    Args:
        file_path (str): Path to the WAV file.
    
    Returns:
        np.ndarray: Feature vector containing means and stds of audio features.
    """
    # Load audio file, resample to 22.05 kHz for consistency
    y, sr = librosa.load(file_path, sr=22050)
    
    # Extract MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Extract spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)
    
    # Extract spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    
    # Extract zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # Combine all features into a single vector
    features = np.concatenate([
        mfccs_mean, mfccs_std,
        [centroid_mean, centroid_std, rolloff_mean, rolloff_std, zcr_mean, zcr_std]
    ])
    return features

def train():
    X = []  # Features
    y = []  # Labels
    for category in categories:
        folder_path = os.path.join(destination_folder, category)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                features = extract_features(file_path)
                X.append(features)
                y.append(categories.index(category))

    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=categories))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(script_dir, 'voice_classifier_model.joblib')
    joblib.dump(clf, model_filename)
    print(f'\nModel saved as {model_filename}')

if __name__ == '__main__':
    train()