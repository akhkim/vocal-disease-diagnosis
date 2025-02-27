import sys
import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

# Define the categories
categories = ['Normal', 'Vox_senilis', 'Laryngocele']

# Function to extract Mel Spectrogram from an audio file
def extract_features(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono by averaging channels if multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
    return mel_spectrogram

# Define the CNN model
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Adaptive pooling to reduce to a fixed size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(-1, 64)       # Flatten to [batch_size, 64]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(script_dir, 'voice_classifier_model.pth')
model = AudioCNN(num_classes=len(categories))
model.load_state_dict(torch.load(model_filename, weights_only=True))
model.eval()

# Function to classify a new audio file
def classify_audio(file_path, model):
    mel_spectrogram = extract_features(file_path)
    mel_spectrogram = mel_spectrogram.unsqueeze(0)
    with torch.no_grad():
        outputs = model(mel_spectrogram)
        _, predicted = torch.max(outputs, 1)
    return categories[predicted.item()]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python evaluate.py path/to/audio.wav')
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path) or not file_path.endswith('.wav'):
        print('Error: File does not exist or is not a WAV file.')
        sys.exit(1)
    
    prediction = classify_audio(file_path, model)
    print(f'The predicted category for the audio is: {prediction}')