import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

destination_folder = 'patient-vocal-dataset'

categories = ['Normal', 'Vox_senilis', 'Laryngocele']

# Function to extract Mel Spectrogram from an audio file
def extract_features(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        n_mels=64
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)
    return mel_spectrogram

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        mel_spectrogram = extract_features(file_path)
        return mel_spectrogram, label

# Collect data from the folder structure
file_paths = []
labels = []
for category in categories:
    folder_path = os.path.join(destination_folder, category)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            file_paths.append(file_path)
            labels.append(categories.index(category))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

def pad_collate(batch):
    # Find the longest spectrogram in the batch (along the time dimension)
    max_len = max([spec.shape[2] for spec, _ in batch])
    
    # Pad each spectrogram to match the maximum length
    padded_specs = []
    labels = []
    for spec, label in batch:
        pad_amount = max_len - spec.shape[2]
        padded_spec = torch.nn.functional.pad(spec, (0, pad_amount, 0, 0, 0, 0), mode='constant', value=0)
        padded_specs.append(padded_spec)
        labels.append(label)
    
    # Stack the padded spectrograms and convert labels to a tensor
    padded_specs = torch.stack(padded_specs)
    labels = torch.tensor(labels)
    return padded_specs, labels

# Create datasets and dataloaders
train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

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

model = AudioCNN(num_classes=len(categories))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

# Evaluate the model
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())

# Print performance metrics
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(confusion_matrix(y_true, y_pred))
print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=categories))

script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(script_dir, 'voice_classifier_model.pth')
torch.save(model.state_dict(), model_filename)
print(f'\nModel saved as {model_filename}')