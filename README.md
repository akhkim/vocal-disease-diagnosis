# Voice Disorder Diagnosis Through Patient Speech

This program aims to diagnose the patient's speech audio and diagnose whether and classify the type of vocal disorder they are experiencing.

The example script currently supports 4 types of disorders, which are Normal, Vox Senilis, Laryngocele, and COVID. 
However, as the file used to train the models is included, additional disorders can be simply added by adding their name to the categories array in both train and evaluate files, and adding the dataset to train it on in patient-vocal-dataset folder, following the folder name formatting used in the repository.

The models currently included in the repository supports regular speech as well as Electroglottography audio for all 4 types of disorders, allowing for more flexible and accurate analysis.
The feature-based model currently shows a higher accuracy of 90%, while the nn-based model still requires optimization and larger sample size.

## Requirements
- Python 3.9 or greater
- Windows 7 or greater

### GPU
GPU execution requires installing PyTorch following the official documentation:

- [PyTorch Get Started Page](https://pytorch.org/get-started/locally/)

## Installation
```
git clone https://github.com/akhkim/vocal-disease-diagnosis
cd vocal-disorder-diagnosis
pip install -r requirements.txt
```

## Command-line Usage
```
cd path\to\downloaded\repo\feature_approach or nn_approach
python evaluate.py path\to\audio
```
