# Music Genre Classification using Deep Learning

A robust machine learning pipeline designed to automatically categorize audio tracks into specific musical genres. This project leverages Digital Signal Processing (DSP) and Deep Neural Networks (DNN) to achieve high-precision classification.

## 📋 Executive Summary

Automating music classification is a fundamental task in music information retrieval (MIR). This repository provides a complete end-to-end workflow—from raw signal processing to model inference—aimed at identifying patterns in acoustic characteristics across various genres.

## 🛠️ Technical Stack

* **Language:** Python 3.x
* **Audio Processing:** [Librosa](https://librosa.org/) (Feature extraction & STFT)
* **Deep Learning Framework:** [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
* **Data Analysis:** NumPy, Pandas, Scikit-learn
* **Visualization:** Matplotlib, Seaborn

## 📊 Methodology & Feature Engineering

The system utilizes specialized acoustic features to capture the temporal and spectral characteristics of sound. The following descriptors are extracted to form the input feature vector:

* **MFCCs (Mel-frequency cepstral coefficients):** Capturing the "texture" of the sound.
* **Spectral Centroid:** Indicating the "brightness" of the audio.
* **Chroma Features:** Representing the harmonic and melodic content.
* **Temporal Features:** Including Root Mean Square (RMS) energy and Zero Crossing Rate (ZCR).

## 🧠 Model Architecture

The core of the project is a deep neural network optimized for multi-class classification. The architecture is designed to minimize loss and prevent overfitting through techniques such as Dropout and Batch Normalization.

<img width="558" height="638" alt="image" src="https://github.com/user-attachments/assets/8308850e-085f-40c7-b4bb-6c0232b09fbe" />


## 🚀 Installation & Usage

### Prerequisites
Ensure you have a Python environment (3.8+) installed.

### Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Karvsky/Music-type-classification.git](https://github.com/Karvsky/Music-type-classification.git)
   cd Music-type-classification
