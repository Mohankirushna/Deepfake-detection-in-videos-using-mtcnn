# 🎥 Deepfake Detection Using Video Frame Analysis

This project demonstrates a deepfake video detection pipeline leveraging face detection, frame extraction, and a Convolutional Neural Network (CNN) model. The model classifies videos as **FAKE** or **REAL** based on analyzing detected faces from video frames.

---

## 📋 Project Overview

- Uses the **Deepfake Detection Challenge** dataset sample videos and metadata.
- Extracts faces from video frames using **MTCNN** (Multi-task Cascaded Convolutional Networks).
- Preprocesses face images and trains a CNN model for binary classification (fake vs real).
- Provides functions to predict whether new videos are fake or real by analyzing faces across frames.
- Visualizes dataset distribution and sample video frames.
- Implements a custom `Sequence` data generator for efficient batch loading during training.

---

## ✨ Features

- **Face detection with MTCNN** 🕵️‍♂️ on video frames.
- **CNN architecture** 🧠 for binary classification of faces.
- **Batch data generator** 🔄 for handling videos with multiple frames.
- Training/validation split with `train_test_split`.
- Model saving/loading for reuse 💾.
- Prediction function for testing new videos 🎞️.
- Visualization of fake vs real video distribution 📊.
- Sample frame display for both fake and real videos 🖼️.

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- MTCNN
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- IPython (for video display in notebooks)

Install dependencies via pip:

```bash
pip install tensorflow mtcnn opencv-python pandas numpy matplotlib seaborn scikit-learn ipython
```
## 📂 Dataset
This project uses the Deepfake Detection Challenge dataset sample videos and metadata, available at Kaggle:
../input/deepfake-detection-challenge/train_sample_videos/

## 🛠️ Usage
1. Load and explore metadata
Load the metadata JSON and visualize distribution of real vs fake videos.

2. Visualize sample frames
Display sample frames from both fake and real videos for manual inspection.

3. Prepare data generators
Use VideoFrameGenerator class to load and preprocess batches of face images extracted from videos.

4. Build and train CNN model
Train a CNN on extracted faces to classify fake vs real.

5. Evaluate model
Evaluate the model on a validation set and save the trained weights.

6. Predict on new videos
Detect faces in new videos and classify them using the trained model.

## 💡 Code Highlights
Face extraction and preprocessing using OpenCV and MTCNN.

Data generator implemented via Keras Sequence for batch processing.

CNN model with multiple Conv2D and MaxPooling2D layers, followed by dense layers.

Prediction function that averages frame-level predictions to classify a full video.

## 🔍 Example Prediction Usage
video_path = '/kaggle/input/deepfake-detection-challenge/test_videos/sample_video.mp4'
prediction_score = predict_video(video_path)

if prediction_score > 0.5:
    print("Predicted: FAKE 🔴")
else:
    print("Predicted: REAL 🟢")

## 📝 Notes
Videos may contain multiple faces; all detected faces are analyzed.

Batch size and target image size are configurable in the data generator.

Training currently set for 2 epochs for demonstration; increase for better results.

Model architecture and hyperparameters can be tuned for improved accuracy.

## 🙏 Acknowledgements
Deepfake Detection Challenge dataset on Kaggle.

Face detection powered by MTCNN.

CNN modeling using TensorFlow and Keras.

## 📄 License
This project is open source under the MIT License.

