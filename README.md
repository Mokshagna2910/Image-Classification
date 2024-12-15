# Image Classification with FastAPI and Streamlit

## Introduction
This project demonstrates an end-to-end image classification system powered by the MobileNetV2 model, FastAPI, and Streamlit. It provides a seamless integration of a robust backend and an intuitive frontend, allowing users to classify images effortlessly.

The application processes images uploaded via the Streamlit interface, performs classification using MobileNetV2 on the backend, and delivers predictions with confidence scores.

## Features
- **Image Classification:** Employs MobileNetV2, pre-trained on ImageNet, to classify images with high accuracy.
- **FastAPI Backend:** Handles classification requests efficiently and reliably.
- **Streamlit Frontend:** Offers a simple and interactive interface for image uploads and results visualization.
- **Pre-trained Model:** Utilizes MobileNetV2 for a lightweight and effective classification pipeline.
- **Local Deployment:** Operates locally without requiring external dependencies beyond Python libraries.

## Model Details
### MobileNetV2
- **Dataset:** Trained on ImageNet
- **Input Requirements:**
  - **Image Size:** 224x224 pixels
  - **Normalization Parameters:**
    - Mean: [0.485, 0.456, 0.406]
    - Std: [0.229, 0.224, 0.225]
- **Output:** Probabilities for 1,000 categories from the ImageNet dataset.

## App Architecture
This project features a modular architecture that integrates a FastAPI backend with a Streamlit frontend to streamline communication between user inputs and model inference.

### API Endpoints
1. **`/health`**
   - **Purpose:** Verifies that the backend is operational.
   - **Response:**
     ```json
     { "message": "Image Classifier API is up and running" }
     ```
2. **`/predict`**
   - **Purpose:** Accepts an image and returns the predicted class along with its confidence score.
   - **Request:**
     - POST `/predict`
     - Field: `"file"` (image in PNG or JPEG format).
   - **Response:**
     ```json
     {
       "predicted_class": "cat",
       "confidence": 0.92
     }
     ```

### Backend
- **Framework:** FastAPI
- **Model Serving:** PyTorch implementation of MobileNetV2.
- **Key Features:**
  - Model stored as `mobilenet_v2.pth`, pre-trained on ImageNet.
  - Endpoints for health checks and predictions.
  - Automatic GPU usage if available for faster inference.

### Frontend
- **Framework:** Streamlit
- **Key Features:**
  1. **File Uploader:** Enables users to upload images.
  2. **Request Handling:** Sends uploaded images to FastAPI for classification.
  3. **Results Display:** Shows predictions, confidence scores, and the uploaded image.

## Installation Instructions
### Prerequisites
- **Python 3.7 or higher**: Ensure it is installed.
- **pip**: The Python package manager to install dependencies.

### Setup Steps
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Required libraries include:
   - `torch`, `torchvision`: For model inference.
   - `fastapi`, `uvicorn`: To set up the backend API.
   - `streamlit`: For the frontend interface.
   - `Pillow`: For image preprocessing.
3. **Start the Backend:**
   ```bash
   uvicorn app:app --reload
   ```
   - Accessible at `http://localhost:8000`.
   - Use endpoints `/predict` and `/health` for testing.
4. **Launch the Frontend:**
   ```bash
   streamlit run app.py
   ```
   - Opens a web interface for uploading images and viewing predictions.

## Usage Instructions
### Workflow Overview
1. **Upload Image:**
   - Users upload an image through the Streamlit interface.
2. **Backend Processing:**
   - FastAPI resizes and normalizes the image before running it through MobileNetV2 for classification.
3. **Display Results:**
   - The frontend shows the predicted class, confidence score, and the uploaded image.

### API Usage
1. **POST `/predict`:**
   - **Request Format:**
     ```python
     files = {"file": uploaded_file.getvalue()}
     response = requests.post("http://localhost:8000/predict", files=files)
     ```
   - **Response:**
     ```json
     {
       "predicted_class": "dog",
       "confidence": 0.87
     }
     ```
2. **GET `/health`:**
   - **Request:**
     ```bash
     GET http://localhost:8000/health
     ```
   - **Response:**
     ```json
     { "message": "Image Classifier API is up and running" }
     ```

## Troubleshooting
1. **Backend Debugging:**
   - Enable detailed logging by setting the logging level to `DEBUG`.
   - Ensure the model file (`mobilenet_v2.pth`) exists and is accessible.
   - Confirm that the model is in inference mode by calling `model.eval()`.
2. **Frontend Issues:**
   - Ensure the Streamlit app starts without errors.
   - Verify that the backend URL is set to `http://localhost:8000`.
3. **General Problems:**
   - Use images in PNG or JPEG format that meet size and preprocessing requirements.
   - Verify all dependencies are installed using `pip install -r requirements.txt`.

## Key Benefits
- Lightweight and efficient classification with MobileNetV2.
- Simple and reliable deployment using FastAPI and Streamlit.
- User-friendly interface designed for both technical and non-technical users.

---

_This project is an example of how modern tools like FastAPI and Streamlit can be combined with advanced deep learning models to create powerful and intuitive applications._

