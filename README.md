Image Classification with FastAPI and Streamlit
This project demonstrates a simple but powerful image classification pipeline. It uses FastAPI for serving a pre-trained MobileNetV2 model and Streamlit for the frontend interface. This solution is designed to be accessible, easy to use, and flexible for anyone interested in integrating image classification into their applications.

Project Overview
The purpose of this project is to create an easy-to-use platform where users can upload images, get predictions based on a pre-trained MobileNetV2 model, and view the results through a frontend built with Streamlit. The backend uses FastAPI, which provides a fast, efficient, and lightweight server to handle classification requests.

By uploading an image, the backend (FastAPI) performs the classification using the MobileNetV2 model, processes the request, and sends back a response containing the class and the associated confidence score. The frontend then displays this result to the user.

Features
Image Classification: The project classifies uploaded images into pre-trained classes using MobileNetV2.
FastAPI Backend: FastAPI serves as the backend to handle requests and responses for image classification.
Streamlit Frontend: A clean and simple UI for easy image uploads and displaying predictions.
Pre-trained Model: Uses the MobileNetV2 model from PyTorch’s torchvision library, which is trained on the ImageNet dataset.
Local Deployment: This project runs entirely locally, with no external dependencies required except for the provided libraries.

Installation
To get started, follow the installation steps outlined below.

Prerequisites
Ensure that you have the following installed on your system:

Python 3.7 or higher.
pip (Python package manager).
Step 1: Clone the Repository
Clone the repository to your local machine:

git clone <repository_url>
cd <project_directory>

Step 2: Install Dependencies
To run the backend and frontend applications, you'll need to install the necessary dependencies. Run the following command:

pip install -r requirements.txt

The requirements.txt file should contain the necessary libraries:

torch, torchvision (for MobileNetV2 model)
fastapi (for the backend API)
uvicorn (for running FastAPI server)
streamlit (for the frontend interface)
Pillow (for image processing)

Step 3: Start the FastAPI Backend
The backend is responsible for accepting image files, processing them with the MobileNetV2 model, and returning classification results. To start the FastAPI server, use the following command:

uvicorn app:app --reload

The server will start at http://localhost:8000. The backend provides two main endpoints:

POST /predict: Accepts an image file, performs classification, and returns the predicted class and confidence score.
GET /health: A health check endpoint to verify if the API is up and running.

Step 4: Run the Streamlit Frontend
To run the frontend, which allows users to upload images and view predictions, run the following Streamlit command:

streamlit run app.py

The Streamlit app will launch in your browser, where you can upload images for classification.

How the Project Works
Image Upload and Classification Flow
The process begins when the user uploads an image through the Streamlit frontend. The uploaded image is sent to the FastAPI backend for processing. Here’s the flow:

Upload Image: The user selects an image file from their local machine and uploads it through the Streamlit interface.
Send Image to Backend: The Streamlit frontend sends the uploaded image to the FastAPI backend through a POST request.
Image Preprocessing: FastAPI processes the image using the necessary preprocessing steps such as resizing, normalization, and conversion into a tensor suitable for the MobileNetV2 model.
Prediction: FastAPI then uses the MobileNetV2 model to predict the class of the image.
Result: The prediction, along with the confidence score, is returned to Streamlit.
Display Result: The frontend displays the predicted class and confidence to the user.
Backend - FastAPI
The backend is responsible for classifying images. The key components of the backend are as follows:

MobileNetV2 Model: A pre-trained MobileNetV2 model from PyTorch is used for image classification. The model is saved as mobilenet_v2.pth in the local directory. When the server starts, it checks whether the model is available and loads it into memory.
Endpoints:
POST /predict: This endpoint handles image classification requests. The image is read, preprocessed, and passed through the MobileNetV2 model for prediction. The result is returned as JSON.
GET /health: A simple health check to verify if the API is running.
CUDA: The backend can utilize GPU acceleration if available by moving the model and image tensor to the GPU.
Frontend - Streamlit
The Streamlit app provides a simple and intuitive interface for users. The steps involved are:

Upload Image: The user uploads an image using the Streamlit file uploader.
Send Image to Backend: Once the image is uploaded, it is sent to the backend using the requests library. The backend responds with the predicted class and confidence score.
Display Result: The frontend displays the predicted class and confidence, and also shows the uploaded image.
API Endpoints
POST /predict
This endpoint accepts an image file and returns the classification result. Here is an example of how to send a request using the Streamlit frontend:

files = {"file": uploaded_file.getvalue()}
response = requests.post("http://localhost:8000/predict", files=files)

Request Format
The image file should be sent as form data under the file field.

Response Format
The response is a JSON object with the following fields:

predicted_class: The predicted class of the image.
confidence: The confidence score of the prediction.
Example Response:
{
  "predicted_class": "cat",
  "confidence": 0.92
}
GET /health
This endpoint provides a simple health check for the API. It returns a message indicating that the server is running.

Example Response:
{
  "message": "Image Classifier API is up and running"
}
Model
This project uses the MobileNetV2 model, which is pre-trained on the ImageNet dataset. MobileNetV2 is a lightweight model that is highly efficient for mobile and embedded applications.

How the Model is Used
Input: The model accepts images of size 224x224 with three color channels (RGB). The input images are normalized based on the ImageNet mean and standard deviation.
Output: The model returns a class prediction and a confidence score. The class is one of the 1000 classes from the ImageNet dataset.
The model is stored as mobilenet_v2.pth. When the FastAPI server starts, it checks if this model file exists and loads it into memory.
