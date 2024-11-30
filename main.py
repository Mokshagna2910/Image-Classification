import torch
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import imagenet

# Instantiate FastAPI app
app = FastAPI()

# Path where the model will be saved
MODEL_PATH = "mobilenet_v2.pth"

# Function to download and save the model if not already downloaded
def download_and_save_model():
    if not os.path.exists(MODEL_PATH):  # Check if the model already exists
        print("Downloading the pre-trained MobileNetV2 model...")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        torch.save(model.state_dict(), MODEL_PATH)  # Save model weights
        print("Model downloaded and saved as mobilenet_v2.pth")
    else:
        print("Model already exists, loading the saved model...")

# Load the model
def load_model():
    model = models.mobilenet_v2()  # Initialize the model architecture
    model.load_state_dict(torch.load(MODEL_PATH))  # Load the saved model weights
    model.eval()  # Set model to evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()  # Move model to GPU if available
    return model

# Download and save the model (if not already saved)
download_and_save_model()

# Load the pre-trained model
model = load_model()

# Image preprocessing function for MobileNetV2
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "Image Classifier API is up and running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint that takes an image file and returns the predicted class.

    Args:
        file (UploadFile): Image file to classify.

    Returns:
        Dict: Predicted class and confidence score.
    """
    try:
        # Read and preprocess the uploaded image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Move input tensor to GPU if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_idx = torch.topk(probabilities, 1)

        # Get the predicted class label and confidence score
        predicted_class = imagenet.image[top_idx.item()]  # You can map this index to class labels if needed
        confidence = top_prob.item()
        
        return {"predicted_class": predicted_class, 
                "confidence": confidence}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
