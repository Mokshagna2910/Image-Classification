import streamlit as st
import requests
from PIL import Image
import io

# Streamlit interface
st.title("Image Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded, show it and process it
if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Displaying a loading message
    st.write("Classifying...")

    # Send the image to FastAPI backend for prediction
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/predict", files=files)

    if response.status_code == 200:
        result = response.json()
        predicted_class = result.get("predicted_class")
        confidence = result.get("confidence")

        # Display the result
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
    else:
        st.write("Error: Could not get a response from the model. Please try again.")
