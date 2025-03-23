import streamlit as st
import tensorflow.lite as tflite
import numpy as np
from PIL import Image

# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="efficientnetb0_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Image Preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize for model
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Run Inference
def predict(image):
    img_array = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Streamlit UI
st.title("🧠 MRI-Based Alzheimer’s Detection")
st.write("Upload an MRI scan to classify it using an optimized EfficientNetB0 model.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    if st.button("Analyze MRI"):
        with st.spinner("Processing..."):
            prediction = predict(image)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

        st.success(f"Prediction: Class {predicted_class} (Confidence: {confidence:.2f})")
