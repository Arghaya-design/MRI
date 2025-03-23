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

# Get model's expected input size
input_shape = input_details[0]['shape']  # Example: [1, 126, 126, 3]
img_height, img_width = input_shape[1], input_shape[2]  # Extract expected size

# Image Preprocessing
def preprocess_image(image):
    image = image.resize((img_width, img_height))  # Resize to match model input
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize (0-1)
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
            try:
                prediction = predict(image)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                st.success(f"Prediction: Class {predicted_class} (Confidence: {confidence:.2f})")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
