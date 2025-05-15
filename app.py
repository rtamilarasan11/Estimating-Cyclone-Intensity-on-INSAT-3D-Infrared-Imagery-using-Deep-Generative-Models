import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Streamlit page config
st.set_page_config(page_title="Cyclone Intensity Estimator", layout="wide")
st.title("🌪️ Cyclone Intensity Estimator")
st.markdown("---")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

model = load_model()

# Image preprocessing function
def load_and_prep_image(uploaded_file, img_shape=512):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((img_shape, img_shape))
    img_array = np.array(image) / 255.0  # normalize
    return img_array

# Sidebar - ground truth input
#with st.sidebar:
    #st.header("Ground Truth (Optional)")
    #actual_intensity = st.number_input("Enter actual intensity (knots)", min_value=0.0, max_value=300.0, value=120.0)
    #st.markdown("---")
    #st.info("Upload an IR satellite image of a cyclone to estimate intensity.")

# File uploader
uploaded_file = st.file_uploader("Upload an Infrared Cyclone Image (JPG/JPEG)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = load_and_prep_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        image_tensor = tf.expand_dims(image, axis=0)  # add batch dimension
        prediction = model.predict(image_tensor)
        intensity_pred = float(prediction[0][0])

        # Display results
        st.subheader("📊 Prediction Results")
        st.write(f"**Predicted Intensity:** `{intensity_pred:.2f} knots`")
        #st.write(f"**Actual Intensity (optional):** `{actual_intensity:.2f} knots`")
        
        st.success("✅ Prediction completed successfully!")

    except Exception as e:
        st.error("⚠️ Error processing image.")
        st.exception(e)
else:
    st.info("ℹ️ Please upload a cyclone satellite image to begin.")
