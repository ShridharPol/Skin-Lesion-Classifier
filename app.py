import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model_final_MobNet.keras")

model = load_model()

# Define class names
class_names = [
    "Actinic keratoses",
    "Basal Cell Carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic Nevi",
    "Melanoma",
    "Vascular Lesions"
]

# Title and description
st.title("Skin Lesion Classifier")
st.write("Upload a skin lesion image and click **Predict** to identify the condition.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Add a predict button
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predict_button = st.button("🔍 Predict")

    if predict_button:
        with st.spinner("Analyzing..."):
            # Preprocess image
            image = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0) / 255.0

            # Predict
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

        # Output
        st.success(f"**Predicted Class:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # Optional: show top 3 predictions
        st.subheader("Top 3 Predictions:")
        top_k = predictions[0].argsort()[-3:][::-1]
        for i in top_k:
            st.write(f"- {class_names[i]}: {predictions[0][i]*100:.2f}%")
