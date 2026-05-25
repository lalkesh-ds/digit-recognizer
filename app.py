import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import joblib

# ----------------------------
# Load trained Random Forest model
# ----------------------------
model = joblib.load("digit_rf_model.pkl")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below and click Predict")

# ----------------------------
# Canvas
# ----------------------------
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Convert image
        img = Image.fromarray(
            (canvas_result.image_data[:, :, 0]).astype("uint8")
        )

        # Resize to 28x28
        img = img.resize((28, 28))

        # Convert to grayscale
        img = img.convert("L")

        # Invert colors
        img_array = 255 - np.array(img)

        # Normalize
        img_array = img_array / 255.0

        # Flatten
        img_flat = img_array.reshape(1, -1)

        # Predict
        prediction = model.predict(img_flat)[0]
        probabilities = model.predict_proba(img_flat)[0]

        st.success(f"Predicted Digit: **{prediction}**")

        st.subheader("Confidence Scores")
        st.bar_chart(probabilities)

    else:
        st.warning("Please draw a digit first.")