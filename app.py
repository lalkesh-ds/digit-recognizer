import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import joblib
import pandas as pd

# Load model
model = joblib.load("digit_rf_model.pkl")

# Page config
st.set_page_config(page_title="Digit Recognizer", page_icon="✍️")

st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) and click Predict")

# Canvas
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

if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Convert image
        img = Image.fromarray(
            canvas_result.image_data[:, :, 0].astype("uint8")
        )

        # Resize to 28x28
        img = img.resize((28, 28))

        # Grayscale
        img = img.convert("L")

        # Invert colors
        img_array = 255 - np.array(img)

        # Normalize + flatten
        img_flat = img_array.reshape(1, -1) / 255.0

        # Convert to DataFrame
        img_df = pd.DataFrame(img_flat)

        # Predict
        prediction = model.predict(img_df)[0]
        probabilities = model.predict_proba(img_df)[0]

        st.success(f"Predicted Digit: {prediction}")
        st.bar_chart(probabilities)

    else:
        st.warning("Please draw a digit first.")