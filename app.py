import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load CNN model
model = load_model("digit_cnn_model.h5")

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

        # Convert image to grayscale
        img = Image.fromarray(
            canvas_result.image_data[:, :, 0].astype("uint8")
        )

        img = img.resize((28, 28))
        img = img.convert("L")

        # Convert to numpy
        img = np.array(img)

        # Invert colors (MNIST format)
        img = 255 - img

        # Normalize
        img = img.astype("float32") / 255.0

        # Reshape for CNN
        img = img.reshape(1, 28, 28, 1)

        # Prediction
        probabilities = model.predict(img, verbose=0)

        prediction = np.argmax(probabilities)
        confidence = np.max(probabilities)

        st.success(f"Predicted Digit: {prediction}")
        st.write(f"Confidence: {confidence:.2%}")

        chart_data = {
            str(i): float(probabilities[0][i])
            for i in range(10)
        }

        st.bar_chart(chart_data)

    else:
        st.warning("Please draw a digit first.")