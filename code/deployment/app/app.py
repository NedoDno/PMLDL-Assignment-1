import streamlit as st
import requests
from PIL import Image
import io

# Title of the web app
st.title("Digit Recognition App")

# Subtitle
st.write("Upload an image of a digit, and the model will predict which digit it is!")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# When an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    # Add a 'Predict' button
    if st.button('Predict'):
        # Send the image to FastAPI backend
        with st.spinner('Sending the image for prediction...'):
            files = {'file': uploaded_file.getvalue()}
            response = requests.post("http://fastapi-backend:8000/predict/", files=files)

            if response.status_code == 200:
                result = response.json()
                predicted_digit = result.get("predicted_digit")
                st.success(f"Predicted Digit: {predicted_digit}")
            else:
                st.error(f"Error: {response.text}")

