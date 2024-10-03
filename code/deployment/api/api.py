from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
import io
from code.models.model import predict_number

app = FastAPI()


# Function to preprocess the image
def preprocess_image(image_bytes):
    # Open the image and convert it to grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert('L')  # 'L' mode is for grayscale
    img = np.array(img)

    # Resize the image to 28x28 pixels
    img_resized = cv2.resize(img, (28, 28))

    # Normalize the image (scale pixel values to [0, 1])
    img_normalized = img_resized / 255.0

    # Reshape the image to match model input (1, 28, 28)
    img_reshaped = np.expand_dims(img_normalized, axis=0)

    return img_reshaped

# Endpoint to predict the number from an image
@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        # Read image bytes from the uploaded file
        image_bytes = await file.read()

        # Preprocess the image
        preprocessed_image = preprocess_image(image_bytes)

        # Use the model to make a prediction
        prediction = predict_number(preprocessed_image)
        # Return the predicted digit in a JSON response
        return JSONResponse(content={"predicted_digit": int(prediction)})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Health check route
@app.get("/")
def read_root():
    return {"message": "API is working!"}
