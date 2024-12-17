import os
import numpy as np
from tensorflow import keras
import cv2

class UploadService:
    def __init__(self):
        try:
            # Define path to the model
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'malnutrition_model.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            # Load the model
            self.model = keras.models.load_model(model_path)
            self.target_size = (224, 224)  # Example image size
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, file):
        try:
            # Read the image file
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")

            # Resize image
            img = cv2.resize(img, self.target_size)

            # Normalize image
            img = img.astype("float32") / 255.0

            # Convert image to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Expand dimensions for model input
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def predict(self, file):
        try:
            if self.model is None:
                return {"status": "Error", "confidence": 0.0, "message": "Model not loaded"}

            # Preprocess the uploaded image
            image = self.preprocess_image(file)
            if image is None:
                return {"status": "Error", "confidence": 0.0, "message": "Invalid image"}

            # Predict using the model
            predictions = self.model.predict(image, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])

            # Map prediction to label
            status = "Normal" if class_idx == 0 else "Malnutrition"
            return {"status": status, "confidence": confidence, "message": "Prediction successful"}
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {"status": "Error", "confidence": 0.0, "message": str(e)}
