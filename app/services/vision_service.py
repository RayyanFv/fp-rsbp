import numpy as np
import cv2
import os
import base64
from tensorflow import keras

class VisionService:
    def __init__(self):
        try:
            # Get absolute path to the model file
            base_dir = os.path.abspath(os.path.dirname(__file__))  # Current file's directory
            model_path = os.path.join(base_dir, '..', 'models', 'malnutrition_model.h5')  # Go up one level to /models/

            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"Model file not found at: {model_path}")
                self.model = None
                self.model_loaded = False
                return

            # Load model
            self.model = keras.models.load_model(model_path)
            self.model_loaded = True
            print(f"Model loaded successfully from: {model_path}")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            self.model_loaded = False

        # Define image size
        self.target_size = (224, 224)

    def preprocess_image(self, image_data):
        try:
            # Handle base64 string input
            if isinstance(image_data, str):
                # Remove header if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                # Decode base64
                image_bytes = base64.b64decode(image_data.encode('utf-8'))
            else:
                # Handle file-like object
                image_bytes = image_data.read()

            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image")

            # Resize
            img = cv2.resize(img, self.target_size)

            # Convert to RGB (karena OpenCV menggunakan BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Scale pixel values
            img = img.astype("float32") / 255.0

            # Expand dimensions untuk batch
            img = np.expand_dims(img, axis=0)

            return img

        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, image_data):
        try:
            # Check if model is loaded
            if not self.model_loaded:
                return {
                    "status": "Error",
                    "confidence": 0.0,
                    "message": "Model not loaded. Please check model file existence."
                }

            # Preprocess image
            processed_image = self.preprocess_image(image_data)

            if processed_image is None:
                raise ValueError("Failed to preprocess image")

            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)

            # Get the highest probability class
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])

            # Map class index to label
            status = "Normal" if class_idx == 0 else "Malnutrisi"

            return {
                "status": status,
                "confidence": confidence,
                "message": "Prediction successful"
            }

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {
                "status": "Error",
                "confidence": 0.0,
                "message": str(e)
            }
