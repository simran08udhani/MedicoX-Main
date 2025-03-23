import numpy as np
import tensorflow as tf
import os
import cv2
import base64

class XRayModel:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Load the X-ray model from .h5 file
        """
        try:
            # Path to the .h5 file - adjust this to your actual file location
            model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'pneumonia_classification_model.h5')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("X-ray model loaded successfully")
            else:
                print(f"X-ray model file not found at {model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading X-ray model: {e}")
            self.model = None
    
    # def preprocess_image(self, image_path):
    #     """
    #     Preprocess the image for the model.
    #     """
    #     try:
    #         # Load and preprocess the image - adjust according to your model requirements
    #         img = cv2.imread(image_path)
    #         if img is None:
    #             return None
                
    #         # Resize to expected dimensions
    #         img = cv2.resize(img, (299, 299))
            
    #         # Convert to grayscale (common for X-ray)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    #         # Apply CLAHE for better contrast (often used for X-rays)
    #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #         img = clahe.apply(img)
            
    #         # Normalize pixel values
    #         img = img / 255.0
            
    #         # Expand dimensions for batch and channel
    #         img = np.expand_dims(img, axis=0)
    #         img = np.expand_dims(img, axis=-1)
            
    #         return img
    #     except Exception as e:
    #         print(f"Error preprocessing X-ray image: {e}")
    #         return None
    
    # def predict(self, image_path):
    #     """
    #     Run prediction on the image.
    #     """
    #     if self.model is None:
    #         return {"error": "Model not loaded"}
        
    #     try:
    #         # Preprocess the image
    #         preprocessed_img = self.preprocess_image(image_path)
    #         if preprocessed_img is None:
    #             return {"error": "Failed to preprocess image"}
            
    #         # Get predictions from model
    #         predictions = self.model.predict(preprocessed_img)
            
    #         # Process and return results according to your model's output format
    #         # Adjust this section based on your model's specific output structure
    #         result = {
    #             "prediction_scores": predictions.tolist(),
    #             "predicted_class": int(np.argmax(predictions[0])),
    #             "confidence": float(np.max(predictions[0])),
    #         }
    #         return result
    #     except Exception as e:
    #         return {"error": str(e)}

    def preprocess_image(self, image_path):
        """
        Preprocess the image for the X-ray model.
        Following the same preprocessing steps used during training.
        """
        try:
            # Load image in grayscale as done in training
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
                
            # Resize to expected dimensions (256, 256) as used during training
            img_resized = cv2.resize(img, (256, 256))
            
            # Normalize pixel values to [0,1]
            img_normalized = img_resized / 255.0
            
            # Expand dimensions for batch and channel
            # First add channel dimension (H, W) -> (H, W, 1)
            img_input = np.expand_dims(img_normalized, axis=-1)
            # Then add batch dimension (H, W, 1) -> (1, H, W, 1)
            img_input = np.expand_dims(img_input, axis=0)
            
            return img_input
        except Exception as e:
            print(f"Error preprocessing X-ray image: {e}")
            return None

    def predict(self, image_path):
        """
        Run prediction on the X-ray image.
        Returns probability, prediction class, and additional metrics.
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess the image
            preprocessed_img = self.preprocess_image(image_path)
            if preprocessed_img is None:
                return {"error": "Failed to preprocess image"}
            
            # Get prediction from model
            probability = float(self.model.predict(preprocessed_img)[0][0])
            
            # Use the threshold of 0.65 as specified in your code
            prediction = "Pneumonia" if probability > 0.65 else "Normal"
            
            # Create visualization for frontend
            original_img = cv2.imread(image_path)
            if original_img is not None:
                # Resize for display purposes
                display_img = cv2.resize(original_img, (400, 400))
                # Convert to base64 for frontend display
                _, buffer = cv2.imencode('.jpg', display_img)
                img_str = base64.b64encode(buffer).decode('utf-8')
            else:
                img_str = None
            
            # Calculate confidence percentage for display
            confidence_percentage = probability * 100 if prediction == "Pneumonia" else (1 - probability) * 100
            
            # Add medical context based on probability ranges
            severity_level = "High" if probability > 0.85 else "Moderate" if probability > 0.75 else "Low"
            
            result = {
                "raw_probability": probability,
                "prediction": prediction,
                "confidence_percentage": f"{confidence_percentage:.2f}%",
                "image_data": img_str,
                "threshold_used": 0.65,
                "severity_level": severity_level if prediction == "Pneumonia" else "N/A",
                "recommendation": "Consult with healthcare provider" if prediction == "Pneumonia" else "Normal findings, routine follow-up recommended"
            }
            
            return result
        except Exception as e:
            return {"error": str(e)}