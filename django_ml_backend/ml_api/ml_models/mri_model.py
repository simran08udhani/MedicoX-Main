import numpy as np
import tensorflow as tf
import os
import cv2
import base64
import io
# Set Matplotlib to non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MRIModel:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Load the MRI model from .h5 file
        """
        try:
            # Path to the .h5 file - adjust this to your actual file location
            model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'model.h5')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("MRI model loaded successfully")
            else:
                print(f"MRI model file not found at {model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading MRI model: {e}")
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
    #         img = cv2.resize(img, (256, 256))
            
    #         # Convert to grayscale (often used for MRI)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    #         # Normalize pixel values
    #         img = img / 255.0
            
    #         # Expand dimensions for batch and channel
    #         img = np.expand_dims(img, axis=0)
    #         img = np.expand_dims(img, axis=-1)
            
    #         return img
    #     except Exception as e:
    #         print(f"Error preprocessing MRI image: {e}")
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
        Preprocess the image for the MRI model.
        Following the same preprocessing steps used during training.
        """
        try:
            # Load image in grayscale as done in training
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
                
            # Resize to expected dimensions (128, 128) as used during training
            img_resized = cv2.resize(img, (128, 128))
            
            # Normalize pixel values to [0,1]
            img_normalized = img_resized / 255.0
            
            # Convert grayscale to 3 channels (model input shape is (IMG_HEIGHT, IMG_WIDTH, 3))
            # Repeating the same grayscale image across 3 channels
            img_rgb = np.stack([img_normalized] * 3, axis=-1)
            
            # Add batch dimension (H, W, 3) -> (1, H, W, 3)
            img_input = np.expand_dims(img_rgb, axis=0)
            
            return img_input
        except Exception as e:
            print(f"Error preprocessing MRI image: {e}")
            return None

    def predict(self, image_path):
        """
        Run prediction on the MRI image.
        Returns class probabilities, predicted class, and additional metrics.
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Define class names
            class_names = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]
            
            # Preprocess the image
            preprocessed_img = self.preprocess_image(image_path)
            if preprocessed_img is None:
                return {"error": "Failed to preprocess image"}
            
            # Get predictions from model
            predictions = self.model.predict(preprocessed_img)[0]
            predicted_class_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_class_idx]
            confidence = float(predictions[predicted_class_idx])
            
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
            
            # Generate probability chart for frontend
            plt.figure(figsize=(8, 4))
            plt.bar(class_names, predictions)
            plt.xlabel('Alzheimer\'s Categories')
            plt.ylabel('Probability')
            plt.title('Alzheimer\'s Classification Probabilities')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # # Convert plot to base64 string
            # buf = io.BytesIO()
            # plt.savefig(buf, format='png')
            # buf.seek(0)
            # plot_str = base64.b64encode(buf.read()).decode('utf-8')
            # plt.close()

            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close('all')  # Close all figures to prevent memory leaks
            
            # Format data for React frontend
            prediction_data = [
                {"category": class_name, "probability": float(prob)} 
                for class_name, prob in zip(class_names, predictions)
            ]
            
            # Provide appropriate medical guidance based on predicted class
            guidance = {
                "Mild_Demented": "Early signs of Alzheimer's detected. Regular monitoring and lifestyle modifications recommended.",
                "Moderate_Demented": "Moderate Alzheimer's indicators present. Medical intervention and care planning advised.",
                "Non_Demented": "No significant signs of Alzheimer's detected in this scan.",
                "Very_Mild_Demented": "Very mild indicators of Alzheimer's detected. Follow-up evaluation recommended."
            }
            
            # result = {
            #     "prediction_scores": predictions.tolist(),
            #     "predicted_class": predicted_class,
            #     "confidence": confidence,
            #     "image_data": img_str,
            #     "probability_chart": plot_str,
            #     "medical_guidance": guidance[predicted_class],
            #     "recommendation": "Please consult with a neurologist to discuss these findings. This AI analysis is meant to assist, not replace professional medical advice.",
            #     "next_steps": [
            #         "Schedule a consultation with a neurologist",
            #         "Discuss medication options if applicable",
            #         "Consider lifestyle modifications for brain health",
            #         "Plan for regular follow-up evaluations"
            #     ]
            # }

            result = {
                #"success": True,
                #"prediction_scores": predictions.tolist(),
                "prediction_data": prediction_data,  # Formatted for easy React chart libraries
                "predicted_class": predicted_class,
                "confidence": confidence,
                #"image_data": f"data:image/jpeg;base64,{img_str}" if img_str else None,
                "probability_chart": f"data:image/png;base64,{plot_str}",
                "medical_guidance": guidance[predicted_class],
                #"recommendation": "Please consult with a neurologist to discuss these findings. This AI analysis is meant to assist, not replace professional medical advice.",
                "next_steps": [
                    "Schedule a consultation with a neurologist",
                    "Discuss medication options if applicable",
                    #"Consider lifestyle modifications for brain health",
                    #"Plan for regular follow-up evaluations"
                ]
            }
            
            return result
        except Exception as e:
            return {"error": str(e)}
