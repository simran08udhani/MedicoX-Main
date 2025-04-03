import numpy as np
import tensorflow as tf
import os
import cv2
import base64
import matplotlib.pyplot as plt
import io
from sklearn.metrics import confusion_matrix
import seaborn as sns

class CTScanModel:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Load the CT scan model from .h5 file
        """
        try:
            # Path to the .h5 file - adjust this to your actual file location
            model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'vgg19_lung_cancer_model.h5')  
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("CT scan model loaded successfully")
            else:
                print(f"CT scan model file not found at {model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading CT scan model: {e}")
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
    #         img = cv2.resize(img, (224, 224))
            
    #         # Convert to RGB if needed
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    #         # Normalize pixel values
    #         img = img / 255.0
            
    #         # Expand dimensions for batch
    #         img = np.expand_dims(img, axis=0)
            
    #         return img
    #     except Exception as e:
    #         print(f"Error preprocessing CT scan image: {e}")
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
        Preprocess the CT scan image for the model.
        Following the same preprocessing steps used during training.
        """
        try:
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Resize to expected dimensions (224, 224) as used during training
            img_resized = cv2.resize(img, (224, 224))
            
            # Convert BGR to RGB (model was trained on RGB)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0,1] as done in training
            img_normalized = img_rgb / 255.0
            
            # Add batch dimension (H, W, 3) -> (1, H, W, 3)
            img_input = np.expand_dims(img_normalized, axis=0)
            
            return img_input
        except Exception as e:
            print(f"Error preprocessing CT scan image: {e}")
            return None

    def predict(self, image_path):
        """
        Run prediction on the CT scan image.
        Returns class probabilities, predicted class, and additional metrics.
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Define class names based on your model's training
            class_names = ["Benign Lung Cancer", "Malignant Lung Cancer", "Normal"]
            
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
            plt.figure(figsize=(10, 6))
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            bar_plot = plt.bar(class_names, predictions, color=colors)
            
            # Add value labels on top of bars
            for bar, prob in zip(bar_plot, predictions):
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + 0.01, 
                        f'{prob:.2%}', 
                        ha='center')
                
            plt.xlabel('Lung CT Scan Categories')
            plt.ylabel('Probability')
            plt.title('Lung Cancer Classification Probabilities')
            plt.ylim(0, 1.1)  # Set y-axis limit to accommodate labels
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plot_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Provide appropriate medical guidance based on predicted class
            guidance = {
                "Benign Lung Cancer": "Benign lung nodules detected.",
                "Malignant Lung Cancer": "Potential malignant indicators detected.",
                "Normal": "No significant abnormalities detected in the lung CT scan."
            }
            
            # Severity and risk assessment
            severity = {
                "Benign Lung Cancer": "Low to Moderate",
                "Malignant Lung Cancer": "High",
                "Normal": "None"
            }
            
            # Model performance metrics (based on your reported accuracy)
            model_metrics = {
                "accuracy": 98.55,
                "reliability": "This model has demonstrated high accuracy in controlled testing environments, but should not replace professional medical diagnosis."
            }
            
            result = {
                #"prediction_scores": predictions.tolist(),
                #"predicted_class": predicted_class,
                "confidence": confidence,
                #"image_data": img_str,
                "probability_chart": plot_str,
                "medical_guidance": guidance[predicted_class],
                "severity_level": severity[predicted_class],
                "model_metrics": model_metrics,
                #"recommendation": "This is an AI-assisted analysis and should be confirmed by a medical professional. Please consult with a pulmonologist or oncologist to discuss these findings.",
                "next_steps": [
                    "Consult with a pulmonologist to review these results",
                    "Schedule additional diagnostic tests if recommended",
                    #"Discuss treatment options if applicable",
                    #"Consider follow-up scans to monitor any changes"
                ]
            }
            
            return result
        except Exception as e:
            return {"error": str(e)}
