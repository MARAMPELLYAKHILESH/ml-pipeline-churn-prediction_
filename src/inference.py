"""
Inference Module
Handles predictions for new data
"""

import pickle
import numpy as np
import pandas as pd

class ModelPredictor:
    """Handles model inference"""
    
    def __init__(self, model_path, preprocessor_path):
        """Initialize predictor with model and preprocessor"""
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)
    
    def _load_model(self, path):
        """Load trained model"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def _load_preprocessor(self, path):
        """Load preprocessor"""
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        return preprocessor
    
    def preprocess_input(self, data):
        """Preprocess input data"""
        # If data is a dict, convert to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Scale features
        scaler = self.preprocessor['scaler']
        feature_names = self.preprocessor['feature_names']
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in data.columns:
                data[feature] = 0
        
        # Select and order features
        data = data[feature_names]
        
        # Scale
        data_scaled = scaler.transform(data)
        
        return data_scaled
    
    def predict(self, data):
        """Make prediction"""
        # Preprocess
        X = self.preprocess_input(data)
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            probability = float(probabilities[1])  # Probability of positive class
        
        return {
            'prediction': int(prediction),
            'probability': probability,
            'prediction_label': 'Churn' if prediction == 1 else 'No Churn'
        }
    
    def predict_batch(self, data_list):
        """Make batch predictions"""
        results = []
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        return results

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = ModelPredictor(
        model_path='models/best_model.pkl',
        preprocessor_path='data/processed/preprocessor.pkl'
    )
    
    # Example data
    sample_data = {
        'age': 35,
        'tenure': 24,
        'monthly_charges': 65.5,
        'total_charges': 1572.0,
        # Add other features...
    }
    
    # Make prediction
    result = predictor.predict(sample_data)
    
    print("Prediction Result:")
    print(f"  Prediction: {result['prediction_label']}")
    if result['probability']:
        print(f"  Probability: {result['probability']:.3f}")