"""
Machine Learning Service
========================
Handles model loading, predictions, and analysis.
This service manages all ML operations for the API.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from app.config import MODELS_DIR, MODEL_FILES, BINARY_LABELS, MULTICLASS_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLService:
    """
    Service class for ML operations.
    
    Responsibilities:
    1. Load trained models from disk
    2. Process CSV data and validate features
    3. Make predictions using loaded models
    4. Calculate confidence scores
    5. Generate prediction summaries
    """
    
    def __init__(self):
        """Initialize the ML service and load models."""
        self.models = {}
        self.models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """
        Load all trained models from disk.
        
        Models loaded:
        - Random Forest Binary
        - Random Forest Multi-Class
        - XGBoost Binary
        - XGBoost Multi-Class
        """
        logger.info("Loading ML models...")
        
        try:
            for model_key, model_file in MODEL_FILES.items():
                model_path = MODELS_DIR / model_file
                
                if model_path.exists():
                    self.models[model_key] = joblib.load(model_path)
                    logger.info(f"✓ Loaded: {model_key}")
                else:
                    logger.warning(f"✗ Not found: {model_key} at {model_path}")
            
            # Check if we have at least the primary models
            if "xgb_binary" in self.models and "xgb_multiclass" in self.models:
                self.models_loaded = True
                logger.info("ML models loaded successfully!")
            else:
                logger.error("Failed to load required models")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False
    
    def validate_and_prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Validate CSV data and prepare features for prediction.
        
        Expected features: 41 network traffic features
        (same features used during training)
        
        Args:
            df: Input DataFrame with network traffic data
            
        Returns:
            Tuple of (feature_matrix, feature_names)
            
        Raises:
            ValueError: If data validation fails
        """
        logger.info(f"Validating data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Remove label columns if they exist
        label_columns = ['label', 'label_original', 'label_binary', 'label_encoded']
        feature_df = df.drop(columns=[col for col in label_columns if col in df.columns], errors='ignore')
        
        # Get feature columns
        feature_columns = feature_df.columns.tolist()
        
        # Check if we have the right number of features
        # NSL-KDD has 41 features (after removing labels)
        if len(feature_columns) < 10:
            raise ValueError(f"Too few features: expected at least 10, got {len(feature_columns)}")
        
        # Convert to numpy array
        X = feature_df.values
        
        # Check for missing values
        if np.isnan(X).any():
            logger.warning("Found missing values, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, feature_columns
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        Make predictions using all loaded models.
        
        Args:
            X: Feature matrix (samples x features)
            
        Returns:
            Dictionary containing predictions and probabilities from all models
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Cannot make predictions.")
        
        results = {}
        
        # Binary predictions (using XGBoost as primary)
        if "xgb_binary" in self.models:
            model = self.models["xgb_binary"]
            results["binary_pred"] = model.predict(X)
            
            # Get probability scores if available
            if hasattr(model, "predict_proba"):
                results["binary_proba"] = model.predict_proba(X)
            else:
                # Create dummy probabilities based on predictions
                results["binary_proba"] = np.zeros((len(X), 2))
                for i, pred in enumerate(results["binary_pred"]):
                    results["binary_proba"][i, int(pred)] = 0.9
                    results["binary_proba"][i, 1 - int(pred)] = 0.1
        
        # Multi-class predictions (using XGBoost as primary)
        if "xgb_multiclass" in self.models:
            model = self.models["xgb_multiclass"]
            results["multiclass_pred"] = model.predict(X)
            
            # Get probability scores if available
            if hasattr(model, "predict_proba"):
                results["multiclass_proba"] = model.predict_proba(X)
            else:
                # Create dummy probabilities
                n_classes = 5
                results["multiclass_proba"] = np.zeros((len(X), n_classes))
                for i, pred in enumerate(results["multiclass_pred"]):
                    results["multiclass_proba"][i, int(pred)] = 0.8
                    # Distribute remaining probability
                    remaining = 0.2 / (n_classes - 1)
                    for j in range(n_classes):
                        if j != int(pred):
                            results["multiclass_proba"][i, j] = remaining
        
        logger.info(f"Predictions complete for {X.shape[0]} samples")
        
        return results
    
    def format_predictions(self, predictions: Dict, limit: int = None) -> List[Dict]:
        """
        Format predictions into a list of dictionaries for API response.
        
        Args:
            predictions: Raw predictions from predict()
            limit: Maximum number of predictions to return (None = all)
            
        Returns:
            List of prediction dictionaries
        """
        n_samples = len(predictions["binary_pred"])
        if limit:
            n_samples = min(n_samples, limit)
        
        formatted = []
        
        for i in range(n_samples):
            # Binary prediction
            binary_class = int(predictions["binary_pred"][i])
            binary_label = BINARY_LABELS[binary_class]
            binary_conf = float(predictions["binary_proba"][i, binary_class])
            
            # Multi-class prediction
            multi_class = int(predictions["multiclass_pred"][i])
            multi_label = MULTICLASS_LABELS[multi_class]
            multi_conf = float(predictions["multiclass_proba"][i, multi_class])
            
            formatted.append({
                "sample_id": i,
                "binary_prediction": binary_label,
                "binary_confidence": round(binary_conf, 4),
                "multiclass_prediction": multi_label,
                "multiclass_confidence": round(multi_conf, 4)
            })
        
        return formatted
    
    def calculate_summary(self, predictions: Dict) -> Dict:
        """
        Calculate summary statistics from predictions.
        
        Args:
            predictions: Raw predictions from predict()
            
        Returns:
            Dictionary with summary statistics
        """
        total = len(predictions["binary_pred"])
        
        # Binary distribution
        binary_counts = np.bincount(predictions["binary_pred"].astype(int), minlength=2)
        binary_dist = [
            {
                "label": BINARY_LABELS[i],
                "count": int(binary_counts[i]),
                "percentage": round(float(binary_counts[i]) / total * 100, 2)
            }
            for i in range(len(binary_counts))
        ]
        
        # Multi-class distribution
        multi_counts = np.bincount(predictions["multiclass_pred"].astype(int), minlength=5)
        multi_dist = [
            {
                "label": MULTICLASS_LABELS[i],
                "count": int(multi_counts[i]),
                "percentage": round(float(multi_counts[i]) / total * 100, 2)
            }
            for i in range(len(multi_counts))
        ]
        
        # Attack vs Normal percentages
        normal_pct = round(float(binary_counts[0]) / total * 100, 2)
        attack_pct = round(float(binary_counts[1]) / total * 100, 2)
        
        return {
            "total_samples": total,
            "binary_distribution": binary_dist,
            "multiclass_distribution": multi_dist,
            "normal_percentage": normal_pct,
            "attack_percentage": attack_pct
        }
    
    def get_model_status(self) -> Dict[str, bool]:
        """
        Get the loading status of all models.
        
        Returns:
            Dictionary mapping model names to loaded status
        """
        return {
            model_key: model_key in self.models
            for model_key in MODEL_FILES.keys()
        }


# Global ML service instance (singleton pattern)
ml_service = MLService()
