"""
Machine Learning Service - Flexible Feature Handling
====================================================
Handles model loading, predictions, and analysis.
Supports datasets with varying feature counts through feature adaptation.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler

from app.config import MODELS_DIR, MODEL_FILES, BINARY_LABELS, MULTICLASS_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLService:
    """
    Flexible ML Service with feature adaptation.
    
    Supports:
    1. Load trained models from disk
    2. Adapt to any number of input features
    3. Feature engineering and selection
    4. Make predictions using loaded models
    5. Calculate confidence scores
    """
    
    def __init__(self):
        """Initialize the ML service and load models."""
        self.models = {}
        self.models_loaded = False
        self.expected_features = None  # Will be set from trained model
        self.feature_names = None
        self.scaler = StandardScaler()
        self._load_models()
    
    def _load_models(self):
        """
        Load all trained models from disk.
        Also loads feature metadata if available.
        """
        logger.info("Loading ML models...")
        
        try:
            for model_key, model_file in MODEL_FILES.items():
                model_path = MODELS_DIR / model_file
                
                if model_path.exists():
                    self.models[model_key] = joblib.load(model_path)
                    logger.info(f"✓ Loaded: {model_key}")
                    
                    # Try to get expected features from model
                    if hasattr(self.models[model_key], 'n_features_in_'):
                        if self.expected_features is None:
                            self.expected_features = self.models[model_key].n_features_in_
                            logger.info(f"Model expects {self.expected_features} features")
                else:
                    logger.warning(f"✗ Not found: {model_key} at {model_path}")
            
            # Load feature metadata if exists
            metadata_path = MODELS_DIR / "feature_metadata.joblib"
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                self.feature_names = metadata.get('feature_names')
                self.scaler = metadata.get('scaler', StandardScaler())
                logger.info(f"✓ Loaded feature metadata")
            
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
        Flexible data validation and preparation.
        Adapts to any number of input features.
        
        Args:
            df: Input DataFrame with network traffic data
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        logger.info(f"Validating data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Remove label columns if they exist
        label_columns = ['label', 'label_original', 'label_binary', 'label_encoded']
        feature_df = df.drop(columns=[col for col in label_columns if col in df.columns], errors='ignore')
        
        # Get feature columns
        feature_columns = feature_df.columns.tolist()
        actual_features = len(feature_columns)
        
        logger.info(f"Found {actual_features} features in dataset")
        
        # Convert to numpy array
        X = feature_df.values
        
        # Handle missing and infinite values
        if np.isnan(X).any():
            n_missing = np.isnan(X).sum()
            logger.warning(f"Found {n_missing} missing values, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        if np.isinf(X).any():
            n_inf = np.isinf(X).sum()
            logger.warning(f"Found {n_inf} infinite values, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature adaptation: align with model expectations
        X = self._adapt_features(X, feature_columns)
        
        logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, feature_columns
    
    def _adapt_features(self, X: np.ndarray, input_features: List[str]) -> np.ndarray:
        """
        Adapt input features to match model requirements.
        
        Strategies:
        1. If features match: use as-is
        2. If fewer features: pad with statistical features
        3. If more features: select most relevant
        
        Args:
            X: Input feature matrix
            input_features: List of input feature names
            
        Returns:
            Adapted feature matrix
        """
        current_features = X.shape[1]
        
        # If model has no expected features, use input as-is
        if self.expected_features is None:
            logger.warning("No expected features defined. Using input as-is.")
            return X
        
        # Perfect match - no adaptation needed
        if current_features == self.expected_features:
            logger.info(f"✓ Feature count matches: {current_features}")
            return X
        
        # Need adaptation
        logger.info(f"Adapting features: {current_features} → {self.expected_features}")
        
        if current_features < self.expected_features:
            # Pad with engineered features
            return self._pad_features(X)
        else:
            # Select top features
            return self._select_features(X, input_features)
    
    def _pad_features(self, X: np.ndarray) -> np.ndarray:
        """
        Pad feature matrix with statistical features.
        
        Creates additional features from existing ones:
        - Mean, std, min, max of existing features
        - Interaction terms
        - Polynomial features (squares)
        """
        current = X.shape[1]
        needed = self.expected_features - current
        
        logger.info(f"Padding {needed} features using feature engineering")
        
        # Create statistical features
        additional_features = []
        
        # Row-wise statistics
        if needed > 0:
            additional_features.append(np.mean(X, axis=1, keepdims=True))
            needed -= 1
        if needed > 0:
            additional_features.append(np.std(X, axis=1, keepdims=True))
            needed -= 1
        if needed > 0:
            additional_features.append(np.min(X, axis=1, keepdims=True))
            needed -= 1
        if needed > 0:
            additional_features.append(np.max(X, axis=1, keepdims=True))
            needed -= 1
        
        # Add squared features (polynomial)
        if needed > 0 and current > 0:
            n_squared = min(needed, current)
            squared = X[:, :n_squared] ** 2
            additional_features.append(squared)
            needed -= n_squared
        
        # Add interaction terms if still needed
        if needed > 0 and current >= 2:
            n_interactions = min(needed, current // 2)
            for i in range(n_interactions):
                interaction = X[:, i] * X[:, (i + 1) % current]
                additional_features.append(interaction.reshape(-1, 1))
                needed -= 1
                if needed <= 0:
                    break
        
        # Fill remaining with zeros if still needed
        if needed > 0:
            additional_features.append(np.zeros((X.shape[0], needed)))
        
        # Concatenate all features
        X_padded = np.hstack([X] + additional_features)
        
        # Ensure exact match
        if X_padded.shape[1] > self.expected_features:
            X_padded = X_padded[:, :self.expected_features]
        
        logger.info(f"✓ Padded to {X_padded.shape[1]} features")
        return X_padded
    
    def _select_features(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Select most relevant features using variance-based selection.
        
        Args:
            X: Input feature matrix
            feature_names: Feature names
            
        Returns:
            Selected feature matrix
        """
        logger.info(f"Selecting top {self.expected_features} features by variance")
        
        # Calculate feature variances
        variances = np.var(X, axis=0)
        
        # Select indices of top features
        top_indices = np.argsort(variances)[-self.expected_features:]
        
        # Select features
        X_selected = X[:, top_indices]
        
        logger.info(f"✓ Selected {X_selected.shape[1]} most informative features")
        return X_selected
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        Make predictions using all loaded models.
        
        Args:
            X: Feature matrix (samples x features)
            
        Returns:
            Dictionary containing predictions and probabilities from all models
            
        Raises:
            RuntimeError: If models are not loaded
            ValueError: If input data is invalid
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Cannot make predictions.")
        
        # Validate input
        if X.shape[0] == 0:
            raise ValueError("Cannot make predictions on empty dataset")
        
        if X.shape[1] == 0:
            raise ValueError("Cannot make predictions with zero features")
        
        results = {}
        
        try:
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
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Model prediction failed: {str(e)}")
    
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
