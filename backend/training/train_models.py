"""
Machine Learning Model Training Script
=======================================
This script trains Random Forest and XGBoost models for network intrusion detection.

Models trained:
1. Random Forest Classifier - Ensemble of decision trees
2. XGBoost Classifier - Gradient boosting with better performance

Both models are trained for:
- Binary classification (normal vs attack)
- Multi-class classification (normal, DoS, Probe, R2L, U2R)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import joblib

# Scikit-learn imports for ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import xgboost as xgb

# Setup logging to track training progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and evaluates Random Forest and XGBoost models.
    
    The class handles:
    - Data loading and preparation
    - Feature-target split
    - Train-test split (80-20)
    - Model training with optimized parameters
    - Model evaluation with multiple metrics
    - Model saving for deployment
    """
    
    def __init__(self, data_path: str, model_save_dir: str):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to cleaned dataset CSV
            model_save_dir: Directory to save trained models
        """
        self.data_path = Path(data_path)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train_binary = None
        self.y_test_binary = None
        self.y_train_multi = None
        self.y_test_multi = None
        
        # Store trained models
        self.models = {}
        
    def load_and_prepare_data(self):
        """
        Step 1: Load the cleaned dataset and prepare features.
        
        What happens here:
        1. Load CSV file into pandas DataFrame
        2. Identify feature columns (exclude label columns)
        3. Separate features (X) and targets (y for binary and multi-class)
        4. Split into training (80%) and testing (20%) sets
        """
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded. Shape: {self.df.shape}")
        
        # Feature columns: All numerical columns except labels
        # We exclude the label columns to get only the network traffic features
        label_columns = ['label', 'label_original', 'label_binary', 'label_encoded']
        feature_columns = [col for col in self.df.columns if col not in label_columns]
        
        logger.info(f"Number of features: {len(feature_columns)}")
        
        # Prepare feature matrix (X) - All network traffic features
        X = self.df[feature_columns].values
        
        # Prepare target vectors (y)
        # Binary: 0 = normal traffic, 1 = attack
        y_binary = self.df['label_binary'].values
        
        # Multi-class: 0=normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
        y_multi = self.df['label_encoded'].values
        
        # Split data: 80% training, 20% testing
        # random_state=42 ensures reproducibility (same split every time)
        # stratify ensures balanced distribution of classes in train/test
        logger.info("Splitting data into train (80%) and test (20%) sets...")
        
        self.X_train, self.X_test, self.y_train_binary, self.y_test_binary = train_test_split(
            X, y_binary, 
            test_size=0.2,      # 20% for testing
            random_state=42,    # Reproducible split
            stratify=y_binary   # Keep class balance
        )
        
        # Same split for multi-class (same samples, different labels)
        _, _, self.y_train_multi, self.y_test_multi = train_test_split(
            X, y_multi,
            test_size=0.2,
            random_state=42,
            stratify=y_multi
        )
        
        logger.info(f"Training set size: {self.X_train.shape[0]} samples")
        logger.info(f"Testing set size: {self.X_test.shape[0]} samples")
        logger.info(f"Feature dimensions: {self.X_train.shape[1]} features")
        
    def train_random_forest_binary(self):
        """
        Step 2a: Train Random Forest for Binary Classification.
        
        Random Forest works by:
        1. Creating multiple decision trees (100 trees here)
        2. Each tree votes on the prediction
        3. Majority vote wins
        
        Advantages:
        - Handles high-dimensional data well
        - Resistant to overfitting
        - Fast training and prediction
        """
        logger.info("\n" + "="*70)
        logger.info("Training Random Forest - Binary Classification (Normal vs Attack)")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Initialize Random Forest Classifier
        # n_estimators=100: Use 100 decision trees
        # max_depth=20: Limit tree depth to prevent overfitting
        # min_samples_split=10: Minimum samples needed to split a node
        # n_jobs=-1: Use all CPU cores for parallel training
        rf_binary = RandomForestClassifier(
            n_estimators=100,        # Number of trees in the forest
            max_depth=20,            # Maximum depth of each tree
            min_samples_split=10,    # Min samples to split internal node
            min_samples_leaf=4,      # Min samples required at leaf node
            random_state=42,         # Reproducible results
            n_jobs=-1,               # Use all CPU cores
            verbose=1                # Show progress
        )
        
        # Train the model
        logger.info("Training started...")
        rf_binary.fit(self.X_train, self.y_train_binary)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set
        self._evaluate_model(
            model=rf_binary,
            X_test=self.X_test,
            y_test=self.y_test_binary,
            model_name="Random Forest Binary",
            class_names=['Normal', 'Attack']
        )
        
        # Save the model
        model_path = self.model_save_dir / "random_forest_binary.joblib"
        joblib.dump(rf_binary, model_path)
        logger.info(f"Model saved to {model_path}")
        
        self.models['rf_binary'] = rf_binary
        return rf_binary
    
    def train_random_forest_multiclass(self):
        """
        Step 2b: Train Random Forest for Multi-Class Classification.
        
        Multi-class distinguishes between 5 attack types:
        - Normal traffic
        - DoS (Denial of Service)
        - Probe (Scanning/Probing)
        - R2L (Remote to Local attacks)
        - U2R (User to Root privilege escalation)
        """
        logger.info("\n" + "="*70)
        logger.info("Training Random Forest - Multi-Class Classification")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Similar configuration as binary but for 5 classes
        rf_multi = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Training started...")
        rf_multi.fit(self.X_train, self.y_train_multi)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set
        self._evaluate_model(
            model=rf_multi,
            X_test=self.X_test,
            y_test=self.y_test_multi,
            model_name="Random Forest Multi-Class",
            class_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        )
        
        # Save the model
        model_path = self.model_save_dir / "random_forest_multiclass.joblib"
        joblib.dump(rf_multi, model_path)
        logger.info(f"Model saved to {model_path}")
        
        self.models['rf_multi'] = rf_multi
        return rf_multi
    
    def train_xgboost_binary(self):
        """
        Step 3a: Train XGBoost for Binary Classification.
        
        XGBoost (eXtreme Gradient Boosting) works by:
        1. Building trees sequentially
        2. Each new tree corrects errors of previous trees
        3. Uses gradient descent for optimization
        
        Advantages over Random Forest:
        - Often higher accuracy
        - Better handling of imbalanced data
        - Built-in regularization
        """
        logger.info("\n" + "="*70)
        logger.info("Training XGBoost - Binary Classification (Normal vs Attack)")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Initialize XGBoost Classifier
        # n_estimators=100: Number of boosting rounds
        # max_depth=6: Maximum depth of each tree (shallower than RF)
        # learning_rate=0.1: Step size for each iteration
        # subsample=0.8: Use 80% of samples for each tree
        xgb_binary = xgb.XGBClassifier(
            n_estimators=100,           # Number of boosting rounds
            max_depth=6,                # Maximum tree depth
            learning_rate=0.1,          # Learning rate (eta)
            subsample=0.8,              # Subsample ratio of training data
            colsample_bytree=0.8,       # Subsample ratio of features
            objective='binary:logistic', # Binary classification
            random_state=42,
            n_jobs=-1,
            verbosity=1                 # Show progress
        )
        
        logger.info("Training started...")
        xgb_binary.fit(self.X_train, self.y_train_binary)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set
        self._evaluate_model(
            model=xgb_binary,
            X_test=self.X_test,
            y_test=self.y_test_binary,
            model_name="XGBoost Binary",
            class_names=['Normal', 'Attack']
        )
        
        # Save the model
        model_path = self.model_save_dir / "xgboost_binary.joblib"
        joblib.dump(xgb_binary, model_path)
        logger.info(f"Model saved to {model_path}")
        
        self.models['xgb_binary'] = xgb_binary
        return xgb_binary
    
    def train_xgboost_multiclass(self):
        """
        Step 3b: Train XGBoost for Multi-Class Classification.
        
        Uses softmax objective for multi-class prediction.
        Predicts probability for each of the 5 attack types.
        """
        logger.info("\n" + "="*70)
        logger.info("Training XGBoost - Multi-Class Classification")
        logger.info("="*70)
        
        start_time = time.time()
        
        # XGBoost for multi-class uses 'multi:softmax' objective
        xgb_multi = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',   # Multi-class classification
            num_class=5,                 # 5 attack types
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        
        logger.info("Training started...")
        xgb_multi.fit(self.X_train, self.y_train_multi)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set
        self._evaluate_model(
            model=xgb_multi,
            X_test=self.X_test,
            y_test=self.y_test_multi,
            model_name="XGBoost Multi-Class",
            class_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        )
        
        # Save the model
        model_path = self.model_save_dir / "xgboost_multiclass.joblib"
        joblib.dump(xgb_multi, model_path)
        logger.info(f"Model saved to {model_path}")
        
        self.models['xgb_multi'] = xgb_multi
        return xgb_multi
    
    def _evaluate_model(self, model, X_test, y_test, model_name, class_names):
        """
        Step 4: Evaluate Model Performance.
        
        Calculates and displays:
        1. Accuracy: Overall correct predictions
        2. Precision: Of all predicted attacks, how many were actual attacks
        3. Recall: Of all actual attacks, how many did we catch
        4. F1-Score: Harmonic mean of precision and recall
        5. Confusion Matrix: Detailed breakdown of predictions
        6. Per-class metrics: Performance for each attack type
        """
        logger.info(f"\n--- {model_name} Evaluation ---")
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # For multi-class, use weighted average
        avg_type = 'binary' if len(class_names) == 2 else 'weighted'
        
        precision = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)
        
        # Display metrics
        logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        
        # Confusion Matrix
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n{cm}")
        
        # Detailed classification report
        logger.info("\nDetailed Classification Report:")
        report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            zero_division=0
        )
        logger.info(f"\n{report}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def train_all_models(self):
        """
        Main training pipeline.
        
        Executes in order:
        1. Load and prepare data
        2. Train Random Forest (binary and multi-class)
        3. Train XGBoost (binary and multi-class)
        4. Save all models
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Prepare data
        self.load_and_prepare_data()
        
        # Step 2: Train Random Forest models
        self.train_random_forest_binary()
        self.train_random_forest_multiclass()
        
        # Step 3: Train XGBoost models
        self.train_xgboost_binary()
        self.train_xgboost_multiclass()
        
        logger.info("\n" + "="*70)
        logger.info("ALL MODELS TRAINED SUCCESSFULLY!")
        logger.info(f"Models saved to: {self.model_save_dir}")
        logger.info("="*70)


def main():
    """
    Main execution function.
    
    Training on NSL-KDD dataset because:
    - Smaller size (4,430 samples) - faster training
    - Already balanced - all classes have 886 samples
    - Good for testing and validation
    
    For production, you would train on CICIDS2017 (2.5M samples).
    """
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    
    # Use NSL-KDD for training (smaller, balanced dataset)
    data_path = project_root / "dataset" / "cleaned" / "nsl_kdd_cleaned.csv"
    
    # Directory to save trained models
    model_save_dir = project_root / "backend" / "models" / "trained"
    
    # Check if data exists
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        logger.error("Please run data_cleaning.py first to generate cleaned data.")
        return
    
    # Initialize trainer and train all models
    trainer = ModelTrainer(
        data_path=str(data_path),
        model_save_dir=str(model_save_dir)
    )
    
    # Train all 4 models
    trainer.train_all_models()
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    logger.info("Models trained and saved:")
    logger.info("1. random_forest_binary.joblib - Binary classification")
    logger.info("2. random_forest_multiclass.joblib - 5-class classification")
    logger.info("3. xgboost_binary.joblib - Binary classification")
    logger.info("4. xgboost_multiclass.joblib - 5-class classification")
    logger.info("\nYou can now use these models for prediction!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
