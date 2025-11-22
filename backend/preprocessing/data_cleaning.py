"""
Data Cleaning Script for NSL-KDD and CICIDS2017 Datasets
This script handles data preprocessing, cleaning, and validation for network intrusion detection datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Data cleaning and preprocessing class for network intrusion detection datasets.
    
    NSL-KDD Dataset Features:
    - 41 numerical features (all normalized between 0-1)
    - Original features represent network connection attributes like:
      * Basic features: duration, protocol_type, service, flag, src_bytes, dst_bytes
      * Content features: hot, num_failed_logins, logged_in, num_compromised, etc.
      * Traffic features: count, srv_count, serror_rate, srv_serror_rate, etc.
      * Host-based features: dst_host_count, dst_host_srv_count, etc.
    - Labels: normal, DoS, Probe, R2L, U2R (5 classes)
      * normal: Normal network traffic
      * DoS: Denial of Service attacks
      * Probe: Surveillance and probing attacks
      * R2L: Remote to Local attacks (unauthorized access from remote)
      * U2R: User to Root attacks (privilege escalation)
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.df = None
        self.feature_names = None
        self.label_column = 'label'
        
    def load_data(self):
        """Load the dataset from CSV file."""
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def inspect_data(self):
        """Inspect the dataset for basic information."""
        if self.df is None:
            logger.error("Dataset not loaded. Call load_data() first.")
            return
        
        logger.info("\n=== Dataset Inspection ===")
        logger.info(f"Shape: {self.df.shape}")
        logger.info(f"Columns: {self.df.columns.tolist()}")
        logger.info(f"\nData types:\n{self.df.dtypes}")
        logger.info(f"\nMissing values:\n{self.df.isnull().sum()}")
        logger.info(f"\nBasic statistics:\n{self.df.describe()}")
        
        if self.label_column in self.df.columns:
            logger.info(f"\nLabel distribution:\n{self.df[self.label_column].value_counts()}")
    
    def handle_missing_values(self, strategy='drop'):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: 'drop' to remove rows with missing values, 
                     'mean' to fill with mean, 
                     'median' to fill with median,
                     'mode' to fill with mode
        """
        if self.df is None:
            logger.error("Dataset not loaded.")
            return
        
        missing_count = self.df.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found in the dataset.")
            return
        
        logger.info(f"Found {missing_count} missing values. Applying strategy: {strategy}")
        
        if strategy == 'drop':
            self.df = self.df.dropna()
            logger.info(f"Dropped rows with missing values. New shape: {self.df.shape}")
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            logger.info("Filled missing values with mean.")
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
            logger.info("Filled missing values with median.")
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else self.df[col])
            logger.info("Filled missing values with mode.")
        
        return self.df
    
    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        if self.df is None:
            logger.error("Dataset not loaded.")
            return
        
        original_shape = self.df.shape
        self.df = self.df.drop_duplicates()
        removed_count = original_shape[0] - self.df.shape[0]
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows. New shape: {self.df.shape}")
        else:
            logger.info("No duplicate rows found.")
        
        return self.df
    
    def handle_outliers(self, method='iqr', threshold=1.5):
        """
        Detect and handle outliers in numerical features.
        
        Args:
            method: 'iqr' for Interquartile Range method, 'zscore' for Z-score method
            threshold: IQR multiplier (default 1.5) or Z-score threshold (default 3)
        """
        if self.df is None:
            logger.error("Dataset not loaded.")
            return
        
        # Get numerical columns (excluding label)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.label_column in numeric_cols:
            numeric_cols.remove(self.label_column)
        
        logger.info(f"Detecting outliers using {method} method...")
        outlier_count = 0
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    logger.info(f"  {col}: {outliers} outliers detected")
                    outlier_count += outliers
                    # Cap outliers instead of removing (preserve data for ML)
                    self.df[col] = self.df[col].clip(lower_bound, upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(self.df[col]))
                outliers = (z_scores > threshold).sum()
                if outliers > 0:
                    logger.info(f"  {col}: {outliers} outliers detected")
                    outlier_count += outliers
        
        logger.info(f"Total outliers detected: {outlier_count}")
        if method == 'iqr':
            logger.info("Outliers capped to IQR bounds.")
        
        return self.df
    
    def validate_data_ranges(self):
        """Validate that normalized features are in the expected range [0, 1]."""
        if self.df is None:
            logger.error("Dataset not loaded.")
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.label_column in numeric_cols:
            numeric_cols.remove(self.label_column)
        
        logger.info("Validating data ranges...")
        invalid_ranges = []
        
        for col in numeric_cols:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            
            if min_val < 0 or max_val > 1:
                invalid_ranges.append((col, min_val, max_val))
                # Clip values to [0, 1] range
                self.df[col] = self.df[col].clip(0, 1)
        
        if invalid_ranges:
            logger.warning(f"Found {len(invalid_ranges)} columns with values outside [0, 1] range:")
            for col, min_val, max_val in invalid_ranges:
                logger.warning(f"  {col}: [{min_val:.4f}, {max_val:.4f}] -> clipped to [0, 1]")
        else:
            logger.info("All numerical features are within [0, 1] range.")
        
        return self.df
    
    def encode_labels(self):
        """
        Encode categorical labels for machine learning.
        Creates both binary (normal vs attack) and multi-class encoding.
        """
        if self.df is None:
            logger.error("Dataset not loaded.")
            return
        
        if self.label_column not in self.df.columns:
            logger.error(f"Label column '{self.label_column}' not found.")
            return
        
        logger.info("Encoding labels...")
        
        # Store original labels
        self.df['label_original'] = self.df[self.label_column].copy()
        
        # Binary classification: 0 = normal, 1 = attack
        self.df['label_binary'] = (self.df[self.label_column] != 'normal').astype(int)
        
        # Multi-class encoding: map each attack type to a number
        label_mapping = {
            'normal': 0,
            'DoS': 1,
            'Probe': 2,
            'R2L': 3,
            'U2R': 4
        }
        self.df['label_encoded'] = self.df[self.label_column].map(label_mapping)
        
        # Check for any unmapped labels
        unmapped = self.df[self.df['label_encoded'].isna()][self.label_column].unique()
        if len(unmapped) > 0:
            logger.warning(f"Unmapped labels found: {unmapped}")
        
        logger.info(f"Label encoding complete:")
        logger.info(f"  Original labels: {self.df['label_original'].value_counts().to_dict()}")
        logger.info(f"  Binary encoding: {self.df['label_binary'].value_counts().to_dict()}")
        logger.info(f"  Multi-class encoding: {self.df['label_encoded'].value_counts().to_dict()}")
        
        return self.df
    
    def balance_dataset(self, method='oversample', random_state=42):
        """
        Balance the dataset to handle class imbalance.
        
        Args:
            method: 'oversample' (random oversampling), 'undersample' (random undersampling),
                   'smote' (Synthetic Minority Over-sampling)
        """
        if self.df is None:
            logger.error("Dataset not loaded.")
            return
        
        if 'label_encoded' not in self.df.columns:
            logger.warning("Labels not encoded. Calling encode_labels() first.")
            self.encode_labels()
        
        logger.info(f"Balancing dataset using {method} method...")
        logger.info(f"Original class distribution:\n{self.df['label_encoded'].value_counts()}")
        
        # This is a placeholder - actual implementation would use imblearn library
        # For now, just log the intent
        logger.info(f"Dataset balancing with {method} method would be applied here.")
        logger.info("Note: Install imbalanced-learn library for actual implementation.")
        
        return self.df
    
    def save_cleaned_data(self, output_path: str):
        """Save the cleaned dataset to a CSV file."""
        if self.df is None:
            logger.error("No data to save.")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.df.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
            logger.info(f"Final dataset shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
            raise
    
    def get_feature_importance_data(self):
        """Prepare data for feature importance analysis."""
        if self.df is None:
            logger.error("Dataset not loaded.")
            return None, None
        
        if 'label_encoded' not in self.df.columns:
            logger.warning("Labels not encoded. Calling encode_labels() first.")
            self.encode_labels()
        
        # Get feature columns (exclude label columns)
        label_cols = ['label', 'label_original', 'label_binary', 'label_encoded']
        feature_cols = [col for col in self.df.columns if col not in label_cols]
        
        X = self.df[feature_cols]
        y = self.df['label_encoded']
        
        return X, y


def clean_nsl_kdd_dataset(input_path: str, output_path: str):
    """
    Main function to clean NSL-KDD dataset.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the cleaned CSV file
    """
    logger.info("=" * 50)
    logger.info("Starting NSL-KDD Dataset Cleaning Process")
    logger.info("=" * 50)
    
    # Initialize cleaner
    cleaner = DataCleaner(input_path)
    
    # Load data
    cleaner.load_data()
    
    # Inspect data
    cleaner.inspect_data()
    
    # Cleaning steps
    cleaner.handle_missing_values(strategy='drop')
    cleaner.remove_duplicates()
    cleaner.validate_data_ranges()
    
    # Handle outliers (using IQR method with capping instead of removal)
    cleaner.handle_outliers(method='iqr', threshold=1.5)
    
    # Encode labels
    cleaner.encode_labels()
    
    # Save cleaned data
    cleaner.save_cleaned_data(output_path)
    
    logger.info("=" * 50)
    logger.info("NSL-KDD Dataset Cleaning Complete")
    logger.info("=" * 50)
    
    return cleaner.df


def clean_cicids_dataset(input_path: str, output_path: str):
    """
    Main function to clean CICIDS2017 dataset.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the cleaned CSV file
    """
    logger.info("=" * 50)
    logger.info("Starting CICIDS2017 Dataset Cleaning Process")
    logger.info("=" * 50)
    
    # Initialize cleaner
    cleaner = DataCleaner(input_path)
    
    # Load data
    cleaner.load_data()
    
    # Inspect data
    cleaner.inspect_data()
    
    # Cleaning steps
    cleaner.handle_missing_values(strategy='drop')
    cleaner.remove_duplicates()
    
    # For CICIDS, might need normalization if not already normalized
    # This would be dataset-specific
    
    # Handle outliers
    cleaner.handle_outliers(method='iqr', threshold=1.5)
    
    # Encode labels
    cleaner.encode_labels()
    
    # Save cleaned data
    cleaner.save_cleaned_data(output_path)
    
    logger.info("=" * 50)
    logger.info("CICIDS2017 Dataset Cleaning Complete")
    logger.info("=" * 50)
    
    return cleaner.df


if __name__ == "__main__":
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    dataset_dir = project_root / "dataset"
    output_dir = dataset_dir / "cleaned"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Clean NSL-KDD dataset
    nsl_kdd_input = dataset_dir / "nsl_kdd_dataset.csv"
    nsl_kdd_output = output_dir / "nsl_kdd_cleaned.csv"
    
    if nsl_kdd_input.exists():
        clean_nsl_kdd_dataset(str(nsl_kdd_input), str(nsl_kdd_output))
    else:
        logger.warning(f"NSL-KDD dataset not found at {nsl_kdd_input}")
    
    # Clean CICIDS2017 dataset
    cicids_input = dataset_dir / "cicids2017_cleaned.csv"
    cicids_output = output_dir / "cicids2017_final_cleaned.csv"
    
    if cicids_input.exists():
        clean_cicids_dataset(str(cicids_input), str(cicids_output))
    else:
        logger.warning(f"CICIDS2017 dataset not found at {cicids_input}")
