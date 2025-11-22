"""
Data Validation Utilities
Provides functions to validate dataset quality and integrity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates dataset quality and integrity."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_report = {}
    
    def check_data_types(self):
        """Check if data types are appropriate."""
        logger.info("Checking data types...")
        type_info = self.df.dtypes.to_dict()
        self.validation_report['data_types'] = type_info
        return type_info
    
    def check_missing_values(self):
        """Check for missing values."""
        logger.info("Checking missing values...")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_info = {
            'total_missing': missing.sum(),
            'missing_by_column': missing[missing > 0].to_dict(),
            'missing_percentage': missing_pct[missing_pct > 0].to_dict()
        }
        
        self.validation_report['missing_values'] = missing_info
        return missing_info
    
    def check_data_quality(self):
        """Overall data quality check."""
        logger.info("Checking overall data quality...")
        
        quality_report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        self.validation_report['quality'] = quality_report
        return quality_report
    
    def check_label_distribution(self, label_column='label'):
        """Check distribution of labels."""
        if label_column not in self.df.columns:
            logger.warning(f"Label column '{label_column}' not found.")
            return None
        
        logger.info("Checking label distribution...")
        distribution = self.df[label_column].value_counts().to_dict()
        distribution_pct = (self.df[label_column].value_counts(normalize=True) * 100).to_dict()
        
        label_info = {
            'distribution': distribution,
            'distribution_percentage': distribution_pct,
            'unique_labels': self.df[label_column].nunique()
        }
        
        self.validation_report['label_distribution'] = label_info
        return label_info
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        self.check_data_types()
        self.check_missing_values()
        self.check_data_quality()
        self.check_label_distribution()
        
        return self.validation_report
    
    def print_report(self):
        """Print validation report to console."""
        report = self.generate_report()
        
        print("\n" + "=" * 70)
        print("DATA VALIDATION REPORT")
        print("=" * 70)
        
        print("\n--- Data Quality ---")
        for key, value in report.get('quality', {}).items():
            print(f"{key}: {value}")
        
        print("\n--- Missing Values ---")
        missing = report.get('missing_values', {})
        print(f"Total missing values: {missing.get('total_missing', 0)}")
        if missing.get('missing_by_column'):
            print("Missing values by column:")
            for col, count in missing['missing_by_column'].items():
                pct = missing['missing_percentage'].get(col, 0)
                print(f"  {col}: {count} ({pct:.2f}%)")
        else:
            print("No missing values found!")
        
        print("\n--- Label Distribution ---")
        label_info = report.get('label_distribution', {})
        if label_info:
            print(f"Unique labels: {label_info.get('unique_labels', 0)}")
            print("Distribution:")
            for label, count in label_info.get('distribution', {}).items():
                pct = label_info['distribution_percentage'].get(label, 0)
                print(f"  {label}: {count} ({pct:.2f}%)")
        
        print("\n" + "=" * 70)


def validate_dataset(file_path: str):
    """
    Validate a dataset file.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        validation report dictionary
    """
    df = pd.read_csv(file_path)
    validator = DataValidator(df)
    validator.print_report()
    return validator.validation_report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        validate_dataset(file_path)
    else:
        print("Usage: python data_validator.py <path_to_csv>")
