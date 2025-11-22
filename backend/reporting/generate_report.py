"""
ML Model Report Generator
==========================
Generates comprehensive PDF reports with:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrices with heatmaps
- Feature importance rankings
- ROC curves and AUC scores
- Performance comparison plots
- Training summary

The report is saved as HTML (easy to view) with embedded plots.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime
import json

# ML and metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# Plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive ML model evaluation reports.
    
    What this does:
    1. Loads trained models and test data
    2. Evaluates each model's performance
    3. Generates visualizations (confusion matrix, ROC curves, feature importance)
    4. Creates HTML report with all metrics and plots
    5. Saves report to reports folder
    """
    
    def __init__(self, data_path: str, model_dir: str, report_dir: str):
        """
        Initialize report generator.
        
        Args:
            data_path: Path to cleaned dataset
            model_dir: Directory containing trained models
            report_dir: Directory to save reports
        """
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for plots
        self.plots_dir = self.report_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Style for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
        self.models = {}
        self.results = {}
        self.plots = {}
        
    def load_data(self):
        """
        Load and prepare data for evaluation.
        
        Steps:
        1. Load cleaned dataset
        2. Split into features and labels
        3. Use last 20% as test set (same as training split)
        """
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Get feature columns (exclude labels)
        label_columns = ['label', 'label_original', 'label_binary', 'label_encoded']
        self.feature_columns = [col for col in df.columns if col not in label_columns]
        
        # Use last 20% for testing (consistent with training)
        test_size = int(len(df) * 0.2)
        test_df = df.tail(test_size)
        
        # Prepare features and labels
        self.X_test = test_df[self.feature_columns].values
        self.y_test_binary = test_df['label_binary'].values
        self.y_test_multi = test_df['label_encoded'].values
        self.y_test_labels = test_df['label'].values
        
        # Store original dataframe for feature importance
        self.df = df
        
        logger.info(f"Test samples: {len(test_df)}")
        logger.info(f"Features: {len(self.feature_columns)}")
        
    def load_models(self):
        """
        Load all trained models from disk.
        
        Loads:
        - Random Forest Binary
        - Random Forest Multi-Class
        - XGBoost Binary
        - XGBoost Multi-Class
        """
        logger.info(f"Loading models from {self.model_dir}")
        
        model_files = {
            'RF_Binary': 'random_forest_binary.joblib',
            'RF_MultiClass': 'random_forest_multiclass.joblib',
            'XGB_Binary': 'xgboost_binary.joblib',
            'XGB_MultiClass': 'xgboost_multiclass.joblib'
        }
        
        for name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                logger.info(f"  Loaded: {name}")
            else:
                logger.warning(f"  Not found: {name}")
                
    def evaluate_model(self, model, X_test, y_test, model_name, is_binary=True):
        """
        Evaluate a single model and collect metrics.
        
        Metrics collected:
        - Accuracy: Overall correctness
        - Precision: Correct positive predictions / All positive predictions
        - Recall: Correct positive predictions / All actual positives
        - F1-Score: Harmonic mean of precision and recall
        - Confusion Matrix: Detailed prediction breakdown
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True labels
            model_name: Name for reporting
            is_binary: Whether this is binary or multi-class
        
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # For multi-class, use weighted average
        avg_type = 'binary' if is_binary else 'weighted'
        
        precision = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report (detailed per-class metrics)
        if is_binary:
            class_names = ['Normal', 'Attack']
        else:
            class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'class_names': class_names
        }
    
    def plot_confusion_matrix(self, cm, class_names, model_name):
        """
        Create confusion matrix heatmap.
        
        A confusion matrix shows:
        - Rows: Actual labels
        - Columns: Predicted labels
        - Diagonal: Correct predictions
        - Off-diagonal: Misclassifications
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap with annotations
        sns.heatmap(
            cm,
            annot=True,           # Show numbers in cells
            fmt='d',              # Integer format
            cmap='Blues',         # Color scheme
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path.name)
    
    def plot_roc_curve(self, model, X_test, y_test, model_name):
        """
        Plot ROC (Receiver Operating Characteristic) curve.
        
        ROC Curve shows:
        - X-axis: False Positive Rate
        - Y-axis: True Positive Rate
        - AUC (Area Under Curve): Overall model quality (higher is better)
        - Perfect classifier: AUC = 1.0
        - Random classifier: AUC = 0.5
        """
        plt.figure(figsize=(10, 8))
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]  # Probability of attack class
        else:
            # For models without probability support
            y_proba = model.predict(X_test)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{model_name}_roc_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path.name), auc
    
    def plot_feature_importance(self, model, model_name, top_n=20):
        """
        Plot feature importance for tree-based models.
        
        Feature importance shows which network traffic features
        are most important for detecting attacks.
        
        Higher importance = more useful for classification
        """
        # Get feature importances from the model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning(f"Model {model_name} doesn't support feature importance")
            return None
        
        # Create DataFrame for easy sorting
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importances
        })
        
        # Sort by importance and take top N
        feature_importance_df = feature_importance_df.sort_values(
            'Importance', ascending=False
        ).head(top_n)
        
        # Plot horizontal bar chart
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bars
        plt.barh(
            range(len(feature_importance_df)),
            feature_importance_df['Importance'],
            color='steelblue',
            edgecolor='navy',
            alpha=0.8
        )
        
        # Add feature names
        plt.yticks(
            range(len(feature_importance_df)),
            feature_importance_df['Feature'],
            fontsize=10
        )
        
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {model_name}', 
                  fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{model_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path.name), feature_importance_df
    
    def plot_model_comparison(self):
        """
        Create comparison bar chart of all models.
        
        Compares models on:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        """
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.results.keys())
        
        # Extract metrics for each model
        metric_data = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            for metric in metrics:
                metric_data[metric].append(self.results[model_name][metric])
        
        # Create subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Create bar chart
            bars = ax.bar(
                range(len(model_names)),
                metric_data[metric],
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=1.5
            )
            
            # Add value labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{metric_data[metric][i]:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )
            
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=15, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path.name)
    
    def generate_html_report(self):
        """
        Generate comprehensive HTML report with all metrics and plots.
        
        Report includes:
        1. Executive Summary
        2. Model Performance Metrics
        3. Confusion Matrices
        4. ROC Curves
        5. Feature Importance
        6. Model Comparison
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start HTML document
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Model Evaluation Report - SurakshaNET</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .timestamp {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 10px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h3 {{
            color: #764ba2;
            margin-top: 25px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card .label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-top: 10px;
        }}
        .plot-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .good {{ color: #2ca02c; font-weight: bold; }}
        .medium {{ color: #ff7f0e; font-weight: bold; }}
        .poor {{ color: #d62728; font-weight: bold; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Network Intrusion Detection</h1>
        <h2>ML Model Evaluation Report</h2>
        <div class="timestamp">Generated: {timestamp}</div>
    </div>
    
    <div class="section">
        <h2>📊 Executive Summary</h2>
        <p>This report provides a comprehensive evaluation of machine learning models trained for network intrusion detection. 
        The models were evaluated on NSL-KDD dataset with {len(self.y_test_binary)} test samples containing 
        {len(self.feature_columns)} network traffic features.</p>
        
        <h3>Models Evaluated:</h3>
        <ul>
            <li><strong>Random Forest Binary:</strong> Binary classification (Normal vs Attack)</li>
            <li><strong>Random Forest Multi-Class:</strong> 5-class classification (Normal, DoS, Probe, R2L, U2R)</li>
            <li><strong>XGBoost Binary:</strong> Binary classification with gradient boosting</li>
            <li><strong>XGBoost Multi-Class:</strong> 5-class classification with gradient boosting</li>
        </ul>
    </div>
"""
        
        # Add results for each model
        for model_name, results in self.results.items():
            accuracy_class = 'good' if results['accuracy'] > 0.8 else ('medium' if results['accuracy'] > 0.6 else 'poor')
            
            html += f"""
    <div class="section">
        <h2>📈 {model_name.replace('_', ' ')}</h2>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Accuracy</div>
                <div class="value {accuracy_class}">{results['accuracy']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Precision</div>
                <div class="value">{results['precision']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Recall</div>
                <div class="value">{results['recall']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">F1-Score</div>
                <div class="value">{results['f1_score']:.4f}</div>
            </div>
        </div>
        
        <h3>Classification Report</h3>
        <table>
            <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
"""
            
            # Add per-class metrics
            for class_name in results['class_names']:
                class_metrics = results['classification_report'][class_name]
                html += f"""
            <tr>
                <td>{class_name}</td>
                <td>{class_metrics['precision']:.4f}</td>
                <td>{class_metrics['recall']:.4f}</td>
                <td>{class_metrics['f1-score']:.4f}</td>
                <td>{int(class_metrics['support'])}</td>
            </tr>
"""
            
            html += """
        </table>
"""
            
            # Add plots
            if 'confusion_matrix_plot' in results:
                html += f"""
        <h3>Confusion Matrix</h3>
        <div class="plot-container">
            <img src="plots/{results['confusion_matrix_plot']}" alt="Confusion Matrix">
        </div>
"""
            
            if 'roc_curve_plot' in results:
                html += f"""
        <h3>ROC Curve (AUC = {results['auc']:.4f})</h3>
        <div class="plot-container">
            <img src="plots/{results['roc_curve_plot']}" alt="ROC Curve">
        </div>
"""
            
            if 'feature_importance_plot' in results:
                html += f"""
        <h3>Feature Importance (Top 20)</h3>
        <div class="plot-container">
            <img src="plots/{results['feature_importance_plot']}" alt="Feature Importance">
        </div>
"""
            
            html += """
    </div>
"""
        
        # Add model comparison
        if 'comparison_plot' in self.plots:
            html += f"""
    <div class="section">
        <h2>📊 Model Comparison</h2>
        <div class="plot-container">
            <img src="plots/{self.plots['comparison_plot']}" alt="Model Comparison">
        </div>
    </div>
"""
        
        # Footer
        html += """
    <div class="footer">
        <p>🛡️ SurakshaNET - Network Intrusion Detection System</p>
        <p>Report generated automatically by ML Model Evaluation Pipeline</p>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        report_path = self.report_dir / f"model_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to: {report_path}")
        return report_path
    
    def generate_report(self):
        """
        Main method to generate complete report.
        
        Workflow:
        1. Load data and models
        2. Evaluate each model
        3. Generate all plots
        4. Create HTML report
        """
        logger.info("="*70)
        logger.info("GENERATING MODEL EVALUATION REPORT")
        logger.info("="*70)
        
        # Step 1: Load data and models
        self.load_data()
        self.load_models()
        
        # Step 2: Evaluate binary models
        for model_name in ['RF_Binary', 'XGB_Binary']:
            if model_name in self.models:
                results = self.evaluate_model(
                    self.models[model_name],
                    self.X_test,
                    self.y_test_binary,
                    model_name,
                    is_binary=True
                )
                
                # Generate plots
                results['confusion_matrix_plot'] = self.plot_confusion_matrix(
                    results['confusion_matrix'],
                    results['class_names'],
                    model_name
                )
                
                roc_plot, auc = self.plot_roc_curve(
                    self.models[model_name],
                    self.X_test,
                    self.y_test_binary,
                    model_name
                )
                results['roc_curve_plot'] = roc_plot
                results['auc'] = auc
                
                feat_plot, feat_df = self.plot_feature_importance(
                    self.models[model_name],
                    model_name
                )
                if feat_plot:
                    results['feature_importance_plot'] = feat_plot
                    results['feature_importance_data'] = feat_df
                
                self.results[model_name] = results
        
        # Step 3: Evaluate multi-class models
        for model_name in ['RF_MultiClass', 'XGB_MultiClass']:
            if model_name in self.models:
                results = self.evaluate_model(
                    self.models[model_name],
                    self.X_test,
                    self.y_test_multi,
                    model_name,
                    is_binary=False
                )
                
                # Generate plots
                results['confusion_matrix_plot'] = self.plot_confusion_matrix(
                    results['confusion_matrix'],
                    results['class_names'],
                    model_name
                )
                
                feat_plot, feat_df = self.plot_feature_importance(
                    self.models[model_name],
                    model_name
                )
                if feat_plot:
                    results['feature_importance_plot'] = feat_plot
                    results['feature_importance_data'] = feat_df
                
                self.results[model_name] = results
        
        # Step 4: Generate comparison plot
        if len(self.results) > 1:
            comparison_plot = self.plot_model_comparison()
            self.plots['comparison_plot'] = comparison_plot
        
        # Step 5: Generate HTML report
        report_path = self.generate_html_report()
        
        logger.info("="*70)
        logger.info("REPORT GENERATION COMPLETE!")
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Plots saved to: {self.plots_dir}")
        logger.info("="*70)
        
        return report_path


def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "dataset" / "cleaned" / "nsl_kdd_cleaned.csv"
    model_dir = project_root / "backend" / "models" / "trained"
    report_dir = project_root / "reports"
    
    # Check if models exist
    if not model_dir.exists() or not any(model_dir.glob("*.joblib")):
        logger.error("No trained models found. Please run train_models.py first.")
        return
    
    # Generate report
    generator = ReportGenerator(
        data_path=str(data_path),
        model_dir=str(model_dir),
        report_dir=str(report_dir)
    )
    
    report_path = generator.generate_report()
    
    logger.info("\n" + "="*70)
    logger.info("To view the report, open:")
    logger.info(f"  {report_path}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
