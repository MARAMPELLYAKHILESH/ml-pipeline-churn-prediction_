"""
Model Evaluation Script
Evaluates trained model and generates comprehensive metrics
"""

import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import json
import os

def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data(data_path):
    """Load test data"""
    print(f"Loading test data from {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data['texts'], data['labels']

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    }
    
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
        except:
            pass
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_pred_proba, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to {save_path}")

def generate_classification_report(y_true, y_pred, save_path):
    """Generate and save classification report"""
    report = classification_report(y_true, y_pred, 
                                  target_names=['No Churn', 'Churn'],
                                  output_dict=True)
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print to console
    print("\nClassification Report:")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
    print("="*50)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate ML model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output', type=str, default='evaluation_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model and data
    model = load_model(args.model)
    X_test, y_test = load_data(args.data)
    
    print(f"\nEvaluating model on {len(X_test)} test samples...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("="*50)
    
    # Save metrics
    metrics_path = os.path.join(args.output, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, cm_path)
    
    # Plot ROC curve if probabilities available
    if y_pred_proba is not None:
        roc_path = os.path.join(args.output, 'roc_curve.png')
        plot_roc_curve(y_test, y_pred_proba, roc_path)
    
    # Generate classification report
    report_path = os.path.join(args.output, 'classification_report.json')
    generate_classification_report(y_test, y_pred, report_path)
    
    print("\n✓ Evaluation completed successfully!")
    print(f"All results saved to: {args.output}")

if __name__ == "__main__":
    main()