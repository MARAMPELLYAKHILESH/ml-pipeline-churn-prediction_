"""
Model Training with MLflow Experiment Tracking
Trains multiple models and tracks experiments
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, experiment_name="churn_prediction"):
        """Initialize MLflow experiment"""
        mlflow.set_experiment(experiment_name)
        self.models = {}
        self.results = {}
        
    def load_data(self, train_path, test_path):
        """Load preprocessed training and test data"""
        print(f"Loading data from {train_path} and {test_path}")
        
        # Load pickle files
        if train_path.endswith('.pkl'):
            train_data = pd.read_pickle(train_path)
            test_data = pd.read_pickle(test_path)
            
            print(f"Train data type: {type(train_data)}")
            
            # Handle different formats
            if isinstance(train_data, dict):
                print(f"Dict keys: {list(train_data.keys())}")
                # Try different key combinations
                if 'X' in train_data and 'y' in train_data:
                    self.X_train = train_data['X']
                    self.y_train = train_data['y']
                    self.X_test = test_data['X']
                    self.y_test = test_data['y']
                elif 'features' in train_data and 'target' in train_data:
                    self.X_train = train_data['features']
                    self.y_train = train_data['target']
                    self.X_test = test_data['features']
                    self.y_test = test_data['target']
                elif 'texts' in train_data and 'labels' in train_data:
                    # Handle texts/labels format
                    self.X_train = train_data['texts']
                    self.y_train = train_data['labels']
                    self.X_test = test_data['texts']
                    self.y_test = test_data['labels']
                else:
                    # Unknown dict format - print and raise error
                    print(f"Unknown dict format with keys: {list(train_data.keys())}")
                    raise ValueError(f"Unexpected dict keys: {list(train_data.keys())}")
            elif isinstance(train_data, tuple):
                # Handle tuple format (X, y)
                print("Data is in tuple format (X, y)")
                self.X_train, self.y_train = train_data
                self.X_test, self.y_test = test_data
            else:
                # Assume it's a DataFrame
                print("Data is DataFrame format")
                self.X_train = train_data.drop('churn', axis=1)
                self.y_train = train_data['churn']
                self.X_test = test_data.drop('churn', axis=1)
                self.y_test = test_data['churn']
        else:
            # CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            self.X_train = train_df.drop('churn', axis=1)
            self.y_train = train_df['churn']
            self.X_test = test_df.drop('churn', axis=1)
            self.y_test = test_df['churn']
        
        print(f"Train shape: {self.X_train.shape}")
        print(f"Test shape: {self.X_test.shape}")
        print(f"Target train shape: {self.y_train.shape}")
        print(f"Target test shape: {self.y_test.shape}")
        print(f"Class distribution:\n{self.y_train.value_counts()}")
        
    def train_model(self, model_name, model, params=None):
        """Train a model and log with MLflow"""
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print(f"{'='*50}")
            
            # Hyperparameter tuning if params provided
            if params:
                print("Performing hyperparameter tuning...")
                grid_search = GridSearchCV(
                    model, params, cv=5, scoring='f1', 
                    n_jobs=-1, verbose=1
                )
                grid_search.fit(self.X_train, self.y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                mlflow.log_params(best_params)
                print(f"Best parameters: {best_params}")
            else:
                best_model = model
                best_model.fit(self.X_train, self.y_train)
                mlflow.log_params(model.get_params())
            
            # Cross-validation
            cv_scores = cross_val_score(
                best_model, self.X_train, self.y_train, 
                cv=5, scoring='f1'
            )
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()
            
            mlflow.log_metric("cv_f1_mean", mean_cv_score)
            mlflow.log_metric("cv_f1_std", std_cv_score)
            print(f"Cross-validation F1: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
            
            # Predictions
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1] \
                           if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_pred, y_pred_proba)
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
                print(f"{metric_name}: {value:.4f}")
            
            # Feature importance (if available)
            if hasattr(best_model, 'feature_importances_'):
                self._log_feature_importance(best_model, model_name)
            
            # Confusion matrix
            self._log_confusion_matrix(y_pred, model_name)
            
            # ROC curve
            if y_pred_proba is not None:
                self._log_roc_curve(y_pred_proba, model_name)
            
            # Log classification report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")
            
            # Save and log model
            model_path = f"models/{model_name}_model.pkl"
            joblib.dump(best_model, model_path)
            mlflow.sklearn.log_model(best_model, "model")
            mlflow.log_artifact(model_path)
            
            # Store results
            self.models[model_name] = best_model
            self.results[model_name] = metrics
            
            print(f"\n✓ {model_name} training completed!")
            
            return best_model, metrics
    
    def _calculate_metrics(self, y_pred, y_pred_proba=None):
        """Calculate all evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
        
        return metrics
    
    def _log_feature_importance(self, model, model_name):
        """Log feature importance plot"""
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title(f'{model_name} - Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig(f'artifacts/{model_name}_feature_importance.png')
        mlflow.log_artifact(f'artifacts/{model_name}_feature_importance.png')
        plt.close()
        
        # Log as JSON
        importance_dict = importance_df.to_dict('records')
        mlflow.log_dict(importance_dict, "feature_importance.json")
    
    def _log_confusion_matrix(self, y_pred, model_name):
        """Log confusion matrix plot"""
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'artifacts/{model_name}_confusion_matrix.png')
        mlflow.log_artifact(f'artifacts/{model_name}_confusion_matrix.png')
        plt.close()
    
    def _log_roc_curve(self, y_pred_proba, model_name):
        """Log ROC curve plot"""
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'artifacts/{model_name}_roc_curve.png')
        mlflow.log_artifact(f'artifacts/{model_name}_roc_curve.png')
        plt.close()
    
    def train_all_models(self):
        """Train all models and compare"""
        
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr_params = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        self.train_model('LogisticRegression', lr, lr_params)
        
        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        self.train_model('RandomForest', rf, rf_params)
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(random_state=42)
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
        self.train_model('GradientBoosting', gb, gb_params)
        
        # Compare results
        self._compare_models()
    
    def _compare_models(self):
        """Compare all trained models"""
        print(f"\n{'='*50}")
        print("MODEL COMPARISON")
        print(f"{'='*50}")
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        print(comparison_df.to_string())
        
        # Save comparison
        comparison_df.to_csv('artifacts/model_comparison.csv')
        
        # Log to MLflow
        with mlflow.start_run(run_name="model_comparison"):
            mlflow.log_artifact('artifacts/model_comparison.csv')
            
            # Plot comparison
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]
                comparison_df[metric].plot(kind='barh', ax=ax)
                ax.set_title(f'{metric.capitalize()} Comparison')
                ax.set_xlabel('Score')
            
            plt.tight_layout()
            plt.savefig('artifacts/model_comparison.png')
            mlflow.log_artifact('artifacts/model_comparison.png')
            plt.close()
        
        # Select best model
        best_model_name = comparison_df['f1_score'].idxmax()
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"F1 Score: {comparison_df.loc[best_model_name, 'f1_score']:.4f}")
        
        # Save best model
        joblib.dump(best_model, 'models/best_model.pkl')
        
        return best_model_name, best_model

def main():
    parser = argparse.ArgumentParser(description='Train ML models with MLflow tracking')
    parser.add_argument('--train', type=str, default='data/processed/train.csv',
                       help='Path to training data')
    parser.add_argument('--test', type=str, default='data/processed/test.csv',
                       help='Path to test data')
    parser.add_argument('--experiment', type=str, default='churn_prediction',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Create artifacts directory
    import os
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(experiment_name=args.experiment)
    
    # Load data
    trainer.load_data(args.train, args.test)
    
    # Train all models
    trainer.train_all_models()
    
    print("\n✓ Training pipeline completed!")
    print("View experiments: mlflow ui")

if __name__ == "__main__":
    main()