"""
Data Preprocessing for ML Pipeline
Handles data cleaning, feature engineering, and train-test split
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle
import os
import json

class DataPreprocessor:
    """Handles all data preprocessing steps"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        print("\nHandling missing values...")
        
        # Numeric columns - fill with median
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  Filled {col} with median")
        
        # Categorical columns - fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  Filled {col} with mode")
        
        return df
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # One-hot encoding for categorical variables
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
                print(f"  Encoded {col} → {len(dummies.columns)} columns")
        
        return df_encoded
    
    def engineer_features(self, df):
        """Create new features"""
        print("\nEngineering features...")
        
        # Example feature engineering (customize based on your data)
        if 'tenure' in df.columns and 'monthly_charges' in df.columns:
            df['total_value'] = df['tenure'] * df['monthly_charges']
            print("  Created total_value feature")
        
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                                       labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
            # Convert to dummies
            tenure_dummies = pd.get_dummies(df['tenure_group'], prefix='tenure_group')
            df = pd.concat([df, tenure_dummies], axis=1)
            df.drop('tenure_group', axis=1, inplace=True)
            print("  Created tenure_group features")
        
        return df
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        print("\nScaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def balance_classes(self, X, y):
        """Balance classes using SMOTE"""
        print("\nBalancing classes with SMOTE...")
        
        print(f"Before SMOTE: {dict(pd.Series(y).value_counts())}")
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"After SMOTE: {dict(pd.Series(y_balanced).value_counts())}")
        
        return X_balanced, y_balanced
    
    def preprocess(self, filepath, target_column, test_size=0.2, balance=True):
        """Complete preprocessing pipeline"""
        
        # Load data
        df = self.load_data(filepath)
        
        # Check for target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Separate features and target
        y = df[target_column]
        X = df.drop(target_column, axis=1)
        
        # DROP customer_id if it exists (it's just an identifier, not a feature!)
        if 'customer_id' in X.columns:
            print(f"\n⚠️  Dropping 'customer_id' column (identifier, not a feature)")
            X = X.drop('customer_id', axis=1)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical variables
        if categorical_cols:
            X = self.encode_categorical(X, categorical_cols)
        
        # Engineer features
        X = self.engineer_features(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode target if categorical
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Balance training data if requested
        if balance:
            X_train, y_train = self.balance_classes(X_train, y_train)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Convert back to DataFrame for easy handling
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_df = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        return X_train_df, X_test_df, y_train, y_test
    
    def save_preprocessor(self, filepath):
        """Save preprocessor objects"""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        print(f"\nPreprocessor saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess data for ML pipeline')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--target', type=str, default='churn', help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--balance', action='store_true', help='Balance classes with SMOTE')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        args.input, args.target, args.test_size, args.balance
    )
    
    # Save processed data
    train_data = {'texts': X_train, 'labels': y_train}
    test_data = {'texts': X_test, 'labels': y_test}
    
    with open(os.path.join(args.output, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(args.output, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    # Save preprocessor
    preprocessor.save_preprocessor(os.path.join(args.output, 'preprocessor.pkl'))
    
    # Save metadata
    metadata = {
        'n_features': len(preprocessor.feature_names),
        'feature_names': preprocessor.feature_names,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'target_column': args.target
    }
    
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("\n" + "="*50)
    print("Preprocessing completed successfully!")
    print(f"Training data: {os.path.join(args.output, 'train.pkl')}")
    print(f"Test data: {os.path.join(args.output, 'test.pkl')}")
    print(f"Preprocessor: {os.path.join(args.output, 'preprocessor.pkl')}")
    print("="*50)

if __name__ == "__main__":
    main()