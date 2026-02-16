"""
Sample Data Generator for ML Pipeline
Creates synthetic customer churn data for demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(n_samples=10000, output_path='data/raw/sample_data.csv'):
    """
    Generate synthetic customer churn data
    
    Args:
        n_samples: Number of samples to generate
        output_path: Where to save the CSV file
    """
    
    np.random.seed(42)
    
    print(f"Generating {n_samples} sample records...")
    
    # Generate features
    data = {
        'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples).round(2),
        'total_charges': None,  # Will calculate
        'contract_type': np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], 
            n_samples, 
            p=[0.5, 0.3, 0.2]
        ),
        'payment_method': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
            n_samples,
            p=[0.4, 0.2, 0.2, 0.2]
        ),
        'internet_service': np.random.choice(
            ['DSL', 'Fiber optic', 'No'],
            n_samples,
            p=[0.4, 0.4, 0.2]
        ),
        'online_security': np.random.choice([True, False], n_samples),
        'tech_support': np.random.choice([True, False], n_samples),
        'streaming_tv': np.random.choice([True, False], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate total charges
    df['total_charges'] = (df['tenure'] * df['monthly_charges']).round(2)
    
    # Generate churn based on realistic patterns
    churn_prob = 0.3  # 30% base churn rate
    
    # Factors that increase churn probability
    churn_factors = (
        (df['contract_type'] == 'Month-to-month') * 0.3 +
        (df['payment_method'] == 'Electronic check') * 0.2 +
        (df['tenure'] < 12) * 0.2 +
        (df['monthly_charges'] > 80) * 0.15 +
        (~df['online_security']) * 0.1 +
        (~df['tech_support']) * 0.1
    )
    
    # Generate churn with influenced probability
    churn_prob_individual = np.clip(churn_prob + churn_factors, 0, 0.9)
    df['churn'] = (np.random.random(n_samples) < churn_prob_individual).astype(int)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Sample data saved to {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Churn rate: {df['churn'].mean():.2%}")
    print(f"  Features: {len(df.columns)}")
    print(f"\nFeature Summary:")
    print(df.describe())
    print(f"\nChurn Distribution:")
    print(df['churn'].value_counts())
    
    return df

def generate_test_cases(output_path='data/raw/test_cases.csv'):
    """Generate specific test cases for prediction"""
    
    test_cases = [
        {
            'customer_id': 'TEST_001',
            'age': 35,
            'tenure': 24,
            'monthly_charges': 65.5,
            'total_charges': 1572.0,
            'contract_type': 'Two year',
            'payment_method': 'Credit card',
            'internet_service': 'Fiber optic',
            'online_security': True,
            'tech_support': True,
            'streaming_tv': True,
            'expected_churn': 0
        },
        {
            'customer_id': 'TEST_002',
            'age': 42,
            'tenure': 3,
            'monthly_charges': 95.0,
            'total_charges': 285.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic',
            'online_security': False,
            'tech_support': False,
            'streaming_tv': False,
            'expected_churn': 1
        },
        {
            'customer_id': 'TEST_003',
            'age': 28,
            'tenure': 48,
            'monthly_charges': 45.0,
            'total_charges': 2160.0,
            'contract_type': 'One year',
            'payment_method': 'Bank transfer',
            'internet_service': 'DSL',
            'online_security': True,
            'tech_support': False,
            'streaming_tv': True,
            'expected_churn': 0
        }
    ]
    
    df = pd.DataFrame(test_cases)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Test cases saved to {output_path}")
    return df

if __name__ == "__main__":
    # Generate sample training data
    sample_data = generate_sample_data(n_samples=10000)
    
    # Generate test cases
    test_cases = generate_test_cases()
    
    print("\n" + "="*50)
    print("Sample data generation complete!")
    print("You can now run the ML pipeline:")
    print("  1. python src/data_preprocessing.py --input data/raw/sample_data.csv")
    print("  2. python src/model_training.py --train data/processed/train.pkl")
    print("  3. uvicorn api.main:app --reload")
    print("="*50)