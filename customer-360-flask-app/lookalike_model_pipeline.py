import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib # Import joblib for saving the model
import json   # Import json for saving column names

def generate_synthetic_customer_data(num_customers=10000):
    """Generates a synthetic dataset simulating customer profiles and behaviors."""
    print(f"Step 1: Generating synthetic data for {num_customers} customers...")
    locations = ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ', 'Austin, TX', 'Miami, FL']
    interests = ['Tech Gadgets', 'Sustainable Living', 'Fitness & Wellness', 'Travel & Outdoors', 'Finance & Investing']
    last_purchase_category = ['Electronics', 'Home Goods', 'Apparel', 'Books', 'Health']
    data = {
        'age': np.random.randint(18, 65, size=num_customers),
        'location': np.random.choice(locations, size=num_customers),
        'primary_interest': np.random.choice(interests, size=num_customers),
        'last_purchase_category': np.random.choice(last_purchase_category, size=num_customers),
        'total_spend': np.random.gamma(2, 150, size=num_customers),
        'purchase_frequency': np.random.randint(1, 50, size=num_customers),
        'avg_session_duration_min': np.random.uniform(1, 30, size=num_customers),
        'ads_clicked_last_30d': np.random.randint(0, 25, size=num_customers),
    }
    df = pd.DataFrame(data)
    # Add a customer_id column for clarity
    df.insert(0, 'customer_id', [f'cust_{i}' for i in range(num_customers)])
    print("✓ Synthetic data generated.\n")
    return df

def define_seed_audience(df):
    """Applies business rules to identify and label 'high-value' customers."""
    print("Step 2: Defining the 'High-Value' seed audience...")
    spend_threshold = df['total_spend'].quantile(0.80)
    frequency_threshold = 10
    df['is_high_value'] = 0
    df.loc[
        (df['total_spend'] >= spend_threshold) & (df['purchase_frequency'] >= frequency_threshold),
        'is_high_value'
    ] = 1
    seed_count = df['is_high_value'].sum()
    print(f"✓ Identified {seed_count} high-value customers as the seed audience.\n")
    return df

def train_and_save_model(df):
    """
    Preprocesses data, trains a classification model, and saves it to a file.
    """
    print("Step 3: Preprocessing data and training the look-alike model...")
    # Balance the classes for more stable training
    seed_df = df[df['is_high_value'] == 1]
    non_seed_df = df[df['is_high_value'] == 0].sample(n=len(seed_df), random_state=42)
    training_data = pd.concat([seed_df, non_seed_df])
    X = training_data.drop(columns=['customer_id', 'is_high_value'])
    y = training_data['is_high_value']

    # Save the column order and types for later use in the UI
    model_columns = X.columns.tolist()
    with open('model_columns.json', 'w') as f:
        json.dump(model_columns, f)
    print("✓ Model columns saved to model_columns.json")

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])

    model_pipeline.fit(X, y)
    print("✓ Model training complete.")

    # Save the entire pipeline to a file
    joblib.dump(model_pipeline, 'lookalike_model.pkl')
    print("✓ Model pipeline saved to lookalike_model.pkl\n")
    return model_pipeline

if __name__ == "__main__":
    raw_df = generate_synthetic_customer_data()
    labeled_df = define_seed_audience(raw_df)
    train_and_save_model(labeled_df)
    print("✅ Pipeline finished. You can now run the Flask app.")