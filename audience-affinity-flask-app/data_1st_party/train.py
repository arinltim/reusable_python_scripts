import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("Loading rich_audience_data.csv...")
df = pd.read_csv('rich_audience_data.csv')

# --- 1. Train and Save the Clustering Model ---
print("Training Clustering model...")
# Features for clustering (behavioral and demographic)
clustering_features = [
    'time_on_social_media_min', 'time_on_streaming_video_min',
    'time_on_podcasts_min', 'monthly_conversions',
    'age_group', 'income_bracket'
]
X_cluster = df[clustering_features]

# Define preprocessing for numeric and categorical features
numeric_features = [f for f in X_cluster.columns if df[f].dtype in ['int64', 'float64']]
categorical_features = [f for f in X_cluster.columns if df[f].dtype == 'object']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline with preprocessing and K-Means clustering (e.g., finding 5 segments)
clustering_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clusterer', KMeans(n_clusters=5, random_state=42, n_init=10))
])

# Train the clustering model
clustering_pipeline.fit(X_cluster)
joblib.dump(clustering_pipeline, 'clustering_pipeline.pkl')
print("SUCCESS: Clustering model trained and saved to 'clustering_pipeline.pkl'")


# --- 2. Train and Save the Churn Prediction Model ---
print("\nTraining Churn Prediction model...")
# Features for churn prediction
churn_features = ['avg_monthly_spend', 'last_seen_days_ago', 'support_tickets_raised']
X_churn = df[churn_features]
y_churn = df['churned_in_last_90_days']

# Simple pipeline with just scaling
churn_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Train the model
churn_pipeline.fit(X_churn, y_churn)
joblib.dump(churn_pipeline, 'churn_model.pkl')
print("SUCCESS: Churn prediction model trained and saved to 'churn_model.pkl'")