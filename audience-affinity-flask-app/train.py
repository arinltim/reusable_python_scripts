import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("Loading enriched audience_data.csv...")
df = pd.read_csv('rich_audience_data.csv')

# --- 1. Train and Save the Enriched Clustering Model ---
print("\n--- Training Enriched Clustering Model ---")
# Features for clustering now include the new 2nd and 3rd party data
clustering_features = [
    'time_on_social_media_min', 'time_on_streaming_video_min',
    'time_on_podcasts_min', 'monthly_conversions',
    'age_group', 'income_bracket',
    'partner_purchase_categories', 'traveler_type',
    'household_size', 'education_level',
    'homeowner_status', 'offline_purchase_intent'
]
X_cluster = df[clustering_features]

# Define preprocessing for numeric and categorical features
numeric_features_cluster = [f for f in X_cluster.columns if df[f].dtype in ['int64', 'float64']]
categorical_features_cluster = [f for f in X_cluster.columns if df[f].dtype == 'object']

# This preprocessor handles the mix of data types
preprocessor_cluster = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features_cluster),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_cluster)
    ])

# Create a pipeline with preprocessing and K-Means clustering
clustering_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_cluster),
    ('clusterer', KMeans(n_clusters=5, random_state=42, n_init=10))
])

# Train the new clustering model
clustering_pipeline.fit(X_cluster)
joblib.dump(clustering_pipeline, 'clustering_pipeline.pkl')
print("SUCCESS: Enriched clustering model saved to 'clustering_pipeline.pkl'")


# --- 2. Train and Save the Enriched Churn Prediction Model ---
print("\n--- Training Enriched Churn Prediction Model ---")
# Added the most impactful new features to the churn model
churn_features = [
    'avg_monthly_spend', 'last_seen_days_ago', 'support_tickets_raised',
    'household_size', 'education_level', 'homeowner_status', 'offline_purchase_intent'
]
X_churn = df[churn_features]
y_churn = df['churned_in_last_90_days']

# Define preprocessing specifically for the churn features
numeric_features_churn = [f for f in X_churn.columns if df[f].dtype in ['int64', 'float64']]
categorical_features_churn = [f for f in X_churn.columns if df[f].dtype == 'object']

preprocessor_churn = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features_churn),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_churn)
    ])

# Create an updated pipeline for churn prediction
churn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_churn),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train the new churn model
churn_pipeline.fit(X_churn, y_churn)

joblib.dump(churn_pipeline, 'churn_model.pkl')
print("SUCCESS: Enriched churn prediction model saved to 'churn_model.pkl'")