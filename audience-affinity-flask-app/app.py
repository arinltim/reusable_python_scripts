import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from flask import Flask, render_template, request, jsonify
import plotly.express as px
import plotly
from sklearn.decomposition import PCA
import joblib
import google.generativeai as genai

# --- 1. Load Assets & Configure Gemini ---
app = Flask(__name__)
print("Loading models and data...")
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    print("Gemini API configured successfully.")
except (TypeError, ValueError):
    print("CRITICAL: GEMINI_API_KEY environment variable not set. GenAI features will be disabled.")
    genai = None

# Assumes you have run the new generate_data.py script
df = pd.read_csv('rich_audience_data.csv')
# These are the new, enriched models from the updated train.py
clustering_pipeline = joblib.load('clustering_pipeline.pkl')
churn_model = joblib.load('churn_model.pkl')

# --- ENHANCEMENT: Update feature lists to include new 2nd and 3rd party data ---
FEATURES_FOR_CLUSTERING = [
    'time_on_social_media_min', 'time_on_streaming_video_min',
    'time_on_podcasts_min', 'monthly_conversions',
    'age_group', 'income_bracket',
    'partner_purchase_categories', 'traveler_type',
    'household_size', 'education_level',
    'homeowner_status', 'offline_purchase_intent'
]
CHURN_FEATURES = [
    'avg_monthly_spend', 'last_seen_days_ago', 'support_tickets_raised',
    'household_size', 'education_level', 'homeowner_status', 'offline_purchase_intent'
]
# ---------------------------------------------------------------------------------

CHANNEL_COLS = ['time_on_social_media_min', 'time_on_streaming_video_min', 'time_on_podcasts_min']
CHANNEL_NAMES = {'time_on_social_media_min': 'Social Media', 'time_on_streaming_video_min': 'Streaming Video', 'time_on_podcasts_min': 'Podcasts'}


# --- 2. Augment DataFrame ---
# The predict and transform steps now use the updated, richer feature lists
df['cluster'] = clustering_pipeline.predict(df[FEATURES_FOR_CLUSTERING])
df['cluster'] = df['cluster'].astype(str)
df['churn_risk_proba'] = churn_model.predict_proba(df[CHURN_FEATURES])[:, 1]
df['customer_since_date'] = pd.to_datetime(df['customer_since_date'])
now = datetime.now()
df['tenure_months'] = (now.year - df['customer_since_date'].dt.year) * 12 + (now.month - df['customer_since_date'].dt.month)
df['clv'] = df['avg_monthly_spend'] * df['tenure_months']
preprocessor = clustering_pipeline.named_steps['preprocessor']
X_processed = preprocessor.transform(df[FEATURES_FOR_CLUSTERING])
pca = PCA(n_components=2, random_state=42)
df[['pca1', 'pca2']] = pca.fit_transform(X_processed)
print("Data processing complete using enriched features.")


# --- 3. GenAI Helper Functions (Unchanged) ---
def generate_persona_from_stats(stats):
    if not genai: return {"segment_name": "GenAI Disabled", "persona_summary": "API key not configured."}
    llm = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = (
        f"You are a senior market research analyst. Based on the following statistics for a customer segment, "
        "create a marketing persona. Respond in a valid JSON format with two keys: \"segment_name\" (a catchy name) "
        "and \"persona_summary\" (a paragraph describing their likely motivations and values).\n\n"
        f"Statistics: {json.dumps(stats)}\n\nJSON Response:"
    )
    try:
        response = llm.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        return {"segment_name": "Analysis Failed", "persona_summary": f"Error: {e}"}

def generate_strategy_from_stats(stats):
    if not genai: return ["GenAI is not configured. Please set your API key."]
    llm = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = (
        f"You are a strategic marketing consultant. For a customer segment with the following profile, "
        "provide 2-3 bullet points of immediate, actionable strategic advice. Focus on how to maximize value "
        "and mitigate risk. Respond only with a valid JSON array of strings.\n\n"
        f"Profile: {json.dumps(stats)}\n\nJSON Response:"
    )
    try:
        response = llm.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        return [f"Error generating strategy: {e}"]

def generate_angles_from_persona(persona_summary):
    if not genai: return {"error": "GenAI not configured"}
    llm = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = (
        f"You are a creative director. For the audience persona described as: '{persona_summary}', "
        "brainstorm three distinct creative angles for a marketing campaign. For each angle, provide a unique "
        "'title' and a short 'description' of the core message. "
        "Respond in a valid JSON format like: {\"angles\": [{\"title\": \"...\", \"description\": \"...\"}]}"
    )
    # print(prompt)
    try:
        response = llm.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        return {"error": str(e)}

# --- 4. Plotting Function (Unchanged) ---
def create_scatter_plot(dataframe):
    sorted_clusters = sorted(dataframe['cluster'].unique())
    fig = px.scatter(
        dataframe, x='pca1', y='pca2', color='cluster',
        category_orders={"cluster": sorted_clusters},
        hover_name='cluster',
        custom_data=['user_id', 'clv'],
        labels={'pca1': 'Engagement Style', 'pca2': 'Behavioral Profile', 'cluster': 'Segment'}
    )
    fig.update_traces(hovertemplate="<b>Cluster %{hovertext}</b><br>User ID: %{customdata[0]}<br>CLV: $%{customdata[1]:,.2f}<extra></extra>")
    fig.update_layout(
        margin=dict(t=60, b=80, l=90, r=40),
        title_text='Audience Segments by Behavior and Value', title_font_size=20, title_x=0.5,
        legend_title_text='Segments',
        xaxis_showticklabels=True, yaxis_showticklabels=True,
        annotations=[
            {'x': 0, 'y': -0.18, 'xref': 'paper', 'yref': 'paper', 'text': '← Video/Social Focus', 'showarrow': False, 'font': dict(size=11, color='grey'), 'xanchor': 'left'},
            {'x': 1, 'y': -0.18, 'xref': 'paper', 'yref': 'paper', 'text': 'Podcast/Audio Focus →', 'showarrow': False, 'font': dict(size=11, color='grey'), 'xanchor': 'right'},
            {
                'x': 0.02, 'y': 0.98, 'xref': 'paper', 'yref': 'paper', 'text': '↑ Frequent Buyers',
                'showarrow': False, 'font': dict(size=11, color='grey'),
                'xanchor': 'left', 'yanchor': 'top'
            },
            {
                'x': 0.02, 'y': 0.02, 'xref': 'paper', 'yref': 'paper', 'text': '↓ Occasional Buyers',
                'showarrow': False, 'font': dict(size=11, color='grey'),
                'xanchor': 'left', 'yanchor': 'bottom'
            }
        ]
    )
    return fig

# --- 5. Flask Routes ---
@app.route('/')
def index():
    fig = create_scatter_plot(df)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', graphJSON=graphJSON, clusters=sorted(df['cluster'].unique()))

@app.route('/get_initial_cluster_data/<cluster_id>')
def get_initial_cluster_data(cluster_id):
    if cluster_id not in df['cluster'].unique():
        return jsonify({"error": "Cluster not found"}), 404

    cluster_df = df[df['cluster'] == cluster_id]

    overall_channel_avg = df[CHANNEL_COLS].mean()
    cluster_channel_avg = cluster_df[CHANNEL_COLS].mean()
    affinity = (cluster_channel_avg / overall_channel_avg) * 100
    affinity = affinity.rename(index=CHANNEL_NAMES).sort_values(ascending=False)

    return jsonify({
        'avg_clv': f"${cluster_df['clv'].mean():,.2f}",
        'avg_churn_risk': f"{cluster_df['churn_risk_proba'].mean() * 100:.1f}%",
        'channel_affinity': {'channels': affinity.index.tolist(), 'scores': affinity.values.tolist()}
    })

@app.route('/get_genai_insights/<cluster_id>')
def get_genai_insights(cluster_id):
    if not genai:
        return jsonify({
            "persona": {"segment_name": "GenAI Disabled", "persona_summary": "API key not configured."},
            "strategy": ["Please configure your Gemini API key to enable this feature."]
        }), 503

    if cluster_id not in df['cluster'].unique():
        return jsonify({"error": "Cluster not found"}), 404

    cluster_df = df[df['cluster'] == cluster_id]
    primary_channel_tech_name = cluster_df[CHANNEL_COLS].mean().idxmax()
    primary_channel_friendly = CHANNEL_NAMES.get(primary_channel_tech_name, primary_channel_tech_name)

    # --- ENHANCEMENT: Send a much richer set of stats to the GenAI model ---
    stats_for_genai = {
        "avg_clv": f"${cluster_df['clv'].mean():,.2f}",
        "avg_churn_risk": f"{cluster_df['churn_risk_proba'].mean():.1%}",
        "dominant_age_group": cluster_df['age_group'].mode()[0],
        "dominant_income_bracket": cluster_df['income_bracket'].mode()[0],
        "primary_channel": primary_channel_friendly,
        "dominant_education": cluster_df['education_level'].mode()[0],
        "homeowner_status": cluster_df['homeowner_status'].mode()[0],
        "common_purchase_intent": cluster_df['offline_purchase_intent'].mode()[0]
    }

    persona_data = generate_persona_from_stats(stats_for_genai)
    strategy_data = generate_strategy_from_stats(stats_for_genai)

    return jsonify({'persona': persona_data, 'strategy': strategy_data})

@app.route('/generate_campaign_angles', methods=['POST'])
def generate_campaign_angles():
    if not genai: return jsonify({"error": "GenAI not configured"}), 503
    data = request.get_json()
    persona_summary = data.get('persona_summary', '')
    if not persona_summary: return jsonify({"error": "Persona summary is required"}), 400

    angles = generate_angles_from_persona(persona_summary)
    return jsonify(angles)

if __name__ == '__main__':
    app.run(debug=True)