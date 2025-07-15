# app.py

import os
import json
import joblib
import pandas as pd
import google.generativeai as genai
import markdown2 # Import the new library
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- Load Models and Configuration ---
MODEL_FILE = 'lookalike_model.pkl'
COLUMNS_FILE = 'model_columns.json'

try:
    ml_model = joblib.load(MODEL_FILE)
    with open(COLUMNS_FILE, 'r') as f:
        model_columns = json.load(f)
    print("✅ ML model and columns loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model or column file not found. Please run the 'lookalike_model_pipeline.py' script first.")
    ml_model = None
    model_columns = []

# Configure the Gemini API
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    generative_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    print("✅ Gemini model configured.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    generative_model = None

def enrich_with_gemini(profile_data: dict, score: float) -> str:
    """Uses Gemini to generate a rich marketing persona from profile data and a score."""
    if not generative_model:
        return "<h3>Error</h3><p>Gemini model is not configured. Cannot enrich profile.</p>"

    score_percentage = f"{score:.0%}"

    # **FIX:** Added explicit instruction to return clean Markdown.
    prompt = f"""
    You are a senior marketing analyst. Your task is to create a marketing persona summary for a customer profile.
    Based on the data below, including their "Look-Alike Score", generate a concise and actionable summary.

    **Customer Data:**
    - Age: {profile_data.get('age')}
    - Location: {profile_data.get('location')}
    - Primary Interest: {profile_data.get('primary_interest')}
    - Look-Alike Score: {score_percentage}

    **IMPORTANT: Your response MUST be in clean, simple Markdown format.** Include headings, bold text, and lists where appropriate.

    **Your Response should contain:**
    1.  **Persona Name:** A catchy, descriptive name.
    2.  **Executive Summary:** A short paragraph describing this person.
    3.  **Marketing Angles & Channels:** Suggest 2-3 specific strategies.
    4.  **Potential Offer:** Recommend a type of product or message.
    """
    try:
        response = generative_model.generate_content(prompt)
        # **FIX:** Convert the Markdown response to HTML before sending it to the frontend.
        html_output = markdown2.markdown(response.text)
        return html_output
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"<h3>Error</h3><p>An error occurred while generating the enrichment: {e}</p>"

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    **NEW:** This is the asynchronous endpoint for analyzing profiles.
    It returns data in JSON format.
    """
    if not ml_model:
        return jsonify({'error': "ML model not loaded. Please run the training script."}), 500

    try:
        input_data_raw = request.json

        # Create a DataFrame with the correct column order
        input_df = pd.DataFrame([input_data_raw], columns=model_columns)

        # 1. Get score from the ML model
        lookalike_score = ml_model.predict_proba(input_df)[0][1]

        # 2. Enrich the result with Gemini
        enriched_persona = enrich_with_gemini(input_data_raw, lookalike_score)

        return jsonify({
            'score': lookalike_score,
            'persona': enriched_persona
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)