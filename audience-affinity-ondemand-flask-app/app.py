import faiss
import pickle
import numpy as np
import os
import json
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- 1. Load Assets & Configure Gemini ---
print("Loading models and data...")
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    print("Gemini API configured successfully.")
except (TypeError, ValueError):
    print("CRITICAL: GEMINI_API_KEY environment variable not set. GenAI features will be disabled.")
    genai = None

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('users.faiss')
with open('users.pkl', 'rb') as f:
    df = pickle.load(f)
print("Server is ready.")

app = Flask(__name__)

# --- 2. GenAI Helper Functions ---
def enhance_prompt_with_genai(query):
    if not genai: return query
    llm = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"You are an expert marketing assistant. Expand the user's simple search query into a rich, descriptive paragraph for a semantic search system. Describe a hypothetical person who fits the query. Respond only with the single descriptive paragraph. User Query: '{query}'\n\nDescriptive Paragraph:"
    try:
        response = llm.generate_content(prompt)
        return response.text.strip().replace('\n', ' ')
    except Exception as e:
        print(f"Error during prompt enhancement: {e}")
        return query

def generate_persona_with_genai(user_profiles_sample):
    if not genai: return {"segment_name": "N/A", "persona_summary": "GenAI is not configured."}
    llm = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = (
        "You are a senior market research analyst. Based on the following sample of user profiles from an audience segment, do two things:\n"
        "1. Give this segment a catchy, descriptive name (e.g., 'Urban Tech Explorers').\n"
        "2. Write a brief, insightful persona summary describing their likely motivations, goals, and communication preferences.\n\n"
        "Respond in a valid JSON format with two keys: \"segment_name\" and \"persona_summary\".\n\n"
        f"User Profiles:\n---\n{user_profiles_sample}\n---\n\nJSON Response:"
    )
    try:
        response = llm.generate_content(prompt)
        # Clean up the response to be valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error generating persona: {e}")
        return {"segment_name": "Analysis Failed", "persona_summary": "Could not generate persona due to an API error."}

def suggest_related_queries_with_genai(original_query):
    if not genai: return []
    llm = genai.GenerativeModel('gemini-1.5-pro-latest')

    prompt = (
        f"You are a strategic marketing consultant. A user has created a segment based on the query: '{original_query}'. "
        "Suggest three related but distinct audience segments they could also explore. Phrase them as new search queries. "
        "Respond only with a valid JSON array of strings, like [\"query 1\", \"query 2\", \"query 3\"]."
    )

    try:
        response = llm.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error suggesting queries: {e}")
        return []

# --- 3. Flask Routes ---
@app.route('/')
def home():
    all_users_vis = df[['pca1', 'pca2', 'name', 'job', 'city']].to_dict(orient='records')
    return render_template('index.html', all_users_vis=all_users_vis)

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query_text = data.get('query', '')
    k = int(data.get('count', 25))

    enhanced_query = enhance_prompt_with_genai(query_text)
    query_vector = model.encode([enhanced_query])
    query_vector = np.array(query_vector).astype('float32')
    distances, indices = index.search(query_vector, k)

    results_df = df.iloc[indices[0]]
    results_list = results_df.to_dict(orient='records')

    persona_data = {}
    related_queries = []
    if results_list:
        # Take top 5 profiles for persona generation
        sample_profiles = "\n".join(results_df['profile_text'].head(5).tolist())
        persona_data = generate_persona_with_genai(sample_profiles)
        related_queries = suggest_related_queries_with_genai(query_text)

    # Engagement propensity calculation remains the same
    engagement_propensity = 1.0
    if "tech" in query_text.lower() or "photography" in query_text.lower(): engagement_propensity = 1.5
    elif "sustainable" in query_text.lower() or "yoga" in query_text.lower(): engagement_propensity = 0.8
    segment_metrics = {"size": len(results_list), "engagement_propensity": engagement_propensity}

    return jsonify({
        "original_query": query_text, "enhanced_query": enhanced_query,
        "users": results_list, "segment_metrics": segment_metrics,
        "persona": persona_data, "related_queries": related_queries
    })

@app.route('/generate_ad_copy', methods=['POST'])
def generate_ad_copy():
    if not genai: return jsonify({"error": "GenAI not configured"}), 500
    data = request.get_json()
    persona_summary = data.get('persona_summary', '')

    llm = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = (
        f"You are an expert direct-response copywriter. For the audience persona described as: '{persona_summary}', "
        "generate two distinct ad variations for a social media campaign. Each variation must have a 'headline' (max 10 words) "
        "and 'body' text (max 50 words). Also provide one compelling 'email_subject' line. "
        "Respond in a valid JSON format like: "
        "{\"ad_variations\": [{\"headline\": \"...\", \"body\": \"...\"}], \"email_subject\": \"...\"}"
    )
    try:
        response = llm.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return jsonify(json.loads(cleaned_response))
    except Exception as e:
        print(f"Error generating ad copy: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)