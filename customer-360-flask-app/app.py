# app.run(host='0.0.0.0', port=5000, debug=True)

import os
import json
import uuid
from collections import defaultdict
import re
import markdown2
from flask import Flask, jsonify, render_template
from thefuzz import fuzz
import google.generativeai as genai

app = Flask(__name__)

# --- Configuration (Unchanged) ---
RAW_DATA_FILE = 'raw_customer_data.json'
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


# --- Identity Resolution Logic (with the definitive scoring function) ---

def calculate_match_score(rec1, rec2):
    """
    **DEFINITIVE SCORING LOGIC**
    This version uses clear, high-confidence rules to match records.
    """
    # Standardize attributes for comparison
    email1 = rec1.get('email', '').lower()
    email2 = rec2.get('email', '').lower()
    phone1 = rec1.get('phone_number')
    phone2 = rec2.get('phone_number')
    name1 = rec1.get('customer_name', f"{rec1.get('first_name', '')} {rec1.get('last_name', '')}").strip().lower()
    name2 = rec2.get('customer_name', f"{rec2.get('first_name', '')} {rec2.get('last_name', '')}").strip().lower()

    # Rule 1: Exact match on a unique identifier is a 100% match.
    if email1 and email1 == email2:
        return 100
    if phone1 and phone1 == phone2:
        return 100

    # Rule 2: A nearly perfect name match is a very strong signal (e.g., to link different emails).
    if name1 and name2 and fuzz.token_sort_ratio(name1, name2) > 95:
        return 80  # This score is high enough to pass the threshold on its own.

    # If no high-confidence rules are met, they are not a match.
    return 0

# The graph traversal and golden record creation logic remain the same, as they are correct.
def create_golden_record(cluster, all_records_flat):
    golden = {'c360_id': f'c360_{uuid.uuid4().hex[:12]}', 'source_records': {}}
    all_emails, all_names, all_phones = set(), set(), set()
    for index in cluster:
        source, rec = all_records_flat[index]
        golden['source_records'][source] = rec.get('order_id') or rec.get('session_id') or rec.get('email')
        if rec.get('email'): all_emails.add(rec['email'].lower())
        name = rec.get('customer_name', f"{rec.get('first_name', '')} {rec.get('last_name', '')}").strip()
        if name: all_names.add(name)
        if rec.get('phone_number'): all_phones.add(rec['phone_number'])
        for key, value in rec.items():
            if key not in golden and value:
                golden[key] = value
    golden['canonical_name'] = max(all_names, key=len) if all_names else 'N/A'
    golden['all_emails'] = list(all_emails)
    golden['all_phones'] = list(all_phones)
    for key in ['customer_name', 'first_name', 'last_name', 'email', 'phone_number']:
        golden.pop(key, None)
    return golden

def resolve_identities_graph(all_records):
    num_records = len(all_records)
    adj = defaultdict(list)
    match_threshold = 75  # Set a clear, high-confidence threshold

    for i in range(num_records):
        for j in range(i + 1, num_records):
            source1, rec1 = all_records[i]
            source2, rec2 = all_records[j]
            if calculate_match_score(rec1, rec2) >= match_threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * num_records
    clusters = []
    for i in range(num_records):
        if not visited[i]:
            component = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                component.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            clusters.append(component)
    return [create_golden_record(cluster, all_records) for cluster in clusters]


# --- Generative AI Enrichment (Unchanged) ---
def get_batch_c360_personas(profiles):
    if not generative_model: return []
    profiles_for_prompt = [{
        "c360_id": p.get('c360_id'), "name": p.get('canonical_name'), "emails": p.get('all_emails', []),
        "sources": list(p.get('source_records', {}).keys()), "ecommerce_spend": p.get('order_total', 0),
        "app_minutes": p.get('time_spent_min', 0), "ad_clicked": p.get('clicked_ad', 'Unknown')
    } for p in profiles]
    prompt = f"""
    You are a Customer 360 analyst. I will provide a JSON array of unified customer profiles.
    For each profile, generate a holistic persona summary in clean Markdown.
    Return a single, valid JSON array. Each object must contain a "c360_id" and a "persona_summary".
    Input Data:
    ```json
    {json.dumps(profiles_for_prompt, indent=2)}
    ```
    """
    try:
        response = generative_model.generate_content(prompt)
        json_text = re.search(r'```json\s*([\s\S]*?)\s*```', response.text).group(1)
        enriched_data = json.loads(json_text)
        for item in enriched_data:
            if 'persona_summary' in item:
                item['persona_summary'] = markdown2.markdown(item['persona_summary'])
        return enriched_data
    except Exception as e:
        print(f"Error in batch persona generation: {e}")
        return []


# --- Flask Routes (Unchanged) ---
@app.route('/')
def index():
    with open(RAW_DATA_FILE, 'r') as f:
        raw_data = json.load(f)
    return render_template('index.html', raw_data=raw_data)

@app.route('/resolve-and-enrich')
def resolve_and_enrich():
    with open(RAW_DATA_FILE, 'r') as f:
        raw_data = json.load(f)
    all_records = []
    for source, records in raw_data.items():
        for record in records:
            all_records.append((source, record))

    golden_records = resolve_identities_graph(all_records)

    enriched_personas = get_batch_c360_personas(golden_records)
    enriched_map = {p['c360_id']: p['persona_summary'] for p in enriched_personas}
    for record in golden_records:
        record['persona_summary'] = enriched_map.get(record['c360_id'], "<p>Error: Persona not generated.</p>")

    return jsonify(golden_records)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)