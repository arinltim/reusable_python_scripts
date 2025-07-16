# app.py
import os
import json
import uuid
from collections import defaultdict
import re
import markdown2
from flask import Flask, jsonify, render_template
from thefuzz import fuzz
import google.generativeai as genai
import concurrent.futures

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


# --- Identity Resolution Logic (Definitive Two-Pass Version) ---

def get_record_identifiers(rec):
    """Helper to get all identifiers from a record."""
    ids = {
        'email': rec.get('email', '').lower(),
        'phone': rec.get('phone_number'),
        'device': rec.get('device_id'),
        'name': rec.get('customer_name', f"{rec.get('first_name', '')} {rec.get('last_name', '')}").strip().lower()
    }
    return {k: v for k, v in ids.items() if v}

def resolve_identities_graph(all_records):
    """
    **DEFINITIVE TWO-PASS RESOLUTION LOGIC**
    1. Clusters based on high-confidence links (email, phone, device).
    2. Merges clusters based on weaker links (name), ONLY if no conflicts are found.
    """
    num_records = len(all_records)
    adj = defaultdict(list)

    # Pass 1: Build graph using ONLY high-confidence links.
    for i in range(num_records):
        for j in range(i + 1, num_records):
            ids1 = get_record_identifiers(all_records[i][1])
            ids2 = get_record_identifiers(all_records[j][1])
            # Check for exact match on email, phone, or device
            if (ids1.get('email') and ids1.get('email') == ids2.get('email')) or \
                    (ids1.get('phone') and ids1.get('phone') == ids2.get('phone')) or \
                    (ids1.get('device') and ids1.get('device') == ids2.get('device')):
                adj[i].append(j)
                adj[j].append(i)

    # Find initial "islands" or components based on strong links
    visited = [False] * num_records
    initial_clusters = []
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
            initial_clusters.append(set(component))

    # Pass 2: Iteratively merge clusters based on weaker name links, if safe.
    merged = True
    while merged:
        merged = False
        for i in range(len(initial_clusters)):
            for j in range(i + 1, len(initial_clusters)):
                c1 = initial_clusters[i]
                c2 = initial_clusters[j]

                # Check for a name-based link between the two clusters
                should_merge = False
                for rec_idx1 in c1:
                    for rec_idx2 in c2:
                        name1 = get_record_identifiers(all_records[rec_idx1][1]).get('name')
                        name2 = get_record_identifiers(all_records[rec_idx2][1]).get('name')
                        if name1 and name2 and fuzz.token_sort_ratio(name1, name2) > 95:
                            should_merge = True
                            break
                    if should_merge:
                        break

                if should_merge:
                    # Before merging, check for conflicts (e.g., different device IDs)
                    devices1 = {get_record_identifiers(all_records[idx][1]).get('device') for idx in c1 if get_record_identifiers(all_records[idx][1]).get('device')}
                    devices2 = {get_record_identifiers(all_records[idx][1]).get('device') for idx in c2 if get_record_identifiers(all_records[idx][1]).get('device')}

                    # If both clusters have devices and the sets are disjoint, it's a conflict.
                    if devices1 and devices2 and devices1.isdisjoint(devices2):
                        continue # Do not merge

                    # No conflict, so merge c2 into c1
                    initial_clusters[i].update(c2)
                    initial_clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    # Final clusters are ready, now create the golden records
    return [create_golden_record(cluster, all_records) for cluster in initial_clusters]

# The `create_golden_record` and other functions remain the same as the last version.
def create_golden_record(cluster, all_records_flat):
    golden = {'c360_id': f'c360_{uuid.uuid4().hex[:12]}', 'source_records': {}}
    all_emails, all_names, all_phones, all_devices = set(), set(), set(), set()
    for index in cluster:
        source, rec = all_records_flat[index]
        source_key = rec.get('order_id') or rec.get('session_id') or rec.get('viewing_id') or rec.get('email')
        golden['source_records'][source] = source_key
        if rec.get('email'): all_emails.add(rec['email'].lower())
        name = rec.get('customer_name', f"{rec.get('first_name', '')} {rec.get('last_name', '')}").strip()
        if name: all_names.add(name)
        if rec.get('phone_number'): all_phones.add(rec['phone_number'])
        if rec.get('device_id'): all_devices.add(rec['device_id'])
        for key, value in rec.items():
            if key not in golden and value:
                golden[key] = value
    golden['canonical_name'] = max(all_names, key=len) if all_names else 'N/A'
    golden['all_emails'] = list(all_emails)
    golden['all_phones'] = list(all_phones)
    golden['all_devices'] = list(all_devices)
    for key in ['customer_name', 'first_name', 'last_name', 'email', 'phone_number', 'device_id']:
        golden.pop(key, None)
    return golden

def get_batch_c360_personas(profiles):
    if not generative_model: return []
    profiles_for_prompt = [{
        "c360_id": p.get('c360_id'), "name": p.get('canonical_name'), "emails": p.get('all_emails', []),
        "sources": list(p.get('source_records', {}).keys()), "ecommerce_spend": p.get('order_total', 0),
        "app_minutes": p.get('time_spent_min', 0), "ad_clicked": p.get('clicked_ad', 'Unknown'),
        "ott_hours": p.get('hours_watched_last_month', 0)
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

    batch_size = 5
    all_enriched_personas = []

    chunks = [golden_records[i:i + batch_size] for i in range(0, len(golden_records), batch_size)]
    print(f"Enriching {len(golden_records)} profiles concurrently in {len(chunks)} batches...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(get_batch_c360_personas, chunks)
        for enriched_chunk in results:
            all_enriched_personas.extend(enriched_chunk)

    print("✅ Enrichment complete.")

    enriched_map = {p['c360_id']: p['persona_summary'] for p in all_enriched_personas}
    for record in golden_records:
        record['persona_summary'] = enriched_map.get(record['c360_id'], "<p>Error: Persona not generated.</p>")

    return jsonify(golden_records)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)