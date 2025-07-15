import flask
from flask import request, jsonify, render_template
import json
import spacy
import google.generativeai as genai
import os

# --- 1. Configuration ---
# It's best practice to set your API key as an environment variable
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

if not GOOGLE_API_KEY:
    print("FATAL: GEMINI_API_KEY environment variable not set.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
# Load the pre-trained English model from spaCy
nlp = spacy.load("en_core_web_sm")
app = flask.Flask(__name__)
MASTER_DATA_FILE = 'master_data.json'

# --- 2. Conversation Memory ---
# This dictionary holds the context of an ongoing conversation.
conversation_state = {}

# --- 3. The CORRECTED Hybrid AI Logic (spaCy + Gemini + Context) ---
def analyze_user_request_hybrid(text: str, context: dict = None) -> dict:
    """
    Combines spaCy for pre-processing and Gemini for deep understanding,
    now with full conversation context.
    """
    # <<< THIS IS THE CORRECTED PART >>>
    # Step 1: spaCy runs first for fast, local entity extraction.
    doc = nlp(text)
    spacy_entities = [f"'{ent.text}' ({ent.label_})" for ent in doc.ents]
    spacy_findings = ", ".join(spacy_entities) if spacy_entities else "None"
    # <<< END OF CORRECTION >>>

    context_prompt = ""
    if context and context.get('pending_intent'):
        intent = context['pending_intent']
        entities = context.get('collected_entities', {})
        context_prompt = f"""
        The user is in the middle of a conversation. The current goal is to '{intent}'.
        So far, we have collected these details: {json.dumps(entities)}.
        The user's latest message is likely providing the missing information.
        Analyze their new message to fill in the blanks.
        """

    # <<< THE PROMPT IS NOW ENRICHED WITH SPACY'S FINDINGS >>>
    prompt = f"""
    You are an intelligent SAP master data assistant.
    Your task is to analyze the user's request and extract parameters.

    A preliminary analysis using a fast NLP tool (spaCy) found these named entities: {spacy_findings}.
    Use these findings as hints to improve your accuracy, but rely on the full user request and conversation context for the final analysis.

    {context_prompt}

    The user's full, most recent request is: "{text}"

    You must identify one of the following intents: 'create_cost_center', 'check_status_cost_center', or 'unclear'.
    Based on the intent, extract the required entities.

    Respond ONLY with a valid JSON object. Do not include any other text.

    JSON structure for 'create_cost_center':
    {{
      "intent": "create_cost_center",
      "is_complete": boolean,
      "entities": {{"id": "...", "name": "...", "manager": "..."}},
      "reply": "Clarifying question if incomplete."
    }}

    JSON structure for 'check_status_cost_center':
    {{
      "intent": "check_status_cost_center",
      "is_complete": boolean,
      "entities": {{"id": "..."}},
      "reply": "Clarifying question if incomplete."
    }}
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        json_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_response_text)
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return {"intent": "error", "reply": "Sorry, an error occurred while connecting to the AI model."}

# --- 4. Backend Logic (Now with Read and Write) ---
def _write_cost_center_to_db(payload: dict) -> str:
    """Writes new cost center data to the JSON file."""
    with open(MASTER_DATA_FILE, 'r+') as f:
        db = json.load(f)
        # Check for duplicates
        if any(cc.get('id') == payload.get('id') for cc in db['cost_centers']):
            return f"Error: Cost Center with ID '{payload.get('id')}' already exists."
        db['cost_centers'].append(payload)
        f.seek(0)
        json.dump(db, f, indent=4)
    return f"Success! Cost Center '{payload.get('name')}' with ID '{payload.get('id')}' has been created."

def _read_cost_center_from_db(cost_center_id: str) -> str:
    """Reads a cost center's status from the JSON file."""
    with open(MASTER_DATA_FILE, 'r') as f:
        db = json.load(f)
        for cc in db['cost_centers']:
            if cc.get('id') == cost_center_id:
                return f"Status for Cost Center '{cost_center_id}':\nName: {cc.get('name')}\nManager: {cc.get('manager', 'N/A')}"
    return f"Sorry, I could not find a Cost Center with ID '{cost_center_id}'."

# --- 5. Flask Routes (Updated with State Management) ---
@app.route('/')
def home():
    """Serves the main chat page."""
    global conversation_state
    conversation_state.clear() # Clear memory on page load/refresh
    return render_template('index.html')

@app.route('/process_message', methods=['POST'])
def process_message():
    """Main endpoint, now with logic to handle conversational state."""
    global conversation_state
    user_message = request.json.get('message', '')

    # If we are in the middle of a conversation, merge new info
    if conversation_state.get('pending_intent'):
        # Pass the context to the AI
        ai_analysis = analyze_user_request_hybrid(user_message, context=conversation_state)
        # Merge newly found entities with previously collected ones
        new_entities = ai_analysis.get('entities', {})
        conversation_state.get('collected_entities', {}).update(
            {k: v for k, v in new_entities.items() if v} # only update with non-empty values
        )
        ai_analysis['entities'] = conversation_state.get('collected_entities', {})
    else:
        # This is a new request
        ai_analysis = analyze_user_request_hybrid(user_message)

    bot_reply = "I'm sorry, I couldn't process that. Please be more specific."
    intent = ai_analysis.get('intent')

    # --- Intent Routing ---
    if intent == 'create_cost_center':
        # Check if all required entities (id, name) are present
        entities = ai_analysis.get('entities', {})
        if entities.get('id') and entities.get('name'):
            bot_reply = _write_cost_center_to_db(entities)
            conversation_state.clear() # End of conversation
        else:
            # Still incomplete, save state and ask for more info
            conversation_state['pending_intent'] = 'create_cost_center'
            conversation_state['collected_entities'] = entities
            bot_reply = ai_analysis.get('reply', "I need more information. What is the ID and Name?")

    elif intent == 'check_status_cost_center':
        entities = ai_analysis.get('entities', {})
        if entities.get('id'):
            bot_reply = _read_cost_center_from_db(entities['id'])
            conversation_state.clear() # End of conversation
        else:
            conversation_state['pending_intent'] = 'check_status_cost_center'
            conversation_state['collected_entities'] = entities
            bot_reply = ai_analysis.get('reply', "Which Cost Center ID would you like to check?")

    elif intent in ['error', 'unclear']:
        bot_reply = ai_analysis.get('reply', bot_reply)
        conversation_state.clear()

    return jsonify({"reply": bot_reply})

# --- 6. Main Execution Block ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)