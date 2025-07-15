# Python Voice AI Agent with RAG, In-Memory Vector Store, and Feedback Loop
# Uses SpeechRecognition, pyttsx3, sentence-transformers, and faiss-cpu.

import speech_recognition as sr
import pyttsx3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import re # For parsing feedback

# --- Configuration ---
MICROPHONE_INDEX = None  # Set to a specific index if default doesn't work
MODEL_NAME = 'all-MiniLM-L6-v2' # Lightweight sentence transformer model
SIMILARITY_THRESHOLD = 0.6 # Adjust as needed for FAISS search relevance (lower means more strict)

# --- Global Variables ---
engine = None
recognizer = None
microphone = None
model = None
index = None # FAISS index
documents = [] # List to store original text documents
last_query_text = "" # To store the last user query for context in feedback
last_retrieved_doc_index = -1 # To store the index of the last retrieved document

# --- Initial Knowledge Base ---
# Based on WBD Technology Architecture information
initial_documents = [
    "The Technology Architecture translates the Logical Architecture into a concrete implementation using a combination of AWS services.",
    "Key components of the WBD Technology Architecture include a Snowflake Data Lakehouse for data storage, Databricks for advanced analytics and ML, AWS Glue for data integration, EMR Serverless for batch processing, Lambda for real-time processing and microservices, API Gateway for API management, and Step Functions for workflow orchestration.",
    "The WBD Technology Architecture emphasizes scalability, security, and cost-effectiveness, leveraging cloud-native services and a focus on data governance.",
    "The WBD Technology Architecture is indicative and will be further refined and tailored based on detailed discovery sessions and ongoing collaboration with WBD.",
    "The purpose of the WBD Technology Architecture is empowering data-driven campaign management from planning to reporting with a modern and scalable cloud platform.",
    "The WBD Technology Architecture allows for choosing the right tools and possibly reusing existing ones during the analysis phase.",
    "The different layers of the WBD Technology Architecture would use a subset of these technologies and it is important to define a boundary set of tech."
]

# --- Helper Functions ---

def list_microphones():
    """Lists available microphone devices and their indices."""
    print("Available microphone devices:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"Microphone with name \"{name}\" found for `Microphone(device_index={index})`")

def initialize_systems():
    """Initializes TTS, STT, Sentence Transformer, and FAISS index."""
    global engine, recognizer, microphone, model, documents, index

    # Initialize TTS
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
    except Exception as e:
        print(f"Error initializing TTS engine: {e}. Exiting.")
        exit()

    # Initialize STT
    recognizer = sr.Recognizer()
    try:
        microphone = sr.Microphone(device_index=MICROPHONE_INDEX)
        # Calibrate for ambient noise once at the beginning
        with microphone as source:
            print("Calibrating microphone for ambient noise... Please be quiet.")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Calibration complete.")
    except Exception as e:
        print(f"Error initializing microphone: {e}. Do you have a microphone connected? Exiting.")
        # list_microphones() # Uncomment to see available mics if there's an issue
        exit()

    # Initialize Sentence Transformer Model
    try:
        print(f"Loading sentence transformer model '{MODEL_NAME}'... (This may take a moment on first run)")
        model = SentenceTransformer(MODEL_NAME)
        print("Sentence transformer model loaded.")
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}. Ensure you have internet for the first download. Exiting.")
        exit()

    # Initialize documents and FAISS index
    documents.extend(initial_documents) # Start with initial docs
    build_faiss_index()

def speak(text):
    """Converts text to speech."""
    print(f"Agent: {text}")
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during speech: {e}")

def listen_for_audio(prompt="Listening..."):
    """Listens for audio input and returns recognized text."""
    if not recognizer or not microphone:
        speak("Speech recognition system not initialized.")
        return None

    with microphone as source:
        speak(prompt) # Speak the prompt to the user
        # recognizer.adjust_for_ambient_noise(source, duration=0.5) # Quick adjustment
        try:
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            # speak("I didn't hear anything. Please try speaking again.") # Agent speaks this now
            return None # Let the main loop handle speaking "didn't hear anything"
        except Exception as e:
            speak(f"Microphone error during listening: {e}")
            return None

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text.lower()
    except sr.UnknownValueError:
        # speak("Sorry, I could not understand what you said.") # Agent speaks this now
        return "___UNKNOWN_SPEECH___" # Special marker
    except sr.RequestError as e:
        speak(f"Could not request results from speech recognition service; {e}. Check internet.")
        return None
    except Exception as e:
        speak(f"An unexpected error occurred during speech recognition: {e}")
        return None

def build_faiss_index():
    """Builds or rebuilds the FAISS index from the current documents."""
    global index, documents, model
    if not documents:
        print("No documents to index.")
        index = None
        return

    print("Building FAISS index...")
    try:
        doc_embeddings = model.encode(documents, convert_to_tensor=False)
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension) # Using L2 distance
        index.add(np.array(doc_embeddings, dtype=np.float32))
        print(f"FAISS index built with {len(documents)} documents.")
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        index = None # Invalidate index on error

def query_vector_store(query_text, k=1):
    """Queries the FAISS index for relevant documents."""
    global index, model, documents, last_retrieved_doc_index
    last_retrieved_doc_index = -1 # Reset
    if index is None or not query_text:
        return None, -1.0 # No document, invalid score

    try:
        query_embedding = model.encode([query_text], convert_to_tensor=False)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)

        distances, indices = index.search(query_embedding_np, k)

        if indices.size > 0 and indices[0][0] != -1: # faiss returns -1 if no results
            # Normalize L2 distance to a similarity score (0 to 1, higher is better)
            # This is a heuristic. For L2, smaller distance is better.
            # We can invert and scale, e.g., similarity = 1 / (1 + distance)
            # Or, if distances are somewhat bounded, (max_dist - dist) / max_dist
            # For simplicity, let's use 1 / (1 + distance)
            similarity_score = 1 / (1 + distances[0][0])

            print(f"Found document with index {indices[0][0]}, L2 distance: {distances[0][0]:.4f}, Similarity score: {similarity_score:.4f}")

            if similarity_score >= SIMILARITY_THRESHOLD:
                last_retrieved_doc_index = indices[0][0]
                return documents[indices[0][0]], similarity_score
            else:
                print(f"Best match similarity {similarity_score:.4f} is below threshold {SIMILARITY_THRESHOLD}.")
                return None, similarity_score # Below threshold
        return None, -1.0
    except Exception as e:
        print(f"Error querying FAISS index: {e}")
        return None, -1.0

def handle_feedback(feedback_text):
    """Processes user feedback to update the knowledge base."""
    global documents, last_query_text, last_retrieved_doc_index

    if not feedback_text:
        return

    feedback_text_lower = feedback_text.lower()

    if feedback_text_lower == "correct":
        speak("Great! I'm glad that was helpful.")
    elif feedback_text_lower.startswith("incorrect the answer should be"):
        new_answer = feedback_text[len("incorrect the answer should be"):].strip()
        if new_answer:
            speak(f"Thank you for the correction. I will add '{new_answer}' to my knowledge for future reference.")
            documents.append(new_answer)
            # Optionally, try to remove or down-weight the previously retrieved incorrect document if last_retrieved_doc_index is valid
            # For simplicity now, we just add the new one. A more advanced system might try to replace.
            build_faiss_index()
        else:
            speak("It seems the new answer was empty. No changes made.")
    elif feedback_text_lower.startswith("add this information") or feedback_text_lower.startswith("add new information"):
        prefix_options = ["add this information:", "add this information", "add new information:", "add new information"]
        new_info = feedback_text
        for prefix in prefix_options:
            if feedback_text_lower.startswith(prefix):
                new_info = feedback_text[len(prefix):].strip()
                break

        if new_info:
            speak(f"Understood. I'm adding the information: '{new_info}'.")
            documents.append(new_info)
            build_faiss_index()
        else:
            speak("It seems the additional information was empty. No changes made.")
    elif feedback_text_lower == "skip feedback":
        speak("Okay, skipping feedback.")
    else:
        speak("I didn't quite understand that feedback. Please try 'correct', 'incorrect, the answer should be...', 'add this information...', or 'skip feedback'.")

# --- Main Application Loop ---
def main():
    global last_query_text
    initialize_systems()

    speak("Voice agent with RAG and feedback activated. How can I assist you today? Say 'goodbye' to exit.")

    while True:
        user_input = listen_for_audio("What is your query?")
        last_query_text = user_input # Store for feedback context

        if not user_input: # Timeout
            speak("I didn't hear anything. Please try speaking again.")
            continue
        if user_input == "___UNKNOWN_SPEECH___": # STT could not understand
            speak("Sorry, I could not understand what you said. Please try again.")
            continue

        if "goodbye" in user_input or "exit" in user_input or "quit" in user_input:
            speak("Goodbye! Have a great day.")
            break

        retrieved_doc, score = query_vector_store(user_input)

        if retrieved_doc:
            response = f"Based on my information: {retrieved_doc}"
            speak(response)
        else:
            speak("I couldn't find specific information related to your query in my current knowledge base. You can try rephrasing or add this as new information during feedback.")

        # Feedback loop
        feedback_prompt = "Was this helpful? You can say 'correct', 'incorrect the answer should be [new answer]', 'add this information [new info]', or 'skip feedback'."
        user_feedback = listen_for_audio(feedback_prompt)

        if not user_feedback: # Timeout on feedback
            speak("No feedback received. Moving on.")
            continue
        if user_feedback == "___UNKNOWN_SPEECH___":
            speak("I didn't understand your feedback. Moving on.")
            continue

        handle_feedback(user_feedback)

        # Small delay before next interaction
        time.sleep(0.5)

if __name__ == "__main__":
    # list_microphones() # Uncomment this if you need to find your microphone index
    main()
