import speech_recognition as sr
import pyttsx3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import re

# --- Sound Playback (Windows-specific) ---
try:
    import winsound # For playing beep sound on Windows
except ImportError:
    winsound = None # winsound is not available on non-Windows systems

# --- Configuration ---
MICROPHONE_INDEX = None  # Set to a specific index if default doesn't work (e.g., 0, 1, etc.)
MODEL_NAME = 'all-MiniLM-L6-v2' # Lightweight sentence transformer model
SIMILARITY_THRESHOLD = 0.50 # Adjust as needed for FAISS search relevance (0.0 to 1.0)
BEEP_SOUND_FILE = 'beep.wav'
MAX_FEEDBACK_RETRIES = 3 # Number of times the agent will retry listening for feedback

# --- Global Variables ---
engine = None
recognizer = None
microphone = None
model = None
index = None # FAISS index
documents = [] # List to store original text documents
last_query_text = "" # To store the last user query for context in feedback
last_retrieved_doc_index = -1 # To store the index of the last retrieved document that formed the basis of an answer

# --- Initial Knowledge Base (Generic Tech Trends) ---
initial_documents = [
    "Artificial intelligence is rapidly transforming various industries, from healthcare to finance.",
    "Cloud computing continues to be a dominant trend, offering scalability and flexibility for businesses.",
    "The Internet of Things (IoT) is connecting more devices every day, generating vast amounts of data.",
    "Cybersecurity remains a critical concern as digital threats become more sophisticated.",
    "Big data analytics helps organizations derive insights and make informed decisions.",
    "Edge computing is gaining traction for processing data closer to its source, reducing latency.",
    "Quantum computing, while still in its early stages, holds the potential to solve complex problems currently intractable.",
    "The IT market is characterized by continuous innovation and a high demand for skilled professionals.",
    "Sustainability in technology, or Green IT, is becoming increasingly important.",
    "Blockchain technology is finding applications beyond cryptocurrencies, such as in supply chain management."
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
        engine.setProperty('rate', 165) # Slightly faster speech
    except Exception as e:
        print(f"Error initializing TTS engine: {e}. Exiting.")
        exit()

    # Initialize STT
    recognizer = sr.Recognizer()
    try:
        microphone = sr.Microphone(device_index=MICROPHONE_INDEX)
        with microphone as source:
            print("Calibrating microphone for ambient noise... Please be quiet.")
            # Adjust for ambient noise once at startup
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Calibration complete.")
    except Exception as e:
        print(f"Error initializing microphone: {e}. Do you have a microphone connected? Exiting.")
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
    documents.extend(initial_documents)
    build_faiss_index()

def speak(text):
    """Converts text to speech."""
    print(f"Agent: {text}")
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during speech: {e}")

def play_beep():
    """Plays a beep sound to indicate listening has started using winsound (Windows-specific)."""
    if winsound:
        try:
            # SND_FILENAME: The sound parameter is the name of a WAV file.
            # SND_ASYNC: The sound is played asynchronously and the function returns immediately.
            winsound.PlaySound(BEEP_SOUND_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Error playing beep sound with winsound: {e}. Ensure '{BEEP_SOUND_FILE}' exists and is a valid WAV file.")
    else:
        print("winsound module not available (non-Windows system). Cannot play beep.")

def listen_for_audio(prompt="Listening..."):
    """
    Listens for audio input. It waits for initial speech and then captures the phrase
    until a period of silence is detected or a phrase_time_limit is reached.
    """
    if not recognizer or not microphone:
        speak("Speech recognition system not initialized.")
        return None

    print(f"Agent is listening... (Prompt: {prompt})")
    try:
        with microphone as source:
            print("Please speak now...")
            # Listen for up to 5 seconds for initial sound (timeout),
            # then capture the phrase for a maximum of 60 seconds (phrase_time_limit).
            # If timeout is None, it waits indefinitely for speech.
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=60)
    except sr.WaitTimeoutError:
        print("No speech detected within the timeout.")
        return None # Indicate no speech
    except Exception as e:
        speak(f"Microphone error during listening: {e}")
        return None

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return "___UNKNOWN_SPEECH___"
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
        # Ensure all documents are strings
        valid_documents = [str(doc) for doc in documents if doc is not None]
        if not valid_documents:
            print("No valid documents to index after filtering.")
            index = None
            return

        doc_embeddings = model.encode(valid_documents, convert_to_tensor=False)
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(doc_embeddings, dtype=np.float32))
        # Update the main documents list to only contain the valid ones that were indexed
        documents = valid_documents
        print(f"FAISS index built with {len(documents)} documents.")
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        index = None

def query_vector_store(query_text, k=1):
    """Queries the FAISS index for relevant documents."""
    global index, model, documents, last_retrieved_doc_index

    # Reset last_retrieved_doc_index before each query.
    # It will be set if a relevant document is found and used for the answer.
    last_retrieved_doc_index = -1

    if index is None or not query_text:
        return None, -1.0

    try:
        query_embedding = model.encode([query_text], convert_to_tensor=False)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)

        distances, indices = index.search(query_embedding_np, k)

        if indices.size > 0 and indices[0][0] != -1 and indices[0][0] < len(documents):
            similarity_score = 1 / (1 + distances[0][0])

            print(f"Found document index {indices[0][0]} ('{documents[indices[0][0]][:50]}...'), L2 distance: {distances[0][0]:.4f}, Similarity: {similarity_score:.4f}")

            if similarity_score >= SIMILARITY_THRESHOLD:
                # This document is considered relevant enough to be the basis of the answer
                last_retrieved_doc_index = indices[0][0]
                return documents[indices[0][0]], similarity_score
            else:
                print(f"Best match similarity {similarity_score:.4f} is below threshold {SIMILARITY_THRESHOLD}.")
                # No specific document was relevant enough to form the answer, so last_retrieved_doc_index remains -1
                return None, similarity_score
        return None, -1.0
    except Exception as e:
        print(f"Error querying FAISS index: {e}")
        return None, -1.0

def handle_feedback(feedback_text):
    """
    Processes user feedback to update the knowledge base.
    Returns True if feedback was successfully processed (a known command was understood and acted upon),
    False otherwise (if the command was not recognized).
    """
    global documents, last_retrieved_doc_index, index

    if not feedback_text:
        return False # No feedback text

    feedback_text_lower = feedback_text.lower()

    if feedback_text_lower == "correct":
        speak("Great! I'm glad that was helpful.")
        return True
    elif feedback_text_lower.startswith(("incorrect the answer should be", "incorrect the answer is")):
        new_answer = re.sub(r"^(incorrect the answer should be|incorrect the answer is)[:\s]*", "", feedback_text, flags=re.IGNORECASE).strip()
        if new_answer:
            if last_retrieved_doc_index != -1 and last_retrieved_doc_index < len(documents):
                speak(f"Thank you. I'm updating my previous information with: '{new_answer}'. This will help with similar queries in the future.")
                documents[last_retrieved_doc_index] = new_answer
            else:
                speak(f"Thank you for the correction. I will add '{new_answer}' to my knowledge base.")
                documents.append(new_answer)
            build_faiss_index()
            return True
        else:
            speak("It seems the new answer was empty. No changes made.")
            return False # Failed to extract new answer
    elif feedback_text_lower.startswith(("add this information", "at this information", "add new information")):
        # More robust parsing for the information to add
        new_info_candidate = re.sub(r"^(add this information|at this information|add new information)[:\s]*", "", feedback_text, flags=re.IGNORECASE).strip()
        if new_info_candidate:
            if last_retrieved_doc_index != -1 and last_retrieved_doc_index < len(documents):
                speak(f"Understood. I'm adding more details to my previous response: '{new_info_candidate}'.")
                documents[last_retrieved_doc_index] += ". " + new_info_candidate # Append
            else:
                speak(f"Okay, I'm adding this as new information to my knowledge base: '{new_info_candidate}'.")
                documents.append(new_info_candidate) # Add as new document
            build_faiss_index()
            return True
        else:
            speak("It seems the additional information was empty. No changes made.")
            return False # Failed to extract new info
    elif feedback_text_lower == "skip feedback":
        speak("Okay, skipping feedback.")
        return True # Considered processed
    else:
        # This branch means the recognized feedback didn't match any known command.
        speak("I didn't quite understand that feedback. Please try 'correct', 'incorrect, the answer should be...', 'add this information...', or 'skip feedback'.")
        return False # Did not understand command

# --- Main Application Loop ---
def main():
    global last_query_text

    initialize_systems()
    speak("Hello Arindam -- Voice agent is activated. I will beep when you can start speaking. Say 'goodbye' to exit.")

    while True:
        # --- Query Phase ---
        speak("Ready for your query.")
        play_beep() # Beep before listening for query
        user_input = listen_for_audio(prompt="Listening for your query...")

        if not user_input: # No speech detected at all within timeout
            speak("I didn't hear anything. Please try speaking again.")
            continue
        if user_input == "___UNKNOWN_SPEECH___": # Speech detected but not understood
            speak("Sorry, I could not understand what you said. Please try again.")
            continue

        # Check for exit commands as part of the main query
        if "goodbye" in user_input or "exit" in user_input or "quit" in user_input:
            speak("Goodbye Arindam! Have a great day in LTIM.")
            break

        last_query_text = user_input # Store the query
        print(f"Processing user input: {user_input}")

        # Query the vector store
        retrieved_doc, score = query_vector_store(user_input)

        if retrieved_doc:
            response = f"Based on my information: {retrieved_doc}"
            speak(response)
        else:
            speak("I couldn't find specific information for your query in my current knowledge. You can try rephrasing, or add it as new information during feedback.")

        # --- Feedback Phase with Retries ---
        feedback_attempt = 0
        feedback_processed_successfully = False # Flag to track if feedback was truly processed
        while feedback_attempt < MAX_FEEDBACK_RETRIES and not feedback_processed_successfully:
            feedback_prompt_text = "Was this helpful? You can say 'correct', 'incorrect the answer should be [new answer]', 'add this information [new info]', or 'skip feedback'."
            if feedback_attempt > 0: # Only add retry prompt after the first attempt
                feedback_prompt_text = f"I didn't quite get that. Please try again. You have {MAX_FEEDBACK_RETRIES - feedback_attempt} attempts left. " + feedback_prompt_text

            speak(feedback_prompt_text)
            play_beep() # Beep before listening for feedback
            user_feedback = listen_for_audio("Feedback")

            if user_feedback and user_feedback != "___UNKNOWN_SPEECH___":
                # Call handle_feedback and check its return value
                if handle_feedback(user_feedback):
                    feedback_processed_successfully = True # Set to True only if handle_feedback succeeded
                else:
                    # handle_feedback already spoke "I didn't understand that feedback."
                    # So, we just increment attempt and loop again.
                    feedback_attempt += 1
            elif user_feedback == "___UNKNOWN_SPEECH___":
                speak("Sorry, I could not understand your feedback.")
                feedback_attempt += 1
            else: # user_feedback is None (no speech detected)
                speak("No feedback received.")
                feedback_attempt += 1

        if not feedback_processed_successfully:
            speak("Moving on, as I couldn't get clear feedback after several attempts.")
        else:
            # Explicit confirmation after successful feedback, before moving to next query
            # This message will only play if feedback was successfully processed (even on a retry)
            speak("Thank you for your feedback. I'm ready for your next query.")
        # --- End Feedback loop with retries ---

        time.sleep(0.5) # Short pause before starting the next query cycle

if __name__ == "__main__":
    # list_microphones() # Uncomment this if you need to find your microphone index
    main()
