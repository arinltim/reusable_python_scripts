# Python Voice AI Agent for Customer Queries
# Uses SpeechRecognition for STT and pyttsx3 for TTS.
# Knowledge base is focused on Technology Architecture.

import speech_recognition as sr
import pyttsx3
import time

# --- Configuration ---
# You might need to adjust MICROPHONE_INDEX if you have multiple mics.
# Run the list_microphones() function once to see available mics and their indices.
MICROPHONE_INDEX = None # Set to a specific index if default doesn't work, e.g., 1

# --- Text-to-Speech Engine Initialization ---
try:
    engine = pyttsx3.init()
    # Optional: Adjust voice properties
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[1].id) # Example: use the second available voice
    engine.setProperty('rate', 160)  # Speed of speech
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    print("Please ensure you have a TTS engine installed (e.g., SAPI5 on Windows, NSSpeechSynthesizer on macOS, or espeak on Linux).")
    exit()

# --- Speech-to-Text Recognizer Initialization ---
recognizer = sr.Recognizer()
microphone = sr.Microphone(device_index=MICROPHONE_INDEX)

# --- Knowledge Base ---
# Derived from the Technology Architecture information provided by the user.
knowledge_base = {
    "introduction": "The Technology Architecture translates the Logical Architecture into a concrete implementation using a combination of AWS services. Key components include a Snowflake Data Lakehouse for data storage, Databricks for advanced analytics and ML, AWS Glue for data integration, EMR Serverless for batch processing, Lambda for real-time processing and microservices, API Gateway for API management, and Step Functions for workflow orchestration. This architecture emphasizes scalability, security, and cost-effectiveness, leveraging cloud-native services and a focus on data governance. It is important to note that this is an indicative architecture and will be further refined and tailored based on detailed discovery sessions and ongoing collaboration with others.",
    "data storage": "A Snowflake Data Lakehouse is used for data storage.",
    "analytics and ml": "Databricks is used for advanced analytics and ML.",
    "data integration": "AWS Glue is used for data integration.",
    "batch processing": "EMR Serverless is used for batch processing.",
    "real-time processing": "Lambda is used for real-time processing and microservices.",
    "microservices": "Lambda is used for real-time processing and microservices.",
    "api management": "API Gateway is used for API management.",
    "workflow orchestration": "Step Functions are used for workflow orchestration.",
    "key components": "Key components include Snowflake, Databricks, AWS Glue, EMR Serverless, Lambda, API Gateway, and Step Functions.",
    "architecture principles": "The architecture emphasizes scalability, security, and cost-effectiveness, leveraging cloud-native services and a focus on data governance.",
    "finality": "This architecture is indicative and will be further refined and tailored based on detailed discovery sessions and ongoing collaboration with the org.",
    "purpose": "The goal is empowering data-driven campaign management—from planning to reporting—with a modern and scalable cloud platform, with the ability to choose the right tools and possibly reuse existing ones during the analysis phase.",
    "greeting_response": "Hello! I am a voice assistant. I can answer questions about Technology Architecture. How can I help you today?",
    "unknown_query": "I'm sorry, I don't have specific information on that topic. My knowledge is focused on the Technology Architecture. Could you ask something related to that?",
    "goodbye_message": "Goodbye! Have a great day."
}

# --- Helper Functions ---

def list_microphones():
    """Lists available microphone devices and their indices."""
    print("Available microphone devices:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"Microphone with name \"{name}\" found for `Microphone(device_index={index})`")

def speak(text):
    """Converts text to speech."""
    print(f"Agent: {text}")
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during speech: {e}")

def listen_for_audio():
    """Listens for audio input from the microphone and returns the recognized text."""
    with microphone as source:
        print("\nAdjusting for ambient noise... Please wait.")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            # Increased timeout and phrase_time_limit for potentially longer queries
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            # speak("I didn't hear anything. Please try speaking again.")
            print("Agent: I didn't hear anything. Please try speaking again.") # Speak might be too slow here
            return None
        except Exception as e:
            speak(f"Microphone error: {e}")
            return None

    try:
        print("Recognizing speech...")
        # Using Google Web Speech API for recognition
        # This requires an internet connection.
        # For offline, consider recognizer.recognize_sphinx(audio) - requires CMU Sphinx
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text.lower()
    except sr.UnknownValueError:
        speak("Sorry, I could not understand what you said. Please try again.")
        return None
    except sr.RequestError as e:
        speak(f"Could not request results from Google Speech Recognition service; {e}. Please check your internet connection.")
        return None
    except Exception as e:
        speak(f"An unexpected error occurred during speech recognition: {e}")
        return None

def query_knowledge_base(query_text):
    """Processes the query text and returns a response from the knowledge base."""
    if not query_text:
        return None # No input to process

    # Simple keyword-based NLU
    if any(word in query_text for word in ["hello", "hi", "hey"]):
        return knowledge_base["greeting_response"]
    if "technology architecture" in query_text or "architecture overview" in query_text:
        return knowledge_base["introduction"]
    if "data storage" in query_text or "store data" in query_text or "snowflake" in query_text:
        return knowledge_base["data storage"]
    if ("advanced analytics" in query_text or "machine learning" in query_text or "ml" in query_text or "databricks" in query_text) and not "aws glue" in query_text :
        return knowledge_base["analytics and ml"]
    if "data integration" in query_text or "aws glue" in query_text: # Made "aws glue" more specific
        return knowledge_base["data integration"]
    if "batch processing" in query_text or "emr serverless" in query_text:
        return knowledge_base["batch processing"]
    if ("real-time processing" in query_text or "lambda" in query_text) and "microservices" not in query_text:
        return knowledge_base["real-time processing"]
    if "microservices" in query_text: # Can also be triggered by "lambda" if context implies microservices
        return knowledge_base["microservices"]
    if "api management" in query_text or "api gateway" in query_text:
        return knowledge_base["api management"]
    if "workflow orchestration" in query_text or "step functions" in query_text:
        return knowledge_base["workflow orchestration"]
    if "key components" in query_text:
        return knowledge_base["key components"]
    if any(word in query_text for word in ["scalability", "security", "cost-effectiveness", "data governance", "principles"]):
        return knowledge_base["architecture principles"]
    if any(word in query_text for word in ["final", "refined", "indicative"]):
        return knowledge_base["finality"]
    if any(word in query_text for word in ["purpose", "goal", "campaign management"]):
        return knowledge_base["purpose"]
    if any(word in query_text for word in ["bye", "exit", "quit", "goodbye"]):
        return knowledge_base["goodbye_message"]

    return knowledge_base["unknown_query"]

# --- Main Application Loop ---
def main():
    """Main function to run the voice agent."""
    # Optional: List microphones if you're unsure which one to use
    # list_microphones()

    speak("Voice agent activated. How can I assist you with the Technology Architecture today? Say 'goodbye' to exit.")

    while True:
        user_input = listen_for_audio()

        if user_input:
            response = query_knowledge_base(user_input)
            if response:
                speak(response)
                if response == knowledge_base["goodbye_message"]:
                    break
            # else: # This case is handled if query_knowledge_base returns None (e.g. from no input)
            # or if it returns unknown_query which is then spoken.
            # No need for an explicit speak here if unknown_query is the default.
        else:
            # If listen_for_audio returned None due to timeout or error, the listen_for_audio function
            # would have already spoken an error message.
            # We can add a small delay to prevent rapid re-prompts if needed.
            time.sleep(0.5)

if __name__ == "__main__":
    main()
