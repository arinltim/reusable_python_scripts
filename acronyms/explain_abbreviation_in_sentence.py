import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def detect_abbreviations(text):
    """
    Detects potential abbreviations in text using regular expressions.
    Same detection as before.

    Args:
        text (str): Input text.

    Returns:
        list: List of potential abbreviations found in the text.
    """
    abbreviations = []
    pattern = r'\b(?:[A-Z]\.?[A-Za-z0-9]?)+\b\.|\b[A-Za-z]{2,}\.'
    matches = re.finditer(pattern, text)
    for match in matches:
        abbreviation = match.group(0)
        abbreviations.append(abbreviation)
    return abbreviations

def expand_abbreviations_nltk_no_dict(text):
    """
    Attempts to expand abbreviations using ONLY NLTK and very basic heuristics,
    WITHOUT ANY PREDEFINED DICTIONARIES OR EXTERNAL KNOWLEDGE.
    This is VERY LIMITED and INACCURATE. Demonstrative only.

    Args:
        text (str): The input text.

    Returns:
        str: Text with extremely limited and likely incorrect "expansions".
    """
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    detected_abbreviations = detect_abbreviations(text)

    expanded_tokens = []
    i = 0
    while i < len(tagged_tokens):
        token, tag = tagged_tokens[i]
        expanded = False

        for abbreviation in detected_abbreviations:
            if token == abbreviation: # Match detected abbreviation

                if re.match(r'Dr\.', token, re.IGNORECASE): # Very basic rule for "Dr." - still relies on *knowing* "Dr."
                    if i + 1 < len(tagged_tokens) and tagged_tokens[i+1][1] == 'NNP':
                        expanded_tokens.append("Doctor") # Guess "Doctor" if followed by Proper Noun (still heuristic)
                        expanded = True
                        break # Stop after applying this rule (very limited ruleset)
                    else:
                        expanded_tokens.append(token) # Keep original "Dr." if rule not met (very basic fallback)
                        expanded = True
                        break

                elif re.match(r'St\.', token, re.IGNORECASE): # Very basic rule for "St." - still relies on *knowing* "St."
                    if i + 1 < len(tagged_tokens) and tagged_tokens[i+1][1] in ('NNP', 'NN'):
                        expanded_tokens.append("Street") # Guess "Street" if followed by Noun (very basic guess)
                        expanded = True
                        break
                    else:
                        expanded_tokens.append(token) # Keep original "St." if rule not met (very basic fallback)
                        expanded = True
                        break

                else: # For other abbreviations, no rules - just keep original (very limited approach)
                    expanded_tokens.append(token)
                    expanded = True
                    break # Stop after "handling" (which is mostly doing nothing)

        if not expanded:
            expanded_tokens.append(token) # Keep original token if no abbreviation rule applied
        i += 1

    return " ".join(expanded_tokens)


# Example Sentences (same as before)
sentences = [
    "Visit Dr. Smith at 123 Main St. for Co. Corp. Ltd. services.",
    "Is this the right way to go to St. Louis or St. Peter's Church?",
    "The address is 456 Park Ave. and 789 Oak Rd.",
    "Call Inc. for more information.",
    "The Org. meeting is scheduled for Dept. heads.",
    "Govt. policies are under review.",
    "U.S.A. is a country.",
    "The patient saw a MD. specialist.",
    "Please contact R&D Dept. for details.",
    "He works at IBM Corp.",
    "Dr. John went to St. Mary's Ave.",
    "This is on Main Rd. near St. Joseph hospital." ,
    "Attending the IBM conference.",
    "R&D is crucial for innovation.",
    "What is the GDP of U.K.?",
    "I need to see a Ph.D.",
    "Go down Dr. for two blocks.",
    "Main Rd is closed.",
    "Park Ave is open."
]

print("Example Sentence Expansions (Rule-Based NLTK - NO DICTIONARY - CPU CODE - VERY LIMITED):")
for sentence in sentences:
    expanded_sentence_nltk_no_dict = expand_abbreviations_nltk_no_dict(sentence)
    print("-" * 50)
    print("Original Sentence:", sentence)
    print("Expanded Sentence (NLTK No Dict):", expanded_sentence_nltk_no_dict)