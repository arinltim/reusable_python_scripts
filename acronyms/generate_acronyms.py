import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import itertools
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def generic_acronym_v2(words, remove_stopwords=True):
    """
    Generates acronyms with context-aware logic:
    - First char for adjacent/stopword-separated words.
    - At least first two chars for single words, retaining leading vowel.

    Args:
        words (list): List of words to generate acronym from.
        remove_stopwords (bool): Whether to remove stop words before acronym generation.

    Returns:
        str: Generated acronym.
    """
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    acronym_parts = []

    if len(words) > 1: # Adjacent or stopword-separated words
        for word in words:
            if remove_stopwords and word.lower() in stop_words:
                continue # Skip stop words
            acronym_parts.append(word[0].upper())
    elif words: # Single word
        word = words[0]
        if word[0].lower() in 'aeiou': # Starts with vowel
            acronym_parts.append(word[:2].upper()) # Keep first two (including vowel)
        else:
            acronym_parts.append(word[:2].upper()) # First two chars for consonant start - can be adjusted to more logic if needed like first two consonants etc.


    return "".join(acronym_parts)


def generate_dynamic_acronym_variations_v5(sentence, remove_stopwords_from_phrase=True):
    """
    Generates sentence variations with dynamic acronyms using generic_acronym_v2.

    Args:
        sentence (str): The input sentence.
        remove_stopwords_from_phrase (bool): Whether to remove stop words from phrases before acronym generation.

    Returns:
        list: List of sentence variations with dynamic acronyms.
    """

    words = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    variants = [sentence] # Start with original sentence
    noun_phrases_indices = []

    i = 0
    while i < len(pos_tags):
        word, tag = pos_tags[i]
        if tag.startswith(('NNP', 'NNPS', 'NN', 'NNS')): # Start of a noun phrase
            phrase_indices = [i]
            j = i + 1
            while j < len(pos_tags): # Extend phrase to include consecutive nouns and '&' and now any word (for stopword separation)
                next_word, next_tag = pos_tags[j]
                if next_tag.startswith(('NNP', 'NNPS', 'NN', 'NNS')) or next_word == '&' or (remove_stopwords_from_phrase and next_word.lower() in set(stopwords.words('english'))): # Include stopwords in phrase detection if remove_stopwords_from_phrase is True
                    phrase_indices.append(j)
                    j += 1
                else:
                    break # End of noun phrase
            noun_phrases_indices.append(phrase_indices)
            i = j # Continue from after the noun phrase
        else:
            i += 1


    for phrase_indices in noun_phrases_indices:
        phrase_words = [words[idx] for idx in phrase_indices]

        acronym = generic_acronym_v2(phrase_words, remove_stopwords=remove_stopwords_from_phrase)

        # Create variation
        temp_words = list(words)
        first_index_to_replace = min(phrase_indices)
        temp_words[first_index_to_replace] = acronym
        for index_to_remove in sorted(list(phrase_indices)[1:], reverse=True):
            del temp_words[index_to_remove]
        variants.append(" ".join(temp_words))


    return list(dict.fromkeys(variants)) # Remove duplicate variations


# Example Usage
sentence1 = "Proctor & Gamble is a large organization."
sentence2 = "The Information Technology Company Limited develops system."
sentence3 = "National Aeronautics and Space Administration."
sentence4 = "Organization for international business." # Example with stopword "for"
sentence5 = "Apple organization." # Example with vowel start
sentence6 = "System design." # Example with two adjacent words
sentence7 = "Very important organization." # Example with stopword in between


print("Sentence 1 Variations:")
variations1 = generate_dynamic_acronym_variations_v5(sentence1)
for variation in variations1:
    print(variation)

print("\nSentence 2 Variations:")
variations2 = generate_dynamic_acronym_variations_v5(sentence2)
for variation in variations2:
    print(variation)

print("\nSentence 3 Variations:")
variations3 = generate_dynamic_acronym_variations_v5(sentence3)
for variation in variations3:
    print(variation)

print("\nSentence 4 Variations:")
variations4 = generate_dynamic_acronym_variations_v5(sentence4)
for variation in variations4:
    print(variation)

print("\nSentence 5 Variations:")
variations5 = generate_dynamic_acronym_variations_v5(sentence5)
for variation in variations5:
    print(variation)

print("\nSentence 6 Variations:")
variations6 = generate_dynamic_acronym_variations_v5(sentence6)
for variation in variations6:
    print(variation)

print("\nSentence 7 Variations:")
variations7 = generate_dynamic_acronym_variations_v5(sentence7)
for variation in variations7:
    print(variation)