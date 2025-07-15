from transformers import pipeline
import docx
import pdfplumber
import re


def preprocess_text(text):
    """Removes unwanted characters and normalizes line breaks."""
    # Remove non-standard whitespace characters (e.g., \xa0)
    text = text.replace('\xa0', ' ')
    # Replace multiple consecutive line breaks with a single one
    text = re.sub(r'\n+', '\n', text)
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    # Join the lines back together
    text = '\n'.join(lines)
    # Remove any remaining leading/trailing whitespace from the entire text
    text = text.strip()
    return text

def extract_text_from_doc(file_path):
    """Extracts text from a Word document."""
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF document."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

def analyze_rfp(raw_rfp_text):
    """Analyzes the RFP text to extract key information."""
    # Preprocess the text
    cleaned_rfp_text = preprocess_text(raw_rfp_text)

    print(cleaned_rfp_text[:500])
    print("Length of RFP text:", len(cleaned_rfp_text))

    # Initialize pipelines for different tasks
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cpu")
    Youtubeer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device="cpu")

    # --- Key Solution Tenets ---
    print("## Key Solution Tenets:\n")
    tenet_summary = summarizer(cleaned_rfp_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    print(tenet_summary)
    print("\n---")

    # --- High-Level Architecture Elements ---
    print("\n## High-Level Architecture Elements:\n")
    architecture_elements = []

    # Define relevant keywords and questions based on your expertise and the provided context
    architecture_related_keywords = ["architecture", "system", "platform", "data flow", "integration", "components", "services"]
    architecture_questions = [
        "What are the main components of the proposed solution?",
        "How will the different systems integrate with each other?",
        "What is the overall architecture of the platform?",
        "What are the key services involved?",
    ]

    for question in architecture_questions:
        answer = Youtubeer(question=question, context=cleaned_rfp_text)
        if answer['score'] > 0.5:  # Adjust score threshold as needed
            architecture_elements.append(f"- {answer['answer']}")

    if architecture_elements:
        for element in architecture_elements:
            print(element)
    else:
        print("Could not identify specific architecture elements. Please review the RFP for details.")
    print("\n---")

    # --- Text for Diagram (PPT) ---
    print("\n## Text for Diagram (PPT):\n")
    diagram_text = []
    if architecture_elements:
        for element in architecture_elements:
            # Simplify for diagram representation
            diagram_text.append(element.replace("- ", ""))
        for item in diagram_text:
            print(f"- {item}")
    else:
        print("No specific text generated for the diagram. Refer to the architecture elements.")
    print("\n---")

    # --- Key Benefits of the Solution ---
    print("\n## Key Benefits of the Solution:\n")
    benefit_keywords = ["benefit", "advantage", "improve", "increase", "reduce", "enhance", "faster", "better", "scalable", "cost-effective"]
    benefit_sentences = []

    for sentence in cleaned_rfp_text.split('.'):
        for keyword in benefit_keywords:
            if keyword in sentence.lower():
                benefit_sentences.append(sentence.strip())
                break

    if benefit_sentences:
        for benefit in sorted(list(set(benefit_sentences))): # Remove duplicates and sort
            print(f"- {benefit}")
    else:
        print("Could not identify specific benefits. Please review the RFP for details.")
    print("\n---")

    # --- Phase-wise Execution Plan ---
    print("\n## Phase-wise Execution Plan (Initial Scan):\n")
    phase_keywords = ["phase", "stage", "milestone", "implement", "deploy", "develop", "design", "plan", "rollout"]
    phase_related_text = []

    for paragraph in cleaned_rfp_text.split('\n\n'): # Split by paragraphs for broader context
        for keyword in phase_keywords:
            if keyword in paragraph.lower():
                phase_related_text.append(paragraph.strip())
                break

    if phase_related_text:
        for item in sorted(list(set(phase_related_text))): # Remove duplicates and sort
            print(f"- {item}")
    else:
        print("Could not identify a clear phase-wise execution plan. Please review the RFP for details.")
    print("\n---")

if __name__ == "__main__":
    file_path = input("Enter the path to the RFP document (Word or PDF): ")

    if file_path.lower().endswith(".docx"):
        rfp_text = extract_text_from_doc(file_path)
    elif file_path.lower().endswith(".pdf"):
        rfp_text = extract_text_from_pdf(file_path)
    else:
        print("Unsupported file format. Please provide a .docx or .pdf file.")
        exit()

    if rfp_text:
        analyze_rfp(rfp_text)