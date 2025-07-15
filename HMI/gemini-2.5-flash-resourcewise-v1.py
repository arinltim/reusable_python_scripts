import pandas as pd
import google.generativeai as genai
import os
import sys
import re # Import regex for parsing

API_KEY = os.environ["GEMINI_API_KEY"]

if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set the GEMINI_API_KEY environment variable with your API key.")
    sys.exit(1)

# --- Excel File Details ---
EXCEL_FILE_PATH = '13_associates_data.xlsx' # <<<--- REPLACE with the path to your input Excel file
OUTPUT_EXCEL_FILE = '13_associates_data_analysis_associates.xlsx' # Name for the output Excel file
ASSOCIATE_NAME_COLUMN = 'Assignee' # Column containing associate names
SUMMARY_COLUMN = 'Summary'         # Column containing Jira story summaries
DESCRIPTION_COLUMN = 'Description' # Column containing Jira story descriptions

GEMINI_MODEL = 'gemini-2.5-flash-preview-04-17'
# Or you could try 'gemini-2.0-flash' which is generally available.

# --- Helper Function to Parse Gemini Output ---
def parse_gemini_response(text: str):
    """
    Parses the Gemini response text based on expected markdown headings.

    Args:
        text: The raw text response from Gemini.

    Returns:
        A dictionary with keys 'Roles/Responsibilities', 'Activities',
        and 'Technical Stack/Skills'. Values are extracted text or 'N/A'.
    """
    parsed_data = {
        'Roles/Responsibilities': 'N/A',
        'Activities': 'N/A',
        'Technical Stack/Skills': 'N/A'
    }

    # Use regex to split based on the markdown headings (## Heading)
    # Look for "## Heading\n" - the \n helps avoid splitting mid-sentence if a heading appears in text
    # The (?=...) is a positive lookahead assertion, it matches the headings but doesn't consume them in the split
    sections = re.split(r'(?=## Roles/Responsibilities\n|## Activities\n|## Technical Stack/Skills\n)', text)

    current_section = None
    content_buffer = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue

        # Identify the section based on the heading
        if part.startswith('## Roles/Responsibilities'):
            # Store previous section's content if any
            if current_section and content_buffer:
                parsed_data[current_section] = content_buffer.strip()
            # Start new section
            current_section = 'Roles/Responsibilities'
            # Remove heading from content
            content_buffer = part.replace('## Roles/Responsibilities', '').strip()
        elif part.startswith('## Activities'):
            if current_section and content_buffer:
                parsed_data[current_section] = content_buffer.strip()
            current_section = 'Activities'
            content_buffer = part.replace('## Activities', '').strip()
        elif part.startswith('## Technical Stack/Skills'):
            if current_section and content_buffer:
                parsed_data[current_section] = content_buffer.strip()
            current_section = 'Technical Stack/Skills'
            content_buffer = part.replace('## Technical Stack/Skills', '').strip()
        else:
            # Append content if it belongs to the current section
            # Add back newline potentially lost in split/strip, ensure spacing between parts
            if current_section is not None:
                content_buffer += "\n" + part
            # If no section started yet, ignore leading text (like a title before the first ##)
            # You might want to handle this case differently if there's important preamble.


    # Store the last section's content
    if current_section and content_buffer:
        parsed_data[current_section] = content_buffer.strip()

    # Fallback if regex split didn't work as expected (e.g., no headings found)
    # You might add more robust fallback parsing here if needed
    if all(value == 'N/A' for value in parsed_data.values()):
        print("Warning: Could not parse specific sections based on headings. Storing raw output in 'Activities' field.")
        parsed_data['Activities'] = text # Store raw output as fallback

    return parsed_data


# --- Helper Function for Gemini API Call ---
def analyze_text_with_gemini(text_to_analyze: str, api_key: str, model_name: str):
    """
    Calls the Gemini API to analyze the provided text and parses the response.

    Args:
        text_to_analyze: The concatenated summaries and descriptions for an associate.
        api_key: Your Gemini API key.
        model_name: The name of the Gemini model to use.

    Returns:
        A dictionary containing the parsed analysis results (Roles, Activities, Skills)
        or None if an error occurs.
    """

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # --- !! Refined Prompt !! ---
        # Explicitly asking for markdown headings and bullet points
        # Added instructions for the model to ONLY use provided text
        prompt = f"""
        Analyze the following combined Jira summaries and descriptions for a single associate.
        Extract and present the information using markdown headings and bullet points as specified below.
        Base your analysis *strictly* on the provided text and do not infer information not present.

        --- START JIRA DETAILS ---
        {text_to_analyze}
        --- END JIRA DETAILS ---

        Based *only* on the details provided above:

        ## Roles/Responsibilities
        List the primary roles or responsibilities demonstrated (e.g., Developer, Tester, Analyst, Lead) as bullet points. If no clear role is evident, state "Role information not explicitly mentioned".

        ## Activities
        Summarize the main activities and tasks performed as bullet points. Be concise but cover the key actions.

        ## Technical Stack/Skills
        List the specific technical skills, programming languages, software, frameworks, platforms, or tools explicitly mentioned or strongly implied as required to achieve these tasks. List them clearly as bullet points. If no specific technical details are mentioned, state "Technical stack/skills not explicitly mentioned".
        """

        print(f"\n--- Sending text to Gemini ({model_name}) for analysis (length: {len(text_to_analyze)} chars) ---")
        # You might need to adjust generation_config based on model capabilities or desired output randomness
        # For consistent output, a lower temperature is generally better.
        # generation_config = genai.GenerationConfig(temperature=0.1)
        response = model.generate_content(prompt) # , generation_config=generation_config)
        print("--- Received response from Gemini ---" + response.text)

        # Parse the response text
        parsed_analysis = parse_gemini_response(response.text)
        return parsed_analysis

    except Exception as e:
        print(f"Error calling Gemini API or parsing response: {e}")
        return None


# --- Main Script Logic ---
def main():
    # --- 1. Read Excel File ---
    try:
        print(f"Reading Excel file: {EXCEL_FILE_PATH}...")
        df = pd.read_excel(EXCEL_FILE_PATH, header=0)
        print(f"Successfully read {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: File not found at {EXCEL_FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

    # --- 2. Validate Columns ---
    required_columns = [ASSOCIATE_NAME_COLUMN, SUMMARY_COLUMN, DESCRIPTION_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in the Excel file: {missing_columns}")
        print(f"Available columns are: {list(df.columns)}")
        sys.exit(1)

    # --- 3. Get Unique Associate Names ---
    # Get all unique associate names from the specified column, drop any NaN values
    unique_associates = df[ASSOCIATE_NAME_COLUMN].dropna().unique().tolist()

    if not unique_associates:
        print(f"No unique associate names found in the column '{ASSOCIATE_NAME_COLUMN}'.")
        sys.exit(0) # Exit gracefully if no associates are found

    print(f"\nFound {len(unique_associates)} unique associates to analyze: {unique_associates}")

    # --- 4. Process Each Associate ---
    results_list = [] # Store results as a list of dictionaries for DataFrame creation

    for associate_name in unique_associates:
        print(f"\n--- Processing Associate: {associate_name} ---")

        # Filter DataFrame for the current associate
        associate_df = df[df[ASSOCIATE_NAME_COLUMN] == associate_name].copy() # Use .copy() to avoid SettingWithCopyWarning

        if associate_df.empty:
            print(f"No data found for associate: {associate_name}")
            results_list.append({
                'Associate Name': associate_name,
                'Roles/Responsibilities': 'No data found in Excel.',
                'Activities': 'No data found in Excel.',
                'Technical Stack/Skills': 'No data found in Excel.'
            })
            continue

        # Combine Summary and Description, handling potential NaN values
        # Use a clearer separator between entries
        combined_texts = []
        for index, row in associate_df.iterrows():
            summary = str(row[SUMMARY_COLUMN]) if pd.notna(row[SUMMARY_COLUMN]) else "No Summary"
            description = str(row[DESCRIPTION_COLUMN]) if pd.notna(row[DESCRIPTION_COLUMN]) else "No Description"
            combined_texts.append(f"Jira Item:\nSummary: {summary}\nDescription: {description}")

        all_combined_text = "\n\n---\n\n".join(combined_texts)


        if not all_combined_text.strip(): # Check if the combined text is just whitespace
            print(f"No combined Summary/Description found for associate: {associate_name}")
            results_list.append({
                'Associate Name': associate_name,
                'Roles/Responsibilities': 'No text found.',
                'Activities': 'No text found.',
                'Technical Stack/Skills': 'No text found.'
            })
            continue

        print(f"Found {len(associate_df)} relevant rows for {associate_name}.")

        # --- 5. Analyze with Gemini ---
        analysis_result = analyze_text_with_gemini(all_combined_text, API_KEY, GEMINI_MODEL)

        # Append results to the list
        result_entry = {'Associate Name': associate_name}
        if analysis_result:
            result_entry.update(analysis_result) # Add parsed sections
        else:
            # Populate with error message if analysis failed
            result_entry.update({
                'Roles/Responsibilities': 'Analysis Failed.',
                'Activities': 'Analysis Failed.',
                'Technical Stack/Skills': 'Analysis Failed.'
            })
        results_list.append(result_entry)

    # --- 6. Create Output Excel ---
    print("\n\n--- Analysis Complete ---")
    if not results_list:
        print("No results generated.")
        return

    # Create DataFrame from the results list
    output_df = pd.DataFrame(results_list)

    # Reorder columns for clarity
    output_df = output_df[['Associate Name', 'Roles/Responsibilities', 'Activities', 'Technical Stack/Skills']]

    try:
        print(f"Writing results to Excel file: {OUTPUT_EXCEL_FILE}...")
        # Use index=False to avoid writing the DataFrame index as a column
        output_df.to_excel(OUTPUT_EXCEL_FILE, index=False, engine='openpyxl')
        print("Successfully wrote output Excel file.")
    except Exception as e:
        print(f"Error writing output Excel file: {e}")

if __name__ == "__main__":
    main()