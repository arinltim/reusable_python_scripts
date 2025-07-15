import pandas as pd
import google.generativeai as genai
import os
import sys
import re # Import regex for parsing

# --- Configure API Key ---
API_KEY = "AIzaSyD_Pp19LfP3OzHuBtu96_Q6hP13wMV7uxQ"

if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set the GEMINI_API_KEY environment variable with your API key.")
    sys.exit(1)

# --- Excel File Details ---
EXCEL_FILE_PATH = '13_associates_data_with_client_draft.xlsx' # <<<--- REPLACE with the path to your input Excel file
OUTPUT_EXCEL_FILE = '13_associates_data_analysis_clientwise.xlsx' # Name for the output Excel file (changed name to reflect filtering)
CLIENT_NAME_COLUMN = 'Client' # Column containing client names
SUMMARY_COLUMN = 'Summary'         # Column containing Jira story summaries
DESCRIPTION_COLUMN = 'Description' # Column containing Jira story descriptions

# --- Specific List of Clients to Process (Case-Insensitive Match) ---
# The script will include a row for analysis if ANY of the comma-separated
# client names in the 'Client' cell match (case-insensitive) a name in this list.
TARGET_CLIENT_NAMES = [
    'AMC',
    'LiveRamp',
    'NFL',
    'ADT',
    'Adtalem',
    'Big Lots',
    'BlueTriton',
    'Bob\'s', # Escape the apostrophe or use double quotes for the string
    'Capital One',
    'care.com',
    'CBS'
]


# --- Gemini Model Configuration ---
GEMINI_MODEL = 'gemini-2.5-flash-preview-04-17'
# Or you could try 'gemini-2.0-flash' which is generally available.
# GEMINI_MODEL = 'gemini-2.0-flash'


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
        text_to_analyze: The concatenated summaries and descriptions for a client.
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
        Analyze the following combined Jira summaries and descriptions for a single client.
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


# --- Helper function for case-insensitive flexible client check ---
def contains_client_flexible_case_insensitive(cell_value, target_client_lower):
    """
    Checks if a case-insensitive target client name is present as a distinct part
    within a string, considering various separators like spaces and commas.
    Handles potential NaN or non-string values.
    """
    if not isinstance(cell_value, str) or not cell_value.strip():
        return False

    # Convert cell value to lowercase
    s = cell_value.lower()

    # Replace common separators with spaces to facilitate splitting
    # We'll replace commas, spaces, hyphens, underscores, slashes, and potentially parentheses/brackets
    # Adjust this regex based on what you observe as separators in your data
    # Use a regex that looks for the target word surrounded by non-word characters or string boundaries
    # This is more precise than just splitting and checking if the part exists
    # Use \b for word boundaries.
    # The pattern is r'\b' + re.escape(target_client_lower) + r'\b'
    # re.escape() is important if the client name contains special regex characters (like care.com)

    # Check if the target client name exists as a whole word (case-insensitive)
    # using word boundaries.
    pattern = r'\b' + re.escape(target_client_lower) + r'\b'
    return re.search(pattern, s) is not None


# --- Main Script Logic ---
def main():
    # --- 1. Read Excel File ---
    try:
        print(f"Reading Excel file: {EXCEL_FILE_PATH}...")
        # Read the entire column as string to ensure .apply() with string methods works
        df = pd.read_excel(EXCEL_FILE_PATH, header=0, dtype={CLIENT_NAME_COLUMN: str})
        print(f"Successfully read {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: File not found at {EXCEL_FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

    # --- 2. Validate Columns ---
    required_columns = [CLIENT_NAME_COLUMN, SUMMARY_COLUMN, DESCRIPTION_COLUMN]
    missing_columns = [col for col in df.columns if col not in required_columns]
    # Check if required columns are missing, ignore extra columns
    if set(required_columns).issubset(df.columns) is False:
        print(f"Error: Missing required columns in the Excel file: {missing_columns}")
        print(f"Available columns are: {list(df.columns)}")
        sys.exit(1)


    # --- 3. Process Specified Clients (Case-Insensitive, Comma-Separated Check) ---
    # Use the predefined list of target clients
    # No need to create a lowercase version of the column beforehand,
    # as the check is done within the helper function row-by-row.
    clients_to_process = TARGET_CLIENT_NAMES # Use the original list for looping

    if not clients_to_process:
        print("TARGET_CLIENT_NAMES list is empty. No clients to process.")
        sys.exit(0)

    print(f"\nProcessing {len(clients_to_process)} specified clients: {TARGET_CLIENT_NAMES}")

    results_list = [] # Store results as a list of dictionaries for DataFrame creation

    for target_client_name in clients_to_process:
        # Convert target client name to lowercase ONCE for efficient checking
        target_client_lower = target_client_name.lower()

        print(f"\n--- Processing Client (targeting case-insensitive, comma-separated): {target_client_name} ---")

        # Filter DataFrame for rows where the Client column CONTAINS the target client name
        # (case-insensitive, checking comma-separated values)
        # Use .loc[] for explicit indexing to potentially avoid SettingWithCopyWarning later
        client_df = df.loc[
            df[CLIENT_NAME_COLUMN].apply(
                lambda x: contains_client_flexible_case_insensitive(x, target_client_lower)
            )
        ].copy() # Use .copy() after filtering


        if client_df.empty:
            print(f"No data found in the Excel file for client matching (case-insensitive, comma-separated): '{target_client_name}'")
            results_list.append({
                'Client Name': target_client_name,
                'Roles/Responsibilities': 'No data found in Excel matching the specified client.',
                'Activities': 'No data found in Excel matching the specified client.',
                'Technical Stack/Skills': 'No data found in Excel matching the specified client.'
            })
            continue

        # Combine Summary and Description, handling potential NaN values
        # Use a clearer separator between entries
        combined_texts = []
        for index, row in client_df.iterrows():
            summary = str(row[SUMMARY_COLUMN]) if pd.notna(row[SUMMARY_COLUMN]) else "No Summary"
            description = str(row[DESCRIPTION_COLUMN]) if pd.notna(row[DESCRIPTION_COLUMN]) else "No Description"
            combined_texts.append(f"Jira Item:\nSummary: {summary}\nDescription: {description}")

        all_combined_text = "\n\n---\n\n".join(combined_texts)


        if not all_combined_text.strip(): # Check if the combined text is just whitespace
            print(f"No combined Summary/Description found for client: {target_client_name}")
            results_list.append({
                'Client Name': target_client_name,
                'Roles/Responsibilities': 'No text found.',
                'Activities': 'No text found.',
                'Technical Stack/Skills': 'No text found.'
            })
            continue

        print(f"Found {len(client_df)} relevant rows for {target_client_name}.")

        # --- 5. Analyze with Gemini ---
        analysis_result = analyze_text_with_gemini(all_combined_text, API_KEY, GEMINI_MODEL)

        # Append results to the list
        result_entry = {'Client Name': target_client_name} # Use the name from the target list
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
    output_df = output_df[['Client Name', 'Roles/Responsibilities', 'Activities', 'Technical Stack/Skills']]

    # --- 7. Save Output Excel ---
    try:
        print(f"Writing results to Excel file: {OUTPUT_EXCEL_FILE}...")
        # Use index=False to avoid writing the DataFrame index as a column
        output_df.to_excel(OUTPUT_EXCEL_FILE, index=False, engine='openpyxl')
        print("Successfully wrote output Excel file.")
    except Exception as e:
        print(f"Error writing output Excel file: {e}")


if __name__ == "__main__":
    main()