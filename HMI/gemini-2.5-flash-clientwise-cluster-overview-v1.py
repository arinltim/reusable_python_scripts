import pandas as pd
import google.generativeai as genai
import os
import sys
import re # Import regex for parsing

# --- Configure API Key ---
# It's safer to load the API key from environment variables
# rather than hardcoding it directly in the script.
# However, for demonstration based on your previous code,
# we will use the provided key.
# In a production scenario, prefer os.getenv('GEMINI_API_KEY')
API_KEY = "AIzaSyD_Pp19LfP3OzHuBtu96_Q6hP13wMV7uxQ"

if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set the GEMINI_API_KEY environment variable with your API key.")
    sys.exit(1)

# --- Excel File Details ---
EXCEL_FILE_PATH = '13_associates_data_with_client_3months.xlsx' # <<<--- REPLACE with the path to your input Excel file
OUTPUT_EXCEL_FILE = '13_associates_data_with_client_3months_cluster.xlsx' # Name for the output Excel file
CLIENT_NAME_COLUMN = 'Client' # Column containing client names
SUMMARY_COLUMN = 'Summary'         # Column containing Jira story summaries
DESCRIPTION_COLUMN = 'Description' # Column containing Jira story descriptions

# --- Gemini Model Configuration ---
GEMINI_MODEL = 'gemini-2.5-flash-preview-04-17'
# Or you could try 'gemini-2.0-flash' which is generally available.
# GEMINI_MODEL = 'gemini-2.0-flash'


# --- Helper Function to Parse Gemini Output ---
def parse_gemini_response(text: str):
    """
    Parses the Gemini response text based on expected markdown headings,
    including headings for clustering and overview.

    Args:
        text: The raw text response from Gemini.

    Returns:
        A dictionary with keys 'Client Cluster', 'Client Overview',
        'Roles/Responsibilities', 'Activities', and 'Technical Stack/Skills'.
        Values are extracted text or 'N/A'.
    """
    parsed_data = {
        'Client Cluster': 'N/A',       # New column
        'Client Overview': 'N/A',      # New column
        'Roles/Responsibilities': 'N/A',
        'Activities': 'N/A',
        'Technical Stack/Skills': 'N/A'
    }

    # Use regex to split based on the markdown headings (## Heading)
    # Includes headings for Client Cluster and Client Overview
    sections = re.split(
        r'(?=## Client Cluster\n|## Client Overview\n|## Roles/Responsibilities\n|## Activities\n|## Technical Stack/Skills\n)',
        text
    )

    current_section = None
    content_buffer = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue

        # Identify the section based on the heading
        if part.startswith('## Client Cluster'):
            # Store previous section's content if any
            if current_section and content_buffer:
                parsed_data[current_section] = content_buffer.strip()
            # Start new section
            current_section = 'Client Cluster'
            # Remove heading from content
            content_buffer = part.replace('## Client Cluster', '').strip()
        elif part.startswith('## Client Overview'):
            if current_section and content_buffer:
                parsed_data[current_section] = content_buffer.strip()
            current_section = 'Client Overview'
            content_buffer = part.replace('## Client Overview', '').strip()
        elif part.startswith('## Roles/Responsibilities'):
            if current_section and content_buffer:
                parsed_data[current_section] = content_buffer.strip()
            current_section = 'Roles/Responsibilities'
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
    # Check if ANY expected field was parsed
    if all(key not in parsed_data or value == 'N/A' for key, value in parsed_data.items() if key not in ['Client Cluster', 'Client Overview']):
        # This fallback might need refinement depending on how crucial the new fields are
        # Currently, it triggers if the original 3 fields weren't parsed
        print("Warning: Could not parse specific sections based on headings. Storing raw output in 'Activities' field.")
        parsed_data['Activities'] = text # Store raw output as fallback
        # Consider also putting raw output in new fields or marking them as parsing failed
        if parsed_data['Client Cluster'] == 'N/A': parsed_data['Client Cluster'] = 'Parsing Failed'
        if parsed_data['Client Overview'] == 'N/A': parsed_data['Client Overview'] = 'Parsing Failed'


    return parsed_data


# --- Helper Function for Gemini API Call ---
def analyze_text_with_gemini(text_to_analyze: str, api_key: str, model_name: str):
    """
    Calls the Gemini API to analyze the provided text and parses the response,
    including requests for clustering and overview.

    Args:
        text_to_analyze: The concatenated summaries and descriptions for a client.
        api_key: Your Gemini API key.
        model_name: The name of the Gemini model to use.

    Returns:
        A dictionary containing the parsed analysis results (Cluster, Overview,
        Roles, Activities, Skills) or None if an error occurs.
    """

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # --- !! Updated Prompt !! ---
        # Added requests for Client Cluster and Client Overview with specific headings
        prompt = f"""
        Analyze the following combined Jira summaries and descriptions for a single client.
        Extract and present the information using markdown headings and bullet points as specified below.
        Base your analysis *strictly* on the provided text and do not infer information not present.

        --- START JIRA DETAILS ---
        {text_to_analyze}
        --- END JIRA DETAILS ---

        Based *only* on the details provided above:

        ## Client Cluster
        Categorize this client into a high-level cluster or bucket based on the type of work evident in the Jira items (e.g., Retail, Finance, Media, Technology, CPG, etc.). Provide a single category or a brief categorization phrase. If categorization is unclear, state "Categorization Unclear".

        ## Client Overview
        Provide a concise summary (1-3 sentences) of what this client does or the primary type of work performed for them, based on the activities and context in the Jira items. If a summary cannot be formed, state "Overview Unclear".

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
        # Read the entire column as string to ensure consistent handling
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
    # Check if required columns are missing, ignore extra columns
    if not set(required_columns).issubset(df.columns):
        missing = set(required_columns) - set(df.columns)
        print(f"Error: Missing required columns in the Excel file: {list(missing)}")
        print(f"Available columns are: {list(df.columns)}")
        sys.exit(1)


    # --- 3. Get Unique Client Names (Exact Case-Insensitive Match) ---
    print("\nGetting unique client names from the Client column (using exact case-insensitive match)...")
    # Convert the client column to lowercase to find unique names case-insensitively
    # Fill NaN values first to avoid errors with .str accessor
    df['client_name_lower'] = df[CLIENT_NAME_COLUMN].fillna('').str.lower()
    unique_client_lowercased = df['client_name_lower'].unique().tolist()

    # Clean up the temporary lowercase column from the original dataframe
    # This ensures the original df is not modified persistently, important if df is used elsewhere
    df.drop(columns=['client_name_lower'], inplace=True)


    # Remove any empty strings that resulted from fillna('') and unique()
    unique_clients_to_process = [client for client in unique_client_lowercased if client.strip()]

    if not unique_clients_to_process:
        print(f"No unique client names found in the column '{CLIENT_NAME_COLUMN}'.")
        sys.exit(0) # Exit gracefully if no clients are found

    # Optional: If you want the output 'Client Name' column to have the original casing,
    # you'd need a mapping from lowercase unique name to an example original casing.
    # For simplicity here, we'll use the lowercase unique names in the output.
    print(f"\nFound {len(unique_clients_to_process)} unique clients to analyze: {unique_clients_to_process}")

    results_list = [] # Store results as a list of dictionaries for DataFrame creation

    # Iterate through the lowercase unique client names found
    for target_client_lower in unique_clients_to_process:
        # The target client name for the output will just be the lowercase version
        target_client_name_for_output = target_client_lower

        print(f"\n--- Processing Client: {target_client_name_for_output} ---")

        # Filter DataFrame for rows where the Client column EXACTLY matches the target client name
        # (case-insensitive match)
        # Convert the original column to lowercase for filtering on the fly
        client_df = df.loc[
            df[CLIENT_NAME_COLUMN].fillna('').str.lower() == target_client_lower
            ].copy() # Use .copy() after filtering


        if client_df.empty:
            # This case should ideally not happen if the name was extracted as unique,
            # but including for robustness.
            print(f"Warning: No data found in the Excel file for client matching '{target_client_name_for_output}' (this is unexpected based on unique extraction).")
            results_list.append({
                'Client Name': target_client_name_for_output,
                'Client Cluster': 'No data found.', # Populate new columns
                'Client Overview': 'No data found.', # Populate new columns
                'Roles/Responsibilities': 'No data found.',
                'Activities': 'No data found.',
                'Technical Stack/Skills': 'No data found.'
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
            print(f"No combined Summary/Description found for client: {target_client_name_for_output}")
            results_list.append({
                'Client Name': target_client_name_for_output,
                'Client Cluster': 'No text found.', # Populate new columns
                'Client Overview': 'No text found.', # Populate new columns
                'Roles/Responsibilities': 'No text found.',
                'Activities': 'No text found.',
                'Technical Stack/Skills': 'No text found.'
            })
            continue

        print(f"Found {len(client_df)} relevant rows for {target_client_name_for_output}.")

        # --- 5. Analyze with Gemini ---
        # The analyze_text_with_gemini function is updated to request new fields
        analysis_result = analyze_text_with_gemini(all_combined_text, API_KEY, GEMINI_MODEL)

        # Append results to the list
        result_entry = {'Client Name': target_client_name_for_output} # Use the standardized lowercase name
        if analysis_result:
            result_entry.update(analysis_result) # Add parsed sections (now including new ones)
        else:
            # Populate with error message if analysis failed
            result_entry.update({
                'Client Cluster': 'Analysis Failed.',      # Populate new columns
                'Client Overview': 'Analysis Failed.',     # Populate new columns
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

    # Reorder columns for clarity - including the new columns
    output_df = output_df[['Client Name', 'Client Cluster', 'Client Overview', 'Roles/Responsibilities', 'Activities', 'Technical Stack/Skills']]

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