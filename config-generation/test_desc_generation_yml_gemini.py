import json
import sys
import os
import re
import yaml # Requires PyYAML
import google.generativeai as genai

class DbtDocGenerator:
    """
    Reads a _struct_config.json file and generates a dbt-style YAML
    documentation file with AI-powered descriptions for each column.
    """

    def __init__(self):
        """Initializes the generator and configures the AI model if possible."""
        self.ai_model = None
        api_key = 'AIzaSyA2YXUaYM3lwc0msb08ANZ-a8FbvpV--PE'
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.ai_model = genai.GenerativeModel('gemini-2.0-flash')
                print("--- AI description generation is ENABLED. ---", file=sys.stderr)
            except Exception as e:
                print(f"Warning: AI model initialization failed. Falling back to basic descriptions. Error: {e}", file=sys.stderr)
        else:
            print("--- Warning: GOOGLE_API_KEY not set. Falling back to basic descriptions. ---", file=sys.stderr)

    def _get_description_ai(self, column_name: str) -> str:
        """
        Generates a human-readable description for a column using the Gemini API.
        """
        if not self.ai_model:
            return self._get_description_basic(column_name)

        try:
            # A prompt specifically tailored for writing dbt model documentation
            prompt = (
                "You are a helpful data catalog assistant writing documentation for a dbt model. "
                f"Provide a short, clear, one-sentence description for the column named `{column_name}`. "
                "The description should explain the business purpose of the column. "
                "For example, for 'ACCOUNT_ID', the description should be 'The unique identifier for an account.'"
            )
            response = self.ai_model.generate_content(prompt)
            # Clean up the response text from any potential markdown or extra spaces
            clean_text = response.text.strip().replace('"', '')
            return clean_text
        except Exception as e:
            print(f"Warning: AI API call failed for column '{column_name}'. Falling back to basic description. Error: {e}", file=sys.stderr)
            return self._get_description_basic(column_name)

    def _get_description_basic(self, column_name: str) -> str:
        """A simple, rule-based fallback for generating descriptions."""
        words = column_name.replace('_', ' ').lower()
        if "id" in words:
            return f"Unique identifier for the {words.replace(' id', '')}."
        return f"Name of the {words}."

    def generate_dbt_yaml(self, config_file_path):
        """
        Reads a structure config file and generates dbt YAML documentation for each source.
        """
        print(f"--- Reading config file: {config_file_path} ---", file=sys.stderr)
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Error: Could not read or parse config file '{config_file_path}'. {e}", file=sys.stderr)
            return

        structure_config = config_data.get("structure_config", {})
        if not structure_config:
            print("Error: No 'structure_config' key found in the JSON file.", file=sys.stderr)
            return

        for source_id, config in structure_config.items():
            print(f"--- Generating docs for source: {source_id} ---", file=sys.stderr)
            silver_config = config.get("silver")
            if not silver_config or not isinstance(silver_config, dict):
                print(f"Warning: No valid 'silver' configuration found for source '{source_id}'. Skipping.", file=sys.stderr)
                continue

            columns_for_yaml = []
            # The rules are the values of the silver config dictionary
            for rule in silver_config.values():
                versioned_rule = rule.get("1.0", {})
                column_name = versioned_rule.get("target_name")
                if column_name:
                    print(f"  - Generating description for column: {column_name}", file=sys.stderr)
                    description = self._get_description_ai(column_name)
                    columns_for_yaml.append({
                        "name": column_name.lower(), # dbt convention is lowercase
                        "description": description
                    })

            if not columns_for_yaml:
                print(f"Warning: No columns found to document for source '{source_id}'.", file=sys.stderr)
                continue

            # Create the final dbt YAML structure
            dbt_yaml_structure = {
                "version": 2,
                "models": [
                    {
                        "name": source_id.lower(), # Model name from source_id
                        "description": f"Cleansed and conformed data for {source_id.replace('_', ' ').title()}.",
                        "columns": columns_for_yaml
                    }
                ]
            }

            # Write the output to a YAML file
            base_path = os.path.splitext(config_file_path)[0]
            out_file_name = f"{base_path}_{source_id.lower()}_dbt_docs.yml"

            print(f"Writing dbt documentation to: {out_file_name}", file=sys.stderr)
            try:
                with open(out_file_name, "w", encoding='utf-8') as f:
                    yaml.dump(dbt_yaml_structure, f, sort_keys=False, indent=2)
                print("Done.", file=sys.stderr)
            except Exception as e:
                print(f"Error writing YAML file: {e}", file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_config.py <path_to_file>", file=sys.stderr)
        config_file_path = 'b2c_bce_json_file_struct_config.json'
    else:
        config_file_path = sys.argv[1]

    generator = DbtDocGenerator()
    generator.generate_dbt_yaml(config_file_path)

if __name__ == '__main__':
    main()
