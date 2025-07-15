# pip install PyYAML google-generativeai
import json
import sys
import os
import re
import yaml
import google.generativeai as genai

class DbtDocGenerator:
    """
    Reads a _struct_config.json file and generates a dbt-style YAML
    documentation file using a single, efficient batch call to the Gemini API.
    """

    def __init__(self):
        """Initializes the generator and configures the AI model if possible."""
        self.ai_model = None
        # SECURITY BEST PRACTICE: Read API key from an environment variable
        api_key = 'AIzaSyA2YXUaYM3lwc0msb08ANZ-a8FbvpV--PE'
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Use 'gemini-2.0-flash' for speed or 'gemini-2.0-pro' for potentially higher quality
                self.ai_model = genai.GenerativeModel('gemini-2.0-flash')
                print("--- AI description generation is ENABLED. ---", file=sys.stderr)
            except Exception as e:
                print(f"Warning: AI model initialization failed. Falling back to basic descriptions. Error: {e}", file=sys.stderr)
        else:
            print("--- Warning: GOOGLE_API_KEY environment variable not set. ---", file=sys.stderr)
            print("--- Falling back to basic, non-AI descriptions. ---", file=sys.stderr)


    def _get_batch_descriptions_ai(self, column_names: list[str]) -> dict:
        """
        (UPDATED) Generates descriptions for a list of columns in a single batch request
        using an improved, example-driven (few-shot) prompt for higher quality.
        """
        if not self.ai_model:
            return {name: self._get_description_basic(name) for name in column_names}

        try:
            column_list_str = json.dumps(column_names, indent=2)

            # A more robust prompt that includes a high-quality example (few-shot learning)
            # to guide the model to produce better results.
            prompt = (
                "You are an expert data catalog assistant writing documentation for a dbt model.\n"
                "I will provide a JSON list of technical database column names. Your task is to return a single, valid JSON object.\n"
                "The keys of your returned JSON object must be the exact column names I provide.\n"
                "The values must be a clear and specific two to three sentence description explaining the business purpose of each column.\n\n"
                "Here is an example of the quality I expect:\n"
                "EXAMPLE INPUT:\n"
                '["SESSION_ID", "TRANSACTION_TIMESTAMP_UTC", "USER_LIFETIME_VALUE_USD"]\n\n'
                "EXAMPLE OUTPUT:\n"
                '{\n'
                '  "SESSION_ID": "The unique identifier for the user session.",\n'
                '  "TRANSACTION_TIMESTAMP_UTC": "Specifies the exact timestamp of the transaction in Coordinated Universal Time (UTC).",\n'
                '  "USER_LIFETIME_VALUE_USD": "Represents the total lifetime value of the user, measured in United States Dollars."\n'
                '}\n\n'
                "Now, generate the descriptions for the following list:\n"
                f"{column_list_str}"
            )

            response = self.ai_model.generate_content(prompt)

            text_response = response.text.strip()
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", text_response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text_response

            descriptions = json.loads(json_str)
            return descriptions

        except Exception as e:
            print(f"Warning: Batch AI processing failed. Falling back to one-by-one generation. Error: {e}", file=sys.stderr)
            return {name: self._get_description_ai(name) for name in column_names}


    def _get_description_ai(self, column_name: str) -> str:
        """
        Generates a human-readable description for a single column (used as a fallback).
        """
        if not self.ai_model:
            return self._get_description_basic(column_name)
        try:
            prompt = (
                "You are a helpful data catalog assistant writing documentation for a dbt model. "
                f"Provide a short, clear, one-sentence description for the column named `{column_name}`. "
                "The description should explain the business purpose of the column."
            )
            response = self.ai_model.generate_content(prompt)
            return response.text.strip().replace('"', '')
        except Exception as e:
            print(f"Warning: AI API call failed for column '{column_name}'. Falling back. Error: {e}", file=sys.stderr)
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

            column_names_to_process = [
                rule.get("1.0", {}).get("target_name")
                for rule in silver_config.values()
                if rule.get("1.0", {}).get("target_name")
            ]

            if not column_names_to_process:
                print(f"Warning: No columns found to document for source '{source_id}'.", file=sys.stderr)
                continue

            print(f"  - Requesting descriptions for {len(column_names_to_process)} columns in a single batch...", file=sys.stderr)
            descriptions = self._get_batch_descriptions_ai(column_names_to_process)

            columns_for_yaml = []
            for col_name in column_names_to_process:
                desc = descriptions.get(col_name, self._get_description_basic(col_name))
                columns_for_yaml.append({
                    "name": col_name.lower(),
                    "description": desc
                })

            dbt_yaml_structure = {
                "version": 2,
                "models": [{"name": source_id.lower(), "description": f"Cleansed and conformed data for {source_id.replace('_', ' ').title()}.", "columns": columns_for_yaml}]
            }

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
        config_file_path = 'Executive_list_records_20250622_struct_config.json'
    else:
        config_file_path = sys.argv[1]
    generator = DbtDocGenerator()
    generator.generate_dbt_yaml(config_file_path)

if __name__ == '__main__':
    main()
