# pip install transformers accelerate sentencepiece PyYAML torch
import json
import sys
import os
import re
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class DbtDocGenerator:
    """
    Reads a _struct_config.json file and generates a dbt-style YAML
    documentation file using a batch-processing call to a local AI model.
    """

    def __init__(self):
        """
        Initializes the generator and loads a lightweight open-source model.
        """
        self.pipe = None
        try:
            model_id = "google/gemma-2b-it"
            print(f"--- Initializing model '{model_id}'... ---", file=sys.stderr)
            print("--- This may take a while on the first run as the model is downloaded. ---", file=sys.stderr)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            print("--- Local AI model loaded successfully. ---", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load local AI model. Falling back to basic descriptions. Error: {e}", file=sys.stderr)

    def _get_batch_descriptions_llm(self, column_names: list[str]) -> dict:
        """
        (NEW) Generates descriptions for a list of columns in a single batch request.
        """
        if not self.pipe:
            return {name: self._get_description_basic(name) for name in column_names}

        try:
            # Create a JSON-formatted string of the column list as input for the prompt
            column_list_str = json.dumps(column_names, indent=2)

            # A more complex prompt that asks for a structured JSON output
            prompt_messages = [
                {
                    "role": "user",
                    "content": (
                        "You are a helpful data catalog assistant writing documentation for a dbt model. "
                        "I will provide you with a JSON list of technical column names. Your task is to return a single, valid JSON object. "
                        "The keys of your returned JSON object must be the exact column names I provide, and the values must be a short, clear, one-sentence description explaining the business purpose of each column."
                        "For example, for '2025 - mar' column, the description should be 'The identifier for March 2025'."
                        "Another example, for 'ACCOUNT_ID' column, the description should be 'The unique identifier for an account'.\n\n"
                        f"Here is the list of column names:\n{column_list_str}"
                    )
                }
            ]

            prompt = self.pipe.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

            # Generate the response from the model
            outputs = self.pipe(
                prompt,
                max_new_tokens=1024, # Increase token limit for a larger payload
                do_sample=False,
            )

            generated_text = outputs[0]['generated_text'][len(prompt):].strip()

            # Find and parse the JSON block from the model's response
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no markdown block, assume the whole response is the JSON
                json_str = generated_text

            descriptions = json.loads(json_str)
            return descriptions

        except Exception as e:
            print(f"Warning: Batch AI processing failed. Falling back to one-by-one generation. Error: {e}", file=sys.stderr)
            # Fallback to the single-item method if batching or JSON parsing fails
            return {name: self._get_description_llm(name) for name in column_names}

    def _get_description_llm(self, column_name: str) -> str:
        """Generates a description for a single column (used as a fallback)."""
        if not self.pipe: return self._get_description_basic(column_name)
        try:
            messages = [{"role": "user", "content": f"Provide a short, clear, one-sentence description for the database column named `{column_name}`."}]
            prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = self.pipe(prompt, max_new_tokens=50, do_sample=False)
            description = outputs[0]['generated_text'][len(prompt):].strip()
            return description.replace('"', '')
        except Exception:
            return self._get_description_basic(column_name)

    def _get_description_basic(self, column_name: str) -> str:
        """A simple, rule-based fallback for generating descriptions."""
        words = column_name.replace('_', ' ').lower()
        if "id" in words: return f"Unique identifier for the {words.replace(' id', '')}."
        return f"Name of the {words}."

    def generate_dbt_yaml(self, config_file_path):
        """Reads a config file and generates dbt YAML documentation for each source."""
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

            # Step 1: Collect all column names first
            column_names_to_process = [
                rule.get("1.0", {}).get("target_name")
                for rule in silver_config.values()
                if rule.get("1.0", {}).get("target_name")
            ]

            if not column_names_to_process:
                print(f"Warning: No columns found to document for source '{source_id}'.", file=sys.stderr)
                continue

            # Step 2: Get all descriptions in a single batch call
            print(f"  - Requesting descriptions for {len(column_names_to_process)} columns in a single batch...", file=sys.stderr)
            descriptions = self._get_batch_descriptions_llm(column_names_to_process)

            # Step 3: Assemble the final YAML
            columns_for_yaml = []
            for col_name in column_names_to_process:
                columns_for_yaml.append({
                    "name": col_name.lower(),
                    "description": descriptions.get(col_name, self._get_description_basic(col_name))
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
        config_file_path = 'tm1_fact_struct_config.json'
    else:
        config_file_path = sys.argv[1]

    generator = DbtDocGenerator()
    generator.generate_dbt_yaml(config_file_path)

if __name__ == '__main__':
    main()