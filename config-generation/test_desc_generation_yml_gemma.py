import json
import sys
import os
import re
import yaml # Requires 'pip install PyYAML'
import torch # Requires 'pip install torch'
# Requires 'pip install transformers accelerate'
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login


HF_TOKEN = ""
login(token=HF_TOKEN)

class DbtDocGenerator:
    """
    Reads a _struct_config.json file and generates a dbt-style YAML
    documentation file using a natively small Hugging Face model.
    """

    def __init__(self):
        """
        Initializes the generator and loads a lightweight open-source model
        directly into memory using the transformers library.
        """
        self.pipe = None
        try:
            # 1. Define the natively small and efficient model to use
            model_id = "google/gemma-2b-it"

            print(f"--- Initializing model '{model_id}'... ---", file=sys.stderr)
            print("--- This may take a while on the first run as the model is downloaded. ---", file=sys.stderr)

            # 2. Load the model using a memory-efficient data type
            # device_map="auto" will use a GPU if available, otherwise CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16, # Use bfloat16 for memory efficiency
            )

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # 3. Create a text-generation pipeline for easy inference
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            print("--- Local AI model loaded successfully. ---", file=sys.stderr)

        except Exception as e:
            print(f"Warning: Failed to load local AI model. Falling back to basic descriptions. Error: {e}", file=sys.stderr)
            print("Please ensure you have an internet connection for the first-time model download.", file=sys.stderr)

    def _get_description_llm(self, column_name: str) -> str:
        """
        Generates a description using the loaded Hugging Face pipeline.
        """
        if not self.pipe:
            return self._get_description_basic(column_name)

        try:
            # Create the chat-style messages for the prompt
            messages = [
                {
                    "role": "user",
                    "content": "You are a helpful data catalog assistant writing documentation for a dbt model. "
                               f"Provide a short, clear, one-sentence description for the column named `{column_name}`. "
                               "The description should explain the business purpose of the column. For example, for 'ACCOUNT_ID', "
                               "the description should be 'The unique identifier for an account.'"
                }
            ]

            # Gemma's template requires a specific format which the pipeline handles
            prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Generate text using the pipeline
            outputs = self.pipe(
                prompt,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
                top_k=50,
                top_p=0.95
            )

            # Extract the generated text from the pipeline's output
            generated_text = outputs[0]['generated_text']
            # The pipeline output includes the prompt, so we split to get only the assistant's reply
            description = generated_text[len(prompt):].strip()

            return description.replace('"', '')

        except Exception as e:
            print(f"Warning: AI inference failed for column '{column_name}'. Falling back. Error: {e}", file=sys.stderr)
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
            for rule in silver_config.values():
                versioned_rule = rule.get("1.0", {})
                column_name = versioned_rule.get("target_name")
                if column_name:
                    print(f"  - Generating description for column: {column_name}", file=sys.stderr)
                    description = self._get_description_llm(column_name)
                    print(description)
                    columns_for_yaml.append({
                        "name": column_name.lower(),
                        "description": description
                    })

            if not columns_for_yaml:
                print(f"Warning: No columns found to document for source '{source_id}'.", file=sys.stderr)
                continue

            dbt_yaml_structure = {
                "version": 2,
                "models": [
                    {
                        "name": source_id.lower(),
                        "description": f"Cleansed and conformed data for {source_id.replace('_', ' ').title()}.",
                        "columns": columns_for_yaml
                    }
                ]
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