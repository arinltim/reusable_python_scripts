import json
import sys
import os
import re
import pandas as pd

class StructureConfigGenerator:
    """
    Generates a two-layer (Bronze, Silver) structure_config JSON
    by robustly analyzing an input JSON, CSV, or Parquet file's structure.
    The output is a versioned dictionary of rules.
    """

    def _create_versioned_entry(self, rule_dict):
        """Wraps a rule dictionary in the required versioning structure."""
        return {"1.0": rule_dict}

    def _get_silver_type_json(self, value):
        if isinstance(value, bool): return "BOOLEAN"
        if isinstance(value, int): return "INTEGER"
        if isinstance(value, float): return "FLOAT"
        return "STRING"

    # --- JSON-Specific Recursive Methods ---

    def _build_bronze_rules_json(self, data, parent_path=""):
        """
        (CORRECTED) Recursively builds BRONZE rules. Iterates through all items
        in lists and correctly handles lists containing mixed or primitive values.
        """
        rules = {}
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{parent_path}_{key}" if parent_path else key

                # --- FINAL, ROBUST LOGIC FOR LIST HANDLING ---
                if isinstance(value, list) and any(isinstance(item, dict) for item in value):
                    # It's a list containing at least one object.
                    # Iterate through all items and recurse only on the objects.
                    for item in value:
                        if isinstance(item, dict):
                            rules.update(self._build_bronze_rules_json(item, current_path))
                elif isinstance(value, dict) and value:
                    # It's a non-empty nested object. Recurse into it.
                    rules.update(self._build_bronze_rules_json(value, current_path))
                else:
                    # This now handles all primitives (strings, numbers, bools, nulls),
                    # empty lists, lists of primitives, and empty dictionaries.
                    target_name = current_path.upper()
                    rule_dict = {"path": key, "target_name": target_name, "type": "STRING"}
                    rules[target_name] = self._create_versioned_entry(rule_dict)
                # ---------------------------------------------
        return rules

    def _build_silver_rules_json(self, data, parent_path=""):
        """
        (CORRECTED) Recursively builds SILVER rules with robust list handling.
        """
        rules = {}
        if isinstance(data, dict):
            for key, value in data.items():
                bronze_target_name = f"{parent_path}_{key}" if parent_path else key

                # --- FINAL, ROBUST LOGIC FOR LIST HANDLING ---
                if isinstance(value, list) and any(isinstance(item, dict) for item in value):
                    # It's a list containing at least one object.
                    for item in value:
                        if isinstance(item, dict):
                            rules.update(self._build_silver_rules_json(item, bronze_target_name))
                elif isinstance(value, dict) and value:
                    # It's a non-empty nested object.
                    rules.update(self._build_silver_rules_json(value, bronze_target_name))
                else:
                    # This handles primitives and lists of primitives.
                    silver_name = bronze_target_name.upper()
                    rule_dict = {"path": silver_name, "target_name": silver_name, "type": self._get_silver_type_json(value)}
                    if re.search(r'date|time|timestamp', key, re.IGNORECASE):
                        rule_dict["type"] = "TIMESTAMP_NTZ(6)"
                    # This specifically handles lists of primitives for silver
                    elif isinstance(value, list):
                        rule_dict["array_to_str"] = True
                    rules[silver_name] = self._create_versioned_entry(rule_dict)
                # ---------------------------------------------
        return rules

    # --- Tabular (CSV/Parquet) Processing Methods ---

    def _map_pandas_dtype_to_silver(self, dtype, col_name):
        if re.search(r'date|time|timestamp', col_name, re.IGNORECASE):
            return "TIMESTAMP_NTZ(6)"
        dtype_str = str(dtype).lower()
        if 'int' in dtype_str: return "INTEGER"
        if 'float' in dtype_str: return "FLOAT"
        if 'bool' in dtype_str: return "BOOLEAN"
        if 'datetime' in dtype_str: return "TIMESTAMP_NTZ(6)"
        return "STRING"

    def _build_rules_from_dataframe(self, df):
        bronze_rules_dict = {}
        silver_rules_dict = {}
        for i, col in enumerate(df.columns):
            target_name = col.upper()
            bronze_rule = {"path": f"${i+1}", "target_name": target_name, "type": "STRING"}
            bronze_rules_dict[target_name] = self._create_versioned_entry(bronze_rule)
            silver_rule = {"path": target_name, "target_name": target_name, "type": self._map_pandas_dtype_to_silver(df[col].dtype, col)}
            silver_rules_dict[target_name] = self._create_versioned_entry(silver_rule)
        return bronze_rules_dict, silver_rules_dict

    # --- Main Generator ---

    def generate(self, file_path):
        print(f"--- Processing file: {file_path} ---", file=sys.stderr)
        base_name = os.path.basename(file_path)
        source_id, extension = os.path.splitext(base_name)
        extension = extension.lower()
        final_config = {"structure_config": {}}
        silver_audit_fields = {
            "HASH_ID": self._create_versioned_entry({"path": "HASH_ID", "target_name": "HASH_ID", "type": "INTEGER"}),
            "SNOWFLAKE_MODIFIED_DATE": self._create_versioned_entry({"path": "SNOWFLAKE_MODIFIED_DATE", "target_name": "SNOWFLAKE_MODIFIED_DATE", "type": "TIMESTAMP_NTZ(6)"})
        }
        try:
            if extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                if all(isinstance(v, dict) for v in input_data.values()):
                    for sub_source_id, data_object in input_data.items():
                        print(f"Detected multi-source JSON object: {sub_source_id}", file=sys.stderr)
                        bronze_rules = self._build_bronze_rules_json(data_object)
                        silver_rules = self._build_silver_rules_json(data_object)
                        silver_rules.update(silver_audit_fields)
                        final_config["structure_config"][sub_source_id.upper()] = {"bronze": bronze_rules, "silver": silver_rules}
                    return final_config
                else:
                    print(f"Detected single-source JSON document.", file=sys.stderr)
                    bronze_rules = self._build_bronze_rules_json(input_data)
                    silver_rules = self._build_silver_rules_json(input_data)
            elif extension in ['.csv', '.parquet']:
                df = pd.read_csv(file_path) if extension == '.csv' else pd.read_parquet(file_path)
                print("\n--- File Content Preview ---\n", df.head().to_string(), "\n--------------------------\n", file=sys.stderr)
                bronze_rules, silver_rules = self._build_rules_from_dataframe(df)
            else:
                raise ValueError(f"Unsupported file type '{extension}'")
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}", file=sys.stderr)
            return None
        silver_rules.update(silver_audit_fields)
        final_config["structure_config"][source_id.upper()] = {"bronze": bronze_rules, "silver": silver_rules}
        return final_config

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_config.py <path_to_file>", file=sys.stderr)
        input_file_path = 'Executive_list_records_20250622.json'
        if not os.path.exists(input_file_path):
            print(f"Default file '{input_file_path}' not found. Please provide a file path.", file=sys.stderr)
            sys.exit(1)
    else:
        input_file_path = sys.argv[1]
    generator = StructureConfigGenerator()
    generated_config = generator.generate(input_file_path)
    if generated_config:
        out_file_name = os.path.splitext(input_file_path)[0] + "_struct_config.json"
        print(f"Writing configuration to: {out_file_name}", file=sys.stderr)
        with open(out_file_name, "w", encoding='utf-8') as f:
            json.dump(generated_config, f, indent=2)
        print("Done.", file=sys.stderr)

if __name__ == '__main__':
    main()