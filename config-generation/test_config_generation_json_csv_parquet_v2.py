import json
import sys
import os
import re
import pandas as pd

class StructureConfigGenerator:
    """
    Generates a two-layer (Bronze, Silver) structure_config JSON.
    This version uses an external text file as the source of truth for the schema
    and looks up data types in the source JSON. It correctly handles JSON arrays.
    """

    def _create_versioned_entry(self, rule_dict):
        """Wraps a rule dictionary in the required versioning structure."""
        return {"1.0": rule_dict}

    def _get_silver_type(self, value):
        """Infers the data type for the Silver layer from an example value."""
        if isinstance(value, bool): return "BOOLEAN"
        if isinstance(value, int): return "INTEGER"
        if isinstance(value, float): return "FLOAT"
        return "STRING"

    def _get_nested_value(self, record, path):
        """Safely retrieves a value from a nested dict using a dot-notation path."""
        keys = path.split('.')
        val = record
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return None # Path not found
        return val

    def _build_lookup_map(self, record_list, expected_columns_set):
        """
        Scans the JSON records once to find an example value for each expected column.
        Handles both top-level keys and nested 'fields' keys.
        """
        print("Scanning JSON to build data type lookup map...", file=sys.stderr)
        lookup_map = {}
        found_columns = set()

        for record in record_list:
            if not isinstance(record, dict): continue
            if len(found_columns) == len(expected_columns_set): break

            for col_name in expected_columns_set:
                if col_name in found_columns: continue

                # Check for dot notation first (e.g., "Created By.id")
                if '.' in col_name:
                    value = self._get_nested_value(record.get('fields', {}), col_name)
                    if value is not None:
                        lookup_map[col_name] = value
                        found_columns.add(col_name)
                        continue

                # Check top-level fields
                if col_name in record:
                    lookup_map[col_name] = record[col_name]
                    found_columns.add(col_name)
                # Check nested 'fields'
                elif 'fields' in record and col_name in record['fields']:
                    lookup_map[col_name] = record['fields'][col_name]
                    found_columns.add(col_name)

        print(f"Found example values for {len(lookup_map)} out of {len(expected_columns_set)} expected columns.", file=sys.stderr)
        return lookup_map

    def _build_rules(self, expected_columns, lookup_map, prefix):
        """Builds Bronze and Silver rules from the master list and lookup map."""
        bronze_rules, silver_rules = {}, {}

        for column_path in expected_columns:
            # Determine if the path indicates a nested field for naming purposes
            is_nested_in_fields = column_path.lower() not in ['id', 'createdtime']

            # Construct the target name with the correct format
            if is_nested_in_fields:
                target_name = f"{prefix}_FIELDS_{re.sub(r'[^A-Z0-9_]+', '_', column_path.upper()).strip('_')}"
            else:
                target_name = f"{prefix}_{re.sub(r'[^A-Z0-9_]+', '_', column_path.upper()).strip('_')}"

            # --- Bronze Rule ---
            bronze_rule = {"path": column_path, "target_name": target_name, "type": "STRING"}
            bronze_rules[target_name] = self._create_versioned_entry(bronze_rule)

            # --- Silver Rule ---
            example_value = lookup_map.get(column_path)
            silver_rule = {"path": target_name, "target_name": target_name, "type": self._get_silver_type(example_value)}
            if re.search(r'date|time|timestamp', column_path, re.IGNORECASE):
                silver_rule["type"] = "TIMESTAMP_NTZ(6)"
            elif isinstance(example_value, list):
                silver_rule["array_to_str"] = True

            silver_rules[target_name] = self._create_versioned_entry(silver_rule)

        return bronze_rules, silver_rules

    def generate_from_json(self, file_path, expected_columns_path, source_id):
        """Generates the config for a JSON file using an external column list."""
        if not expected_columns_path:
            raise ValueError("An expected columns text file is required for JSON processing.")

        with open(file_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        with open(expected_columns_path, 'r', encoding='utf-8-sig') as f:
            expected_columns = sorted([line.strip() for line in f if line.strip()])

        # --- UPDATED LOGIC FOR JSON ARRAY ---
        record_list = []
        if isinstance(input_data, list):
            record_list = input_data
        elif isinstance(input_data, dict):
            # Fallback for old format: find the first list of objects
            for key, value in input_data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    record_list = value
                    break

        if not record_list:
            raise ValueError("Could not find a list of records in the JSON file.")

        # Use the filename-derived source_id as the prefix
        list_key_prefix = source_id.upper()
        lookup_map = self._build_lookup_map(record_list, set(expected_columns))

        return self._build_rules(expected_columns, lookup_map, list_key_prefix)

    def generate(self, file_path, expected_columns_path=None):
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
                bronze_rules, silver_rules = self.generate_from_json(file_path, expected_columns_path, source_id)
            # Placeholder for CSV/Parquet logic if needed in the future
            else:
                raise ValueError(f"Unsupported file type '{extension}'")
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}", file=sys.stderr)
            return None

        silver_rules.update(silver_audit_fields)
        final_config["structure_config"][source_id.upper()] = {
            "bronze": bronze_rules,
            "silver": silver_rules
        }
        return final_config

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_config_generation_json_csv_parquet_v2.py <path_to_data_file> <path_to_expected_columns.txt>", file=sys.stderr)
        input_file_path = 'vedio_productionlist_records_2025062303.json'
        expected_columns_file = 'vedio_productionlist_records_2025062303_columns.txt'
        if not os.path.exists(input_file_path):
            print("Usage: python generate_config.py <path_to_data_file> [path_to_expected_columns.txt]", file=sys.stderr)
            sys.exit(1)
    else:
        input_file_path = sys.argv[1]
        # Check for the optional second argument
        expected_columns_file = sys.argv[2] if len(sys.argv) > 2 else None

    generator = StructureConfigGenerator()
    generated_config = generator.generate(input_file_path, expected_columns_file)

    if generated_config:
        out_file_name = os.path.splitext(input_file_path)[0] + "_struct_config.json"

        source_key = list(generated_config['structure_config'].keys())[0]
        final_bronze_count = len(generated_config['structure_config'][source_key]['bronze'])
        print(f"Final column count before writing: {final_bronze_count}", file=sys.stderr)

        print(f"Writing configuration to: {out_file_name}", file=sys.stderr)
        with open(out_file_name, "w", encoding='utf-8') as f:
            json.dump(generated_config, f, indent=2)
        print("Done.", file=sys.stderr)

if __name__ == '__main__':
    main()