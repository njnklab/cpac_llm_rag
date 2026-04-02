"""YAML Updater Module for C-PAC Configuration

This module provides utilities to update C-PAC YAML configuration files based on JSON input.

Two methods are available:

1. process_and_save() [LEGACY]
   - Input: Complete configuration JSON (nested structure matching YAML)
   - Use case: When LLM generates a full configuration object
   - Example JSON format:
     {
       "anatomical_preproc": {
         "n4_bias_field_correction": {"run": "On"}
       }
     }

2. process_modifications_and_save() [NEW - RECOMMENDED]
   - Input: Modifications JSON with parameter_path and recommended_value
   - Use case: When LLM only suggests parameters that need to be changed
   - Example JSON format:
     {
       "modifications": [
         {
           "parameter_path": "anatomical_preproc.n4_bias_field_correction.run",
           "recommended_value": "On",
           "rationale": "..."
         }
       ]
     }
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import copy


class YmlUpdaterFromJSON:
    """Updates a C-PAC YAML configuration from JSON input.
    
    Supports two JSON formats:
    - Legacy: Complete nested configuration (use process_and_save)
    - New: Modifications list with parameter paths (use process_modifications_and_save)
    """

    def __init__(self, yml_template_path: str):
        self.yml_template_path = Path(yml_template_path)
        self.yml_data = self._load_yml_template()

    def _load_yml_template(self) -> Dict[str, Any]:
        """Loads the base YAML template file."""
        try:
            with open(self.yml_template_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load YAML template {self.yml_template_path}: {e}")

    def _load_json_config(self, json_path: str) -> Dict[str, Any]:
        """Loads the source JSON configuration file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON config {json_path}: {e}")

    def _set_nested_value(self, data: Dict, path: list, value: Any):
        """Sets a value in a nested dictionary based on a path list."""
        current = data
        for key in path[:-1]:
            if key not in current or not isinstance(current.get(key), dict):
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _convert_value(self, value: Any) -> Any:
        """Converts JSON string values to appropriate YAML types (bool, int, float)."""
        if isinstance(value, str):
            if value.lower() in ['true', 'on']:
                return True
            if value.lower() in ['false', 'off']:
                return False
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if isinstance(value, list):
            return [self._convert_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        return value

    def _recursive_update(self, yml_dict: Dict, json_dict: Dict, current_path: list):
        """Recursively traverses the JSON and updates the YAML dictionary."""
        converted_json = self._convert_value(json_dict)
        for key, value in converted_json.items():
            new_path = current_path + [key]
            if isinstance(value, dict):
                self._recursive_update(yml_dict, value, new_path)
            else:
                print(f"  - Updating '{'.'.join(new_path)}' to: {value}")
                self._set_nested_value(yml_dict, new_path, value)

    def _save_updated_yml(self, updated_data: Dict, output_path: str):
        """Saves the updated dictionary to a YAML file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(updated_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)
            print(f"\n\u2713 Successfully saved updated YAML to: {output_path}")
        except Exception as e:
            raise IOError(f"Failed to save YAML file {output_path}: {e}")

    # =========================================================================
    # LEGACY METHOD: For complete nested configuration JSON
    # =========================================================================
    def process_and_save(self, config_json_path: str, output_yml_path: str):
        """[LEGACY] Process a complete nested configuration JSON and update YAML.
        
        This method expects a JSON file with a nested structure that mirrors
        the YAML configuration structure. All parameters in the JSON will be
        applied to the YAML template.
        
        Args:
            config_json_path: Path to the complete configuration JSON file
            output_yml_path: Path where the updated YAML will be saved
            
        Example JSON format:
            {
              "anatomical_preproc": {
                "n4_bias_field_correction": {"run": "On"}
              }
            }
        """
        print(f"\n--- Starting YAML update process (LEGACY MODE) ---")
        print(f"Template YAML: {self.yml_template_path}")
        print(f"Config JSON:   {config_json_path}")

        # Load the JSON config from the LLM
        json_config = self._load_json_config(config_json_path)

        # Create a deep copy of the template to modify
        updated_yml = copy.deepcopy(self.yml_data)

        # Iterate through the top-level modules in the JSON (e.g., 'anatomical_preproc')
        for module_name, module_config in json_config.items():
            if module_name in updated_yml:
                print(f"\nProcessing module: '{module_name}'...")
                self._recursive_update(updated_yml, {module_name: module_config}, [])
            else:
                print(f"\nWarning: Module '{module_name}' from JSON not found in YAML template. Skipping.")

        # Save the final YAML file
        self._save_updated_yml(updated_yml, output_yml_path)
        print(f"--- YAML update process complete ---")

    # =========================================================================
    # NEW METHOD: For modifications-based JSON (recommended)
    # =========================================================================
    def process_modifications_and_save(self, modifications_json_path: str, output_yml_path: str):
        """[NEW] Process a modifications JSON and update only the specified parameters.
        
        This method expects a JSON file with a 'modifications' list, where each
        item specifies a parameter_path and recommended_value. Only the listed
        parameters will be modified; all other parameters keep their default values.
        
        Args:
            modifications_json_path: Path to the modifications JSON file
            output_yml_path: Path where the updated YAML will be saved
            
        Example JSON format:
            {
              "dataset_info": {...},
              "modifications": [
                {
                  "parameter_path": "anatomical_preproc.n4_bias_field_correction.run",
                  "current_default": "Off",
                  "recommended_value": "On",
                  "confidence": "high",
                  "rationale": "N4 bias field correction improves segmentation..."
                }
              ],
              "summary": {...}
            }
        """
        print(f"\n--- Starting YAML update process (MODIFICATIONS MODE) ---")
        print(f"Template YAML: {self.yml_template_path}")
        print(f"Modifications JSON: {modifications_json_path}")

        # Load the modifications JSON
        json_data = self._load_json_config(modifications_json_path)
        
        # Extract the modifications list
        modifications = json_data.get("modifications", [])
        if not modifications:
            print("\nWarning: No modifications found in JSON. Output will be identical to template.")
        
        # Create a deep copy of the template to modify
        updated_yml = copy.deepcopy(self.yml_data)
        
        # Track successful and failed updates
        success_count = 0
        failed_updates = []
        
        print(f"\nApplying {len(modifications)} modification(s)...")
        
        # Process each modification
        for mod in modifications:
            param_path = mod.get("parameter_path", "")
            recommended_value = mod.get("recommended_value")
            confidence = mod.get("confidence", "unknown")
            rationale = mod.get("rationale", "")[:80]  # Truncate for display
            
            if not param_path:
                print(f"  - [SKIP] Empty parameter_path in modification")
                continue
            
            # Convert path string to list (e.g., "a.b.c" -> ["a", "b", "c"])
            path_list = param_path.split(".")
            
            # Convert the value to appropriate type
            converted_value = self._convert_value(recommended_value)
            
            try:
                # Set the value in the YAML structure
                self._set_nested_value(updated_yml, path_list, converted_value)
                print(f"  - [OK] {param_path} = {converted_value} (confidence: {confidence})")
                success_count += 1
            except Exception as e:
                print(f"  - [FAIL] {param_path}: {e}")
                failed_updates.append((param_path, str(e)))
        
        # Summary
        print(f"\n--- Update Summary ---")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {len(failed_updates)}")
        if failed_updates:
            for path, error in failed_updates:
                print(f"    - {path}: {error}")
        
        # Save the final YAML file
        self._save_updated_yml(updated_yml, output_yml_path)
        print(f"--- YAML update process complete ---")
        
        return {
            "success_count": success_count,
            "failed_count": len(failed_updates),
            "failed_updates": failed_updates
        }


# Example usage for command-line execution
def main():
    YML_TEMPLATE_PATH = "/home/a001/zhangyan/cpac/llm_parameters/pipeline_config_default.yml"
    # This is the output from generate_config.py
    CONFIG_JSON_PATH = "/home/a001/zhangyan/cpac/llm_parameters/20250825全参数整理/main_generated_config.json"
    OUTPUT_YML_PATH = "/home/a001/zhangyan/cpac/llm_parameters/4修改yml/final_pipeline_config.yml"

    try:
        updater = YmlUpdaterFromJSON(YML_TEMPLATE_PATH)
        updater.process_and_save(CONFIG_JSON_PATH, OUTPUT_YML_PATH)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
