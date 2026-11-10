import sys
import os
import argparse
import uuid
import ast

def main():
    parser = argparse.ArgumentParser(description="ProtFlow Wrapper for HydraProt")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input PDBs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--hydraprot_dir', type=str, required=True, help='Path to the HydraProt repository')
    parser.add_argument('--override', type=str, default='', 
                        help='Comma-separated key=value pairs to override config (e.g., "in_channels=4,radius=4.5,include_hetatm=False")')
    args = parser.parse_args()

    # 1. Convert directories to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    hydraprot_dir = os.path.abspath(args.hydraprot_dir)

    if not os.path.isdir(hydraprot_dir):
        print(f"Error: HydraProt directory not found at {hydraprot_dir}")
        sys.exit(1)

    # 2. Isolate temporary files as a hidden folder inside the output directory
    run_id = uuid.uuid4().hex[:8]
    custom_temp_dir = os.path.join(output_dir, f".hydraprot_temp_{run_id}")
    os.makedirs(custom_temp_dir, exist_ok=True)

    # 3. Update sys.path and change working directory so HydraProt finds its checkpoints
    sys.path.insert(0, hydraprot_dir)
    original_cwd = os.getcwd()
    os.chdir(hydraprot_dir)

    # 4. Import config into memory
    try:
        from params.prediction_params import config
    except ImportError:
        print("Error: Could not import HydraProt modules. Check --hydraprot_dir.")
        os.rmdir(custom_temp_dir)
        sys.exit(1)

    # 5. Set the required dynamic parameters
    config.pdb_path = input_dir if input_dir.endswith('/') else input_dir + '/'
    config.pdb_list_path = config.pdb_path
    config.results_dir = output_dir if output_dir.endswith('/') else output_dir + '/'
    
    config.unet_results_dir = f"{custom_temp_dir}/unet_prediction_waters/"
    config.mlp_embedding_dir = f"{custom_temp_dir}/mlp_embedding/"

    print("Starting HydraProt Wrapper...")

    # 6. Process custom overrides
    if args.override:
        print("\n--- Applying Custom Overrides ---")
        for pair in args.override.split(','):
            if '=' in pair:
                key, val_str = pair.split('=', 1)
                key = key.strip()
                val_str = val_str.strip()
                
                try:
                    # Safely evaluate strings into ints, floats, bools, or lists
                    parsed_val = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    # If it's just a normal string that can't be evaluated, keep it as a string
                    parsed_val = val_str

                # Warn if trying to set an attribute that doesn't exist in the original config
                if hasattr(config, key):
                    setattr(config, key, parsed_val)
                    print(f"[*] Set config.{key} = {parsed_val} (Type: {type(parsed_val).__name__})")
                else:
                    print(f"[!] Warning: config has no attribute '{key}'. Skipping.")
            else:
                print(f"[!] Warning: Invalid override format '{pair}'. Expected key=value.")
        print("---------------------------------\n")

    # 7. Monkey-patch os.system to intercept the hardcoded temp deletion
    original_os_system = os.system
    def safe_system(cmd):
        if cmd == 'rm ./temp/* -rf':
            original_os_system(f'rm -rf {custom_temp_dir}/*')
        else:
            original_os_system(cmd)
            
    os.system = safe_system

    # 8. Import and execute the main prediction logic
    import predict
    
    print(f"Input: {config.pdb_path}")
    print(f"Output: {config.results_dir}")
    print(f"Temp Dir: {custom_temp_dir}\n")
    
    try:
        predict.main()
    finally:
        # 9. Clean up and securely restore state
        original_os_system(f'rm -rf {custom_temp_dir}')
        os.system = original_os_system
        os.chdir(original_cwd)

if __name__ == '__main__':
    main()