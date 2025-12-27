import subprocess
import time
import argparse
import json
from datetime import datetime

def load_experiment_config(json_path):
    """
    Load experiment configurations from the specified JSON file
    """
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
            if 'experiments' not in data:
                # Handle single experiment case
                return [data]
            print(f"Loaded {len(data['experiments'])} configurations from {json_path}")
            return data['experiments']
        except json.JSONDecodeError as e:
            raise ValueError(f"Error loading {json_path}: {e}")

def run_experiment(config):
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Removed 'sparsity' from log_name since it is now inherent in the trace data
    log_name = (f"{config['para_num']}B_"
                f"{config['C_HBM_max']}GB_"
                f"{timestamp}.txt")
    
    # Build command
    cmd = [
        'python', 'simulator.py', # Assuming simulator.py is in the current directory or Python path
        '--para_num', str(config['para_num']),
        '--C_HBM_max', str(config['C_HBM_max']),
        '--filename', str(config['filename']),
        '--init_class', config['init_class'],
        '--mig_classes', *config['mig_classes'],
        '--plc_classes', *config['plc_classes'],
        '--best', str(config['best']),
        '--inclusive', str(config['inclusive']),
        '--log_file', log_name
    ]

    # Add bandwidth parameters if they exist in config
    if 'B_ext_R' in config:
        cmd.extend(['--B_ext_R', str(config['B_ext_R'])])
    if 'B_ext_W' in config:
        cmd.extend(['--B_ext_W', str(config['B_ext_W'])])
    
    # Run in separate process
    # Print experiment info to terminal
    print(f"\nStarting experiment ({timestamp}):")
    print(f"Model: {config['para_num']}B, HBM: {config['C_HBM_max']}GB")
    print(f"File: {config['filename']}")
    print(f"Strategy: {config['init_class']} + {'+'.join(config['mig_classes'])} + {'+'.join(config['plc_classes'])}")
    if 'B_ext_R' in config:
        print(f"External Read BW: {config['B_ext_R']} GB/s")
    if 'B_ext_W' in config:
        print(f"External Write BW: {config['B_ext_W']} GB/s")
    print(f"Log file: {log_name}\n")
    
    # The original script used 'simulation/simulator.py', adjusting path here for common use
    # If your simulator.py is in a subdirectory, adjust the path in the cmd list above.
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Handle results
    if process.returncode == 0:
        print(f"Completed successfully: {log_name}")
    else:
        print(f"Failed: {log_name}")
        with open(f"ERROR_{log_name}", 'w') as f:
            f.write(stderr.decode())
    
    # Add cooling period between experiments
    time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments from JSON config')
    parser.add_argument('config_file', help='Path to JSON configuration file')
    args = parser.parse_args()
    
    configs = load_experiment_config(args.config_file)
    for idx, config in enumerate(configs, 1):
        print(f"\nRunning experiment {idx}/{len(configs)}")
        print("=" * 80)
        run_experiment(config)
        if idx < len(configs):
            print(f"Cooling period before next experiment...")
            time.sleep(30)  # Longer cooling period between experiments
    
    print("\nAll experiments completed!")