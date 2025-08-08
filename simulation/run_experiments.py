# run_experiments.py (new file)
import subprocess
import time
import argparse
import json
from datetime import datetime

def load_experiment_config(json_path):
    """
    Load a single experiment configuration from the specified JSON file
    """
    with open(json_path, 'r') as f:
        try:
            config = json.load(f)
            print(f"Loaded configuration from {json_path}")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Error loading {json_path}: {e}")

# experiments = [
#     # scaled LLaMA-3-8B 8:1
#     {
#         'para_num': 8,
#         'C_HBM_max': 23,
#         'inclusive': True,
#         'best': True,
#         'filename': 'data/narativeqa/random_low_0.60.csv',
#         'sparsity': 0.60,
#         'init_class': 'HBMInit',
#         'mig_classes': ['NoMigration'],
#         'plc_classes': ['PreferHBM']
#     },
#     {
#         'para_num': 8,
#         'C_HBM_max': 23,
#         'inclusive': True,
#         'best': False,
#         'filename': 'data/narativeqa/random_low_0.60.csv',
#         'sparsity': 0.60,
#         'init_class': 'HBMInit',
#         'mig_classes': ['NoMigration'],
#         'plc_classes': ['PreferHBM']
#     },
#     {
#         'para_num': 8,
#         'C_HBM_max': 23,
#         'inclusive': True,
#         'best': False,
#         'filename': 'data/narativeqa/random_low_0.60.csv',
#         'sparsity': 0.60,
#         'init_class': 'TokenLevelBestRatioInit',
#         'mig_classes': ['NormalMigration'],
#         'plc_classes': ['PreferHBM']
#     },
#     {
#         'para_num': 8,
#         'C_HBM_max': 23,
#         'inclusive': True,
#         'best': False,
#         'filename': 'data/narativeqa/random_low_0.60.csv',
#         'sparsity': 0.60,
#         'init_class': 'HBMInitPaged',
#         'mig_classes': ['PageMigration'],
#         'plc_classes': ['PreferHBMPaged']
#     },
    
# ]

def run_experiment(config):
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = (f"{config['para_num']}B_"
                f"{config['C_HBM_max']}GB_"
                f"{config['sparsity']}_"
                f"{timestamp}.txt")
    
    # Build command
    cmd = [
        'python', 'simulator.py',
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
    
    # Run in separate process
    # Print experiment info to terminal
    print(f"\nStarting experiment ({timestamp}):")
    print(f"Model: {config['para_num']}B, HBM: {config['C_HBM_max']}GB")
    print(f"Sparsity: {config['sparsity']}, File: {config['filename']}")
    print(f"Strategy: {config['init_class']} + {'+'.join(config['mig_classes'])} + {'+'.join(config['plc_classes'])}")
    print(f"Log file: {log_name}\n")
    process = subprocess.Popen(cmd, stdout=None, stderr=subprocess.PIPE)
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

# if __name__ == "__main__":
#     for config in experiments:
#         run_experiment(config)
#         print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a single experiment from JSON config')
    parser.add_argument('config_file', help='Path to JSON configuration file')
    args = parser.parse_args()
    
    config = load_experiment_config(args.config_file)
    run_experiment(config)