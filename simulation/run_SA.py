import subprocess
import time
from datetime import datetime
import json
import argparse
import sys

# --- Configurations to tune ---
# Note: C_HBM_max should be small enough to force competition for HBM
experiments = [
    {
        'setting': "Mixtral Adaptive",
        'para_num': 46.7,      # Mixtral 8x7B equivalent unique params
        'C_HBM_max': 90,       # 10 GB HBM to force cache competition
        'B_ext_R': 450,        # Standard BW
        'B_ext_W': 450,
        'filename': '../data/gov_report/mixtral_gov_16_4096_8192_60.csv', # Placeholder trace
        'inclusive': True,
        'best': False,
        'n_splits': 32,
        'max_iter': 5,
        'initial_window': 12
    },
]

def run_experiment(config):
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = (f"SA_ADAPTIVE_"
                f"{config['C_HBM_max']}GB_"
                f"{config['B_ext_R']}R_"
                f"{timestamp}.txt")
    
    # Build command for the adaptive_tuner.py script
    cmd = [
        'python', 'SA_simulation.py',
        '--para_num', str(config['para_num']),
        '--C_HBM_max', str(config['C_HBM_max']),
        '--B_ext_R', str(config['B_ext_R']),
        '--B_ext_W', str(config['B_ext_W']),
        '--filename', str(config['filename']),
        '--best', str(config['best']),
        '--inclusive', str(config['inclusive']),
        '--log_file', log_name,
        '--n_splits', str(config['n_splits']),
        '--max_iter', str(config['max_iter']),
        '--initial_window', str(config['initial_window'])
    ]
    
    # Run in separate process
    print(f"\nStarting SA Tuning Experiment: {config['setting']}")
    print(f"Log file: {log_name}")
    print(f"trace file: {config['filename']}\n")
    
    # Use subprocess.run to capture the output correctly
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Completed successfully. Output logged to {log_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {log_name}")
        print(f"--- Stdout ---\n{e.stdout}")
        print(f"--- Stderr ---\n{e.stderr}")
        with open(f"ERROR_{log_name}", 'w') as f:
            f.write(f"Stdout:\n{e.stdout}\n\nStderr:\n{e.stderr}")
    
    # Add cooling period
    time.sleep(10)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Note: Ignoring command-line arguments and using hardcoded 'experiments' list.")
        
    for config in experiments:
        run_experiment(config)
        print("="*80)
        time.sleep(30) # Longer break between major experiments

    print("All adaptive tuning experiments completed!")