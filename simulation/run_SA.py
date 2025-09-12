import subprocess
import time
from datetime import datetime

# scaled LLaMA-3-8B
experiments = [
    # {
    #     'para_num': 0.0078125,
    #     'C_HBM_max': 0.0234375,
    #     'inclusive': True,
    #     'best': False,
    #     'filename': './data/qasper/quasper_01_40.csv',
    #     'sparsity': 0.4,
    # },
    # Add more experiment configurations as needed

    {
        'para_num': 8,
        'C_HBM_max': 25,
        'B_ext_R': 300,
        'B_ext_W': 300,
        'inclusive': True,
        'best': False,
        'filename': 'data/narativeqa/narativeqa_60.csv',
        'sparsity': 0.60,
    },
]

def run_experiment(config):
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = (f"SA_{config['para_num']}B_"
                f"{config['C_HBM_max']}GB_"
                f"{config['sparsity']}_"
                f"{timestamp}.txt")
    
    # Build command
    cmd = [
        'python', 'simulation/SA_simulation.py',
        '--para_num', str(config['para_num']),
        '--C_HBM_max', str(config['C_HBM_max']),
        '--filename', str(config['filename']),
        '--best', str(config['best']),
        '--inclusive', str(config['inclusive']),
        '--log_file', log_name
    ]
    
    # Run in separate process
    print(f"Starting SA experiment: {log_name}")
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
    for config in experiments:
        run_experiment(config)
        print("="*80)