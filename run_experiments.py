# run_experiments.py (new file)
import subprocess
import time
from datetime import datetime

experiments = [
    # HBM enough
    {
        'N': 1024*2,
        'N_pre': 512,
        'para_num': 0.1,
        'C_HBM_max': 2,
        'filename': '00_05_2.txt',
        'init_class': 'HBMInit',
        'mig_classes': ['NoMigration', 'AlphaMigration', 'PastWindowMigration'],
        'plc_classes': ['PreferHBM', 'AlphaLayersDistribution']
    },
    {
        'N': 1024*2,
        'N_pre': 512,
        'para_num': 0.1,
        'C_HBM_max': 2,
        'filename': '04_05_2.txt',
        'init_class': 'HBMInit',
        'mig_classes': ['NoMigration', 'AlphaMigration', 'PastWindowMigration'],
        'plc_classes': ['PreferHBM', 'AlphaLayersDistribution']
    },
    # HBM not enough
    {
        'N': 1024*8,
        'N_pre': 512,
        'para_num': 0.1,
        'C_HBM_max': 2,
        'filename': '00_05_8.txt',
        'init_class': 'HBMInit',
        'mig_classes': ['NoMigration', 'AlphaMigration', 'PastWindowMigration'],
        'plc_classes': ['PreferHBM', 'AlphaLayersDistribution']
    },
    {
        'N': 1024*8,
        'N_pre': 512,
        'para_num': 0.1,
        'C_HBM_max': 2,
        'filename': '04_05_8.txt',
        'init_class': 'HBMInit',
        'mig_classes': ['NoMigration', 'AlphaMigration', 'PastWindowMigration'],
        'plc_classes': ['PreferHBM', 'AlphaLayersDistribution']
    }
    
]

def run_experiment(config):
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = (f"{config['N_pre']}_{config['N']}_"
                f"{config['para_num']}B_"
                f"{config['C_HBM_max']}GB_"
                f"{config['init_class']}_"
                f"{timestamp}.txt")
    
    # Build command
    cmd = [
        'python', 'simulator.py',
        '--N', str(config['N']),
        '--N_pre', str(config['N_pre']),
        '--para_num', str(config['para_num']),
        '--C_HBM_max', str(config['C_HBM_max']),
        '--filename', str(config['filename']),
        '--init_class', config['init_class'],
        '--mig_classes', *config['mig_classes'],
        '--plc_classes', *config['plc_classes'],
        '--log_file', log_name
    ]
    
    # Run in separate process
    print(f"Starting experiment: {log_name}")
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