# run_experiments.py (new file)
import subprocess
import time
from datetime import datetime

experiments = [
    {
        'N': 1024*4,
        'N_pre': 512,
        'para_num': 0.25,
        'C_HBM_max': 1,
        'filename': '03_05_4.txt',
        'init_class': 'HBMInit',
        # 'mig_classes': ['NoMigration', 'AlphaMigration'],
        'mig_classes': ['LookAheadBatchMigration', ],
        # 'plc_classes': ['PreferHBM', 'BatchRatio']
        'plc_classes': ['LookAheadBatch', 'AlphaLayersDistribution']
    }
    # {
    #     'N': 1024*10,
    #     'N_pre': 1024*2,
    #     'para_num': 0.5,
    #     'C_HBM_max': 3,
    #     'filename': '03_2_10.txt',
    #     'init_class': 'TokenLevelBestRatioInit',
    #     'mig_classes': ['NoMigration', 'AlphaMigration'],
    #     'plc_classes': ['PreferHBM', 'BatchRatio']
    # },
    # {
    #     'N': 1024*10,
    #     'N_pre': 1024*2,
    #     'para_num': 0.5,
    #     'C_HBM_max': 3,
    #     'filename': '06_2_10.txt',
    #     'init_class': 'HBMInit',
    #     'mig_classes': ['NoMigration', 'AlphaMigration'],
    #     'plc_classes': ['PreferHBM', 'BatchRatio']
    # },
    # {
    #     'N': 1024*10,
    #     'N_pre': 1024*2,
    #     'para_num': 0.5,
    #     'C_HBM_max': 3,
    #     'filename': '06_2_10.txt',
    #     'init_class': 'TokenLevelBestRatioInit',
    #     'mig_classes': ['NoMigration', 'AlphaMigration'],
    #     'plc_classes': ['PreferHBM', 'BatchRatio']
    # }
    # 03_2_1
    # {
    #     'N': 1024*1,
    #     'N_pre': 1024*2,
    #     'para_num': 0.5,
    #     'C_HBM_max': 3,
    #     'filename': '03_2_1.txt',
    #     'init_class': 'HBMInit',
    #     'mig_classes': ['NoMigration', 'AlphaMigration'],
    #     'plc_classes': ['PreferHBM', 'BatchRatio']
    # },
    # {
    #     'N': 1024*1,
    #     'N_pre': 1024*2,
    #     'para_num': 0.5,
    #     'C_HBM_max': 3,
    #     'filename': '03_2_1.txt',
    #     'init_class': 'TokenLevelBestRatioInit',
    #     'mig_classes': ['NoMigration', 'AlphaMigration'],
    #     'plc_classes': ['PreferHBM', 'BatchRatio']
    # },
    # {
    #     'N': 1024*1,
    #     'N_pre': 1024*2,
    #     'para_num': 0.5,
    #     'C_HBM_max': 3,
    #     'filename': '06_2_1.txt',
    #     'init_class': 'HBMInit',
    #     'mig_classes': ['NoMigration', 'AlphaMigration'],
    #     'plc_classes': ['PreferHBM', 'BatchRatio']
    # },
    # {
    #     'N': 1024*1,
    #     'N_pre': 1024*2,
    #     'para_num': 0.5,
    #     'C_HBM_max': 3,
    #     'filename': '06_2_1.txt',
    #     'init_class': 'TokenLevelBestRatioInit',
    #     'mig_classes': ['NoMigration', 'AlphaMigration'],
    #     'plc_classes': ['PreferHBM', 'BatchRatio']
    # }
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