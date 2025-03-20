import numpy as np
import math
import matplotlib.pyplot as plt
import random
from abc import ABC, abstractmethod
from memory_status import ModelConfig, MemStatus, HBMInit, TokenLevelBestRatioInit
from placement import BaseStrategy, PreferHBM, SplitToken, BatchRatio, LookAheadBatch, LayerImportance, AlphaLayersDistribution
from migration import BaseDataMigration, NoMigration, PriorMigration, SkippedTokensMigration, PastWindowMigration, LookAheadMigration, LookAheadBatchMigration, AlphaMigration
import copy
import csv
import re

BYTES_TO_GB = 1024**3

# def load_trace(filename="trace.txt"):
#     trace = {}
#     pattern = re.compile(r"^([^,]+),([^,]+),([^,]+),(\[.*?\]),(.+)$")

#     with open(filename, "r") as f:
#         for line in f:
#             line = line.strip()
#             m = pattern.match(line)
#             if m:
#                 n, l, s, skip_token_kv_str, skip_layer_str = m.groups()
#                 n, l, s = int(n), int(l), int(s)
#                 skip_token_kv = eval(skip_token_kv_str)  # Note: using eval can be dangerous if the file is untrusted
#                 skip_layer = skip_layer_str.strip().lower() == "true"
#                 trace[(n, l, s)] = {"skip_token_kv": skip_token_kv, "skip_layer": skip_layer}
#             else:
#                 raise ValueError("Line doesn't match expected format: " + line)
#     return trace

def load_skip_lists(filename="trace.txt"):
    trace = {}
    pattern = re.compile(r"^([^,]+),([^,]+),([^,]+),(\[.*?\]),(.+)$")

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if m:
                n, l, s, skip_token_kv_str, skip_layer_str = m.groups()
                n, l, s = int(n), int(l), int(s)
                if l == 0 and s == 0:
                    skip_token_kv = eval(skip_token_kv_str)
                    trace[n] = skip_token_kv
            else:
                raise ValueError("Line doesn't match expected format: " + line)
    return trace

class MemorySimulator(ABC):
    def __init__(self, config: ModelConfig, status: MemStatus,
                placement: BaseStrategy, migration: BaseDataMigration, best: bool = False):
        self.cfg = config
        self.plc = placement
        self.mig = migration
        self.status = status
        self.best = best
        self.total_time = 0.0
        self.step_details = []

    def calculate_step_time(self, n: int, l: int, s: int, 
                       alpha, beta: float, 
                       hbm_MR: float, hbm_MW: float,
                       ext_MR: float, ext_MW: float):
        """Calculate time consumption for one step"""
        # Calculate data sizes
        D_R, D_W = self.status.calculate_data_sizes(n, l, s)
        # should modify the way to calculate time, inclusive / exclusive
        # represents the ratio of KV cache.
        # Calculate HBM time
        beta_ext = 0.0
        if self.status.inclusive:
            beta_ext = 1.0
        else:
            beta_ext = 1 - beta

        if self.best:
            alpha = min(self.cfg.C_HBM_max / D_R, self.cfg.best_alpha)

        HBM_read = alpha * D_R
        HBM_write = beta * D_W
        HBM_migration = hbm_MR + hbm_MW
        T_HBM = (HBM_read + HBM_write + HBM_migration) / self.cfg.B_HBM  # in ns
        
        
        # New external read calculation using interface vs internal minimum
        ext_read = (1 - alpha) * D_R / min(self.cfg.B_ext_interface_R, 
                                        self.cfg.B_ext_internal)
        
        # New write + migration calculation
        write_migration = 0.0
        internal_migration = 0.0

        write_migration = (beta_ext * D_W + ext_MW) / self.cfg.B_ext_interface_W if self.cfg.B_ext_interface_R > 0 else 0
        internal_migration = (beta_ext * D_W + ext_MW + ext_MR) / self.cfg.B_ext_internal if self.cfg.B_ext_internal > 0 else 0
        read_migration = ext_MR / self.cfg.B_ext_interface_R if self.cfg.B_ext_interface_R > 0 else 0
        
        ext_write_migration = max(write_migration, read_migration, internal_migration)
        
        T_ext = ext_read + ext_write_migration
        
        return max(T_HBM, T_ext)
    
    def simulate(self):
        """
        Run full simulation
        Args:
            alpha_strategy: Function(n,l,s) -> alpha
            beta_strategy: Function(n,l,s) -> beta
            migration_strategy: Function(n,l,s) -> (D_MR, D_MW)
        """
        self.total_time = 0.0
        self.step_details = []

        for n in range(self.cfg.N_pre, self.cfg.N_pre + self.cfg.N):
            for l in range(self.cfg.L):
                for s in [0, 1]:  # MHA and MLP
                    step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
                    # If this layer is skipped in the trace, no data is processed.
                    if step_info["skip_layer"]:
                        continue
                    # Get strategies
                    alpha = self.plc.alpha_strategy(n, l, s)
                    beta = self.plc.beta_strategy(n, l, s)
                    migration_data = self.mig.migration_strategy(n, l, s)
                    
                    # Calculate step time
                    step_time = self.calculate_step_time(n, l, s, alpha, beta, 
                                                         migration_data[0], migration_data[1],
                                                         migration_data[2], migration_data[3])
                    self.total_time += step_time
                    
                    # Record step details
                    self.step_details.append({
                        'n': n,
                        'l': l,
                        's': s,
                        'time': step_time,
                        'alpha': alpha,
                        'beta': beta
                    })
        return self.total_time

# simulator.py (updated run_simulation function)
def run_simulation(init_class: MemStatus, config_params: dict, 
                  mig_classes: list, plc_classes: list):
    """Run simulation with specified initialization class and config parameters"""
    fn = config_params.get('filename', "trace.txt")
    inclusive = config_params.get('inclusive', False)
    
    # Create config with custom parameters
    config = ModelConfig(
        N=config_params.get('N', 1024*10),
        N_pre=config_params.get('N_pre', 1024*2),
        para_num=config_params.get('para_num', 0.5),
        C_HBM_max=config_params.get('C_HBM_max', 3)
    )
    
    # 🔥 Use passed strategy classes instead of hardcoded
    placement_classes = plc_classes
    migration_classes = mig_classes

    # Run simulation for this initialization class
    config_temp = copy.deepcopy(config)
    trace = load_skip_lists(fn)
    initial_state_clean = init_class(config_temp, trace, inclusive)
    
    # Rest of the original simulation logic...
    initial_state_temp = copy.deepcopy(initial_state_clean)
    initial_state_temp.trace = trace
    best_mig = NoMigration(initial_state_temp.cfg, initial_state_temp)
    best_plc = PreferHBM(initial_state_temp.cfg, initial_state_temp)
    best_simulator = MemorySimulator(initial_state_temp.cfg, initial_state_temp, best_plc, best_mig, best=True)
    upper_bound_time = best_simulator.simulate()
    
    print(f"Read trace file: {fn}")
    print(f"Best Combination:")
    print(f"Total simulation time: {upper_bound_time:.4f} ns, {upper_bound_time/1e9:.4f} seconds")
    print(f"Average time per token: {upper_bound_time/initial_state_temp.cfg.N:.6f} ns")
    print("-" * 50)

    # 🔥 Use passed strategy classes in the loops
    for p_cls in placement_classes:
        for m_cls in migration_classes:
            test_initial_state = copy.deepcopy(initial_state_clean)
            test_initial_state.trace = trace
            mig_instance = m_cls(test_initial_state.cfg, test_initial_state)
            placement_instance = p_cls(test_initial_state.cfg, test_initial_state)
            
            simulator = MemorySimulator(test_initial_state.cfg, test_initial_state, 
                                      placement_instance, mig_instance, best=False)
            total_time = simulator.simulate()
            avg_alpha = sum(step['alpha'] for step in simulator.step_details) / len(simulator.step_details)
            
            print(f"Combination: {p_cls.__name__} + {m_cls.__name__}")
            print(f"Total time: {total_time:.4f} ns, {total_time/1e9:.4f} seconds")
            print(f"Avg alpha: {avg_alpha:.6f}")
            print(f"Alphas:")
            for step in simulator.step_details:
                print(f"{step['alpha']:.4f}")
            print("-" * 50)
    
    return

# simulator.py (add this at the end)
if __name__ == "__main__":
    import argparse
    from argparse import Namespace
    # Redirect output to log file
    import sys

    # Mapping from string names to actual classes
    CLASS_MAPPING = {
        # Initialization classes
        'HBMInit': HBMInit,
        'TokenLevelBestRatioInit': TokenLevelBestRatioInit,
        
        # Migration classes
        'NoMigration': NoMigration,
        'AlphaMigration': AlphaMigration,
        'LookAheadBatchMigration': LookAheadBatchMigration,
        'LookAheadMigration': LookAheadMigration,
        'PriorMigration': PriorMigration,
        'PastWindowMigration': PastWindowMigration,
        # Placement classes
        'PreferHBM': PreferHBM,
        'BatchRatio': BatchRatio,
        'LookAheadBatch': LookAheadBatch,
        'LayerImportance': LayerImportance,
        'AlphaLayersDistribution': AlphaLayersDistribution,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1024*10)
    parser.add_argument('--N_pre', type=int, default=1024*2)
    parser.add_argument('--para_num', type=float, default=0.5)
    parser.add_argument('--C_HBM_max', type=int, default=3)
    parser.add_argument('--inclusive', type=bool, default=False)
    parser.add_argument('--filename', type=str, default="trace.txt")
    parser.add_argument('--init_class', type=str, required=True, 
                       help='Initialization class name')
    parser.add_argument('--mig_classes', type=str, nargs='+', required=True,
                       help='Migration class names separated by spaces')
    parser.add_argument('--plc_classes', type=str, nargs='+', required=True,
                       help='Placement class names separated by spaces')
    parser.add_argument('--log_file', type=str, default="simulation.txt")
    args = parser.parse_args()

    # Validate and convert class names to actual classes
    try:
        init_class = CLASS_MAPPING[args.init_class]
        mig_classes = [CLASS_MAPPING[name] for name in args.mig_classes]
        plc_classes = [CLASS_MAPPING[name] for name in args.plc_classes]
    except KeyError as e:
        print(f"Error: Unknown class name {e.args[0]}")
        sys.exit(1)

    config_params = {
        'N': args.N,
        'N_pre': args.N_pre,
        'para_num': args.para_num,
        'C_HBM_max': args.C_HBM_max,
        'filename': args.filename,
        'inclusive': args.inclusive
    }

    
    with open(args.log_file, 'w') as f:
        sys.stdout = f
        try:
            run_simulation(
                init_class=init_class,
                config_params=config_params,
                mig_classes=mig_classes,
                plc_classes=plc_classes
            )
        except Exception as e:
            print(f"Simulation failed: {str(e)}")
        sys.stdout = sys.__stdout__


    # # Initialize configuration
    # filename = "trace.txt"
    # config = ModelConfig()

    # # add bset time calculation
    
    # # List of placement strategies from placement.py.
    # # placement_classes = [PreferHBM, SplitToken, BatchRatio, LookAheadBatch, LayerImportance]
    # placement_classes = [PreferHBM, BatchRatio]
    # # List of migration strategies from migration.py.
    # # migration_classes = [NoMigration, PriorMigration, SkippedTokensMigration, PastWindowMigration, LookAheadMigration, LookAheadBatchMigration, AlphaMigration]
    # migration_classes = [NoMigration, AlphaMigration]
    # initialization_classes = [HBMInit, TokenLevelBestRatioInit]
    # # HBMInit, 
    # # Iterate over each combination of placement and migration strategy.
    # for init in initialization_classes:
    #     config_temp = copy.deepcopy(config)
    #     initial_state_clean = init(config_temp, filename, config_temp.best_alpha, is_inclusive=False)    
    #     initial_state_temp = copy.deepcopy(initial_state_clean)
    #     # initial_state = init(config, filename, config.best_alpha, is_inclusive=False)
    #     # coarsed upper bound
    #     best_mig = NoMigration(initial_state_temp.cfg, initial_state_temp)
    #     best_plc = PreferHBM(initial_state_temp.cfg, initial_state_temp)
    #     best_simulator = MemorySimulator(initial_state_temp.cfg, initial_state_temp, best_plc, best_mig, best=True)
    #     upper_bound_time = best_simulator.simulate()
        
    #     print(f"Best Combination:")
    #     print(f"Total simulation time: {upper_bound_time:.4f} ns, {upper_bound_time/1e9:.4f} seconds")
    #     print(f"Average time per token: {upper_bound_time/initial_state_temp.cfg.N:.6f} ns")
    #     print("-" * 50)

    #     for p_cls in placement_classes:
    #         for m_cls in migration_classes:
    #             # Clone the initial_state instead of re-initializing
    #             test_initial_state = copy.deepcopy(initial_state_clean)
    #             # test_initial_state = init(config, filename, config.best_alpha, is_inclusive=False)
    #             # Instantiate migration instance (pass config and temporary strategy placeholder).
    #             mig_instance = m_cls(test_initial_state.cfg, test_initial_state)
    #             # Instantiate placement instance, passing the migration instance.
    #             placement_instance = p_cls(test_initial_state.cfg, test_initial_state)
                
    #             # Create the simulator with this combined strategy.
    #             simulator = MemorySimulator(test_initial_state.cfg, test_initial_state, placement_instance, mig_instance, best=False)
    #             total_time = simulator.simulate()
    #             avg_alpha = sum(step['alpha'] for step in simulator.step_details) / len(simulator.step_details)
                
    #             print(f"Combination: Data Placement = {p_cls.__name__}, Data Migration = {m_cls.__name__}")
    #             print(f"Total simulation time: {total_time:.4f} ns, {total_time/1e9:.4f} seconds")
    #             print(f"Average time per token: {total_time/test_initial_state.cfg.N:.6f} ns")
    #             print(f"Avg alpha rate: {avg_alpha:.6f}")
    #             print("-" * 50)
    

    # times = [step['alpha'] for step in simulator.step_details]
    # plt.plot(times)
    # plt.xlabel('Step')
    # plt.ylabel('Time (ns)')
    # plt.title('Step Time Distribution')
    # plt.show()