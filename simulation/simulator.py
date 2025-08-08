
import math
import random
from abc import ABC, abstractmethod
from memory_status import *
from placement import *
from migration import *
import copy
import io
import csv
import re
import ast
import sys
from tqdm.auto import tqdm

BYTES_TO_GB = 1024**3


# Add this new class at the beginning of the file
class TraceReader:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.reader = None
        self._header = None
        self.first_token = None
        self.last_token = None
        self.line_positions = {}
        self.token_cache = {}
        # prefetch how many tokens
        self.cache_window = 128
        self.current_window_start = None
        self._build_index()
    

    def _update_cache(self, n):
        """Update cache with tokens from n to n+cache_window"""
        # Clear existing cache
        self.token_cache.clear()
        self.current_window_start = n
        
        # Initialize cache structure for the window
        for token in range(n, min(n + self.cache_window, self.last_token + 1)):
            self.token_cache[token] = {}

        # Seek to the start position
        if n in self.line_positions:
            self.file.seek(self.line_positions[n])
        else:
            return  # Invalid token number

        # Read and cache all layers for each token in window
        while True:
            line = self.file.readline()
            if not line:
                break

            reader = csv.reader(io.StringIO(line))
            parts = next(reader)
            curr_n = int(parts[0])
            
            # Break if we've gone beyond our window
            if curr_n >= n + self.cache_window:
                break
                
            # Skip if token is before our window
            if curr_n < n:
                continue

            curr_l = int(parts[1])
            read_kv_str = parts[2]
            read_kv = ast.literal_eval(read_kv_str)
            
            # Store in cache
            self.token_cache[curr_n][curr_l] = read_kv
    
    def _is_in_cache(self, n, l):
        """Check if token n and layer l are in cache"""
        return (self.current_window_start is not None and 
                self.current_window_start <= n < self.current_window_start + self.cache_window and
                n in self.token_cache and 
                l in self.token_cache[n])


    
    def _build_index(self):
        """Build an index of file positions for each n value"""
        with open(self.filename, 'r') as f:
            # Read and store the header
            self._header = f.readline().strip()
            
            # Get position at the start of the first data line
            pos = f.tell()
            line = f.readline()
            if not line:  # Handle empty file after header
                return
            
            # Parse first data line
            parts = line.strip().split(',')
            n = int(parts[0])
            self.first_token = n
            self.line_positions[n] = pos  # Position before reading the line
            current_n = n
            
            # Process remaining lines
            while True:
                pos = f.tell()  # Position before reading the next line
                line = f.readline()
                if not line:  # End of file
                    break
                parts = line.strip().split(',')
                n = int(parts[0])
                if n != current_n:
                    self.line_positions[n] = pos  # Record position where new n starts
                    current_n = n
            
            self.last_token = current_n
    
    def __enter__(self):
        """Context manager entry"""
        self.file = open(self.filename, 'r')
        self.reader = csv.reader(self.file)
        self._header = next(self.reader)  # Skip and store header
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.token_cache.clear()
        if self.file:
            self.file.close()
    
    def get_read_tokens(self, n, l):
        if not self._is_in_cache(n, l):
            self._update_cache(n)
        
        # Return from cache if available
        if self._is_in_cache(n, l):
            return self.token_cache[n][l]
            
        return []  # Return empty list if not found


class MemorySimulator(ABC):
    def __init__(self, config: ModelConfig, status: MemStatus,
                 placement: BaseStrategy, migration: BaseDataMigration, 
                 best: bool = False):
        self.cfg = config
        self.plc = placement
        self.mig = migration
        self.status = status
        # Always read in maximum bandwidth
        self.best = best
        self.total_time = 0.0
        self.step_details = []

    def calculate_step_time(self, n: int, l: int, 
                       alpha, beta: float, 
                       hbm_MR: float, hbm_MW: float,
                       ext_MR: float, ext_MW: float):
        """Calculate time consumption for one step"""
        # Calculate data sizes
        D_R, D_W = self.status.calculate_data_sizes(n, l)
        # should modify the way to calculate time, inclusive / exclusive
        # represents the ratio of KV cache.
        # Calculate HBM time

        if self.best:
            alpha = min(self.cfg.C_HBM_max / D_R, self.cfg.best_alpha)

        HBM_read = alpha * D_R
        HBM_write = beta * D_W
        HBM_migration = hbm_MR + hbm_MW
        T_HBM = (HBM_read + HBM_write + HBM_migration) / self.cfg.B_HBM  # in ns
        
        
        # New external read calculation using interface vs internal minimum
        ext_read = ((1 - alpha) * D_R + ext_MR) / min(self.cfg.B_ext_interface_R, 
                                        self.cfg.B_ext_internal)
        
        # when internal bw larger than interface read
        ext_write_large = (D_W) / min(self.cfg.B_ext_interface_W, self.cfg.B_ext_internal - self.cfg.B_ext_interface_R)
        # when internal bw smaller than interface read
        ext_write_small = (D_W) / self.cfg.B_ext_internal
        
        # New write + migration calculation
        # write_migration = 0.0
        # internal_migration = 0.0

        # write_migration = (ext_MW + D_W) / self.cfg.B_ext_interface_W if self.cfg.B_ext_interface_R > 0 else 0 # delete beta_ext * D_W because can be covered in the read stage.
        # internal_migration = (ext_MW + ext_MR + D_W) / self.cfg.B_ext_internal if self.cfg.B_ext_internal > 0 else 0
        # read_migration = ext_MR / self.cfg.B_ext_interface_R if self.cfg.B_ext_interface_R > 0 else 0
        
        # ext_write_migration = max(write_migration, read_migration, internal_migration)
        ext_write_migration = 0
        if self.cfg.B_ext_interface_R <= self.cfg.B_ext_internal:
            ext_write_migration = max(ext_read, ext_write_large)
        else:
            ext_write_migration = ext_read + ext_write_small
        
        T_ext = ext_write_migration
        
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

        total_steps = self.cfg.N * self.cfg.L
        
        # Create progress bar for terminal
        progress_bar = tqdm(
            total=total_steps,
            desc="Simulating",
            unit= "step",
            dynamic_ncols=True,       # auto-resize to terminal width
            file=sys.__stdout__,  # Use system stdout for terminal
            leave=False,
        )

        for n in range(self.cfg.N_pre, self.cfg.N_pre + self.cfg.N):
            for l in range(self.cfg.L):
                alpha = self.plc.alpha_strategy(n, l)
                beta = self.plc.beta_strategy(n, l)
                migration_data = self.mig.migration_strategy(n, l)
                
                # Calculate step time
                step_time = self.calculate_step_time(n, l, alpha, beta, 
                                                        migration_data[0], migration_data[1],
                                                        migration_data[2], migration_data[3])
                self.total_time += step_time
                
                # Record step details
                self.step_details.append({
                    'n': n,
                    'l': l,
                    'time': step_time,
                    'alpha': alpha,
                    'beta': beta
                })
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix(n=n, l=l, refresh=False)
        progress_bar.close()
        return self.total_time

# simulator.py (updated run_simulation function)
def run_simulation(init_class: MemStatus, config_params: dict, 
                  mig_classes: list, plc_classes: list):
    """Run simulation with specified initialization class and config parameters"""
    fn = config_params.get('filename', "trace.txt")
    inclusive = config_params.get('inclusive', False)
    best = config_params.get('best', False)
    # First read to get token information
    with TraceReader(fn) as trace_reader:
        N_pre_tk = trace_reader.first_token
        N_last_tk = trace_reader.last_token
        N_tk = N_last_tk - N_pre_tk + 1

        # Create base config
        config = ModelConfig(
            N=N_tk,
            N_pre=N_pre_tk,
            para_num=config_params.get('para_num', 0.5),
            C_HBM_max=config_params.get('C_HBM_max', 3)
        )

    # 🔥 Use passed strategy classes instead of hardcoded
    placement_classes = plc_classes
    migration_classes = mig_classes
    # best simulation
    # with TraceReader(fn) as trace_reader:
    #     # Rest of the original simulation logic...
    #     initial_state_temp = init_class(config, trace_reader, inclusive)
    
    #     best_mig = NoMigration(config, initial_state_temp)
    #     best_plc = PreferHBM(config, initial_state_temp)
    #     best_simulator = MemorySimulator(config, initial_state_temp, best_plc, best_mig, best=True)
    #     upper_bound_time = best_simulator.simulate()
        
    #     print(f"Read trace file: {fn}")
    #     print(f"Best Combination:")
    #     print(f"Total simulation time: {upper_bound_time:.4f} ns, {upper_bound_time/1e9:.4f} seconds")
    #     print(f"Average time per token: {upper_bound_time/initial_state_temp.cfg.N:.6f} ns")
    #     print("-" * 50)

    # 🔥 Use passed strategy classes in the loops
    for p_cls in placement_classes:
        for m_cls in migration_classes:
            with TraceReader(fn) as trace_reader:

                test_initial_state = init_class(config, trace_reader, inclusive)
                
                mig_instance = m_cls(config, test_initial_state)
                placement_instance = p_cls(config, test_initial_state)
                
                simulator = MemorySimulator(config, test_initial_state, 
                                        placement_instance, mig_instance, best)
                total_time = simulator.simulate()
                # avg_alpha = sum(step['alpha'] for step in simulator.step_details) / len(simulator.step_details)
                
                print(f"Combination: {p_cls.__name__} + {m_cls.__name__}")
                print(f"Total time: {total_time:.8f} ns, {total_time/1e9:.8f} seconds, coarse upper bound: {best}")
                #print(f"Avg alpha: {avg_alpha:.6f}")
                print(f"Alpha:")
                last_n = config.N_pre + config.N - 1
                last_alpha = next(step['alpha'] for step in simulator.step_details
                                if step['n'] == last_n and step['l'] == 31)
                print(f"n={last_n}, alpha = {last_alpha:.8f}")

                # test_initial_state.print_token_layer_status()
    
    return

def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# simulator.py (add this at the end)
if __name__ == "__main__":
    import argparse
    from argparse import Namespace
    # Redirect output to log file
    import sys

    csv.field_size_limit(sys.maxsize)
    # Mapping from string names to actual classes
    CLASS_MAPPING = {
        # Initialization classes
        'HBMInit': HBMInit,
        'TokenLevelBestRatioInit': TokenLevelBestRatioInit,
        'HBMInitPaged': HBMInitPaged,
        
        # Migration classes
        'NoMigration': NoMigration,
        'LookAheadOneMigration': LookAheadOneMigration,
        'NormalMigration': NormalMigration,
        'PageMigration': PageMigration,
        # Placement classes
        'PreferHBM': PreferHBM,
        'LookAheadOnePlacement': LookAheadOnePlacement,
        'PreferHBMPaged': PreferHBMPaged,
        
    }

    parser = argparse.ArgumentParser()
    # parser.add_argument('--N', type=int, default=1024*16)
    # parser.add_argument('--N_pre', type=int, default=1024)
    parser.add_argument('--para_num', type=float, default=0.5)
    parser.add_argument('--C_HBM_max', type=float, default=4)
    parser.add_argument('--inclusive', type=str2bool, default=False)
    parser.add_argument('--filename', type=str, default="04_1_16.txt")
    parser.add_argument('--init_class', type=str, required=True, 
                       help='Initialization class name')
    parser.add_argument('--mig_classes', type=str, nargs='+', required=True,
                       help='Migration class names separated by spaces')
    parser.add_argument('--plc_classes', type=str, nargs='+', required=True,
                       help='Placement class names separated by spaces')
    parser.add_argument('--log_file', type=str, default="simulation.txt")
    parser.add_argument('--best', type=str2bool, default=False)
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
        # 'N': args.N,
        # 'N_pre': args.N_pre,
        'para_num': args.para_num,
        'C_HBM_max': args.C_HBM_max,
        'filename': args.filename,
        'inclusive': args.inclusive,
        'best': args.best
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


