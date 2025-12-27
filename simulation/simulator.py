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

class TraceReader:
    """
    Reads and indexes the 5-column MoE trace file.
    Format: query_num_in_batch,absolute_token_index,layer_num,used_expert_indices,token_indices_by_attention
    """
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self._header = None
        self.first_token = None
        self.last_token = None
        self.batch_size = 0
        self.line_positions = {}
        self.token_cache = {}
        self.cache_window = 128 
        self.current_window_start = None
        
        print("Building trace index...")
        self._build_index()
        print(f"Trace index built. Batch Size: {self.batch_size}, Tokens: {self.first_token} to {self.last_token}")
    
    def _update_cache(self, n):
        self.token_cache.clear()
        self.current_window_start = n
        
        for token_idx in range(n, min(n + self.cache_window, self.last_token + 1)):
            self.token_cache[token_idx] = {}

        if n not in self.line_positions:
            return

        self.file.seek(self.line_positions[n])

        while True:
            line_pos = self.file.tell()
            line = self.file.readline()
            if not line: break 

            try:
                parts = next(csv.reader(io.StringIO(line)))
                if not parts: continue
                
                query_num = int(parts[0])
                curr_n = int(parts[1])
                curr_l = int(parts[2])
                
                if curr_n < n: continue 
                if curr_n >= n + self.cache_window:
                    self.file.seek(line_pos)
                    break
                
                experts = set(ast.literal_eval(parts[3]))
                kv_read = set(ast.literal_eval(parts[4]))
                
                if curr_l not in self.token_cache[curr_n]:
                    self.token_cache[curr_n][curr_l] = {
                        'experts': set(),
                        'kv_read_per_query': {} 
                    }
                
                self.token_cache[curr_n][curr_l]['experts'].update(experts)
                self.token_cache[curr_n][curr_l]['kv_read_per_query'][query_num] = kv_read
                
            except Exception as e:
                continue
    
    def _is_in_cache(self, n, l):
        return (self.current_window_start is not None and 
                self.current_window_start <= n < self.current_window_start + self.cache_window and
                n in self.token_cache and 
                l in self.token_cache[n])
    
    def _build_index(self):
        max_query_num = 0
        with open(self.filename, 'r') as f:
            self._header = f.readline().strip()
            pos = f.tell()
            line = f.readline()
            if not line: return
            
            try:
                parts = next(csv.reader(io.StringIO(line)))
                n = int(parts[1])
                self.first_token = n
                self.line_positions[n] = pos
                current_n = n
                max_query_num = max(max_query_num, int(parts[0]))
            except Exception:
                return

            while True:
                pos = f.tell()
                line = f.readline()
                if not line: break
                
                try:
                    parts = next(csv.reader(io.StringIO(line)))
                    if not parts: continue
                    n = int(parts[1])
                    max_query_num = max(max_query_num, int(parts[0]))
                    if n != current_n:
                        self.line_positions[n] = pos
                        current_n = n
                except Exception:
                    continue 
            
            self.last_token = current_n
            self.batch_size = max_query_num + 1 
    
    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.token_cache.clear()
        if self.file:
            self.file.close()
    
    def _get_step_data(self, n, l):
        if not self._is_in_cache(n, l):
            self._update_cache(n)
        
        if self._is_in_cache(n, l):
            return self.token_cache[n][l]
        return {'experts': set(), 'kv_read_per_query': {}}

    def get_active_experts(self, n, l) -> set:
        return self._get_step_data(n, l)['experts']
        
    def get_all_kv_reads_by_query(self, n, l) -> dict:
        return self._get_step_data(n, l)['kv_read_per_query']

class MemorySimulator(ABC):
    def __init__(self, config: ModelConfig, status: MemStatus,
                 placement: BaseStrategy, migration: BaseDataMigration, 
                 best: bool = False, log_filename: str = None):
        self.cfg = config
        self.plc = placement
        self.mig = migration
        self.status = status
        self.best = best
        self.total_time = 0.0
        self.step_details = []
        self.total_alpha = 0.0
        self.total_model_weight_ratio = 0.0
        self.step_count = 0
        self.model_weight_ratio = 0.0 
        self.log_filename = log_filename

    def calculate_step_time(self, n: int, l: int, 
                       alpha, beta: float, 
                       hbm_MR: float, hbm_MW: float,
                       ext_MR: float, ext_MW: float):
        D_R, D_W = self.status.calculate_data_sizes(n, l)
        
        if D_R > 0:
            kv_reads_all_queries = self.status.trace.get_all_kv_reads_by_query(n, l)
            total_kv_read_count = sum(len(tokens) for tokens in kv_reads_all_queries.values())
            total_kv_read = total_kv_read_count * self.status.single_KV_cache_size
            actual_model_weight = D_R - total_kv_read
            self.model_weight_ratio = actual_model_weight / D_R
        else:
            self.model_weight_ratio = 0.0

        if self.best:
            alpha = min(self.cfg.C_HBM_max / D_R, self.cfg.best_alpha) if D_R > 0 else 0
            beta = 0.0 
            hbm_MR = 0.0
            hbm_MW = 0.0
            ext_MR = 0.0
            ext_MW = 0.0
            D_W = 0.0

        HBM_read = alpha * D_R
        HBM_write = beta * D_W
        HBM_migration = hbm_MR + hbm_MW
        T_HBM = (HBM_read + HBM_write + HBM_migration) / self.cfg.B_HBM if self.cfg.B_HBM > 0 else float('inf')
        
        ext_read_bw = min(self.cfg.B_ext_interface_R, self.cfg.B_ext_internal)
        ext_read = ((1 - alpha) * D_R + ext_MR) / ext_read_bw if ext_read_bw > 0 else float('inf')
        
        write_migration = (ext_MW + D_W) / self.cfg.B_ext_interface_W if self.cfg.B_ext_interface_W > 0 else 0
        
        internal_bw_contention = (ext_MW + ext_MR + D_W + (1 - alpha) * D_R) / self.cfg.B_ext_internal if self.cfg.B_ext_internal > 0 else 0
        
        T_ext = max(ext_read, write_migration, internal_bw_contention)
        
        return max(T_HBM, T_ext)
    
    def simulate(self):
        self.total_time = 0.0
        self.step_details = []
        self.total_alpha = 0.0
        self.step_count = 0
        self.total_model_weight_ratio = 0.0

        if self.log_filename:
            log_handle = open(self.log_filename, 'a')
            log_handle.write(f"\n--- Simulation Start (N_pre={self.cfg.N_pre}) ---\n")
            log_handle.write("n,l,Time_ns,Alpha,Beta,DR_GB,DW_GB,HBM_Rem_GB,hbm_MW_GB,ext_MR_GB\n")
        else:
            log_handle = None

        total_steps = self.cfg.N * self.cfg.L
        
        progress_bar = tqdm(
            total=total_steps,
            desc="Simulating",
            unit= "step",
            dynamic_ncols=True,
            file=sys.__stdout__,
            leave=False,
        )

        for n in range(self.cfg.N_pre, self.cfg.N_pre + self.cfg.N):
            for l in range(self.cfg.L):
                alpha = self.plc.alpha_strategy(n, l)
                beta = self.plc.beta_strategy(n, l)
                migration_data = self.mig.migration_strategy(n, l)
                
                hbm_MR, hbm_MW, ext_MR, ext_MW = migration_data

                self.total_alpha += alpha
                self.total_model_weight_ratio += self.model_weight_ratio
                self.step_count += 1
                
                step_time = self.calculate_step_time(n, l, alpha, beta, 
                                                        hbm_MR, hbm_MW,
                                                        ext_MR, ext_MW)
                self.total_time += step_time
                
                if log_handle:
                    D_R, D_W = self.status.calculate_data_sizes(n, l)
                    line = f"{n},{l},{step_time:.2f},{alpha:.4f},{beta:.4f},"
                    line += f"{D_R/BYTES_TO_GB:.6f},{D_W/BYTES_TO_GB:.6f},"
                    line += f"{self.status.hbm_capacity_remaining/BYTES_TO_GB:.6f},"
                    line += f"{hbm_MW/BYTES_TO_GB:.6f},{ext_MR/BYTES_TO_GB:.6f}\n"
                    log_handle.write(line)
                
                self.step_details.append({
                    'n': n,
                    'l': l,
                    'time': step_time,
                    'alpha': alpha,
                    'beta': beta
                })
                progress_bar.update(1)
                progress_bar.set_postfix(n=n, l=l, refresh=False)
        
        progress_bar.close()
        if log_handle: log_handle.close()
        
        avg_alpha = self.total_alpha / self.step_count if self.step_count > 0 else 0
        avg_model_ratio = self.total_model_weight_ratio / self.step_count if self.step_count > 0 else 0
        
        return self.total_time, avg_alpha, avg_model_ratio

def run_simulation(init_class: MemStatus, config_params: dict, 
                  mig_classes: list, plc_classes: list, log_filename: str = None):
    fn = config_params.get('filename', "trace.txt")
    inclusive = config_params.get('inclusive', False)
    best = config_params.get('best', False)

    print(f"Reading trace file: {fn}")
    with TraceReader(fn) as trace_reader:
        N_pre_tk = trace_reader.first_token
        N_last_tk = trace_reader.last_token
        
        if N_pre_tk is None or N_last_tk is None:
            print(f"Error: No data in trace file {fn}")
            return
            
        N_tk = N_last_tk - N_pre_tk + 1
        batch_size = trace_reader.batch_size

    config = Mixtral8x7BConfig(
        N=N_tk,         
        N_pre=N_pre_tk, 
        para_num=config_params.get('para_num', 46.7), 
        C_HBM_max=config_params.get('C_HBM_max', 100), 
        B_ext_R=config_params.get('B_ext_R', 450),
        B_ext_W=config_params.get('B_ext_W', 450),
        batch_size=batch_size
    )

    placement_classes = plc_classes
    migration_classes = mig_classes

    for p_cls in placement_classes:
        for m_cls in migration_classes:
            with TraceReader(fn) as trace_reader:
                test_initial_state = init_class(config, trace_reader, inclusive)
                
                mig_instance = m_cls(config, test_initial_state)
                placement_instance = p_cls(config, test_initial_state)
                
                simulator = MemorySimulator(config, test_initial_state, 
                                        placement_instance, mig_instance, best, log_filename)
                
                total_time, avg_hit_rate, avg_model_weight_ratio = simulator.simulate()
                
                print(f"\nCombination: {p_cls.__name__} + {m_cls.__name__}")
                print(f"Internal bandwidth for read {config.B_ext_interface_R} GB/s, and write {config.B_ext_interface_W} GB/s")
                print(f"Total time: {total_time:.8f} ns, {total_time/1e9:.8f} seconds")
                
                if simulator.step_details:
                    last_n = config.N_pre + config.N - 1
                    last_l = config.L - 1
                    last_alpha = next((step['alpha'] for step in reversed(simulator.step_details)
                                     if step['n'] == last_n and step['l'] == last_l), 0.0)
                    print(f"Alpha at n={last_n}, l={last_l}: {last_alpha:.8f}")
                
                print(f"Average HBM hit rate (Alpha): {avg_hit_rate:.4f}")
                print(f"Average model weight ratio in total read: {avg_model_weight_ratio:.4f}")
                print("-" * 30)
    
    return

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    import argparse
    from argparse import Namespace
    import sys

    csv.field_size_limit(sys.maxsize)
    
    defined_classes = locals()
    CLASS_MAPPING = {
        'HBMInit': defined_classes.get('HBMInit'),
        'HBMInitPaged': defined_classes.get('HBMInitPaged'),
        'LookAheadInit': defined_classes.get('LookAheadInit'), 
        'NoMigration': defined_classes.get('NoMigration'),
        'JITMigration': defined_classes.get('JITMigration'),
        'PageMigration': defined_classes.get('PageMigration'),
        'OnlineAdaptiveMigration': defined_classes.get('OnlineAdaptiveMigration'),
        'PreferHBM': defined_classes.get('PreferHBM'),
        'PreferHBMPaged': defined_classes.get('PreferHBMPaged'),
        'LookAheadOnePlacement': defined_classes.get('LookAheadOnePlacement'),
    }
    CLASS_MAPPING = {k: v for k, v in CLASS_MAPPING.items() if v is not None}
    
    if not CLASS_MAPPING or 'HBMInit' not in CLASS_MAPPING:
        print("Error: No simulation classes found. Check imports from companion files.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--para_num', type=float, default=46.7)
    parser.add_argument('--C_HBM_max', type=float, default=100)
    parser.add_argument('--inclusive', type=str2bool, default=False)
    parser.add_argument('--filename', type=str, default="mixtral_analysis.csv")
    parser.add_argument('--init_class', type=str, required=True)
    parser.add_argument('--mig_classes', type=str, nargs='+', required=True)
    parser.add_argument('--plc_classes', type=str, nargs='+', required=True)
    parser.add_argument('--B_ext_R', type=float, default=450)
    parser.add_argument('--B_ext_W', type=float, default=450)
    parser.add_argument('--log_file', type=str, default="simulation.txt")
    parser.add_argument('--best', type=str2bool, default=False)
    args = parser.parse_args()

    try:
        init_class = CLASS_MAPPING[args.init_class]
        mig_classes = [CLASS_MAPPING[name] for name in args.mig_classes]
        plc_classes = [CLASS_MAPPING[name] for name in args.plc_classes]
    except KeyError as e:
        print(f"Error: Unknown class name {e.args[0]}")
        sys.exit(1)

    config_params = {
        'para_num': args.para_num,
        'C_HBM_max': args.C_HBM_max,
        'filename': args.filename,
        'inclusive': args.inclusive,
        'best': args.best,
        'B_ext_R': args.B_ext_R,
        'B_ext_W': args.B_ext_W
    }
    
    # --- DISABLE CSV GENERATION ---
    step_log_filename = None

    with open(args.log_file, 'w') as f:
        class Tee(object):
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, f)
        
        try:
            print("--- Starting MoE Simulation ---")
            run_simulation(
                init_class=init_class,
                config_params=config_params,
                mig_classes=mig_classes,
                plc_classes=plc_classes,
                log_filename=step_log_filename 
            )
            print("--- Simulation Finished ---")
        except Exception as e:
            import traceback
            print(f"Simulation failed: {str(e)}")
            traceback.print_exc(file=sys.stdout)
        
        sys.stdout = original_stdout