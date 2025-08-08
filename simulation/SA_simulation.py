from abc import ABC, abstractmethod
from simulator.simulation.memory_status import ModelConfig, MemStatus, ScaledLLaMa3_8BConfig
from collections import Counter
import multiprocessing as mp
from functools import partial

from simulator import ModelConfig, MemorySimulator, TraceReader
from simulator.simulation.memory_status import TokenLevelBestRatioInit
from placement import LookAheadOnePlacement
from simulator.simulation.migration import SAMigration
import random
import math
import numpy as np
import argparse
import sys

# Add str2bool function if not already present
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

def evaluate_latency_parallel(params: dict, window_size: int, topk_ratio: float, n_splits: int = 4) -> float:
    """Parallel version of latency evaluation"""
    try:
        fn = params.get('filename', "trace.txt")
        inclusive = params.get('inclusive', False)
        best = params.get('best', False)

        # First read to get token information
        with TraceReader(fn) as trace_reader:
            N_pre_tk = trace_reader.first_token
            N_last_tk = trace_reader.last_token
            N_tk = N_last_tk - N_pre_tk + 1

        # Divide the token range into n_splits parts
        first_tok = N_pre_tk                     
        last_tok  = N_last_tk                     

        # how many processes you want
        P = n_splits                              # e.g. 2

        # equal-size contiguous chunks
        chunk = math.ceil((last_tok - first_tok + 1) / P)

        slices = []
        for i in range(P):
            s = first_tok + i * chunk
            e = min(last_tok, s + chunk - 1)
            if s <= e:                            # skip empty tails
                slices.append((s, e))
        
        # Create a pool of workers
        with mp.Pool(processes=len(slices)) as pool:
            # Create partial function with fixed parameters
            eval_partial = partial(
                evaluate_split,
                fn=fn,
                params=params,
                inclusive=inclusive,
                best=best,
                window_size=window_size,
                topk_ratio=topk_ratio
            )
            
            # Run evaluations in parallel
            results = pool.map(eval_partial, slices)
            
        # Sum up all latencies
        total_latency = sum(results)
        return total_latency

    except Exception as e:
        print(f"Error in parallel evaluation: {str(e)}")
        return float('inf')
    
def evaluate_split(token_pairs, fn: str, params:dict, inclusive: bool, 
                  best: bool, window_size: int, topk_ratio: float) -> float:
    """Evaluate a specific range of tokens"""
    try:
        start_tk, end_tk = token_pairs
        
        with TraceReader(fn) as trace_reader:

            # Create base config
            slice_cfg = ModelConfig(
                N=end_tk - start_tk + 1,
                N_pre=start_tk,
                para_num=params.get('para_num', 0.5),
                C_HBM_max=params.get('C_HBM_max', 3)
            )

            # Initialize memory status for this split
            mem_status = TokenLevelBestRatioInit(slice_cfg, trace_reader, inclusive)
            
            # Create strategy instances
            placement_instance = LookAheadOnePlacement(slice_cfg, mem_status)
            migration_instance = SAMigration(slice_cfg, mem_status, window_size, topk_ratio)
            
            # Create simulator for this split
            simulator = MemorySimulator(
                config=slice_cfg,
                status=mem_status,
                placement=placement_instance,
                migration=migration_instance,
                best=best
            )
            
            # Run simulation for this split
            split_latency = simulator.simulate()
            return split_latency

    except Exception:
        import traceback, sys
        traceback.print_exc()      # show in worker stderr
        sys.stderr.flush()
        raise   

def evaluate_latency(params: dict, window_size, topk_ratio) -> float:

    try:
        fn = params.get('filename', "trace.txt")
        inclusive = params.get('inclusive', False)
        best = params.get('best', False)
        # First read to get token information
        with TraceReader(fn) as trace_reader:
            N_pre_tk = trace_reader.first_token
            N_last_tk = trace_reader.last_token
            N_tk = N_last_tk - N_pre_tk + 1

            # Create base config
            config = ModelConfig(
                N=N_tk,
                N_pre=N_pre_tk,
                para_num=params.get('para_num', 0.5),
                C_HBM_max=params.get('C_HBM_max', 3)
            )
            # Initialize memory status
            mem_status = TokenLevelBestRatioInit(config, trace_reader, inclusive)
            
            # Create strategy instances
            placement_instance = LookAheadOnePlacement(config, mem_status)
            migration_instance = SAMigration(config, mem_status, window_size, topk_ratio)
            
            # Create and run simulator
            simulator = MemorySimulator(
                config=config,
                status=mem_status,
                placement=placement_instance,
                migration=migration_instance,
                best=best,
            )
            
            total_latency = simulator.simulate()
            return total_latency

    except Exception as e:
        print(f"Error in evaluate_latency: {str(e)}")
        return float('inf')  # Return infinity for failed evaluations
    

def simulated_annealing_parallel(params: dict, log_file: str, max_iter: int = 100,
                               initial_temp: float = 100.0, cooling_rate: float = 0.95,
                               n_splits: int = 4):
    """Parallel version of simulated annealing"""
    # Initialize logging
    with open(log_file, 'w') as f:
        f.write("Starting Parallel Simulated Annealing optimization...\n")
        f.write(f"\nConfiguration:")
        f.write(f"\n  Parameters: {params['para_num']}B")
        f.write(f"\n  HBM Capacity: {params['C_HBM_max']}GB")
        f.write(f"\n  Inclusive: {params['inclusive']}")
        f.write(f"\n  Best Case: {params['best']}")
        f.write(f"\n  Number of parallel splits: {n_splits}\n")
        f.write("\nOptimizing parameters...\n")
        f.flush()
    
    # Initial state
    current_window = 5
    current_topk = 0.5
    current_latency = evaluate_latency_parallel(params, current_window, current_topk, n_splits)
    
    # Best state tracking
    best_window = current_window
    best_topk = current_topk
    best_latency = current_latency
    temp = initial_temp
    
    for i in range(max_iter):
        # Generate neighbor state
        new_window = max(1, current_window + random.randint(-3, 3))
        new_topk = max(0.1, min(1.0, current_topk + random.uniform(-0.3, 0.3)))
        
        # Evaluate new state in parallel
        new_latency = evaluate_latency_parallel(params, new_window, new_topk, n_splits)
        
        # Calculate delta and accept/reject
        delta = new_latency - current_latency
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_window = new_window
            current_topk = new_topk
            current_latency = new_latency
            
            if current_latency < best_latency:
                best_window = current_window
                best_topk = current_topk
                best_latency = current_latency
        
        # Cool down
        temp *= cooling_rate
        
        # Log progress
        with open(log_file, 'a') as f:
            f.write(f"\nIteration {i+1}/{max_iter}:\n")
            f.write(f"  Temperature: {temp:.2f}\n")
            f.write(f"  Current: window={current_window}, topk={current_topk:.2f}\n")
            f.write(f"  Current latency: {current_latency:.8f}\n")
            f.write(f"  Best latency: {best_latency:.8f}\n")
            f.write("="*50 + "\n")
            f.flush()
    
    return best_window, best_topk, best_latency

def simulated_annealing(params: dict, log_file: str, max_iter: int = 100, 
                        initial_temp: float = 100.0, cooling_rate: float = 0.95):
    """
    Simulated Annealing to find optimal parameters for minimal latency.
    
    Args:
        params: Base configuration parameters
        max_iter: Maximum iterations
        initial_temp: Initial temperature
        cooling_rate: Temperature cooling rate
    
    Returns:
        tuple[dict, float]: Best parameters and corresponding latency
    """
    # Current state
    with open(log_file, 'w') as f:  # First clear the file
        f.write("Starting Simulated Annealing optimization...\n")
        f.write("\nConfiguration:")
        f.write(f"\n  Parameters: {params['para_num']}B")
        f.write(f"\n  HBM Capacity: {params['C_HBM_max']}GB")
        f.write(f"\n  Inclusive: {params['inclusive']}")
        f.write(f"\n  Best Case: {params['best']}\n")
        f.write("\nOptimizing parameters...\n")
        f.flush()
    
    current_window = 5  # Start with window size 10
    current_topk = 0.5  # Start with 80% tokens
    
    # Evaluate initial state
    current_latency = evaluate_latency(params, current_window, current_topk)

    # Best state tracking
    best_window = current_window
    best_topk = current_topk
    # best_freq = current_freq
    best_latency = current_latency
    
    # Temperature
    temp = initial_temp
    
    for i in range(max_iter):
        # Generate neighbor state with random modifications
        new_window = max(1, current_window + random.randint(-3, 3))  # Window size ≥ 1 tried 8
        new_topk = max(0.1, min(1.0, current_topk + random.uniform(-0.3, 0.3)))  # TopK ratio [0.1, 1.0]
        # new_freq = not current_freq if random.random() < 0.3 else current_freq
        
        # Evaluate new state
        new_latency = evaluate_latency(params, new_window, new_topk)
        
        # Calculate delta
        delta = new_latency - current_latency
        
        # Accept or reject new state
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_window = new_window
            current_topk = new_topk
            # current_freq = new_freq
            current_latency = new_latency
            
            # Update best if current is better
            if current_latency < best_latency:
                best_window = current_window
                best_topk = current_topk
                # best_freq = current_freq
                best_latency = current_latency
        
        # Cool down
        temp *= cooling_rate
        
        # Log progress in real-time
        with open(log_file, 'a') as f:
            f.write(f"\nIteration {i+1}/{max_iter}:\n")
            f.write(f"  Temperature: {temp:.2f}\n")
            f.write(f"  Current: window={current_window}, topk={current_topk:.2f}\n")
            f.write(f"  Current latency: {current_latency:.8f}\n")
            f.write(f"  Best latency: {best_latency:.8f}\n")
            f.write("="*50 + "\n")
            f.flush()  # Ensure writing to disk immediately
    
    # Write final results
    with open(log_file, 'a') as f:
        f.write("\nOptimization completed!\n")
        f.write("\nBest parameters found:\n")
        f.write(f"  Window Size: {best_window}\n")
        f.write(f"  Top-k Ratio: {best_topk:.3f}\n")
        f.write(f"\nBest latency: {best_latency:.8f} ns ({best_latency/1e9:.8f} seconds)\n")
        f.flush()

def argparse_params():
    parser = argparse.ArgumentParser(description="Evaluate latency for memory simulation.")
    parser.add_argument('--para_num', type=float, default=0.5, help='Number of parameters in billions.')
    parser.add_argument('--C_HBM_max', type=float, default=3, help='Maximum HBM capacity in GB.')
    parser.add_argument('--filename', type=str, default='trace.txt', help='Trace file name.')
    parser.add_argument('--best', type=str2bool, default=False, help='Use best case scenario.')
    parser.add_argument('--inclusive', type=str2bool, help='Use inclusive memory model.')
    parser.add_argument('--log_file', type=str, default="sa_simulation.txt",
                      help='Output log filename')
    return parser.parse_args()

def main():

    args = argparse_params()
    test_params = {
        'para_num': args.para_num,  # 0.5B parameters
        'C_HBM_max': args.C_HBM_max,   # 3GB HBM
        'filename': args.filename,
        'inclusive': args.inclusive,
        'best': args.best
    }

    # Get number of CPU cores (leave one core free for system)
    n_splits = max(1, mp.cpu_count() - 3)

    try:
        best_window, best_topk, best_latency = simulated_annealing_parallel(
            test_params,
            args.log_file,
            max_iter=40,
            initial_temp=100.0,
            cooling_rate=0.95,
            n_splits=n_splits
        )
        
        with open(args.log_file, 'a') as f:
            f.write("\nOptimization completed!\n")
            f.write("\nBest parameters found:\n")
            f.write(f"  Window Size: {best_window}\n")
            f.write(f"  Top-k Ratio: {best_topk:.3f}\n")
            f.write(f"\nBest latency: {best_latency:.8f} ns ({best_latency/1e9:.8f} seconds)\n")
    except Exception as e:
        with open(args.log_file, 'a') as f:
            f.write(f"\nOptimization failed: {str(e)}\n")

    # try:
    #     simulated_annealing(
    #         test_params,
    #         args.log_file,
    #         max_iter=40,
    #         initial_temp=100.0,
    #         cooling_rate=0.95
    #     )
    # except Exception as e:
    #     with open(args.log_file, 'a') as f:
    #         f.write(f"\nOptimization failed: {str(e)}\n")
    #         f.flush()


if __name__ == "__main__":
    main()