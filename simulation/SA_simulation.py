import multiprocessing as mp
from functools import partial
import random
import math
import argparse
import sys
from datetime import datetime

# Import classes from companion scripts
from simulator import TraceReader, MemorySimulator
from memory_status import Mixtral8x7BConfig, LookAheadInit, MemStatus # Ensure MemStatus classes are imported
from placement import LookAheadOnePlacement
from migration import OnlineAdaptiveMigration
from placement import BaseStrategy
from migration import BaseDataMigration

# Increase CSV limit for large trace files
import csv
csv.field_size_limit(sys.maxsize)


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

def evaluate_split(token_pairs, fn: str, params: dict, inclusive: bool,
                  best: bool, window_size: int, split_id: int) -> tuple[float, float, float]:
    """
    Evaluates a specific contiguous range of tokens using the target strategy.
    Returns: (split_latency, avg_hit_rate, avg_model_weight_ratio)
    """
    try:
        start_tk, end_tk = token_pairs

        # split_log_filename = f"sa_split_{split_id}_w{window_size}_t{start_tk}.csv" # NEW
        split_log_filename = None
        
        # We must create a new TraceReader for each process/split
        with TraceReader(fn) as trace_reader:

            # Get necessary batch size from trace
            batch_size = trace_reader.batch_size
            
            # 1. Create split-specific config (N and N_pre are local to the slice)
            slice_cfg = Mixtral8x7BConfig(
                N=end_tk - start_tk + 1,
                N_pre=start_tk,
                para_num=params.get('para_num', 46.7),
                C_HBM_max=params.get('C_HBM_max', 100),
                B_ext_R=params.get('B_ext_R', 450),
                B_ext_W=params.get('B_ext_W', 450),
                batch_size=batch_size
            )

            # 2. Instantiate strategy classes (fixed combination)
            # Initialization class must be LookAheadInit to set up the initial state
            mem_status: MemStatus = LookAheadInit(slice_cfg, trace_reader, inclusive)
            
            # Placement: LookAheadOnePlacement (beta and alpha)
            placement_instance: BaseStrategy = LookAheadOnePlacement(slice_cfg, mem_status)
            
            # Migration: OnlineAdaptiveMigration (using the current window_size being tuned)
            migration_instance: BaseDataMigration = OnlineAdaptiveMigration(
                slice_cfg, mem_status, window_size=window_size
            )
            
            # 3. Create and run simulator
            simulator = MemorySimulator(
                config=slice_cfg,
                status=mem_status,
                placement=placement_instance,
                migration=migration_instance,
                best=best,
                log_filename=split_log_filename
            )
            
            split_latency, avg_hit_rate, avg_model_weight_ratio = simulator.simulate()
            
            # Return raw latency, average alpha, and average model weight ratio
            return split_latency, avg_hit_rate, avg_model_weight_ratio

    except Exception as e:
        print(f"Error in split {start_tk}-{end_tk}: {str(e)}", file=sys.stderr)
        # Return infinite latency for failed splits
        return float('inf'), 0.0, 0.0
    
def _sa_split_worker(
    split_pair: tuple[tuple[int, int], int], # (token_pairs, split_id)
    fn: str, 
    params: dict, 
    inclusive: bool, 
    best: bool, 
    window_size: int
) -> tuple[float, float, float]:
    """
    Wrapper to align all arguments passed from pool.starmap with the core logic.
    """
    # Unpack the varying arguments
    token_pairs, split_id = split_pair
    
    # Call the original evaluation function
    return evaluate_split(
        token_pairs=token_pairs, 
        fn=fn, 
        params=params, 
        inclusive=inclusive, 
        best=best, 
        window_size=window_size, 
        split_id=split_id # Now correctly passed
    )

def evaluate_latency_parallel(params: dict, window_size: int, n_splits: int) -> tuple[float, float, float]:
    """
    Runs the full simulation in parallel splits and aggregates the results.
    """
    fn = params.get('filename', "trace.txt")
    inclusive = params.get('inclusive', False)
    best = params.get('best', False)
    
    # 1. First read to get token information for slicing
    with TraceReader(fn) as trace_reader:
        N_pre_tk = trace_reader.first_token
        N_last_tk = trace_reader.last_token
        
    if N_pre_tk is None or N_last_tk is None:
        raise ValueError("Trace file does not contain valid token indices.")

    # 2. Divide the token range into n_splits parts
    first_tok = N_pre_tk                     
    last_tok  = N_last_tk                     
    chunk = math.ceil((last_tok - first_tok + 1) / n_splits)

    slices = []
    for i in range(n_splits):
        s = first_tok + i * chunk
        e = min(last_tok, s + chunk - 1)
        if s <= e:
            slices.append((s, e))
    
    if not slices:
        return 0.0, 0.0, 0.0
        
    # 3. Create a pool of workers and map the evaluation function
    with mp.Pool(processes=len(slices)) as pool:
        # split_args = [(slices[i], i) for i in range(len(slices))]

        # eval_partial = partial(
        #     evaluate_split,
        #     fn=fn,
        #     params=params,
        #     inclusive=inclusive,
        #     best=best,
        #     window_size=window_size
        # )
        
        # # results is a list of (latency, alpha, model_ratio) tuples
        # results = pool.map(eval_partial, split_args)

        # Create a list of the *varying* arguments: [((start, end), id), ...]
        split_pair_args = [((s, e), i) for i, (s, e) in enumerate(slices)]
        # Fixed arguments for the worker function
        fixed_args = (fn, params, inclusive, best, window_size)
        
        # The pool.starmap iterable needs to be structured:
        # [(split_args_0 + fixed_args), (split_args_1 + fixed_args), ...]
        
        # Prepare the final argument list for starmap
        starmap_args = [
            (split_pair_arg,) + fixed_args 
            for split_pair_arg in split_pair_args
        ]
        # results is a list of (latency, alpha, model_ratio) tuples
        results = pool.starmap(_sa_split_worker, starmap_args)
        
    # 4. Sum up all latencies and average hit rates
    valid_results = [res for res in results if res[0] != float('inf')]
    
    if not valid_results:
        print("All splits failed.")
        return float('inf'), 0.0, 0.0

    total_latency = sum(latency for latency, _, _ in valid_results)
    
    # Average the metrics (alpha, model_weight) over the number of valid splits
    avg_hit_rate = sum(hit_rate for _, hit_rate, _ in valid_results) / len(valid_results)
    avg_model_weight_ratio = sum(model_weight for _, _, model_weight in valid_results) / len(valid_results)
    
    return total_latency, avg_hit_rate, avg_model_weight_ratio


def simulated_annealing_parallel(params: dict, log_file: str, max_iter: int = 100,
                               initial_temp: float = 100.0, cooling_rate: float = 0.95,
                               n_splits: int = 4, initial_window: int = 50):
    """
    Parallel Simulated Annealing focusing only on window_size for the
    LookAheadInit + OnlineAdaptiveMigration + LookAheadOnePlacement strategy.
    """
    
    # Initialize logging
    with open(log_file, 'w') as f:
        f.write(f"Starting Parallel Simulated Annealing optimization at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...\n")
        f.write(f"\nTarget Strategy: LookAheadInit + OnlineAdaptiveMigration + LookAheadOnePlacement")
        f.write(f"\nConfiguration: {params['para_num']}B | HBM: {params['C_HBM_max']}GB | Splits: {n_splits}")
        f.write(f"\nTrace File: {params['filename']}")
        f.write(f"\nOptimization Parameters: Max Iter={max_iter}, Initial Temp={initial_temp}, Cooling Rate={cooling_rate}\n")
        f.write("\nOptimizing window_size...\n")
        f.flush()
    
    # Initial state
    current_window = initial_window
    current_latency, current_hit_rate, current_model_weight = evaluate_latency_parallel(params, current_window, n_splits)
    
    # Best state tracking
    best_window = current_window
    best_latency = current_latency
    best_hit_rate = current_hit_rate
    best_model_weight = current_model_weight
    temp = initial_temp
    
    # Search space boundaries
    MIN_W = 5
    MAX_W = 500
    
    for i in range(max_iter):
        
        if temp < 1e-4:
            print("Temperature effectively zero. Stopping.")
            break
            
        # Generate neighbor state: step size decreases as temp decreases
        step_size = max(1, int(10 * (temp / initial_temp)))
        
        new_window = current_window + random.randint(-step_size, step_size)
        new_window = max(MIN_W, min(MAX_W, new_window)) # Clamp to bounds
        
        # Ensure we don't re-evaluate the same state unnecessarily (optional optimization)
        if new_window == current_window:
            continue
            
        # Evaluate new state in parallel
        new_latency, new_hit_rate, new_model_weight = evaluate_latency_parallel(params, new_window, n_splits)
        
        # Calculate delta and accept/reject
        delta = new_latency - current_latency
        
        # Accept if better (delta < 0) or by probability (if worse)
        if delta < 0 or (new_latency != float('inf') and random.random() < math.exp(-delta / temp)):
            current_window = new_window
            current_latency = new_latency
            current_hit_rate = new_hit_rate
            current_model_weight = new_model_weight
            
            if current_latency < best_latency:
                best_window = current_window
                best_latency = current_latency
                best_hit_rate = current_hit_rate
                best_model_weight = current_model_weight
        
        # Cool down
        temp *= cooling_rate
        
        # Log progress
        with open(log_file, 'a') as f:
            f.write(f"\nIteration {i+1}/{max_iter}:\n")
            f.write(f"  Temperature: {temp:.2f}\n")
            f.write(f"  Current: window={current_window}\n")
            f.write(f"  Current latency: {ns_to_s(current_latency):.8f} s\n") 
            f.write(f"  Best latency: {ns_to_s(best_latency):.8f} s (Window {best_window})\n")
            f.write("="*50 + "\n")
            f.flush()
    
    return best_window, best_latency, best_hit_rate, best_model_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Simulated Annealing optimization for adaptive migration window size.")
    parser.add_argument('--para_num', type=float, default=46.7, help='Number of parameters in billions.')
    parser.add_argument('--C_HBM_max', type=float, default=100.0, help='Maximum HBM capacity in GB.')
    parser.add_argument('--B_ext_R', type=float, default=450, help='External read bandwidth in GB/s.')
    parser.add_argument('--B_ext_W', type=float, default=450, help='External write bandwidth in GB/s.')
    parser.add_argument('--filename', type=str, required=True, help='Path to the MoE trace file.')
    parser.add_argument('--best', type=str2bool, default=False, help='Use best case scenario (alpha constraint override).')
    parser.add_argument('--inclusive', type=str2bool, default=False, help='Use inclusive memory model.')
    parser.add_argument('--log_file', type=str, default="sa_tuning_log.txt", help='Output log filename.')
    parser.add_argument('--n_splits', type=int, default=4, help='Number of parallel processes to split the trace into.')
    parser.add_argument('--max_iter', type=int, default=40, help='Maximum SA iterations.')
    parser.add_argument('--initial_window', type=int, default=50, help='Initial window size for SA search.')
    
    args = parser.parse_args()
    
    # Convert latency to seconds for logging (as simulator returns ns)
    def ns_to_s(ns):
        return ns / 1e9

    test_params = {
        'para_num': args.para_num,
        'C_HBM_max': args.C_HBM_max,
        'B_ext_R': args.B_ext_R,
        'B_ext_W': args.B_ext_W,
        'filename': args.filename,
        'inclusive': args.inclusive,
        'best': args.best
    }

    try:
        best_window, best_latency_ns, best_hit_rate, best_model_weight = simulated_annealing_parallel(
            test_params,
            args.log_file,
            max_iter=args.max_iter,
            initial_temp=100.0,
            cooling_rate=0.95,
            n_splits=args.n_splits,
            initial_window=args.initial_window
        )
        
        best_latency_s = ns_to_s(best_latency_ns)
        
        with open(args.log_file, 'a') as f:
            f.write("\nOptimization completed!\n")
            f.write("\nBest parameters found:\n")
            f.write(f"  Window Size (W): {best_window}\n")
            f.write(f"\nBest latency: {best_latency_ns:.1f} ns ({best_latency_s:.8f} seconds)\n")
            f.write(f"Best HBM hit rate (Alpha): {best_hit_rate:.4f}\n")
            f.write(f"Best model weight ratio: {best_model_weight:.4f}\n")
    except Exception as e:
        with open(args.log_file, 'a') as f:
            f.write(f"\nOptimization failed: {str(e)}\n")