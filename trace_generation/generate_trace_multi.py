import random
import math
import csv
import multiprocessing
import os
import shutil

# Configuration parameters
N = 41538 - 31299      # Number of tokens to generate in decode stage
N_pre = 31299     # Starting token index for decode stage
L = 32            # Total layers (matches ModelConfig.L)
sparsity = 0.6    # Target sparsity ratio (now represents skip ratio) 0.4 0.6
read_ratio = 1 - sparsity  # Ratio of tokens to read
diff_ratio = 0.1  # Maximum difference ratio between consecutive read sets
threshold_factor = 0.95  # Additional factor for threshold

def generate_initial_read_tokens(n, read_ratio, threshold_factor):
    """Generate initial read tokens for token n, biased towards recent tokens."""
    if n <= 1:
        return []
    threshold = min(1.0, read_ratio + threshold_factor * (1 - read_ratio))
    min_read_index = math.floor((1 - threshold) * (n - 1))  # Lower limit for read tokens
    k = int(round(read_ratio * (n - 1)))  # Desired number of read tokens
    if k <= 0:
        return []
    # Select k tokens from min_read_index to n-1
    available = list(range(min_read_index, n))
    if len(available) < k:
        return sorted(available)
    return sorted(random.sample(available, k))

def generate_similar_read_tokens(R_prev, n, read_ratio, diff_ratio, threshold_factor):
    """Generate read tokens for token n based on R_prev, favoring recent tokens."""
    if n <= 1:
        return []
    k_n = int(round(read_ratio * (n - 1)))  # Desired size of R_n
    if k_n <= 0:
        return []
    R_n = list(R_prev)  # Start with the previous read set
    threshold = min(1.0, read_ratio + threshold_factor * (1 - read_ratio))
    min_read_index = math.floor((1 - threshold) * (n - 1))  # Range for read tokens
    
    # Available tokens to add: min_read_index to n-1 excluding current R_n
    available_to_add = list(set(range(min_read_index, n)) - set(R_n))
    
    # Adjust size based on delta
    delta = k_n - len(R_prev)
    if delta > 0:
        add_count = min(delta, len(available_to_add))
        if add_count > 0:
            to_add = random.sample(available_to_add, add_count)
            R_n.extend(to_add)
    elif delta < 0:
        remove_count = min(-delta, len(R_n))
        if remove_count > 0:
            R_n_sorted = sorted(R_n)  # Sort ascending to prioritize removing older tokens
            to_remove = R_n_sorted[:remove_count]
            R_n = [t for t in R_n if t not in to_remove]
    
    # Introduce controlled variation by swapping tokens
    swap_count = random.randint(0, int(diff_ratio * len(R_n)))
    available_to_remove = sorted(R_n)  # Prefer older tokens
    available_to_add = list(set(range(min_read_index, n)) - set(R_n))
    swap_count = min(swap_count, len(available_to_remove), len(available_to_add))
    if swap_count > 0:
        to_remove = available_to_remove[:swap_count]
        to_add = random.sample(available_to_add, swap_count)
        R_n = [t for t in R_n if t not in to_remove]
        R_n.extend(to_add)
    
    return sorted(R_n)

def generate_for_layer(l, N_pre, N, read_ratio, diff_ratio, threshold_factor, temp_folder):
    layer_file = f"{temp_folder}/layer_{l}.csv"
    with open(layer_file, "w", newline='') as f:
        writer = csv.writer(f)
        R_prev = generate_initial_read_tokens(N_pre, read_ratio, threshold_factor)
        read_str = f"[{','.join(map(str, R_prev))}]" if R_prev else "[]"
        writer.writerow([N_pre, l, read_str])
        for n in range(N_pre + 1, N_pre + N):
            R_n = generate_similar_read_tokens(
                R_prev, n, read_ratio, diff_ratio, threshold_factor
            )
            read_str = f"[{','.join(map(str, R_n))}]" if R_n else "[]"
            writer.writerow([n, l, read_str])
            R_prev = R_n

output_filename = f"trace_{sparsity:.2f}.csv"
temp_folder = "temp_traces"

if __name__ == "__main__":
    os.makedirs(temp_folder, exist_ok=True)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(generate_for_layer, [(l, N_pre, N, read_ratio, diff_ratio, threshold_factor, temp_folder) for l in range(L)])
    
    # Merge incrementally
    with open(output_filename, "w", newline='') as final_f:
        writer = csv.writer(final_f)
        writer.writerow(['n', 'l', 'read_kv'])
        files = [open(f"{temp_folder}/layer_{l}.csv", "r") for l in range(L)]
        readers = [csv.reader(f) for f in files]
        for _ in range(N):
            for l in range(L):
                row = next(readers[l])
                writer.writerow(row)
        for f in files:
            f.close()  # Close the file objects directly

    # Clean up
    shutil.rmtree(temp_folder)

    print(f"Trace file '{output_filename}' generated successfully.")