import random
import math

# Configuration parameters
N = 1024 * 16      # Number of tokens to generate in decode stage
N_pre = 1024      # Starting token index for decode stage
L = 32            # Total layers (matches ModelConfig.L)
sparsity = 0.2    # Target sparsity ratio
diff_ratio = 0.03 # Maximum difference ratio between consecutive skip sets
threshold_factor = 0.1  # Additional factor for threshold (e.g., threshold = sparsity + threshold_factor)

def generate_initial_skipped_tokens(n, sparsity, threshold_factor):
    """Generate initial skipped tokens for token n, biased towards older tokens."""
    if n <= 1:
        return []
    threshold = min(1.0, sparsity + threshold_factor)  # Cap threshold at 1.0
    max_skip_index = math.floor(threshold * (n - 1))  # Upper limit for skipped tokens
    k = int(round(sparsity * (n - 1)))  # Desired number of skipped tokens
    if k <= 0:
        return []
    # Select k tokens from 0 to max_skip_index
    available = list(range(max_skip_index + 1))
    if len(available) < k:
        return sorted(available)  # Return all available if k exceeds range
    return sorted(random.sample(available, k))

def generate_similar_skipped_tokens(S_prev, n, sparsity, diff_ratio, threshold_factor):
    """Generate skipped tokens for token n based on S_prev, favoring older tokens."""
    if n <= 1:
        return []
    k_n = int(round(sparsity * (n - 1)))  # Desired size of S_n
    if k_n <= 0:
        return []
    S_n = list(S_prev)  # Start with the previous skip set
    threshold = min(1.0, sparsity + threshold_factor)
    max_skip_index = math.floor(threshold * (n - 1))  # Range for skipped tokens
    
    # Available tokens to add: 0 to max_skip_index excluding current S_n
    available_to_add = list(set(range(max_skip_index + 1)) - set(S_n))
    
    # Adjust size based on delta
    delta = k_n - len(S_prev)
    if delta > 0:
        # Add delta tokens from the threshold-limited range
        add_count = min(delta, len(available_to_add))
        if add_count > 0:
            to_add = random.sample(available_to_add, add_count)
            S_n.extend(to_add)
    elif delta < 0:
        # Remove -delta tokens, preferring more recent ones in S_n
        remove_count = min(-delta, len(S_n))
        if remove_count > 0:
            # Sort S_n descending to prioritize removing recent tokens
            S_n_sorted = sorted(S_n, reverse=True)
            to_remove = S_n_sorted[:remove_count]
            S_n = [t for t in S_n if t not in to_remove]
    
    # Introduce controlled variation by swapping tokens
    swap_count = int(diff_ratio * len(S_prev))  # Max tokens to swap
    available_to_remove = sorted(S_n, reverse=True)[:swap_count]  # Prefer recent tokens
    available_to_add = list(set(range(max_skip_index + 1)) - set(S_n))
    swap_count = min(swap_count, len(available_to_remove), len(available_to_add))
    if swap_count > 0:
        to_remove = available_to_remove[:swap_count]
        to_add = random.sample(available_to_add, swap_count)
        S_n = [t for t in S_n if t not in to_remove]
        S_n.extend(to_add)
    
    return sorted(S_n)

# Generate trace file
with open("trace.txt", "w") as f:
    # Initialize S_prev for n = N_pre
    S_prev = generate_initial_skipped_tokens(N_pre, sparsity, threshold_factor)
    skipped_str = f"[{','.join(map(str, S_prev))}]" if S_prev else "[]"
    for l in range(L):
        for s in [0, 1]:
            skip_kv = skipped_str if s == 0 else "[]"
            f.write(f"{N_pre},{l},{s},{skip_kv},False\n")
    
    # Generate subsequent tokens
    for n in range(N_pre + 1, N_pre + N):
        S_n = generate_similar_skipped_tokens(S_prev, n, sparsity, diff_ratio, threshold_factor)
        skipped_str = f"[{','.join(map(str, S_n))}]" if S_n else "[]"
        for l in range(L):
            for s in [0, 1]:
                skip_kv = skipped_str if s == 0 else "[]"
                f.write(f"{n},{l},{s},{skip_kv},False\n")
        S_prev = S_n  # Update S_prev for the next iteration

print("Trace file 'trace.txt' generated successfully.")