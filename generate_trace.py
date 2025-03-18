import random

# Configuration parameters
N = 1024 * 16      # Number of tokens to generate in decode stage
N_pre = 1024      # Starting token index for decode stage
L = 32            # Total layers (matches ModelConfig.L)
sparsity = 0.1    # Target sparsity ratio
diff_ratio = 0.03  # Maximum difference ratio between consecutive skip sets

def generate_initial_skipped_tokens(n, sparsity):
    """Generate initial skipped tokens for token n."""
    num_previous = n  # Previous tokens are 0..n-1
    if num_previous == 0:
        return []
    k = int(round(sparsity * (n - 1)))
    if k <= 0:
        return []
    # Randomly select k tokens from 0 to n-1
    return sorted(random.sample(range(n), k))

def generate_similar_skipped_tokens(S_prev, n, sparsity, diff_ratio):
    """Generate skipped tokens for token n based on previous skip set S_prev."""
    k_n = int(round(sparsity * (n - 1)))  # Desired size of S_n
    if k_n <= 0:
        return []
    S_n = list(S_prev)  # Start with previous skip set
    delta = k_n - len(S_prev)  # Number of tokens to add or remove
    
    # Available tokens to add: 0 to n-1 excluding current S_n
    available_to_add = list(set(range(n)) - set(S_n))
    
    # Adjust size for delta
    if delta > 0:
        # Add delta tokens if k_n > len(S_prev)
        add_count = min(delta, len(available_to_add))
        if add_count > 0:
            to_add = random.sample(available_to_add, add_count)
            S_n.extend(to_add)
            available_to_add = [t for t in available_to_add if t not in to_add]
    elif delta < 0:
        # Remove -delta tokens if k_n < len(S_prev)
        remove_count = min(-delta, len(S_n))
        if remove_count > 0:
            to_remove = random.sample(S_n, remove_count)
            S_n = [t for t in S_n if t not in to_remove]
    
    # Introduce variation by swapping tokens
    swap_count = int(diff_ratio * len(S_prev))  # Number of tokens to swap
    available_to_remove = S_n
    available_to_add = list(set(range(n)) - set(S_n))
    swap_count = min(swap_count, len(available_to_remove), len(available_to_add))
    if swap_count > 0:
        to_remove = random.sample(available_to_remove, swap_count)
        S_n = [t for t in S_n if t not in to_remove]
        to_add = random.sample(available_to_add, swap_count)
        S_n.extend(to_add)
    
    return sorted(S_n)

# Generate trace file
with open("trace.txt", "w") as f:
    # Initialize S_prev for n = N_pre
    S_prev = generate_initial_skipped_tokens(N_pre, sparsity)
    skipped_str = f"[{','.join(map(str, S_prev))}]" if S_prev else "[]"
    for l in range(L):
        for s in [0, 1]:
            skip_kv = skipped_str if s == 0 else "[]"
            f.write(f"{N_pre},{l},{s},{skip_kv},False\n")
    
    # Generate subsequent tokens
    for n in range(N_pre + 1, N_pre + N):
        S_n = generate_similar_skipped_tokens(S_prev, n, sparsity, diff_ratio)
        skipped_str = f"[{','.join(map(str, S_n))}]" if S_n else "[]"
        for l in range(L):
            for s in [0, 1]:
                skip_kv = skipped_str if s == 0 else "[]"
                f.write(f"{n},{l},{s},{skip_kv},False\n")
        S_prev = S_n  # Update S_prev for next iteration

print("Trace file 'trace.txt' generated successfully.")