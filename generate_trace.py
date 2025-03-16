import random

# Configuration parameters
N = 1024 * 10      # Number of tokens to generate in decode stage
N_pre = 1024 * 2  # Starting token index for decode stage
L = 32          # Total layers (matches ModelConfig.L)
sparsity = 0.6  # Target sparsity ratio (adjust as needed)

def generate_skipped_tokens(n, sparsity):
    """Generate skipped tokens for token n with age-biased sparsity."""
    num_previous = n  # Previous tokens are 0..n-1 (total = n)
    if num_previous == 0:
        return []
    
    # Calculate number of tokens to skip
    k = int(round(sparsity * (n - 1)))
    if k <= 0:
        return []
    
    # Generate weighted scores (older tokens get higher weight)
    scores = [(i, (n - i) * random.random()) for i in range(n)]
    # Sort by descending score to prioritize older tokens
    scores.sort(key=lambda x: -x[1])
    # Select top k tokens and sort for consistency
    skipped = sorted([i for i, _ in scores[:k]])
    return skipped

# Generate trace file
with open("trace.txt", "w") as f:
    for n in range(N_pre, N_pre + N):
        skipped = generate_skipped_tokens(n, sparsity)
        skipped_str = f"[{','.join(map(str, skipped))}]" if skipped else "[]"
        
        # Write entries for all layers and stages
        for l in range(L):
            for s in [0, 1]:
                skip_kv = skipped_str if s == 0 else "[]"
                f.write(f"{n},{l},{s},{skip_kv},False\n")

print("Trace file 'trace.txt' generated successfully.")