import random
# modify to include prefill tokens

# Configuration parameters
N = 4096         # total tokens (should match ModelConfig.N)
L = 32           # total layers (should match ModelConfig.L)

# Probabilities for dynamic token importance (these can be tuned)
# p_return: probability that a token in the skipped list returns to being important
# p_skip: probability to mark a previous token as skip-worthy in the current token generation
p_return = 0.3
p_skip = 0.05

# Limits based on token index
def get_skip_limit(n):
    if n > 4000:
        return 30
    elif n > 3000:
        return 25
    elif n > 2000:
        return 20
    elif n > 1000:
        return 15
    else:
        return 10

def get_skip_layer(n):
    if n > 3000:
        return 0
    elif n > 1500:
        return 1
    elif n > 1000:
        return 2
    elif n > 500:
        return 3
    else:
        return 5

# Global list for skipped tokens (stores token indices that are considered "skipped")
skipped_tokens = []

# Open file for writing the trace
with open("trace.txt", "w") as f:
    # Write header
    f.write("n,l,s,skip_token_kv,skip_layer\n")
    
    # For each token generation step
    for n in range(N):
        # Step 1: For tokens already in the skip list, randomly decide to return them (simulate becoming important)
        new_skipped = []
        for token in skipped_tokens:
            if random.random() > p_return:  # keep as skipped with probability (1-p_return)
                new_skipped.append(token)
        skipped_tokens = new_skipped

        # Step 2: Iterate through previous tokens (0 to n-1) and decide to mark them as skipped
        skip_limit = get_skip_limit(n)
        # Only consider tokens that are not already in the list
        for i in range(n):
            if i not in skipped_tokens:
                if random.random() < p_skip:
                    skipped_tokens.append(i)
                    # If we've reached the limit, break out
                    if len(skipped_tokens) >= skip_limit:
                        break
        # Ensure we do not exceed the limit (if over, trim randomly)
        if len(skipped_tokens) > skip_limit:
            skipped_tokens = random.sample(skipped_tokens, skip_limit)

        # Step 3: For the current token n, randomly choose SKIP_LAYERS_PER_TOKEN layers to skip
        skipped_layers = set(random.sample(range(L), get_skip_layer(n)))

        # For each layer and each stage, write a trace record.
        for l in range(L):
            for s in [0, 1]:
                # For stage 0 (e.g., MHA) we record the current list of skipped tokens,
                # for stage 1 (e.g., MLP) it’s not used.
                if s == 0:
                    skip_token_kv = skipped_tokens.copy()
                else:
                    skip_token_kv = []
                # If the layer is in the skipped layers for this token, mark skip_layer True
                skip_layer = (l in skipped_layers)
                
                # Write record as: n, l, s, [list], skip_layer (as True/False)
                # The format follows the pattern expected by the simulator's regex.
                f.write(f"{n},{l},{s},{skip_token_kv},{skip_layer}\n")

print("Trace file 'trace.txt' generated successfully.")
