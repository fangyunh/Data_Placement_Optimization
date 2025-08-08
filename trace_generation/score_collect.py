#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Collect *ranked* cross-token attention indices while decoding on CPU
# – Stores descending-order token indices instead of raw FP32 weights –
# ─────────────────────────────────────────────────────────────────────────────
import os, csv, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ─────────── configurable parameters ───────────
MODEL_NAME      = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR      = "attention_scores"
MAX_NEW_TOKENS  = 10240                # generated tokens per sample
DATASETS        = ["narrativeqa"]           # LongBench subsets
DEVICE          = torch.device("cpu")  # force CPU
DTYPE           = torch.float32
# ────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.set_default_dtype(DTYPE)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)
model.eval();  model.config.use_cache = True

# ─────────── helper ───────────
def _maybe_clip(num_new: int, prompt_len: int, max_ctx: int) -> int:
    """Clip new-token budget so we never exceed the model context window."""
    return max(0, min(num_new, max_ctx - prompt_len))

# ─────────── core routine ───────────
@torch.no_grad()
def collect_attention_scores(sample_id: str, prompt: str, max_new_tokens: int):
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=model.config.max_position_embeddings)
    input_ids      = enc.input_ids.to(DEVICE)           # (1, L₀)
    attention_mask = enc.attention_mask.to(DEVICE)      # (1, L₀)
    prompt_len     = input_ids.size(-1)

    new_tok_budget = _maybe_clip(
        max_new_tokens, prompt_len, model.config.max_position_embeddings
    )
    if new_tok_budget == 0:
        print(f"⚠️  Prompt too long for {sample_id}; skipped.")
        return

    # ── PREFILL ──
    prefill_out = model(
        input_ids      = input_ids,
        attention_mask = attention_mask,
        use_cache=True,
        output_attentions=False,
    )
    past_kv    = prefill_out.past_key_values
    next_token = prefill_out.logits.argmax(dim=-1)[:, -1:]   # (1,1)
    position   = prompt_len                                  # first gen idx

    csv_path = os.path.join(OUTPUT_DIR, f"{sample_id}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["position", "layer", "indices_desc"])  # header

        for _ in tqdm(range(new_tok_budget), desc=f"decode {sample_id}", leave=False):
            # extend mask BEFORE forward pass
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1
            )

            out = model(
                input_ids         = next_token,
                attention_mask    = attention_mask,
                past_key_values   = past_kv,
                use_cache         = True,
                output_attentions = True,
            )

            # ── store *ranked* indices per layer ──
            for layer_idx, attn in enumerate(out.attentions):
                vec     = attn[0].mean(0)[0]                    # (kv_len,)
                indices = torch.argsort(vec, descending=True)   # (kv_len,)
                writer.writerow([position, layer_idx,
                                 json.dumps(indices.tolist())])

            # prepare next step
            past_kv    = out.past_key_values
            next_token = out.logits.argmax(dim=-1)[:, -1:]
            position  += 1

            if next_token.item() == tok.eos_token_id:
                break

            del out, attn, vec, indices   # free RAM each loop

# ─────────── run over dataset(s) ───────────

if __name__ == "__main__":
    for ds in DATASETS:
        split = load_dataset("THUDM/LongBench", f"{ds}", split="test")
        for index, ex in enumerate(split):
            if index == 0:  # Skip first sample
                continue

            if index >= 5:  # limit to 5 samples
                print(f"Collected attention scores for {index} samples.")
                break

            prompt = (
                f"Context: {ex['context']}\n"
                f"Question: {ex['input']}\n"
                f"Answer:"
            )
            collect_attention_scores(
                sample_id      = ex["_id"],
                prompt         = prompt,
                max_new_tokens = MAX_NEW_TOKENS,
            )
            
            
