#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Collect Attention & Expert Indices for Mixtral-8x7B (LongBench Batching)
# ─────────────────────────────────────────────────────────────────────────────
# 1. Loads specific LongBench sub-dataset (e.g. 'qasper').
# 2. Selects a specific batch of samples (Start Index -> Start + Batch Size).
# 3. Enforces strict PREFILL_LEN and DECODE_LEN for every query.
# 4. Runs sequentially on CPU and merges into a synchronized batch trace.
# ─────────────────────────────────────────────────────────────────────────────

import os
import csv
import json
import torch
import shutil
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ─────────── Configurable Parameters ───────────
MODEL_NAME        = "mistralai/Mixtral-8x7B-Instruct-v0.1" # Or local path
OUTPUT_FILE       = "mixtral_fixed_batch_trace.csv"
TEMP_DIR          = "temp_query_traces"

# STRICT LENGTH CONTROLS
PREFILL_LEN       = 4096           # Fixed prompt length for all queries
DECODE_LEN        = 8192            # Fixed generation steps for all queries

# DATASET CONTROLS
DATASET_SUBSET    = "gov_report"       # Which LongBench dataset to load
DATASET_START_IDX = 4              # Start index in the dataset
BATCH_SIZE        = 32              # How many samples to run (e.g. 4, 5, 6, 7)

DEVICE            = torch.device("cpu")
DTYPE             = torch.float32  # CPU usually requires float32
TOP_K_EXPERTS     = 2              # Mixtral specific
# ────────────────────────────────────────────────

def get_expert_hook(layer_idx, storage_list):
    """
    Hook to capture router logits from Mixtral's MoE gate.
    Stores Top-K expert indices.
    """
    def hook(module, input, output):
        with torch.no_grad():
            # output shape: (batch_size * sequence_length, num_experts)
            probs = torch.softmax(output, dim=-1)
            topk_indices = torch.topk(probs, k=TOP_K_EXPERTS, dim=-1).indices
            
            # Store as list of integers [e1, e2]
            indices_list = topk_indices[0].tolist() 
            storage_list[layer_idx] = indices_list
    return hook

@torch.no_grad()
def run_single_query(query_idx_in_batch, sample_data, model, tokenizer, temp_dir):
    """
    Runs inference for a single query with strict length enforcement.
    query_idx_in_batch: 0 to BATCH_SIZE-1 (used for file naming)
    """
    # 1. Form Proper Prompt
    # LongBench Format: Context (Long) + Input (Question/Command)
    prompt = f"Context: {sample_data['context']}\n\nQuestion: {sample_data['input']}\n\nAnswer:"
    
    print(f"--> [Batch Query {query_idx_in_batch}] Tokenizing & enforcing length...")
    
    # Encode without strict length first
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids
    
    # 2. Enforce Strict PREFILL_LEN
    curr_len = input_ids.size(1)
    
    if curr_len > PREFILL_LEN:
        # Truncate to keep the last PREFILL_LEN tokens (usually contains the Question)
        input_ids = input_ids[:, -PREFILL_LEN:]
        attention_mask = torch.ones_like(input_ids)
    elif curr_len < PREFILL_LEN:
        # Pad to the left
        pad_len = PREFILL_LEN - curr_len
        pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=input_ids.dtype)
        input_ids = torch.cat([pad_ids, input_ids], dim=1)
        
        pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
        real_mask = torch.ones((1, curr_len), dtype=torch.long)
        attention_mask = torch.cat([pad_mask, real_mask], dim=1)
    else:
        attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    
    print(f"    Input shape enforced: {input_ids.shape} (Expected 1, {PREFILL_LEN})")

    # 3. Register Hooks
    current_step_experts = {}
    handles = []
    
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'block_sparse_moe'):
            h = layer.block_sparse_moe.gate.register_forward_hook(
                get_expert_hook(i, current_step_experts)
            )
            handles.append(h)
    
    # 4. Prefill (Batch Process Prompt)
    print(f"--> [Batch Query {query_idx_in_batch}] Prefilling...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=False
    )
    past_key_values = outputs.past_key_values
    next_token = outputs.logits.argmax(dim=-1)[:, -1:]
    
    # 5. Decode Loop
    temp_csv_path = os.path.join(temp_dir, f"q_{query_idx_in_batch}.csv")
    with open(temp_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "layer_id", "experts", "attention"])
        
        # Absolute token index starts exactly at PREFILL_LEN
        current_token_absolute_idx = PREFILL_LEN
        
        for _ in tqdm(range(DECODE_LEN), desc=f"Query {query_idx_in_batch} Decoding", leave=False):
            current_step_experts.clear()
            
            # Extend mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1
            )
            
            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True
            )
            
            # Write rows for this step
            for layer_idx, attn_tensor in enumerate(out.attentions):
                # attn_tensor: (1, 32, 1, seq_len) -> Average heads -> (seq_len,)
                attn_avg = attn_tensor[0].mean(dim=0)[0] 
                
                # Get Ranked Indices
                sorted_indices = torch.argsort(attn_avg, descending=True).tolist()
                
                # Get Experts
                experts = current_step_experts.get(layer_idx, [])
                
                writer.writerow([
                    current_token_absolute_idx,
                    layer_idx,
                    json.dumps(experts),
                    json.dumps(sorted_indices)
                ])
            
            past_key_values = out.past_key_values
            next_token = out.logits.argmax(dim=-1)[:, -1:]
            current_token_absolute_idx += 1
                
    for h in handles: h.remove()
    del past_key_values, outputs, out
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def merge_and_sort_traces(temp_dir, output_file, batch_size, model_layers):
    """
    Merges traces assuming perfect synchronization.
    Order: AbsoluteTokenID -> LayerID -> QueryID
    """
    print("\nMerging traces...")
    
    file_handles = [open(os.path.join(temp_dir, f"q_{q}.csv"), 'r') for q in range(batch_size)]
    csv_readers = [csv.reader(f) for f in file_handles]
    
    # Skip Headers
    for r in csv_readers: next(r)
    
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["query_id", "token_id", "layer_id", "experts", "attention"])
        
        start_idx = PREFILL_LEN
        end_idx = PREFILL_LEN + DECODE_LEN
        
        total_steps = (end_idx - start_idx) * model_layers
        pbar = tqdm(total=total_steps, desc="Merging Batches")
        
        for t_idx in range(start_idx, end_idx):
            for l_idx in range(model_layers):
                for q in range(batch_size):
                    try:
                        row = next(csv_readers[q])
                        experts = row[2]
                        attn = row[3]
                        writer.writerow([q, t_idx, l_idx, experts, attn])
                    except StopIteration:
                        pass
                
                pbar.update(1)
        pbar.close()

    for f in file_handles: f.close()
    print(f"Trace generation complete: {output_file}")


def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    torch.set_default_dtype(DTYPE)
    
    print(f"Loading Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    model.eval()
    
    print(f"Loading LongBench Dataset: '{DATASET_SUBSET}'...")
    # Load specific sub-dataset
    dataset = load_dataset("THUDM/LongBench", DATASET_SUBSET, split="test")
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_SUBSET}")
    print(f"  Start Index: {DATASET_START_IDX}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Prefill Length: {PREFILL_LEN}")
    print(f"  Decode Length: {DECODE_LEN}")
    
    # Validate indices
    if DATASET_START_IDX + BATCH_SIZE > len(dataset):
        print(f"Error: Requested batch exceeds dataset size ({len(dataset)}).")
        return

    # Iterate through the selected batch
    for i in range(BATCH_SIZE):
        global_idx = DATASET_START_IDX + i
        sample = dataset[global_idx]
        print(f"\nProcessing Batch Item {i+1}/{BATCH_SIZE} (Dataset Index {global_idx}, ID: {sample.get('_id', 'N/A')})")
        
        # We pass 'i' as query_id so the merge function sees 0, 1, 2...
        run_single_query(i, sample, model, tokenizer, TEMP_DIR)
    
    num_layers = len(model.model.layers)
    merge_and_sort_traces(TEMP_DIR, OUTPUT_FILE, BATCH_SIZE, num_layers)
    
    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()