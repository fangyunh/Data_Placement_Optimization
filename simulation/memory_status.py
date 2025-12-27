import random
import csv
import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
import ast # Import ast for literal_eval
import sys
from tqdm.auto import tqdm


BYTES_TO_GB = 1024**3

@dataclass
class ModelConfig():
    # Model architecture parameters
    def __init__(self, N=1024*8, N_pre=256, para_num=8, 
                C_HBM_max=3, B_ext_R=450, B_ext_W=450, batch_size=1):
        self.L: int = 32          # Number of layers
        self.d: int = 4096        # Hidden dimension
        self.h: int = 32          # Number of attention heads
        self.d_ff: int = 14000    # Feed-forward dimension
        self.dtype_size: int = 2  # Bytes per parameter (e.g., 2 for FP16)
        self.para_num: int = para_num * 1000000000 
        
        # --- MoE Specific Parameters ---
        self.num_experts: int = 1       # Total number of experts
        self.num_experts_per_tok: int = 1 # Number of experts activated per token
        
        # --- KV Cache & GQA Parameters ---
        self.num_key_value_heads: int = self.h
        self.head_dim: int = self.d // self.h # d_k and d_v
        
        # Memory parameters
        self.B_HBM: float = 4900  # HBM bandwidth in GB/s
        self.B_ext_interface_R: float = B_ext_R  # External memory interface read (GB/s)
        self.B_ext_interface_W: float = B_ext_W  # External memory interface write (GB/s)
        self.B_ext_internal: float = 500    # External memory internal bandwidth (GB/s)
        self.C_HBM_max: float = C_HBM_max * BYTES_TO_GB # HBM capacity in B
        
        # Inference parameters
        self.N: int = N           # Total tokens to generate
        self.N_pre: int = N_pre   # Previous tokens from prefilling
        self.batch_size: int = batch_size # Batch size
        
        self.best_alpha = self.B_HBM / (self.B_HBM + min(self.B_ext_interface_R, self.B_ext_internal))

class ScaledLLaMa3_8BConfig(ModelConfig):
    def __init__(self, N=1024, N_pre=256, para_num=0.125, C_HBM_max=0.375):
        super().__init__(N, N_pre, para_num, C_HBM_max)
        self.L = 28          # Number of layers
        self.d = 512        # Hidden dimension e.g. 
        self.h = 4          # Number of attention heads
        self.d_ff = 2048    # Feed-forward dimension e.e. 16384

class Mixtral8x7BConfig(ModelConfig):
    """
    Configuration for Mixtral-8x7B based on the official config.json.
    """
    def __init__(self, N=1024*8, N_pre=256, para_num=46.7, 
                 C_HBM_max=100, B_ext_R=450, B_ext_W=450, batch_size=1):
        # Set para_num to 46.7B (total unique params)
        super().__init__(N, N_pre, para_num, C_HBM_max, B_ext_R, B_ext_W, batch_size)
        
        # --- Architecture from config.json ---
        self.L: int = 32
        self.d: int = 4096
        self.h: int = 32
        self.d_ff: int = 14336 
        
        # --- MoE Specific ---
        self.num_experts: int = 8
        self.num_experts_per_tok: int = 2
        
        # --- GQA Specific ---
        self.num_key_value_heads: int = 8
        self.head_dim: int = self.d // self.h # 4096 // 32 = 128
        
class MemStatus(ABC):
    def __init__(self, config: ModelConfig, trace_reader, is_inclusive: bool):
        self.trace = trace_reader
        self.cfg = config
        
        self.total_model_weights: float =  self.cfg.para_num * self.cfg.dtype_size 
        self.start_token_id = self.cfg.N_pre
        self.threshold = 0.99
        self.inclusive = is_inclusive

        # --- KV HBM Tracking ---
        # Stores (query_num, token_index) tuples
        self.hbm_kv_tokens = [set() for _ in range(self.cfg.L)]
        
        # --- Expert HBM Tracking ---
        # Stores expert_id (e.g., 0-7)
        self.hbm_expert_slices = [set() for _ in range(self.cfg.L)]

        # --- KV Cache Size Calculation (GQA Aware) ---
        self.single_KV_cache_size = (self.cfg.num_key_value_heads * self.cfg.head_dim * self.cfg.dtype_size) + \
                                    (self.cfg.num_key_value_heads * self.cfg.head_dim * self.cfg.dtype_size)
        
        # --- MoE Model Weight Calculation ---
        # 1. Static Weights (MHA + Gating) - Per Layer
        mha_q_size = (self.cfg.d * self.cfg.h * self.cfg.head_dim) * self.cfg.dtype_size
        mha_k_size = (self.cfg.d * self.cfg.num_key_value_heads * self.cfg.head_dim) * self.cfg.dtype_size
        mha_v_size = (self.cfg.d * self.cfg.num_key_value_heads * self.cfg.head_dim) * self.cfg.dtype_size
        mha_o_size = (self.cfg.h * self.cfg.head_dim * self.cfg.d) * self.cfg.dtype_size
        gating_weight = (self.cfg.d * self.cfg.num_experts) * self.cfg.dtype_size
        self.static_weight_per_layer_size = mha_q_size + mha_k_size + mha_v_size + mha_o_size + gating_weight
        self.total_static_weights_geom = self.static_weight_per_layer_size * self.cfg.L
        
        # 2. Calculate Total Expert Weights
        expert_mlp_params_per_layer = (self.cfg.d * self.cfg.d_ff * 2) * self.cfg.dtype_size 
        total_expert_weight_geom = expert_mlp_params_per_layer * self.cfg.num_experts * self.cfg.L
        
        # 3. Calculate Missing Component
        total_counted_weight = self.total_static_weights_geom + total_expert_weight_geom
        self.missing_static_size = self.total_model_weights - total_counted_weight
        if self.missing_static_size < 0:
            self.missing_static_size = 0.0
        
        # --- Final Memory Allocations ---
        self.total_static_allocation = self.total_static_weights_geom + self.missing_static_size
        
        total_expert_slices = self.cfg.L * self.cfg.num_experts
        if total_expert_slices > 0:
            self.single_expert_slice_size = total_expert_weight_geom / total_expert_slices
        else:
            self.single_expert_slice_size = 0.0
            
        # --- HBM Capacity Management ---
        self.hbm_capacity_remaining = self.cfg.C_HBM_max - self.total_static_allocation
        
        if self.hbm_capacity_remaining < 0:
            print(f"ERROR: Static weights ({self.total_static_allocation/BYTES_TO_GB:.2f} GB) exceed total HBM ({self.cfg.C_HBM_max/BYTES_TO_GB:.2f} GB).")
        
        self.foresight_window = 0
        self.initialize_memory()
    
    def initialize_memory(self):
        """Initialize HBM with model parameters and KV cache."""
        self.initial_tokens_placement()
        # print(f"Initialization complete. {self.hbm_capacity_remaining / BYTES_TO_GB:.2f} GB HBM remaining.")

    # --- HBM Tracking Methods ---
    
    # --- KV Cache ---
    def is_kv_in_hbm(self, query: int, token: int, layer: int) -> bool:
        return (query, token) in self.hbm_kv_tokens[layer]
    
    def add_kv_to_hbm(self, query: int, token: int, layer: int):
        if self.is_kv_in_hbm(query, token, layer):
            return True # Already present
        
        if self.hbm_capacity_remaining >= self.single_KV_cache_size:
            self.hbm_kv_tokens[layer].add((query, token))
            self.hbm_capacity_remaining -= self.single_KV_cache_size
            
            # --- CRITICAL SAFETY CHECK ---
            if self.hbm_capacity_remaining < -1e-9:
                 raise ValueError(f"CRASH: HBM Underflow! KV Add L{layer} Q{query} T{token}. Capacity: {self.hbm_capacity_remaining/BYTES_TO_GB:.10f} GB")
            return True
        else:
            return False # HBM is full
    
    def remove_kv_from_hbm(self, query: int, token: int, layer: int):
        if self.is_kv_in_hbm(query, token, layer):
            self.hbm_kv_tokens[layer].discard((query, token))
            self.hbm_capacity_remaining += self.single_KV_cache_size
            
            # --- CRITICAL SAFETY CHECK ---
            if self.hbm_capacity_remaining > self.cfg.C_HBM_max * 1.000000001:
                 raise ValueError(f"CRASH: HBM Overflow! KV Remove L{layer} Q{query} T{token}. Capacity: {self.hbm_capacity_remaining/BYTES_TO_GB:.10f} GB")

    # --- Expert Slices ---
    def is_expert_in_hbm(self, expert_id: int, layer: int) -> bool:
        return expert_id in self.hbm_expert_slices[layer]

    def add_expert_to_hbm(self, expert_id: int, layer: int):
        if self.is_expert_in_hbm(expert_id, layer):
            return True # Already present
        
        if self.hbm_capacity_remaining >= self.single_expert_slice_size:
            self.hbm_expert_slices[layer].add(expert_id)
            self.hbm_capacity_remaining -= self.single_expert_slice_size
            
            # --- CRITICAL SAFETY CHECK ---
            if self.hbm_capacity_remaining < -1e-9:
                 raise ValueError(f"CRASH: HBM Underflow! Expert Add L{layer} E{expert_id}. Capacity: {self.hbm_capacity_remaining/BYTES_TO_GB:.10f} GB")
            return True
        else:
            return False # HBM is full
    
    def remove_expert_from_hbm(self, expert_id: int, layer: int):
        if self.is_expert_in_hbm(expert_id, layer):
            self.hbm_expert_slices[layer].discard(expert_id)
            self.hbm_capacity_remaining += self.single_expert_slice_size
            
            # --- CRITICAL SAFETY CHECK ---
            if self.hbm_capacity_remaining > self.cfg.C_HBM_max * 1.000000001:
                 raise ValueError(f"CRASH: HBM Overflow! Expert Remove L{layer} E{expert_id}. Capacity: {self.hbm_capacity_remaining/BYTES_TO_GB:.10f} GB")
    
    # --- End of HBM Tracking Methods ---
    
    def calculate_data_sizes(self, n: int, l: int):
        active_experts = self.trace.get_active_experts(n, l)
        kv_reads_all_queries = self.trace.get_all_kv_reads_by_query(n, l)

        # Model Weight Read
        total_expert_weight = len(active_experts) * self.single_expert_slice_size
        total_model_weight = self.static_weight_per_layer_size + total_expert_weight

        # KV Cache Read
        total_kv_read_count = sum(len(tokens) for tokens in kv_reads_all_queries.values())
        total_kv_read_size = total_kv_read_count * self.single_KV_cache_size
        
        D_R = total_model_weight + total_kv_read_size
        
        # Data Write
        D_W = self.single_KV_cache_size * self.cfg.batch_size
            
        return D_R, D_W

    def max_alpha(self, n: int, l: int):
        active_experts = self.trace.get_active_experts(n, l)
        kv_reads_all_queries = self.trace.get_all_kv_reads_by_query(n, l)

        # --- Calculate Model Weight Component ---
        total_model_weight = 0
        model_weight_in_hbm = 0
        
        # 1. Static Weights (Assumed always in HBM)
        model_weight_in_hbm += self.static_weight_per_layer_size
        total_model_weight += self.static_weight_per_layer_size
        
        # 2. Expert Weights
        for expert_id in active_experts:
            total_model_weight += self.single_expert_slice_size
            if self.is_expert_in_hbm(expert_id, l):
                model_weight_in_hbm += self.single_expert_slice_size

        # --- Calculate KV Cache Component ---
        total_kv_read_size = 0
        kv_cache_in_hbm_size = 0
        
        for query_num, token_list in kv_reads_all_queries.items():
            for token in token_list:
                total_kv_read_size += self.single_KV_cache_size
                if self.is_kv_in_hbm(query_num, token, l):
                    kv_cache_in_hbm_size += self.single_KV_cache_size

        # --- Calculate Final Alpha ---
        D_R = total_model_weight + total_kv_read_size
        if D_R <= 0:
            return 0.0
        
        total_hbm_read = model_weight_in_hbm + kv_cache_in_hbm_size
        max_alpha = total_hbm_read / D_R
            
        return max(0.0, min(max_alpha, 1.0))
    
    @abstractmethod
    def initial_tokens_placement(self):
        pass

class HBMInit(MemStatus):
    def __init__(self, config, trace, is_inclusive):
        self.model_weight_ratio = 1.0
        super().__init__(config, trace, is_inclusive)
        
    def initial_tokens_placement(self):
        # 1. Place Expert Slices
        expert_placement_successful = True
        for l in range(self.cfg.L):
            for e in range(self.cfg.num_experts):
                if not self.add_expert_to_hbm(e, l):
                    expert_placement_successful = False
                    break
            if not expert_placement_successful: break

        # 2. Place Prefill KV Cache
        kv_placement_successful = True
        for q in range(self.cfg.batch_size):
            for n in range(self.cfg.N_pre):
                for l in range(self.cfg.L):
                    if not self.add_kv_to_hbm(q, n, l):
                        kv_placement_successful = False
                        break
                if not kv_placement_successful: break
            if not kv_placement_successful: break

class HBMInitPaged(MemStatus):
    def __init__(self, config, trace, is_inclusive):
        self.model_weight_ratio = 1.0
        self.page_size = 16
        super().__init__(config, trace, is_inclusive)
    
    def initial_tokens_placement(self):
        # 1. Place Expert Slices
        expert_placement_successful = True
        for l in range(self.cfg.L):
            for e in range(self.cfg.num_experts):
                if not self.add_expert_to_hbm(e, l):
                    expert_placement_successful = False
                    break
            if not expert_placement_successful: break

        # 2. Place Prefill KV Cache (Paged)
        num_pages = self.cfg.N_pre // self.page_size
        kv_placement_successful = True
        
        for page in range(num_pages):
            start_token = page * self.page_size
            end_token = start_token + self.page_size
            for q in range(self.cfg.batch_size):
                for token in range(start_token, end_token):
                    for l in range(self.cfg.L):
                        if not self.add_kv_to_hbm(q, token, l):
                            kv_placement_successful = False
                            break
                    if not kv_placement_successful: break
                if not kv_placement_successful: break
            if not kv_placement_successful: break
                
class LookAheadInit(MemStatus):
    def __init__(self, config, trace, is_inclusive):
        self.model_weight_ratio = 1.0
        super().__init__(config, trace, is_inclusive)
        
    def _collect_required_working_set(self, n_start: int):
        required_items_set = set() 
        for l in range(self.cfg.L):
            try:
                experts_read = self.trace.get_active_experts(n_start, l)
                kv_reads = self.trace.get_all_kv_reads_by_query(n_start, l)
            except (IOError, IndexError, KeyError):
                continue
            
            for e in experts_read:
                item_tuple = (self.single_expert_slice_size, "EXPERT", l, e)
                required_items_set.add(item_tuple)
            
            for q, tokens in kv_reads.items():
                for t in tokens:
                    if t < self.cfg.N_pre:
                        item_tuple = (self.single_KV_cache_size, "KV", l, q, t)
                        required_items_set.add(item_tuple)
        return list(required_items_set)

    def initial_tokens_placement(self):
        n_start = self.cfg.N_pre
        # 1. Collect required working set
        unique_required_items = self._collect_required_working_set(n_start)
        
        # --- FALLBACK PROTECTION ---
        if not unique_required_items:
            # print(f"Warning: Trace read failed or empty for n={n_start}. Falling back to HBMInit expert placement.", file=sys.stderr)
            expert_placement_successful = True
            for l in range(self.cfg.L):
                for e in range(self.cfg.num_experts):
                    if not self.add_expert_to_hbm(e, l):
                        expert_placement_successful = False
                        break
                if not expert_placement_successful:
                    break
            return

        # 2. Sort items: Experts first
        unique_required_items.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Place items in HBM
        for item in unique_required_items:
            size, item_type, l = item[0], item[1], item[2]
            
            if self.hbm_capacity_remaining < size:
                break 

            if item_type == "KV":
                q, t = item[3], item[4]
                self.add_kv_to_hbm(q, t, l)
            elif item_type == "EXPERT":
                e = item[3]
                self.add_expert_to_hbm(e, l)