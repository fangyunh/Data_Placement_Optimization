import random
import numpy as np
import csv
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
BYTES_TO_GB = 1024**3

@dataclass
class ModelConfig():
    # Model architecture parameters
    # llama 3.3
    def __init__(self, N=1024*2, N_pre=1024*2, para_num=0.5, 
                C_HBM_max=3):
        self.L: int = 32          # Number of layers
        self.d: int = 16384        # Hidden dimension 3.g. 8192
        self.h: int = 32          # Number of attention heads
        self.d_ff: int = 8192    # Feed-forward dimension e.e. 16384
        self.dtype_size: int = 2  # Bytes per parameter (e.g., 2 for FP16)
        self.para_num: int = para_num * 1000000000 # how many parameters in the model 2B 4GB
        
        # Memory parameters
        self.B_HBM: float = 4915  # HBM bandwidth in GB/s  e.g 4.8TB/s  B/ns
        self.B_ext_interface_R: float = 900  # External memory interface read (GB/s) B/ns
        self.B_ext_interface_W: float = 900  # External memory interface write (GB/s) B/ns
        self.B_ext_internal: float = 1900    # External memory internal bandwidth (GB/s) B/ns
        self.C_HBM_max: float = C_HBM_max * BYTES_TO_GB          # HBM capacity in B, 10GB
        self.C_HBM: float = 0.0
        # Inference parameters
        self.N: int = N       # Total tokens 2GB
        self.N_pre: int = N_pre   # Previous tokens from prefilling 1.71GB
        self.best_alpha = self.B_HBM / (self.B_HBM + min(self.B_ext_interface_R, self.B_ext_internal))

# Records each token's KV caches store at where
class MemStatus(ABC):
    def __init__(self, config: ModelConfig, trace, weight_on_HBM, is_inclusive: bool):
        self.trace = trace
        self.cfg = config
        self.token_layer_status = {}
        self.total_model_weights: float =  self.cfg.para_num * self.cfg.dtype_size 
        self.start_token_id = self.cfg.N_pre
        # memory threshold rate
        self.threshold = 0.99
        self.model_weight_ratio = weight_on_HBM
        self.inclusive = is_inclusive
        self.initialize_memory()
    
    def initialize_memory(self):
        """Initialize HBM with model parameters and KV cache."""
        self.cfg.C_HBM = 0.0
        HBM_model_size = self.total_model_weights * self.model_weight_ratio
        self.store_data(HBM_model_size)

        self.initial_tokens_placement()
        print(f"Initialization complete, HBM utilizaiton rate: {self.get_HBM_util_rate() * 100}%.")

    
    def initialize_token(self, token_id):
        """Ensure that a token has been initialized in token_layer_status."""
        if token_id not in self.token_layer_status:
            # 0: on HBM, 1: on the external memory, 2: The layer's KV cache
            # was not calculated (skip), 3: initial state, unarranged.
            # Default: all layers set to 3 (undecided)
            self.token_layer_status[token_id] = [3] * self.cfg.L
    
    def get_layer_location(self, token_id, layer: int) -> int:
        """Return the location of a token's KV cache at a specific layer.
           0: HBM, 1: External, 2: Skipped.
        """
        self.initialize_token(token_id)
        return self.token_layer_status[token_id][layer]
    
    def update_token_layer(self, token_id, layer: int, location: int):
        """
        Update the location for a given token and layer.
        location should be 0 (HBM), 1 (External), or 2 (Skipped).
        """
        self.initialize_token(token_id)
        self.token_layer_status[token_id][layer] = location
    
    def get_effective_token_size(self, token_id) -> int:
        """
        Returns the effective KV cache size for a given token.
        For each layer not marked as skipped (i.e. location != 2), add 2*d*dtype_size.
        """
        self.initialize_token(token_id)
        effective_layers = sum(1 for loc in self.token_layer_status[token_id] if loc != 2)
        return effective_layers * 2 * self.cfg.d * self.cfg.dtype_size

    def get_single_KV_cache_size(self) -> int:
        return 2 * self.cfg.d * self.cfg.dtype_size
    
    # Attention model weight soze
    def get_layer_md_weight_size(self) -> float:
        return 4 * self.cfg.d**2 * self.cfg.dtype_size
    
    def get_HBM_util_rate(self) -> float:
        return self.cfg.C_HBM / self.cfg.C_HBM_max
    
    def exceed_threshold(self) -> bool:
        flag = self.get_HBM_util_rate() >= self.threshold
        return flag

    def store_data(self, data_size):
        """Attempt to store data in HBM, return True if successful."""
        remaining = self.cfg.C_HBM_max - self.cfg.C_HBM
        if remaining <= 0:
            return False
        if data_size <= remaining:
            self.cfg.C_HBM += data_size
            return True
        return False
    
    def calculate_data_sizes(self, n: int, l: int, s: int):
        """Calculate read/write data sizes for current step."""
        step_info = self.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        # If this layer is skipped in the trace, no data is processed.
        if step_info["skip_layer"]:
            return 0, 0
        skip_tokens = set(step_info["skip_token_kv"])

        if s == 0:  # MHA
            D_R = self.get_layer_md_weight_size() + n * self.get_single_KV_cache_size()
            D_W = 2 * self.cfg.d * self.cfg.dtype_size
            count = 0
            for token_id, layer_status in self.token_layer_status.items():
                if token_id in skip_tokens:
                    continue
                if layer_status[l] == 2:
                    count += 1
            # Reduce D_R by size of skipped KV caches           
            skipped_kv_size = (len(step_info["skip_token_kv"]) + count) * 2 * self.cfg.d * self.cfg.dtype_size
            D_R -= skipped_kv_size
            
        else:  # MLP
            D_R = 2 * self.cfg.d * self.cfg.d_ff * self.cfg.dtype_size
            D_W = 0
        return D_R, D_W

    def max_and_best_alphas(self, n: int, l: int, s: int):
        D_R, _ = self.calculate_data_sizes(n, l, s)
        if D_R <= 0:
            return [0.0, 0.0]
        
        if s == 1:
            if self.inclusive:
                return [self.model_weight_ratio, min(self.cfg.best_alpha, self.model_weight_ratio)]
            else:
                return [self.model_weight_ratio, self.model_weight_ratio]
        
        # Model weight part: assume that the model weight component in D_R is 4*d^2*dtype_size.
        model_weight_component = self.get_layer_md_weight_size()
        effective_model_weight = self.model_weight_ratio * model_weight_component

        # Retrieve current step's trace to get skip_token_kv list.
        step_info = self.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        skip_tokens = set(step_info["skip_token_kv"])

        # Count over all tokens that have been processed so far (or that are tracked)
        count_hbm = 0
        count_tot_tokens = 0
        for token_id, layer_status in self.token_layer_status.items():
            # Only consider tokens that are NOT skipped for the current step.
            if token_id in skip_tokens:
                continue
            if layer_status[l] == 0:
                count_hbm += 1
            if layer_status[l] == 0 or layer_status[l] == 1:
                count_tot_tokens += 1

        # count_hbm should also include prefill tokens
        effective_KV_cache = count_hbm * self.get_single_KV_cache_size()
        
        max_alpha = (effective_model_weight + effective_KV_cache) / D_R
        
        # Calculate the best alpha for inclusive HBM
        best_alpha = 0.0
        best_HBM_tokens = math.floor(self.cfg.best_alpha * count_tot_tokens)
        best_effective_KV_cahe = min(best_HBM_tokens, count_hbm) * self.get_single_KV_cache_size()
        if self.inclusive:
            best_alpha = (model_weight_component * min(self.cfg.best_alpha, self.model_weight_ratio) 
                        + best_effective_KV_cahe) / D_R
            
        # Ensure alpha is within [0, 1]
        return [max(0.0, min(max_alpha, 1.0)), max(0.0, min(best_alpha, 1.0))]
    
    
    @abstractmethod
    def initial_tokens_placement(self):
        pass

# Firstly, records model weights and prefill KV cache in the HBM.
# IF the space is not enough, store in the external memory.
class HBMInit(MemStatus):
    def __init__(self, config, trace, weight_on_HBM, is_inclusive):
        super().__init__(config, trace, weight_on_HBM, is_inclusive)
    
    def initial_tokens_placement(self):
        print(f"Start HBMInit initialization")
        kv_cache_size = self.get_single_KV_cache_size()
        # store prefill tokens on HBM until full
        for n in range (self.cfg.N_pre):
            for l in range(self.cfg.L):
                if self.store_data(kv_cache_size):
                    self.update_token_layer(n, l, 0)
                else:
                    self.update_token_layer(n, l, 1)
        print(f"End HBMInit initialization")


# Store best ratio of prefill tokens on HBM (token level)
class TokenLevelBestRatioInit(MemStatus):
    def __init__(self, config, trace, weight_on_HBM, is_inclusive):
        super().__init__(config, trace, weight_on_HBM, is_inclusive)
    
    def initial_tokens_placement(self):
        print(f"Start TokenLevelInit initialization")
        kv_cache_size = self.get_single_KV_cache_size()
        batch = 32
        on_HBM_tokens = math.floor(batch * self.cfg.best_alpha)
        for n in range (self.cfg.N_pre):
            for l in range(self.cfg.L):
                if (n % batch <= on_HBM_tokens):
                    if self.store_data(kv_cache_size):
                        self.update_token_layer(n, l, 0)
                    else:
                        self.update_token_layer(n, l, 1)
                else:
                    self.update_token_layer(n, l, 1)
        print(f"End TokenLevelInit initialization")

