import random
import csv
import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
BYTES_TO_GB = 1024**3

@dataclass
class ModelConfig():
    # Model architecture parameters
    # llama 3.1 8B settng
    def __init__(self, N=1024*8, N_pre=256, para_num=8, 
                C_HBM_max=3):
        self.L: int = 32          # Number of layers
        self.d: int = 4096        # Hidden dimension e.g. 8192, with 8192, 1024 tokens = 1GB, 1 layer = 32KB, md_read_l = 0.5GB
                                  # With 4096, 1 layer = 16KB, 2048 tokens = 1GB, md_read_l = 0.0078125 GB
        self.h: int = 32          # Number of attention heads
        self.d_ff: int = 14000    # Feed-forward dimension e.e. 16384
        self.dtype_size: int = 2  # Bytes per parameter (e.g., 2 for FP16)
        self.para_num: int = para_num * 1000000000 # how many parameters in the model 2B 4GB
        
        # Memory parameters
        self.B_HBM: float = 4900  # HBM bandwidth in GB/s  e.g 4.8TB/s  B/ns
        self.B_ext_interface_R: float = 450  # External memory interface read (GB/s) B/ns
        self.B_ext_interface_W: float = 450  # External memory interface write (GB/s) B/ns
        self.B_ext_internal: float = 500    # External memory internal bandwidth (GB/s) B/ns
        self.C_HBM_max: float = C_HBM_max * BYTES_TO_GB          # HBM capacity in B, 10GB
        # Inference parameters
        self.N: int = N       # Total tokens 2GB
        self.N_pre: int = N_pre   # Previous tokens from prefilling
        self.best_alpha = self.B_HBM / (self.B_HBM + min(self.B_ext_interface_R, self.B_ext_internal))

# class ScaledLLaMa3_8BConfig(ModelConfig):
#     def __init__(self, N=1024*2, N_pre=256, para_num=0.5, C_HBM_max=1.5):
#         super().__init__(N, N_pre, para_num, C_HBM_max)
#         # scale the model from 8B to 0.5B, 4:1 ratio, final latency need to multiply by 16 to reach original latency
#         # prefill 256, decode 1024*20 for sparsity 90%
#         self.L = 32          # Number of layers
#         self.d = 1024        # Hidden dimension e.g. 8192, with 8192, 1024 tokens = 1GB, 1 layer = 32KB, md_read_l = 0.5GB
#         self.h = 8          # Number of attention heads
#         self.d_ff = 3584    # Feed-forward dimension e.e. 16384

class ScaledLLaMa3_8BConfig(ModelConfig):
    def __init__(self, N=1024, N_pre=256, para_num=0.125, C_HBM_max=0.375):
        super().__init__(N, N_pre, para_num, C_HBM_max)
        # scale the model from 8B to 0.125B, 8:1 ratio, final latency need to multiply by 64 to reach original latency
        # prefill 256, decode 1024*10 for sparsity 90%\
        # 2KB for one KV cache, 64KB for a token
        self.L = 28          # Number of layers
        self.d = 512        # Hidden dimension e.g. 
        self.h = 4          # Number of attention heads
        self.d_ff = 2048    # Feed-forward dimension e.e. 16384

# class ScaledLLaMa3_8BConfig(ModelConfig):
#     def __init__(self, N=1024, N_pre=256, para_num=0.125, C_HBM_max=0.375):
#         super().__init__(N, N_pre, para_num, C_HBM_max)
#         # scale the model from 8B to 0.125B, 8:1 ratio, final latency need to multiply by 64 to reach original latency
#         # prefill 256, decode 1024*10 for sparsity 90%\
#         # 2KB for one KV cache, 64KB for a token
#         self.L = 32          # Number of layers
#         self.d = 128        # Hidden dimension e.g. 
#         self.h = 4          # Number of attention heads
#         self.d_ff = 512    # Feed-forward dimension e.e. 16384


# Records each token's KV caches store at where
class MemStatus(ABC):
    def __init__(self, config: ModelConfig, trace_reader, is_inclusive: bool):
        # self.trace = trace
        self.trace = trace_reader
        self.cfg = config
        
        self.total_model_weights: float =  self.cfg.para_num * self.cfg.dtype_size 
        self.start_token_id = self.cfg.N_pre
        # memory threshold rate
        self.threshold = 0.99
        self.inclusive = is_inclusive

        # Initialize hbm_tokens as a list of sets for each layer
        # Using sets for O(1) lookup time
        self.hbm_tokens = [set() for _ in range(self.cfg.L)]

        # Single KV cache size
        self.single_KV_cache_size = 2 * self.cfg.d * self.cfg.dtype_size
        # moodel weight read size per step
        self.model_weight_read_per_step = 4 * self.cfg.d**2 * self.cfg.dtype_size + (2 * self.cfg.d * self.cfg.d_ff * self.cfg.dtype_size) / 8
        # number of KV cache we can store in HBM
        self.max_KV_num = math.floor((self.cfg.C_HBM_max - self.total_model_weights) / self.single_KV_cache_size)
        self.current_hbm_kv_count = 0

        # other methods variables
        self.foresight_window = 0
        self.initialize_memory()
    
    def initialize_memory(self):
        """Initialize HBM with model parameters and KV cache."""
        self.initial_tokens_placement()
        print(f"Initialization complete.")

    
    # def print_token_layer_status(self):
    #     print("Final Layer distribution:")
    #     for token_id, layers in self.token_layer_status.items():
    #         print(f"Token {token_id}: {layers}")

    def is_token_in_hbm(self, token: int, layer: int) -> bool:
        """
        Check if a token's layer is in HBM
        Args:
            token: token number
            layer: layer number
        Returns:
            bool: True if token's layer is in HBM
        """
        return token in self.hbm_tokens[layer]
    
    def count_tokens_in_hbm(self, tokens: list, layer: int) -> int:
        """
        Count how many tokens from the given list are in HBM for a specific layer
        
        Args:
            tokens: list of token numbers to check
            layer: layer number to check against
            
        Returns:
            int: number of tokens from the list that are in HBM for the given layer
        """
        return sum(1 for token in tokens if self.is_token_in_hbm(token, layer))

    def remove_token_from_hbm(self, token: int, layer: int):
        """
        Remove a token's layer from HBM
        Args:
            token: token number
            layer: layer number
        """
        self.hbm_tokens[layer].discard(token)  # Using discard instead of remove to avoid KeyError
        self.current_hbm_kv_count -= 1
    
    
    def get_read_token_kv(self, n, l=0):
        read_tokens = self.trace.get_read_tokens(n, l)
        return read_tokens
    

    def add_token_to_hbm(self, token: int, layer: int):
        """Attempt to store data in HBM, return True if successful."""
        if self.is_token_in_hbm(token, layer):
            return True
        
        self.current_hbm_kv_count += 1
        remaining = self.max_KV_num - self.current_hbm_kv_count
        if remaining <= 0:
            self.current_hbm_kv_count -= 1
            return False
        else:
            self.hbm_tokens[layer].add(token)
            return True
    
    def calculate_data_sizes(self, n: int, l: int):
        """Calculate read/write data sizes for current step."""
        step_info = self.get_read_token_kv(n, l)

        # model weight read + read KV caches
        D_R = self.model_weight_read_per_step + len(step_info) * self.single_KV_cache_size
    
        D_W = self.single_KV_cache_size
            
        return D_R, D_W

    # return the maximum ratio that can be read from HBM
    def max_alpha(self, n: int, l: int):
        D_R, _ = self.calculate_data_sizes(n, l)
        if D_R <= 0:
            return 0.0
        
        # Model weight part: assume that the model weight component in D_R is 4*d^2*dtype_size.
        model_weight_component = self.model_weight_read_per_step

        # Retrieve current step's trace to get skip_token_kv list.
        step_info = self.get_read_token_kv(n, l)

        tk_on_hbm = sum(1 for token in step_info if self.is_token_in_hbm(token, l))

        # count_hbm should also include prefill tokens
        effective_KV_cache = tk_on_hbm * self.single_KV_cache_size
        
        max_alpha = (model_weight_component + effective_KV_cache) / D_R
            
        # Ensure alpha is within [0, 1]
        return max(0.0, min(max_alpha, 1.0))
    
    
    @abstractmethod
    def initial_tokens_placement(self):
        pass

# Firstly, records model weights and prefill KV cache in the HBM.
# IF the space is not enough, store in the external memory.
class HBMInit(MemStatus):
    def __init__(self, config, trace, is_inclusive):
        self.model_weight_ratio = 1.0
        super().__init__(config, trace, is_inclusive)
        

    def initial_tokens_placement(self):
        print(f"Start HBMInit initialization")
        # store prefill tokens on HBM until full
        for n in range (self.cfg.N_pre):
            for l in range(self.cfg.L):
                if self.add_token_to_hbm(n, l):
                    continue
                else:
                    print(f"HBM full, end HBMInit initialization")
                    return
        print(f"End HBMInit initialization")

class HBMInitPaged(MemStatus):
    def __init__(self, config, trace, is_inclusive):
        self.model_weight_ratio = 1.0
        self.page_size = 16  # Number of tokens per page
        super().__init__(config, trace, is_inclusive)
    
    def initial_tokens_placement(self):
        print(f"Start HBMInitPaged initialization")
        # Calculate number of complete pages
        num_pages = self.cfg.N_pre // self.page_size
        # Each token needs L KV caches
        kv_caches_needed = self.page_size * self.cfg.L
        
        # Process each complete page
        for page in range(num_pages):
            start_token = page * self.page_size
            end_token = start_token + self.page_size - 1
            
            # Check if we have enough space for the entire page
            if (self.current_hbm_kv_count + kv_caches_needed) > self.max_KV_num:
                print(f"HBM full at page {page}, end HBMInitPaged initialization")
                return
            
            # Add all tokens in the page with their L layers
            for token in range(start_token, end_token):
                for l in range(self.cfg.L):
                    if self.add_token_to_hbm(token, l):
                        continue
                    else:
                        print(f"HBM full, end HBMInit initialization")
                        return
    
        print(f"End HBMInitPaged initialization")


# Store best ratio of prefill tokens on HBM (token level)
class TokenLevelBestRatioInit(MemStatus):
    def __init__(self, config, trace, is_inclusive):
        self.future_window = 64
        self.model_weight_ratio = 1.0
        super().__init__(config, trace, is_inclusive)
        

    def initial_tokens_placement(self):
        print(f"Start TokenLevelInit initialization with {self.future_window} layer lookahead")
        # Scan future window to count token access frequency
        n_start = self.cfg.N_pre  # First decoding token

        for n in range(n_start, n_start + self.cfg.N):
            for l in range(self.cfg.L):
                read_tokens = self.get_read_token_kv(n, l)
                for token in read_tokens:
                    if token > self.cfg.N_pre:
                        continue
                    if not self.add_token_to_hbm(token, l):
                        print(f"HBM full, end TokenLevelInit initialization")
                        return
        print(f"End TokenLevelInit initialization")
