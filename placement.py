from dataclasses import dataclass
import numpy as np
import csv
import re
import math
from abc import ABC, abstractmethod
from migration import BaseDataMigration
BYTES_TO_GB = 1024**3

@dataclass
class ModelConfig:
    # Model architecture parameters
    # llama 3.3
    L: int = 32          # Number of layers
    d: int = 4096        # Hidden dimension
    h: int = 32          # Number of attention heads
    d_ff: int = 16384    # Feed-forward dimension
    dtype_size: int = 2  # Bytes per parameter (e.g., 2 for FP16)
    para_num: int = 75000000000 # how many parameters in the model
    
    # Memory parameters
    B_HBM: float = 4915  # HBM bandwidth in GB/s  e.g 4.8TB/s  B/ns
    B_ext_interface_R: float = 900  # External memory interface read (GB/s) B/ns
    B_ext_interface_W: float = 900  # External memory interface write (GB/s) B/ns
    B_ext_internal: float = 1900    # External memory internal bandwidth (GB/s) B/ns
    C_HBM_max: float = 141 * BYTES_TO_GB          # HBM capacity in B
    C_HBM: float = 0.0
    # Inference parameters
    N: int = 4096         # Total tokens
    N_pre: int = 30000     # Previous tokens from prefilling
    model_weight_ratio = 0.84

class BaseStrategy(ABC):
    def __init__(self, config: ModelConfig, migration: BaseDataMigration):
        self.cfg = config
        self.mig = migration
        
        
        self.initialize_memory()

    def initialize_memory(self):
        """Initialize HBM with model parameters and KV cache."""
        self.cfg.C_HBM = 0.0
        model_size = self.cfg.para_num * self.cfg.dtype_size * self.cfg.model_weight_ratio
        self.store_data(model_size)
        prev_kv_cache_size = 2 * self.cfg.d * self.cfg.dtype_size * self.cfg.N_pre * self.cfg.model_weight_ratio
        self.store_data(prev_kv_cache_size)
        print(f"HBM initial utilizaiton rate: {self.cfg.C_HBM/self.cfg.C_HBM_max * 100}%.")

    def store_data(self, data_size):
        """Attempt to store data in HBM, return True if successful."""
        remaining = self.cfg.C_HBM_max - self.cfg.C_HBM
        if remaining <= 0:
            return False
        if data_size <= remaining:
            self.cfg.C_HBM += data_size
            return True
        self.cfg.C_HBM = self.cfg.C_HBM_max
        return False

    def calculate_data_sizes(self, n: int, l: int, s: int):
        """Calculate read/write data sizes for current step."""
        step_info = self.mig.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        # If this layer is skipped in the trace, no data is processed.
        if step_info["skip_layer"]:
            return 0, 0
        skip_tokens = set(step_info["skip_token_kv"])

        if s == 0:  # MHA
            D_R = (4 * self.cfg.d**2 + 2 * (n + self.cfg.N_pre) * self.cfg.d) * self.cfg.dtype_size
            D_W = 2 * self.cfg.d * self.cfg.dtype_size
            count = 0
            for token_id, layer_status in self.mig.token_layer_status.items():
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

    def alpha_strategy(self, n: int, l: int, s: int) -> float:
        if s == 1:
            return self.cfg.model_weight_ratio
        
        D_R, _ = self.calculate_data_sizes(n, l, s)
        if D_R <= 0:
            return 1.0
        
        # Model weight part: assume that the model weight component in D_R is 4*d^2*dtype_size.
        model_weight_component = 4 * self.cfg.d**2 * self.cfg.dtype_size
        effective_model_weight = self.cfg.model_weight_ratio * model_weight_component

        # Retrieve current step's trace to get skip_token_kv list.
        step_info = self.mig.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        skip_tokens = set(step_info["skip_token_kv"])

        # Count over all tokens that have been processed so far (or that are tracked)
        count_hbm = 0
        for token_id, layer_status in self.mig.token_layer_status.items():
            # Only consider tokens that are NOT skipped for the current step.
            if token_id in skip_tokens:
                continue
            if layer_status[l] == 0:
                count_hbm += 1

        effective_KV_cache = (count_hbm + self.cfg.N_pre) * (2 * self.cfg.d * self.cfg.dtype_size)
        
        alpha = (effective_model_weight + effective_KV_cache) / D_R
        # Ensure alpha is within [0, 1]
        return max(0.0, min(alpha, 1.0))

    @abstractmethod
    def beta_strategy(self, n: int, l: int, s: int) -> float:
        """Define fraction of writes to HBM."""
        pass

class PreferHBM(BaseStrategy):
    def __init__(self, config: ModelConfig, migration: BaseDataMigration):
        super().__init__(config, migration)

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        
        step_info = self.mig.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        if step_info['skip_layer']:
            self.mig.update_token_layer(n, l, 2)
            return 0.0
        
        _, D_W = self.calculate_data_sizes(n, l, s)
        if (self.store_data(D_W)):
            self.mig.update_token_layer(n, l, 0)
            return 1.0
        else:
            self.mig.update_token_layer(n, l, 1)
            return 0.0

# prior layers of a token to HBM and later layers to the external memory
class SplitToken(BaseStrategy):
    def __init__(self, config: ModelConfig, migration: BaseDataMigration):
        super().__init__(config, migration)
        self.ratio = self.cfg.B_HBM / (self.cfg.B_HBM + min(self.cfg.B_ext_interface_R, self.cfg.B_ext_internal))

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        
        step_info = self.mig.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        if step_info['skip_layer']:
            self.mig.update_token_layer(n, l, 2)
            return 0.0
        
        _, D_W = self.calculate_data_sizes(n, l, s)
        
        layer = math.floor(self.cfg.L * self.ratio)
        if (l <= layer):
            if (self.store_data(D_W)):
                self.mig.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.mig.update_token_layer(n, l, 1)
                return 0.0
        else:
            self.mig.update_token_layer(n, l, 1)
            return 0.0

# According to a ratio, store some l-th layers on HBM and some l-th layers on the external memory
class BatchRatio(BaseStrategy):
    def __init__(self, config: ModelConfig, migration: BaseDataMigration):
        super().__init__(config, migration)
        self.ratio = self.cfg.B_HBM / (self.cfg.B_HBM + min(self.cfg.B_ext_interface_R, self.cfg.B_ext_internal))


    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        
        batch_num = 16
        step_info = self.mig.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        if step_info['skip_layer']:
            self.mig.update_token_layer(n, l, 2)
            return 0.0
        
        batch = math.floor(batch_num * self.ratio)
        _, D_W = self.calculate_data_sizes(n, l, s)

        if (n % batch_num <= batch):
            if (self.store_data(D_W)):
                self.mig.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.mig.update_token_layer(n, l, 1)
                return 0.0
        else:
            self.mig.update_token_layer(n, l, 1)
            return 0.0

# look ahead to see if the token or layer is skipped or not.
class LookAheadBatch(BaseStrategy):
    def __init__(self, config: ModelConfig, migration: BaseDataMigration):
        super().__init__(config, migration)
        self.ratio = self.cfg.B_HBM / (self.cfg.B_HBM + min(self.cfg.B_ext_interface_R, self.cfg.B_ext_internal))

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        
        step_info = self.mig.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        if step_info['skip_layer']:
            self.mig.update_token_layer(n, l, 2)
            return 0.0
        
        batch_size = 16
        # Gather skip lists from tokens n+1 to n+batch_size for the same (l, s)
        batch_skip_lists = []
        for j in range(1, batch_size + 1):
            key = (n + j, l, s)
            info = self.mig.trace.get(key, {"skip_token_kv": [], "skip_layer": False})
            batch_skip_lists.append(set(info["skip_token_kv"]))

        _, D_W = self.calculate_data_sizes(n, l, s)

        if n in batch_skip_lists:
            self.mig.update_token_layer(n, l, 1)
            return 0.0
        else:
            if (self.store_data(D_W)):
                self.mig.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.mig.update_token_layer(n, l, 1)
                return 0.0
            
# Consider each layer's importance. If a layer was skipped frequentely we place it
# at the external memory.
class LayerImportance(BaseStrategy):
    def __init__(self, config: ModelConfig, migration: BaseDataMigration):
        super().__init__(config, migration)
        self.ratio = self.cfg.B_HBM / (self.cfg.B_HBM + min(self.cfg.B_ext_interface_R, self.cfg.B_ext_internal))

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        limit = 10
        count = 0
        for token_id, layer_status in self.mig.token_layer_status.items():
            if layer_status[l] == 2:
                count += 1

        _, D_W = self.calculate_data_sizes(n, l, s)

        if count > limit:
            self.mig.update_token_layer(n, l, 1)
            return 0.0
        else:
            if (self.store_data(D_W)):
                self.mig.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.mig.update_token_layer(n, l, 1)
                return 0.0

# The class tracks the previous layers distribution to decide this layer
# writes to the HBM or not.
class AlphaLayersDistribution(BaseStrategy):
    def __init__(self, config: ModelConfig, migration: BaseDataMigration):
        super().__init__(config, migration)
        self.ratio = self.cfg.B_HBM / (self.cfg.B_HBM + min(self.cfg.B_ext_interface_R, self.cfg.B_ext_internal))

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        count = 0
        for token_id, layer_status in self.mig.token_layer_status.items():
            if layer_status[l] == 0:
                count += 1

        _, D_W = self.calculate_data_sizes(n, l, s)
        if (n == 0):
            if (self.store_data(D_W)):
                self.mig.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.mig.update_token_layer(n, l, 1)
                return 0.0
            
        if ((count / n) < self.ratio):
            if (self.store_data(D_W)):
                self.mig.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.mig.update_token_layer(n, l, 1)
                return 0.0
        else:
            self.mig.update_token_layer(n, l, 1)
            return 0.0