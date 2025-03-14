import numpy as np
import math
from abc import ABC, abstractmethod
from memory_status import ModelConfig, MemStatus
BYTES_TO_GB = 1024**3


class BaseStrategy(ABC):
    def __init__(self, config: ModelConfig, status: MemStatus):
        self.cfg = config
        self.status = status

    
    def alpha_strategy(self, n: int, l: int, s: int) -> float:
        alphas = self.status.max_and_best_alphas(n, l, s)
        if self.status.inclusive:
            return alphas[1]
        
        return alphas[0]

    @abstractmethod
    def beta_strategy(self, n: int, l: int, s: int) -> float:
        """Define fraction of writes to HBM."""
        pass
        

class PreferHBM(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        
        step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        if step_info['skip_layer']:
            self.status.update_token_layer(n, l, 2)
            return 0.0
        
        _, D_W = self.status.calculate_data_sizes(n, l, s)
        if (self.status.store_data(D_W)):
            self.status.update_token_layer(n, l, 0)
            return 1.0
        else:
            self.status.update_token_layer(n, l, 1)
            return 0.0

# prior layers of a token to HBM and later layers to the external memory
class SplitToken(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        
        step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        if step_info['skip_layer']:
            self.status.update_token_layer(n, l, 2)
            return 0.0
        
        _, D_W = self.status.calculate_data_sizes(n, l, s)
        
        layer = math.floor(self.cfg.L * self.cfg.best_alpha)
        if (l <= layer):
            if (self.status.store_data(D_W)):
                self.status.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.status.update_token_layer(n, l, 1)
                return 0.0
        else:
            self.status.update_token_layer(n, l, 1)
            return 0.0

# According to a ratio, store some l-th layers on HBM and some l-th layers on the external memory
class BatchRatio(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        
        batch_num = 16
        step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        if step_info['skip_layer']:
            self.status.update_token_layer(n, l, 2)
            return 0.0
        
        batch = math.floor(batch_num * self.cfg.best_alpha)
        _, D_W = self.status.calculate_data_sizes(n, l, s)

        if (n % batch_num <= batch):
            if (self.status.store_data(D_W)):
                self.status.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.status.update_token_layer(n, l, 1)
                return 0.0
        else:
            self.status.update_token_layer(n, l, 1)
            return 0.0

# look ahead to see if the token or layer is skipped or not.
class LookAheadBatch(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        
        step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        if step_info['skip_layer']:
            self.status.update_token_layer(n, l, 2)
            return 0.0
        
        batch_size = 16
        # Gather skip lists from tokens n+1 to n+batch_size for the same (l, s)
        batch_skip_lists = []
        for j in range(1, batch_size + 1):
            key = (n + j, l, s)
            info = self.status.trace.get(key, {"skip_token_kv": [], "skip_layer": False})
            batch_skip_lists.append(set(info["skip_token_kv"]))

        _, D_W = self.status.calculate_data_sizes(n, l, s)

        if n in batch_skip_lists:
            self.status.update_token_layer(n, l, 1)
            return 0.0
        else:
            if (self.status.store_data(D_W)):
                self.status.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.status.update_token_layer(n, l, 1)
                return 0.0
            
# Consider each layer's importance. If a layer was skipped frequentely we place it
# at the external memory.
class LayerImportance(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        limit = 20
        count = 0
        for token_id, layer_status in self.status.token_layer_status.items():
            if layer_status[l] == 2:
                count += 1

        _, D_W = self.status.calculate_data_sizes(n, l, s)

        if count > limit:
            self.status.update_token_layer(n, l, 1)
            return 0.0
        else:
            if (self.status.store_data(D_W)):
                self.status.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.status.update_token_layer(n, l, 1)
                return 0.0

# The class tracks the previous layers distribution to decide this layer
# writes to the HBM or not.
class AlphaLayersDistribution(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l, s):
        if s == 1:
            return 0.0
        count = 0
        for token_id, layer_status in self.status.token_layer_status.items():
            if layer_status[l] == 0:
                count += 1

        _, D_W = self.status.calculate_data_sizes(n, l, s)
            
        if ((count / n) < self.cfg.best_alpha):
            if (self.status.store_data(D_W)):
                self.status.update_token_layer(n, l, 0)
                return 1.0
            else:
                self.status.update_token_layer(n, l, 1)
                return 0.0
        else:
            self.status.update_token_layer(n, l, 1)
            return 0.0