
import math
from abc import ABC, abstractmethod
from memory_status import ModelConfig, MemStatus
BYTES_TO_GB = 1024**3


class BaseStrategy(ABC):
    def __init__(self, config: ModelConfig, status: MemStatus):
        self.cfg = config
        self.status = status

    
    def alpha_strategy(self, n: int, l: int) -> float:
        alpha = self.status.max_alpha(n, l)
        if self.status.inclusive:
            return min(self.cfg.best_alpha, alpha)
        
        return alpha

    @abstractmethod
    def beta_strategy(self, n: int, l: int) -> float:
        """Define fraction of writes to HBM."""
        pass
        

class PreferHBM(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l):
        if self.status.add_token_to_hbm(n, l):
            return 1.0
        else:
            return 0.0
        
class PreferHBMPaged(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l):
        page_size = self.status.page_size
        # Calculate which page this token belongs to
        current_page = n // page_size
        page_start_token = current_page * page_size
        
        # If this is the first token of a page, check if we can store the whole page
        if n == page_start_token:
            # Calculate space needed for whole page
            kv_caches_needed = page_size * self.cfg.L
            remaining_space = self.status.max_KV_num - self.status.current_hbm_kv_count

            if remaining_space >= kv_caches_needed:
                # We can store the whole page
                return 1.0 if self.status.add_token_to_hbm(n, l) else 0.0
            else:
                # Not enough space for whole page
                return 0.0
        else:
            # For non-starting tokens, follow the same pattern as page start token
            if self.status.is_token_in_hbm(page_start_token, l):
                return 1.0 if self.status.add_token_to_hbm(n, l) else 0.0
            else:
                return 0.0

# The class tracks the previous layers distribution to decide this layer
# writes to the HBM or not.
class LookAheadOnePlacement(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l):
        
        tokens_for_next_read = self.status.get_read_token_kv(n + 1, l)
            
        if n in tokens_for_next_read:
            # If the token is in the next read set, we place it in HBM
            if self.status.add_token_to_hbm(n, l):
                return 1.0
            else:
                for old_token in range(n):
                    if (self.status.is_token_in_hbm(old_token, l) and 
                        old_token not in tokens_for_next_read):
                        self.status.remove_token_from_hbm(old_token, l)
                        if self.status.add_token_to_hbm(n, l):
                            return 1.0
            
                return 0.0
                        
        else:
            return 0.0
        

            

        