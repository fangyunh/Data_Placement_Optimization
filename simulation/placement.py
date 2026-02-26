
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
    """
    A simple placement strategy:
    Always try to write the new KV cache for ALL queries in the
    batch to HBM.
    """
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l):
        """
        Tries to add the KV cache unit (q, n, l) for all queries q.
        Returns the fraction of queries that were successfully added.
        """
        if self.cfg.batch_size == 0:
            return 0.0
            
        added_count = 0
        for q in range(self.cfg.batch_size):
            # Try to add the new KV unit for query q, token n, layer l
            if self.status.add_kv_to_hbm(q, n, l):
                added_count += 1
        
        # Beta is the fraction of D_W (batch_size * 1 KV unit) written to HBM
        return added_count / self.cfg.batch_size
        
class PreferHBMPaged(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)
        self.page_size = getattr(self.status, 'page_size', 16)

    def beta_strategy(self, n, l):
        if self.cfg.batch_size == 0:
            return 0.0
            
        current_page = n // self.page_size
        page_start_token = current_page * self.page_size
        
        added_count = 0
        
        if n == page_start_token:
            # First token of a new page. Make a policy decision.
            # Check for space for the whole page *at this layer* for *all queries*.
            space_needed = self.page_size * self.cfg.batch_size * self.status.single_KV_cache_size
            
            if self.status.hbm_capacity_remaining >= space_needed:
                # We can store the whole page. Store the first token batch.
                for q in range(self.cfg.batch_size):
                    if self.status.add_kv_to_hbm(q, n, l):
                        added_count += 1
            else:
                # Not enough space for the page. Store nothing in HBM.
                return 0.0
        else:
            # Not the first token. Check if the start of this page was stored
            # for query 0 (as a proxy for the page).
            if self.status.is_kv_in_hbm(query=0, token=page_start_token, layer=l):
                # Page was approved. Try to store this token's batch.
                for q in range(self.cfg.batch_size):
                    if self.status.add_kv_to_hbm(q, n, l):
                        added_count += 1
            else:
                # Page was not approved. Store nothing in HBM.
                return 0.0

        return added_count / self.cfg.batch_size

# The class tracks the previous layers distribution to decide this layer
# writes to the HBM or not.
class LookAheadOnePlacement(BaseStrategy):
    def __init__(self, config: ModelConfig, status: MemStatus):
        super().__init__(config, status)

    def beta_strategy(self, n, l):
        if self.cfg.batch_size == 0:
            return 0.0

        # 1. Look ahead to step (n+1, l)
        try:
            kv_reads_all_queries = self.status.trace.get_all_kv_reads_by_query(n + 1, l)
        except (IOError, IndexError, KeyError):
            # Handle end of trace or trace errors by falling back to PreferHBM
            added_count = 0
            for q in range(self.cfg.batch_size):
                if self.status.add_kv_to_hbm(q, n, l):
                    added_count += 1
            return added_count / self.cfg.batch_size if self.cfg.batch_size > 0 else 0.0

        # 2. Find all unique tokens read next step and check if 'n' is among them
        all_tokens_read_next = set()
        is_token_n_read_next = False
        for token_set in kv_reads_all_queries.values():
            all_tokens_read_next.update(token_set)
            if n in token_set:
                is_token_n_read_next = True

        added_count = 0
        if not is_token_n_read_next:
            # Token 'n' is not immediately needed. Write to external.
            return 0.0

        # 3. Token 'n' *is* read next. Try to add to HBM.
        # Keep track of which queries failed to add.
        failed_queries = []
        for q in range(self.cfg.batch_size):
            if self.status.add_kv_to_hbm(q, n, l):
                added_count += 1
            else:
                failed_queries.append(q)

        if not failed_queries:
            # Successfully added all, we are done.
            return 1.0

        # 4. HBM is full. Try to evict useless tokens to make space.
        # Iterate over a copy of the HBM KV set for this layer
        kv_in_hbm_at_l = list(self.status.hbm_kv_tokens[l])
        
        for (evict_q, evict_t) in kv_in_hbm_at_l:
            if evict_t not in all_tokens_read_next:
                # This KV unit (evict_q, evict_t) is in HBM but
                # its token 'evict_t' is NOT needed next step. Evict it.
                self.status.remove_kv_from_hbm(evict_q, evict_t, l)
                
                # Try again to add the failed queries
                remaining_failed = []
                for q in failed_queries:
                    if self.status.add_kv_to_hbm(q, n, l):
                        added_count += 1
                    else:
                        remaining_failed.append(q)
                
                failed_queries = remaining_failed
                if not failed_queries:
                    # Made enough space
                    break 
        
        # Return the final fraction that was successfully added
        return added_count / self.cfg.batch_size
        

            

        