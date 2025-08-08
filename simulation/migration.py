from abc import ABC, abstractmethod
from memory_status import ModelConfig, MemStatus
from collections import Counter, deque
from math import ceil 

# Add this to your migration.py or a utility file
def binary_search(sorted_list, target):
    """
    Check if `target` exists in `sorted_list` using binary search.
    Returns True if found, False otherwise.
    """
    left = 0
    right = len(sorted_list) - 1
    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] == target:
            return True
        elif sorted_list[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False
    
class BaseDataMigration(ABC):
    def __init__(self, config: ModelConfig, status: MemStatus):
        # Maintain sets of token IDs stored in HBM and external memory.
        # Initially, you might decide that all tokens start in external memory.
        self.cfg = config
        self.status = status
    
    # Returen [hbm_MR, hbm_MW, ext_MR, ext_MW]
    @abstractmethod
    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        """Define migration sizes (D_MR, D_MW)."""
        pass
    

class NoMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
    
    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        """No data migration"""
        return [0.0, 0.0, 0.0, 0.0]

# Look ahead the n+1 token's alpha, try to maintain it at the best ratio.
class LookAheadOneMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)

    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        layer_size = self.status.single_KV_cache_size
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0

        next_n = n + 1
    
        if next_n >= (self.cfg.N_pre + self.cfg.N):
            return [0.0, 0.0, 0.0, 0.0]    
        
        # Collect current and next read information
        read_tokens_cur = self.status.get_read_token_kv(n, l)
        read_tokens_next = self.status.get_read_token_kv(next_n, l)
        tokens_in_hbm_count = self.status.count_tokens_in_hbm(read_tokens_next, l)

        D_R, _ = self.status.calculate_data_sizes(next_n, l)
        model_weight = self.status.model_weight_read_per_step * self.status.model_weight_ratio

         # Adjust tokens to maintain best_alpha
        target = self.cfg.best_alpha * D_R - model_weight
        target_tokens = int(target / layer_size)
        delta = target_tokens - tokens_in_hbm_count

        if delta > 0:
            migrate_out = min(delta, tokens_in_hbm_count)
            # Simple heuristic: migrate oldest tokens (implementation-specific)
            migrated = 0

            # collect tokens that can be migrated out
            remove_tokens = deque()
            tokens_to_add = []
            for token in read_tokens_cur:
                # remove token later
                if token not in read_tokens_next and self.status.is_token_in_hbm(token, l):
                    remove_tokens.append(token)
                
                if token in read_tokens_next and not self.status.is_token_in_hbm(token, l):
                    tokens_to_add.append(token)

            for token in tokens_to_add:
                if migrated >= migrate_out:
                    break

                # if HBM is full, evict tokens
                while not self.status.add_token_to_hbm(token, l):
                    if remove_tokens:
                        self.status.remove_token_from_hbm(remove_tokens.popleft(), l)
                    else:
                        break

                hbm_MW += layer_size
                migrated += 1
                    

        if self.status.inclusive:
            return [0.0, hbm_MW, ext_MR, 0.0]   
        
        return [hbm_MR, hbm_MW, ext_MR, ext_MW]

class PageMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)

    def transfer_page_to_hbm(self, n:int)->bool:
        """
        Check if the page containing token n is fully in HBM.
        """
        page_size = self.status.page_size
        page_start = (n // page_size) * page_size
        page_end = page_start + page_size - 1
    
        # Calculate KV caches needed for entire page
        kv_caches_needed = page_size * self.cfg.L
        
        while True:
            # Check if we have enough space for the entire page
            remaining_space = self.status.max_KV_num - self.status.current_hbm_kv_count
            
            if remaining_space >= kv_caches_needed:
                # Keep track of successfully added tokens for rollback
                added_tokens = []
                
                # Try to transfer all tokens in the page
                for t in range(page_start, page_end + 1):
                    for layer in range(self.cfg.L):
                        if not self.status.is_token_in_hbm(t, layer):
                            success = self.status.add_token_to_hbm(t, layer)
                            if not success:
                                # Rollback: remove all previously added tokens
                                for rollback_t, rollback_l in added_tokens:
                                    self.status.remove_token_from_hbm(rollback_t, rollback_l)
                                return False  # Not enough space for the entire page
                        added_tokens.append((t, layer))
                return True  # Successfully added the entire page
            else:
                # Try to remove oldest page to make space
                evict_success = False
                for old_page in range(n // page_size):
                    old_page_start = old_page * page_size
                    old_page_end = old_page_start + page_size - 1
                    
                    # Check if any token from this page is in HBM
                    page_found = False
                    if self.status.is_token_in_hbm(old_page_start, 0):
                        page_found = True
                    
                    if page_found:
                        # Remove entire page
                        for t in range(old_page_start, old_page_end + 1):
                            for l in range(self.cfg.L):
                                if self.status.is_token_in_hbm(t, l):
                                    self.status.remove_token_from_hbm(t, l)
                        evict_success = True
                        break
                    else:
                        # No tokens in this page, continue to next page
                        continue
                if not evict_success:
                    return False  # No pages to evict, cannot transfer the page

    
    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        layer_size = self.status.single_KV_cache_size
        page_size = self.status.page_size
        # data amount per page
        page_amount = page_size * self.cfg.L * layer_size
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0

        next_n = n + 1
    
        if next_n >= (self.cfg.N_pre + self.cfg.N):
            return [0.0, 0.0, 0.0, 0.0]

        # Collect current and next read information
        read_tokens_cur = self.status.get_read_token_kv(n, l)
        read_tokens_next = self.status.get_read_token_kv(next_n, l)
        for token in read_tokens_next:
            # Check if the token belongs to a page that can be fully transferred
            if self.status.is_token_in_hbm(token, l):
                continue
            else:
                if self.transfer_page_to_hbm(token):
                    if token in read_tokens_cur:
                        ext_MR += page_amount
                    hbm_MW += page_amount

        if self.status.inclusive:
            return [0.0, hbm_MW, ext_MR, 0.0]
            
        return [hbm_MR, hbm_MW, ext_MR, ext_MW]

            
class NormalMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
        self.last_removed_token = 0

    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        layer_size = self.status.single_KV_cache_size
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0

        # Convert to set for O(1) lookups
        read_tokens_cur = self.status.get_read_token_kv(n, l)
        
        # Find tokens that need to be added to HBM
        tokens_to_add = [t for t in read_tokens_cur if not self.status.is_token_in_hbm(t, l)]
        
        # Early return if no tokens need to be added
        if not tokens_to_add:
            return [0.0, 0.0, 0.0, 0.0]

        
        # Try to add each token
        for token in tokens_to_add:
            while not self.status.add_token_to_hbm(token, l):
                for old_token in range(n):
                    if self.status.is_token_in_hbm(old_token, l):
                        # Remove this token to make space
                        self.status.remove_token_from_hbm(old_token, l)

            hbm_MW += layer_size         

        if self.status.inclusive:
            return [0.0, hbm_MW, ext_MR, 0.0]
            
        return [hbm_MR, hbm_MW, ext_MR, ext_MW]

def calculate_token_score(offsets: set) -> float:
    """Calculate token score based on offsets and current read status"""
    score = sum(1.0/offset for offset in offsets)
    return score
    
class SAMigration(BaseDataMigration):
    def __init__(self, config, status, read_window_size=5, topk_ratio=1.0):
        super().__init__(config, status)
        self.window_read = read_window_size
        self.top_k_ratio = topk_ratio
        # self.frequency_priority = freq_prior
    
    def read_migration(self, n, l, mig_queue, no_remove, top_k) -> tuple[float, float, float, float]:
        layer_size = self.status.single_KV_cache_size
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0

        migrated = 0
        # Convert to set for O(1) lookups
        # Get current read tokens
        while mig_queue and migrated < top_k:
            token = mig_queue.popleft()

            add_success = True
            # If HBM has space, directly add the token
            while not self.status.add_token_to_hbm(token, l):
                removed = False
                # If HBM is full, remove tokens from no_remove
                for token_to_remove in self.status.hbm_tokens[l]:
                    if token_to_remove not in no_remove:
                        self.status.remove_token_from_hbm(token_to_remove, l)
                        removed = True
                        break
                
                if not removed:
                    add_success = False
                    break

            if add_success:
                hbm_MW += layer_size      
                migrated += 1
            
        return [hbm_MR, hbm_MW, ext_MR, ext_MW]

    def collect_future_tokens(self, n: int, l: int, read_tokens_cur: set) -> tuple[deque, set]:
        """
        Efficiently collect and prioritize future tokens based on their read offsets.
        Returns (priority_queue, remove_queue).
        
        Logic:
        1. Collect tokens that will be read in future window and their offsets
        2. Track all future read tokens (both in HBM and not) for remove queue generation
        3. Only generate remove queue if we have tokens to migrate
        """
        # Initialize data structures
        token_to_offsets = {}  # {token: set(offsets)} - for tokens that can be migrated
        all_future_reads = set()  # Track all tokens that will be read (for remove queue)

        # Single pass to collect all future tokens and their offsets
        for offset in range(1, self.window_read + 1):
            next_n = n + offset
            if next_n >= (self.cfg.N_pre + self.cfg.N):
                break
                
            # Get all tokens that will be read in this step
            step_tokens = self.status.get_read_token_kv(next_n, l)
            
            # Update all future reads set (for tokens <= n)
            # valid_future_reads = {t for t in step_tokens if t <= n and t in read_tokens_cur}
            valid_tokens = set()
            valid_future_reads = set()
            for t in step_tokens:
                if t <= n:
                    valid_tokens.add(t)
                    if t in read_tokens_cur:
                        valid_future_reads.add(t)
            
            all_future_reads.update(valid_tokens)
            
            # Record offsets only for tokens that can be migrated
            valid_migration_tokens = [t for t in valid_future_reads 
                                    if not self.status.is_token_in_hbm(t, l)]
            
            for token in valid_migration_tokens:
                if token not in token_to_offsets:
                    token_to_offsets[token] = {offset}
                else:
                    token_to_offsets[token].add(offset)

        # If no tokens to migrate, return empty queues
        if not token_to_offsets:
            return deque(), set()

        # Calculate scores and create priority queue
        scored_tokens = [
            (token, calculate_token_score(offsets))
            for token, offsets in token_to_offsets.items()
        ]
        scored_tokens.sort(key=lambda x: (-x[1], x[0]))  # Sort by score desc, token ID asc
        priority_tokens = deque(token for token, _ in scored_tokens)

        # Build remove queue - only for tokens in HBM that won't be read in future window
        # remove_tokens = [
        #     token for token in self.status.hbm_tokens[l]
        #     if token not in all_future_reads  # Check against all future reads
        # ]
        # remove_tokens.sort()  # Sort by token ID
        
        return priority_tokens, all_future_reads

    def top_k_in_read(self, priority_tokens: deque, read_tokens_cur: set) -> int:
        return sum(1 for token in priority_tokens if token in read_tokens_cur)

    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        mig_overhead = [0.0, 0.0, 0.0, 0.0]
        
        # Read tokens once
        read_tokens_cur = set(self.status.get_read_token_kv(n, l))
        priority_tokens, no_remove_tokens = self.collect_future_tokens(n, l, read_tokens_cur)
        # priority_tokens, remove_queue = self.collect_future_tokens_cur_read(n, l, s)

        if not priority_tokens:
            return mig_overhead
        
        # number of tokens to migrate
        top_k = ceil(self.top_k_ratio * len(priority_tokens))

        # log:
        if n == self.cfg.N_pre + self.cfg.N - self.window_read:
            print(f"top_k: {top_k}\n")
            print(f"Migration tokens len: {len(priority_tokens)}\n")
            print(f"Count in read: {self.top_k_in_read(priority_tokens, read_tokens_cur)}\n")

        return self.read_migration(n, l, priority_tokens, no_remove_tokens, top_k)










        
        
        

        