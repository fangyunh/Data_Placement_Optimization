from abc import ABC, abstractmethod
from memory_status import ModelConfig, MemStatus

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
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        """Define migration sizes (D_MR, D_MW)."""
        pass
    

class NoMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
    
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        """No data migration"""
        return [0.0, 0.0, 0.0, 0.0]

# migrate previous tokens if reach the threshold
class PriorMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
    
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        # initialize return values
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0

        num_to_migrate = 32
        layer_size = self.status.get_single_KV_cache_size()  # 2 * d * dtype_size
        
        # Only perform migration if the HBM utilization rate exceeds the threshold.
        if self.status.exceed_threshold():
            tokens_in_hbm = []
            # step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
            # skipped_tokens = sorted(step_info["skip_token_kv"])
            skipped_tokens = sorted(self.status.get_skip_token_kv(n, l, s))

            # We assume that if a token has any layer with value 0, it is eligible.
            for token_id, layer_status in self.status.token_layer_status.items():
                if 0 in layer_status:
                    tokens_in_hbm.append(token_id)
                    if len(tokens_in_hbm) >= num_to_migrate:
                        break
                    continue
            tokens_in_hbm = sorted(tokens_in_hbm)
            
            # If fewer than 32 tokens are found, adjust the number.
            if len(tokens_in_hbm) < num_to_migrate:
                num_to_migrate = len(tokens_in_hbm)
            
            # Select the tokens to migrate (for example, the earliest tokens by ID).
            tokens_to_migrate = tokens_in_hbm[:num_to_migrate]
            
            # For each selected token, iterate over its layers.
            for token_id in tokens_to_migrate:
                # Ensure the token is initialized (should be by now).
                self.status.initialize_token(token_id)
                for layer in range(self.cfg.L):
                    # Only migrate layers currently in HBM (status 0)
                    if self.status.get_layer_location(token_id, layer) == 0:
                        # Update the status to 1 (external memory).
                        self.status.update_token_layer(token_id, layer, 1)
                        # Decrease the HBM occupancy in the placement strategy.
                        self.cfg.C_HBM -= layer_size

                        if layer != l or binary_search(skipped_tokens, token_id):
                            hbm_MR += layer_size
                        ext_MW += layer_size

            if self.status.inclusive:
                return [0.0, 0.0, 0.0, 0.0]      
            
            return [hbm_MR, hbm_MW, ext_MR, ext_MW]
        return [0.0, 0.0, 0.0, 0.0] 

# Migrate skipped tokens to external memory
class SkippedTokensMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
    
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0

        layer_size = self.status.get_single_KV_cache_size()  # per-layer KV cache size: 2 * d * dtype_size
        
        # Proceed only if the HBM utilization exceeds the threshold.
        if self.exceed_threshold():
            # print(f"Exceed threshold!")
            # Retrieve the trace for the current step.
            # step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
            # skipped_tokens = sorted(step_info["skip_token_kv"])
            skipped_tokens = sorted(self.status.get_skip_token_kv(n, l, s))
            
            # Iterate over each token in the skipped list.
            for token_id in skipped_tokens:
                self.status.initialize_token(token_id)
                # For each layer in the token:
                for layer in range(self.cfg.L):
                    # If the KV cache for this layer is in HBM (status 0), migrate it.
                    if self.status.get_layer_location(token_id, layer) == 0:
                        # Update layer status to external memory.
                        self.status.update_token_layer(token_id, layer, 1)
                        # Decrease the HBM occupancy in the placement strategy.
                        self.cfg.C_HBM -= layer_size

                        hbm_MR += layer_size
                        ext_MW += layer_size
            if self.status.inclusive:
                return [0.0, 0.0, 0.0, 0.0]  
            return [hbm_MR, hbm_MW, ext_MR, ext_MW]
        return [0.0, 0.0, 0.0, 0.0]

    
class PastWindowMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
        self.window_size = 16

    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0

        # step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        # skipped_tokens = sorted(step_info["skip_token_kv"])
        skipped_tokens = sorted(self.status.get_skip_token_kv(n, l, s))

        layer_size = self.status.get_single_KV_cache_size()  # per-layer size = 2 * d * dtype_size

        # Define the time window.
        start = max(0, n - self.window_size + 1)
        # Gather the union of all tokens marked as skipped in this window for the current (l, s).
        tokens_to_migrate = set()
        for idx in range(start, n + 1):
            key = (idx, l, s)
            info = self.status.trace.get(key)
            if info is not None:
                tokens_to_migrate.update(info.get("skip_token_kv", []))
        
        # For each token in the union, check each layer.
        for token in tokens_to_migrate:
            self.status.initialize_token(token)
            for layer in range(self.cfg.L):
                # If the KV cache at this layer is currently in HBM, migrate it.
                if self.status.get_layer_location(token, layer) == 0:
                    # Update layer status to external memory.
                    self.status.update_token_layer(token, layer, 1)
                    self.cfg.C_HBM -= layer_size       # Update HBM occupancy.
                    if layer != l or binary_search(skipped_tokens, token):
                        hbm_MR += layer_size
                    ext_MW += layer_size

        if self.status.inclusive:
            return [0.0, 0.0, 0.0, 0.0]      
        
        return [hbm_MR, hbm_MW, ext_MR, ext_MW]


class LookAheadMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
    
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0
        layer_size = self.status.get_single_KV_cache_size()
        # step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        # skipped_tokens = sorted(step_info["skip_token_kv"])
        skipped_tokens = sorted(self.status.get_skip_token_kv(n, l, s))
        
        next_key = (n + 1, l, s)
        if next_key not in self.status.trace:
            return [0.0, 0.0, 0.0, 0.0]   
        
        next_info = self.status.trace[next_key]
        next_skipped_tokens = sorted(next_info["skip_token_kv"])

        full = False
        
        # PART 1: Migrate out layers for tokens that token n+1 wants to skip.
        for token in next_skipped_tokens:
            self.initialize_token(token)
            # For each layer in this token:
            for layer in range(self.cfg.L):
                if self.status.get_layer_location(token, layer) == 0:
                    # Update layer status to external memory.
                    self.status.update_token_layer(token, layer, 1)
                    # Decrease HBM occupancy.
                    self.cfg.C_HBM -= layer_size
                    if layer != l or binary_search(skipped_tokens, token):
                        hbm_MR += layer_size
                    ext_MW += layer_size
        
        # PART 2: Migrate in layers for tokens that are not in the skip list.
        # We iterate over all tokens known in our token_layer_status.
        for token, layer_status in self.status.token_layer_status.items():
            if binary_search(next_skipped_tokens, token):
                continue  # skip tokens already processed for migration out.
            self.initialize_token(token)
            # For each layer in this token:
            for layer in range(self.cfg.L):
                if self.status.get_layer_location(token, layer) == 1:
                    if self.status.store_data(layer_size):
                        # Migrate in: update layer status from 1 (External) to 0 (HBM).
                        self.status.update_token_layer(token, layer, 0)
                    
                        hbm_MW += layer_size
                        if layer != l or binary_search(skipped_tokens, token):
                            ext_MR += layer_size
                    else:
                        full = True
                        break
            if full:
                break
        
        if self.status.inclusive:
            return [0.0, hbm_MW, ext_MR, 0.0]   
        
        return [hbm_MR, hbm_MW, ext_MR, ext_MW]


class LookAheadBatchMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
        self.batch_size = 16

    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0
        
        # step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        # skipped_tokens = step_info["skip_token_kv"]
        layer_size = self.status.get_single_KV_cache_size()

        # Gather skip lists from tokens n+1 to n+batch_size for the same (l, s)
        batch_skip_lists = []
        for j in range(1, self.batch_size + 1):
            key = (n + j, l, s)
            # info = self.status.trace.get(key, {"skip_token_kv": []})
            # batch_skip_lists.append(set(info["skip_token_kv"]))
            info = sorted(self.status.get_skip_token_kv(n, l, s))
            batch_skip_lists.append(set(info))
            
        
        if not batch_skip_lists:
            return [0.0, 0.0, 0.0, 0.0] 
        
        # Compute the intersection: tokens that every token in the batch wants to skip.
        consistent_skipped = set.intersection(*batch_skip_lists)
        
        # PART 1: Migrate out layers for tokens that are consistently skipped.
        for token in consistent_skipped:
            self.status.initialize_token(token)
            if self.status.get_layer_location(token, l) == 0:
                # Update layer status to external memory.
                self.status.update_token_layer(token, l, 1)
                # Decrease HBM occupancy.
                self.cfg.C_HBM -= layer_size
                
                hbm_MR += layer_size
                ext_MW += layer_size
        
        # PART 2: For tokens that are not consistently skipped, try to migrate in layers.
        # Iterate over all tokens we have tracked.
        for token, layer_status in self.status.token_layer_status.items():
            if token in consistent_skipped:
                continue  # Skip tokens already processed.
            self.status.initialize_token(token)
            # If a layer is in external memory (status 1), attempt to migrate it into HBM.
            if self.status.get_layer_location(token, l) == 1:
                if self.status.store_data(layer_size):
                    # Migrate in: update layer status from 1 (External) to 0 (HBM).
                    self.status.update_token_layer(token, l, 0)
                
                    hbm_MW += layer_size
                    
                    ext_MR += layer_size
                else:
                    break
                
        
        if self.status.inclusive:
            return [0.0, hbm_MW, ext_MR, 0.0]   
        
        return [hbm_MR, hbm_MW, ext_MR, ext_MW]

# Look ahead the n+1 token's alpha, try to maintain it at the best ratio.
class AlphaMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
    
    def move_out_unimportant_tokens(self, skip_tokens, layer) -> bool:
        for token in skip_tokens:
            if self.status.get_layer_location(token, layer) == 0:
                self.status.update_token_layer(token, layer, 1)
                self.cfg.C_HBM -= self.status.get_single_KV_cache_size()
                return True
        
        return False

    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        layer_size = self.status.get_single_KV_cache_size()
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0

        deviation = 0.01
        next_n = n + 1

        next_key = (next_n, l, s)
        if next_key not in self.status.trace:
            return [0.0, 0.0, 0.0, 0.0] 
        
        if s != 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        # Get current HBM count and skipped tokens
        current_tokens_on_hbm = self.status.hbm_token_counts[l]

        # step_info = self.status.trace.get((next_n, l, s), {"skip_token_kv": [], "skip_layer": False})
        # skipped_tokens = sorted(step_info["skip_token_kv"])
        # step_info_cur_l = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        # skipped_tokens_cu_l = sorted(step_info_cur_l["skip_token_kv"])

        skipped_tokens = sorted(self.status.get_skip_token_kv(next_n, l, s))
        skipped_tokens_cu_l = sorted(self.status.get_skip_token_kv(n, l, s))

        # Calculate effective HBM tokens for alpha
        skipped_in_hbm = sum(1 for token in skipped_tokens if self.status.get_layer_location(token, l) == 0)
        effective_tokens_on_hbm = current_tokens_on_hbm - skipped_in_hbm
        D_R, _ = self.status.calculate_data_sizes(next_n, l, s)
        model_weight = self.status.get_layer_md_weight_size() * self.status.model_weight_ratio

         # Adjust tokens to maintain best_alpha
        target = self.cfg.best_alpha * D_R - model_weight
        target_tokens = int(target / layer_size)
        delta = target_tokens - effective_tokens_on_hbm

        if delta < 0:
            if self.status.inclusive:
                return [0.0, 0.0, 0.0, 0.0]
            migrate_out = min(-delta, current_tokens_on_hbm)
            # Simple heuristic: migrate oldest tokens (implementation-specific)
            migrated = 0
            for token in sorted(self.status.token_layer_status.keys()):
                if migrated >= migrate_out:
                    break
                if binary_search(skipped_tokens_cu_l, token):
                    continue
                if self.status.get_layer_location(token, l) == 0:
                    self.status.update_token_layer(token, l, 1)
                    ext_MW += layer_size
                    self.cfg.C_HBM -= layer_size
                    migrated += 1
        elif delta > 0:
            migrated = 0
            for token in sorted(self.status.token_layer_status.keys(), reverse=True):  # Newest first
                if migrated >= delta:
                    break

                if (self.status.get_layer_location(token, l) == 1 and 
                    binary_search(skipped_tokens, token) == False):

                    if (self.status.store_data(layer_size)):
                        self.status.update_token_layer(token, l, 0)
                        hbm_MW += layer_size
                        migrated += 1
                    else:
                        if self.move_out_unimportant_tokens(skipped_tokens, l):
                            if (self.status.store_data(layer_size)):
                                self.status.update_token_layer(token, l, 0)
                                hbm_MW += layer_size
                                migrated += 1
                        else:
                            break


        if self.status.inclusive:
            return [0.0, hbm_MW, ext_MR, 0.0]   
        
        return [hbm_MR, hbm_MW, ext_MR, ext_MW]
    
        
