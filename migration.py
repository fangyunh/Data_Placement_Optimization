from abc import ABC, abstractmethod
from memory_status import ModelConfig, MemStatus
    
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
            step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
            skipped_tokens = step_info["skip_token_kv"]

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

                        if layer != l or token_id in skipped_tokens:
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
            step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
            skipped_tokens = step_info["skip_token_kv"]
            
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
        self.window_size = 50

    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0

        step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        skipped_tokens = step_info["skip_token_kv"]

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
                    if layer != l or token in skipped_tokens:
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
        step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        skipped_tokens = step_info["skip_token_kv"]
        
        next_key = (n + 1, l, s)
        if next_key not in self.status.trace:
            return [0.0, 0.0, 0.0, 0.0]   
        
        next_info = self.status.trace[next_key]
        next_skipped_tokens = next_info["skip_token_kv"]

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
                    if layer != l or token in skipped_tokens:
                        hbm_MR += layer_size
                    ext_MW += layer_size
        
        # PART 2: Migrate in layers for tokens that are not in the skip list.
        # We iterate over all tokens known in our token_layer_status.
        for token, layer_status in self.status.token_layer_status.items():
            if token in next_skipped_tokens:
                continue  # skip tokens already processed for migration out.
            self.initialize_token(token)
            # For each layer in this token:
            for layer in range(self.cfg.L):
                if self.status.get_layer_location(token, layer) == 1:
                    if self.status.store_data(layer_size):
                        # Migrate in: update layer status from 1 (External) to 0 (HBM).
                        self.status.update_token_layer(token, layer, 0)
                    
                        hbm_MW += layer_size
                        if layer != l or token in skipped_tokens:
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
            info = self.status.trace.get(key, {"skip_token_kv": []})
            batch_skip_lists.append(set(info["skip_token_kv"]))
        
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
        self.batch_size = 16

    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float, float, float]:
        layer_size = self.status.get_single_KV_cache_size()
        hbm_MR = 0.0 
        hbm_MW = 0.0
        ext_MR = 0.0
        ext_MW = 0.0
        
        step_info = self.status.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
        skipped_tokens = step_info["skip_token_kv"]
        deviation = 0.005
        next_n = n
        next_l = l + 1
        next_s = 0
        if next_l >= self.cfg.L:
            next_l = 0
            next_n = n + 1

        next_key = (next_n, next_l, next_s)
        if next_key not in self.status.trace:
            return [0.0, 0.0, 0.0, 0.0] 
        
        if s != 0:
            return [0.0, 0.0, 0.0, 0.0] 
        
        full = False

        current_alpha = self.status.max_and_best_alphas(next_n, next_l, next_s)
        while not full and abs(current_alpha[0] - self.cfg.best_alpha) > deviation:
            if current_alpha[0] > self.cfg.best_alpha:
                # Too high: we want to reduce HBM contribution.
                # Find one token (from those tracked) that currently has its next_l layer in HBM.
                adjusted = False
                for token in self.status.token_layer_status.keys():
                    if self.status.get_layer_location(token, next_l) == 0:
                        # Update layer status to external memory.
                        self.status.update_token_layer(token, next_l, 1)
                        # Decrease HBM occupancy.
                        self.cfg.C_HBM -= layer_size
                        hbm_MR += layer_size
                        ext_MW += layer_size
                        adjusted = True
                        break

                if not adjusted:
                    # No candidate found, mark full.
                    full = True
            else:
                # Too low: we want to increase HBM contribution.
                # Find one token that has its next_l layer in External memory.
                adjusted = False
                for token in self.status.token_layer_status.keys():
                    if self.status.get_layer_location(token, next_l) == 1:
                        if self.status.store_data(layer_size):
                            if token in skipped_tokens:
                                continue
                            
                            # Migrate in: update layer status from 1 (External) to 0 (HBM).
                            self.status.update_token_layer(token, next_l, 0)
                        
                            hbm_MW += layer_size
                            ext_MR += layer_size
                            adjusted = True
                            break
                        else:
                            full = True
                            break
                        
                    
                if not adjusted:
                    full = True
            # Recompute the alpha after adjustment.
            current_alpha = self.status.max_and_best_alphas(next_n, next_l, next_s)
            
        if self.status.inclusive:
            return [0.0, hbm_MW, ext_MR, 0.0]   
        
        return [hbm_MR, hbm_MW, ext_MR, ext_MW]
        
