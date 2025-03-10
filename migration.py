from dataclasses import dataclass
import re
import math
from abc import ABC, abstractmethod

def load_trace(filename="trace.txt"):
    trace = {}
    pattern = re.compile(r"^([^,]+),([^,]+),([^,]+),(\[.*?\]),(.+)$")

    with open(filename, "r") as f:
        next(f)  # Skip header if there is one
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if m:
                n, l, s, skip_token_kv_str, skip_layer_str = m.groups()
                n, l, s = int(n), int(l), int(s)
                skip_token_kv = eval(skip_token_kv_str)  # Note: using eval can be dangerous if the file is untrusted
                skip_layer = skip_layer_str.strip().lower() == "true"
                trace[(n, l, s)] = {"skip_token_kv": skip_token_kv, "skip_layer": skip_layer}
            else:
                raise ValueError("Line doesn't match expected format: " + line)
    return trace
    
class BaseDataMigration(ABC):
    def __init__(self, config, placement):
        # Maintain sets of token IDs stored in HBM and external memory.
        # Initially, you might decide that all tokens start in external memory.
        self.cfg = config
        self.plc = placement
        self.threshold = 1.0
        self.trace = load_trace()
        self.token_layer_status = {}
    
    @abstractmethod
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float]:
        """Define migration sizes (D_MR, D_MW)."""
        pass

    def initialize_token(self, token_id):
        """Ensure that a token has been initialized in token_layer_status."""
        if token_id not in self.token_layer_status:
            # Default: all layers set to 1 (external memory)
            self.token_layer_status[token_id] = [2] * self.cfg.L
    
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

    def get_layer_size(self) -> int:
        return 2 * self.cfg.d * self.cfg.dtype_size
    
    def get_HBM_util_rate(self) -> float:
        return self.cfg.C_HBM / self.cfg.C_HBM_max
    
    def exceed_threshold(self) -> bool:
        flag = self.get_HBM_util_rate() >= self.threshold
        return flag
    
    def is_token_in_hbm(self, token_id, layer) -> bool:
        self.initialize_token(token_id)
        # For our purposes, we say a token is in HBM if at least one layer is stored there.
        return self.token_layer_status[token_id][layer] == 0
    

class NoMigration(BaseDataMigration):
    def __init__(self, config, placement):
        super().__init__(config, placement)
    
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float]:
        """No data migration"""
        return 0, 0

# migrate previous tokens if reach the threshold
class PriorMigration(BaseDataMigration):
    def __init__(self, config, placement):
        super().__init__(config, placement)
    
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float]:
        """
        When the HBM utilization exceeds the threshold, migrate 16 tokens' KV caches 
        from HBM to external memory at the layer level.
        
        For each of up to 16 tokens (selected based on token IDs that have at least one layer in HBM),
        iterate over all layers. For each layer that is currently in HBM (value 0) and not already 
        migrated (or skipped, value 1 or 2), update that layer's status to 1 (external memory) and 
        subtract the per-layer size from the HBM usage. Accumulate the migrated data sizes in layer-level 
        precision.
        
        Returns:
            D_MR: Total migrated data read from HBM (in bytes). (Source data read)
            D_MW: Total migrated data written to external memory (in bytes). (Destination data write)
        """
        D_MR = 0
        D_MW = 0
        num_to_migrate = 16
        layer_size = self.get_layer_size()  # 2 * d * dtype_size
        
        # Only perform migration if the HBM utilization rate exceeds the threshold.
        if self.exceed_threshold():
            # print(f"Exceed threshold!")
            # Gather tokens that have at least one layer in HBM.
            tokens_in_hbm = []
            # We assume that if a token has any layer with value 0, it is eligible.
            for token_id, layer_status in self.token_layer_status.items():
                if 0 in layer_status:
                    tokens_in_hbm.append(token_id)
                    if len(tokens_in_hbm) >= num_to_migrate:
                        break
                    continue
            tokens_in_hbm = sorted(tokens_in_hbm)
            
            # If fewer than 16 tokens are found, adjust the number.
            if len(tokens_in_hbm) < num_to_migrate:
                num_to_migrate = len(tokens_in_hbm)
            
            # Select the tokens to migrate (for example, the earliest tokens by ID).
            tokens_to_migrate = tokens_in_hbm[:num_to_migrate]
            
            # For each selected token, iterate over its layers.
            for token_id in tokens_to_migrate:
                # Ensure the token is initialized (should be by now).
                self.initialize_token(token_id)
                for layer in range(self.cfg.L):
                    # Only migrate layers currently in HBM (status 0)
                    if self.token_layer_status[token_id][layer] == 0:
                        # Update the status to 1 (external memory).
                        self.token_layer_status[token_id][layer] = 1
                        # Decrease the HBM occupancy in the placement strategy.
                        self.cfg.C_HBM -= layer_size
                        # Accumulate the migrated data sizes.
                        # For migration from HBM to external, D_MR is the amount read from HBM,
                        # and D_MW is the amount written to external memory.
                        D_MW += layer_size
            return D_MR, D_MW
        
        # If threshold is not exceeded, no migration occurs.
        return 0, 0

# Migrate skipped tokens to external memory
class SkippedTokensMigration(BaseDataMigration):
    def __init__(self, config, placement):
        super().__init__(config, placement)
    
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float]:
        """
        For the current processing step (n, l, s), migrate tokens marked as skipped in the trace
        from HBM to external memory at layer level.
        
        For each token in the skipped tokens list from the trace, iterate over all layers.
        For each layer that is currently in HBM (status 0) (and not already migrated or skipped),
        update that layer's status to 1 (external memory), subtract the per-layer size from the HBM occupancy,
        and accumulate the migrated data sizes.
        
        In this simulation:
          - D_MR represents the migrated data read (from HBM, conceptually, but we target external memory).
          - D_MW represents the migrated data written to external memory.
        """
        D_MR = 0
        D_MW = 0
        layer_size = self.get_layer_size()  # per-layer KV cache size: 2 * d * dtype_size
        
        # Proceed only if the HBM utilization exceeds the threshold.
        if self.exceed_threshold():
            # print(f"Exceed threshold!")
            # Retrieve the trace for the current step.
            step_info = self.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
            skipped_tokens = step_info["skip_token_kv"]
            
            # Iterate over each token in the skipped list.
            for token_id in skipped_tokens:
                self.initialize_token(token_id)
                # For each layer in the token:
                for layer in range(self.cfg.L):
                    # If the KV cache for this layer is in HBM (status 0), migrate it.
                    if self.token_layer_status[token_id][layer] == 0:
                        # Update layer status to external memory.
                        self.token_layer_status[token_id][layer] = 1
                        # Decrease the HBM occupancy in the placement strategy.
                        self.cfg.C_HBM -= layer_size
                        # Accumulate migrated data sizes.
                        D_MW += layer_size
            return D_MR, D_MW
        return 0, 0

    
class PastWindowMigration(BaseDataMigration):
    def __init__(self, config, placement):
        super().__init__(config, placement)
        self.window_size = 50

    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float]:
        """
        Look ahead the next token to migrate important tokens that from the external memory to HBM. Unimportant tokens from the HBM to the external memory..

        Steps:
          1. Define the window as tokens from max(0, n - window_size + 1) to n.
          2. For each token in this window, retrieve its trace record and form the union of all tokens
             marked as skipped (using the "skip_token_kv" list).
          3. For each token in this union, iterate over all layers. For each layer that is currently in HBM 
             (i.e. token_layer_status[token][layer] == 0), update its status to 1 (external memory) and 
             subtract the per-layer size from HBM usage.
          4. Accumulate the migrated data sizes at layer-level precision. Here, D_MR represents the total data
             read from HBM (migrated out) and D_MW represents the total data written to external memory.
        """
        token_layer_size = self.get_layer_size()  # per-layer size = 2 * d * dtype_size
        D_MR = 0
        D_MW = 0

        # Define the time window.
        start = max(0, n - self.window_size + 1)
        # Gather the union of all tokens marked as skipped in this window for the current (l, s).
        tokens_to_migrate = set()
        for idx in range(start, n + 1):
            key = (idx, l, s)
            info = self.trace.get(key)
            if info is not None:
                tokens_to_migrate.update(info.get("skip_token_kv", []))
        
        # For each token in the union, check each layer.
        for token in tokens_to_migrate:
            self.initialize_token(token)
            for layer in range(self.cfg.L):
                # If the KV cache at this layer is currently in HBM, migrate it.
                if self.token_layer_status[token][layer] == 0:
                    self.token_layer_status[token][layer] = 1  # Migrate to external memory.
                    self.cfg.C_HBM -= token_layer_size       # Update HBM occupancy.
                    D_MW += token_layer_size

        return D_MR, D_MW


class LookAheadMigration(BaseDataMigration):
    def __init__(self, config, placement):
        super().__init__(config, placement)
    
    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float]:
        """
        Look ahead at token n+1. For token n+1:
          - For each token in its skip_token_kv list, migrate layers currently in HBM (status 0)
            to external memory (set status to 1). (Migrating out.)
          - For tokens NOT in the skip list, attempt to migrate layers that are in external memory (status 1)
            into HBM (set status to 0) until the HBM capacity threshold is reached. (Migrating in.)
        
        In our simulation:
          - D_MR represents the migrated data read:
                • When migrating out: the data read from HBM.
                • When migrating in: the data read from external memory.
          - D_MW represents the migrated data written to external memory (when migrating out)
            or written to HBM (when migrating in).
        
        Both values are accumulated at layer-level (using get_layer_size() for a single layer).
        """
        layer_size = self.get_layer_size()
        D_MR = 0
        D_MW = 0
        
        next_key = (n + 1, l, s)
        if next_key not in self.trace:
            return 0, 0
        
        next_info = self.trace[next_key]
        skipped_tokens = next_info["skip_token_kv"]
        
        # PART 1: Migrate out layers for tokens that token n+1 wants to skip.
        for token in skipped_tokens:
            self.initialize_token(token)
            # For each layer in this token:
            for layer in range(self.cfg.L):
                if self.token_layer_status[token][layer] == 0:
                    # Migrate out: update layer status from 0 (HBM) to 1 (External).
                    self.token_layer_status[token][layer] = 1
                    # Decrease HBM occupancy.
                    self.cfg.C_HBM -= layer_size
                    # Accumulate migrated sizes:
                    # When migrating out, we read from HBM and write to external memory.
                    D_MW += layer_size
        
        # PART 2: Migrate in layers for tokens that are not in the skip list.
        # We iterate over all tokens known in our token_layer_status.
        for token in list(self.token_layer_status.keys()):
            if token in skipped_tokens:
                continue  # skip tokens already processed for migration out.
            self.initialize_token(token)
            # For each layer in this token:
            for layer in range(self.cfg.L):
                if self.token_layer_status[token][layer] == 1:
                    # Only attempt to migrate in if HBM capacity is below threshold.
                    if self.get_HBM_util_rate() < self.threshold:
                        # Migrate in: update layer status from 1 (External) to 0 (HBM).
                        self.token_layer_status[token][layer] = 0
                        # Increase HBM occupancy.
                        self.cfg.C_HBM += layer_size
                        # Accumulate migrated sizes:
                        # When migrating in, we read from external memory and write to HBM.
                        D_MR += layer_size
                    else:
                        # If HBM capacity threshold is reached, stop migrating in further layers.
                        break
        
        return D_MR, D_MW


class LookAheadBatchMigration(BaseDataMigration):
    def __init__(self, config, placement):
        super().__init__(config, placement)
        self.batch_size = 50

    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float]:
        """
        Look ahead at a batch of tokens (from n+1 to n+batch_size) for the current (l, s).
        
        For each token in the batch, we gather the skip_token_kv lists.
          - Tokens that are consistently skipped across the batch (i.e. appear in every skip list)
            are migrated from HBM to external memory (i.e. each layer in HBM (status 0) is set to 1).
          - Tokens that are not consistently skipped are ensured to reside in HBM.
            For those, for each layer in external memory (status 1), we attempt to migrate it in (set to 0)
            if the HBM capacity is below the threshold.
        
        The migration is done at layer-level, so that each migrated layer contributes 
        get_layer_size() bytes. In our simulation:
          - D_MR represents the data read from the source memory (HBM when migrating out, external memory when migrating in).
          - D_MW represents the data written to the destination memory.
        """
        layer_size = self.get_layer_size()
        D_MR = 0
        D_MW = 0

        # Gather skip lists from tokens n+1 to n+batch_size for the same (l, s)
        batch_skip_lists = []
        for j in range(1, self.batch_size + 1):
            key = (n + j, l, s)
            info = self.trace.get(key, {"skip_token_kv": []})
            batch_skip_lists.append(set(info["skip_token_kv"]))
        
        if not batch_skip_lists:
            return 0, 0
        
        # Compute the intersection: tokens that every token in the batch wants to skip.
        consistent_skipped = set.intersection(*batch_skip_lists)
        
        # PART 1: Migrate out layers for tokens that are consistently skipped.
        for token in consistent_skipped:
            self.initialize_token(token)
            for layer in range(self.cfg.L):
                # If this layer is currently in HBM (status 0), migrate it out.
                if self.token_layer_status[token][layer] == 0:
                    self.token_layer_status[token][layer] = 1  # Migrate out: update to external memory.
                    self.cfg.C_HBM -= layer_size            # Decrease HBM usage.
                    D_MW += layer_size
        
        # PART 2: For tokens that are not consistently skipped, try to migrate in layers.
        # Iterate over all tokens we have tracked.
        for token in list(self.token_layer_status.keys()):
            if token in consistent_skipped:
                continue  # Skip tokens already processed.
            self.initialize_token(token)
            for layer in range(self.cfg.L):
                # If a layer is in external memory (status 1), attempt to migrate it into HBM.
                if self.token_layer_status[token][layer] == 1:
                    if self.get_HBM_util_rate() < self.threshold:
                        self.token_layer_status[token][layer] = 0  # Migrate in: update to HBM.
                        self.cfg.C_HBM += layer_size            # Increase HBM usage.
                        D_MR += layer_size
                    else:
                        # If HBM capacity threshold is reached, stop trying to migrate more layers in.
                        break
        
        return D_MR, D_MW

# Look ahead the n+1 token's alpha, try to maintain it at the best ratio.
class AlphaMigration(BaseDataMigration):
    def __init__(self, config, placement):
        super().__init__(config, placement)
        self.batch_size = 16

    def migration_strategy(self, n: int, l: int, s: int) -> tuple[float, float]:
        layer_size = self.get_layer_size()
        D_MR = 0
        D_MW = 0
        deviation = 0.01
        next_n = n
        next_l = l + 1
        next_s = 0
        if next_l >= self.cfg.L:
            next_l = 0
            next_n = n + 1

        next_key = (next_n, next_l, next_s)
        if next_key not in self.trace:
            return 0, 0
        
        if s != 0:
            return 0, 0
        
        ratio = self.cfg.B_HBM / (self.cfg.B_HBM + min(self.cfg.B_ext_interface_R, self.cfg.B_ext_internal))
        full = False

        current_alpha = self.plc.alpha_strategy(next_n, next_l, next_s)
        while not full and abs(current_alpha - ratio) > deviation:
            if current_alpha > ratio:
                # Too high: we want to reduce HBM contribution.
                # Find one token (from those tracked) that currently has its next_l layer in HBM.
                adjusted = False
                for token in self.token_layer_status.keys():
                    if self.token_layer_status[token][next_l] == 0:
                        # Migrate out this layer: update status to 1.
                        self.token_layer_status[token][next_l] = 1
                        # Decrease HBM occupancy via the placement instance.
                        self.cfg.C_HBM -= layer_size
                        D_MW += layer_size  # data written to external memory.
                        adjusted = True
                        break
                if not adjusted:
                    # No candidate found, mark full.
                    full = True
            else:
                # Too low: we want to increase HBM contribution.
                # Find one token that has its next_l layer in External memory.
                adjusted = False
                for token in self.token_layer_status.keys():
                    if self.token_layer_status[token][next_l] == 1:
                        if self.get_HBM_util_rate() < self.threshold:
                            self.token_layer_status[token][next_l] = 0
                            self.cfg.C_HBM += layer_size
                            D_MR += layer_size  # data read from external memory.
                            adjusted = True
                            break
                        else:
                            full = True
                            break
                if not adjusted:
                    full = True
            # Recompute the alpha after adjustment.
            current_alpha = self.plc.alpha_strategy(next_n, next_l, next_s)
            
        return D_MR, D_MW
        
