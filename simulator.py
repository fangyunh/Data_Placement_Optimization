import numpy as np
import math
import matplotlib.pyplot as plt
import random
from memory_status import ModelConfig, MemStatus
from placement import BaseStrategy, PreferHBM, SplitToken, BatchRatio, LookAheadBatch, LayerImportance, AlphaLayersDistribution
from migration import BaseDataMigration, NoMigration, PriorMigration, SkippedTokensMigration, PastWindowMigration, LookAheadMigration, LookAheadBatchMigration, AlphaMigration

BYTES_TO_GB = 1024**3


class MemorySimulator:
    def __init__(self, config: ModelConfig, strategy, migration, HBM_inclusive, trace_file):
        self.cfg = config
        self.stg = strategy
        self.mig = migration
        self.is_inclusive = HBM_inclusive
        self.total_time = 0.0
        self.step_details = []
        self.trace = load_trace(trace_file)


    def calculate_step_time(self, n: int, l: int, s: int, 
                       alpha: float, beta: float, D_MR: float, D_MW: float):
        """Calculate time consumption for one step"""
        # Calculate data sizes
        D_R, D_W = self.stg.calculate_data_sizes(n, l, s)
        # should modify the way to calculate time, inclusive / exclusive
        # represents the ratio of KV cache.
        # Calculate HBM time
        HBM_read = alpha * D_R
        HBM_write = beta * D_W
        HBM_migration = D_MR + D_MW
        T_HBM = (HBM_read + HBM_write + HBM_migration) / self.cfg.B_HBM  # in ns
        
        
        # New external read calculation using interface vs internal minimum
        ext_read = (1 - alpha) * D_R / min(self.cfg.B_ext_interface_R, 
                                        self.cfg.B_ext_internal)
        
        # New write + migration calculation
        write_migration = ((1 - beta) * D_W + D_MW) / self.cfg.B_ext_interface_W if self.cfg.B_ext_interface_R > 0 else 0
        migration_read = D_MR / self.cfg.B_ext_interface_R if self.cfg.B_ext_interface_R > 0 else 0
        internal_migration = ((1 - beta) * D_W + HBM_migration) / self.cfg.B_ext_internal if self.cfg.B_ext_internal > 0 else 0
        ext_write_migration = max(write_migration, migration_read, internal_migration)
        
        T_ext = ext_read + ext_write_migration
        
        return max(T_HBM, T_ext)
    
    def simulate(self):
        """
        Run full simulation
        Args:
            alpha_strategy: Function(n,l,s) -> alpha
            beta_strategy: Function(n,l,s) -> beta
            migration_strategy: Function(n,l,s) -> (D_MR, D_MW)
        """
        self.total_time = 0.0
        self.step_details = []

        for n in range(self.cfg.N_pre, self.cfg.N_pre + self.cfg.N):
            for l in range(self.cfg.L):
                for s in [0, 1]:  # MHA and MLP
                    step_info = self.mig.trace.get((n, l, s), {"skip_token_kv": [], "skip_layer": False})
                    # If this layer is skipped in the trace, no data is processed.
                    if step_info["skip_layer"]:
                        continue
                    # Get strategies
                    alpha = self.stg.alpha_strategy(n, l, s)
                    beta = self.stg.beta_strategy(n, l, s)
                    D_MR, D_MW = self.mig.migration_strategy(n, l, s)
                    
                    # Calculate step time
                    step_time = self.calculate_step_time(n, l, s, alpha, beta, D_MR, D_MW)
                    self.total_time += step_time
                    
                    # Record step details
                    self.step_details.append({
                        'n': n,
                        'l': l,
                        's': s,
                        'time': step_time,
                        'alpha': alpha,
                        'beta': beta,
                        'D_MR': D_MR,
                        'D_MW': D_MW
                    })
        return self.total_time

if __name__ == "__main__":
    # Initialize configuration
    config = ModelConfig()

    # List of placement strategies from placement.py.
    # placement_classes = [PreferHBM, SplitToken, BatchRatio, LookAheadBatch, LayerImportance]
    placement_classes = [LayerImportance, AlphaLayersDistribution]
    # List of migration strategies from migration.py.
    # migration_classes = [NoMigration, PriorMigration, SkippedTokensMigration, PastWindowMigration, LookAheadMigration, LookAheadBatchMigration, AlphaMigration]
    migration_classes = [AlphaMigration]
    # Iterate over each combination of placement and migration strategy.
    for p_cls in placement_classes:
        for m_cls in migration_classes:
            # Instantiate migration instance (pass config and temporary strategy placeholder).
            mig_instance = m_cls(config, None)
            # Instantiate placement instance, passing the migration instance.
            placement_instance = p_cls(config, mig_instance)

            mig_instance.plc = placement_instance
            
            # Create the simulator with this combined strategy.
            simulator = MemorySimulator(config, placement_instance, mig_instance)
            total_time = simulator.simulate()
            avg_alpha = sum(step['alpha'] for step in simulator.step_details) / len(simulator.step_details)
            
            print(f"Combination: Data Placement = {p_cls.__name__}, Data Migration = {m_cls.__name__}")
            print(f"Total simulation time: {total_time:.4f} ns, {total_time/1e9:.4f} seconds")
            print(f"Average time per token: {total_time/config.N:.6f} ns")
            print(f"Avg alpha rate: {avg_alpha:.6f}")
            print("-" * 50)
    

    # times = [step['alpha'] for step in simulator.step_details]
    # plt.plot(times)
    # plt.xlabel('Step')
    # plt.ylabel('Time (ns)')
    # plt.title('Step Time Distribution')
    # plt.show()