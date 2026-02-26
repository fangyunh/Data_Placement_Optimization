from abc import ABC, abstractmethod
from memory_status import ModelConfig, MemStatus
from collections import Counter, deque
from math import ceil
import random

class BaseDataMigration(ABC):
    def __init__(self, config: ModelConfig, status: MemStatus):
        self.cfg = config
        self.status = status
    
    @abstractmethod
    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        """Define migration sizes (D_MR, D_MW)."""
        pass

class NoMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
    
    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        return [0.0, 0.0, 0.0, 0.0]
    
class JITMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)

    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        hbm_MW = 0.0
        ext_MR = 0.0

        try:
            experts_read = self.status.trace.get_active_experts(n, l)
            kv_reads = self.status.trace.get_all_kv_reads_by_query(n, l)
        except (IOError, IndexError, KeyError):
            return [0.0, 0.0, 0.0, 0.0]

        kv_read_set = set()
        for q, tokens in kv_reads.items():
            for t in tokens:
                kv_read_set.add((q, t))

        items_to_add = []
        for e in experts_read:
            if not self.status.is_expert_in_hbm(e, l):
                items_to_add.append(("EXPERT", e))
        for (q, t) in kv_read_set:
            if not self.status.is_kv_in_hbm(q, t, l):
                items_to_add.append(("KV", (q, t)))

        if not items_to_add:
            return [0.0, 0.0, 0.0, 0.0]

        expert_victims = [e for e in self.status.hbm_expert_slices[l] if e not in experts_read]
        kv_victims = [(q, t) for (q, t) in self.status.hbm_kv_tokens[l] if (q, t) not in kv_read_set]
        
        all_victims = [("EXPERT", e) for e in expert_victims] + [("KV", (q, t)) for (q, t) in kv_victims]
        random.shuffle(all_victims)
        victim_queue = deque(all_victims)

        for item_type, data in items_to_add:
            if item_type == "EXPERT":
                expert_id = data
                while not self.status.add_expert_to_hbm(expert_id, l):
                    if not victim_queue: break
                    victim_type, victim_data = victim_queue.popleft()
                    if victim_type == "EXPERT":
                        self.status.remove_expert_from_hbm(victim_data, l)
                    elif victim_type == "KV":
                        self.status.remove_kv_from_hbm(victim_data[0], victim_data[1], l)
                else:
                    size = self.status.single_expert_slice_size
                    hbm_MW += size
            
            elif item_type == "KV":
                q, t = data
                while not self.status.add_kv_to_hbm(q, t, l):
                    if not victim_queue: break
                    victim_type, victim_data = victim_queue.popleft()
                    if victim_type == "EXPERT":
                        self.status.remove_expert_from_hbm(victim_data, l)
                    elif victim_type == "KV":
                        self.status.remove_kv_from_hbm(victim_data[0], victim_data[1], l)
                else:
                    size = self.status.single_KV_cache_size
                    hbm_MW += size

        return [0.0, hbm_MW, ext_MR, 0.0]

class LookAheadOneMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)

    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        next_n = n + 1
        if next_n >= (self.cfg.N_pre + self.cfg.N):
            return [0.0, 0.0, 0.0, 0.0] 
        
        hbm_MW = 0.0
        ext_MR = 0.0

        try:
            experts_needed_next = self.status.trace.get_active_experts(next_n, l)
            kv_reads_needed_next = self.status.trace.get_all_kv_reads_by_query(next_n, l)
        except (IOError, IndexError, KeyError):
            return [0.0, 0.0, 0.0, 0.0]

        items_to_add = []
        for e in experts_needed_next:
            if not self.status.is_expert_in_hbm(e, l):
                items_to_add.append(("EXPERT", e))

        kv_needed_next_set = set()
        for q, tokens in kv_reads_needed_next.items():
            for t in tokens:
                kv_needed_next_set.add((q, t))
                if not self.status.is_kv_in_hbm(q, t, l):
                    items_to_add.append(("KV", (q, t)))

        if not items_to_add:
            return [0.0, 0.0, 0.0, 0.0]

        eviction_candidates = []
        for e in self.status.hbm_expert_slices[l]:
            if e not in experts_needed_next:
                eviction_candidates.append(("EXPERT", e))

        for (q, t) in self.status.hbm_kv_tokens[l]:
            if (q, t) not in kv_needed_next_set:
                eviction_candidates.append(("KV", (q, t)))
        
        random.shuffle(eviction_candidates)
        
        for item_type, data in items_to_add:
            if item_type == "EXPERT":
                expert_id = data
                while not self.status.add_expert_to_hbm(expert_id, l):
                    if not eviction_candidates: break
                    victim_type, victim_data = eviction_candidates.pop()
                    if victim_type == "EXPERT":
                        self.status.remove_expert_from_hbm(victim_data, l)
                    elif victim_type == "KV":
                        self.status.remove_kv_from_hbm(victim_data[0], victim_data[1], l)
                else:
                    size = self.status.single_expert_slice_size
                    hbm_MW += size
                    ext_MR += size
            
            elif item_type == "KV":
                q, t = data
                while not self.status.add_kv_to_hbm(q, t, l):
                    if not eviction_candidates: break
                    victim_type, victim_data = eviction_candidates.pop()
                    if victim_type == "EXPERT":
                        self.status.remove_expert_from_hbm(victim_data, l)
                    elif victim_type == "KV":
                        self.status.remove_kv_from_hbm(victim_data[0], victim_data[1], l)
                else:
                    size = self.status.single_KV_cache_size
                    hbm_MW += size
                    ext_MR += size

        return [0.0, hbm_MW, ext_MR, 0.0]

class PageMigration(BaseDataMigration):
    def __init__(self, config, status):
        super().__init__(config, status)
        self.page_size = getattr(self.status, 'page_size', 16)
        if self.page_size <= 0: self.page_size = 16
        self.num_prefill_pages = (self.cfg.N_pre + self.page_size - 1) // self.page_size

    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        hbm_MW = 0.0
        ext_MR = 0.0

        try:
            experts_read = self.status.trace.get_active_experts(n, l)
            kv_reads = self.status.trace.get_all_kv_reads_by_query(n, l)
        except (IOError, IndexError, KeyError):
            return [0.0, 0.0, 0.0, 0.0]

        kv_read_set = set()
        for q, tokens in kv_reads.items():
            for t in tokens:
                kv_read_set.add((q, t))

        experts_to_add = []
        for e in experts_read:
            if not self.status.is_expert_in_hbm(e, l):
                experts_to_add.append(e)

        pages_to_add_set = set()
        for (q, t) in kv_read_set:
            if t < self.cfg.N_pre:
                if not self.status.is_kv_in_hbm(q, t, l):
                    page_index = t // self.page_size
                    pages_to_add_set.add((q, page_index))
        
        expert_victims = [e for e in self.status.hbm_expert_slices[l] if e not in experts_read]
        
        kv_page_victims = []
        for q in range(self.cfg.batch_size):
            for p_idx in range(self.num_prefill_pages):
                page_start = p_idx * self.page_size
                if self.status.is_kv_in_hbm(q, page_start, l):
                    is_victim = True
                    for t_in_page in range(page_start, min(page_start + self.page_size, self.cfg.N_pre)):
                        if (q, t_in_page) in kv_read_set:
                            is_victim = False
                            break
                    if is_victim:
                        kv_page_victims.append((q, p_idx))

        all_victims = [("EXPERT", e) for e in expert_victims] + \
                      [("KV_PAGE", (q, p_idx)) for (q, p_idx) in kv_page_victims]
        random.shuffle(all_victims)
        victim_queue = deque(all_victims)

        for expert_id in experts_to_add:
            while not self.status.add_expert_to_hbm(expert_id, l):
                if not victim_queue: break
                
                victim_type, victim_data = victim_queue.popleft()
                if victim_type == "EXPERT":
                    self.status.remove_expert_from_hbm(victim_data, l)
                elif victim_type == "KV_PAGE":
                    (q_v, p_idx_v) = victim_data
                    page_start_v = p_idx_v * self.page_size
                    for t_v in range(page_start_v, min(page_start_v + self.page_size, self.cfg.N_pre)):
                        self.status.remove_kv_from_hbm(q_v, t_v, l)
            else:
                size = self.status.single_expert_slice_size
                hbm_MW += size
        
        for (q, page_index) in pages_to_add_set:
            page_start = page_index * self.page_size
            page_end = min(page_start + self.page_size, self.cfg.N_pre)
            
            tokens_to_add_in_page = []
            page_hbm_mw_cost = 0.0
            page_ext_mr_cost = 0.0
            
            for t in range(page_start, page_end):
                if not self.status.is_kv_in_hbm(q, t, l):
                    tokens_to_add_in_page.append(t)
                    size = self.status.single_KV_cache_size
                    page_hbm_mw_cost += size
                    if (q, t) not in kv_read_set:
                        page_ext_mr_cost += size
            
            if not tokens_to_add_in_page: continue

            failed_tokens = []
            for t in tokens_to_add_in_page:
                if not self.status.add_kv_to_hbm(q, t, l):
                    failed_tokens.append(t)
            
            while failed_tokens:
                if not victim_queue: break
                
                victim_type, victim_data = victim_queue.popleft()
                if victim_type == "EXPERT":
                    self.status.remove_expert_from_hbm(victim_data, l)
                elif victim_type == "KV_PAGE":
                    (q_v, p_idx_v) = victim_data
                    page_start_v = p_idx_v * self.page_size
                    for t_v in range(page_start_v, min(page_start_v + self.page_size, self.cfg.N_pre)):
                        self.status.remove_kv_from_hbm(q_v, t_v, l)
                
                still_failed = []
                for t in failed_tokens:
                     if not self.status.add_kv_to_hbm(q, t, l):
                         still_failed.append(t)
                failed_tokens = still_failed
            
            if not failed_tokens:
                hbm_MW += page_hbm_mw_cost
                ext_MR += page_ext_mr_cost

        return [0.0, hbm_MW, ext_MR, 0.0]

class OnlineAdaptiveMigration(BaseDataMigration):
    def __init__(self, config, status, window_size: int = 100):
        super().__init__(config, status)
        self.window_size = window_size
        self.total_tokens = self.cfg.N_pre + self.cfg.N

    def _calculate_future_accesses(self, n: int, l: int):
        future_accesses = {}
        scan_start_n = n + 1
        scan_end_n = min(scan_start_n + self.window_size, self.total_tokens)

        for scan_n in range(scan_start_n, scan_end_n):
            for scan_l in range(self.cfg.L):
                try:
                    experts_read = self.status.trace.get_active_experts(scan_n, scan_l)
                    kv_reads = self.status.trace.get_all_kv_reads_by_query(scan_n, scan_l)
                except (IOError, IndexError, KeyError):
                    continue

                for e in experts_read:
                    key = ("EXPERT", e, scan_l)
                    future_accesses[key] = future_accesses.get(key, 0) + 1

                for q, tokens in kv_reads.items():
                    for t in tokens:
                        key = ("KV", q, t, scan_l)
                        future_accesses[key] = future_accesses.get(key, 0) + 1
                        
        return future_accesses

    def _calculate_budgets(self, n: int, l: int):
        D_R, _ = self.status.calculate_data_sizes(n, l)
        alpha = self.status.max_alpha(n, l)
        
        if D_R == 0: return 0.0, 0.0

        T_HBM_compute = (alpha * D_R) / self.cfg.B_HBM
        T_ext_compute = ((1 - alpha) * D_R) / self.cfg.B_ext_interface_R 
        
        M_HBM_MW = 0.0 
        M_Ext_MR = 0.0 

        if T_ext_compute > T_HBM_compute:
            slack_time = T_ext_compute - T_HBM_compute
            M_HBM_MW = self.cfg.B_HBM * slack_time
        elif T_HBM_compute > T_ext_compute:
            slack_time = T_HBM_compute - T_ext_compute
            M_Ext_MR = self.cfg.B_ext_interface_R * slack_time
            
        return M_HBM_MW, M_Ext_MR

    def _get_item_cost_and_size(self, item_id: tuple):
        item_type = item_id[0]
        size = 0.0
        if item_type == "EXPERT":
            size = self.status.single_expert_slice_size
        elif item_type == "KV":
            size = self.status.single_KV_cache_size
        return size

    def migration_strategy(self, n: int, l: int) -> tuple[float, float, float, float]:
        experts_read = self.status.trace.get_active_experts(n, l)
        kv_reads = self.status.trace.get_all_kv_reads_by_query(n, l)
        WS_current = set([("EXPERT", e, l) for e in experts_read] + 
                         [("KV", q, t, l) for q, tokens in kv_reads.items() for t in tokens])
        
        future_accesses = self._calculate_future_accesses(n, l)
        M_HBM_MW, M_Ext_MR = self._calculate_budgets(n, l)
        
        if M_HBM_MW <= 0.0 and M_Ext_MR <= 0.0:
            return [0.0, 0.0, 0.0, 0.0]

        candidate_list = []
        for e in range(self.cfg.num_experts):
            if not self.status.is_expert_in_hbm(e, l):
                candidate_list.append(("EXPERT", e, l))

        for q in range(self.cfg.batch_size):
            for t in range(self.cfg.N_pre):
                if not self.status.is_kv_in_hbm(q, t, l):
                    candidate_list.append(("KV", q, t, l))

        migration_candidates = []
        for item_id in candidate_list:
            size = self._get_item_cost_and_size(item_id)
            future_count = future_accesses.get(item_id, 0)
            cost_factor = 2 if item_id not in WS_current else 1
            ratio = future_count * size / (cost_factor * size) if size > 0 else 0.0
            migration_candidates.append((ratio, size, cost_factor, item_id))

        migration_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        total_hbm_mw = 0.0
        total_ext_mr = 0.0

        expert_victims = [e for e in self.status.hbm_expert_slices[l] if ("EXPERT", e, l) not in WS_current]
        kv_victims = [(q, t) for q in range(self.cfg.batch_size) for t in range(self.cfg.N_pre) 
                      if self.status.is_kv_in_hbm(q, t, l) and ("KV", q, t, l) not in WS_current]
        
        # --- CORRECT PACKING OF VICTIMS ---
        all_victims = [("EXPERT", (e, l)) for e in expert_victims] + \
                      [("KV", (q, t, l)) for q, t in kv_victims]
        random.shuffle(all_victims) 
        victim_queue = deque(all_victims)

        for ratio, size, cost_factor, item_id in migration_candidates:
            cost_hbm = size 
            cost_ext = size if cost_factor == 2 else 0.0 

            if (total_hbm_mw + cost_hbm > M_HBM_MW) and (M_HBM_MW > 0.0): continue
            if (total_ext_mr + cost_ext > M_Ext_MR) and (M_Ext_MR > 0.0): continue

            item_type = item_id[0]
            success = False

            while True:
                is_full = False
                
                if item_type == "EXPERT":
                    e = item_id[1]
                    if self.status.hbm_capacity_remaining < size:
                        is_full = True
                    else:
                        self.status.add_expert_to_hbm(e, l)
                        success = True
                        break
                
                elif item_type == "KV":
                    q, t = item_id[1], item_id[2]
                    if self.status.hbm_capacity_remaining < size:
                        is_full = True
                    else:
                        self.status.add_kv_to_hbm(q, t, l)
                        success = True
                        break
                
                if is_full:
                    # --- SILENCED DEBUG NOISE, KEPT CRITICAL CHECKS ---
                    if not victim_queue: 
                        import sys
                        print(f"ERROR: HBM FULL, Victim queue empty. Cannot add {item_type} at n={n}, l={l}.", file=sys.stderr)
                        break

                    victim_type, victim_data = victim_queue.popleft()
                    
                    if victim_type == "EXPERT":
                        e_v, l_v = victim_data
                        self.status.remove_expert_from_hbm(e_v, l_v)
                    elif victim_type == "KV":
                        # --- CORRECT UNPACKING OF VICTIMS ---
                        q_v, t_v, l_v = victim_data 
                        self.status.remove_kv_from_hbm(q_v, t_v, l_v)
                else:
                    break

            if success:
                total_hbm_mw += cost_hbm
                total_ext_mr += cost_ext
                M_HBM_MW -= cost_hbm
                M_Ext_MR -= cost_ext
        
        return [0.0, total_hbm_mw, total_ext_mr, 0.0]