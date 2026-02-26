# Agent Background

## Project context
This project simulates step-by-step LLM inference latency under a two-level memory system:
- External memory: effectively unlimited capacity, lower bandwidth.
- HBM: limited capacity, high bandwidth.
- Both memories can be accessed by the processor concurrently.

The simulator studies how different placement and migration strategies for KV cache and MoE expert weights affect inference latency and bandwidth utilization. It also includes a high-oracle strategy that assumes perfect future knowledge of what KV/expert data will be accessed.

## Current implementation (core files)
- simulation/simulator.py: Main simulation loop, time model, trace reading, and strategy execution.
- simulation/memory_status.py: Model config, memory state tracking, size accounting, and initialization policies.
- simulation/placement.py: Placement (write) strategies for KV cache.
- simulation/migration.py: Migration strategies between external memory and HBM.
- simulation/run_experiments.py: Batch experiment runner driven by JSON settings.
- simulation/SA_simulation.py and simulation/run_SA.py: Simulated annealing to tune adaptive migration parameters.
- simulation/files_descriptions.md: High-level description of scripts.
- results/: All results will be recorded in the corresponding folder (batch num, decode token start from and to) with different sparsities.
- trace_generation/score_collect.py: Collects Mixtral MoE traces with fixed prefill and decode lengths, for a fixed batch from LongBench. Exports a 5-column CSV (query_id, token_id, layer_id, experts, attention).
- trace_generation/split.py: Applies attention sparsity by keeping top-k attention indices and produces per-sparsity trace files.

## Trace collection notes
- Current trace collection uses Mixtral with a fixed batch (e.g., 16) and fixed sequence lengths (prefill 4096, decode 4096 to 8192).
- Traces are collected in a large CSV per batch, then sparsified into multiple variants via split.py.
- The trace format is the input to TraceReader in simulation/simulator.py.

## Goals
- Improve simulator correctness, efficiency, and extensibility.
- Align latency model and memory accounting with the latest formulas and experimental directions.
- Support inclusive/exclusive memory modes, token skipping, and advanced scheduling policies.
- Provide reliable metrics for comparing strategies and a clean interface for experiments.

## My role in this project
- Read and understand current simulation logic and trace semantics.
- Propose and implement fixes, optimizations, and new features.
- Keep memory accounting safe and consistent (no underflow/overflow).
- Ensure changes are reproducible and documented.
- Suggest tests and sanity checks when modifying core logic.

## Working agreement
- Scope: All files under scheData/ are in scope for this project.
- I will read this file before performing future tasks to stay aligned with the goals and role.
