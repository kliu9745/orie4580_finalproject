# ORIE 4580 Final Project: GPU Scheduling Simulation

This repository contains a simulation of GPU scheduling strategies for language
model inference workloads. The simulation models the prefill and decode phases
of processing queries, comparing two scheduling approaches:
First-Come-First-Served (FCFS) with process-to-completion, and prefill batching
with sequential decode.

## Overview

The `simulation.py` script implements discrete-event simulations of two main
scheduling strategies:

1. **FCFS (Process-to-Completion)**: Each job is processed entirely before the
   next begins, including both prefill (prompt processing) and decode (token
   generation) phases.

2. **Prefill Batching**: Jobs are batched during the prefill phase to reduce
   setup overhead, then decoded sequentially.

## Dependencies

The simulation requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`

Install them using pip:

```bash
pip install numpy pandas matplotlib
```

## Running the Simulation

To run the simulation with default parameters:

```bash
python simulation.py
```

This will execute both simulation functions and display CDF plots for TTFT and TBT distributions. 
You will see a series of windows open. Once you see one plot, exit out of the window and that will lead
you to the next plot. There are 4 plots in total. The first two are the TTFT and TBT of Prefill-Priority, 
and the last two are the TTFT and TBT for Decode-Priority (Batching)

## Simulation Parameters

### Common Parameters

- `SIM_TIME`: Total simulation time in seconds (default: 200)
- `LAMBDA`: Query arrival rate in queries/second (default: 15)
- `B`: Number of output tokens per request (default: 16)
- `c`: Setup cost in seconds (default: 20e-3)
- `a`: Per-token marginal cost in seconds (default: 0.3e-3)
- `b0`: Token threshold for service model (default: 64)

### Prefill Batching Specific
- `K`: Maximum tokens per prefill batch (default: 128)

## Key Metrics

- **TTFT (Time To First Token)**: Time from query arrival until the first output
  token is generated
- **TBT (Time Between Tokens)**: Average time between consecutive output tokens

## Output
Each simulation function returns:

- `results`: A pandas DataFrame with per-job metrics
  - arrival: Time the job entered the system.
  - ttft: Time To First Token. The latency from arrival until the first token is generated.
  - completion: Total time until the job is fully finished.
  - prefill_service: Time spent processing the prompt.
  - decode_service: Time spent generating new tokens.

- `per_token_tbt`: List of individual token generation times for TBT analysis

The script automatically prints summary statistics and generates CDF plots for
TTFT and TBT.

# 1. FCFS Strategy (No Batching)
Use simulation_priorize_decode to model a system where every job is processed immediately upon arrival (sequentially).

## Run simulation with no batching
results, tbt_data = simulation_priorize_decode(
    SIM_TIME=200, 
    LAMBDA=15
)

print(f"Mean TTFT: {results['ttft'].mean():.4f}s")


from llm_sim import simulation_priorize_prefill

# 2. Batching Strategy
Use simulation_priorize_prefill to model a system that groups incoming requests into batches before processing.

## Run simulation with a batch limit of 128 tokens
results, tbt_data = simulation_priorize_prefill(
    SIM_TIME=200, 
    LAMBDA=15, 
    K=128
)

print(f"Mean TTFT: {results['ttft'].mean():.4f}s")

----

## Impact of Batching Experiment

This directory contains a batching experiment for a single-GPU LLM-serving simulator.  
We compare fixed-size batching against dynamic batching and evaluate their impact on
throughput and user-facing latency metrics.

## How to Run

From the directory batching_experiments, run:
```bash
python batching_impact_experiment_runner.py
```

## Outputs

Running the script generates the following plots:

- Completion latency distributions across fixed batch sizes
- P95 completion latency vs batch size
- P95 TTFT vs batch size
- Throughput vs batch size
- TBT (time between tokens) distributions
- Fixed batching vs dynamic batching comparison plots

## Key parameters to tweak

Open `batching_impact_experiment_runner.py` and edit:

- `SIM_TIME`: simulation horizon
- `LAMBDA`: arrival rate (load)
- `B`: output tokens per request
- Service model: `c` (setup cost), `a` (per-token time scale), `b0` (threshold)
- Fixed batching list: `B_LIST`
- Timeouts / guards:
  - `FIXED_TIMEOUT_S`
  - dynamic `latency_target_s`
- Warm-up trimming:
  - `WARMUP_COMPLETED`

----

## Chunked Prefill Experiment

This directory contains the chunked-prefill experiment for a single-GPU LLM-serving simulator.
We study how splitting prompt prefill into smaller chunks affects latency, throughput,
and decode stalls.

## How to Run

From the project root, run:
```bash
python chunked_prefill_experiment.py
```

## Outputs

Running the script generates the following plots:

- Mean TTFT vs chunk size
- Throughput vs chunk size
- Total decode stall time vs chunk size

The script also prints a summary table containing, for each chunk size:

- Throughput
- GPU utilization
- Mean and P95 TTFT
- Mean and P95 completion latency
- Mean and P95 TBT
- Total decode stall time

## Key Parameters

The following parameters can be modified directly in the script:

Simulation parameters:
- `SIM_TIME` — total simulation horizon
- `LAMBDA` — arrival rate (queries/sec)
- `prompt_lengths` — prompt tokens per request
- `B` — output tokens per request

Chunked-prefill parameters:
- `chunk_size` — maximum prefill tokens taken from one query per iteration
- `K` — total token budget per GPU iteration
- `chunk_sizes` — list of chunk sizes swept in the experiment

Service model parameters:
- `c` — per-batch setup cost
- `a` — per-token marginal cost
- `b0` — token threshold

----

## Workload Sensitivity Experiment

This directory contains the workload sensitivity experiment for the single-GPU LLM-serving simulator.

## How to Run

From the project root, run:
```bash
python workload_sensitivity_experiment.py
```

## Outputs

Running the script generates the following plots:

- TTFT vs arrival rate
- TTFT vs prompt length
- TBT distributions vs output length

## Key Parameters

The following parameters can be modified directly in  
`workload_sensitivity_experiment.py`:

- `SIM_TIME` — total simulation horizon
- `WARMUP_COMPLETED` — number of completed requests discarded as warm-up
- `LAMBDA_LIST` — arrival rate values to sweep
- `PROMPT_LENGTH_LIST` — prompt length values to sweep
- `OUTPUT_LENGTH_LIST` — output length values to sweep
