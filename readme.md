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

This will execute both simulation functions and display results including:

- Mean and 95th percentile TTFT (Time To First Token)
- Mean and 95th percentile completion times
- Mean and 95th percentile TBT (Time Between Tokens)
- GPU utilization
- CDF plots for TTFT and TBT distributions

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
- **Completion Time**: Total time from arrival to completion of all output
  tokens
- **TBT (Time Between Tokens)**: Average time between consecutive output tokens
- **Utilization**: Fraction of time the GPU is busy processing requests

## Output

Each simulation function returns:

- `results`: A pandas DataFrame with per-job metrics
- `per_token_tbt`: List of individual token generation times for TBT analysis

The script automatically prints summary statistics and generates CDF plots for
TTFT and TBT.
