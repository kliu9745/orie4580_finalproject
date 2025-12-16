# Impact of Batching (ORIE 4580/5580 Simulation Project)
This contains a small, **reproducible** batching experiment for a *single-GPU* LLM-serving simulator.
We compare:

- **Fixed-size batching**: wait until the queue has accumulated a target **token load** `b_target` (or a timeout), then dispatch a batch.
- **Dynamic batching**: adapt the batch target to current queue depth and use a **latency guard** to dispatch early if the oldest queued job is waiting too long.

The goal is to quantify the trade-off between **throughput** and **tail latency** (P95 TTFT / completion / TBT) under different batching strategies.

---

## Files

- `simulation_batching_impact.py`  
  Core discrete-event simulator + policy implementations. Exposes:
  - `simulation_fixed_size_batching(...)`
  - `simulation_dynamic_batching(...)`

- `batching_impact_experiment_runner.py`  
  A runnable script that:
  - runs dynamic once (baseline),
  - runs fixed for `b ∈ {32, 64, 96, 128, 160, 192, 256}`,
  - generates plots + prints a summary table.

---

## Model (high level)

### Workload
- Query arrivals follow a Poisson process with rate `LAMBDA` (queries/sec).
- Each query has:
  - prompt length `L` (tokens) — in this simplified experiment, we use a constant `prompt_length`
  - output budget `B` (tokens) — constant here

### Prefill + Decode
Each request is modeled as:
1. **Prefill**: a single job of size `L` tokens
2. **Decode**: `B` sequential 1-token jobs  
   (only the next decode token becomes available after the previous one completes)

### GPU execution
- The GPU processes **batches** of jobs.
- Batch service time includes a per-batch setup cost plus a marginal token component (beyond `b0`), with randomness.

---

## Quickstart

1. Create a new notebook.
2. Upload the two `.py` files into the Colab session:
   - `simulation_batching_impact.py`
   - `batching_impact_experiment_runner.py`
3. Run:

```bash
!python batching_impact_experiment_runner.py
```
---

## Quickstart (Local)

```bash
pip install numpy pandas matplotlib
python batching_impact_experiment_runner.py
```

---

## What you’ll see

The runner generates:

1. **Completion latency density overlay** across fixed `b` values  
2. **TBT (time-between-tokens) density overlay** across fixed `b` values  
3. **Per-b completion histograms**: fixed `b` vs dynamic  
4. **Per-b TBT histograms**: fixed `b` vs dynamic  
5. **Throughput / P95 completion / P95 TTFT vs fixed b**, with dynamic shown as a dashed baseline  

Finally, it prints a table like:

- `throughput_per_s`
- `p95_ttft_s`
- `p95_completion_s`
- `p95_tbt_s`
- `utilization`

---

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
---

## Reproducibility

- The simulator uses a fixed RNG seed by default (inside `simulation_batching_impact.py`).
- If you want multiple replications, you can:
  1) change the seed and rerun, or  
  2) wrap the runner logic in a loop over seeds and aggregate results (recommended for confidence intervals).

---

