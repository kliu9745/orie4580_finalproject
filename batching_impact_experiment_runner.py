
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation_batching_impact import (
    simulation_fixed_size_batching,
    simulation_dynamic_batching,
)

# Experiment configuration
SIM_TIME = 200
LAMBDA = 15
B = 16

# Service model
c = 20e-3
a = 0.30e-3
b0 = 64
prompt_length = 20

# Fixed batching experiment
B_LIST = [32, 64, 96, 128, 160, 192, 256]
FIXED_TIMEOUT_S = 0.05

# Dynamic batching parameters
DYN_PARAMS = dict(
    b_min=32,
    b_mid=64,
    b_max=128,
    switch_mid=64,
    switch_max=256,
    latency_target_s=0.08,
)

# drop first N completed requests when computing summaries
WARMUP_COMPLETED = 1500  

def _trim_warmup(results_df: pd.DataFrame, warmup_completed: int) -> pd.DataFrame:
    df = results_df.dropna(subset=["completion"]).copy()
    df["completion_abs"] = df["arrival"] + df["completion"]
    df = df.sort_values("completion_abs").reset_index(drop=True)
    if warmup_completed > 0 and warmup_completed < len(df):
        df = df.iloc[warmup_completed:].copy()
    return df

def summarize(results_df: pd.DataFrame, per_token_tbt, warmup_completed: int) -> dict:
    df = _trim_warmup(results_df, warmup_completed)

    out = {
        "completed": int(df.shape[0]),
        "throughput_per_s": float(df.shape[0] / SIM_TIME),
        "p95_ttft_s": float(df["ttft"].quantile(0.95)),
        "p95_completion_s": float(df["completion"].quantile(0.95)),
        "mean_completion_s": float(df["completion"].mean()),
        "utilization": np.nan,
    }

    if per_token_tbt is not None and len(per_token_tbt) > 0:
        out["p95_tbt_s"] = float(np.quantile(per_token_tbt, 0.95))
        out["mean_tbt_s"] = float(np.mean(per_token_tbt))
    else:
        out["p95_tbt_s"] = np.nan
        out["mean_tbt_s"] = np.nan

    return out

# Dynamic baseline
dyn_results, dyn_tbt, dyn_batch, dyn_util = simulation_dynamic_batching(
    SIM_TIME=SIM_TIME, LAMBDA=LAMBDA, B=B,
    c=c, a=a, b0=b0, prompt_length=prompt_length,
    **DYN_PARAMS
)
dyn_summary = summarize(dyn_results, dyn_tbt, WARMUP_COMPLETED)
dyn_summary["utilization"] = float(dyn_util) if dyn_util is not None else np.nan
dyn_summary["policy"] = "dynamic"

# -----------------------------
# Run fixed experiment
# -----------------------------
fixed_runs = {}
summary_rows = []

for b in B_LIST:
    res, tbt, batch_df, util = simulation_fixed_size_batching(
        SIM_TIME=SIM_TIME, LAMBDA=LAMBDA, B=B,
        b_target=b, timeout_s=FIXED_TIMEOUT_S,
        c=c, a=a, b0=b0, prompt_length=prompt_length,
    )
    fixed_runs[b] = (res, tbt)
    s = summarize(res, tbt, WARMUP_COMPLETED)
    s["utilization"] = float(util) if util is not None else np.nan
    s["policy"] = f"fixed_b={b}"
    s["b"] = b
    summary_rows.append(s)

summary_rows.append(dyn_summary)
summary_df = pd.DataFrame(summary_rows)


# Plot 1: Completion latency density overlay across fixed b
plt.figure(figsize=(7,4))
for b in B_LIST:
    df = _trim_warmup(fixed_runs[b][0], WARMUP_COMPLETED)
    plt.hist(df["completion"], bins=60, density=True, alpha=0.35, label=f"b={b}")
plt.xlabel("Completion latency (s)")
plt.ylabel("density")
plt.title("Completion latency (density overlay) across fixed b")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# Plot 2: TBT density overlay across fixed b
plt.figure(figsize=(7,4))
for b in B_LIST:
    tbt = np.array(fixed_runs[b][1], dtype=float)
    if len(tbt) == 0:
        continue
    plt.hist(tbt, bins=80, density=True, alpha=0.35, label=f"b={b}")
plt.xlabel("TBT (s)")
plt.ylabel("density")
plt.title("TBT (density overlay) across fixed b")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# Plot 3: Per-b completion histograms vs dynamic
n = len(B_LIST)
cols = 3
rows = int(np.ceil(n / cols))
plt.figure(figsize=(5*cols, 3.5*rows))

dyn_df = _trim_warmup(dyn_results, WARMUP_COMPLETED)

for i, b in enumerate(B_LIST):
    plt.subplot(rows, cols, i+1)
    df = _trim_warmup(fixed_runs[b][0], WARMUP_COMPLETED)
    plt.hist(df["completion"], bins=40, alpha=0.6, label=f"Fixed b={b}")
    plt.hist(dyn_df["completion"], bins=40, alpha=0.4, label="Dynamic")
    plt.xlabel("Completion latency (s)")
    plt.ylabel("count")
    plt.title(f"Completion latency: Fixed b={b} vs Dynamic")
    plt.legend(fontsize=8)

plt.tight_layout()
plt.show()

# Plot 4: Per-b TBT histograms vs dynamic
plt.figure(figsize=(5*cols, 3.5*rows))

dyn_tbt_arr = np.array(dyn_tbt, dtype=float)

for i, b in enumerate(B_LIST):
    plt.subplot(rows, cols, i+1)
    tbt = np.array(fixed_runs[b][1], dtype=float)
    if len(tbt) == 0:
        continue
    plt.hist(tbt, bins=60, alpha=0.6, label=f"Fixed b={b}")
    plt.hist(dyn_tbt_arr, bins=60, alpha=0.4, label="Dynamic")
    plt.xlabel("TBT (s)")
    plt.ylabel("count")
    plt.title(f"TBT: Fixed b={b} vs Dynamic")
    plt.legend(fontsize=8)

plt.tight_layout()
plt.show()

# Plot 5: Throughput + tail metrics vs fixed b (dashed = dynamic)
fixed_only = summary_df[summary_df["policy"].str.contains("fixed")].sort_values("b").copy()

plt.figure(figsize=(6,4))
plt.plot(fixed_only["b"], fixed_only["throughput_per_s"], marker="o")
plt.axhline(float(dyn_summary["throughput_per_s"]), linestyle="--")
plt.xlabel("Fixed batching token target b")
plt.ylabel("Throughput (completed / s)")
plt.title("Throughput vs fixed batch token target (dashed = dynamic)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(fixed_only["b"], fixed_only["p95_completion_s"], marker="o")
plt.axhline(float(dyn_summary["p95_completion_s"]), linestyle="--")
plt.xlabel("Fixed batching token target b")
plt.ylabel("P95 completion latency (s)")
plt.title("Tail completion latency vs fixed b (dashed = dynamic)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(fixed_only["b"], fixed_only["p95_ttft_s"], marker="o")
plt.axhline(float(dyn_summary["p95_ttft_s"]), linestyle="--")
plt.xlabel("Fixed batching token target b")
plt.ylabel("P95 TTFT (s)")
plt.title("P95 TTFT vs fixed b (dashed = dynamic)")
plt.tight_layout()
plt.show()

print("\nSummary (including dynamic):")
print(summary_df[[
    "policy","b","throughput_per_s","p95_ttft_s","p95_completion_s","p95_tbt_s","utilization"
]].sort_values(["policy","b"], na_position="last").to_string(index=False))
