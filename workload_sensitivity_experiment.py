import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)

def service_time_batch(token_count, a, c, b0):
    """
    Batch service time:
        S(b) = c + A * max(0, b - b0)
    where A ~ Exp(a)
    """
    A = rng.exponential(a)
    return c + A * max(0, token_count - b0)

def decode_time_tokens(num_tokens, a):
    """
    Sequential decode token times (no setup cost).
    """
    return rng.exponential(a, size=num_tokens)

def simulation_priorize_decode(
    SIM_TIME=200,
    LAMBDA=15,
    L=20,
    B=16,
    c=20e-3,
    a=0.3e-3,
    b0=64
):
    # Generate arrivals
    arrival_times = []
    t = 0.0
    while t < SIM_TIME:
        t += rng.exponential(1.0 / LAMBDA)
        if t < SIM_TIME:
            arrival_times.append(t)

    jobs = pd.DataFrame({
        "arrival": arrival_times,
        "L": L,
        "B": B
    })

    records = []
    gpu_free_time = 0.0
    per_token_tbt = []

    for i, row in jobs.iterrows():
        arrival = row["arrival"]

        # PREFILL
        prefill_start = max(gpu_free_time, arrival)
        prefill_S = service_time_batch(L, a, c, b0)
        prefill_end = prefill_start + prefill_S
        ttft = prefill_end - arrival
        gpu_free_time = prefill_end

        # DECODE
        decode_start = gpu_free_time
        token_times = decode_time_tokens(B, a)
        per_token_tbt.extend(token_times)
        decode_S = token_times.sum() + c
        decode_end = decode_start + decode_S
        gpu_free_time = decode_end

        completion = decode_end - arrival

        records.append({
            "arrival": arrival,
            "ttft": ttft,
            "completion": completion,
            "prefill_service": prefill_S,
            "decode_service": decode_S
        })

    return pd.DataFrame(records), per_token_tbt

def simulation_priorize_prefill(
    SIM_TIME=200,
    LAMBDA=15,
    L=20,
    B=16,
    K=128,
    c=20e-3,
    a=0.3e-3,
    b0=64
):
    # Generate arrivals
    arrival_times = []
    t = 0.0
    while t < SIM_TIME:
        t += rng.exponential(1.0 / LAMBDA)
        if t < SIM_TIME:
            arrival_times.append(t)

    jobs = pd.DataFrame({
        "arrival": arrival_times,
        "L": L,
        "B": B
    }).sort_values("arrival").reset_index(drop=True)

    records = []
    gpu_free_time = 0.0
    per_token_tbt = []

    batch = []
    batch_tokens = 0

    def flush_batch(batch, batch_tokens, gpu_free_time):
        start = max(gpu_free_time, batch[-1]["arrival"])
        S = service_time_batch(batch_tokens, a, c, b0)
        prefill_end = start + S

        new_gpu_time = prefill_end
        recs = []

        for job in batch:
            arrival = job["arrival"]
            ttft = prefill_end - arrival

            decode_start = new_gpu_time
            token_times = decode_time_tokens(B, a)
            per_token_tbt.extend(token_times)
            decode_S = token_times.sum() + c
            decode_end = decode_start + decode_S
            new_gpu_time = decode_end

            completion = decode_end - arrival

            recs.append({
                "arrival": arrival,
                "ttft": ttft,
                "completion": completion,
                "prefill_service": S,
                "decode_service": decode_S
            })

        return new_gpu_time, recs

    for _, row in jobs.iterrows():
        if batch_tokens + L <= K:
            batch.append(row)
            batch_tokens += L
        else:
            gpu_free_time, recs = flush_batch(batch, batch_tokens, gpu_free_time)
            records.extend(recs)
            batch = [row]
            batch_tokens = L

    if batch:
        gpu_free_time, recs = flush_batch(batch, batch_tokens, gpu_free_time)
        records.extend(recs)

    return pd.DataFrame(records), per_token_tbt

def summarize(results, per_token_tbt):
    return {
        "mean_ttft": results["ttft"].mean(),
        "p95_ttft": results["ttft"].quantile(0.95),
        "mean_completion": results["completion"].mean(),
        "p95_completion": results["completion"].quantile(0.95),
        "mean_tbt": ((results["completion"] - results["ttft"]) / 16).mean(),
        "p95_tbt": ((results["completion"] - results["ttft"]) / 16).quantile(0.95),
        "p95_token_tbt": np.quantile(per_token_tbt, 0.95)
    }

arrival_rates = [5, 10, 15, 20, 25]
prompt_lengths = [8, 20, 64, 128]
output_budgets = [8, 16, 32]

records = []

for lam in arrival_rates:
    for L in prompt_lengths:
        for B in output_budgets:

            # FCFS
            res, tbt = simulation_priorize_decode(
                LAMBDA=lam, L=L, B=B
            )
            summary = summarize(res, tbt)
            records.append({
                "scheduler": "fcfs",
                "lambda": lam,
                "L": L,
                "B": B,
                **summary
            })

            # Prefill batching
            res, tbt = simulation_priorize_prefill(
                LAMBDA=lam, L=L, B=B
            )
            summary = summarize(res, tbt)
            records.append({
                "scheduler": "prefill",
                "lambda": lam,
                "L": L,
                "B": B,
                **summary
            })

df = pd.DataFrame(records)

# P95 TTFT vs arrival rate
plt.figure(figsize=(7,5))
for sched in ["fcfs", "prefill"]:
    sub = df[(df.scheduler == sched) & (df.L == 20) & (df.B == 16)]
    plt.plot(sub["lambda"], sub["p95_ttft"], marker="o", label=sched)
plt.xlabel("Arrival rate λ (qps)")
plt.ylabel("P95 TTFT (s)")
plt.title("Tail TTFT vs Arrival Rate")
plt.legend()
plt.grid(True)
plt.show()

# P95 TTFT vs prompt length
plt.figure(figsize=(7,5))
for sched in ["fcfs", "prefill"]:
    sub = df[(df["scheduler"] == sched) & (df["lambda"] == 15) & (df["B"] == 16)]
    plt.plot(sub["L"], sub["p95_ttft"], marker="o", label=sched)
plt.xlabel("Prompt length L (tokens)")
plt.ylabel("P95 TTFT (s)")
plt.title("Tail TTFT vs Prompt Length")
plt.legend()
plt.grid(True)
plt.show()

# P95 TBT vs output budget
plt.figure(figsize=(7,5))
for sched in ["fcfs", "prefill"]:
    sub = df[(df["scheduler"] == sched) & (df["lambda"] == 15) & (df["L"] == 20)]
    plt.plot(sub["B"], sub["p95_tbt"], marker="o", label=sched)
plt.xlabel("Output budget B (tokens)")
plt.ylabel("P95 TBT (s/token)")
plt.title("Tail TBT vs Output Length")
plt.legend()
plt.grid(True)
plt.show()
