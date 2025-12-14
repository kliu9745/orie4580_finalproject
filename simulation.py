import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SIM_TIME = 200.0 # seconds
# LAMBDA = 15  # arrival rate (queries/sec)
# B = 16           # output tokens Bi (fixed)

# # service model parameters (given in ms, converted to seconds)
# c = 20e-3      # setup cost in seconds
# a = 0.30e-3      # per-token marginal cost in seconds
# b0 = 64          # token threshold

rng = np.random.default_rng(12345)

def service_time_batch(token_count, a, c, b0):
    """
    Batch service time S(b) = c + a * max(0, b - b0)
    token_count is total tokens in the batch.
    """
    A = rng.exponential(a, size = 1)[0]
    return c + A * max(0, token_count - b0)
    # return A * token_count + c

# For decode we model sequential token generation without paying c for each token.
# We use a simple linear per-token decode time of 'a' seconds per token.
def decode_time_tokens(num_tokens, a):
    decode_token_times = rng.exponential(a, size=num_tokens)
    return decode_token_times

def simulation_priorize_decode(SIM_TIME=200, LAMBDA=15, B=16, c=20e-3, a=0.3e-3, b0=64): # no batching
    arrival_times = []
    t = 0.0
    while t < SIM_TIME:
        t += rng.exponential(1.0 / LAMBDA)
        if t < SIM_TIME:
            arrival_times.append(t)

    N = len(arrival_times)
    prompt_lengths = 20

    jobs = pd.DataFrame({
        "arrival": arrival_times,
        "L": prompt_lengths,
        "B": [B] * N
    })

    records = []
    gpu_free_time = 0.0
    per_token_tbt = []
    for i, row in jobs.iterrows():
        arrival = row["arrival"]
        L_i = int(row["L"])
        B_i = int(row["B"])

        # wait until arrival and GPU free
        prefill_start = max(gpu_free_time, arrival)
        # PREFILL should process the prompt tokens only (L_i)
        prefill_tokens = L_i
        prefill_S = service_time_batch(prefill_tokens, a, c, b0)
        prefill_end = prefill_start + prefill_S
        # TTFT is the time from arrival until the first output token is produced
        # Under this simple model, that occurs at the end of prefill
        ttft = prefill_end - arrival
        gpu_free_time = prefill_end

        # DECODE: sequentially produce B_i tokens, modeled as linear per-token cost
        decode_start = gpu_free_time
        decode_S = decode_time_tokens(B_i, a)
        per_token_tbt.extend(decode_S)

        decode_S = decode_S.sum() + c
        decode_end = decode_start + decode_S
        gpu_free_time = decode_end

        completion = decode_end - arrival

        records.append({
            "job": i,
            "arrival": arrival,
            "L": L_i,
            "B": B_i,
            "prefill_start": prefill_start,
            "prefill_end": prefill_end,
            "prefill_service": prefill_S,
            "decode_start": decode_start,
            "decode_end": decode_end,
            "decode_service": decode_S,
            "ttft": ttft,
            "completion": completion,
            "phase": "fcfs_fixed"
        })

    results = pd.DataFrame(records).sort_values("job").reset_index(drop=True)
    return results, per_token_tbt

results, per_token_tbt = simulation_priorize_decode()
mean_ttft = results["ttft"].mean()
p95_ttft = results["ttft"].quantile(0.95)
mean_comp = results["completion"].mean()
p95_comp = results["completion"].quantile(0.95)

# TBT: average time between tokens approximated as (completion - ttft) / B
results["tbt"] = (results["completion"] - results["ttft"]) / results["B"]
mean_tbt = results["tbt"].mean()
p95_tbt = results["tbt"].quantile(0.95)

# Utilization: fraction of time GPU was busy between first start and last end
total_service = results["prefill_service"].sum() + results["decode_service"].sum()
min_start = results["prefill_start"].min()
max_end = results["decode_end"].max()
actual_sim_time = max_end - min_start if max_end > min_start else 1.0
utilization = total_service / actual_sim_time

print("=" * 60)
print("FCFS (Process-to-completion) Scheduling")
print("=" * 60)
print(f"Jobs simulated: {len(results)}")
print(f"Mean TTFT: {mean_ttft:.4f} s     P95 TTFT: {p95_ttft:.4f} s")
print(f"Mean completion: {mean_comp:.4f} s  P95 completion: {p95_comp:.4f} s")
print(f"Mean TBT: {mean_tbt:.6f} s/token  P95 TBT: {p95_tbt:.6f} s/token")
print("=" * 60)

# TTFT CDF
sorted_ttft = np.sort(results["ttft"])
cdf = np.linspace(0, 1, len(sorted_ttft), endpoint=False)
plt.figure(figsize=(6,4))
plt.plot(sorted_ttft, cdf, linewidth=2)
plt.xlabel("TTFT (s)")
plt.ylabel("CDF")
plt.title("TTFT CDF (FCFS)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# TBT plot
plt.figure(figsize=(6,4))

plt.plot(
    np.sort(per_token_tbt),
    np.linspace(0, 1, len(per_token_tbt)),
    label="TBT"
)

plt.xlabel("Time Between Tokens (seconds)")
plt.ylabel("CDF")
plt.title("Time Between Tokens (TBT) CDF")
plt.grid(True)
plt.show()

def simulation_priorize_prefill(SIM_TIME=200, LAMBDA=15, B=16, K=128, c=20e-3, a=0.3e-3, b0=64):
    arrival_times = []
    t = 0.0
    while t < SIM_TIME:
        t += rng.exponential(1.0 / LAMBDA)
        if t < SIM_TIME:
            arrival_times.append(t)

    N = len(arrival_times)
    prompt_lengths = 20

    jobs = pd.DataFrame({
        "arrival": arrival_times,
        "L": prompt_lengths,
        "B": B
    }).sort_values("arrival").reset_index(drop=True)

    records = []
    gpu_free_time = 0.0
    per_token_tbt = []

    batch = []
    batch_token_sum = 0

    def flush_batch(batch, batch_token_sum, gpu_free_time):
        """Process a full prefill batch; return updated gpu_free_time and per-job records."""
        if len(batch) == 0:
            return gpu_free_time, []

        start_time = max(gpu_free_time, batch[-1]["arrival"])
        S = service_time_batch(batch_token_sum, a, c, b0)
        prefill_end = start_time + S

        batch_records = []
        new_gpu_time = prefill_end

        # Each job then decodes individually (sequential FCFS per batch)
        for job in batch:
            arrival = job["arrival"]
            L_i = job["L"]
            B_i = job["B"]

            ttft = prefill_end - arrival

            decode_start = new_gpu_time
            token_times = decode_time_tokens(B_i, a)
            per_token_tbt.extend(token_times)
            decode_S = token_times.sum() + c

            decode_end = decode_start + decode_S
            completion = decode_end - arrival

            new_gpu_time = decode_end

            batch_records.append({
                "job": job["job"],
                "arrival": arrival,
                "L": L_i,
                "B": B_i,
                "prefill_start": start_time,
                "prefill_end": prefill_end,
                "prefill_service": S,
                "decode_start": decode_start,
                "decode_end": decode_end,
                "decode_service": decode_S,
                "ttft": ttft,
                "completion": completion,
                "phase": "prefill_batching"
            })

        return new_gpu_time, batch_records


    # Build batches in arrival order
    for idx, row in jobs.iterrows():
        L_i = int(row["L"])

        # If we can add to batch, do it
        if batch_token_sum + L_i <= K:
            batch.append({
                "job": idx,
                "arrival": row["arrival"],
                "L": L_i,
                "B": B
            })
            batch_token_sum += L_i
        else:
            # Flush previous batch
            gpu_free_time, recs = flush_batch(batch, batch_token_sum, gpu_free_time)
            records.extend(recs)

            # Start new batch
            batch = [{
                "job": idx,
                "arrival": row["arrival"],
                "L": L_i,
                "B": B
            }]
            batch_token_sum = L_i

    # Flush final batch
    gpu_free_time, recs = flush_batch(batch, batch_token_sum, gpu_free_time)
    records.extend(recs)

    results = pd.DataFrame(records).sort_values("job").reset_index(drop=True)
    return results, per_token_tbt

results, per_token_tbt = simulation_priorize_prefill()
mean_ttft = results["ttft"].mean()
p95_ttft = results["ttft"].quantile(0.95)
mean_comp = results["completion"].mean()
p95_comp = results["completion"].quantile(0.95)

results["tbt"] = (results["completion"] - results["ttft"]) / results["B"]
mean_tbt = results["tbt"].mean()
p95_tbt = results["tbt"].quantile(0.95)

total_service = results["prefill_service"].sum() + results["decode_service"].sum()
min_start = results["prefill_start"].min()
max_end = results["decode_end"].max()
utilization = total_service / (max_end - min_start)

print("=" * 60)
print("Prefill Batching (Max Token Load K) - Simulation Results")
print("=" * 60)
print(f"Jobs: {len(results)}")
print(f"Batch token limit K: {128}")
print("-")
print(f"Mean TTFT: {mean_ttft:.4f}  |  P95 TTFT: {p95_ttft:.4f}  (seconds)")
print(f"Mean completion: {mean_comp:.4f}  |  P95 completion: {p95_comp:.4f}")
print(f"Mean TBT: {mean_tbt:.6f} s/token   |   P95 TBT: {p95_tbt:.6f} s/token")
print("=" * 60)

# TBT CDF
plt.figure(figsize=(6,4))
plt.plot(np.sort(per_token_tbt), np.linspace(0,1,len(per_token_tbt)), label="TBT")
plt.xlabel("Time Between Tokens (seconds)")
plt.ylabel("CDF")
plt.title("TBT CDF (Prefill Batching)")
plt.grid(True)
plt.tight_layout()
plt.show()

# TTFT CDF
sorted_ttft = np.sort(results["ttft"])
cdf = np.linspace(0,1,len(sorted_ttft), endpoint=False)
plt.figure(figsize=(6,4))
plt.plot(sorted_ttft, cdf, linewidth=2)
plt.xlabel("TTFT (s)")
plt.ylabel("CDF")
plt.title("TTFT CDF (Prefill Batching)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
