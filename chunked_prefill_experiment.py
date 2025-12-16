import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)

def service_time_batch(token_count, a, c, b0):
    """
    Batch service time S(b) = c + a * max(0, b - b0)
    token_count is total tokens in the batch.
    """
    A = rng.exponential(a, size = 1)[0]
    return c + A * max(0, token_count - b0)

def simulation_chunked_prefill(
    SIM_TIME=200,
    LAMBDA=15,
    B=16,
    K=128,            # token budget per GPU iteration (same meaning as your prefill batching code)
    chunk_size=32,    # NEW: max prefill tokens taken from ONE query per prefill iteration
    c=20e-3,
    a=0.3e-3,
    b0=64,
    prompt_lengths=20,  # keep fixed to match your current code (you can change later)
):
    """
    Chunked Prefill (single GPU, iteration-level):
    - Prefill work is broken into chunks of size <= chunk_size per query per prefill iteration.
    - GPU runs either:
        (A) a PREFILL iteration batch (token load <= K) OR
        (B) a DECODE iteration batch (1 token per query, up to K queries).
    Scheduling rule (simple + matches your project text):
      - If prefill backlog >= K tokens, run prefill.
      - Otherwise, if decode work exists, run decode.
      - If only prefill exists (even if < K), run prefill.
    Metrics:
      - TTFT = time(first decoded token completes) - arrival
      - per_token_tbt = actual gaps between successive decoded tokens (good for TBT CDF)
      - decode_stall_time = total time decode was waiting while GPU executed prefill
    """

    # arrivals
    arrival_times = []
    t = 0.0
    while t < SIM_TIME:
        t += rng.exponential(1.0 / LAMBDA)
        if t < SIM_TIME:
            arrival_times.append(t)

    N = len(arrival_times)
    jobs = pd.DataFrame({
        "job": np.arange(N),
        "arrival": arrival_times,
        "L": [prompt_lengths] * N,
        "B": [B] * N
    }).sort_values("arrival").reset_index(drop=True)

    # per-job state
    rem_prefill = jobs["L"].to_numpy().astype(int)   # remaining prefill tokens
    rem_decode  = jobs["B"].to_numpy().astype(int)   # remaining decode tokens

    token_times = [[] for _ in range(N)]             # decoded token completion times per job
    first_token_time = np.full(N, np.nan)
    completion_time  = np.full(N, np.nan)

    # queues hold job indices in arrival order
    prefill_q = []
    decode_q = []
    next_idx = 0

    # utilization bookkeeping
    idle_time = 0.0
    decode_stall_time = 0.0

    # per-token TBT samples = actual gaps between consecutive tokens within each job
    per_token_tbt = []

    # helper: push arrivals into prefill_q
    def push_arrivals(up_to_t):
        nonlocal next_idx
        while next_idx < N and jobs.loc[next_idx, "arrival"] <= up_to_t + 1e-12:
            prefill_q.append(next_idx)
            next_idx += 1

    t = 0.0
    while True:
        push_arrivals(t)

        # If nothing to do, jump to next arrival (GPU idle)
        if not prefill_q and not decode_q:
            if next_idx < N:
                next_arrival = jobs.loc[next_idx, "arrival"]
                if next_arrival > t:
                    idle_time += (next_arrival - t)
                    t = next_arrival
                continue
            else:
                break

        prefill_backlog = int(rem_prefill[prefill_q].sum()) if prefill_q else 0

        # Decide whether to run PREFILL or DECODE
        if prefill_q and prefill_backlog >= K:
            run_prefill = True
        elif decode_q:
            run_prefill = False
        else:
            run_prefill = True  # only prefill exists


        # PREFILL ITERATION
        if run_prefill:
            # If decode work exists, this prefill batch is stalling decode
            stalling_decode = (len(decode_q) > 0)

            budget = K
            batch = []  # list of (job_idx, tokens_taken)
            for j in prefill_q:
                if budget <= 0:
                    break
                take = min(chunk_size, rem_prefill[j], budget)
                if take > 0:
                    batch.append((j, take))
                    budget -= take

            token_load = sum(take for _, take in batch)
            if token_load <= 0:
                # no progress possible (shouldn't happen)
                break

            S = service_time_batch(token_load, a, c, b0)
            if stalling_decode:
                decode_stall_time += S
            t += S

            # apply prefill progress and move completed-prefill jobs to decode_q
            new_prefill_q = []
            taken_map = {j: take for j, take in batch}
            for j in prefill_q:
                rem_prefill[j] -= taken_map.get(j, 0)
                if rem_prefill[j] <= 0:
                    decode_q.append(j)
                else:
                    new_prefill_q.append(j)
            prefill_q = new_prefill_q

        # DECODE ITERATION (batched)
        else:
            m = min(len(decode_q), K)   # each query contributes 1 token
            active = decode_q[:m]
            decode_q = decode_q[m:]

            token_load = len(active)
            S = service_time_batch(token_load, a, c, b0)
            t += S

            for j in active:
                # decoded token completes at time t
                token_times[j].append(t)
                k = len(token_times[j])

                if k == 1:
                    first_token_time[j] = t
                else:
                    gap = token_times[j][-1] - token_times[j][-2]
                    per_token_tbt.append(gap)

                rem_decode[j] -= 1
                if rem_decode[j] <= 0:
                    completion_time[j] = t
                else:
                    decode_q.append(j)  # round-robin

        # stop when all arrived and all done
        if next_idx >= N and not prefill_q and not decode_q:
            break

    # build results --------
    records = []
    for idx in range(N):
        if np.isnan(first_token_time[idx]) or np.isnan(completion_time[idx]):
            continue

        arrival = jobs.loc[idx, "arrival"]
        L_i = int(jobs.loc[idx, "L"])
        B_i = int(jobs.loc[idx, "B"])

        ttft = first_token_time[idx] - arrival
        completion = completion_time[idx] - arrival

        gaps = np.diff(token_times[idx]) if len(token_times[idx]) >= 2 else np.array([])
        tbt_mean = float(gaps.mean()) if len(gaps) else np.nan

        records.append({
            "job": idx,
            "arrival": arrival,
            "L": L_i,
            "B": B_i,
            "ttft": ttft,
            "completion": completion,
            "tbt_mean": tbt_mean,
            "first_token_time": first_token_time[idx],
            "completion_time": completion_time[idx],
            "phase": f"chunked_prefill_{chunk_size}",
        })

    results = pd.DataFrame(records).sort_values("job").reset_index(drop=True)

    makespan = results["completion_time"].max() if len(results) else SIM_TIME
    throughput = len(results) / makespan if makespan > 0 else 0.0
    utilization = 1.0 - (idle_time / makespan) if makespan > 0 else 0.0

    return results, per_token_tbt, decode_stall_time, throughput, utilization

chunk_sizes = [256, 128, 64, 32, 16, 8]

rows = []
for cs in chunk_sizes:
    res, tbt_samples, stall, thr, util = simulation_chunked_prefill(
        SIM_TIME=200, LAMBDA=15, B=16,
        K=128, chunk_size=cs,
        c=20e-3, a=0.3e-3, b0=64,
        prompt_lengths=20
    )

    rows.append({
        "chunk_size": cs,
        "jobs": len(res),
        "throughput": thr,
        "utilization": util,
        "mean_ttft": res["ttft"].mean(),
        "p95_ttft": res["ttft"].quantile(0.95),
        "mean_completion": res["completion"].mean(),
        "p95_completion": res["completion"].quantile(0.95),
        "mean_tbt": np.mean(tbt_samples) if len(tbt_samples) else np.nan,
        "p95_tbt": np.quantile(tbt_samples, 0.95) if len(tbt_samples) else np.nan,
        "decode_stall_time": stall
    })

sweep = pd.DataFrame(rows).sort_values("chunk_size", ascending=False)
print(sweep)

# Plots
plt.figure(figsize=(6,4))
plt.plot(sweep["chunk_size"], sweep["mean_ttft"], marker="o")
plt.gca().invert_xaxis()
plt.xlabel("chunk size (tokens)")
plt.ylabel("mean TTFT (s)")
plt.title("Chunked-prefill sensitivity: TTFT vs chunk size")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(sweep["chunk_size"], sweep["throughput"], marker="o")
plt.gca().invert_xaxis()
plt.xlabel("chunk size (tokens)")
plt.ylabel("throughput (jobs/s)")
plt.title("Chunked-prefill sensitivity: throughput vs chunk size")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(sweep["chunk_size"], sweep["decode_stall_time"], marker="o")
plt.gca().invert_xaxis()
plt.xlabel("chunk size (tokens)")
plt.ylabel("decode stall time (s)")
plt.title("Chunked-prefill sensitivity: decode stalls vs chunk size")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
