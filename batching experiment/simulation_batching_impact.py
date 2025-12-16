
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

# Extending and Adding on simulation.py:
# simulation_fixed_size_batching (accumulate b tokens or timeout)
# simulation_dynamic_batching (queue-depth target + latency guard)
# This file intentionally does NOT run simulations at import time.

rng = np.random.default_rng(12345)

def service_time_batch(token_count: int, a: float, c: float, b0: int) -> float:
    """Batch service time: S(b) = c + A * max(0, b - b0), where A ~ Exp(mean=a)."""
    A = rng.exponential(a, size=1)[0]
    return c + A * max(0, int(token_count) - int(b0))

def decode_time_tokens(num_tokens: int, a: float) -> np.ndarray:
    """Sample per-token decode times (i.i.d. Exp(mean=a))."""
    return rng.exponential(a, size=int(num_tokens))

# Existing baseline strategies
def simulation_priorize_decode(
    SIM_TIME: float = 200,
    LAMBDA: float = 15,
    B: int = 16,
    c: float = 20e-3,
    a: float = 0.3e-3,
    b0: int = 64,
    prompt_length: int = 20,
):
    """FCFS / process-to-completion baseline."""
    arrival_times = []
    t = 0.0
    while t < SIM_TIME:
        t += rng.exponential(1.0 / LAMBDA)
        if t < SIM_TIME:
            arrival_times.append(t)

    N = len(arrival_times)

    jobs = pd.DataFrame({
        "arrival": arrival_times,
        "L": [prompt_length] * N,
        "B": [B] * N
    })

    records = []
    gpu_free_time = 0.0
    per_token_tbt = []

    for i, row in jobs.iterrows():
        arrival = float(row["arrival"])
        L_i = int(row["L"])
        B_i = int(row["B"])

        prefill_start = max(gpu_free_time, arrival)
        prefill_S = service_time_batch(L_i, a, c, b0)
        prefill_end = prefill_start + prefill_S
        ttft = prefill_end - arrival
        gpu_free_time = prefill_end

        decode_start = gpu_free_time
        token_times = decode_time_tokens(B_i, a)
        per_token_tbt.extend(token_times.tolist())
        decode_S = float(token_times.sum()) + c
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


def simulation_priorize_prefill(
    SIM_TIME: float = 200,
    LAMBDA: float = 15,
    B: int = 16,
    K: int = 128,
    c: float = 20e-3,
    a: float = 0.3e-3,
    b0: int = 64,
    prompt_length: int = 20,
):
    """Prefill batching baseline """
    arrival_times = []
    t = 0.0
    while t < SIM_TIME:
        t += rng.exponential(1.0 / LAMBDA)
        if t < SIM_TIME:
            arrival_times.append(t)

    N = len(arrival_times)

    jobs = pd.DataFrame({
        "arrival": arrival_times,
        "L": [prompt_length] * N,
        "B": [B] * N
    }).sort_values("arrival").reset_index(drop=True)

    records = []
    gpu_free_time = 0.0
    per_token_tbt = []

    batch = []
    batch_token_sum = 0

    def flush_batch(batch, batch_token_sum, gpu_free_time):
        if len(batch) == 0:
            return gpu_free_time, []

        start_time = max(gpu_free_time, batch[-1]["arrival"])
        S = service_time_batch(batch_token_sum, a, c, b0)
        prefill_end = start_time + S

        batch_records = []
        new_gpu_time = prefill_end

        for job in batch:
            arrival = float(job["arrival"])
            L_i = int(job["L"])
            B_i = int(job["B"])

            ttft = prefill_end - arrival

            decode_start = new_gpu_time
            token_times = decode_time_tokens(B_i, a)
            per_token_tbt.extend(token_times.tolist())
            decode_S = float(token_times.sum()) + c
            decode_end = decode_start + decode_S
            completion = decode_end - arrival
            new_gpu_time = decode_end

            batch_records.append({
                "job": int(job["job"]),
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

    for idx, row in jobs.iterrows():
        L_i = int(row["L"])
        if batch_token_sum + L_i <= K:
            batch.append({"job": idx, "arrival": float(row["arrival"]), "L": L_i, "B": int(row["B"])})
            batch_token_sum += L_i
        else:
            gpu_free_time, recs = flush_batch(batch, batch_token_sum, gpu_free_time)
            records.extend(recs)
            batch = [{"job": idx, "arrival": float(row["arrival"]), "L": L_i, "B": int(row["B"])}]
            batch_token_sum = L_i

    gpu_free_time, recs = flush_batch(batch, batch_token_sum, gpu_free_time)
    records.extend(recs)

    results = pd.DataFrame(records).sort_values("job").reset_index(drop=True)
    return results, per_token_tbt

# New: batching-impact simulator (fixed-size vs dynamic batching)

@dataclass
class _ReqState:
    rid: int
    arrival: float
    L: int
    B: int
    remaining_decode: int
    prefill_start: Optional[float] = None
    prefill_end: Optional[float] = None
    decode_start: Optional[float] = None
    completion_time: Optional[float] = None
    token_times: Optional[List[float]] = None
    prefill_service: float = 0.0
    decode_service: float = 0.0

    def __post_init__(self):
        if self.token_times is None:
            self.token_times = []

@dataclass(frozen=True)
class _Job:
    rid: int
    kind: str         
    tokens: int       
    created: float    
    token_index: int = 0  

def _generate_arrivals(SIM_TIME: float, LAMBDA: float) -> List[float]:
    arrivals = []
    t = 0.0
    while t < SIM_TIME:
        t += rng.exponential(1.0 / LAMBDA)
        if t < SIM_TIME:
            arrivals.append(t)
    return arrivals

def _simulate_token_batched(
    SIM_TIME: float,
    LAMBDA: float,
    B: int,
    b0: int,
    c: float,
    a: float,
    prompt_length: int,
    policy: str,
    # fixed policy
    b_target: int = 128,
    timeout_s: float = 0.05,
    # dynamic policy
    b_min: int = 32,
    b_mid: int = 64,
    b_max: int = 128,
    switch_mid: int = 64,
    switch_max: int = 256,
    latency_target_s: float = 0.08,
    # operational limits
    K_max_jobs: int = 256,
):
    """Discrete-event simulation where the queue holds token-sized jobs."""
    arrivals = _generate_arrivals(SIM_TIME, LAMBDA)
    N = len(arrivals)

    req: Dict[int, _ReqState] = {
        i: _ReqState(rid=i, arrival=arrivals[i], L=prompt_length, B=B, remaining_decode=B)
        for i in range(N)
    }

    next_arr = 0
    now = 0.0

    q: Deque[_Job] = deque()
    q_tokens = 0

    def push(job: _Job):
        nonlocal q_tokens
        q.append(job)
        q_tokens += int(job.tokens)

    def add_arrivals_up_to(t_now: float):
        nonlocal next_arr
        while next_arr < N and arrivals[next_arr] <= t_now:
            rid = next_arr
            push(_Job(rid=rid, kind="prefill", tokens=prompt_length, created=arrivals[rid], token_index=0))
            next_arr += 1

    gpu_busy_until = 0.0
    running_batch: List[_Job] = []
    batch_dispatch_time = 0.0
    busy_time = 0.0
    first_dispatch = None
    last_done = 0.0

    batch_log: List[Dict[str, float]] = []

    def choose_b_target() -> int:
        if policy == "fixed":
            return int(b_target)
        if q_tokens >= switch_max:
            return int(b_max)
        if q_tokens >= switch_mid:
            return int(b_mid)
        return int(b_min)

    def choose_deadline(oldest_created: float) -> float:
        if policy == "fixed":
            return oldest_created + float(timeout_s)
        return oldest_created + float(latency_target_s)

    def pop_batch(b_lim: int) -> Tuple[List[_Job], int]:
        nonlocal q_tokens
        if not q:
            return [], 0
        batch: List[_Job] = []
        tok = 0

        jb = q.popleft()
        q_tokens -= int(jb.tokens)
        batch.append(jb)
        tok += int(jb.tokens)

        while q and len(batch) < K_max_jobs:
            nxt = q[0]
            if tok + int(nxt.tokens) <= b_lim:
                q.popleft()
                q_tokens -= int(nxt.tokens)
                batch.append(nxt)
                tok += int(nxt.tokens)
            else:
                break

        return batch, tok

    while True:
        if now < gpu_busy_until:
            now = gpu_busy_until
            last_done = now
            add_arrivals_up_to(now)

            # complete running batch
            for jb in running_batch:
                st = req[jb.rid]

                if jb.kind == "prefill":
                    if st.prefill_start is None:
                        st.prefill_start = batch_dispatch_time
                    st.prefill_end = now
                    st.token_times.append(now)  # token0

                    if st.remaining_decode > 0:
                        push(_Job(rid=jb.rid, kind="decode", tokens=1, created=now, token_index=1))

                else:
                    if jb.token_index == 1 and st.decode_start is None:
                        st.decode_start = batch_dispatch_time
                    st.token_times.append(now)
                    st.remaining_decode -= 1

                    if st.remaining_decode <= 0:
                        st.completion_time = now
                    else:
                        push(_Job(rid=jb.rid, kind="decode", tokens=1, created=now, token_index=jb.token_index + 1))

            running_batch = []
            continue

        add_arrivals_up_to(now)

        if not q and next_arr >= N:
            break

        if not q:
            if next_arr < N:
                now = arrivals[next_arr]
                add_arrivals_up_to(now)
                continue
            break

        oldest = q[0].created
        b_lim = choose_b_target()
        deadline = choose_deadline(oldest)

        if q_tokens >= b_lim or now >= deadline:
            batch, b_load = pop_batch(b_lim)

            batch_dispatch_time = now
            if first_dispatch is None:
                first_dispatch = now

            S = service_time_batch(b_load, a, c, b0)
            gpu_busy_until = now + S
            busy_time += S
            running_batch = batch

            if b_load > 0:
                for jb in batch:
                    share = S * (jb.tokens / b_load)
                    if jb.kind == "prefill":
                        req[jb.rid].prefill_service += share
                    else:
                        req[jb.rid].decode_service += share

            batch_log.append({
                "t_dispatch": now,
                "t_done": gpu_busy_until,
                "batch_jobs": len(batch),
                "batch_tokens": b_load,
                "service_s": S,
                "queued_tokens_after": q_tokens,
                "queued_len_after": len(q),
            })
            continue

        next_time = arrivals[next_arr] if next_arr < N else float("inf")
        now = min(next_time, deadline)

    rows = []
    per_token_tbt: List[float] = []

    for rid, st in req.items():
        ttft = np.nan if st.prefill_end is None else (st.prefill_end - st.arrival)
        completion = np.nan if st.completion_time is None else (st.completion_time - st.arrival)

        # realized token gaps (include queueing/batching)
        times = st.token_times
        if times and len(times) >= 2:
            gaps = [times[i] - times[i-1] for i in range(1, len(times))]
            per_token_tbt.extend(gaps)

        rows.append({
            "job": rid,
            "arrival": st.arrival,
            "L": st.L,
            "B": st.B,
            "prefill_start": st.prefill_start,
            "prefill_end": st.prefill_end,
            "decode_start": st.decode_start,
            "decode_end": st.completion_time,
            "prefill_service": st.prefill_service,
            "decode_service": st.decode_service,
            "ttft": ttft,
            "completion": completion,
            "phase": policy,
        })

    results = pd.DataFrame(rows).sort_values("job").reset_index(drop=True)
    batch_df = pd.DataFrame(batch_log)

    utilization = np.nan
    if first_dispatch is not None and last_done > first_dispatch:
        utilization = busy_time / (last_done - first_dispatch)

    return results, per_token_tbt, batch_df, utilization


def simulation_fixed_size_batching(
    SIM_TIME: float = 200,
    LAMBDA: float = 15,
    B: int = 16,
    b_target: int = 128,
    timeout_s: float = 0.05,
    c: float = 20e-3,
    a: float = 0.3e-3,
    b0: int = 64,
    prompt_length: int = 20,
    K_max_jobs: int = 256,
):
    """Fixed-size batching: dispatch when queued token load >= b_target or timeout."""
    return _simulate_token_batched(
        SIM_TIME=SIM_TIME, LAMBDA=LAMBDA, B=B, b0=b0, c=c, a=a,
        prompt_length=prompt_length,
        policy="fixed",
        b_target=b_target, timeout_s=timeout_s,
        K_max_jobs=K_max_jobs,
    )


def simulation_dynamic_batching(
    SIM_TIME: float = 200,
    LAMBDA: float = 15,
    B: int = 16,
    c: float = 20e-3,
    a: float = 0.3e-3,
    b0: int = 64,
    prompt_length: int = 20,
    b_min: int = 32,
    b_mid: int = 64,
    b_max: int = 128,
    switch_mid: int = 64,
    switch_max: int = 256,
    latency_target_s: float = 0.08,
    K_max_jobs: int = 256,
):
    """Dynamic batching: queue-depth batch target + latency guard."""
    return _simulate_token_batched(
        SIM_TIME=SIM_TIME, LAMBDA=LAMBDA, B=B, b0=b0, c=c, a=a,
        prompt_length=prompt_length,
        policy="dynamic",
        b_min=b_min, b_mid=b_mid, b_max=b_max,
        switch_mid=switch_mid, switch_max=switch_max,
        latency_target_s=latency_target_s,
        K_max_jobs=K_max_jobs,
    )


if __name__ == "__main__":
    res_f, tbt_f, batch_f, util_f = simulation_fixed_size_batching(b_target=128)
    res_d, tbt_d, batch_d, util_d = simulation_dynamic_batching()
    print("Fixed b=128 utilization:", util_f)
    print("Dynamic utilization:", util_d)
