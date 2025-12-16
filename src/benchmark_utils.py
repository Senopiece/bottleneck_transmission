import math
import multiprocessing as mp
import queue
import random
import traceback
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

from impls._interface import Producer, Recoverer


def _default_sample_burst_size() -> int:
    return random.randint(1, 5)


def _default_sample_data_size() -> int:
    return random.randint(3, 30)


def _run_benchmark_chunk(
    producer_constructor: Callable[[int, int, int], Producer],
    recoverer_constructor: Callable[[int, int], Recoverer],
    N: int,
    D: int,
    passes: int,
    sample_burst_size: Callable[[], int],
    sample_data_size: Callable[[], int],
    iters_bound: int,
    progress_reporter: Callable[[int], None] | None = None,
    progress_step: int = 100,
):
    """Run a benchmark slice and optionally report progress via `progress_reporter`."""
    time_to_recover_distribution = defaultdict(lambda: 0)
    burst_sum = 0
    burst_count = 0
    data_size_sum = 0
    data_size_count = 0
    failures = 0
    mismatch = 0

    processed = 0
    reported = 0
    step = max(1, progress_step)

    for _ in range(passes):
        processed += 1

        # Make random binary data matrix
        data = random.randint(0, D - 1)
        recovered = None

        # Initialize producer and recoverer
        producer = producer_constructor(data, N, D)
        recoverer = recoverer_constructor(N, D)
        time_to_recover = 0

        for _ in range(iters_bound):
            # Skip samples
            burst_size = sample_burst_size()
            burst_sum += burst_size
            burst_count += 1
            time_to_recover += burst_size
            for _ in range(burst_size):
                producer.generate()

            # Use samples
            data_size = sample_data_size()
            data_size_sum += data_size
            data_size_count += 1
            for _ in range(data_size):
                sample = producer.generate()
                assert (
                    sample.shape == (N,)
                    and sample.dtype == np.uint8
                    and np.all((sample <= 1))
                ), "Generated sample has invalid format"
                time_to_recover += 1
                recovered = recoverer.feed(sample)
                if recovered is not None:
                    break

            if recovered is not None:
                break

            recoverer.feed(None)  # Indicate end of continuity

        if recovered is None:
            failures += 1
        elif not np.array_equal(recovered, data):
            mismatch += 1
        else:
            time_to_recover_distribution[time_to_recover] += 1

        if progress_reporter and (processed - reported >= step):
            delta = processed - reported
            progress_reporter(delta)
            reported = processed

    if progress_reporter and processed > reported:
        progress_reporter(processed - reported)

    return {
        "distribution_counts": dict(time_to_recover_distribution),
        "burst_sum": burst_sum,
        "burst_count": burst_count,
        "data_size_sum": data_size_sum,
        "data_size_count": data_size_count,
        "failures": failures,
        "mismatches": mismatch,
        "successes": sum(time_to_recover_distribution.values()),
    }


def _worker_entry(
    producer_constructor: Callable[[int, int, int], Producer],
    recoverer_constructor: Callable[[int, int], Recoverer],
    N: int,
    D: int,
    passes: int,
    sample_burst_size: Callable[[], int],
    sample_data_size: Callable[[], int],
    iters_bound: int,
    progress_queue: mp.Queue,
    progress_step: int,
    result_queue: mp.Queue,
    error_queue: mp.Queue,
):
    def report(delta: int):
        progress_queue.put(delta)

    try:
        result = _run_benchmark_chunk(
            producer_constructor,
            recoverer_constructor,
            N,
            D,
            passes,
            sample_burst_size,
            sample_data_size,
            iters_bound,
            report,
            progress_step,
        )
    except Exception:
        # Surface worker failures to the master so it can halt everything immediately.
        error_queue.put(traceback.format_exc())
        return

    progress_queue.put(None)
    result_queue.put(result)


def _finalize_results(worker_results: list[dict]):
    merged_distribution = defaultdict(int)
    burst_sum = 0
    burst_count = 0
    data_size_sum = 0
    data_size_count = 0
    failures = 0
    mismatches = 0

    for res in worker_results:
        for k, v in res["distribution_counts"].items():
            merged_distribution[k] += v
        burst_sum += res["burst_sum"]
        burst_count += res["burst_count"]
        data_size_sum += res["data_size_sum"]
        data_size_count += res["data_size_count"]
        failures += res["failures"]
        mismatches += res["mismatches"]

    successes = sum(merged_distribution.values())
    if failures:
        print(f"{failures} failures.")
    if mismatches:
        print(f"{mismatches} MISMATCHES!!!")

    expected_burst_size = burst_sum / burst_count if burst_count > 0 else float("nan")
    expected_data_size = (
        data_size_sum / data_size_count if data_size_count > 0 else float("nan")
    )

    if successes == 0:
        distribution = {}
    else:
        distribution = {k: v / successes for k, v in merged_distribution.items()}

    return {
        "distribution": distribution,
        "expected_sample_burst_size": expected_burst_size,
        "expected_sample_data_size": expected_data_size,
    }


def benchmark(
    producer_constructor: Callable[[int, int, int], Producer],
    recoverer_constructor: Callable[[int, int], Recoverer],
    N: int,
    D: int,
    passes: int = 100,
    sample_burst_size: Callable[[], int] = _default_sample_burst_size,
    sample_data_size: Callable[[], int] = _default_sample_data_size,
    iters_bound: int = 10000,
    processes: int | None = None,
    progress_update: int = 100,
):
    total_passes = max(0, int(passes))
    if total_passes == 0:
        return _finalize_results(
            [
                {
                    "distribution_counts": {},
                    "burst_sum": 0,
                    "burst_count": 0,
                    "data_size_sum": 0,
                    "data_size_count": 0,
                    "failures": 0,
                    "mismatches": 0,
                    "successes": 0,
                }
            ]
        )

    if processes is None:
        processes = min(mp.cpu_count() or 1, total_passes)
    processes = max(1, min(int(processes), total_passes))

    if processes == 1:
        with tqdm(total=total_passes, desc="benchmark") as pbar:
            result = _run_benchmark_chunk(
                producer_constructor,
                recoverer_constructor,
                N,
                D,
                total_passes,
                sample_burst_size,
                sample_data_size,
                iters_bound,
                pbar.update,
                progress_update,
            )
        return _finalize_results([result])

    ctx = mp.get_context("spawn")
    progress_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    error_queue: mp.Queue = ctx.Queue()

    base, remainder = divmod(total_passes, processes)
    passes_per_worker = [base + (1 if i < remainder else 0) for i in range(processes)]
    passes_per_worker = [p for p in passes_per_worker if p > 0]

    procs = []
    for worker_passes in passes_per_worker:
        p = ctx.Process(
            target=_worker_entry,
            args=(
                producer_constructor,
                recoverer_constructor,
                N,
                D,
                worker_passes,
                sample_burst_size,
                sample_data_size,
                iters_bound,
                progress_queue,
                progress_update,
                result_queue,
                error_queue,
            ),
        )
        p.start()
        procs.append(p)

    finished = 0
    worker_error = None
    with tqdm(total=total_passes, desc="benchmark") as pbar:
        while finished < len(passes_per_worker):
            if worker_error is None:
                try:
                    worker_error = error_queue.get_nowait()
                except queue.Empty:
                    worker_error = None

            if worker_error is not None:
                break

            try:
                msg = progress_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if msg is None:
                finished += 1
            else:
                pbar.update(int(msg))

    # Check for errors after the loop in case they arrived during shutdown.
    if worker_error is None:
        try:
            worker_error = error_queue.get_nowait()
        except queue.Empty:
            worker_error = None

    if worker_error is not None:
        print("Benchmark worker failed with an exception:")
        print(worker_error)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
        raise RuntimeError("Benchmark worker failed; see worker traceback above.")

    worker_results = [result_queue.get() for _ in passes_per_worker]
    for p in procs:
        p.join()

    return _finalize_results(worker_results)


def compute_distribution_stats(N: int, D: int, benchmark_result: dict):
    """
    Take raw distribution (dict: time -> probability) and optional metadata
    from benchmark() and return minimal useful stats:
        - expected_time
        - plot_df (possibly downsampled and smoothed)
        - sigma, t_min, t_max, D
        - expected_sample_burst_size, expected_sample_data_size
    """

    # Allow callers to pass either the raw distribution or the dict returned by benchmark()
    distribution_raw = benchmark_result.get("distribution", {})
    distribution = {}
    for k, v in distribution_raw.items():
        try:
            key_int = int(k)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid time bucket key {k!r} in distribution")
        distribution[key_int] = v
    expected_burst_size = benchmark_result.get("expected_sample_burst_size")
    expected_data_size = benchmark_result.get("expected_sample_data_size")

    assert expected_burst_size is not None, "expected_sample_burst_size missing"
    assert expected_data_size is not None, "expected_sample_data_size missing"

    if not distribution:
        raise ValueError("Empty distribution provided")

    # Sorted distribution table
    df = pd.DataFrame(list(distribution.items()), columns=["time_to_recover", "prob"])
    if not df.empty:
        df = df.astype({"time_to_recover": "int64", "prob": float})
    df = df.sort_values("time_to_recover").reset_index(drop=True)

    t_min = int(df["time_to_recover"].min())
    t_max = int(df["time_to_recover"].max())

    # Fill missing buckets
    full_idx = pd.Series(range(t_min, t_max + 1), name="time_to_recover")
    df_full = pd.DataFrame({"time_to_recover": full_idx.astype("int64")})
    df_full = df_full.merge(df, on="time_to_recover", how="left").fillna(0)

    # Expected value
    times = df["time_to_recover"].to_numpy(dtype=float)
    probs = df["prob"].to_numpy(dtype=float)
    total = probs.sum()
    if abs(total - 1.0) > 1e-8:
        probs = probs / total
    expected_time = float(np.sum(times * probs))

    # Downsample for plotting
    range_len = t_max - t_min + 1
    max_points = 500
    if range_len > max_points:
        bin_size = int(math.ceil(range_len / max_points))
        bins = list(range(t_min, t_max + bin_size, bin_size))
        df_full["bin"] = pd.cut(df_full["time_to_recover"], bins=bins, right=False)

        df_plot = (
            df_full.groupby("bin")
            .agg(
                time_to_recover=("time_to_recover", "mean"),
                prob=("prob", "sum"),
            )
            .reset_index(drop=True)
        )
    else:
        df_plot = df_full[["time_to_recover", "prob"]].copy()

    payload_bits = math.log2(D)
    transmitted_bits = expected_time * N
    saturation = payload_bits / N
    permeability = expected_data_size / (expected_data_size + expected_burst_size)
    ideal_time = expected_time * permeability
    ideal_transmitted_bits = transmitted_bits * permeability
    return {
        "plot_df": df_plot,
        "metrics": {
            "basic": {
                "expected_sample_data_size": expected_data_size,
                "expected_sample_burst_size": expected_burst_size,
                "expected_time_to_recover": expected_time,
                "packet_size": N,
                "payload_bits": payload_bits,
            },
            "derived": {
                "transmitted_bits": transmitted_bits,
                "saturation": saturation,  # how many packets are needed to enocode raw payload
                "bit_efficency": payload_bits
                / transmitted_bits,  # how many payload bits per transmitted bit
                "packet_efficiency": payload_bits
                / expected_time,  # how many payload bits per transmitted packet
                "bit_redundancy": transmitted_bits
                - payload_bits,  # how many extra bits were sent
                "packet_redundancy": expected_time
                - saturation,  # how many extra bits were sent
                "permeability": permeability,
                "ideal_time_to_recover": ideal_time,
                "ideal_transmitted_bits": ideal_transmitted_bits,
                "ideal_bit_efficency": payload_bits
                / ideal_transmitted_bits,  # bit_efficency / permeability
                "ideal_packet_efficiency": payload_bits
                / ideal_time,  # packet_efficiency / permeability
                "ideal_bit_redundancy": ideal_transmitted_bits - payload_bits,
                "ideal_packet_redundancy": ideal_time - saturation,
            },
        },
    }


def render_distribution(stats):
    plot_df = stats["plot_df"]
    if plot_df is None or plot_df.empty:
        print("Nothing to plot.")
        return

    expected_time = stats["metrics"]["basic"]["expected_time_to_recover"]

    fig = go.Figure()

    # Probability bars
    fig.add_trace(
        go.Bar(
            x=plot_df["time_to_recover"],
            y=plot_df["prob"],
            name="probability",
            marker_color="steelblue",
            opacity=0.6,
        )
    )

    # Vertical expected value line
    ymax = float(plot_df["prob"].max())

    fig.add_shape(
        type="line",
        x0=expected_time,
        x1=expected_time,
        y0=0,
        y1=ymax,
        line=dict(color="green", width=2, dash="dash"),
    )

    fig.update_layout(
        title="Time-to-Recover Distribution",
        xaxis_title="time to recover",
        yaxis_title="probability",
        template="plotly_white",
    )

    fig.show()


def visual_benchmark(
    producer_constructor: Callable[[int, int, int], Producer],
    recoverer_constructor: Callable[[int, int], Recoverer],
    N: int,
    D: int,
    passes: int = 100,
    sample_burst_size: Callable[[], int] = _default_sample_burst_size,
    sample_data_size: Callable[[], int] = _default_sample_data_size,
    iters_bound: int = 300,
    processes: int | None = None,
    progress_update: int = 100,
):
    return compute_distribution_stats(
        N,
        D,
        benchmark(
            producer_constructor,
            recoverer_constructor,
            N,
            D,
            passes,
            sample_burst_size,
            sample_data_size,
            iters_bound,
            processes,
            progress_update,
        ),
    )
