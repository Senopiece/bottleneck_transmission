import random
from collections import defaultdict
from typing import Callable
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from tqdm import tqdm

from impls._interface import Producer, Recoverer


def benchmark(
    producer_constructor: Callable[[int, int, int], Producer],
    recoverer_constructor: Callable[[int, int], Recoverer],
    N: int,
    D: int,
    passes: int = 100,
    sample_burst_size: Callable[[], int] = lambda: random.randint(1, 5),
    sample_data_size: Callable[[], int] = lambda: random.randint(3, 30),
    iters_bound: int = 10000,
):
    time_to_recover_distribution = defaultdict(lambda: 0)

    init_pat = 3
    patience = init_pat
    mismatch = 0

    for _ in tqdm(range(passes)):
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
            time_to_recover += burst_size
            for _ in range(burst_size):
                producer.generate()

            # Use samples
            for _ in range(sample_data_size()):
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
            if patience > 0:
                print(f"Failed to recover {data} within iteration bound")
            patience -= 1
            passes -= 1
            continue

        # Check correctness
        # assert np.array_equal(recovered, data), f"Expected {data}, got {recovered}"
        if not np.array_equal(recovered, data):
            if patience > 0:
                print(f"Data mismatch: expected {data}, got {recovered}")
                patience -= 1
            mismatch += 1
            passes -= 1
            continue

        time_to_recover_distribution[time_to_recover] += 1

    print(f"{init_pat - patience} failures.")
    if mismatch > 0:
        print(f"{mismatch} MISMATCHES!!!")
    return {k: v / passes for k, v in time_to_recover_distribution.items()}


def compute_distribution_stats(D: int, distribution: dict):
    """
    Take raw distribution (dict: time -> probability)
    and return minimal useful stats:
        - expected_time
        - plot_df (possibly downsampled and smoothed)
        - sigma, t_min, t_max, D
    """

    if not distribution:
        return {
            "expected_time": None,
            "plot_df": None,
            "sigma": None,
            "t_min": None,
            "t_max": None,
            "D": D,
        }

    # Sorted distribution table
    df = pd.DataFrame(list(distribution.items()), columns=["time_to_recover", "prob"])
    df = df.sort_values("time_to_recover").reset_index(drop=True)

    t_min = int(df["time_to_recover"].min())
    t_max = int(df["time_to_recover"].max())

    # Fill missing buckets
    full_idx = pd.Series(range(t_min, t_max + 1), name="time_to_recover")
    df_full = pd.DataFrame({"time_to_recover": full_idx})
    df_full = df_full.merge(df, on="time_to_recover", how="left").fillna(0)

    # Gaussian smoothing
    range_len = t_max - t_min + 1
    sigma = max(1.0, range_len / 100.0)
    radius = int(max(1, int(3 * sigma)))

    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    smoothed = np.convolve(df_full["prob"].to_numpy(), kernel, mode="same")
    df_full["smoothed"] = smoothed

    # Expected value
    times = df["time_to_recover"].to_numpy(dtype=float)
    probs = df["prob"].to_numpy(dtype=float)
    total = probs.sum()
    if abs(total - 1.0) > 1e-8:
        probs = probs / total
    expected_time = float(np.sum(times * probs))

    # Downsample for plotting
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
                smoothed=("smoothed", "mean"),
            )
            .reset_index(drop=True)
        )
    else:
        df_plot = df_full[["time_to_recover", "prob", "smoothed"]].copy()

    return {
        "expected_time": expected_time,
        "plot_df": df_plot,
        "sigma": sigma,
        "t_min": t_min,
        "t_max": t_max,
        "D": D,
    }


def render_distribution(stats):
    plot_df = stats["plot_df"]
    if plot_df is None or plot_df.empty:
        print("Nothing to plot.")
        return

    expected_time = stats["expected_time"]
    sigma = stats["sigma"]

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

    # Smoothed line
    fig.add_trace(
        go.Scatter(
            x=plot_df["time_to_recover"],
            y=plot_df["smoothed"],
            mode="lines",
            name=f"smoothed (sigma={sigma:.2f})",
            line=dict(color="firebrick", width=2),
        )
    )

    # Vertical expected value line
    ymax = max(float(plot_df["prob"].max()), float(plot_df["smoothed"].max()))

    fig.add_shape(
        type="line",
        x0=expected_time,
        x1=expected_time,
        y0=0,
        y1=ymax,
        line=dict(color="green", width=2, dash="dash"),
    )

    # Invisible trace to add a legend entry for expected value
    fig.add_trace(
        go.Scatter(
            x=[expected_time],
            y=[ymax],
            mode="markers",
            marker=dict(color="green", size=8),
            name=f"expected = {expected_time:.2f}",
            showlegend=True,
        )
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
    sample_burst_size: Callable[[], int] = lambda: random.randint(1, 10),
    sample_data_size: Callable[[], int] = lambda: random.randint(3, 5),
    iters_bound: int = 300,
):
    return compute_distribution_stats(
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
        ),
    )
