"""
Summarise results.csv (written by sp.cpp) and produce plots.

Usage:
    python analyze.py --results results --data data

Outputs (in --results):
    summary.csv / summary.md    Part E table (averages over the query pairs)
    graph_stats.md              Part B table (from data/stats.csv), if available
    plots/*.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ALGOS = {"dijkstra": "Dijkstra", "astar": "A*"}
STYLE = {"dijkstra": ("tab:blue", "o"), "astar": ("tab:orange", "s")}


def line_plot(summary, column, ylabel, title, path, log=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, label in ALGOS.items():
        d = summary[summary["algo"] == algo].sort_values("size")
        color, marker = STYLE[algo]
        ax.plot(d["size"], d[column], marker=marker, color=color, label=label)
    ax.set_xlabel("Graph size (vertices)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.results, "results.csv"))
    df = df[df["found"] == 1]

    summary = (
        df.groupby(["size", "algo"])
        .agg(
            pairs=("pair", "count"),
            avg_cost=("cost", "mean"),
            avg_explored=("explored", "mean"),
            avg_edges=("edges", "mean"),
            avg_time_ms=("time_ms", "mean"),
            avg_path_len=("path_len", "mean"),
        )
        .reset_index()
    )
    summary["algo"] = pd.Categorical(summary["algo"], ["dijkstra", "astar"])
    summary = summary.sort_values(["size", "algo"]).reset_index(drop=True)
    summary["algo"] = summary["algo"].astype(str)
    summary.to_csv(os.path.join(args.results, "summary.csv"), index=False)

    # Part E table (markdown)
    pretty = summary.copy()
    pretty["algo"] = pretty["algo"].map(ALGOS)
    pretty = pretty.rename(columns={
        "size": "Graph Size", "algo": "Algorithm", "pairs": "Pairs",
        "avg_cost": "Avg Cost", "avg_explored": "Avg Vertices Explored",
        "avg_edges": "Avg Edges Examined", "avg_time_ms": "Avg Time (ms)",
        "avg_path_len": "Avg Path Length",
    })
    md = pretty.to_markdown(index=False, floatfmt=".2f")
    with open(os.path.join(args.results, "summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)

    # A* vs Dijkstra ratios per size
    piv = summary.pivot(index="size", columns="algo")
    ratio = pd.DataFrame({
        "explored_ratio (A*/Dijkstra)": piv["avg_explored"]["astar"] / piv["avg_explored"]["dijkstra"],
        "time_ratio (A*/Dijkstra)": piv["avg_time_ms"]["astar"] / piv["avg_time_ms"]["dijkstra"],
    })
    print("\n" + ratio.to_markdown(floatfmt=".3f"))

    # cost agreement check
    wide = df.pivot_table(index=["size", "pair"], columns="algo", values="cost")
    mismatches = int((wide["dijkstra"] != wide["astar"]).sum())
    print(f"\nPairs where costs differ: {mismatches} of {len(wide)}")

    # Part B table
    stats_path = os.path.join(args.data, "stats.csv")
    if os.path.exists(stats_path):
        with open(os.path.join(args.results, "graph_stats.md"), "w") as f:
            f.write(pd.read_csv(stats_path).to_markdown(index=False) + "\n")

    # plots
    plots = os.path.join(args.results, "plots")
    os.makedirs(plots, exist_ok=True)
    line_plot(summary, "avg_time_ms", "Average time (ms)",
              "Execution time vs graph size", os.path.join(plots, "time.png"))
    line_plot(summary, "avg_explored", "Average vertices explored",
              "Vertices explored vs graph size", os.path.join(plots, "explored.png"))
    line_plot(summary, "avg_edges", "Average edges examined",
              "Edges examined vs graph size", os.path.join(plots, "edges.png"))
    line_plot(summary, "avg_time_ms", "Average time (ms)",
              "Execution time vs graph size (log-log)",
              os.path.join(plots, "time_loglog.png"), log=True)
    print(f"\nPlots saved to {plots}")


if __name__ == "__main__":
    main()