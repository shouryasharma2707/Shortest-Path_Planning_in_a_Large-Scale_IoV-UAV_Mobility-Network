"""
Preprocess the DIMACS 9th Challenge BAY road network.

Usage:
    python preprocess.py --gr USA-road-d.BAY.gr --co USA-road-d.BAY.co --out data

Outputs (in --out):
    bay_<N>.edges.csv   columns: u,v,w      (0-indexed, renumbered, directed arcs)
    bay_<N>.coords.csv  columns: id,lon,lat (degrees)
    stats.csv           graph statistics for every size
"""
import argparse
import csv
import os
from collections import deque

import numpy as np

SIZES = [1000, 5000, 10000, 25000, 50000]
SEED_NODE = 0          # fixed seed => reproducible subgraphs
EARTH_R = 6371000.0    # metres


def open_any(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt")
    return open(path, "r")


def load_graph(gr_path):
    """Read .gr file -> (n, list of (u, v, w)) with 0-indexed ids."""
    n = m = None
    arcs = []
    with open_any(gr_path) as f:
        for line in f:
            if line.startswith("p"):
                _, _, n, m = line.split()
                n, m = int(n), int(m)
            elif line.startswith("a"):
                _, u, v, w = line.split()
                arcs.append((int(u) - 1, int(v) - 1, int(w)))
    assert n is not None and len(arcs) == m, "header / arc count mismatch"
    return n, arcs


def load_coords(co_path, n):
    """Read .co file -> array (n, 2) of [lon, lat] in degrees."""
    xy = np.zeros((n, 2), dtype=np.float64)
    with open_any(co_path) as f:
        for line in f:
            if line.startswith("v"):
                _, i, x, y = line.split()
                xy[int(i) - 1] = (int(x) / 1e6, int(y) / 1e6)
    return xy


def build_adj(n, arcs):
    adj = [[] for _ in range(n)]
    for u, v, w in arcs:
        adj[u].append((v, w))
    return adj


def haversine(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance in metres."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def heuristic_ratio(arcs, xy):
    """min(weight / haversine) over arcs. Scale haversine by this to keep h admissible."""
    a = np.array(arcs, dtype=np.float64)
    u, v, w = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2]
    d = haversine(xy[u, 0], xy[u, 1], xy[v, 0], xy[v, 1])
    mask = d > 0
    r = w[mask] / d[mask]
    return r.min(), (r < 1).sum(), mask.sum()


def bfs_nodes(adj, seed, size):
    """First `size` nodes reached by BFS from seed (follows arc direction)."""
    seen = {seed}
    order = [seed]
    q = deque([seed])
    while q and len(order) < size:
        u = q.popleft()
        for v, _ in adj[u]:
            if v not in seen:
                seen.add(v)
                order.append(v)
                q.append(v)
                if len(order) == size:
                    break
    return order


def induced_subgraph(adj, nodes):
    """Keep only arcs with both endpoints in `nodes`; renumber to 0..k-1."""
    new_id = {old: i for i, old in enumerate(nodes)}
    sub = [[] for _ in nodes]
    for old in nodes:
        for v, w in adj[old]:
            if v in new_id:
                sub[new_id[old]].append((new_id[v], w))
    return sub


def graph_stats(adj):
    deg = np.array([len(a) for a in adj])
    return {
        "vertices": len(adj),
        "arcs": int(deg.sum()),
        "avg_out_degree": round(float(deg.mean()), 3),
        "max_out_degree": int(deg.max()),
    }


def save_subgraph(out_dir, name, sub, xy_sub):
    with open(os.path.join(out_dir, f"{name}.edges.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["u", "v", "w"])
        for u, nbrs in enumerate(sub):
            for v, w in nbrs:
                wr.writerow([u, v, w])
    with open(os.path.join(out_dir, f"{name}.coords.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "lon", "lat"])
        for i, (lon, lat) in enumerate(xy_sub):
            wr.writerow([i, f"{lon:.6f}", f"{lat:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr", required=True)
    ap.add_argument("--co", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=SEED_NODE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n, arcs = load_graph(args.gr)
    xy = load_coords(args.co, n)
    adj = build_adj(n, arcs)

    full = graph_stats(adj)
    print("Full graph:", full)

    ratio, below1, total = heuristic_ratio(arcs, xy)
    print(f"min(w/haversine) = {ratio:.4f}; arcs with ratio<1: {below1}/{total}")

    rows = [{"size": "full", **full}]
    for size in SIZES:
        nodes = bfs_nodes(adj, args.seed, size)
        sub = induced_subgraph(adj, nodes)
        stats = graph_stats(sub)
        save_subgraph(args.out, f"bay_{size}", sub, xy[nodes])
        print(f"{size}:", stats)
        rows.append({"size": size, **stats})

    with open(os.path.join(args.out, "stats.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()