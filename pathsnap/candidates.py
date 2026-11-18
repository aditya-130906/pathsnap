"""Inference-compatible road candidates and features (numpy only).

This module NEVER reads .route and never sees labels. The same code runs on training, student-test
and evaluator trips. Arc IDs are kept only as trip-local identifiers (`candidate_arc_id`); they are
not model features.

Candidate search
----------------
Points are sampled every `sample_step_m` along each arc near the trajectory and hashed into a uniform
grid with cell size `radius_m`. For each GPS observation, the 3x3 neighbouring cells give every arc
with a sample within `radius_m + sample_step_m / 2`, which contains every arc whose *segment* lies within
`radius_m` (unlike endpoint/midpoint shortlists, long arcs are not missed). Exact point-to-segment
distance then filters and ranks them; the nearest `max_candidates` are kept. An observation with no arc
inside the radius keeps its single nearest arc, flagged by `outside_radius = 1`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .data import Trip
from .geo import angle_difference_deg, bearing_deg, point_segment_distance, project_local_meters, trip_origin


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 50.0
    max_candidates: int = 12
    sample_step_m: float = 10.0
    speed_cap_mps: float = 80.0
    heading_min_step_m: float = 3.0
    heading_max_dt_s: float = 60.0
    large_gap_s: float = 60.0
    chunk_observations: int = 2000

    def to_dict(self) -> dict:
        return asdict(self)


ID_COLUMNS = ["trip_id", "observation_index", "timestamp_s", "candidate_arc_id"]

FEATURE_COLUMNS = [
    # distance
    "distance_m", "log_distance", "candidate_rank", "candidate_count", "distance_minus_min",
    "distance_ratio_to_min", "outside_radius",
    # road segment geometry / length
    "segment_length_m", "projected_fraction", "distance_to_start_m", "distance_to_end_m",
    # heading
    "heading_difference_deg", "cos_heading_difference", "heading_reliable",
    # movement / speed / gaps
    "step_distance_m", "speed_mps", "time_delta_s", "large_time_gap", "speed_clipped",
    # connectivity (road graph + neighbouring observations' candidates, no route)
    "prev_candidate_link", "next_candidate_link", "from_node_in_degree", "to_node_out_degree",
    "has_reverse_arc",
]


def _trip_geometry(trip: Trip, origin: np.ndarray):
    nodes_xy = project_local_meters(trip.nodes, origin)
    starts = nodes_xy[trip.arcs[:, 0]]
    ends = nodes_xy[trip.arcs[:, 1]]
    return starts, ends


def _movement(xy: np.ndarray, t: np.ndarray, cfg: CandidateConfig):
    """Per-observation step, speed, heading (from the previous observation) and gap flags."""
    n = len(xy)
    change = np.zeros_like(xy)
    change[1:] = np.diff(xy, axis=0)
    step = np.hypot(change[:, 0], change[:, 1])
    dt = np.zeros(n)
    dt[1:] = np.diff(t)
    raw_speed = np.divide(step, dt, out=np.zeros(n), where=dt > 0)
    speed = np.minimum(raw_speed, cfg.speed_cap_mps)
    heading = bearing_deg(change)
    reliable = (step >= cfg.heading_min_step_m) & (dt > 0) & (dt <= cfg.heading_max_dt_s)
    # first observation: borrow the heading of the next step when it is reliable
    if n > 1 and reliable[1]:
        heading[0], reliable[0] = heading[1], True
    return dict(step=step, dt=dt, speed=speed, speed_clipped=raw_speed > cfg.speed_cap_mps,
                heading=heading, heading_reliable=reliable, large_gap=dt > cfg.large_gap_s)


def _grid_pairs(xy, starts, ends, arc_ids, cfg: CandidateConfig):
    """Yield (obs_index, arc_id) arrays of arcs with a sample point within reach, chunked."""
    lengths = np.hypot(*(ends[arc_ids] - starts[arc_ids]).T)
    n_samples = np.maximum(np.ceil(lengths / cfg.sample_step_m).astype(np.int64), 1) + 1
    owner = np.repeat(arc_ids, n_samples)
    offsets = np.arange(len(owner)) - np.repeat(np.cumsum(n_samples) - n_samples, n_samples)
    frac = offsets / np.repeat(n_samples - 1, n_samples)
    pts = starts[owner] + frac[:, None] * (ends[owner] - starts[owner])

    cell = cfg.radius_m
    origin = pts.min(axis=0) - cell
    ci = np.floor((pts - origin) / cell).astype(np.int64)
    width = int(ci[:, 0].max()) + 3
    key = ci[:, 1] * width + ci[:, 0]
    order = np.argsort(key, kind="stable")
    key_sorted, owner_sorted, pts_sorted = key[order], owner[order], pts[order]
    reach2 = (cfg.radius_m + cfg.sample_step_m / 2 + 0.5) ** 2

    for lo in range(0, len(xy), cfg.chunk_observations):
        hi = min(lo + cfg.chunk_observations, len(xy))
        q = xy[lo:hi]
        qc = np.floor((q - origin) / cell).astype(np.int64)
        obs_parts, arc_parts = [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cx, cy = qc[:, 0] + dx, qc[:, 1] + dy
                valid = (cx >= 0) & (cy >= 0) & (cx < width)
                k = cy * width + cx
                left = np.searchsorted(key_sorted, k, side="left")
                right = np.searchsorted(key_sorted, k, side="right")
                counts = np.where(valid, right - left, 0)
                total = int(counts.sum())
                if total == 0:
                    continue
                obs_local = np.repeat(np.arange(hi - lo), counts)
                start_rep = np.repeat(left, counts)
                within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                sample_idx = start_rep + within
                d2 = ((pts_sorted[sample_idx] - q[obs_local]) ** 2).sum(axis=1)
                keep = d2 <= reach2
                obs_parts.append(obs_local[keep] + lo)
                arc_parts.append(owner_sorted[sample_idx[keep]])
        if obs_parts:
            obs = np.concatenate(obs_parts)
            arc = np.concatenate(arc_parts)
            pair = np.unique(obs.astype(np.int64) * (len(starts) + 1) + arc)
            yield pair // (len(starts) + 1), pair % (len(starts) + 1)


def generate_candidates(trip: Trip, cfg: CandidateConfig = CandidateConfig()) -> dict:
    """Return a column dict (numpy arrays) with one row per (observation, candidate arc).

    Raises if the trip carries a reference route, to make accidental leakage impossible.
    """
    if trip.route is not None:
        raise ValueError("generate_candidates needs an inference-only Trip (include_reference=False)")
    track = trip.track
    origin = trip_origin(track)
    xy = project_local_meters(track[:, :2], origin)
    starts, ends = _trip_geometry(trip, origin)
    n_arcs, n_obs = len(starts), len(xy)

    # arcs whose bounding box comes within reach of the trajectory's bounding box
    reach = cfg.radius_m + cfg.sample_step_m
    lo, hi = xy.min(axis=0) - reach, xy.max(axis=0) + reach
    near = ~((np.maximum(starts, ends) < lo) | (np.minimum(starts, ends) > hi)).any(axis=1)
    near_ids = np.flatnonzero(near)

    obs_list, arc_list = [], []
    if len(near_ids):
        for o, a in _grid_pairs(xy, starts, ends, near_ids, cfg):
            obs_list.append(o)
            arc_list.append(a)
    obs = np.concatenate(obs_list) if obs_list else np.zeros(0, np.int64)
    arc = np.concatenate(arc_list) if arc_list else np.zeros(0, np.int64)
    dist, frac = point_segment_distance(xy[obs, 0], xy[obs, 1], starts[arc, 0], starts[arc, 1],
                                        ends[arc, 0], ends[arc, 1])
    inside = dist <= cfg.radius_m
    obs, arc, dist, frac = obs[inside], arc[inside], dist[inside], frac[inside]

    # fallback: nearest arc (over all arcs) for observations with nothing inside the radius
    has = np.zeros(n_obs, bool)
    has[obs] = True
    missing = np.flatnonzero(~has)
    outside = np.zeros(len(obs), bool)
    if len(missing):
        fb_arc = np.empty(len(missing), np.int64)
        fb_dist = np.empty(len(missing))
        fb_frac = np.empty(len(missing))
        for j, i in enumerate(missing):
            d, f = point_segment_distance(xy[i, 0], xy[i, 1], starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1])
            k = int(np.argmin(d))
            fb_arc[j], fb_dist[j], fb_frac[j] = k, d[k], f[k]
        obs = np.concatenate([obs, missing])
        arc = np.concatenate([arc, fb_arc])
        dist = np.concatenate([dist, fb_dist])
        frac = np.concatenate([frac, fb_frac])
        outside = np.concatenate([outside, np.ones(len(missing), bool)])

    # rank by distance within each observation (ties broken by arc id for reproducibility)
    order = np.lexsort((arc, dist, obs))
    obs, arc, dist, frac, outside = obs[order], arc[order], dist[order], frac[order], outside[order]
    first = np.r_[True, obs[1:] != obs[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(obs)), 0))
    rank = np.arange(len(obs)) - group_start + 1
    keep = rank <= cfg.max_candidates
    obs, arc, dist, frac, outside, rank = obs[keep], arc[keep], dist[keep], frac[keep], outside[keep], rank[keep]

    count = np.bincount(obs, minlength=n_obs)[obs]
    dmin = np.full(n_obs, np.inf)
    np.minimum.at(dmin, obs, dist)
    dmin = dmin[obs]

    mv = _movement(xy, track[:, 2], cfg)
    seg = ends[arc] - starts[arc]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    road_bearing = bearing_deg(seg)
    hrel = mv["heading_reliable"][obs]
    hdiff = angle_difference_deg(mv["heading"][obs], road_bearing)
    hdiff = np.where(hrel, hdiff, 90.0)  # undefined heading -> neutral value, flagged by heading_reliable

    # road-graph connectivity
    from_node, to_node = trip.arcs[arc, 0], trip.arcs[arc, 1]
    in_deg = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes))
    out_deg = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
    n_nodes = len(trip.nodes) + 1
    pair_keys = np.unique(trip.arcs[:, 0].astype(np.int64) * n_nodes + trip.arcs[:, 1])
    rev_keys = to_node.astype(np.int64) * n_nodes + from_node
    pos = np.searchsorted(pair_keys, rev_keys)
    has_reverse = (pos < len(pair_keys)) & (pair_keys[np.minimum(pos, len(pair_keys) - 1)] == rev_keys)

    # link to neighbouring observations' candidate sets (same arc, or arc continues/precedes it)
    stride = np.int64(max(n_arcs, len(trip.nodes)) + 1)
    def member(query, table):
        table = np.unique(table)
        p = np.searchsorted(table, query)
        return (p < len(table)) & (table[np.minimum(p, len(table) - 1)] == query)
    o64 = obs.astype(np.int64)
    same_prev = member((o64 - 1) * stride + arc, o64 * stride + arc)
    into = member((o64 - 1) * stride + from_node, o64 * stride + to_node)   # prev candidate ends where this starts
    same_next = member((o64 + 1) * stride + arc, o64 * stride + arc)
    onto = member((o64 + 1) * stride + to_node, o64 * stride + from_node)   # next candidate starts where this ends
    prev_link = np.where(obs > 0, (same_prev | into), False)
    next_link = np.where(obs < n_obs - 1, (same_next | onto), False)

    return {
        "trip_id": np.full(len(obs), trip.trip_id, dtype=object),
        "observation_index": obs,
        "timestamp_s": track[obs, 2],
        "candidate_arc_id": arc,
        "distance_m": dist,
        "log_distance": np.log1p(dist),
        "candidate_rank": rank,
        "candidate_count": count,
        "distance_minus_min": dist - dmin,
        "distance_ratio_to_min": dist / (dmin + 1.0),
        "outside_radius": outside.astype(np.int8),
        "segment_length_m": seg_len,
        "projected_fraction": frac,
        "distance_to_start_m": np.hypot(*(xy[obs] - starts[arc]).T),
        "distance_to_end_m": np.hypot(*(xy[obs] - ends[arc]).T),
        "heading_difference_deg": hdiff,
        "cos_heading_difference": np.where(hrel, np.cos(np.radians(hdiff)), 0.0),
        "heading_reliable": hrel.astype(np.int8),
        "step_distance_m": mv["step"][obs],
        "speed_mps": mv["speed"][obs],
        "time_delta_s": mv["dt"][obs],
        "large_time_gap": mv["large_gap"][obs].astype(np.int8),
        "speed_clipped": mv["speed_clipped"][obs].astype(np.int8),
        "prev_candidate_link": prev_link.astype(np.int8),
        "next_candidate_link": next_link.astype(np.int8),
        "from_node_in_degree": in_deg[from_node],
        "to_node_out_degree": out_deg[to_node],
        "has_reverse_arc": has_reverse.astype(np.int8),
    }
