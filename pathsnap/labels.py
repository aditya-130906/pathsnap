"""Observation-label APPROXIMATION from .route (training supervision and evaluation only).

The .route file is an ordered sequence of arc IDs, NOT one label per GPS observation (e.g. train trip
00000001 has 1,315 observations and 525 route arcs). This module derives an approximate reference arc for
each observation by monotonic dynamic-programming alignment of the GPS sequence onto the route sequence:

    D[i, k] = c(i, k) + min_{k' <= k} D[i-1, k'],   c = min(point-to-arc distance, clip_m)

* time order is preserved: an observation may stay on the same route position or move forward any number
  of positions (1 Hz sampling skips short arcs), never backwards;
* revisited arcs keep separate route positions, so loops and U-turns align in order;
* cost clipping keeps a few wild GPS points from dragging the whole alignment;
* a heading term separates the two directions of a two-way road, which have identical distances.

Limitations (report these, do not overstate label certainty)
-------------------------------------------------------------
* It is an approximation: near intersections or on short arcs the boundary between consecutive arcs can be
  off by one or two observations.
* The reference route is itself not guaranteed error-free (a few routes have arc endpoint breaks).
* GPS stretches that are off the route (start/end outside the route, detours, long gaps) align to the nearest
  allowed position at a large distance. Observations whose aligned arc is farther than `valid_max_m` are
  marked `label_valid = 0` and must be excluded from training and scoring.

Nothing produced here (reference arc, distance, validity) may be used as a model input feature.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .data import Trip
from .geo import angle_difference_deg, bearing_deg, point_segment_distance, project_local_meters, trip_origin


@dataclass(frozen=True)
class LabelConfig:
    clip_m: float = 100.0
    valid_max_m: float = 50.0
    # direction term: the two directions of a two-way road have identical distance, so a reliable GPS
    # heading adds heading_weight_m * (1 - cos(heading difference)) / 2 metres of cost (0 aligned, full opposite)
    heading_weight_m: float = 20.0
    heading_min_step_m: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)


def align_route(trip: Trip, cfg: LabelConfig = LabelConfig()) -> dict:
    """Per-observation reference arc (approximate) plus alignment diagnostics."""
    if trip.route is None:
        raise ValueError("align_route needs a Trip loaded with include_reference=True")
    route = trip.route
    origin = trip_origin(trip.track)
    xy = project_local_meters(trip.track[:, :2], origin)
    nodes_xy = project_local_meters(trip.nodes, origin)
    a = nodes_xy[trip.arcs[route, 0]]
    b = nodes_xy[trip.arcs[route, 1]]
    n, k = len(xy), len(route)
    # GPS heading from the centred difference of neighbouring observations
    nxt = xy[np.minimum(np.arange(n) + 1, n - 1)] - xy[np.maximum(np.arange(n) - 1, 0)]
    gps_bearing = bearing_deg(nxt)
    gps_reliable = np.hypot(nxt[:, 0], nxt[:, 1]) >= cfg.heading_min_step_m
    arc_bearing = bearing_deg(b - a)
    back = np.empty((n, k), dtype=np.int32)
    idx = np.arange(k)
    prev = None
    for i in range(n):
        d, _ = point_segment_distance(xy[i, 0], xy[i, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        cost = np.minimum(d, cfg.clip_m)
        if gps_reliable[i]:
            diff = np.radians(angle_difference_deg(gps_bearing[i], arc_bearing))
            cost = cost + cfg.heading_weight_m * (1 - np.cos(diff)) / 2
        if prev is None:
            cur = cost
            back[i] = idx
        else:
            prefix_min = np.minimum.accumulate(prev)
            new_min = np.r_[True, prev[1:] < prefix_min[:-1]]
            back[i] = np.maximum.accumulate(np.where(new_min, idx, 0))
            cur = cost + prefix_min
        prev = cur
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    ref = route[pos]
    dist, _ = point_segment_distance(xy[:, 0], xy[:, 1], a[pos, 0], a[pos, 1], b[pos, 0], b[pos, 1])
    return {
        "trip_id": np.full(n, trip.trip_id, dtype=object),
        "observation_index": np.arange(n),
        "reference_arc_id": ref,
        "route_position": pos,
        "alignment_distance_m": dist,
        "label_valid": (dist <= cfg.valid_max_m).astype(np.int8),
    }


def alignment_quality(trip: Trip, labels: dict, cand_obs: np.ndarray, cand_arc: np.ndarray) -> dict:
    """Per-trip label quality and candidate miss rate (evaluation diagnostics, never model inputs)."""
    d = labels["alignment_distance_m"]
    valid = labels["label_valid"].astype(bool)
    ref = labels["reference_arc_id"]
    hit = np.zeros(len(d), bool)
    match = cand_arc == ref[cand_obs]
    hit[cand_obs[match]] = True
    # naive comparison: nearest route arc ignoring time order
    route = trip.route
    breaks = int(np.count_nonzero(trip.arcs[route[:-1], 1] != trip.arcs[route[1:], 0])) if len(route) > 1 else 0
    return {
        "trip_id": trip.trip_id,
        "observations": int(len(d)),
        "route_arcs": int(len(route)),
        "route_endpoint_breaks": breaks,
        "route_positions_used_share": float(len(np.unique(labels["route_position"])) / max(len(route), 1)),
        "alignment_median_m": float(np.median(d)),
        "alignment_p95_m": float(np.percentile(d, 95)),
        "label_valid_share": float(valid.mean()),
        "candidate_miss_rate": float(1.0 - hit[valid].mean()) if valid.any() else float("nan"),
        "valid_observations": int(valid.sum()),
    }
