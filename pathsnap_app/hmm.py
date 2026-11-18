"""HMM + Viterbi route reconstruction on top of HGB candidate probabilities.

Hidden states   : the candidate road arcs of each GPS observation (HGB pipeline: top-8 arcs within 150 m).
Observations    : the GPS fixes in time order.
Emission score  : w_emit * log P_HGB(arc is the true arc | features)          (log space throughout)
Transition score: road-network route distance d between the two projected positions vs. the straight-line GPS
                  displacement g (Newson & Krumm 2009):  -|d - g| / beta,  beta grows with elapsed time.
                  d = rest of arc a + shortest path (to-node of a -> from-node of b) + part of arc b,
                  with same-arc and directly-connected cases handled exactly. Unreachable within the search limit
                  -> a fixed large penalty (disconnected candidates can still be decoded instead of crashing).
Gaps            : if the time gap exceeds `gap_s` or the GPS jump exceeds `jump_m`, transitions are neutral (0) for
                  that step, so the decoder neither forces nor forbids a path across a dropout.
Decoding        : Viterbi recurrence  delta_t(j) = max_i [delta_{t-1}(i) + T(i, j)] + E_t(j)  with back-pointers.
Route           : the decoded arcs per observation, with consecutive arcs joined by their shortest connecting path,
                  give a reconstructed arc sequence (and its geometry) for the whole trip.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, asdict

import numpy as np

EARTH_R = 6_371_008.8


@dataclass
class HMMConfig:
    w_emit: float = 1.0
    beta_m: float = 10.0          # transition scale (metres) for a 1 s step
    beta_per_s: float = 2.0       # extra metres of tolerance per extra second elapsed
    unreachable_logp: float = -40.0
    gap_s: float = 60.0
    jump_m: float = 1500.0
    search_extra_m: float = 300.0  # Dijkstra limit = 2 * GPS displacement + this
    search_cap_m: float = 5000.0
    # direction handling (v2): the two directions of a two-way road share geometry
    backtrack_tol_m: float = 1e9      # backward movement along the same arc tolerated as GPS jitter (v1: unlimited)
    uturn_logp: float = 0.0           # extra log-penalty for switching to the reverse twin arc (v1: none)
    w_head: float = 0.0               # emission weight of heading agreement, w_head * (cos(diff) - 1) / 2 (v1: none)

    def to_dict(self):
        return asdict(self)


class RoadGraph:
    """Directed road graph of one trip, projected to metres, with cached bounded Dijkstra."""

    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, origin_lonlat=None):
        nodes_lonlat = np.asarray(nodes_lonlat, float)
        self.arcs = np.asarray(arcs, np.int64)
        o = np.median(nodes_lonlat, axis=0) if origin_lonlat is None else np.asarray(origin_lonlat, float)
        self.origin = o
        self.xy = self.project(nodes_lonlat)
        seg = self.xy[self.arcs[:, 1]] - self.xy[self.arcs[:, 0]]
        self.length = np.hypot(seg[:, 0], seg[:, 1])
        order = np.argsort(self.arcs[:, 0], kind="stable")
        self.out_arcs = order
        self.indptr = np.searchsorted(self.arcs[order, 0], np.arange(len(nodes_lonlat) + 1))
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def project(self, lonlat):
        lonlat = np.asarray(lonlat, float)
        d = np.deg2rad(lonlat[..., :2] - self.origin)
        return EARTH_R * d * np.array([np.cos(np.deg2rad(self.origin[1])), 1.0])

    def dijkstra(self, source: int, limit: float):
        """Distances (m) and predecessor arcs from `source` to nodes within `limit`; cached per source."""
        hit = self._cache.get(source)
        if hit is not None and hit[0] >= limit:
            return hit[1], hit[2]
        dist = {source: 0.0}; prev = {}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for k in range(self.indptr[u], self.indptr[u + 1]):
                a = self.out_arcs[k]; v = int(self.arcs[a, 1]); nd = d + self.length[a]
                if nd < dist.get(v, np.inf) and nd <= limit:
                    dist[v] = nd; prev[v] = int(a); heapq.heappush(heap, (nd, v))
        self._cache[source] = (limit, dist, prev)
        return dist, prev

    def path_arcs(self, a: int, b: int, limit: float = 5000.0) -> list[int]:
        """Arcs strictly between arc a and arc b on the shortest connecting path ([] if adjacent/unreachable)."""
        s, t = int(self.arcs[a, 1]), int(self.arcs[b, 0])
        if s == t:
            return []
        dist, prev = self.dijkstra(s, limit)
        if t not in dist:
            return []
        out = []; v = t
        while v != s:
            e = prev[v]; out.append(e); v = int(self.arcs[e, 0])
        return out[::-1]


def route_distance(g: RoadGraph, a: int, fa: float, b: int, fb: float, limit: float, backtrack_tol_m: float = 1e9) -> float:
    la, lb = g.length[a], g.length[b]
    if a == b:
        back = (fa - fb) * la
        if back <= 0:
            return -back                                # forward along the same arc
        return back if back <= backtrack_tol_m else np.inf   # small backward jitter only; larger = impossible
    head = (1 - fa) * la
    s, t = int(g.arcs[a, 1]), int(g.arcs[b, 0])
    if s == t:
        return head + fb * lb
    dist, _ = g.dijkstra(s, limit)
    sp = dist.get(t)
    return np.inf if sp is None else head + sp + fb * lb


def viterbi(obs_cands: list[dict], gps_xy: np.ndarray, times: np.ndarray, g: RoadGraph, cfg: HMMConfig):
    """obs_cands[t] = dict(arc=int array, frac=float array, prob=float array). Returns chosen candidate index per t."""
    T = len(obs_cands)
    log_e = []
    for c in obs_cands:
        e = cfg.w_emit * np.log(np.clip(c["prob"], 1e-6, 1.0))
        if cfg.w_head and "hdiff" in c:
            e = e + np.where(c["hrel"] > 0, cfg.w_head * (np.cos(np.radians(c["hdiff"])) - 1) / 2, 0.0)
        log_e.append(e)
    delta = log_e[0].copy()
    back = [np.zeros(len(log_e[0]), np.int64)]
    for t in range(1, T):
        A, B = obs_cands[t - 1], obs_cands[t]
        dt = max(float(times[t] - times[t - 1]), 1e-3)
        gdist = float(np.hypot(*(gps_xy[t] - gps_xy[t - 1])))
        if dt > cfg.gap_s or gdist > cfg.jump_m:
            trans = np.zeros((len(A["arc"]), len(B["arc"])))
        else:
            beta = cfg.beta_m + cfg.beta_per_s * max(dt - 1.0, 0.0)
            limit = min(2 * gdist + cfg.search_extra_m, cfg.search_cap_m)
            trans = np.empty((len(A["arc"]), len(B["arc"])))
            for i, (a, fa) in enumerate(zip(A["arc"], A["frac"])):
                for j, (b, fb) in enumerate(zip(B["arc"], B["frac"])):
                    d = route_distance(g, int(a), float(fa), int(b), float(fb), limit, cfg.backtrack_tol_m)
                    trans[i, j] = cfg.unreachable_logp if not np.isfinite(d) else -abs(d - gdist) / beta
                    if cfg.uturn_logp and a != b and g.arcs[a, 0] == g.arcs[b, 1] and g.arcs[a, 1] == g.arcs[b, 0]:
                        trans[i, j] += cfg.uturn_logp
        scores = delta[:, None] + trans
        back.append(np.argmax(scores, axis=0))
        delta = scores[back[-1], np.arange(scores.shape[1])] + log_e[t]
    path = np.empty(T, np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t][path[t]]
    return path


def reconstruct_route(g: RoadGraph, decoded_arcs: list[int]) -> list[int]:
    """Deduplicate consecutive arcs and fill gaps with shortest connecting paths."""
    seq = [a for i, a in enumerate(decoded_arcs) if i == 0 or a != decoded_arcs[i - 1]]
    route = []
    for i, a in enumerate(seq):
        if i:
            route.extend(g.path_arcs(seq[i - 1], a))
        route.append(int(a))
    return route


def continuity(g: RoadGraph, arcs_per_obs) -> float:
    a = np.asarray(arcs_per_obs)
    if len(a) < 2:
        return 1.0
    ok = (a[1:] == a[:-1]) | (g.arcs[a[:-1], 1] == g.arcs[a[1:], 0])
    return float(ok.mean())


def route_overlap(pred_route: list[int], ref_route, length: np.ndarray) -> dict:
    """Length-weighted precision/recall/F1 of reconstructed arcs against the reference route arcs."""
    p, r = set(pred_route), set(int(x) for x in ref_route)
    inter = sum(length[a] for a in p & r)
    lp, lr = sum(length[a] for a in p), sum(length[a] for a in r)
    prec = inter / lp if lp else 0.0; rec = inter / lr if lr else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
