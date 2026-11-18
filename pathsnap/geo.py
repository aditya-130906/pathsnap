"""Metric projection and segment geometry (numpy only)."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def trip_origin(track_lonlat: np.ndarray) -> np.ndarray:
    """Projection origin for one trip: the median GPS position (robust to outliers)."""
    return np.median(np.asarray(track_lonlat, dtype=np.float64)[:, :2], axis=0)


def project_local_meters(lonlat: np.ndarray, origin_lonlat: np.ndarray) -> np.ndarray:
    """Local tangent-plane (equirectangular) projection in metres, centred on each trip.

    Trips are city-scale (a few km to tens of km), where the error of this projection is far
    below GPS noise. One origin per trip avoids imposing a single UTM zone on a worldwide dataset.
    Returns (east, north) in metres.
    """
    lonlat = np.asarray(lonlat, dtype=np.float64)
    reference = np.asarray(origin_lonlat, dtype=np.float64)
    diffs = np.deg2rad(lonlat[..., :2] - reference)
    return EARTH_RADIUS_M * diffs * np.array([np.cos(np.deg2rad(reference[1])), 1.0])


def bearing_deg(vectors: np.ndarray) -> np.ndarray:
    """Compass bearing, clockwise from north, for (east, north) vectors."""
    vectors = np.asarray(vectors, dtype=np.float64)
    return (np.degrees(np.arctan2(vectors[..., 0], vectors[..., 1])) + 360.0) % 360.0


def angle_difference_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    return np.abs((np.asarray(first) - np.asarray(second) + 180.0) % 360.0 - 180.0)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Exact distance from points to finite segments (arrays broadcast).

    Returns (distance, fraction) where fraction in [0, 1] is the clamped projection position.
    """
    vx, vy = bx - ax, by - ay
    length_sq = vx * vx + vy * vy
    fraction = ((px - ax) * vx + (py - ay) * vy) / np.maximum(length_sq, 1e-12)
    fraction = np.clip(fraction, 0.0, 1.0)
    dx = px - (ax + fraction * vx)
    dy = py - (ay + fraction * vy)
    return np.hypot(dx, dy), fraction
