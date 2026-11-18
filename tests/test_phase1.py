import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from pathsnap_phase1 import segment_distances, weak_component_count


def test_segment_distance_and_fraction():
    distance, fraction = segment_distances(np.array([3.0, 4.0]), np.array([[0.0, 0.0]]), np.array([[10.0, 0.0]]))
    assert np.allclose(distance, [4.0])
    assert np.allclose(fraction, [0.3])


def test_weak_components():
    arcs = np.array([[0, 1], [1, 2], [3, 4]])
    assert weak_component_count(6, arcs) == 3
