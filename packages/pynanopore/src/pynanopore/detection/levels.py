"""Multi-level / multi-conductance analysis within detected events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pynanopore.io.trace import Trace


@dataclass
class LevelFeatures:
    """Up to two blocked conductance levels plus open-pore reference."""

    n_levels: float = 1.0
    level1_current: float = float("nan")
    level2_current: float = float("nan")
    level1_fraction: float = float("nan")
    level2_fraction: float = float("nan")
    level_sep: float = float("nan")  # |L2 - L1| / |i0| when defined
    level_rms: float = float("nan")

    def as_event_fields(self) -> dict[str, float]:
        return {
            "n_levels": float(self.n_levels),
            "level1_current": float(self.level1_current),
            "level2_current": float(self.level2_current),
            "level1_fraction": float(self.level1_fraction),
            "level2_fraction": float(self.level2_fraction),
            "level_sep": float(self.level_sep),
            "level_rms": float(self.level_rms),
        }


@dataclass
class LevelAssignment:
    """Per-sample level labels inside one event (0 = deepest / level1)."""

    features: LevelFeatures
    labels: NDArray[np.integer]
    centers: NDArray[np.floating]


@dataclass
class MultiLevelIdealization:
    """Trace-length idealization with open / level1 / level2 codes."""

    time: NDArray[np.floating]
    idealized: NDArray[np.floating]
    level_code: NDArray[np.integer]  # 0=open, 1=level1, 2=level2
    events: list  # Event-like objects with i0 / start_idx / end_idx


def _kmeans_1d(x: NDArray[np.floating], k: int, *, n_iter: int = 25) -> tuple[NDArray, NDArray]:
    """Simple 1-D k-means. Returns ``(labels, centers)``."""
    x = np.asarray(x, dtype=float)
    if k <= 1 or len(x) < k:
        return np.zeros(len(x), dtype=int), np.array([float(np.mean(x))])

    qs = np.linspace(0, 100, k)
    centers = np.percentile(x, qs).astype(float)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(n_iter):
        dist = np.abs(x[:, None] - centers[None, :])
        labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = float(np.mean(x[mask]))
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def _bic_1d(x: NDArray[np.floating], labels: NDArray, centers: NDArray) -> float:
    """Gaussian BIC for 1-D mixture with equal variance (rough model-order score)."""
    n = len(x)
    k = len(centers)
    if n < 2:
        return float("inf")
    resid = x - centers[labels]
    var = float(np.mean(resid**2)) + 1e-18
    n_params = k + 1
    ll = -0.5 * n * (np.log(2 * np.pi * var) + 1.0)
    return float(-2.0 * ll + n_params * np.log(n))


def assign_event_levels(
    segment: NDArray[np.floating],
    i0: float,
    *,
    max_levels: int = 2,
    min_samples: int = 12,
) -> LevelAssignment:
    """Cluster an event segment and return labels + summary features."""
    seg = np.asarray(segment, dtype=float)
    if len(seg) < min_samples or max_levels < 1:
        mean = float(np.mean(seg)) if len(seg) else float("nan")
        feats = LevelFeatures(n_levels=1.0, level1_current=mean, level1_fraction=1.0, level_rms=0.0)
        return LevelAssignment(
            features=feats,
            labels=np.zeros(len(seg), dtype=int),
            centers=np.array([mean], dtype=float),
        )

    best_bic = float("inf")
    best_labels: NDArray | None = None
    best_centers: NDArray | None = None
    k_max = min(int(max_levels), 2, len(seg))
    for k in range(1, k_max + 1):
        labels, centers = _kmeans_1d(seg, k)
        bic = _bic_1d(seg, labels, centers)
        if bic < best_bic:
            best_bic = bic
            best_labels = labels
            best_centers = centers

    assert best_labels is not None and best_centers is not None
    order = np.argsort(np.abs(best_centers - i0))[::-1]
    centers_sorted = best_centers[order]
    remap = {int(order[i]): i for i in range(len(order))}
    labels_sorted = np.array([remap[int(lb)] for lb in best_labels], dtype=int)

    n_lev = len(centers_sorted)
    fracs = np.array([(labels_sorted == j).mean() for j in range(n_lev)], dtype=float)
    resid = seg - centers_sorted[labels_sorted]
    rms = float(np.sqrt(np.mean(resid**2)))

    l1 = float(centers_sorted[0])
    f1 = float(fracs[0])
    l2 = float(centers_sorted[1]) if n_lev > 1 else float("nan")
    f2 = float(fracs[1]) if n_lev > 1 else float("nan")
    sep = abs(l2 - l1) / abs(i0) if n_lev > 1 and abs(i0) > 1e-12 else float("nan")

    feats = LevelFeatures(
        n_levels=float(n_lev),
        level1_current=l1,
        level2_current=l2,
        level1_fraction=f1,
        level2_fraction=f2,
        level_sep=sep,
        level_rms=rms,
    )
    return LevelAssignment(features=feats, labels=labels_sorted, centers=centers_sorted)


def analyze_event_levels(
    segment: NDArray[np.floating],
    i0: float,
    *,
    max_levels: int = 2,
    min_samples: int = 12,
) -> LevelFeatures:
    """
    Estimate 1–2 blocked current levels inside an event segment.

    Uses 1-D k-means with BIC to choose ``k ∈ {1, 2}`` (capped by ``max_levels``).
    Levels are ordered by increasing distance from the open-pore ``i0``.
    """
    return assign_event_levels(segment, i0, max_levels=max_levels, min_samples=min_samples).features


def idealize_multilevel(
    trace: Trace,
    events: list,
    *,
    max_levels: int = 2,
) -> MultiLevelIdealization:
    """
    Build a stepwise idealization using open pore + up to two blocked levels per event.
    """
    n = len(trace.current)
    if events:
        open_fill = float(np.median([e.i0 for e in events]))
    else:
        open_fill = float(np.median(trace.current))

    idealized = np.full(n, open_fill, dtype=float)
    level_code = np.zeros(n, dtype=int)

    for ev in events:
        start = ev.start_idx if ev.start_idx >= 0 else int(ev.start_time * trace.sample_rate)
        end = ev.end_idx if ev.end_idx >= 0 else int(ev.end_time * trace.sample_rate)
        start = max(0, min(start, n - 1))
        end = max(start, min(end, n - 1))
        segment = trace.current[start : end + 1]
        assignment = assign_event_levels(segment, float(ev.i0), max_levels=max_levels)
        for i, lab in enumerate(assignment.labels):
            idx = start + i
            idealized[idx] = float(assignment.centers[int(lab)])
            level_code[idx] = int(lab) + 1  # 1 or 2

    return MultiLevelIdealization(
        time=np.asarray(trace.time, dtype=float),
        idealized=idealized,
        level_code=level_code,
        events=list(events),
    )
