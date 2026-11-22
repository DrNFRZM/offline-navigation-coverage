"""Diagnostic holdout and the three training-data conditions.

All selections are sets of transition indices into the flattened recording. A transition is the
stored tuple (o_t, a_t, r_t, o_{t+1}) of one Minari episode, so selecting or dropping
transitions never joins observations that were not consecutive in the recording.
"""

from __future__ import annotations

import hashlib

import numpy as np


def episode_bounds(episode_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique episode ids (in recording order), start index and length of each episode."""
    ids, starts, lengths = [], [], []
    change = np.flatnonzero(np.diff(episode_id) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(episode_id)]])
    ids = episode_id[starts]
    lengths = ends - starts
    return ids, starts, lengths


def holdout_split(episode_id: np.ndarray, block_starts: list[float], block_fraction: float,
                  gap_steps: int) -> dict:
    """Contiguous holdout blocks made of whole stored episodes, separated from training by gaps.

    For each start fraction f the block begins with the first episode whose first transition is at
    or after f * N and extends over whole episodes until it holds >= block_fraction * N transitions.
    Whole episodes adjacent to each block are assigned to a gap (excluded from both training and
    holdout) until each gap holds >= gap_steps transitions. Because the recording is one continuous
    trajectory, the gaps prevent the transitions immediately before/after a holdout block from
    being used for training.
    """
    n = len(episode_id)
    _, starts, lengths = episode_bounds(episode_id)
    n_ep = len(starts)
    role = np.zeros(n_ep, dtype=np.int8)  # 0 = pool, 1 = holdout, 2 = gap
    target = int(round(block_fraction * n))
    for f in block_starts:
        first = int(np.searchsorted(starts, int(f * n), side="left"))
        if first >= n_ep:
            raise ValueError("holdout block start beyond the recording")
        k, size = first, 0
        while k < n_ep and size < target:
            if role[k] != 0:
                raise ValueError("holdout blocks overlap; reduce block_fraction or gap_steps")
            role[k] = 1
            size += lengths[k]
            k += 1
        # trailing gap
        g = 0
        while k < n_ep and g < gap_steps:
            if role[k] == 1:
                raise ValueError("gap overlaps another holdout block")
            role[k] = 2
            g += lengths[k]
            k += 1
        # leading gap
        k, g = first - 1, 0
        while k >= 0 and g < gap_steps:
            if role[k] == 1:
                raise ValueError("gap overlaps another holdout block")
            role[k] = 2
            g += lengths[k]
            k -= 1
    per_transition = np.repeat(role, lengths)
    return {
        "holdout": per_transition == 1,
        "gap": per_transition == 2,
        "pool": per_transition == 0,
        "episode_role": role,
    }


def make_conditions(eligible: np.ndarray, region: np.ndarray, retention: float, data_seed: int,
                    streams: tuple[int, int] = (1, 2)) -> dict:
    """Index sets for the full, restricted and count-matched uniform conditions.

    eligible : boolean mask of the eligible training pool (not holdout/gap, goal-consistent)
    region   : boolean mask of transitions whose origin position lies in the restricted region

    restricted = all eligible transitions outside the region + exactly round(retention * n_region)
                 eligible in-region transitions drawn uniformly without replacement
    uniform    = |restricted| transitions drawn uniformly without replacement from the eligible pool
    """
    full = np.flatnonzero(eligible)
    in_region = np.flatnonzero(eligible & region)
    out_region = np.flatnonzero(eligible & ~region)
    n_keep = int(round(retention * len(in_region)))
    rng_r = np.random.default_rng([int(data_seed), int(streams[0])])
    rng_u = np.random.default_rng([int(data_seed), int(streams[1])])
    kept = rng_r.choice(in_region, size=n_keep, replace=False)
    restricted = np.sort(np.concatenate([out_region, kept]))
    uniform = np.sort(rng_u.choice(full, size=len(restricted), replace=False))
    return {"full": full, "restricted": restricted, "uniform": uniform}


def indices_digest(idx: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.sort(idx).astype(np.int64)).tobytes()).hexdigest()


def save_manifest(path, idx: np.ndarray, n_total: int) -> str:
    """Store a selection as a packed bitmask over all recorded transitions; returns its digest."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(n_total, dtype=bool)
    mask[idx] = True
    np.savez_compressed(path, packed=np.packbits(mask), n_total=np.int64(n_total))
    return indices_digest(idx)


def load_manifest(path) -> np.ndarray:
    z = np.load(path)
    n = int(z["n_total"])
    mask = np.unpackbits(z["packed"])[:n].astype(bool)
    return np.flatnonzero(mask)
