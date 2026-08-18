"""Keyframe filtering described in Algorithm 1 of the STF-ACSR paper."""

from typing import Iterable, List, Sequence

import numpy as np


def load_npy(npy_path: str) -> np.ndarray:
    positions = np.load(npy_path)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "Hand positions must have shape [frames, 2], "
            f"received {positions.shape}."
        )
    return positions


def group_slow_indices(
    indices: Iterable[int], index_distance_threshold: int = 2
) -> List[List[int]]:
    """Group slow-motion indices whose adjacent index distance is at most theta."""
    if index_distance_threshold < 0:
        raise ValueError("index_distance_threshold must be non-negative.")

    ordered = [int(index) for index in indices]
    if not ordered:
        return []
    if any(right <= left for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Slow-motion indices must be strictly increasing.")

    groups = [[ordered[0]]]
    for index in ordered[1:]:
        if index - groups[-1][-1] <= index_distance_threshold:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def median_frame_index(group: Sequence[int]) -> int:
    """Return an actual middle frame from a non-empty ordered group."""
    if not group:
        raise ValueError("Cannot choose a keyframe from an empty group.")
    return int(group[(len(group) - 1) // 2])


def screen_slow_motion_groups(
    hand_positions: np.ndarray,
    speed_threshold: float = 6.0,
    index_distance_threshold: int = 2,
) -> List[List[int]]:
    """Filter slow frames by Euclidean hand-center speed and group them."""
    positions = np.asarray(hand_positions)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "Hand positions must have shape [frames, 2], "
            f"received {positions.shape}."
        )
    if speed_threshold < 0:
        raise ValueError("speed_threshold must be non-negative.")
    if len(positions) < 2:
        return []

    motion = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    # motion[j - 1] describes the transition into original frame j.
    slow_indices = np.flatnonzero(motion <= speed_threshold) + 1
    return group_slow_indices(slow_indices, index_distance_threshold)


def group_elements(
    indices: Iterable[int], index_distance_threshold: int = 2
) -> List[int]:
    """Compatibility wrapper returning one median keyframe per group."""
    return [
        median_frame_index(group)
        for group in group_slow_indices(indices, index_distance_threshold)
    ]


def screen_slow_motion(
    hand_positions: np.ndarray,
    speed_threshold: float = 6.0,
    index_distance_threshold: int = 2,
) -> List[int]:
    return [
        median_frame_index(group)
        for group in screen_slow_motion_groups(
            hand_positions, speed_threshold, index_distance_threshold
        )
    ]


def get_keyframe_groups(
    position_path: str,
    speed_threshold: float = 6.0,
    index_distance_threshold: int = 2,
) -> List[List[int]]:
    return screen_slow_motion_groups(
        load_npy(position_path), speed_threshold, index_distance_threshold
    )


def get_keyframes(
    position_path: str,
    speed_threshold: float = 6.0,
    index_distance_threshold: int = 2,
) -> List[int]:
    return [
        median_frame_index(group)
        for group in get_keyframe_groups(
            position_path, speed_threshold, index_distance_threshold
        )
    ]
