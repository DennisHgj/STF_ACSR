"""Convert CSPM hand labels into the phoneme prompt matrix used by MFM."""

import json
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


TOKEN_TO_ID = {
    "<blank>": 0,
    "<unk>": 1,
    "b": 2,
    "p": 3,
    "m": 4,
    "f": 5,
    "d": 6,
    "t": 7,
    "n": 8,
    "l": 9,
    "g": 10,
    "k": 11,
    "h": 12,
    "j": 13,
    "q": 14,
    "x": 15,
    "zh": 16,
    "ch": 17,
    "sh": 18,
    "r": 19,
    "z": 20,
    "c": 21,
    "s": 22,
    "y": 23,
    "w": 24,
    "yu": 25,
    "a": 26,
    "o": 27,
    "e": 28,
    "i": 29,
    "u": 30,
    "v": 31,
    "ai": 32,
    "ei": 33,
    "ao": 34,
    "ou": 35,
    "er": 36,
    "an": 37,
    "en": 38,
    "ang": 39,
    "eng": 40,
    "ong": 41,
    "-": 42,
}

# Backwards-compatible public name used by the released code.
hashmap = TOKEN_TO_ID

POSITION_TO_VOWELS = {
    0: ("an", "e", "o"),
    1: ("a", "ou", "er", "en"),
    2: ("i", "v", "ang"),
    3: ("ai", "u", "ao"),
    4: ("eng", "ong", "ei"),
}

SHAPE_TO_CONSONANTS = {
    0: ("p", "d", "zh"),
    1: ("k", "q", "z"),
    2: ("s", "r", "h"),
    3: ("b", "n", "yu"),
    4: ("m", "t", "f"),
    5: ("l", "x", "w"),
    6: ("g", "j", "ch"),
    7: ("y", "c", "sh"),
}


def label2phone(
    hand_position: int, hand_shape: int
) -> Tuple[Sequence[str], Sequence[str]]:
    """Map one Mandarin CS position/shape pair to candidate phonemes."""
    if hand_position not in POSITION_TO_VOWELS:
        raise ValueError(f"hand_position must be in [0, 4], got {hand_position}.")
    if hand_shape not in SHAPE_TO_CONSONANTS:
        raise ValueError(f"hand_shape must be in [0, 7], got {hand_shape}.")
    return POSITION_TO_VOWELS[hand_position], SHAPE_TO_CONSONANTS[hand_shape]


def load_npy(npy_path: str) -> np.ndarray:
    positions = np.load(npy_path)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "Hand positions must have shape [frames, 2], "
            f"received {positions.shape}."
        )
    return positions


def group_elements2(
    indices: Iterable[int], index_distance_threshold: int = 2
) -> List[List[int]]:
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


def screen_slow_motion_group(
    hand_positions: np.ndarray,
    speed_threshold: float = 6.0,
    index_distance_threshold: int = 2,
) -> List[List[int]]:
    positions = np.asarray(hand_positions)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "Hand positions must have shape [frames, 2], "
            f"received {positions.shape}."
        )
    if len(positions) < 2:
        return []
    motion = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    slow_indices = np.flatnonzero(motion <= speed_threshold) + 1
    return group_elements2(slow_indices, index_distance_threshold)


def get_keyframe_groups(
    position_path: str,
    speed_threshold: float = 6.0,
    index_distance_threshold: int = 2,
) -> List[List[int]]:
    return screen_slow_motion_group(
        load_npy(position_path), speed_threshold, index_distance_threshold
    )


def select_compatible_groups(
    hand_positions: np.ndarray,
    expected_count: int,
    speed_threshold: float = 6.0,
    index_distance_threshold: int = 2,
) -> List[List[int]]:
    """Select paper grouping, with a fallback for released legacy JSON.

    Early generated recognition files grouped only strictly consecutive slow
    frames (theta=1), while Algorithm 1 and newly generated files use theta=2.
    The result count makes the intended convention unambiguous.
    """
    candidates = []
    for threshold in (index_distance_threshold, 1, 2):
        if threshold in candidates:
            continue
        candidates.append(threshold)
        groups = screen_slow_motion_group(
            hand_positions, speed_threshold, threshold
        )
        if len(groups) == expected_count:
            return groups
    counts = {
        threshold: len(
            screen_slow_motion_group(
                hand_positions, speed_threshold, threshold
            )
        )
        for threshold in candidates
    }
    raise ValueError(
        "CSPM result count does not match any supported keyframe grouping: "
        f"results={expected_count}, group_counts={counts}."
    )


def _shape_label(result: Mapping[str, object]) -> int:
    if "hand_shape" in result:
        return int(result["hand_shape"])
    if "hand_gesture" in result:
        return int(result["hand_gesture"])
    raise KeyError("Each CSPM result must contain hand_shape or hand_gesture.")


def _ordered_results(
    hand_results: Sequence[Mapping[str, object]], keyframes: Sequence[int]
) -> List[Mapping[str, object]]:
    if len(hand_results) != len(keyframes):
        raise ValueError(
            "CSPM result count does not match keyframe count: "
            f"{len(hand_results)} != {len(keyframes)}."
        )

    ids = [result.get("frame_id") for result in hand_results]
    if all(frame_id is not None for frame_id in ids):
        integer_ids = [int(frame_id) for frame_id in ids]
        if len(set(integer_ids)) != len(integer_ids):
            raise ValueError("CSPM result frame_id values must be unique.")
        if set(integer_ids) == set(int(index) for index in keyframes):
            by_id = {
                int(result["frame_id"]): result
                for result in hand_results
            }
            return [by_id[int(index)] for index in keyframes]
    return list(hand_results)


def build_hand_prompt_array(
    hand_results: Sequence[Mapping[str, object]],
    slow_groups: Sequence[Sequence[int]],
    frame_num: int,
    *,
    keyframes: Sequence[int] = (),
    vocabulary_size: int = 44,
) -> np.ndarray:
    """Build H' in R^(T x Z) from CSPM recognition and slow-motion groups."""
    if frame_num < 0:
        raise ValueError("frame_num must be non-negative.")
    if vocabulary_size <= max(TOKEN_TO_ID.values()):
        raise ValueError("vocabulary_size is too small for the Mandarin token map.")
    if len(hand_results) != len(slow_groups):
        raise ValueError(
            "CSPM result count does not match slow-motion group count: "
            f"{len(hand_results)} != {len(slow_groups)}."
        )

    resolved_keyframes = (
        [int(index) for index in keyframes]
        if keyframes
        else [int(group[(len(group) - 1) // 2]) for group in slow_groups]
    )
    ordered = _ordered_results(hand_results, resolved_keyframes)
    matrix = np.zeros((frame_num, vocabulary_size), dtype=np.float32)

    for result, group in zip(ordered, slow_groups):
        if not group:
            raise ValueError("Slow-motion groups must not be empty.")
        start, end = int(group[0]), int(group[-1])
        if start < 0 or end >= frame_num:
            raise ValueError(
                f"Slow-motion group [{start}, {end}] exceeds {frame_num} frames."
            )
        hand_position = int(result["hand_position"])
        hand_shape = _shape_label(result)
        vowels, consonants = label2phone(hand_position, hand_shape)
        token_ids = [TOKEN_TO_ID[token] for token in (*vowels, *consonants)]
        matrix[start : end + 1, token_ids] = 1.0
    return matrix


def load_hand_recog(
    hand_recog_path: str,
    hand_position_path: str,
    frame_num: int,
    *,
    speed_threshold: float = 6.0,
    index_distance_threshold: int = 2,
):
    """Load CSPM JSON and return the torch hand-prompt matrix."""
    import torch

    with open(hand_recog_path, "r", encoding="utf-8") as stream:
        hand_data: Dict[str, object] = json.load(stream)
    hand_results = hand_data.get("recog_results")
    if not isinstance(hand_results, list):
        raise ValueError("CSPM JSON must contain a recog_results list.")

    keyframes = hand_data.get("frame_index", [])
    if not isinstance(keyframes, list):
        raise ValueError("CSPM frame_index must be a list.")
    positions = load_npy(hand_position_path)
    slow_groups = select_compatible_groups(
        positions,
        len(hand_results),
        speed_threshold,
        index_distance_threshold,
    )
    array = build_hand_prompt_array(
        hand_results,
        slow_groups,
        frame_num,
        keyframes=keyframes,
    )
    return torch.from_numpy(array)
