import unittest

import numpy as np

from LipModel_MFM.datamodule.hand_util import (
    TOKEN_TO_ID,
    build_hand_prompt_array,
    label2phone,
    select_compatible_groups,
)


class HandPromptTest(unittest.TestCase):
    def test_label_mapping_matches_mandarin_cued_speech_rules(self):
        vowels, consonants = label2phone(0, 0)
        self.assertEqual(tuple(vowels), ("an", "e", "o"))
        self.assertEqual(tuple(consonants), ("p", "d", "zh"))

    def test_build_hand_prompt_array_marks_only_the_slow_group(self):
        matrix = build_hand_prompt_array(
            [{"frame_id": 2, "hand_position": 0, "hand_shape": 0}],
            [[1, 2, 3]],
            frame_num=5,
            keyframes=[2],
        )
        active = [
            TOKEN_TO_ID[token]
            for token in ("an", "e", "o", "p", "d", "zh")
        ]
        self.assertEqual(matrix.shape, (5, 44))
        self.assertTrue(np.all(matrix[1:4, active] == 1))
        self.assertEqual(matrix[0].sum(), 0)
        self.assertEqual(matrix[4].sum(), 0)

    def test_legacy_hand_gesture_alias_is_supported(self):
        matrix = build_hand_prompt_array(
            [{"frame_id": 1, "hand_position": 4, "hand_gesture": 7}],
            [[1]],
            frame_num=3,
            keyframes=[1],
        )
        self.assertEqual(matrix[1, TOKEN_TO_ID["eng"]], 1)
        self.assertEqual(matrix[1, TOKEN_TO_ID["sh"]], 1)

    def test_result_group_mismatch_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "count"):
            build_hand_prompt_array([], [[1]], frame_num=2)

    def test_released_json_can_use_strictly_consecutive_groups(self):
        positions = np.array(
            [[0, 0], [1, 0], [20, 0], [21, 0], [40, 0], [41, 0]],
            dtype=np.float32,
        )
        self.assertEqual(
            select_compatible_groups(
                positions,
                expected_count=3,
                index_distance_threshold=2,
            ),
            [[1], [3], [5]],
        )


if __name__ == "__main__":
    unittest.main()
