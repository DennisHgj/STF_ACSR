import unittest

import numpy as np

from CSPM.util import (
    group_slow_indices,
    median_frame_index,
    screen_slow_motion,
    screen_slow_motion_groups,
)


class KeyframeFilterTest(unittest.TestCase):
    def test_paper_threshold_groups_indices_at_distance_two(self):
        self.assertEqual(
            group_slow_indices([1, 3, 6], index_distance_threshold=2),
            [[1, 3], [6]],
        )

    def test_keyframe_is_an_actual_group_member(self):
        self.assertEqual(median_frame_index([4, 5, 6, 7]), 5)

    def test_slow_motion_filter_uses_original_frame_indices(self):
        positions = np.array(
            [[0, 0], [1, 0], [20, 0], [21, 0], [40, 0], [41, 0]],
            dtype=np.float32,
        )
        groups = screen_slow_motion_groups(
            positions, speed_threshold=6, index_distance_threshold=2
        )
        self.assertEqual(groups, [[1, 3, 5]])
        self.assertEqual(screen_slow_motion(positions), [3])

    def test_keyframe_filter_rejects_bad_position_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            screen_slow_motion_groups(np.zeros((4, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
