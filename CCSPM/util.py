import numpy as np


def load_npy(npy_path):
    npy_file = np.load(npy_path)
    # print(hand_position.shape)
    return npy_file


def group_elements(lst):
    if not lst:
        return []

    result = []
    current_group = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] <= 2:
            current_group.append(lst[i])
        else:
            result.append(int(np.mean(current_group)))
            # result.append(current_group)
            current_group = [lst[i]]

    if current_group not in result:
        result.append(int(np.mean(current_group)))
    return result

def screen_slow_motion(hand_position):
    slow_index = []
    for i in range(1, len(hand_position)):
        # print(type(hand_position[i]))
        if np.linalg.norm(hand_position[i] - hand_position[i - 1]) <= 6:
            slow_index.append(i)
    final_slow_index = group_elements(slow_index)

    return final_slow_index

def get_keyframes(position_path):
    hand_position = load_npy(position_path)
    # print(len(hand_position))
    slow_frame_index = screen_slow_motion(hand_position)
    return slow_frame_index