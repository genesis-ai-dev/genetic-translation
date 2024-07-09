import numpy as np

def l_shift(lst, target_seq, n):
    arr = np.array([i for i in lst if i != ''], dtype=object)
    target_len = len(target_seq)

    # Find the starting indices of the target sequences
    indices = [i for i in range(len(arr) - target_len + 1) if np.array_equal(arr[i:i + target_len], target_seq)]

    if len(indices) == 0:
        return lst  # No matching sequences found

    # Remove all occurrences of the target sequences
    for index in reversed(indices):
        arr = np.delete(arr, slice(index, index + target_len))

    # Calculate new positions
    new_indices = np.clip(np.array(indices) + n, 0, len(arr))

    # Insert target sequences at new positions
    for i in sorted(new_indices, reverse=(n > 0)):
        for j, val in enumerate(target_seq):
            arr = np.insert(arr, i + j, val)

    return arr.tolist()



def l_replace(lst, old_seq, new_seq):
    lst = " ".join(lst)
    old_seq = " ".join(old_seq)
    new_seq = " ".join(new_seq)
    result = lst.replace(old_seq, new_seq)

    return result.split()


import numpy as np


def similarity(arr1, arr2):
    # Ensure arrays are numpy arrays
    arr1, arr2 = np.array(arr1), np.array(arr2)

    # Get the length of the longer array
    max_len = max(len(arr1), len(arr2))

    # Pad the shorter array with empty strings
    arr1 = np.pad(arr1, (0, max_len - len(arr1)), constant_values='')
    arr2 = np.pad(arr2, (0, max_len - len(arr2)), constant_values='')

    # Calculate exact matches
    exact_matches = np.sum(arr1 == arr2)

    # Calculate position shifts
    position_similarity = 0
    for word in set(arr1) | set(arr2):
        if word == '':
            continue
        indices1 = np.where(arr1 == word)[0]
        indices2 = np.where(arr2 == word)[0]
        if len(indices1) > 0 and len(indices2) > 0:
            # Calculate the minimum position difference
            min_diff = np.min(np.abs(indices1[:, np.newaxis] - indices2))
            position_similarity += 1 / (1 + min_diff)  # Decay function

    # Combine exact matches and position similarity
    total_similarity = (exact_matches + position_similarity) / max_len

    return total_similarity * 100

