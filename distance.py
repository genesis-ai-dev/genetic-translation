import numpy as np

def l_shift(lst, target_seq, n):
    """
    :param lst: List to modify
    :param target_seq: sublist to shift
    :param n: number of shifts /and direction
    :return: modified list
    """
    arr = np.array([i for i in lst if i != ''], dtype=object)
    target_len = len(target_seq)
    indices = [i for i in range(len(arr) - target_len + 1) if np.array_equal(arr[i:i + target_len], target_seq)]

    if len(indices) == 0:
        return lst

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
    """
    TODO: make something better for lists, but for now this works.
    """
    lst = " ".join(lst)
    old_seq = " ".join(old_seq)
    new_seq = " ".join(new_seq)
    result = lst.replace(old_seq, new_seq)

    return result.split()

