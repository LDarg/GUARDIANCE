import numpy as np

def manhattan_distance(array1, array2):
    return np.sum(np.abs(array1 - array2))

manhattan_distance(np.array([1,1], np.array([3,2])))