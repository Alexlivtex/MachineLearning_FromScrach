import math

def euclidean_distance(x1, x2):
    """
    Calculate the distance between tow points
    """
    assert len(x1) == len(x2)
    distance = 0
    for i in range(len(x1)):
        distance += math.pow(x1[i] - x2[i], 2)
    return math.sqrt(distance)