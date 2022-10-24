import numpy as np


def allocate(availability):
    breakfast_count = np.zeros((10, 5), dtype=int) + 5
    dinner_count = np.zeros((10, 5), dtype=int) + 5
    for i in range(10):
        for j in range(5):
            if availability[i][j] == 1:
                breakfast_count[i][j] = j
            if availability[i][j] == 2:
                dinner_count[i][j] = j
            if availability[i][j] == 3:
                breakfast_count[i][j] = j
                dinner_count[i][j] = j
    breakfast = np.zeros(10, dtype=int) - 1
    dinner = np.zeros(10, dtype=int) - 1
    count = np.array([0, 0, 0, 0, 0], dtype=int)
    n = availability.shape[0]
    min = int(np.floor(0.36 * n))
    max = int(np.ceil(0.44 * n))
    limit = np.floor(0.1 * n)
    for i in range(10):
        for j in range(5):
            if breakfast[i] == -1 and count[j] <= max and breakfast_count[i][j] == j:
                breakfast[i] = j
                count[j] = count[j]+1
    for i in range(10):
        for j in range(5):
            if dinner[i] == -1 and count[j] < max and dinner_count[i][j] == j and breakfast[i] != j:
                dinner[i] = j
                count[j] = count[j]+1
    restaurant = 0
    for i in range(10):
        if breakfast[i] == -1 and restaurant <= limit:
            breakfast[i] = 5
            restaurant = restaurant+1
        if dinner[i] == -1 and restaurant <= limit:
            dinner[i] = 5
            restaurant = restaurant+1
    if breakfast[breakfast<0] is None:
        return None
    else:
        result = (breakfast.tolist(), dinner.tolist())
        return result


if __name__ == '__main__':
    availability = [[2, 0, 2, 1, 2], [3, 3, 1, 0, 0],
                    [0, 1, 0, 3, 0], [0, 0, 2, 0, 3],
                    [1, 0, 0, 2, 1], [0, 0, 3, 0, 2],
                    [0, 2, 0, 1, 0], [1, 3, 3, 2, 0],
                    [0, 0, 1, 2, 1], [2, 0, 0, 3, 0]]
    availability = np.array(availability)
    print(allocate(availability))