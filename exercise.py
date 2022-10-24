import numpy as np


def allocate(availability):
    breakfast_count = np.zeros((10, 5), dtype=int) + 5
    # print(breakfast_count)
    dinner_count = np.zeros((10, 5), dtype=int) + 5
    # print(dinner_count)

    for i in range(10):
        for j in range(5):
            if availability[i][j] == 1:
                breakfast_count[i][j] = j
            if availability[i][j] == 2:
                # print(availability[i][j])
                dinner_count[i][j] = j
            if availability[i][j] == 3:
                breakfast_count[i][j] = j
                dinner_count[i][j] = j

    print(breakfast_count)
    print(dinner_count)
    breakfast = np.zeros(10, dtype=int) - 1
    # print(breakfast)
    dinner = np.zeros(10, dtype=int) - 1
    # print(dinner)

    count = np.array([0, 0, 0, 0, 0], dtype=int)
    # print(count)

    n = availability.shape[0]
    min = int(np.floor(0.36 * n))
    max = int(np.ceil(0.44 * n))
    # print(min,max)
    limit = np.floor(0.1 * n)

    for i in range(10):
        # print(i)
        # for j in range(5):
        for j in range(5):
            # print(breakfast)
            # 如果第i天没人做早饭 且 第j个人没有做够3顿 且 第j个人第i天可以做早饭
            if (breakfast[i] == -1) and (count[j] <= max) and (breakfast_count[i][j] == j):
                breakfast[i] = j
                count[j] = count[j]+1
        print('breakfast', breakfast, count)
    print(count)
    for i in range(10):
        # print(count)
        # for j in range(5):
        for j in range(5):
            # print(count)
            # 如果第i天没人做晚饭 且 第j个人没有做够5顿 且 第j个人第i天可以做晚饭 且 第j个人在第i天没有做早饭
            if (dinner[i] == -1) and (count[j] < max) and (dinner_count[i][j] == j) and (breakfast[i] != j):
                dinner[i] = j
                count[j] = count[j]+1
        print(count)
        print('dinner', dinner)
    # print(breakfast)
    # print(dinner)
    # """
    restaurant = 0
    for i in range(10):
        if breakfast[i] == -1 and restaurant <= limit:
            breakfast[i] = 5
            restaurant = restaurant+1
        if dinner[i] == -1 and restaurant <= limit:
            dinner[i] = 5
            restaurant = restaurant+1
    # print(breakfast[breakfast<0])
    if breakfast[breakfast<0] is None:
        return None
    else:
        result = (breakfast.tolist(), dinner.tolist())
        # print(result)
        return result
    # """


if __name__ == '__main__':
    availability = [[2, 0, 2, 1, 2], [3, 3, 1, 0, 0],
                    [0, 1, 0, 3, 0], [0, 0, 2, 0, 3],
                    [1, 0, 0, 2, 1], [0, 0, 3, 0, 2],
                    [0, 2, 0, 1, 0], [1, 3, 3, 2, 0],
                    [0, 0, 1, 2, 1], [2, 0, 0, 3, 0]]
    availability = np.array(availability)
    print(allocate(availability))