import numpy as np
import scipy as sci

"""
    # 模拟向量
"""
key1 = np.array([1, 0, 0])
key2 = np.array([0, 1, 0])
key3 = np.array([1, 1, 0])
key4 = np.array([0, 0, 1])

query = np.array([0, 1, 1])     # decoder初试向量


"""
    # 相关性（得分）计算
"""
scores = np.array([np.dot(query, key1),
                   np.dot(query, key2),
                   np.dot(query, key3),
                   np.dot(query, key4)])
# print(scores)       # [0 1 1 1]

"""
    # 求注意力权重
"""
weights = np.exp(scores) / sum(np.exp(scores))
# print(weights)      # [0.10923177 0.29692274 0.29692274 0.29692274]


context_value = weights * np.array([key1, key2, key3, key4]).T
print(context_value)
# [[0.10923177 0.         0.29692274 0.        ]
#  [0.         0.29692274 0.29692274 0.        ]
#  [0.         0.         0.         0.29692274]]
# print(context_value.T)
# print(sum(context_value))       # [0.10923177 0.29692274 0.59384548 0.29692274]
# print(sum(context_value.T))     # [0.40615452 0.59384548 0.29692274]
print(np.sum(context_value, axis=1))        # [0.40615452 0.59384548 0.29692274]
