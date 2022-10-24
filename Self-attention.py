import numpy as np

"""
    # 模拟向量
"""
key1 = np.array([1, 0, 0])
key2 = np.array([0, 1, 0])
key3 = np.array([1, 1, 0])
key4 = np.array([0, 0, 1])

"""
    # 相关性（得分）计算
"""
score1 = [
    np.dot(key1.T, key1),
    np.dot(key1.T, key2),
    np.dot(key1.T, key3),
    np.dot(key1.T, key4),
]
# print(score1)           # [1, 0, 1, 0]
score2 = [
    np.dot(key2.T, key1),
    np.dot(key2.T, key2),
    np.dot(key2.T, key3),
    np.dot(key2.T, key4),
]
score3 = [
    np.dot(key3.T, key1),
    np.dot(key3.T, key2),
    np.dot(key3.T, key3),
    np.dot(key3.T, key4),
]
score4 = [
    np.dot(key4.T, key1),
    np.dot(key4.T, key2),
    np.dot(key4.T, key3),
    np.dot(key4.T, key4),
]
# print(score2, score3, score4)   # [0, 1, 1, 0] [1, 1, 2, 0] [0, 0, 0, 1]

"""
    # 求注意力权重
"""
weight1 = np.exp(score1) / sum(np.exp(score1))
# print(weight1)          # [0.36552929 0.13447071 0.36552929 0.13447071]
weight2 = np.exp(score2) / sum(np.exp(score2))
weight3 = np.exp(score3) / sum(np.exp(score3))
weight4 = np.exp(score4) / sum(np.exp(score4))
# print(weight2, weight3, weight4)
# [0.13447071 0.36552929 0.36552929 0.13447071]
# [0.19661193 0.19661193 0.53444665 0.07232949]
# [0.1748777  0.1748777  0.1748777  0.47536689]

weighted_value1 = weight1 * np.array([key1, key2, key3, key4]).T
# print(weighted_value1)
context_value1 = np.sum(weighted_value1, axis=1)
# print(context_value1)   # [0.73105858 0.5        0.13447071]
weighted_value2 = weight2 * np.array([key1, key2, key3, key4]).T
# print(weighted_value1)
context_value2 = np.sum(weighted_value2, axis=1)
# print(context_value2)   # [0.5        0.73105858 0.13447071]
weighted_value3 = weight3 * np.array([key1, key2, key3, key4]).T
# print(weighted_value1)
context_value3 = np.sum(weighted_value3, axis=1)
# print(context_value3)   # [0.73105858 0.73105858 0.07232949]
weighted_value4 = weight1 * np.array([key1, key2, key3, key4]).T
# print(weighted_value1)
context_value4 = np.sum(weighted_value4, axis=1)
# print(context_value4)   # [0.73105858 0.5        0.13447071]

"""
    # 融合
"""
integration_value = np.sum([context_value1,
                            context_value2,
                            context_value3,
                            context_value4], axis=0)
# print(integration_value)    # [2.69317574 2.46211716 0.47574162]

"""
    # 增加可学习的参数(Trick 1)
"""
import torch

"STEP1: 准备输入"
x = [(1, 0, 1, 0),
     (0, 2, 0, 2),
     (1, 1, 1, 1)]
x = torch.tensor(x, dtype=torch.float32)
# print(x)
# tensor([[1., 0., 1., 0.],
#         [0., 2., 0., 2.],
#         [1., 1., 1., 1.]])

"STEP2: 初始化参数（增加可学习的参数）"
w_key = [(0, 1, 1),
         (1, 1, 0),
         (0, 1, 0),
         (1, 1, 0)]
w_key = torch.tensor(w_key, dtype=torch.float32)
# print(w_key.shape)      # torch.Size([4, 3])
# print(w_key)
# tensor([[0., 1., 1.],
#         [1., 1., 0.],
#         [0., 1., 0.],
#         [1., 1., 0.]])
w_query = [(1, 0, 1),
           (1, 0, 0),
           (0, 0, 0),
           (0, 1, 1)]
w_query = torch.tensor(w_query, dtype=torch.float32)
# print(w_query.shape)      # torch.Size([4, 3])
# print(w_query)
# tensor([[1., 0., 1.],
#         [1., 0., 0.],
#         [0., 0., 0.],
#         [0., 1., 1.]])
w_value = [(0, 2, 0),
           (0, 3, 0),
           (1, 0, 3),
           (1, 1, 0)]
w_value = torch.tensor(w_value, dtype=torch.float32)
# print(w_query.shape)      # torch.Size([4, 3])
# print(w_value)
# tensor([[0., 2., 0.],
#         [0., 3., 0.],
#         [1., 0., 3.],
#         [1., 1., 0.]])

# 初始化法二：使用神经网络初始化权重（服从高斯分布）
w = torch.Tensor(4, 3)
# print(torch.nn.init.uniform_(w))

"STEP3: 获取key, query和value"
keys = x @ w_key
# print(x.shape, w_key.shape, keys.shape)     # torch.Size([3, 4]) torch.Size([4, 3]) torch.Size([3, 3])
# print(keys)
# tensor([[0., 2., 1.],
#         [4., 4., 0.],
#         [2., 4., 1.]])
querys = x @ w_query
values = x @ w_value
# print(querys)
# tensor([[1., 0., 1.],
#         [2., 2., 2.],
#         [2., 1., 2.]])
# print(values)
# tensor([[1., 2., 3.],
#         [2., 8., 0.],
#         [2., 6., 3.]])

"STEP4: 计算attention scores"
attn_scores = querys @ keys.T
# print(attn_scores)
# tensor([[ 1.,  4.,  3.],
#         [ 6., 16., 14.],
#         [ 4., 12., 10.]])

"STEP5: 计算attention scores的softmax"
from torch.nn.functional import softmax
attn_scores_softmax = softmax(attn_scores, dim=-1)
# print(attn_scores_softmax.shape)    # torch.Size([3, 3])
# print(attn_scores_softmax)
# tensor([[3.5119e-02, 7.0538e-01, 2.5950e-01],
#         [3.9987e-05, 8.8076e-01, 1.1920e-01],
#         [2.9539e-04, 8.8054e-01, 1.1917e-01]])

"STEP6: 给value乘上score"
weighted_values = values[:, None] * attn_scores_softmax.T[:, :, None]
# print(weighted_values)
# print(weighted_values.shape)        # torch.Size([3, 3, 3])

"STEP7: 给value加权求和获取output"
outputs = weighted_values.sum(dim=0)
# print(outputs)
# tensor([[1.9649, 7.2703, 0.8838],
#         [2.0000, 7.7614, 0.3577],
#         [1.9997, 7.7599, 0.3584]])

"STEP8: 给value加权求和获取output"
# 对 input2 和 input3 也重复的完成步骤4～7的计算。


"""
    # 缩放querys和keys点积(注意力得分)的值(Trick 2)
    # w_ij = querys·keys/sqrt(k)
"""
k = w_key.shape[1]
# print(k)                # 3
attn_scores = querys @ keys.T/np.sqrt(k)
# print(attn_scores)
# tensor([[0.5774, 2.3094, 1.7321],
#         [3.4641, 9.2376, 8.0829],
#         [2.3094, 6.9282, 5.7735]]

"""
    # 多头注意力机制(Trick 3)
    不同的层面（参数维度）的注意力不一样，每一个”头“就代表一个层面
"""
# print(w_key)
# print(w_query)
# print(w_value)

head1_w_key = torch.Tensor(4, 3)
head1_w_query = torch.Tensor(4, 3)
head1_w_value = torch.Tensor(4, 3)
# print(head1_w_key)
print(
    'head1', '\n',
    torch.nn.init.uniform_(head1_w_key), '\n',
    torch.nn.init.uniform_(head1_w_query), '\n',
    torch.nn.init.uniform_(head1_w_value),
)

head2_w_key = torch.Tensor(4, 3)
head2_w_query = torch.Tensor(4, 3)
head2_w_value = torch.Tensor(4, 3)
# print(head1_w_key)
print(
    'head2', '\n',
    torch.nn.init.uniform_(head2_w_key), '\n',
    torch.nn.init.uniform_(head2_w_query), '\n',
    torch.nn.init.uniform_(head2_w_value),
)