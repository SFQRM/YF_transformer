import torch
import torch.nn as nn
import math
from torch.autograd import Variable

# --------------------输入部分-------------------- #

"""
    文本嵌入层的作用：
    ·无论是源文本嵌入还是目标文本嵌入，都是为了将文本中词汇的数字表示转换为向量表示，希望在这样的高维空间捕捉词汇间的关系。
"""


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """ d_model:指嵌入的维度，vocab:指词表的大小"""
        super(Embeddings, self).__init__()
        # 使用nn.Embedding获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """参数x:输入给模型的文本通过词汇映射后的张量"""
        # 乘根号下d_model是为了缩放数据
        return self.lut(x) * math.sqrt(self.d_model)


"""
embedding = nn.Embedding(10, 3)
# input每个数字代表一个词
input = torch.LongTensor([[1, 2, 4, 5],
                          [4, 3, 2, 9]])
# print(input)
# tensor([[1, 2, 4, 5],
#         [4, 3, 2, 9]])
# print(embedding(input))
# torch.Size([2, 4, 3])
"""

# """
# 词嵌入维度是512维
d_model = 512
# 词表大小是1000
vocab = 1000

input_data = Variable(torch.LongTensor([[100, 2, 421, 508],
                               [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr = emb(input_data)
# print("embr:", embr)
# print(embr.shape)
# torch.Size([2, 4, 512])
# """


"""
    位置编码器的作用：
    ·因为在Transformer的编码器结构中，并没有针对词汇位置信息的处理，
     因此需要在Embedding层后加入位置编码器，
     将词汇位置不用可能会产生不同语义的信息加入到词嵌入张量中，以弥补位置信息的缺失。
"""


# 位置编码器的类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        d_model: 词嵌入的维度
        dropout: 置0比率，即让多少个神经元失效
        max_len: 每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 实例化nn中预定义的Dropout层，并将dropout传入其中，获得对象self.dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化一个位置编码矩阵，它是一个全0矩阵，矩阵的大小是max_len x d_model, 句子的长度max_len就是一个句子中单词的个数
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵，词汇的绝对位置就是用它的索引去表示
        position = torch.arange(0, max_len).unsqueeze(1)

        # 绝对位置矩阵初始化之后，需要考虑如何将位置信息加入到位置编码矩阵中
        # 最简单的思路就是先将绝对位置矩阵变换成max_len x d_model的形状，然后覆盖位置编码矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 目前得到的位置编码矩阵pe还只是一个二维矩阵，要想和embedding的输出（一个三维张量）相加，就必须拓展使用unsqueeze拓展一个维度。
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成模型的buffer，在模型保存后重加载时buffer和模型结构及参数一同被加载。
        # buffer被认为是对模型效果有帮助的，但又不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象。
        self.register_buffer('pe', pe)

    def forward(self, x):
        """forward函数的参数是x, 表示文本序列的词嵌入表示"""
        # 在相加之前我们对pe做一些适配工作， 将这个三维张量的第二维即句子最大长度切片到与输入的x的第二维相同即x.size(1)，
        # 因为默认max_len为5000太大了，很难有一个句子包含5000个词汇，所以要进行与输入张量的适配。
        # 最后使用Variable进行封装，使其与x的样式相同，但是它不需要进行梯度求解，因此把requires_grad设置成false
        x = x  + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 最后使用self.dropout对象进行'丢弃'操作, 并返回结果。
        return self.dropout(x)


"""
# 词嵌入维度是512维
d_model = 512
# 置0比率为0.1
dropout = 0.1
# 句子最大长度
max_len = 60
# 输入x是Embedding层的输出的张量, 形状是2 x 4 x 512
# embr = Embeddings(d_model, dropout)
x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)  # (2,4,512)
print("pe_result:", pe_result)
print(pe_result.shape)
"""

# position = torch.arange(0, 5)
# print(position)
# print(position.unsqueeze(1))
# print(position.unsqueeze(1).unsqueeze(0))

"""
# 绘制词汇向量中特征的分布曲线
import matplotlib.pyplot as plt
import numpy as np

# 创建一个15 x 5大小的画布
plt.figure(figsize=(15, 5))

# 实例化PositionalEncoding类得到_pe对象，输入参数是20和0
_pe = PositionalEncoding(20, 0)

# 然后向_pe传入被Variable封装的tensor，这样_pe会直接执行forward函数
# 且这个tensor里的数值都是0，被处理后相当于位置编码张量
y = _pe(Variable(torch.zeros(1, 100, 20)))
print(y)
print(y.shape)
# torch.Size([1, 100, 20])

# 然后定义画布的横纵坐标，横坐标到100的长度，纵坐标是某一个词汇中的某维特征在不同长度下对应的值
# 因为总共有20维，这里只查看4、5、6、7维的值
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())

plt.legend(["dim %d"%p for p in [4, 5, 6, 7]])
plt.show()
"""
"""
    输出效果分析：
    ·每条颜色的曲线代表某一个词汇中的特征在不同位置的含义
    ·保证同一词汇随着所在位置不同它对应位置嵌入向量会发生变化
    ·正弦波和余弦波的值域范围都是1到-1，这又很好的控制了嵌入数值的大小，有助于梯度的快速计算
"""


# --------------------Encoder部分-------------------- #


# --------------------Dncoder部分-------------------- #
