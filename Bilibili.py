import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
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

# """
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
pe_result = pe(x)
# print("pe_result:", pe_result)
# print(pe_result.shape)    # (2,4,512)
# """

# position = torch.arange(0, 5)
# print(position)
# print(position.unsqueeze(1))
# print(position.unsqueeze(1).unsqueeze(0))

"""
# 绘制词汇向量中特征的分布曲线

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
# 生成掩码张量
def subsequent_mask(size):
    """生成向后遮掩的掩码张量, 参数size是掩码张量最后两个维度的大小, 它的最后两维形成一个方阵"""
    # 首先定义掩码张量的形状，1起到扩充维度的作用
    attn_shape = (1, size, size)

    # 然后使用np.ones方法向这个形状中添加元素1，使用np.triu方法生成上三角阵
    # 最后为了节约空间，再使其中的数据类型变为无符号8位整型uint8
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 最后将numpy类型转化为torch中的tensor, 内部做一个1-的操作,
    # 在这其实是做一个三角阵的反转, subsequent_mask中的每个元素都会被1减
    return torch.from_numpy(1 - subsequent_mask)

"""
exam = np.triu([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12]], k=0)
"""
"""
    k=0代表保证主对角线以上的所有部分不被置0
    k=1代表保证主对角线右上一行的所有部分不被置0
    k=-1代表保证主对角线左下一行的所有部分不被置0
"""
# print(exam)
# [[1 2 3]
#  [0 5 6]
#  [0 0 9]
#  [0 0 0]]

"""
# 生成的掩码张量的最后两维的大小
size = 20
sm = subsequent_mask(size)
print("sm:", sm)
print(sm[0])

# 掩码张量的可视化
plt.figure(figsize=(5, 5))
plt.imshow(subsequent_mask(20)[0])
# plt.show()
"""


# 注意力机制
def attention(query, key, value, mask=None, dropout=None):
    """
    注意力机制的实现, 输入分别是query, key, value, mask: 掩码张量,
           dropout是nn.Dropout层的实例化对象, 默认为None
    """
    # 首先取query最后一维的大小, 一般情况下就等同于词嵌入维度
    d_k = query.size(-1)
    # 按照注意力公式, 将query与key的转置相乘, 这里key是将最后两个维度进行转置, 再除以缩放系数根号下d_k, 这种计算方法也称为缩放点积注意力计算
    # 得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 接着判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法, 将掩码张量和scores张量每个位置一一比较, 如果掩码张量处为0，
        # 则对应的scores张量用一个极小的值来替换
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一维进行softmax操作, 获得最终的注意力张量
    p_attn = F.softmax(scores, dim=-1)

    # 之后判断是否使用dropout进行随机置0
    if dropout is not None:
        # 将p_attn传入dropout对象中进行'丢弃'处理
        p_attn = dropout(p_attn)

    # 最后, 根据公式将p_attn与value张量相乘获得最终的query注意力表示, 同时返回注意力张量
    return torch.matmul(p_attn, value), p_attn

"""
query = pe_result
d_k = query.size(-1)
# print(d_k)                                  # 512
# print(query.transpose(-2, -1).shape)        # torch.Size([2, 512, 4])
scores = torch.matmul(query, query.transpose(-2, -1)) / math.sqrt(d_k)
print(scores)
p_attn = F.softmax(scores, dim=-1)
print(p_attn)
"""
"""
input = Variable(torch.randn(5, 5))
# print(input)

mask = Variable(torch.zeros(5, 5))
# print(mask)

input.masked_fill(mask == 0, -1e9)

# 我们令输入的query, key, value都相同, 位置编码的输出
query = key = value = pe_result
# 令mask为一个2x4x4的零张量
mask = Variable(torch.zeros(2, 4, 4))
# attn, p_attn = attention(query, key, value)  # attn(2,4,512) p_attn(2,4,4)
attn, p_attn = attention(query, key, value, mask=mask)
print("attn:", attn)
print("p_attn:", p_attn)
"""


# 首先需要定义克隆函数，因为在多头注意力机制中要用到多个结构相同的线性层。
# 我们将使用clone函数将这些线性层一同初始化在一个网络层列表对象中。
def clones(module, N):
    """用于生成相同网络层的克隆函数, 它的参数module表示要克隆的目标网络层, N代表需要克隆的数量"""
    # 在函数中, 我们通过for循环对module进行N次深度拷贝，使每个module成为独立的层，然后将其放在nn.ModuleList类型的列表中
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 实现多头注意力机制的类
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        在类的初始化时, 会传入三个参数，head代表头数，embedding_dim代表词嵌入的维度，
           dropout代表进行dropout操作时置0比率，默认是0.1
        """
        super(MultiHeadedAttention, self).__init__()

        # 首先使用了一个assert语句判断head是否能被embedding_dim整除，
        # 因为我们之后要给每个头分配等量的词特征，也就是embedding_dim/head个
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head

        # 传入头数head
        self.head = head

        # 然后获得线性层对象，通过nn.Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，然后使用clones函数克隆四个。
        # 为什么是四个呢？这是因为在多头注意力中，Q，K，V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个。
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn代表最后得到的注意力张量
        self.attn = None

        # 最后就是一个self.dropout对象，它通过nn.Dropout实例化
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向逻辑函数, 它的输入参数有四个，前三个就是注意力机制需要的Q, K, V，
           最后一个是注意力机制中可能需要的mask掩码张量，默认是None.
        """

        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeze拓展维度
            mask = mask.unsqueeze(0)

        # batch_size代表有多少条样本，它是query矩阵的第1个维度。
        batch_size = query.size(0)

        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV与三个线性层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值.然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.这样我们就得到了每个头的输入.
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]

        # 得到每个头的输入后，接下来就是将他们传入到attention中，
        # 这里直接调用我们之前实现的attention函数.同时也将mask和dropout传入其中.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法，
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 所以，下一步就是使用view重塑形状，变成和输入形状相同.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出.
        return self.linears[-1](x)
# """
# 实例化若干参数
# 头数head
head = 8
# 词嵌入维度embedding_dim
embedding_dim = 512
# 置零比率dropout
dropout = 0.2
# 假设输入的Q，K，V仍然相等
query = value = key = pe_result
# 输入的掩码张量mask
mask = Variable(torch.zeros(8, 4, 4))

mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
# print(mha_result)           # (2,4,512)
# """


# 构建前馈全连接网络类
# 通过类PositionwiseFeedForward来实现加入位置编码的前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
            初始化函数有三个输入参数分别是d_model,d_ff和dropout=0.1，
            d_model: 线性层的输入维度也是第二个线性层的输出维度，因为通过前馈全连接层后输入和输出的维度不变。
            d_ff: 前馈全连接层的中间维度，即第二个线性层的输入维度和第一个线性层的输出维度。
            dropout: dropout置0比率。
        """
        super(PositionwiseFeedForward, self).__init__()

        # 首先实例化了两个线性层对象：self.w1和self.w2
        # 它们的参数分别是d_model, d_ff和d_ff, d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 然后使用nn.Dropout实例化对象self.dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """输入参数为x，代表来自上一层的输出"""
        # 首先经过第一个线性层，然后使用relu函数进行激活,
        # 再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果。
        return self.w2(self.dropout(F.relu(self.w1(x))))

# """
d_model = 512
# 线性变化的维度
d_ff = 64
dropout = 0.2
# 输入参数x可以是多头注意力机制的输出
x = mha_result
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
# print(ff_result)    # (2,4,512)
# """


# 通过LayerNorm实现规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
            初始化函数有两个参数。
            features: 表示词嵌入的维度;
            eps: 是一个足够小的数，在规范化公式的分母中出现，防止分母为0，默认是1e-6。
        """
        super(LayerNorm, self).__init__()

        # 根据features的形状初始化两个参数张量a2，和b2，第一个初始化为1张量，
        # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数，
        # 因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，
        # 使其既能满足规范化要求，又能不改变针对目标的表征。最后使用nn.parameter封装，代表他们是模型的参数。
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        # 把eps传到类中
        self.eps = eps

    def forward(self, x):
        """输入参数x代表来自上一层的输出"""
        # 在函数中，首先对输入变量x求最后一个维度的均值，并保持输出维度与输入维度一致。
        # 接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果，
        # 最后对结果乘以我们的缩放参数a2，*号代表同型点乘，即对应位置进行乘法操作，加上位移参数b2
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


# """
features = d_model = 512
eps = 1e-6
# 输入x来自前馈全连接层的输出
x = ff_result

ln = LayerNorm(features, eps)
ln_result = ln(x)
# print(ln_result)    # (2,4,512)
# """


# 使用SublayerConnection来实现子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        输入参数有两个，size以及dropout.
        :param size: 词嵌入维度的大小.
        :param dropout: 输出矩阵的随机置0的比率.
        """
        super(SublayerConnection, self).__init__()
        # 实例化了规范化层对象self.norm
        self.norm = LayerNorm(size)
        # 使用nn.Droupout实例化一个self.dropout对象.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """前向逻辑函数中, 接收上一个层或者子层的输入作为第一个参数，
           将该子层连接中的子层函数作为第二个参数"""

        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作，
        # 随机停止一些网络中神经元的作用，来防止过拟合. 最后还有一个add操作，
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出
        return x + self.dropout(sublayer(self.norm(x)))


size = 512
dropout = 0.2
head = 8
d_model = 512

# 令x为位置编码器的输出
x = pe_result
mask = Variable(torch.zeros(8, 4, 4))

# 假设子层中装的是多头注意力层，实例化这个类
self_attn = MultiHeadedAttention(head, d_model)

# 使用lambda获得一个函数类型的子层
"""
    lambda [parameters]: expression
    关键字lamdbda 参数: 函数体
    Lambda是一种不需要名字（即标识符）、由一个单独表达式成的匿名内联函数，函数体会在调用时被求值。
"""
sublayer = lambda x: self_attn(x, x, x, mask)

sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
# print(sc_result)                # (2,4,512)


# 编码器层
# 使用EncoderLayer类实现编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
            初始化函数参数有四个，分别是
            :param size: 词嵌入维度的大小，它也将作为我们编码器层的大小
            :param self_attn: 将传入多头自注意力子层实例化对象, 并且是自注意力机制
            :param feed_forward: 将传入前馈全连接层实例化对象
            :param dropout: 置0比率
        """
        super(EncoderLayer, self).__init__()

        # 首先将self_attn和feed_forward传入
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 把size传入其中
        self.size = size

    def forward(self, x, mask):
        """forward函数中有两个输入参数，x和mask，分别代表上一层的输出，和掩码张量mask。"""
        # 首先通过第一个子层连接结构，其中包含多头自注意力子层，
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层。
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


size = 512
head = 8
d_model = 512
d_ff = 64
x = pe_result
dropout = 0.2
self_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(8, 4, 4))

el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)
# print(el_result)                # （2,4,512）
# print(el_result.shape)


# 使用Encoder类来实现编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """初始化函数的两个参数分别代表编码器层和编码器层的个数"""
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 再初始化一个规范化层, 它将用在编码器的最后面
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """forward函数的输入和编码器层相同, x代表上一层的输出, mask代表掩码张量"""
        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程，就相当于输出的x经过了N个编码器层的处理。
        # 最后再通过规范化层的对象self.norm进行处理，返回结果。
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 第一个实例化参数layer, 它是一个编码器层的实例化对象, 因此需要传入编码器层的参数
# 又因为编码器层中的子层是不共享的, 因此需要使用深度拷贝各个对象
size = 512
head = 8
d_model = 512
d_ff = 64
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)

# 编码器中编码器层的个数N
N = 8
mask = Variable(torch.zeros(8, 4, 4))

en = Encoder(layer, N)
en_result = en(x, mask)
# print(en_result)
# print(en_result.shape)        # （2,4,512）

# --------------------Dncoder部分-------------------- #