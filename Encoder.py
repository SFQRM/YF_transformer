import math
import torch
import torch.nn.functional as F
import re
import copy
import importlib
from torch.autograd import Variable


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


# 模拟数值
key = torch.Tensor(8, 4, 3)     # (batch_size)
value = torch.Tensor(8, 4, 3)
query = torch.Tensor(8, 4, 3)

# 初始化
key = torch.nn.init.uniform_(key)
value = torch.nn.init.uniform_(value)
query = torch.nn.init.uniform_(query)

result = attention(query, key, value, d_k=3)
# print(result)
# print(result.shape)             # torch.Size([8, 4, 3])
# print(result.size(0))           # 8

"""
input_data = torch.randn(10, 2)
my_linear = torch.nn.Linear(2, 1)
print(my_linear)
print(my_linear(input_data))
"""


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):    # (模型的整体维度)
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model//heads
        self.h = heads

        # 可学习的参数
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k)
        q = self.k_linear(q).view(batch_size, -1, self.h, self.d_k)
        v = self.k_linear(v).view(batch_size, -1, self.h, self.d_k)

        # batch_
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(concat)
        return output


# mul_head = MultiHeadAttention(4, 4).forward(query, key, value)
# print(mul_head)


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model):        # (词典的大小，模型的整体维度)
        super().__init__()
        self.d_model = d_model
        # 要把词典中的每一个词转换成一个向量，维度是d_model
        self.embed = torch.nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_module = d_model

        # 根据pos和i创建一个都为0的pe矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i) / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*i) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 让embeddedings vector相对大一些
        x = x * math.sqrt(self.d_module)
        # 增加位置常量到embedding中
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x


class NormLayer(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = torch.nn.Parameter(torch.ones(self.size))
        self.bias = torch.nn.Parameter(torch.ones(self.size))
        self.eps = eps

    def forward(self, x):
        norm = {
            self.alpha * (x-x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        }
        return norm


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


def get_clone(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EncoderLayer(torch.nn.Module):
    pass


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):     # N 代表transformer层数
        super().__init__()
        self.N = N

        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionEncoder(d_model, dropout=dropout)
        self.layers = get_clone(EncoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        s = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Tokenize(object):

    def __init__(self, lang):
        self.nlp = importlib.import_module(lang).load()

    def tokenizer(self, sentence):
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


tokenize = Tokenize('zh_core_web_sm')
tokenize.tokenizer('你好，这里是中国。')