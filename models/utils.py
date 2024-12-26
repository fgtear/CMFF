import torch
import torch.nn as nn
import typing
import random
import numpy
import os
import math


def seed_all(seed):
    """
    设置随机种子
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    if torch.cuda.is_available():  # GPU
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # python 全局
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class ThreePhaseLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(0.05 * total_steps)
        self.maintain_steps = int(0.55 * total_steps)
        self.cosine_steps = total_steps - self.warmup_steps - self.maintain_steps
        super(ThreePhaseLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm up phase
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        elif self.last_epoch < self.warmup_steps + self.maintain_steps:
            # Maintain phase
            return self.base_lrs

        else:
            # Cosine decay phase
            cosine_epoch = self.last_epoch - self.warmup_steps - self.maintain_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_epoch / self.cosine_steps))
            return [base_lr * cosine_factor for base_lr in self.base_lrs]


def masked_mean(output, attentions_length):
    """
    attentions_length : [5,8,3,7,6], 每个数字代表序列的有效长度
    """
    # 获取输出的形状
    bs, length, embedding = output.size()
    # 创建一个长度掩码矩阵
    length_mask = torch.arange(length).to(output.device).expand(bs, length) < attentions_length.unsqueeze(1)
    # 将长度掩码应用到输出上
    masked_output = output * length_mask.unsqueeze(-1)
    # 计算每个序列的总和
    sum_output = masked_output.sum(dim=1)
    # 计算有效长度
    lengths = attentions_length.float().unsqueeze(1)
    # 计算平均值
    mean_output = sum_output / lengths
    return mean_output


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.normalized_shape = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 计算均方根
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        # 进行归一化
        x_normalized = x / (rms + self.eps)
        # 应用缩放参数
        return self.scale * x_normalized


# class RMSNorm(torch.nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def _norm(self, x):
#         # `torch.rsqrt` 是开平方并取倒数
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#     def forward(self, x):
#         output = self._norm(x.float()).type_as(x)
#         # 没有 `bias`
#         return output * self.weight


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(RotaryPositionalEncoding, self).__init__()
        self.d_model = d_model

    def precompute_freqs_cis(self, dim: int, end: int, constant: float = 10000.0):
        """
        计算cos和sin的值，cos值在实部，sin值在虚部，类似于 cosx+j*sinx
        :param dim: q,k,v的最后一维，一般为emb_dim/head_num
        :param end: 句长length
        :param constant： 这里指10000
        :return:
        复数计算 torch.polar(a, t)输出， a*(cos(t)+j*sin(t))
        """
        # freqs: 计算 1/(10000^(2i/d) )，将结果作为参数theta
        # 形式化为 [theta_0, theta_1, ..., theta_(d/2-1)]
        freqs = 1.0 / (constant ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [d/2]

        # 计算m
        t = torch.arange(end, device=freqs.device)  # [length]
        # 计算m*theta
        freqs = torch.outer(t, freqs).float()  # [length, d/2]
        # freqs形式化为 [m*theta_0, m*theta_1, ..., m*theta_(d/2-1)],其中 m=0,1,...,length-1

        # 计算cos(m*theta)+j*sin(m*theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        # freqs_cis: [cos(m*theta_0)+j*sin(m*theta_0),  cos(m*theta_1)+j*sin(m*theta_1),), ..., cos(m*theta_(d/2-1))+j*sin(m*theta_(d/2-1))]
        # 其中j为虚数单位， m=0,1,...,length-1
        return freqs_cis  # [length, d/2]

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]  # (1, length, 1, d/2)
        return freqs_cis.view(*shape)  # [1, length, 1, d/2]

    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        # 先将xq维度变为[bs, length, head,  d/2, 2], 利用torch.view_as_complex转变为复数
        # xq:[q0, q1, .., q(d-1)] 转变为 xq_: [q0+j*q1, q2+j*q3, ..., q(d-2)+j*q(d-1)]
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [bs, length, head, d/2]
        # 同样的，xk_:[k0+j*k1, k2+j*k3, ..., k(d-2)+j*k(d-1)]
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)  # [1, length, 1, d/2]
        # 下式xq_ * freqs_cis形式化输出，以第一个为例, 如下
        # (q0+j*q1)(cos(m*theta_0)+j*sin(m*theta_0)) = q0*cos(m*theta_0)-q1*sin(m*theta_0) + j*(q1*cos(m*theta_0)+q0*sin(m*theta_0))
        # 上式的实部为q0*cos(m*theta_0)-q1*sin(m*theta_0)，虚部为q1*cos(m*theta_0)+q0*sin(m*theta_0)
        # 然后通过torch.view_as_real函数，取出实部和虚部，维度由[bs, length, head, d/2]变为[bs, length, head, d/2, 2]，最后一维放实部与虚部
        # 最后经flatten函数将维度拉平，即[bs, length, head, d]
        # 此时xq_out形式化为 [实部0，虚部0，实部1，虚部1，..., 实部(d/2-1), 虚部(d/2-1)]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # [bs, length, head, d]
        # 即为新生成的q

        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, xq, xk):
        # xq, xk: [bs, length, head, d]
        # xq, xk = xq.float(), xk.float()
        freqs_cis = self.precompute_freqs_cis(dim=self.d_model, end=xq.shape[1])
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)
        return xq, xk


class SinusoidalPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        # >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # [d_model/2, ]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x, batch_first=True):
        if batch_first:
            x = x + self.pe[:, : x.size(1), :]
        else:
            x = x + self.pe[:, : x.size(0), :].transpose(0, 1)
        return self.dropout(x)


class LearnabledPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnabledPositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, embedding)
        pos = torch.arange(0, x.size(1), dtype=torch.long).unsqueeze(0)
        pos = pos.repeat(x.size(0), 1).to(x.device)
        embedding = self.embedding(pos)
        return x + embedding


class DynamicWeightAverage(torch.nn.Module):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.loss_weights = [0.5, 0.5]  # 有几个损失就要有几个权重

    def forward(self, losses, mode: typing.Literal["train", "eval"]):
        losses_inv = [1 / (loss + 1e-8) for loss in losses]  # 加上小常数避免除零
        total_inv = sum(losses_inv)
        new_weights = [inv / total_inv for inv in losses_inv]

        # 应用平滑因子
        weighted_weights = [self.alpha * prev + (1 - self.alpha) * new for prev, new in zip(self.loss_weights, new_weights)]

        # 归一化
        total_weight = sum(weighted_weights)
        normalized_weights = [w / total_weight for w in weighted_weights]

        loss = sum([w * lo for w, lo in zip(normalized_weights, losses)])

        if mode == "train":
            self.loss_weights = [i.detach().clone() for i in normalized_weights]

        return loss


class Adapter(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Adapter, self).__init__()
        self.down_proj_adapter = torch.nn.Linear(input_size, input_size // 2)
        self.non_linearity_adapter = torch.nn.GELU()
        self.up_proj_adapter = torch.nn.Linear(input_size // 2, output_size)
        self.norm_adapter = Normalization(output_size)

    def forward(self, x):
        output = self.down_proj_adapter(x)
        output = self.non_linearity_adapter(output)
        output = self.up_proj_adapter(output)
        output += x
        output = self.norm_adapter(output)
        return output
