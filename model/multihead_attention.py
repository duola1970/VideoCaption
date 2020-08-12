import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(Q, K, V, mask, dropout=None):
    # Q, K, V are (B, *(H), seq_len, d_model//H = d_k)
    # mask is     (B,    1,       1,               Ss)
    d_k = Q.size(-1)
    # (B, H, S, S)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)

    # mask掉那些pad部分，使得注意力机制注意不到
    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))

    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)

    if dropout is not None:
        out = dropout(out)

    # (B, *(H), seq_len, d_model//H = d_k)
    return out


class MultiheadedAttention(nn.Module):

    def __init__(self, d_model_Q, d_model_K, d_model_V, H, dout_p=0.0, d_model=None):
        super(MultiheadedAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p

        if self.d_model is None:
            print(f'd_model: is None')
            self.d_model = self.d_model_Q

        self.d_k = self.d_model // H          # 256

        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)   # 在进行self_attention之前，先将特征映射均映射为1024
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        self.linear_d2Q = nn.Linear(self.d_model, self.d_model_Q)

        self.dropout = nn.Dropout(self.dout_p)

        assert self.d_model % H == 0

    def forward(self, Q, K, V, mask):
        ''' 
            Q, K, V: (B, Sq, Dq), (B, Sk, Dk), (B, Sv, Dv)
            mask: (B, 1, Sk)
            Sk = Sv,         #K和V是一个键值对取自同一个特征流
            Dk != self.d_k
            Also: m1 is the target modality (queries); m2 is the source modality (keys, values)
        '''

        B, Sq, d_model_Q = Q.shape    # (32,Sa,d)
        # (B, Sm, D) <- (B, Sm, Dm)
        Q = self.linear_Q2d(Q)        # (32,Sa,d_model)
        K = self.linear_K2d(K)        # (32,Sv,d_model)
        V = self.linear_V2d(V)        # (32,Sv,d_model)

        # (B, H, Sm, d_k) <- (B, Sm, D)
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)  # (-4, -3*, -2*, -1)  (32,Sa,4,256)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)  # (32,Sv,4,256)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)  # (32,Sv,4,256)

        if mask is not None:
            # the same mask for all heads -> (B, 1, 1, Sm2)
            mask = mask.unsqueeze(1)

        # (B, H, Sq, d_k) <- (B, H, Sq, d_k), (B, H, Sk, d_k), (B, H, Sv, d_k), Sk = Sv
        Q = attention(Q, K, V, mask, self.dropout)
        # (B, Sq, D) <- (B, H, Sq, d_k)
        Q = Q.transpose(-3, -2).contiguous().view(B, Sq, self.d_model)
        # (B, Sq, Dq),输出的维度永远与Q查询向量的维度一致，故Q是什么流的查询向量对应输出的主特征就是哪一流
        Q = self.linear_d2Q(Q)

        return Q
