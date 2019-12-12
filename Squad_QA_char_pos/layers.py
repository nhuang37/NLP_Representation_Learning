"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import util
from util import masked_softmax

import math
import copy

class Embedding(nn.Module):
    """Embedding layer used by QAnet, with the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, pos_vectors, hidden_size, drop_prob, char_emb_dim):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed_words = nn.Embedding.from_pretrained(word_vectors)
        self.embed_chars = nn.Embedding.from_pretrained(char_vectors)
        self.conv2d = nn.Conv2d(char_vectors.size(1), char_emb_dim, kernel_size = (1,5), padding=0, bias=True)
        self.proj = nn.Linear(word_vectors.size(1)+char_emb_dim+10, hidden_size, bias=False) #add POS dim
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word, char,pos):
        word_emb = self.embed_words(word)   # (batch_size, seq_len, embed_size)
        char_emb = self.embed_chars(char)
        char_emb = char_emb.permute(0,3,1,2)
        char_emb = self.conv2d(char_emb)
        char_emb = F.relu(char_emb)
        char_emb, _ = torch.max(char_emb, dim=3)
        char_emb = char_emb.transpose(2,1)
        emb = torch.cat([char_emb, word_emb, pos], dim=2) #add pos embeddings
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

#Added modules for QAnet:
#Source: https://github.com/BangLiu/QANet-PyTorch/blob/master/model/QANet.py
class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    #x: shape [batch_size, length, hidden_size] (from highway output)
    #print(x.shape) #double check whether transpose make sense
    #x = x.transpose(1, 2) #first dimension is batch
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    if torch.cuda.is_available():
        signal = signal.to(x.get_device()) #ensure they are in the same gpu

    return (x + signal).transpose(1, 2) # such that return shape [batch_size, hidden_size, length]


def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))

#source code: https://github.com/harvardnlp/annotated-transformer
def attention(query, key, value, mask=None, dropout=None): #similar to what 'fscore' thing ???
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SelfAttention(nn.Module):
    def __init__(self, d_model=128, h=8,drop_prob=0):
        super(SelfAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=drop_prob)
        self.mem_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)


    def forward(self, queries, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1) #add 1 more fake dim: 64 x 1 x 1 x 256
            #print(mask.shape)
        nbatches = queries.size(0)

        # 0) specific to machine reading, create key and value based on query & convolutions
        memory = queries
        memory = self.mem_conv(memory)  # (batch_size, d_model, d_model*2)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        key, value = torch.split(memory, self.d_model, dim=2) # K and V each equal to size (batch_size, d_model, d_model)
        #print(query.shape, key.shape, value.shape)

        # 1) Do all the linear projections in batch from d_model => h x d_k  ???like flatten all 8 heads, fit W and train them all together
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear to map it back to d_model=128
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int,drop_prob=0):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model=ch_num)
        self.FFN_1 = Initialized_Conv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(ch_num) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(ch_num)
        self.norm_2 = nn.LayerNorm(ch_num)
        self.conv_num = conv_num
        self.drop_prob = drop_prob

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num+1)*blks
        out = PosEncoder(x)
        drop_prob = self.drop_prob
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=drop_prob, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, drop_prob*float(l)/total_layers)
            l += 1
        res = out #add to enforce shape batch  x hidden x len
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=drop_prob, training=self.training)
        out = self.self_att(out, mask) # batch x len x hidden
        out = self.layer_dropout(out.transpose(1,2), res, drop_prob*float(l)/total_layers)
        l += 1
        res = out
        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=drop_prob, training=self.training)
        out = self.FFN_1(out) #a
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, drop_prob*float(l)/total_layers)
        return out.transpose(1,2) # batch x len x hidden

    def layer_dropout(self, inputs, residual,drop_prob):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < drop_prob
            if pred:
                return residual
            else:
                return F.dropout(inputs, drop_prob, training=self.training) + residual
        else:
            return inputs + residual

# for the model encoder layer
class CQAttention(nn.Module):
    def __init__(self):
        super().__init__()
        w4C = torch.empty(D, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, D)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Output_Pointer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = Initialized_Conv1d(hidden_size*2, 1)
        self.w2 = Initialized_Conv1d(hidden_size*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = masked_softmax(self.w1(X1).squeeze(), mask, log_softmax=True)
        Y2 = masked_softmax(self.w2(X2).squeeze(), mask, log_softmax=True)
        return Y1, Y2


# class QANet(nn.Module):
#     def __init__(self, word_mat, char_mat):
#         super().__init__()
#         if config.pretrained_char:
#             self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat))
#         else:
#             char_mat = torch.Tensor(char_mat)
#             self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=False)
#         self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat), freeze=True)
#         self.emb = Embedding()
#         self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
#         self.cq_att = CQAttention()
#         self.cq_resizer = Initialized_Conv1d(D * 4, D)
#         self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7)])
#         self.out = Pointer()

#     def forward(self, Cwid, Ccid, Qwid, Qcid):
#         maskC = (torch.zeros_like(Cwid) != Cwid).float()
#         maskQ = (torch.zeros_like(Qwid) != Qwid).float()
#         Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
#         Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
#         C, Q = self.emb(Cc, Cw, Lc), self.emb(Qc, Qw, Lq)
#         Ce = self.emb_enc(C, maskC, 1, 1)
#         Qe = self.emb_enc(Q, maskQ, 1, 1)
#         X = self.cq_att(Ce, Qe, maskC, maskQ)
#         M0 = self.cq_resizer(X)
#         M0 = F.dropout(M0, p=dropout, training=self.training)
#         for i, blk in enumerate(self.model_enc_blks):
#              M0 = blk(M0, maskC, i*(2+2)+1, 7)
#         M1 = M0
#         for i, blk in enumerate(self.model_enc_blks):
#              M0 = blk(M0, maskC, i*(2+2)+1, 7)
#         M2 = M0
#         M0 = F.dropout(M0, p=dropout, training=self.training)
#         for i, blk in enumerate(self.model_enc_blks):
#              M0 = blk(M0, maskC, i*(2+2)+1, 7)
#         M3 = M0
#         p1, p2 = self.out(M1, M2, M3, maskC)
#         return p1, p2
