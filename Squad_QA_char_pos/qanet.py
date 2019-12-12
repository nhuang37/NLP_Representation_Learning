"""Top-level model classes.

Author:
    Teresa Ningyuan Huang
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QAnet(nn.Module):
    """QAnet implementation for SQuAD.

    Based on the paper:
    QANET: COMBINING LOCAL CONVOLUTION WITH GLOBAL SELF-ATTENTION FOR READING COMPRE- HENSION

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors. (word + character + add features => highway => output)
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors,pos_vectors,hidden_size,drop_prob=0.0,char_emb_dim = 200): #use drop_out = 0.1, plus adding character_level embedding (200 from QA net paper)
        super(QAnet, self).__init__()

        self.tag_emb = nn.Embedding(num_embeddings=52,embedding_dim=10,padding_idx=0) # add POS tag

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    pos_vectors=pos_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    char_emb_dim = char_emb_dim)

        self.enc = layers.EncoderBlock(conv_num=4, ch_num=hidden_size, k=7,
                                     drop_prob=drop_prob)   # QAnet Encoder Block: convolution x #4 + self_attention + feed_forward

        self.att = layers.BiDAFAttention(hidden_size=hidden_size, #just original hidden size, since use self-attention instead of bi-direction LASTM (not 2 times)
                                         drop_prob=drop_prob)
        self.att_resizer = layers.Initialized_Conv1d(hidden_size * 4, hidden_size) #map back to original hiddensize, can consider a linear layer

        self.model_enc_blks = nn.ModuleList([layers.EncoderBlock(conv_num=2, ch_num=hidden_size, k=5) for _ in range(7)]) #QAnet model layer using Encoder Block again

        self.out = layers.Output_Pointer(hidden_size=hidden_size) #QAnet collect output

        self.drop_prob = drop_prob

    def forward(self, cw_idxs, cc_idxs, ct, qw_idxs, qc_idxs, qt): #add POS
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        ct_emb = self.tag_emb(ct) # (batch_size, c_len, 10), POS
        qt_emb = self.tag_emb(qt) # (batch_size, q_len, 10), POS
        c_emb = self.emb(cw_idxs, cc_idxs, ct_emb)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs, qt_emb)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_mask,1,1)    # (batch_size, c_len, hidden_size)
        q_enc = self.enc(q_emb, q_mask,1,1)    # (batch_size, q_len, hidden_size)

        #print(c_enc.shape, q_enc.shape)
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        M0 = self.att_resizer(att.transpose(1, 2))          # (batch_size, hidden_size, c_len)
        # Model Encoder layers
        M0 = F.dropout(M0, p=self.drop_prob, training=self.training).transpose(1,2) # (batch_size,c_len, hidden_size)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, c_mask, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, c_mask, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, c_mask, i*(2+2)+1, 7)
        M3 = M0
        log_p1, log_p2 = self.out(M1.transpose(1,2), M2.transpose(1,2), M3.transpose(1,2), c_mask)
        return log_p1, log_p2
