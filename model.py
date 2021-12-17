import copy

import torch
import torch.nn as nn
import numpy as np
from params_model import *

class Conv1dWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, w_init_gain='tanh'):
        super(Conv1dWithBatchNorm, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=1, padding=kernel_size//2, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self._init_weight(w_init_gain)

    def forward(self, x):
        return self.bn(self.relu(self.conv1d(x)))

    def _init_weight(self, w_init_gain):
        nn.init.xavier_normal_(self.conv1d.weight, gain=nn.init.calculate_gain(w_init_gain))
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)

        nn.init.xavier_normal_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

        if bias:
            nn.init.zeros_(self.linear_layer.bias)

    def forward(self, x):
        return self.linear_layer(x)


def clone(m, N):
    return nn.ModuleList([copy.deepcopy(m) for _ in range(N)])

class Tacotron2Encoder(nn.Module):
    def __init__(self, n_char, char_embed_size, hidden_feature_size, speaker_embedding_size,
                 bidirectional, num_layers, dropout):
        super(Tacotron2Encoder, self).__init__()
        self.embedding = nn.Embedding(n_char, embedding_dim=char_embed_size)
        self.conv3layers = clone(Conv1dWithBatchNorm(char_embed_size, char_embed_size, kernel_size=3), 3)
        self.lstm = nn.LSTM(char_embed_size, hidden_feature_size,
                            bidirectional=bidirectional, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.speaker_embedding_linear = nn.Linear(speaker_embedding_size, hidden_feature_size)
    def forward(self, x, speaker_embedding: torch.Tensor =None):
        out = self.embedding(x)
        for l in self.conv3layers:
            out = l(out)
        out, _ = self.lstm(out)

        if not speaker_embedding:
            return out

        batch, seq_len = out.size(0), out.size(1)

        speaker_embedding = self.speaker_embedding_linear(speaker_embedding)
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.repeat(batch, seq_len, 1)
        elif speaker_embedding.dim() == 2:
            speaker_embedding = speaker_embedding.view(batch, 1, -1).repeat(1, seq_len, 1)


        return x + speaker_embedding

class LocalLayer(nn.Module):
    """
    计算f_ij，即累加注意力的卷积。
    """
    def __init__(self, attention_location_n_filters, attention_kernel_size, attention_dim):
        super(LocalLayer, self).__init__()
        self.conv = Conv1dWithBatchNorm(2, attention_location_n_filters, kernel_size=attention_kernel_size)
        self.linear = LinearNorm(attention_location_n_filters, attention_dim, bias=False, w_init_gain='tanh')

    def foward(self, attention_weight_cat):
        """
        :param attention_weight_cat: [batch, 2, max_time]
        :return: f_ij: [batch, max_time, attention_dim]
        """
        out = self.conv(attention_weight_cat)
        out = out.transpose(1,2)
        out =self.linear(out)
        return out


class Attention(nn.Module):
    def __init__(self, embedding_dim, attention_rnn_dim, attention_dim,
                 attention_location_n_filters, attention_kernel_size, ):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.key_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.location_layer = LocalLayer(attention_location_n_filters, attention_kernel_size, attention_dim)
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, key, attention_weights_cat):
        query_, key_ = self.query_layer(query), self.key_layer(key)
        attention_weights_cat_ = self.location_layer(attention_weights_cat)

        energies = self.v(torch.tanh(query_ + key_ + attention_weights_cat_))
        return energies.squeeze(-1)

    def forward(self, query, key, attention_weights_cat, mask):

        energies = self.get_alignment_energies(query, key, attention_weights_cat)

        if mask is not None:
            energies.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = energies.softmax(dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), key)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, dropout=0.5):
        super(Prenet, self).__init__()
        in_sizes = [in_dim,] + sizes[:-1]

        self.layers = nn.ModuleList([LinearNorm(in_size,out_size, bias=False)
                                     for (in_size, out_size) in zip(in_sizes, sizes)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for l in self.layers:
            x = self.dropout(torch.relu(l(x)))

        return x

class Postnet(nn.Module):
    def __init__(self, n_mel_channel, postnet_embedding_dim,
                 kernel_size, conv_nums, dropout=0.5):
        super(Postnet, self).__init__()
        self.layers = nn.Sequential(
            Conv1dWithBatchNorm(n_mel_channel, postnet_embedding_dim, kernel_size), nn.Dropout(dropout),
            *clone(nn.Sequential(Conv1dWithBatchNorm(postnet_embedding_dim, postnet_embedding_dim, kernel_size),
                                nn.Dropout(dropout)), conv_nums-2),
            Conv1dWithBatchNorm(postnet_embedding_dim, n_mel_channel, kernel_size), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)

class Tacotron2Decoder(nn.Module):
    def __init__(self, hparams):


        super(Tacotron2Decoder, self).__init__()
        self.prenet = Prenet()
class Tacotron2(nn.Module):
    def __init__(self):
        pass