import torch
import torch.nn as nn

from layers import LinearNorm, Conv1dWithBatchNorm, clone, get_mask_from_lengths
from params_model import hparams

class LocalLayer(nn.Module):
    """
    计算f_ij，即累加注意力的卷积。
    """

    def __init__(self, attention_location_n_filters, attention_location_kernel_size, attention_dim):
        super(LocalLayer, self).__init__()
        self.conv = Conv1dWithBatchNorm(2, attention_location_n_filters, kernel_size=attention_location_kernel_size, stride=1)
        self.linear = LinearNorm(attention_location_n_filters, attention_dim, bias=False, w_init_gain='tanh')

    def forward(self, attention_weight_cat):
        """
        :param attention_weight_cat: [batch, 2, max_time]
        :return: f_ij: [batch, max_time, attention_dim]
        """
        out = self.conv(attention_weight_cat)
        out = out.transpose(1, 2)
        out = self.linear(out)
        return out


class Attention(nn.Module):
    """
    用于计算attention_context
    """

    def __init__(self, attention_rnn_dim, embedding_dim,
                 attention_dim, attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.key_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.location_layer = LocalLayer(attention_location_n_filters,
                                         attention_location_kernel_size,
                                         attention_dim)
        self.v = LinearNorm(attention_dim, 1, bias=False, w_init_gain='linear')

        # mask fill value
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, key, attention_weight_cat) -> torch.Tensor:
        """
        用于计算非归一化的非masked attention_weight
        :param query: decoder output last frame [batch, n_mel_channel * n_frames_per_step]
        :param key: encoder output [batch, time_step, embedding]
        :param attention_weight_cat: cumulative and prev. att weights [B, 2, max_time]
        :return: unnormed unmasked_filled attention_weight
        """
        query_, key_ = self.query_layer(query.unsqueeze(1)), self.key_layer(key)
        attention_weight_cat_ = self.location_layer(attention_weight_cat)
        print(query_.shape, key_.shape, attention_weight_cat_.shape)
        energies = self.v(torch.tanh(query_ + key_ + attention_weight_cat_))

        return energies.squeeze(-1)

    def forward(self, query, key, attention_weight_cat, mask):
        energies = self.get_alignment_energies(query, key, attention_weight_cat)

        if mask is not None:
            print(mask.shape)
            print(energies.shape)
            energies.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = energies.softmax(dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), key)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, dropout=0.5):
        super(Prenet, self).__init__()
        in_sizes = [in_dim, ] + sizes[:-1]

        self.layers = nn.ModuleList([LinearNorm(in_size, out_size, bias=False)
                                     for (in_size, out_size) in zip(in_sizes, sizes)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for l in self.layers:
            x = self.dropout(torch.relu(l(x)))

        return x


class Postnet(nn.Module):
    def __init__(self, n_mel_channel, postnet_embedding_dim,
                 postnet_kernel_size, postnet_conv_nums, dropout=0.5):
        super(Postnet, self).__init__()
        self.layers = nn.Sequential(
            Conv1dWithBatchNorm(n_mel_channel, postnet_embedding_dim, postnet_kernel_size), nn.Dropout(dropout),
            *clone(nn.Sequential(Conv1dWithBatchNorm(postnet_embedding_dim, postnet_embedding_dim, postnet_kernel_size),
                                 nn.Dropout(dropout)), postnet_conv_nums - 2),
            Conv1dWithBatchNorm(postnet_embedding_dim, n_mel_channel, postnet_kernel_size), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, hparams=hparams):
        super(Decoder, self).__init__()
        self.n_mel_channel = hparams.n_mel_channel
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(hparams.n_mel_channel * hparams.n_frames_per_step,
                             [hparams.prenet_dim, hparams.prenet_dim],
                             hparams.prenet_dropout)

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim
        )

        self.attention_layer = Attention(
            hparams.attention_rnn_dim,
            hparams.encoder_embedding_dim,
            hparams.attention_dim,
            hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size
        )

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, bias=True
        )

        self.linear_project = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channel * hparams.n_frames_per_step,
            w_init_gain='linear'
        )

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            1, w_init_gain='sigmoid'
        )

    def get_go_frame(self, memory: torch.Tensor):
        """get all zeros frames to use as first decoder frame
        memory: encoder outputs
        """
        batch_size = memory.size(0)

        go_frame = memory.new_zeros((batch_size, self.n_mel_channel*self.n_frames_per_step))

        return go_frame

    def initialize_decoder_stats(self, memory, mask):
        """
        :param memory: encoder outputs
        :return:
        """
        batch_size = memory.size(0)
        MAX_TIME = memory.size(1)
        self.attention_hidden = memory.new_zeros((
            batch_size, self.attention_rnn_dim
        ))
        self.attention_cell= memory.new_zeros((
            batch_size, self.attention_rnn_dim
        ))

        self.decoder_hidden= memory.new_zeros((
            batch_size, self.decoder_rnn_dim
        ))
        self.decoder_cell= memory.new_zeros((
            batch_size, self.decoder_rnn_dim
        ))

        self.attention_weight = memory.new_zeros((
            batch_size, MAX_TIME
        ))
        self.attention_weight_cum = memory.new_zeros((
            batch_size, MAX_TIME
        ))
        self.attention_context = memory.new_zeros((
            batch_size, self.encoder_embedding_dim
        ))

        self.memory = memory
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        :param decoder_inputs: [batch, T_out, n_mel_channel]
        :return:
        """

        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1
        )

        # [batch, T_out, n_mel_channel] -> [T_out, batch, n_mel_channel]
        decoder_inputs = decoder_inputs.transpose(0, 1)

        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):

        """
        :param mel_outputs:
        :param gate_outputs: gate_output
        :param alignments: sequence of attention weights from the decoder
        :return:
        """

        # [T_out, B] -> [B, T_out]
        alignments = torch.stack(alignments).transpose(0, 1)

        # [T_out, B] -> [B, T_out]
        gate_outputs = torch.stack(gate_outputs).transpose(0,1)
        gate_outputs = gate_outputs.contiguous()

        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channel)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):

        cell_input = torch.cat((decoder_input, self.attention_context), dim=-1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )

        self.attention_hidden = torch.dropout(self.attention_hidden,
                                              self.p_attention_dropout, self.training)

        attention_weight_cat = torch.cat([self.attention_weight.unsqueeze(1),
                                          self.attention_weight_cum.unsqueeze(1)], dim=1)
        self.attention_context, self.attention_weight = self.attention_layer(
            self.attention_hidden, self.memory, attention_weight_cat, self.mask
        )
        self.attention_weight_cum += self.attention_weight

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input,
                                                                  (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = torch.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), 1
        )

        decoder_output = self.linear_project(decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, self.attention_weight


    def forward(self, memory, decoder_inputs, memory_lengths):


        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)


        self.initialize_decoder_stats(
            memory, mask=~get_mask_from_lengths(memory_lengths)
        )


        mel_outputs, gate_outputs, alignments = [], [], []

        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weight =\
            self.decode(decoder_input)

            mel_outputs.append(mel_output.squeeze(1))
            gate_outputs.append(gate_output.squeeze(1))
            alignments.append(attention_weight)


        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments