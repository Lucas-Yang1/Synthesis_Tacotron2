import torch
import torch.nn as nn

from layers import LinearNorm, Conv1dWithBatchNorm, clone, get_mask_from_lengths, Conv1dNorm
from params_model import hparams
from synthesisdataset import SynthesisBatchData


class LocalLayer(nn.Module):
    """
    计算f_ij，即累加注意力的卷积。
    """

    def __init__(self, attention_location_n_filters, attention_location_kernel_size, attention_dim):
        super(LocalLayer, self).__init__()
        self.conv = Conv1dNorm(2, attention_location_n_filters, kernel_size=attention_location_kernel_size,
                               stride=1, padding=int((attention_location_kernel_size - 1) / 2), )
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
    ei,j=score(si,cαi−1,hj)=v^T * tanh(Wsi+Vhj+Uf(i,j)+b)
    """

    def __init__(self, attention_rnn_dim,
                 embedding_dim,
                 attention_dim,
                 attention_location_n_filters,
                 attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim,
                                       bias=False, w_init_gain='tanh')
        self.location_layer = LocalLayer(attention_location_n_filters,
                                         attention_location_kernel_size,
                                         attention_dim)
        self.v = LinearNorm(attention_dim, 1, bias=False, w_init_gain='linear')

        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, memory, attention_weight_cat):
        """
        计算未归一化，未mask的 att_weight
        query: decoder output [batch, n_mel_channel * n_frames_per_step]
        memory: encoder output [batch, time_step, embedding]
        attention_weight_cat : cumulative and prev att_ weights [batch, 2, time_step]
        return:
        energies : [batch, time_steps]
        """
        query_, memory_, attention_weight_cat_ = \
            self.query_layer(query).unsqueeze(1), self.memory_layer(memory), self.location_layer(attention_weight_cat)
        energies = self.v(torch.tanh(query_ + memory_ + attention_weight_cat_))

        return energies.squeeze(-1)

    def forward(self, query, memory, attention_weight_cat, mask):
        """
        query : decoder output
        memory: encoder output
        attention_weight_cat :
        mask : for memory
        """

        energies = self.get_alignment_energies(query, memory, attention_weight_cat)

        if mask is not None:
            energies.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = energies.softmax(dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
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
    def __init__(self, hparams=hparams):
        super(Postnet, self).__init__()
        self.layers = nn.Sequential(
            Conv1dWithBatchNorm(hparams.n_mel_channel, hparams.postnet_embedding_dim, hparams.postnet_kernel_size,
                                stride=1,
                                padding=int((hparams.postnet_kernel_size - 1) / 2)),
            nn.Dropout(hparams.p_postnet_dropout),
            *clone(nn.Sequential(Conv1dWithBatchNorm(hparams.postnet_embedding_dim, hparams.postnet_embedding_dim,
                                                     hparams.postnet_kernel_size,
                                                     stride=1, padding=int((hparams.postnet_kernel_size - 1) / 2)),
                                 nn.Dropout(hparams.p_postnet_dropout)), hparams.postnet_conv_nums - 2),
            Conv1dWithBatchNorm(hparams.postnet_embedding_dim, hparams.n_mel_channel, hparams.postnet_kernel_size,
                                stride=1,
                                padding=int((hparams.postnet_kernel_size - 1) / 2)),
            nn.Dropout(hparams.p_postnet_dropout)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, hparams=hparams):
        super(Decoder, self).__init__()
        self.n_mel_channel = hparams.n_mel_channel
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.speaker_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            self.n_mel_channel * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim],
            dropout=hparams.prenet_dropout
        )

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim,
            self.attention_rnn_dim,
        )

        self.attention_layer = Attention(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size
        )

        self.decoder_rnn = nn.LSTMCell(
            self.encoder_embedding_dim + hparams.attention_dim,
            self.decoder_rnn_dim, bias=True
        )

        self.linear_project = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mel_channel * self.n_frames_per_step,
            w_init_gain='linear'
        )

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            1, w_init_gain='sigmoid'
        )

    def get_go_frame(self, memory):
        B = memory.size(0)
        go_frame = memory.new_zeros((B, self.n_frames_per_step * self.n_mel_channel))

        return go_frame

    def initialize_decoder_stats(self, memory, mask):
        """
        memory: encoder output
        mask : mask for memory
        """

        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = memory.new_zeros((
            B, self.attention_rnn_dim
        ))

        self.attention_cell = memory.new_zeros((
            B, self.attention_rnn_dim
        ))

        self.decoder_hidden = memory.new_zeros((
            B, self.decoder_rnn_dim
        ))

        self.decoder_cell = memory.new_zeros((
            B, self.decoder_rnn_dim
        ))

        self.attention_weight = memory.new_zeros((
            B, MAX_TIME
        ))

        self.attention_weight_cum = memory.new_zeros((
            B, MAX_TIME
        ))

        self.attention_context = memory.new_zeros((
            B, self.encoder_embedding_dim
        ))

        self.mask = mask
        self.memory = memory

    def parse_decoder_inputs(self, decoder_inputs):
        """
        decoder_inputs : [batch, T_out, n_mel_channel]
        """

        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            decoder_inputs.size(1) // self.n_frames_per_step, -1
        )

        # [batch, T_out, n_mel_channel] -> [T_out, batch, n_mel_channel]
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """
        mel_outputs : list of mel_output : [batch, 1, n_mel_channel]
        gate_outputs: list of gate_output: [batch, 1, 1]
        alignments : list of att . weight: [batch, T_IN]
        """
        # [T_out, B] -> [B, T_out]
        alignments = torch.stack(alignments).transpose(0, 1)

        # [T_out, B] -> [B, T_out]
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
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
        """
        one step decode process
        """
        cell_input = torch.cat((decoder_input, self.attention_context), dim=-1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )

        self.attention_hidden = torch.dropout(self.attention_cell,
                                              self.p_attention_dropout, self.training)

        attention_weight_cat = torch.cat((self.attention_weight.unsqueeze(1),
                                          self.attention_weight_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weight = self.attention_layer(
            self.attention_hidden, self.memory, attention_weight_cat, self.mask
        )

        self.attention_weight_cum += self.attention_weight

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = torch.dropout(self.decoder_hidden, p=self.p_decoder_dropout, train=self.training)

        decoder_hidden_attention_context = torch.cat((
            self.decoder_hidden, self.attention_context
        ), 1)
        decoder_output = self.linear_project(decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, self.attention_weight

    def forward(self, memory, decoder_inputs, memory_lengths):

        go_frames = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((go_frames, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_stats(memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []

        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weight = \
                self.decode(decoder_input)

            mel_outputs.append(mel_output.squeeze(1))
            gate_outputs.append(gate_output.squeeze(1))
            alignments.append(attention_weight)

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_stats(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []

        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, attention_weight = \
                self.decode(decoder_input)

            mel_outputs.append(mel_output.squeeze(1))
            gate_outputs.append(gate_output.squeeze(1))
            alignments.append(attention_weight)

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning, inference reached the max decoder steps")
                break

            decoder_input = mel_output
        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)


class Encoder(nn.Module):
    """
    3 1-d conv layer
    bi lstm
    """

    def __init__(self, hparams=hparams):
        super(Encoder, self).__init__()
        self.conv_layer = clone(nn.Sequential(Conv1dNorm(
            hparams.encoder_embedding_dim,
            hparams.encoder_embedding_dim,
            kernel_size=hparams.encoder_kernel_size,
            stride=1, padding=int((hparams.encoder_kernel_size - 1) / 2),
            dilation=1, w_init_gain='relu'
        ), nn.Dropout(hparams.encoder_dropout)), 3)
        self.lstm = nn.LSTM(
            hparams.encoder_embedding_dim,
            int(hparams.encoder_embedding_dim / 2), num_layers=1,
            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, speaker_embedding=None):
        """
        x: batch text embedding : [batch, embedding_size, text_length]
        """
        for layer in self.conv_layer:
            x = layer(x)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs [batch, text_length, embedding_size]
        # speaker_embedding: [batch, speaker_embedding_size]
        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.unsqueeze(1).repeat_interleave(outputs.size(1), dim=1)
            outputs = torch.cat((outputs, speaker_embedding), dim=-1)

        return outputs

    def inference(self, x, speaker_embedding=None):
        for layer in self.conv_layer:
            x = layer(x)
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.unsqueeze(1).repeat_interleave(outputs.size(1), dim=1)
            outputs = torch.cat((outputs, speaker_embedding), dim=-1)

        return outputs


class Tacotron(nn.Module):
    def __init__(self, hparams=hparams):
        super(Tacotron, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channel = hparams.n_mel_channel
        self.n_frames_per_step = hparams.n_frames_per_step

        self.embedding = nn.Embedding(hparams.num_embedding, hparams.encoder_embedding_dim)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

        nn.init.xavier_normal_(self.embedding.weight)

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channel, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0] = outputs[0].masked_fill(mask, 0.0)
            outputs[1] = outputs[1].masked_fill(mask, 0.0)
            outputs[2] = outputs[2].masked_fill(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs: SynthesisBatchData):
        text_inputs, text_lengths, speaker_embedding, mels, output_lenghts = \
            inputs.text_inputs, inputs.text_lengths, inputs.speaker_embedding, inputs.mels, inputs.output_lengths

        text_lengths, output_lenghts = text_lengths.data, output_lenghts.data

        embedding_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedding_inputs, text_lengths, speaker_embedding)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, text_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs_postnet + mel_outputs

        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
                                 output_lengths=output_lenghts)

    def inference(self, text_inputs, speaker_embedding):
        embedding_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_output = self.encoder.inference(embedding_inputs, speaker_embedding)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_output)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs_postnet + mel_outputs

        output = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return output

    def get_loss(self, model_output, targets, output_lengths):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad_(False)
        gate_target.requires_grad_(False)
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)

        mel_loss = (nn.MSELoss(reduction='sum')(mel_out, mel_target) +
                    nn.MSELoss(reduction='sum')(mel_out_postnet, mel_target)) / (
                               output_lengths.sum() * self.n_mel_channel)
        gate_loss = nn.BCEWithLogitsLoss(reduction='sum')(gate_out, gate_target) / output_lengths.sum()
        return (mel_loss + gate_loss)


if __name__ == '__main__':
    B = 16
    memory_length = torch.randint(10, 100, (B,))
    memory = torch.randn((B, max(memory_length), hparams.encoder_embedding_dim + hparams.speaker_embedding_dim))

    decoder_inputs = torch.randn((B, 100, hparams.n_mel_channel))
    decoder = Decoder(hparams)
    output = decoder(memory, decoder_inputs, memory_length)
    infer = decoder.inference(memory[:1])

    encoder = Encoder(hparams)
    text_ = torch.randn((B, hparams.encoder_embedding_dim, max(memory_length)))
   # memory = encoder(text_, memory_length, torch.randn(B, hparams.speaker_embedding_dim))
   # infer_ = decoder.inference(memory[:1])

    # memory_length = torch.randint(10, 50, (B,))
    #
    # text_input = torch.randint(0, 44, (B, max(memory_length)))
    # speaker_embedding = torch.randn(B, hparams.speaker_embedding_dim)
    # mel_lengths = torch.randint(10, 100, (B,))
    # mel = torch.randn((B, max(mel_lengths), hparams.n_mel_channel))
    #
    # model = Tacotron(hparams)
    # inputs = text_input, memory_length, speaker_embedding, mel, max(mel_lengths), mel_lengths
    #
    # m_output = model(inputs)
