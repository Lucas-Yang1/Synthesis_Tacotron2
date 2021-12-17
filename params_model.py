from symbols import idx2symbol


class Hyparams:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


hparams = Hyparams(
    #
    n_frames_per_step=1,
    n_mel_channel=80,

    # encoder
    n_char=len(idx2symbol),
    encoder_embedding_dim=256,
    # decoder
    prenet_dim=256,
    prenet_dropout=0.5,
    decoder_rnn_dim=256,
    max_decoder_steps=1000,
    gate_threshold=0.5,
    p_attention_dropout=0.1,
    p_decoder_dropout=0.1,

    # attention
    attention_rnn_dim=512,
    attention_dim=512,
    attention_location_n_filters=32,
    attention_location_kernel_size=31,

)
