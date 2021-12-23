from symbols import idx2symbol


class Hyparams:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


hparams = Hyparams(
    #
    ### Signal Processing (used in both synthesizer and vocoder)
    sample_rate=16000,
    n_fft=800,
    num_mels=80,
    hop_size=200,  # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
    win_size=800,  # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
    fmin=55,
    min_level_db=-100,
    ref_level_db=20,
    max_abs_value=4.,  # Gradient explodes if too big, premature convergence if too small.
    preemphasis=0.97,  # Filter coefficient to use if preemphasize is True
    preemphasize=True,
    use_lws=False,
    fmax=7600,  # Should not exceed (sample_rate // 2)
    allow_clipping_in_normalization=True,  # Used when signal_normalization = True
    clip_mels_length=True,  # If true, discards samples exceeding max_mel_frames
    symmetric_mels=True,  # Sets mel range to [-max_abs_value, max_abs_value] if True,
    #               and [0, max_abs_value] if False
    trim_silence=True,  # Use with sample_rate of 16000 for best results

    signal_normalization=True,
    power=1.5,
    griffin_lim_iters=60,
    #
    n_frames_per_step=1,
    n_mel_channel=80,
    speaker_embedding_dim=256,
    mask_padding=True,
    fp16_run = False,

    # encoder
    num_embedding=len(idx2symbol),
    encoder_embedding_dim=256,
    encoder_kernel_size=3,
    encoder_dropout=0.5,
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
    attention_location_kernel_size=3,

    # postnet
    postnet_embedding_dim=256,
    postnet_kernel_size=5,
    postnet_conv_nums=5,
    p_postnet_dropout=0.5

)
