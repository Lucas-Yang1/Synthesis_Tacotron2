import copy

import torch.nn as nn
import torch
class Conv1dWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding=0, dilation=1, w_init_gain='tanh'):
        super(Conv1dWithBatchNorm, self).__init__()
        if padding == None:
            padding = int(kernel_size // 2)
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activate = getattr(torch, w_init_gain) if w_init_gain != 'linear' else lambda x:  x

        nn.init.xavier_normal_(self.conv1d.weight, gain=nn.init.calculate_gain(w_init_gain))
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        return self.bn(self.activate(self.conv1d(x)))

class Conv1dNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding=0, dilation=1, bias=False,w_init_gain='linear'):
        super(Conv1dNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding=padding, dilation=dilation, bias=bias)

        nn.init.xavier_normal_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))
    def forward(self, x):
        return self.conv(x)

class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=False, w_init_gain='tanh'):
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

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()

    ids = torch.arange(0,max_len,device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

