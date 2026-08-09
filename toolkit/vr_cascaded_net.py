"""VR 5.1 CascadedNet, vendored/adapted from python-audio-separator (MIT)
uvr_lib_v5/vr_network/{nets_new,layers_new}.py.

Single functional-neutral change for clean ONNX export:
AdaptiveAvgPool2d((1, None)) -> mean(dim=2, keepdim=True) (numerically identical:
adaptive average pooling to H=1 is exactly the mean over H). state_dict keys are
preserved (the pooling layer holds no parameters; the Sequential index it occupied
is kept by the MeanFreq placeholder).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def crop_center(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    if h1.size(3) == h2.size(3):
        return h1
    s = (h1.size(3) - h2.size(3)) // 2
    return h1[:, :, :, s : s + h2.size(3)]


class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Decoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        h = self.conv1(x)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class MeanFreq(nn.Module):
    """Export-safe replacement for AdaptiveAvgPool2d((1, None))."""

    def forward(self, x):
        return x.mean(dim=2, keepdim=True)


class ASPPModule(nn.Module):
    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super().__init__()
        # Sequential index 1 preserved so state_dict keys match upstream
        self.conv1 = nn.Sequential(MeanFreq(), Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ))
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(nin, nout, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = Conv2DBNActiv(nin, nout, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = Conv2DBNActiv(nin, nout, 3, 1, dilations[2], dilations[2], activ=activ)
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode="bilinear", align_corners=True)
        out = torch.cat((feat1, self.conv2(x), self.conv3(x), self.conv4(x), self.conv5(x)), dim=1)
        out = self.bottleneck(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LSTMModule(nn.Module):
    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super().__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True)
        self.dense = nn.Sequential(nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU())

    def forward(self, x):
        N, _, nbins, nframes = x.size()
        h = self.conv(x)[:, 0]
        h = h.permute(2, 0, 1)
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, h.size(-1)))
        h = h.reshape(nframes, N, 1, nbins)
        return h.permute(1, 2, 3, 0)


class BaseNet(nn.Module):
    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super().__init__()
        self.enc1 = Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = Encoder(nout * 6, nout * 8, 3, 2, 1)
        self.aspp = ASPPModule(nout * 8, nout * 8, dilations, dropout=True)
        self.dec4 = Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        h = self.aspp(e5)
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        return self.dec1(h, e1)


class CascadedNet(nn.Module):
    def __init__(self, n_fft, nn_arch_size=51000, nout=32, nout_lstm=128):
        super().__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64
        nout = 64 if nn_arch_size == 218409 else nout
        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm), Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0)
        )
        self.stg1_high_band_net = BaseNet(2, nout // 4, self.nin_lstm // 2, nout_lstm // 2)
        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm), Conv2DBNActiv(nout, nout // 2, 1, 1, 0)
        )
        self.stg2_high_band_net = BaseNet(nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2)
        self.stg3_full_band_net = BaseNet(3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm)
        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def forward(self, x):
        x = x[:, :, : self.max_bin]
        bandw = x.size(2) // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)
        l2 = self.stg2_low_band_net(torch.cat([l1_in, l1], dim=1))
        h2 = self.stg2_high_band_net(torch.cat([h1_in, h1], dim=1))
        aux2 = torch.cat([l2, h2], dim=2)
        f3 = self.stg3_full_band_net(torch.cat([x, aux1, aux2], dim=1))
        mask = torch.sigmoid(self.out(f3))
        return F.pad(mask, (0, 0, 0, self.output_bin - mask.size(2)), mode="replicate")
