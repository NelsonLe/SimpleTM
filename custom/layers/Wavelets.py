import torch
import torch.nn as nn
import torch.nn.functional as F
from pywt import Wavelet
from abc import ABC, abstractmethod


class WaveletLayer(nn.Module, ABC):
    def __init__(self, channels, m=2, learnable_wavelets=True, wv="db1", pad_mode="circular"):
        super().__init__()

        self.channels = channels
        self.m = m
        self.learnable_wavelets = learnable_wavelets
        self.wv = wv
        self.pad_mode = pad_mode

        # create wavelet for parameter initialization
        self.wavelet = Wavelet(self.wv)
    
    def initialize_weights(self, detail_init, approx_init):
        detail_weights = torch.tensor(detail_init, dtype=torch.float32)
        approx_weights = torch.tensor(approx_init, dtype=torch.float32)

        self.kernel_size = detail_weights.shape[0]

        # using repeat and clone bc its safer when saving and reloading checkpoitns
        detail_weights = detail_weights.view(1, 1, self.kernel_size).repeat(self.channels, 1, 1).clone()
        approx_weights = approx_weights.view(1, 1, self.kernel_size).repeat(self.channels, 1, 1).clone()

        self.detail_weights = nn.Parameter(
            detail_weights,
            requires_grad=self.learnable_wavelets
        )
        self.approx_weights = nn.Parameter(
            approx_weights,
            requires_grad=self.learnable_wavelets
        )

    def _pad(self, x, pad):
        if self.pad_mode == "interpolate":
            return self._interpolate_pad(x, pad)
        return F.pad(x, pad, mode=self.pad_mode)

    def _interpolate_pad(self, x, pad):
        left_pad, right_pad = pad
        parts = []

        if left_pad > 0:
            if x.shape[-1] < 2:
                left = x[..., :1].expand(*x.shape[:-1], left_pad)
            else:
                left_slope = x[..., 1:2] - x[..., 0:1]
                steps = torch.arange(
                    left_pad, 0, -1,
                    device=x.device,
                    dtype=x.dtype
                ).view(*([1] * (x.dim() - 1)), left_pad)
                left = x[..., :1] - steps * left_slope
            parts.append(left)

        parts.append(x)

        if right_pad > 0:
            if x.shape[-1] < 2:
                right = x[..., -1:].expand(*x.shape[:-1], right_pad)
            else:
                right_slope = x[..., -1:] - x[..., -2:-1]
                steps = torch.arange(
                    1, right_pad + 1,
                    device=x.device,
                    dtype=x.dtype
                ).view(*([1] * (x.dim() - 1)), right_pad)
                right = x[..., -1:] + steps * right_slope
            parts.append(right)

        return torch.cat(parts, dim=-1)
    
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward")

class WaveletDecomposition(WaveletLayer):
    def __init__(self, channels, m=2, learnable_wavelets=True, wv="db1", pad_mode="circular"):
        super().__init__(channels, m, learnable_wavelets, wv, pad_mode)
        super().initialize_weights(self.wavelet.dec_hi[::-1], self.wavelet.dec_lo[::-1])
    
    def forward(self, x):

        # get shape of input
        B, C, L_prime = x.shape

        # ensure there are as many channels in the input as expected
        if C != self.channels:
            raise ValueError(f"Mismatch in channels: got {C} but expected {self.channels}.")

        # initialize tensor of coefficients to be modified
        coeffs = torch.zeros((B, C, self.m+1, L_prime), dtype=torch.float32)

        # use x as the first approximate coefficients
        approx = x

        # iterate through each decomposition level
        for i in range(self.m):

            # dilation scales exponentially with level
            dilation = 2 ** i

            # pad the current approximate coefficients to preserve length
            padding = dilation * (self.kernel_size - 1)
            padding_r = (self.kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_padded = self._pad(approx, pad)

            # calculate detail and approx coefficients from wavelet convolution
            detail = F.conv1d(approx_padded, self.detail_weights, dilation=dilation, groups=C)
            approx = F.conv1d(approx_padded, self.approx_weights, dilation=dilation, groups=C)

            # save detailed coefficients
            coeffs[:,:,-i-1,:] = detail
        
        # finally, save approximate coefficients
        coeffs[:,:,0,:] = approx
        return coeffs

    
class WaveletReconstruction(WaveletLayer):
    def __init__(self, channels, m=2, learnable_wavelets=True, wv="db1", pad_mode="circular"):
        super().__init__(channels, m, learnable_wavelets, wv, pad_mode)
        super().initialize_weights(self.wavelet.rec_hi[::-1], self.wavelet.rec_lo[::-1])
    
    def forward(self, x):

        B, C, M_plus_one, L_prime = x.shape
        if C != self.channels:
            raise ValueError(f"Mismatch in channels: got {C} but expected {self.channels}.")
        if M_plus_one != self.m + 1:
            raise ValueError(f"Mismatch in decomposition levels: got {M_plus_one} but expected {self.m + 1}.")

        approx = x[:,:,0,:]

        # iterate through decomposition level
        for i in range(self.m):

            # get detail coefficients for this level
            detail = x[:,:,i+1,:]

            # dilation scales exponentially with level
            # but this time, in reverse!
            dilation = 2 ** (self.m - (i + 1))

            # same story with padding
            padding = dilation * (self.kernel_size - 1)
            padding_l = (self.kernel_size * dilation) // 2
            pad = (padding_l, padding - padding_l)

            # pad both the approx and detail coefficients
            approx_padded = self._pad(approx, pad)
            detail_padded = self._pad(detail, pad)

            # reconstruct signal from approx and detail coefficients
            approx_reconstruct = F.conv1d(approx_padded, self.approx_weights, dilation=dilation, groups=C)
            detail_reconstruct = F.conv1d(detail_padded, self.detail_weights, dilation=dilation, groups=C)
            approx = (approx_reconstruct + detail_reconstruct) / 2
        
        return approx
