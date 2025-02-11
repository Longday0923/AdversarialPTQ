"""
    Quantization Module (https://github.com/skmhrk1209/QuanTorch)
"""
# torch
import torch
from torch import nn
from torch import autograd


# ------------------------------------------------------------------------------
#    Quantizers
# ------------------------------------------------------------------------------
class Round(autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        return torch.floor(inputs + 0.5)

    @staticmethod
    def backward(ctx, grads):
        return grads


class Quantizer(nn.Module):
    def __init__(self, bits_precision, range_tracker):
        super().__init__()
        self.bits_precision = bits_precision
        self.range_tracker = range_tracker
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        self.disabled = False

    def update_params(self):
        raise NotImplementedError

    def quantize(self, inputs):
        outputs = inputs * self.scale - self.zero_point
        return outputs

    def round(self, inputs):
        # outputs = torch.round(inputs) + inputs - inputs.detach()
        outputs = Round.apply(inputs)
        return outputs

    def clamp(self, inputs):
        outputs = torch.clamp(inputs, self.min_val.to(inputs.get_device()), self.max_val.to(inputs.get_device()))
        return outputs

    def dequantize(self, inputs):
        outputs = (inputs + self.zero_point) / self.scale
        return outputs
    
    def disable_param_updates(self):
        self.disabled = True

    def enable_param_updates(self):
        self.disabled = False

    def forward(self, inputs):
        if not self.disabled:
            self.range_tracker(inputs)
        self.update_params()
        outputs = self.quantize(inputs)
        outputs = self.round(outputs)
        outputs = self.clamp(outputs)
        # print(outputs.min(), outputs.max())
        outputs = self.dequantize(outputs)
        return outputs


class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits_precision - 1))))
        self.register_buffer('max_val', torch.tensor((1 << (self.bits_precision - 1)) - 1))


class UnsignedQuantizer(SignedQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(0))
        self.register_buffer('max_val', torch.tensor((1 << self.bits_precision) - 1))


class SymmetricQuantizer(SignedQuantizer):
    def update_params(self):
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))
        float_range = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val))
        self.scale = quantized_range / (float_range + 1e-14)
        self.zero_point = torch.zeros_like(self.scale)


class AsymmetricQuantizer(UnsignedQuantizer):
    def update_params(self):
        quantized_range = self.max_val - self.min_val
        float_range = self.range_tracker.max_val - self.range_tracker.min_val
        self.scale = quantized_range / (float_range + 1e-14)
        self.zero_point = torch.round(self.range_tracker.min_val * self.scale)
