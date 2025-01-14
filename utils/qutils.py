"""
    Utils for the Quantizers
"""
import torch
from torch import nn

# custom
from utils.quantizer import SymmetricQuantizer, AsymmetricQuantizer
from utils.trackers import MovingAverageRangeTracker


# ------------------------------------------------------------------------------
#    To enable / disable quantization functionalities
# ------------------------------------------------------------------------------
class QuantizationEnabler(object):
    def __init__(self, model, wmode, amode, nbits, silent=False):
        self.model = model
        self.wmode = wmode
        self.amode = amode
        self.nbits = nbits
        self.quite = silent

    def __enter__(self):
        # loop over the model
        for module in self.model.modules():
            if isinstance(module, QuantizedConv2d) \
                or isinstance(module, QuantizedLinear):
                module.enable_quantization(self.wmode, self.amode, self.nbits)
                module.disable_parameter_updates()

                # to verbosely show
                if not self.quite:
                    print (type(module).__name__)
                    print (' : enable - ', module.quantization)
                    print (' : w-mode - ', module.wmode)
                    print (' : a-mode - ', module.amode)
                    print (' : n-bits - ', module.nbits)
                    print (' : w-track :', type(module.weight_quantizer).__name__, module.weight_quantizer.range_tracker.track)
                    print (' : a-track :', type(module.activation_quantizer).__name__, module.activation_quantizer.range_tracker.track)

        # report
        if not self.quite:
            print (' : convert to a quantized model [mode: {} / {}-bits]'.format(self.amode, self.nbits))

    def __exit__(self, exc_type, exc_value, traceback):
        # loop over the model
        for module in self.model.modules():
            if isinstance(module, QuantizedConv2d) \
                or isinstance(module, QuantizedLinear):
                module.disable_quantization()

        # report
        if not self.quite:
            print (' : restore a FP model from the quantized one [mode: {} / {}-bits]'.format(self.amode, self.nbits))

class QuantizationParameterEnabler(object):
    def __init__(self, model, wmode, amode, nbits, silent=False, reset=True):
        self.model = model
        self.wmode = wmode
        self.amode = amode
        self.nbits = nbits
        self.quite = silent
        self.reset = reset

    def __enter__(self):
        # loop over the model
        for module in self.model.modules():
            if isinstance(module, QuantizedConv2d) \
                or isinstance(module, QuantizedLinear):
                if self.reset:
                    module.reset_quantization_parameters()
                module.enable_quantization(self.wmode, self.amode, self.nbits)
                module.enable_parameter_updates()

                # to verbosely show
                if not self.quite:
                    print (type(module).__name__)
                    print (' : enable - ', module.quantization)
                    print (' : w-mode - ', module.wmode)
                    print (' : a-mode - ', module.amode)
                    print (' : n-bits - ', module.nbits)
                    print (' : w-track :', type(module.weight_quantizer).__name__, module.weight_quantizer.range_tracker.track)
                    print (' : a-track :', type(module.activation_quantizer).__name__, module.activation_quantizer.range_tracker.track)

        # report
        if not self.quite:
            print (' : convert to a quantized model [mode: {} / {}-bits]'.format(self.amode, self.nbits))

    def __exit__(self, exc_type, exc_value, traceback):
        # loop over the model
        for module in self.model.modules():
            if isinstance(module, QuantizedConv2d) \
                or isinstance(module, QuantizedLinear):
                module.disable_quantization()

        # report
        if not self.quite:
            print (' : restore a FP model from the quantized one [mode: {} / {}-bits]'.format(self.amode, self.nbits))

# ------------------------------------------------------------------------------
#    Quantized layers (Conv2d and Linear)
# ------------------------------------------------------------------------------
class QuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # default behavior is false...
        self.quantization  = False
        self.wmode         = None
        self.amode         = None
        self.nbits         = None

        # set the trackers
        self.reset_quantization_parameters()
        # done.


    """
        To enable / disable quantization functionalities
    """
    def enable_quantization(self, wmode, amode, nbits):
        # set the internals
        self.quantization = True
        self.wmode        = wmode
        self.amode        = amode
        self.nbits        = nbits

        # --------------------------------
        # set the weight / activation tracker channels
        if 'per_layer' in self.wmode:     wtrack_channel = 1
        elif 'per_channel' in self.wmode: wtrack_channel = self.out_channels

        if 'per_layer' in self.amode:     atrack_channel = 1
        elif 'per_channel' in self.amode: atrack_channel = self.out_channels

        # set the weight quantizer
        if 'asymmetric' in self.wmode:
            self.weight_quantizer = AsymmetricQuantizer(
                bits_precision=self.nbits, range_tracker=self.wtracker)
        elif 'symmetric' in self.wmode:
            self.weight_quantizer = SymmetricQuantizer(
                bits_precision=self.nbits, range_tracker=self.wtracker)
        else:
            assert False, ('Error: unknown quantization scheme [w: {}]'.format(self.wmode))

        # set the activation quantizer
        if 'asymmetric' in self.amode:
            self.activation_quantizer = AsymmetricQuantizer( \
                bits_precision=self.nbits, range_tracker=self.atracker)
        elif 'symmetric' in self.amode:
            self.activation_quantizer = SymmetricQuantizer( \
                bits_precision=self.nbits, range_tracker=self.atracker)
        else:
            assert False, ('Error: unknown quantization scheme [a: {}]'.format(self.amode))
        # done.


    def disable_quantization(self):
        self.quantization = False
        self.wmode        = None
        self.amode        = None
        self.nbits        = None
        self.atracker     = self.activation_quantizer.range_tracker
        self.wtracker     = self.weight_quantizer.range_tracker
        # done.

    def enable_parameter_updates(self):
        self.activation_quantizer.enable_param_updates()
        self.weight_quantizer.enable_param_updates()

    def disable_parameter_updates(self):
        self.activation_quantizer.disable_param_updates()
        self.weight_quantizer.disable_param_updates()

    def reset_quantization_parameters(self):
        self.wtracker = MovingAverageRangeTracker(shape = (self.out_channels, 1, 1, 1), momentum=1, track=True)
        self.atracker = MovingAverageRangeTracker(shape = (1, self.out_channels, 1, 1), momentum=1, track=True)


    """
        Forward function
    """
    def forward(self, inputs):
        if self.quantization:
            inputs = self.activation_quantizer(inputs)
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight

        outputs = nn.functional.conv2d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        return outputs


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias = True):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )

        # default behavior is false...
        self.quantization  = False
        self.wmode         = None
        self.amode         = None
        self.nbits         = None

        # set the trackers
        self.reset_quantization_parameters()
        # done.


    """
        To enable / disable quantization functionalities
    """
    def enable_quantization(self, wmode, amode, nbits):
        # set the internals
        self.quantization  = True
        self.wmode         = wmode
        self.amode         = amode
        self.nbits         = nbits

        # --------------------------------
        # set the weight / activation tracker channels
        if 'per_layer' in self.wmode:     wtrack_channel = 1
        elif 'per_channel' in self.wmode: wtrack_channel = self.out_features

        if 'per_layer' in self.amode:     atrack_channel = 1
        elif 'per_channel' in self.amode: atrack_channel = self.out_features

        # set the weight quantizer
        if 'asymmetric' in self.wmode:
            self.weight_quantizer = AsymmetricQuantizer(
                bits_precision=self.nbits, range_tracker=self.wtracker)
        elif 'symmetric' in self.wmode:
            self.weight_quantizer = SymmetricQuantizer(
                bits_precision=self.nbits, range_tracker=self.wtracker)
        else:
            assert False, ('Error: unknown quantization scheme [w: {}]'.format(self.wmode))

        # set the activation quantizer
        if 'asymmetric' in self.amode:
            self.activation_quantizer = AsymmetricQuantizer( \
                bits_precision=self.nbits, range_tracker=self.atracker)
        elif 'symmetric' in self.amode:
            self.activation_quantizer = SymmetricQuantizer( \
                bits_precision=self.nbits, range_tracker=self.atracker)
        else:
            assert False, ('Error: unknown quantization scheme [a: {}]'.format(self.amode))
        # done.


    def disable_quantization(self):
        self.quantization  = False
        self.wmode         = None
        self.amode         = None
        self.nbits         = None
        self.atracker     = self.activation_quantizer.range_tracker
        self.wtracker     = self.weight_quantizer.range_tracker 
        # done.

    def enable_parameter_updates(self):
        self.activation_quantizer.enable_param_updates()
        self.weight_quantizer.enable_param_updates()

    def disable_parameter_updates(self):
        self.activation_quantizer.disable_param_updates()
        self.weight_quantizer.disable_param_updates()

    def reset_quantization_parameters(self):
        self.wtracker = MovingAverageRangeTracker(shape = (self.out_features, 1), momentum=1, track=True)
        self.atracker = MovingAverageRangeTracker(shape = (1, self.out_features), momentum=1, track=True)

    """
        Forward function
    """
    def forward(self, inputs):
        if self.quantization:
            inputs = self.activation_quantizer(inputs)
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight

        outputs = nn.functional.linear(
            input=inputs,
            weight=weight,
            bias=self.bias,
        )

        return outputs
