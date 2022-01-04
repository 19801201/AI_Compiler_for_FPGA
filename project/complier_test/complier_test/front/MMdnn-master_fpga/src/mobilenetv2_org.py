from .network_blocks import *
from torch.quantization import QuantStub, DeQuantStub
import torch
import torch.nn as nn
def create_mobilenet_v2(inverted_residual_setting=None):
    """
    MobileNet V2 main class

    """
    block = InvertedResidual
    input_channel = 32
    last_channel = 1024

    if inverted_residual_setting is None:
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

    # only check the first element, assuming user knows t,c,n,s are required
    if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
        raise ValueError("inverted_residual_setting should be non-empty "
                         "or a 4-element list, got {}".format(inverted_residual_setting))

    # building first layer
    mlist = nn.ModuleList()
    mlist.append(ConvBNReLU(1, input_channel, stride=2))
    # building inverted residual blocks
    for t, c, n, s in inverted_residual_setting:
        output_channel = c
        for i in range(n):
            stride = s if i == 0 else 1
            mlist.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
    # building last several layers
    mlist.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))   #18

    return mlist


class mobilenetv2(nn.Module):

    def __init__(self, anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)]):
        """
        mobilnetv2

        """
        super(mobilenetv2, self).__init__()
        self.anchors = anchors
        self.module_list = create_mobilenet_v2()
        self.conv = nn.Conv2d(1024, len(self.anchors) * 5, 1, 1, 0, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()   

    def forward(self, x):
        x = self.quant(x)
        for i, module in enumerate(self.module_list):
   
            x = module(x)
        output = self.conv(x)
        output= self.dequant(output)
        return output
    def fuse_model(self):
     
     
        for m in self.module_list:

            if isinstance(m, ConvBNReLU):
                torch.quantization.fuse_modules(m, [['0', '1', '2']], inplace=True)
            if isinstance(m, InvertedResidual):
                for i in m.conv:
                    if isinstance(i, ConvBNReLU):
                        torch.quantization.fuse_modules(i, [['0', '1', '2']], inplace=True)

                if (str(m.conv[1])[0:6]=="Conv2d"):
                    torch.quantization.fuse_modules(m.conv, [['1', '2']], inplace=True)


                                    
                if (str(m.conv[2])[0:6]=="Conv2d"):
                    torch.quantization.fuse_modules(m.conv, [['2', '3']], inplace=True) 


                 