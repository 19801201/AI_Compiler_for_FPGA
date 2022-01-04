from .network_blocks import *
from torch.quantization import QuantStub, DeQuantStub
import torch
import torch.nn as nn


class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride

        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, stride, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convs(x)



class mobilenetv2(nn.Module):

    def __init__(self, anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)]):
        """
        mobilnetv2

        """
        super(mobilenetv2, self).__init__()
        self.anchors = anchors  # voc anchors 1.3221
        self.conv_0 = Conv_BN_LeakyReLU(1, 32, 3, 1, 2)
        self.conv_2 = Conv_BN_LeakyReLU(32, 72, 3, 1, 2)
        # self.conv_4 = Conv_BN_LeakyReLU(78, 78, 3, 1, 1)
        # self.conv_5 = Conv_BN_LeakyReLU(78, 78, 3, 1, 1)
        self.conv_6 = Conv_BN_LeakyReLU(72, 152, 3, 1, 2)
        self.conv_8 = Conv_BN_LeakyReLU(152, 152, 3, 1, 1)
        self.conv_9 = Conv_BN_LeakyReLU(152, 152, 3, 1, 1)
        self.conv_10 = Conv_BN_LeakyReLU(152, 312, 3, 1, 2)
        # self.conv_12 = Conv_BN_LeakyReLU(315, 315, 3, 1, 1)
        self.conv_13 = Conv_BN_LeakyReLU(312, 312, 3, 1, 1)
        self.conv_14 = Conv_BN_LeakyReLU(312, 312, 3, 1, 1)
        self.conv_15 = Conv_BN_LeakyReLU(312, 312, 3, 1, 1)
        self.conv_16 = Conv_BN_LeakyReLU(312, 128, 3, 1, 1) #
        self.maxpool_17 = Conv_BN_LeakyReLU(128, 128, 3, 1, 2)
        self.route=nn.quantized.FloatFunctional()
        ###
 
        self.conv_24 = Conv_BN_LeakyReLU(128, 256, 3, 1, 1)#
        self.reorg = reorg_layer(stride=2) 
        self.conv_28 = Conv_BN_LeakyReLU(768, 256, 3, 1, 1)
        self.conv_29 = nn.Conv2d(256, len(self.anchors) * 5, 1, 1, 0, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()   

#1024 - little      label
    def forward(self, x):
        x = self.quant(x)
        x = self.conv_0(x)
        # print(x.shape)

        x = self.conv_2(x)
        # print(x.shape)
        # x = self.conv_4(x)
        # print(x.shape)
        # x = self.conv_5(x)
        # print(x.shape)
        x = self.conv_6(x)
        # print(x.shape)
        x = self.conv_8(x)
        # print(x.shape)
        x = self.conv_9(x)
        # print(x.shape)
        x = self.conv_10(x)
        # print(x.shape)
        # x = self.conv_12(x)
        # print(x.shape)
        x = self.conv_13(x)
        # print(x.shape)
        x = self.conv_14(x)
        # print(x.shape)
        x = self.conv_15(x)
        # print(x.shape)
        x16 = self.conv_16(x)
        # print(x16.shape)
        x = self.maxpool_17(x16)
        # print(x.shape)





        x24 = self.conv_24(x)
        # print(x24.shape)
        # exit()
        x = self.reorg(x16)

        x = self.route.cat([x, x24], 1)
        x = self.conv_28(x)
        output = self.conv_29(x)
        output= self.dequant(output)


        return output                 
    def fuse_model(self):
        for m in self.children():
            if isinstance(m, Conv_BN_LeakyReLU):
                torch.quantization.fuse_modules(m.convs, [['0', '1', '2']], inplace=True)
                