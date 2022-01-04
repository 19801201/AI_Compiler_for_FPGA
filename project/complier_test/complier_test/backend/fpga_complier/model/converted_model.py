from torch.quantization import QuantStub, DeQuantStub
import torch
import torch.nn as nn

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, stride, dilation=1):
        super(Conv_BN_ReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convs(x)



class irmodel(nn.Module):

    def __init__(self, prun_flag=0, anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)],in_channels=[1, 8, 32, 64, 64, 128, 256, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128]  ,out_channels=[8, 32, 64, 64, 128, 256, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128, 25]  ):

        super(irmodel, self).__init__()
        self.anchors = anchors
        self.prun_flag = prun_flag
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_0 = Conv_BN_ReLU(in_channels[0],out_channels[0],3,1,2)
        self.conv_1 = Conv_BN_ReLU(in_channels[1],out_channels[1],3,1,2)
        self.conv_2 = Conv_BN_ReLU(in_channels[2],out_channels[2],3,1,2)
        self.conv_3 = Conv_BN_ReLU(in_channels[3],out_channels[3],3,1,1)
        self.conv_4 = Conv_BN_ReLU(in_channels[4],out_channels[4],3,1,1)
        self.conv_5 = Conv_BN_ReLU(in_channels[5],out_channels[5],3,1,1)
        self.conv_6 = Conv_BN_ReLU(in_channels[6],out_channels[6],3,1,2)
        self.conv_7 = Conv_BN_ReLU(in_channels[7],out_channels[7],3,1,1)
        self.conv_8 = Conv_BN_ReLU(in_channels[8],out_channels[8],3,1,1)
        self.conv_9 = Conv_BN_ReLU(in_channels[9],out_channels[9],3,1,1)
        self.conv_10 = Conv_BN_ReLU(in_channels[10],out_channels[10],3,1,1)
        self.conv_11 = Conv_BN_ReLU(in_channels[11],out_channels[11],3,1,1)
        self.conv_12 = Conv_BN_ReLU(in_channels[12],out_channels[12],3,1,1)
        self.conv_13 = Conv_BN_ReLU(in_channels[13],out_channels[13],3,1,1)
        self.conv_14 = Conv_BN_ReLU(in_channels[14],out_channels[14],3,1,2)
        self.conv_15 = Conv_BN_ReLU(in_channels[15],out_channels[15],3,1,1)




        self.conv_last = nn.Conv2d(out_channels[-2], len(self.anchors) * 5, 1, 1, 0, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()   
    def forward(self, x):
        x = self.quant(x)
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.conv_10(x)
        x = self.conv_11(x)
        x = self.conv_12(x)
        x = self.conv_13(x)
        x = self.conv_14(x)
        x = self.conv_15(x)


        
        x = self.conv_last(x)
        output= self.dequant(x)
        


        return output                 
    def fuse_model(self):
        for m in self.children():
            if isinstance(m, Conv_BN_ReLU):
                torch.quantization.fuse_modules(m.convs, [['0', '1', '2']], inplace=True)
                