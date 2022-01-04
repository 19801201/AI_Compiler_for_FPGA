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
                          (11.2364, 10.0071)],in_channels=  ,out_channels=  ):

        super(irmodel, self).__init__()
        self.anchors = anchors
        self.prun_flag = prun_flag
        self.in_channels = in_channels
        self.out_channels = out_channels




        self.conv_last = nn.Conv2d(out_channels[-2], len(self.anchors) * 5, 1, 1, 0, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()   
    def forward(self, x):
        x = self.quant(x)


        
        x = self.conv_last(x)
        output= self.dequant(x)
        


        return output                 
    def fuse_model(self):
        for m in self.children():
            if isinstance(m, Conv_BN_ReLU):
                torch.quantization.fuse_modules(m.convs, [['0', '1', '2']], inplace=True)
                