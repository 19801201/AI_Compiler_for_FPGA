from .network_blocks import *
from torch.quantization import QuantStub, DeQuantStub
import torch
import torch.nn as nn

def gen_txt(OUT):
        shape=OUT.shape        
        print(shape)    
        out = []
        with open("OUT.txt", "w") as fp:  
                for r in range(shape[0]):#hang
                    for c in range(shape[1]):#lie
                        for ch in range(shape[2]):#channel

                                

  
                                       
                                        m=OUT[r][c][ch].item()
                                        
                                        fp.write(str(m))
                                        fp.write(',\n')
                                        
       
       

        print('OKKKKKKK')


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
        # self.conv_0 = nn.Conv2d(1, 16, 3, 1, 2)
        self.conv_0 = Conv_BN_LeakyReLU(1, 8, 3, 1, 2)
        self.conv_2 = Conv_BN_LeakyReLU(8, 16, 3, 1, 1)
        # self.conv_4 = Conv_BN_LeakyReLU(78, 78, 3, 1, 1)
        # self.conv_5 = Conv_BN_LeakyReLU(78, 78, 3, 1, 1)
        self.conv_6 = Conv_BN_LeakyReLU(16, 16, 3, 1, 1)
        self.conv_8 = Conv_BN_LeakyReLU(16, 32, 3, 1, 1)
        self.conv_9 = Conv_BN_LeakyReLU(32, 32, 3, 1, 1)
        self.conv_10 = Conv_BN_LeakyReLU(32, 32, 3, 1, 2)
        # self.conv_12 = Conv_BN_LeakyReLU(315, 315, 3, 1, 1)
        self.conv_13 = Conv_BN_LeakyReLU(32, 64, 3, 1, 2)
        self.conv_14 = Conv_BN_LeakyReLU(64, 64, 3, 1, 2)
        self.conv_15 = Conv_BN_LeakyReLU(64, 64, 3, 1, 1)
        self.conv_16 = Conv_BN_LeakyReLU(64, 64, 3, 1, 2) #
        self.maxpool_17 = Conv_BN_LeakyReLU(64, 64, 3, 1, 1)
        self.route=nn.quantized.FloatFunctional()
        ###
 
        self.conv_24 = Conv_BN_LeakyReLU(64, 64, 3, 1, 1)# conv16+24
  
        self.conv_28 = Conv_BN_LeakyReLU(64+64, 128, 3, 1, 1)
        self.conv_29 = nn.Conv2d(128, len(self.anchors) * 5, 1, 1, 0, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()   

#1024 - little      label
    def forward(self, x):
        x = self.quant(x)#去掉了

        # 
        # x = torch.zeros((4,1,416,416))
        # x[0][0][0][0] = 1
        # x[1][0][0][0] = 0
        # x[2][0][0][0] = 0
        # x[3][0][0][0] = 0

        # x = torch.zeros((1,1,416,416))
        # x[0][0][0][0] = 1

        # x = self.conv_0.convs[0](x) # 卷积

        # x = torch.zeros((4,16,208,208))
        # x[0][0][0][0] = 1
        # x[1][0][0][0] = 1
        # x[2][0][0][0] = 1
        # x[3][0][0][0] = 1

        # print('weight:',self.conv_0.convs[1].weight)
        # print('bias:',self.conv_0.convs[1].bias)  
        # print('mean:',self.conv_0.convs[1].running_mean)
        # print('var:',self.conv_0.convs[1].running_var)
        # print('num_batches_tracked:',self.conv_0.convs[1].num_batches_tracked)  
                            

        # x = self.conv_0.convs[1](x) #BN 
        
        # print(x[0].shape)
        # exit()
        # print(x[0].shape)#这么写是3各维度
        # gen_txt(x[0])
        # print(x[0])
        
        # exit()       
        #print(x.shape)
       
        x = self.conv_0(x)
        #print(x.shape)
        x = self.conv_2(x)
        #print(x.shape)
        # x = self.conv_4(x)

        # x = self.conv_5(x)
   
        x = self.conv_6(x)
        #print(x.shape)
        x = self.conv_8(x)
        #print(x.shape)
        x = self.conv_9(x)
        #print(x.shape)
        x = self.conv_10(x)
        #print(x.shape)
        # x = self.conv_12(x)
        #print(x.shape)
        x = self.conv_13(x)
        #print(x.shape)
        x = self.conv_14(x)
        #print(x.shape)
        # x = self.conv_15(x)    #we cut this
        #print(x.shape)
        x16 = self.conv_16(x)
        #print(x16.shape)
        x = self.maxpool_17(x16)
        #print(x.shape)





        x24 = self.conv_24(x)
        #print(x24.shape)
        # exit()
  

        x = self.route.cat([x16, x24], 1)


        x = self.conv_28(x)
        #print(x.shape)
       
        
        output = self.conv_29(x)
       # print(output.shape)
        output= self.dequant(output)
        


        return output                 
    def fuse_model(self):
        for m in self.children():
            if isinstance(m, Conv_BN_LeakyReLU):
                torch.quantization.fuse_modules(m.convs, [['0', '1', '2']], inplace=True)
                