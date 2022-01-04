# -*- coding: UTF-8 -*-
#encoding=utf-8
import torch
import torch.nn as nn
import numpy as np 
#np.set_printoptions(threshold=np.inf)
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
sys.path.append("..")
import torch.nn.functional as F

import torch.quantization
from torch.nn.quantized import functional as qF
# # Setup warnings
import warnings
from torch.quantization import QuantStub, DeQuantStub



def gen_M_N(S1,S2,S3):
    M=(S1*S2)/S3
    M=M.numpy()
    daxiao=S2.shape[0]
    SCALE=np.zeros(daxiao, dtype = np.uint32, order = 'C')
    N_REAL=np.zeros(daxiao, dtype = np.uint32, order = 'C')
    for i,ii in enumerate(M):
                
        while not(ii>=0.5 and ii<=1.0):
                ii*=2

        pass
           
        mmmm=ii*(2**32)

        SCALE[i]=mmmm.astype(np.uint32) 


    for i,ii in enumerate(M):
        
        
        N_REAL[i]=round(math.log(SCALE[i]/ii,2))-32-1
        

    return   SCALE,N_REAL     
  

def gen_int_bias(s1,s2,bias_float):
    aa=bias_float/s1
    bb= torch.div(aa, s2)
    for i,m in enumerate(bb):
        bb[i]=round(m.item())                 
    bias=bb.int()
    return bias
    
def gen_M(s1,s2,s3):
    aa=s1*s2
    M=aa/s3
    return M
def dequantize(scale,zero_point,x):
    #qm=nn.quantized.Quantize(scale, zero_point, dtype)
    qt=scale*(x-zero_point)
    return qt


def Quantize(scale,zero_point,dtype,x):
    qm=nn.quantized.Quantize(scale, zero_point, dtype)
    qt = qm(x)
    return qt
def gen_int_image(imgs,quant_scale,quant_zero_point):

    #img_q=torch.quantize_per_tensor(imgs, float(quant_scale), int(quant_zero_point), dtype=torch.quint8)
    img_q=Quantize(quant_scale,quant_zero_point,torch.quint8,imgs)
    img_q=img_q.int_repr()
    return img_q  

class Conv2d_Q(nn.Module):
    def __init__(
        self,

        quant_scale1=None,
        quant_zero_point1=None,
        quant_scale2=None,
        quant_zero_point2=None,  
        quant_scale3=None,
        quant_zero_point3=None,             

    ):
        super(Conv2d_Q, self).__init__()
        self.quant_scale1 = quant_scale1
        self.quant_zero_point1 = quant_zero_point1
        self.quant_scale2 = quant_scale2
        self.quant_zero_point2 = quant_zero_point2
        self.quant_scale3 = quant_scale3
        self.quant_zero_point3 = quant_zero_point3


    def forward(self, x,q_weight,bias,path1,coee=0):

        bias = np.array(bias.data.cpu().numpy(), dtype=np.uint32)
       
        
        SCALE,N_REAL=gen_M_N(self.quant_scale1,self.quant_scale2,self.quant_scale3)

        fp1 = open(path1, "wb") # 打开fp1 ab+追加写入

        q_weight=torch.as_tensor(q_weight,dtype=torch.int8)
        q_weight = np.array(q_weight.data.cpu().numpy(), dtype=np.int8) 
    
        shape = q_weight.shape
        if(shape[0]%8 !=0):
            kernel_num=shape[0]+8-shape[0]%8
        else:
            kernel_num=shape[0]
        if(shape[1]%8 !=0):
            channel_in_num=shape[1]+8-shape[1]%8
        else:
            channel_in_num=shape[1] 
        new_weig = np.zeros((kernel_num,channel_in_num,shape[2],shape[3]))
        new_weight = add_weight_channel(new_weig,q_weight,shape)
        new_shape = new_weight.shape
        daxiao=new_shape[0]*new_shape[1]*new_shape[2]*new_shape[3]
        
        weight=np.zeros(daxiao, dtype = np.uint8, order = 'C')
        get_weight(new_weight,new_shape,weight)
        if(coee==1):
            out=[]

            with open("bias.coe", "a+") as fp: 
                for r in weight:
                    out.append(r)
                    if len(out) == 32:
                        out.reverse()
                        for m in out:
                            
                            m=m.item()
                           
                            fp.write('%02x'%m)

                        fp.write(',\n')
                        out = []    
            with open("bias.coe", "a+") as fp: 
                for r in bias:
                    out.append(r)
                    if len(out) == 8:
                        out.reverse()
                        for m in out:
                            m=m.item()
                            fp.write('%08x'%m)

                        fp.write(',\n')
                        out = []  
            with open("bias.coe", "a+") as fp: 
                for r in SCALE:
                    out.append(r)
                    if len(out) == 8:
                        out.reverse()
                        for m in out:
                            m=m.item()
                            fp.write('%08x'%m)
                        fp.write(',\n')
                        out = []   
            with open("bias.coe", "a+") as fp: 
                print(N_REAL)
                for r in N_REAL:
                    out.append(r)
                    if len(out) == 8:
                        out.reverse()
                        for m in out:
                            m=m.item()
                            fp.write('%08x'%m) 
                        fp.write(',\n')
                        out = []                                   
                    

        weight.tofile(fp1)

        bias.tofile(fp1)        # 追加
        SCALE.tofile(fp1)
        N_REAL.tofile(fp1)
      
        fp1.close()
        # fp2 = open(path2, "ab+")
        # weight.tofile(fp2)
        # fp2.close()      

def get_weight(new_weight,shape,weight):
 
        j=0
        for i in range(shape[2]): #row
            for ii in range(shape[3]):#col
                for kernel_times in range(shape[0]>>3):#kernel_num/8
                    for channel_in_times in range(shape[1]>>3):#channel_in_num/8
                        for iii in range(8):
                            for iiii in range(8):
                                weight[j]=new_weight[kernel_times*8+iii][channel_in_times*8+iiii][i][ii]
                                j+=1

def add_weight_channel(new_weig,weig,shape):
    for kernel_num in range(shape[0]): 
        for channel_in_num in range(shape[1]):
            for row in range(shape[2]):
                for col in range(shape[3]):
                    new_weig[kernel_num][channel_in_num][row][col]=weig[kernel_num][channel_in_num][row][col]
    return new_weig        
def tensorr(x):
    tensor_py  = torch.from_numpy(np.load(x))
    return tensor_py
def gen_txt(OUT):
        shape=OUT.shape        
        print(shape)    
        out = []
        with open("INT8SHOUXIE.txt", "w") as fp:  
                for r in range(shape[2]):#hang
                    for c in range(shape[3]):#lie
                        for ch in range(shape[1]):#channel
                            for n in range(shape[0]):#image_num 
                                out.append(OUT[n][ch][r][c])

                                if len(out) == 32: 
                                    out.reverse()
                                    for m in out:
                                        m=m.int()
                                        m=m.item()
                                        
                                        fp.write('%02x'%m)
                                    fp.write(',\n')
                                    out = []
       

        print('OKKKKKKK')
        exit()   
       
class QuantizableYolo_tiny(nn.Module):

    def __init__(self, img_size=416):
        super(QuantizableYolo_tiny,self).__init__() 

        s1=tensorr('./q_paras/quant.scale.npy')
        z1=tensorr('./q_paras/quant.zero_point.npy')
        s2=tensorr('./q_paras/conv_0.convs.0.weight.scale.npy')
        z2=tensorr('./q_paras/conv_0.convs.0.weight.zero_point.npy')
        s3=tensorr('./q_paras/conv_0.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_0.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_0.convs.0.bias.npy')
        bias=gen_int_bias(s1,s2,bias_f)
     

        self.bias0 =bias            
        self.convs0 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        # gogogo
        s1 = tensorr('./q_paras/conv_0.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_1.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_1.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_1.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_1.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_1.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias1 = bias
        self.conv1 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_1.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_2.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_2.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_2.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_2.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_2.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias2 = bias
        self.conv2 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_2.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_3.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_3.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_3.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_3.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_3.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias3 = bias
        self.conv3 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_3.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_4.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_4.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_4.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_4.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_4.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias4 = bias
        self.conv4 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_4.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_5.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_5.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_5.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_5.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_5.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias5 = bias
        self.conv5 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_5.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_6.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_6.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_6.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_6.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_6.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias6 = bias
        self.conv6 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_6.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_7.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_7.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_7.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_7.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_7.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias7 = bias
        self.conv7 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_7.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_8.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_8.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_8.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_8.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_8.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias8 = bias
        self.conv8 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_8.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_9.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_9.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_9.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_9.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_9.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias9 = bias
        self.conv9 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_9.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_10.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_10.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_10.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_10.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_10.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias10 = bias
        self.conv10 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_10.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_11.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_11.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_11.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_11.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_11.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias11 = bias
        self.conv11 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_11.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_12.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_12.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_12.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_12.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_12.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias12 = bias
        self.conv12 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_12.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_13.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_13.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_13.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_13.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_13.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias13 = bias
        self.conv13 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_13.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_14.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_14.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_14.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_14.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_14.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias14 = bias
        self.conv14 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)
        s1 = tensorr('./q_paras/conv_14.convs.0.scale.npy')
        s2 = tensorr('./q_paras/conv_15.convs.0.weight.scale.npy')
        z2 = tensorr('./q_paras/conv_15.convs.0.weight.zero_point.npy')
        s3 = tensorr('./q_paras/conv_15.convs.0.scale.npy')
        z3=tensorr('./q_paras/conv_15.convs.0.zero_point.npy')
        bias_f=tensorr('./q_paras/conv_15.convs.0.bias.npy')
        gen_int_bias(s1,s2,bias_f)
        self.bias15 = bias
        self.conv15 = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)


    def forward(self,x): 

        path='biasscaleshift.bin'


        ########################################################
        weight=tensorr('./q_paras/conv_0.convs.0.weight.int.npy')

        x  =  self.convs0(1,weight,self.bias0,path,coee=0)
        ### aoaoao
        weight_convs1=tensorr('./q_paras/conv_1.convs.0.weight.int.npy')
        x  =  self.conv1(1,weight_convs1,self.bias1,path,coee=0)
        weight_convs2=tensorr('./q_paras/conv_2.convs.0.weight.int.npy')
        x  =  self.conv2(1,weight_convs2,self.bias2,path,coee=0)
        weight_convs3=tensorr('./q_paras/conv_3.convs.0.weight.int.npy')
        x  =  self.conv3(1,weight_convs3,self.bias3,path,coee=0)
        weight_convs4=tensorr('./q_paras/conv_4.convs.0.weight.int.npy')
        x  =  self.conv4(1,weight_convs4,self.bias4,path,coee=0)
        weight_convs5=tensorr('./q_paras/conv_5.convs.0.weight.int.npy')
        x  =  self.conv5(1,weight_convs5,self.bias5,path,coee=0)
        weight_convs6=tensorr('./q_paras/conv_6.convs.0.weight.int.npy')
        x  =  self.conv6(1,weight_convs6,self.bias6,path,coee=0)
        weight_convs7=tensorr('./q_paras/conv_7.convs.0.weight.int.npy')
        x  =  self.conv7(1,weight_convs7,self.bias7,path,coee=0)
        weight_convs8=tensorr('./q_paras/conv_8.convs.0.weight.int.npy')
        x  =  self.conv8(1,weight_convs8,self.bias8,path,coee=0)
        weight_convs9=tensorr('./q_paras/conv_9.convs.0.weight.int.npy')
        x  =  self.conv9(1,weight_convs9,self.bias9,path,coee=0)
        weight_convs10=tensorr('./q_paras/conv_10.convs.0.weight.int.npy')
        x  =  self.conv10(1,weight_convs10,self.bias10,path,coee=0)
        weight_convs11=tensorr('./q_paras/conv_11.convs.0.weight.int.npy')
        x  =  self.conv11(1,weight_convs11,self.bias11,path,coee=0)
        weight_convs12=tensorr('./q_paras/conv_12.convs.0.weight.int.npy')
        x  =  self.conv12(1,weight_convs12,self.bias12,path,coee=0)
        weight_convs13=tensorr('./q_paras/conv_13.convs.0.weight.int.npy')
        x  =  self.conv13(1,weight_convs13,self.bias13,path,coee=0)
        weight_convs14=tensorr('./q_paras/conv_14.convs.0.weight.int.npy')
        x  =  self.conv14(1,weight_convs14,self.bias14,path,coee=0)
        weight_convs15=tensorr('./q_paras/conv_15.convs.0.weight.int.npy')
        x  =  self.conv15(1,weight_convs15,self.bias15,path,coee=0)



        
        return x



QuantizableYolo_tiny()(1)
