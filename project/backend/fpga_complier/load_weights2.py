#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model.converted_model import irmodel
import torch
import numpy as np
import torch.nn as nn


# In[2]:


model = irmodel()


# In[3]:


i_conv = 0
i_bn = 0
num_conv = 0
num_bn = 0
for l in model.modules():
    if isinstance(l,nn.Conv2d):
        num_conv = num_conv+1
for ll in model.modules():
    if isinstance(ll,nn.BatchNorm2d):
        num_bn = num_bn+1
# 写到num_bn层参数提取了
for m in model.modules():
    if i_conv > num_conv:
        break
    if isinstance(m, nn.Conv2d):
        if i_conv != (num_conv - 1):
            weights = torch.from_numpy(np.array(np.load("paras/conv_%s_weights.npy"%i_conv),dtype=float)).permute(3,2,0,1).type(torch.FloatTensor)
            m.weight.data = weights.clone()            
            bias = torch.from_numpy(np.array(np.load("paras/conv_%s_bias.npy"%i_conv),dtype=float)).type(torch.FloatTensor)
            m.bias.data = bias
        else: # last conv
            try:
                weights = torch.from_numpy(np.array(np.load("paras/conv_%s_weights.npy"%i_conv),dtype=float)).permute(3,2,0,1).type(torch.FloatTensor)
                m.weight.data = weights.clone()   
            except:
                pass
        i_conv = i_conv+1
        
    elif isinstance(m, nn.BatchNorm2d):
        if i_bn <= num_bn:
            scale = torch.from_numpy(np.array(np.load("paras/bn%s_scale.npy"%i_bn),dtype=float)).type(torch.FloatTensor)
            var = torch.from_numpy(np.array(np.load("paras/bn%s_var.npy"%i_bn),dtype=float)).type(torch.FloatTensor)
            bias = torch.from_numpy(np.array(np.load("paras/bn%s_bias.npy"%i_bn),dtype=float)).type(torch.FloatTensor)
            mean = torch.from_numpy(np.array(np.load("paras/bn%s_mean.npy"%i_bn),dtype=float)).type(torch.FloatTensor)
            eps = np.array(np.load("paras/bn%s_eps.npy"%i_bn),dtype=float)
            m.weight.data = scale.clone()
            m.bias.data = bias.clone()
            m.running_var = var.clone()
            m.running_mean = mean.clone()
            m.eps = float(eps)
            i_bn = i_bn+1
    elif isinstance(m,nn.ReLU):
        pass


# In[4]:


torch.save(model.state_dict(),"./model/weights.pth")
torch.save(model,"./model/model.pth")


# # test

# In[5]:


input1 = torch.ones((1,1,416,416))
model(input1)


# In[ ]:




