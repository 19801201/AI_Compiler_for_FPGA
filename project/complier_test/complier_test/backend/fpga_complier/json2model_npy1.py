#!/usr/bin/env python
# coding: utf-8

# # json文件转化为py模型

# In[1]:


import json
with open("converted.json",'r') as load_f:
    load_dict = json.load(load_f)


# ##         原封不用：irs  自动生成：paras model  

# ## 需要 converted.json converted.npy ./irs/ir_model.py

# In[2]:


a = load_dict


# In[3]:


in_channels=[]
out_channels = []
k_sizes = []
strides = []
paddings = []
num_conv = 0
for i in range(len(a['node'])):
    b = a['node'][i]['op']
    if b in 'Conv':
        num_conv = num_conv+1
        kernels = a['node'][i]['attr']['kernel_shape']['list']['i']
        stridea = a['node'][i]['attr']['strides']['list']['i']
        pads = a['node'][i]['attr']['pads']['list']['i'] 
        
        k_size = int(kernels[0])
        in_channel = int(kernels[2])
        out_channel = int(kernels[3])
        stride = int(stridea[1])
        padding = int(pads[1])
        k_sizes.append(k_size)
        in_channels.append(in_channel)
        out_channels.append(out_channel)
        strides.append(stride)
        paddings.append(padding)

    


# # 构造backbone

# In[4]:


num_conv = num_conv - 1
f1 = open('init_model.txt','w+')
for i in range(num_conv):
    strin ="        self.conv_%s = Conv_BN_ReLU(in_channels[%s],out_channels[%s],%s,%s,%s)\n"%(i,i,i,k_sizes[i],paddings[i],strides[i])
    f1.write(strin)
f1.close()


# In[5]:


# f2 = open('in_channel.txt','w+')
# f2.write(str(in_channels))
# f2.close()
# f3 = open('out_channel.txt','w+')
# f3.write(str(out_channels))
# f3.close()
f4 = open('forward.txt','w+')
for i in range(num_conv):
    stri ="        x = self.conv_%s(x)\n"%(i)
    f4.write(stri)
f4.close()


# # backbone插入代码（自动生成）

# In[6]:


from shutil import copyfile
from sys import exit
copyfile('./irs/ir_model.py', './model/converted_model.py')

f5 = open('./model/converted_model.py','r')
content = f5.read()
inputstr = str(in_channels)
keyword = 'in_channels='
post = content.find(keyword)
if post != -1:
    content = content[:post+len(keyword)]+inputstr+content[post+len(keyword):]
    f5 = open('./model/converted_model.py','w')
    f5.write(content)
    f5.close()
    
    
f5 = open('./model/converted_model.py','r')
content = f5.read()
inputstr = str(out_channels)
keyword = 'out_channels='
post = content.find(keyword)
if post != -1:
    content = content[:post+len(keyword)]+inputstr+content[post+len(keyword):]
    f5 = open('./model/converted_model.py','w')
    f5.write(content)
    f5.close()
    
    
    
    
    
with open('init_model.txt','r',encoding='utf-8') as f11:
    content1 = f11.read()
with open('forward.txt','r',encoding='utf-8') as f22:
    content2 =  f22.read()
    
f5 = open('./model/converted_model.py','r')
content = f5.read()
inputstr = content1
keyword = 'self.out_channels = out_channels\n'
post = content.find(keyword)
if post != -1:
    content = content[:post+len(keyword)]+inputstr+content[post+len(keyword):]
    f5 = open('./model/converted_model.py','w')
    f5.write(content)
    f5.close()    
    
    
f5 = open('./model/converted_model.py','r')
content = f5.read()
inputstr = content2
keyword = 'x = self.quant(x)\n'
post = content.find(keyword)
if post != -1:
    content = content[:post+len(keyword)]+inputstr+content[post+len(keyword):]
    f5 = open('./model/converted_model.py','w')
    f5.write(content)
    f5.close()    
    


# # 保存权重

# In[7]:


import numpy as np
import json


# In[8]:


conv_list = []
bn_list = []
eps = []
with open("converted.json",'r') as load_f:
    aa = json.load(load_f)
for i in range(len(aa['node'])):
    bb = aa['node'][i]['op']
    if bb in 'Conv':
        conv_list.append(aa['node'][i]['name'])
    if bb in 'BatchNorm':
        bn_list.append(aa['node'][i]['name'])
        eps.append(aa['node'][i]['attr']['epsilon']['f'])
        


# In[9]:


c = np.load("converted.npy",allow_pickle=True).item()


# In[10]:


key_conv = []
for i in c.keys():
    key_conv.append(i)


# In[11]:


for i,j in enumerate(key_conv):
    if j in conv_list:
#         print(conv_list.index(j))  # 文件名顺序 0.npy 1.npy
#         print(j)
#         print(c[j]['weights'].shape)
        if 'weights' in c[j]:
            np.save("paras/conv_%s_weights.npy"%(conv_list.index(j)),c[j]['weights'])
        if 'bias' in c[j]:
            if c[j]['bias'].shape != 0:
                np.save("paras/conv_%s_bias.npy"%(conv_list.index(j)),c[j]['bias'])
    if j in bn_list:
#         print(bn_list.index(j))
#         print(j)
#         print(c[j].keys())
        if 'scale' in c[j]:
            np.save("paras/bn%s_scale.npy"%(bn_list.index(j)),c[j]['scale'])
        if 'var' in c[j]:
            np.save("paras/bn%s_var.npy"%(bn_list.index(j)),c[j]['var'])
        if 'bias' in c[j]:
            np.save("paras/bn%s_bias.npy"%(bn_list.index(j)),c[j]['bias'])            
        if 'mean' in c[j]:
#             print(c[j]['mean'].shape)
            np.save("paras/bn%s_mean.npy"%(bn_list.index(j)),c[j]['mean'])
            np.save("paras/bn%s_eps.npy"%(bn_list.index(j)),eps[(bn_list.index(j))])


# # 清除无用文件

# In[12]:


import os
os.remove("init_model.txt")
os.remove("forward.txt")


# In[ ]:




