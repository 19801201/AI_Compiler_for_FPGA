#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
# f1 = open('./logs/flag_prun.txt')

# prun_flag = f1.read()
# f1.close()
prun_flag = 0
if prun_flag == 0:
    model = torch.load('./model/model.pth', map_location='cpu')
#     model.load_state_dict(torch.load('./model/weights.pth',map_location='cpu'))
else:
    model = torch.load('./prun_model/model.pth', map_location='cpu')
#     model.load_state_dict(torch.load('./prun_model/weights.pth',map_location='cpu'))        
ic_ls = model.in_channels[:-1]
oc_ls = model.out_channels[:-1]


# In[4]:


import json
with open("converted.json",'r') as load_f:
    load_dict = json.load(load_f)
a = load_dict
in_channels=[]
out_channels = []
k_sizes = []
strides = []
paddings = []
output_shapes_ls = []
num_conv = 0
for i in range(len(a['node'])):
    b = a['node'][i]['op']
    if b in 'Conv':
        num_conv = num_conv+1
        kernels = a['node'][i]['attr']['kernel_shape']['list']['i']
        stridea = a['node'][i]['attr']['strides']['list']['i']
        pads = a['node'][i]['attr']['pads']['list']['i'] 
        output_shapes = int(a['node'][i]['attr']['_output_shapes']['list']['shape'][0]['dim'][1]['size'])
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
        output_shapes_ls.append(output_shapes)
    if b in 'DataInput':
        img_size = int(a['node'][i]['attr']['_output_shapes']['list']['shape'][0]['dim'][1]['size'])
stride_ls = strides[:-1]  
output_shapes_ls = output_shapes_ls[:-1]
input_shapes_ls = output_shapes_ls.copy()
input_shapes_ls.insert(0,img_size)
input_shapes_ls = input_shapes_ls[:-1]
n_0 = [0] * len(input_shapes_ls)
n_1 = [1] * len(input_shapes_ls)
n_9 = [9] * len(input_shapes_ls)


# In[5]:


fo = open("para.txt", "w")
for i in range(len(input_shapes_ls)):
    fo.write(str(ic_ls[i]))
    fo.write(' ')
    fo.write(str(oc_ls[i]))
    fo.write(' ')
    fo.write(str(n_9[i]))
    fo.write(' ')
    fo.write(str(n_0[i]))
    fo.write(' ')
    fo.write(str(n_0[i]))
    fo.write(' ')
    fo.write(str(n_0[i]))
    fo.write(' ')
    fo.write(str(n_0[i]))
    fo.write(' ')
    fo.write(str(n_1[i]))
    fo.write(' ')
    fo.write(str(stride_ls[i]))
    fo.write(' ')
    fo.write(str(n_1[i]))
    fo.write(' ')
    fo.write(str(input_shapes_ls[i]))
    fo.write(' ')
    fo.write(str(output_shapes_ls[i]))
    fo.write('\n')

fo.close()


# In[6]:


out_size = []
for i,s in enumerate(output_shapes_ls):
    a = out_channels[i]*s*s*4
    out_size.append(a)
max_addr = max(out_size)
f1 = open('tmp_addr.txt','w')
f1.write(str(max_addr))
f1.close()


# In[9]:


# ic_ls
# oc_ls
# n_9
# n_0
# n_0
# n_0
# n_0
# n_1
# stride_ls
# n_1
# input_shapes_ls
# output_shapes_ls


# In[18]:





# In[21]:





# In[ ]:




