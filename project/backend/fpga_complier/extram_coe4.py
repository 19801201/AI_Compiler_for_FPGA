#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 卷积层数
import shutil 
import torch.nn as nn
from model.converted_model import irmodel
model = irmodel()
numconv = 0
for l in model.modules():
    if isinstance(l,nn.Conv2d):
        numconv = numconv+1
conv_num =numconv-1

shutil.copyfile("./irs/extra_prame_coe.py", "./extra_prame_coe.py")  


# In[ ]:





# In[2]:


# __init__
#conv 记得减一，第一层写好了
fp = open("postir.txt","w+")
for i in range(conv_num-1):
    strin1 = "        s1 = tensorr('./q_paras/conv_%s.convs.0.scale.npy')\n"%(i)
    strin2 = "        s2 = tensorr('./q_paras/conv_%s.convs.0.weight.scale.npy')\n"%(i+1)
    strin3 = "        z2 = tensorr('./q_paras/conv_%s.convs.0.weight.zero_point.npy')\n"%(i+1)
    strin4 = "        s3 = tensorr('./q_paras/conv_%s.convs.0.scale.npy')\n"%(i+1)
    strin5 = "        z3=tensorr('./q_paras/conv_%s.convs.0.zero_point.npy')\n"%(i+1)

    biass = "        bias_f=tensorr('./q_paras/conv_%s.convs.0.bias.npy')\n"%(i+1)
    bias = "        gen_int_bias(s1,s2,bias_f)\n"

    selfbias = "        self.bias%s = bias\n"%(i+1)
    selfconvs = "        self.conv%s = Conv2d_Q(quant_scale1=s1,quant_zero_point1=z1,quant_scale2=s2,quant_zero_point2=z2,quant_scale3=s3,quant_zero_point3=z3)\n"%(i+1)
    fp.write(strin1)
    fp.write(strin2)
    fp.write(strin3)
    fp.write(strin4)
    fp.write(strin5)
    fp.write(biass)
    fp.write(bias)
    fp.write(selfbias)
    fp.write(selfconvs)

fp.close()


with open('postir.txt','r',encoding='utf-8') as f11:
    content1 = f11.read()
    f11.close()

##############插入
f5 = open('extra_prame_coe.py','r')
content = f5.read()
inputstr = content1
keyword = '# gogogo\n'
post = content.find(keyword)
if post != -1:
    content = content[:post+len(keyword)]+inputstr+content[post+len(keyword):]
    f5 = open('extra_prame_coe.py','w')
    f5.write(content)
    f5.close()    
    


# In[3]:


# forward
# 这块不用减一
fp = open("postir2.txt","w+")
for i in range(1,conv_num):
    strin = "        weight_convs%s=tensorr('./q_paras/conv_%s.convs.0.weight.int.npy')\n"%(i,i)
    strin1 = "        x  =  self.conv%s(1,weight_convs%s,self.bias%s,path,coee=0)\n"%(i,i,i)
    fp.write(strin)
    fp.write(strin1)
fp.close()


with open('postir2.txt','r',encoding='utf-8') as f22:
    content2 =  f22.read()
    f22.close()
    
    
f5 = open('extra_prame_coe.py','r')
content = f5.read()
inputstr = content2
keyword = '### aoaoao\n'
post = content.find(keyword)
if post != -1:
    content = content[:post+len(keyword)]+inputstr+content[post+len(keyword):]
    f5 = open('extra_prame_coe.py','w')
    f5.write(content)
    f5.close()    


# In[4]:


import os
os.remove('postir.txt')
os.remove('postir2.txt')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# createVar = locals()

# city = ['a','b']
# for i in city:
#     createVar[i] = 16
    
    
# createVar = locals()

# conv_list = []
# for i in range(numconv):
#     convstr = 'self.conv_%s'%i
#     conv_list.append(convstr)
    
# for i in conv_list:
#     createVar[i] = 16

# class Employee: 
#     def __init__(self):
#         createVar = locals()

#         createVar['a']=3
#         self.d = 1
#     def displayCount(self):
#         print(self.a)
# b = Employee()
# b.displayCount()

