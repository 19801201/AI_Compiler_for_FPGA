import numpy as np
from nets.yolo4 import YoloBody
import random
import torch.nn as nn
import torch
np.random.seed(0)
model = YoloBody(3,20)
aaa = np.random.random((4,3,10,10))*255
aaa = aaa.astype(np.uint8)
aaa = torch.as_tensor(aaa,dtype=torch.float)
aaa1 = np.ones((4,1,1,1))
aaa2 = np.ones((4,1,1,1))*2
aaa3 = np.ones((4,1,1,1))*3
aaa4 = [aaa1,aaa2,aaa3]
exit()
x2, x1, x0= model.backbone(aaa)
P5 = model.conv1(x0)
pool_sizes=[13, 9, 5]
xxx=nn.MaxPool2d(13, 1, 13//2)
xxx = xxx(P5)
yyy=nn.MaxPool2d(9, 1, 9//2)
zzz=nn.MaxPool2d(5, 1, 5//2)
yyy = yyy(P5)
zzz = zzz(P5)
bbb=[xxx,yyy,zzz]
print(bbb)

# print(bbb.shape)
features = torch.cat(bbb + [P5], dim=1)
P5 = model.SPP(P5)
print(P5.shape)
# print(torch.equal(features,P5))