import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from src.utils import *
from src.evaluate_f1_qun import evaluate_f1

from torch.utils.data import DataLoader
from src.ship_dataset import SHIPDataset
from model.converted_model import irmodel
import copy 
### need dir:prun_model  src/*  file: model/converted_model.py  model/weights.pth  model/model.pth  
def test():
    model.eval()
    model.cpu()
    label_path = './images/labels'
    image_path = './images/images'
    total_ship, detected_ship, false_alarm, precision, recall, f1 = evaluate_f1(model, label_path, image_path, 800, thr=0.17,nms_threshold = 0.5)# 0.33 0.4
    # print("f1:",f1)
    return f1

parser = argparse.ArgumentParser()

# percent(剪枝率)
parser.add_argument('--optlevel', type=float, default=2,
                    help='level') ###
# 正常|规整剪枝标志
parser.add_argument('--normal_regular', type=int, default=8,
                    help='--normal_regular_flag (default: normal) 8倍数')
# 稀疏训练后的model
parser.add_argument('--iweights', default='model/weights.pth', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)') ###
# jiao
parser.add_argument('--imgtype',default=1,type=int, help='rgb=3,gray=1')

parser.add_argument("--futine", type=bool, default=False, help="finetune:True or False")

args = parser.parse_args()
base_number = args.normal_regular

if base_number <= 0:
    print('\r\n!base_number is error!\r\n')
    base_number = 1
##加载模型
# model = irmodel()
model = torch.load('./model/model.pth')
layers = len(model.in_channels)

if args.iweights:
    if os.path.isfile(args.iweights):
        print("=> loading checkpoint '{}'".format(args.iweights))
        load=torch.load(args.iweights)
        model.load_state_dict(load)
    else:
        print("=> no checkpoint found at '{}'".format(args.iweights))

# print('旧模型: ', model)
# ff = test()
# print("old model f1:",ff)

# exit()


## 计算剪枝阈值
total = 0
i = 0
for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if i < layers - 1:
                i += 1
                total += m.weight.data.shape[0]

# 确定剪枝的全局阈值
bn = torch.zeros(total)
index = 0
i = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
y, j = torch.sort(bn)
percent = 0.5
if args.optlevel == 0:
    percent = 0
elif args.optlevel == 1:
    percent = 0.3
elif args.optlevel == 2:
    percent = 0.5
elif args.optlevel == 3:
    percent = 0.7
thre_index = int(total * percent)
if thre_index == total:
    thre_index = total - 1
thre_0 = y[thre_index]

#********************************预剪枝*********************************
pruned = 0
cfg_0 = []
cfg = []
cfg_mask = []
i = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1

            weight_copy = m.weight.data.clone()

            mask = weight_copy.abs().gt(thre_0).float()
            remain_channels = torch.sum(mask)

            # m = weight_copy.tolist()
            # t = copy.deepcopy(m)
            # # 求m个最大的数值及其索引
            # max_number = []
            # max_index = []
            # for _ in range(8):
            #     number = max(t)
            #     index = t.index(number)
            #     t[index] = 0
            #     max_number.append(number)
            #     max_index.append(index)
            # t = []
            # print(max_index)
            if remain_channels < 8: ################################对最大的8个进行排序，并保存索引，对应mask设置为1
                sss = weight_copy.tolist()
                t = copy.deepcopy(sss)  
                max_number =[]
                max_index = []
                for _ in range(8):
              	
                    number = max(t)
                    index = t.index(number)
                    t[index] = 0
                    max_number.append(number)
                    max_index.append(index)
                t = []
                # print(max_index)   
                for jjj in max_index:
                    mask[jjj] = 1
                remain_channels = 8

                # remain_channels = weight_copy.shape[0]
                # mask = torch.ones(remain_channels)
                # print(remain_channels)

            if remain_channels == 0:
                print('\r\n!please turn down the prune_ratio!\r\n')
                remain_channels = 8
                mask[int(torch.argmax(weight_copy))]=1

            # ******************规整剪枝******************
            v = 0
            n = 1

            if remain_channels % base_number != 0:
                if remain_channels > base_number:
                    while v < remain_channels:
                        n += 1
                        v = base_number * n

                    if remain_channels - (v - base_number) < v - remain_channels:
                        remain_channels = v - base_number
                    else:
                        remain_channels = v
                    if remain_channels > m.weight.data.size()[0]:
                        remain_channels = m.weight.data.size()[0]
                    remain_channels = torch.tensor(remain_channels)
                        
                    y, j = torch.sort(weight_copy.abs())

                    thre_1 = y[-remain_channels]

                    mask = weight_copy.abs().ge(thre_1).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
         
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_0.append(mask.shape[0])
            cfg.append(int(remain_channels))
            cfg_mask.append(mask.clone())
            print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                format(k, mask.shape[0], int(torch.sum(mask)), (mask.shape[0] - torch.sum(mask)) / mask.shape[0]))
pruned_ratio = float(pruned/total)
print('\r\n!预剪枝完成!')
print('total_pruned_ratio: ', pruned_ratio)
# print(cfg)


#********************************预剪枝后model测试*********************************



# def test():
#     test_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR10(root = args.data, train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
#         batch_size = 64, shuffle=False, num_workers=1)
#     model.eval()
#     correct = 0
    
#     for data, target in test_loader:
#         if not args.cpu:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         output = model(data)
#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#     acc = 100. * float(correct) / len(test_loader.dataset)
#     print('Accuracy: {:.2f}%\n'.format(acc))
#     return
# print('************预剪枝模型测试************')
# if not args.cpu:
#     model.cuda()
# test()
#********************************剪枝*********************************
# cfg to   in/out_chanles 构造剪枝后的模型 (无参数)
newmodel = irmodel()

inputchanel = cfg.copy()
inputchanel.insert(0,newmodel.in_channels[0])
# inputchanel.append(newmodel.in_channels[-1])

outputchanel = cfg.copy()
outputchanel.append(newmodel.out_channels[-1])

newmodel = irmodel(in_channels=inputchanel,out_channels=outputchanel)
# print(newmodel)
# 加载参数
# if not args.cpu:
#     newmodel.cuda()
layer_id_in_cfg = 0
start_mask = torch.ones(args.imgtype)   #### 灰度图像还是rgb图像
end_mask = cfg_mask[layer_id_in_cfg]
i = 0
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    # print('******************')
    # print(m0)
    # print('++++++++++++++++++')
    # print(m1)
    # print('------------------')
    if isinstance(m0, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1

            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Conv2d):
        if i < layers - 1:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy()))) ## startmask 是当前层的mask end是当前层输出的mask
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
       
            w = m0.weight.data[:, idx0, :, :].clone() #前一层所连接

            m1.weight.data = w[idx1, :, :, :].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
    
        else:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if i ==layers - 1:
                m1.weight.data = m0.weight.data[:, idx0, :, :].clone() # 以前第二位是idx0
                i = i+1
            else:
                m1.weight.data = m0.weight.data[:, :, :, :].clone() # 以前第二位是idx0


            if hasattr(m1.bias,'data'):
                m1.bias.data = m0.bias.data.clone()
    elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
    #elif isinstance(m0,nn.route)
#******************************剪枝后model测试*********************************
# print('新模型: ', newmodel)
print('**********剪枝后新模型测试*********')
##############################################################################finetune

if args.futine:
    # train()
    pass



################################################################################
#******************************剪枝后model保存*********************************
newmodel.prun_flag = 1

print('**********剪枝后新模型保存*********')
torch.save(newmodel.state_dict(),"./prun_model/weights.pth")
torch.save(newmodel,"./prun_model/model.pth")
# torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)   ./model/weights.pth    ./model/model.pth
print('**********保存成功*********\r\n')

#*****************************剪枝前后model对比********************************
print('************旧模型结构************')
print(cfg_0)
print('************新模型结构************')
print(cfg, '\r\n')