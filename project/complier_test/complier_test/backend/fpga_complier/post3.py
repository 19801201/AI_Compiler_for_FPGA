# -*- coding: UTF-8 -*-
#encoding=utf-8
import os
import argparse
from torch.utils.data import DataLoader
from src.ship_dataset import SHIPDataset
from src.utils import *
import warnings
from src.evaluate_f1_qun import evaluate_f1
import numpy as np
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser("Mobilenetv2+yolov2loss")
    parser.add_argument("--image_size", type=int, default=416, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--num_epoches", type=int, default=1)
    parser.add_argument("--test_interval", type=int, default=0, help="Number of epoches between testing phases")
    parser.add_argument("--object_scale", type=float, default=5.0)
    parser.add_argument("--noobject_scale", type=float, default=1.0)
    parser.add_argument("--coord_scale", type=float, default=1.0)
    parser.add_argument("--reduction", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="./data/", help="the root folder of dataset")
    parser.add_argument("--iweight", type=str, default="./prun_model/weights.pth")
    parser.add_argument("--imodel", type=str, default="./prun_model/model.pth")
    parser.add_argument("--prun", type=bool, default="0")

    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--test", type=bool, default=True)
    args = parser.parse_args()
    return args

start_epoch = 0
def train(opt):
    device='cpu'  
    learning_rate_schedule = {"0": 1e-5, "5": 1e-4, "20": 1e-3, "45": 1e-4, "65": 1e-5, "80": 1e-6}
    training_para1ms = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

    training_set = SHIPDataset(opt.data_path,  opt.image_size)
    training_generator = DataLoader(training_set, **training_para1ms)

    coco_anchors = [(17, 12), (18, 37), (48, 24), (68, 69), (199,173)]
    coco_anchors = np.array(coco_anchors)
    coco_anchors = coco_anchors.tolist()
    if opt.prun:
        model = torch.load(opt.imodel, map_location='cpu')
        model.load_state_dict(torch.load(opt.iweight,map_location='cpu'))
    else:

        model = torch.load('./prun_model/model.pth', map_location='cpu')
        model.load_state_dict(torch.load('./prun_model/weights.pth',map_location='cpu'))        
    model.eval().fuse_model()
 
      
    if opt.test == True:
        logfile = open('logs/logs.txt', 'w')


    ENGINE = 'fbgemm'
    torch.backends.quantized.engine = ENGINE
    model.qconfig = torch.quantization.get_default_qat_qconfig(ENGINE)
    torch.quantization.prepare(model, inplace=True)


   
    for epoch in np.arange(start_epoch, opt.num_epoches):


        for iter, batch in enumerate(training_generator):
   

            image, label = batch
   
           
            image = image/255

            image = Variable(image.to(device))
            #image = Variable(image, requires_grad=True)
          
            logits = model(image)
            print(iter)
            if iter==50:
                break


    torch.quantization.convert(model, inplace=True)
    if opt.test == True:
        label_path = './images/labels'
        image_path = './images/images'
        if epoch >= 0:
            total_ship, detected_ship, false_alarm, precision, recall, f1 = evaluate_f1(model, label_path, image_path, opt.image_size, thr=0.3)
            logfile.writelines(str(epoch+1) + '\t' + 'total_ship: ' + str(total_ship) + '\t' + 'detected_ship: ' + str(detected_ship)
                               + '\t' + 'false_alarm: ' + str(false_alarm) + '\t' + 'precision: ' + str(precision)+ '\t' + 'recall: ' + str(recall) + '\t' + 'f1: ' + str(f1) + '\n')
            logfile.flush()
 
    torch.jit.save(torch.jit.script(model.eval()),"./YOLO_quantization_post.pth")

  
    print("quantized=================================================")
    print(model)
   
    for k,v in model.state_dict().items():
        # print(k)
        # print(v)

        if not('NoneType' in str(type(v))):
            if 'weight' in k:           
                np.save('./q_paras/'+k+'.scale',v.q_per_channel_scales())
                np.save('./q_paras/'+k+'.zero_point',v.q_per_channel_zero_points())
                np.save('./q_paras/'+k+'.int',v.int_repr())
                np.save('./q_paras/'+k,v.dequantize().numpy()) 
            elif 'bias' in k:
                np.save('./q_paras/'+k,v.detach().numpy()) 
            elif 'zero_point' in k:
                np.save('./q_paras/'+k,v.detach().numpy()) 
            elif 'scale' in k:
                np.save('./q_paras/'+k,v.detach().numpy()) 
    if opt.test == True:
    
        logfile.close()
    


if __name__ == "__main__":
    opt = get_args()
    train(opt)
