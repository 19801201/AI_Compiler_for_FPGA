# -*- coding: UTF-8 -*-
#encoding=utf-8
import os
import argparse
from torch.utils.data import DataLoader
import warnings
import numpy as np
from PIL import Image
import torch
from relayIR.yolov4_tiny_quan import YoloBody
import glob
warnings.filterwarnings("ignore")


def train():
    device='cpu'
    model_float_path='1.7-LEAKY_RELU_01_Epoch14-Total_Loss0.5940-Val_Loss0.9286_714.pth'
    model = YoloBody(3,1).eval()
    state_dict = torch.load(model_float_path, map_location='cpu')
    model.load_state_dict(state_dict)
    # print(model)
    # exit()
    # 改一下保存模型格式
    # model_float_path='logs/model.pth'
    # model = torch.load(model_float_path, map_location='cpu')
    # load=torch.load('logs/model.pth',map_location='cpu')
    # model.load_state_dict(load)

    model.eval().fuse_model()
    ENGINE = 'fbgemm'
    torch.backends.quantized.engine = ENGINE
    model.qconfig = torch.quantization.get_default_qat_qconfig(ENGINE)
    torch.quantization.prepare(model, inplace=True)
    image_path = 'VOCdevkit/VOC2007/JPEGImages'
   
    for epoch in range(0, 1):
        
        for iter,img_path in enumerate(glob.glob(os.path.join(image_path, '*.bmp'))):
            model.train()
            image = Image.open(img_path)
            image_shape = np.array(np.shape(image)[0:2])
            crop_img = image.convert('L')
            crop_img = crop_img.resize((416, 416), Image.BICUBIC)
            photo = np.array(crop_img, dtype=np.float32) / 255.0
            # photo = np.transpose(photo, (2, 0, 1))
            photo = np.expand_dims(photo, axis=0)

            images = [photo]

            with torch.no_grad():
                images = torch.from_numpy(np.asarray(images))
                outputs = model(images)
            print(iter)

            if iter==100:
                break


        torch.quantization.convert(model, inplace=True)
      
        # label_path = './images/labels'
        # image_path = './images/images'
        # if epoch >= 0:
        #     total_ship, detected_ship, false_alarm, precision, recall, f1 = evaluate_f1(model, label_path, image_path, opt.image_size, thr=0.3)
        #     logfile.writelines(str(epoch+1) + '\t' + 'total_ship: ' + str(total_ship) + '\t' + 'detected_ship: ' + str(detected_ship)
        #                            + '\t' + 'false_alarm: ' + str(false_alarm) + '\t' + 'precision: ' + str(precision)+ '\t' + 'recall: ' + str(recall) + '\t' + 'f1: ' + str(f1) + '\n')
        #     logfile.flush()
     
        torch.jit.save(torch.jit.script(model.eval()), "quan_pth/Epoch{}-YOLOV4_quantization_post_jit.pth".format(epoch+1))
        torch.save(model.state_dict(), "quan_pth/Epoch{}-YOLOV4_quantization_post_save.pth".format(epoch+1))
        print("quantized=================================================")
        print(model)
        # for k,v in model.state_dict().items():
        #     # print(k)
        #     # if 'scale' in k:
        #     #     print(v)

        #     # if not('NoneType' in str(type(v))):
        #         if 'weight' in k:
        #             print(k)
        #             np.save('./para800_1028/'+k+'.scale',v.q_per_channel_scales())
        #             np.save('./para800_1028/'+k+'.zero_point',v.q_per_channel_zero_points())
        #             np.save('./para800_1028/'+k+'.int',v.int_repr())
        #             np.save('./para800_1028/'+k,v.dequantize().numpy())
        #         elif 'bias' in k:
        #             np.save('./para800_1028/'+k,v.detach().numpy())
        #         elif 'zero_point' in k:
        #             np.save('./para800_1028/'+k,v.detach().numpy())
        #         elif 'scale' in k:
        #             np.save('./para800_1028/'+k,v.detach().numpy())
        # logfile.close()
        # break


if __name__ == "__main__":
    train()
