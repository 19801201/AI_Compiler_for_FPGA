import os
import glob
import cv2
from src.utils import *
import numpy as np

def bboxes_iou(bbox_a, bbox_b):
    """
    计算矩形框之间的重叠率，bbox_a 与 bbox_b 格式均为 [x1, y1, x2, y2]
    """
    if bbox_a.shape[0] != 4 or bbox_b.shape[0] != 4:
        return 0  # 如果box格式有问题则重叠率判为0

    tl = np.maximum(bbox_a[:2], bbox_b[:2])
    br = np.minimum(bbox_a[2:], bbox_b[2:])
    area_a = np.prod(bbox_a[2:] - bbox_a[:2])
    area_b = np.prod(bbox_b[2:] - bbox_b[:2])
    en = np.prod(tl < br)
    area_i = np.prod(br - tl) * en
    return area_i / (area_a + area_b - area_i)

def evaluate_f1(model, label_path, image_path, image_size,thr=0.4,nms_threshold=0.5):
    model.eval()

    conf_threshold = thr
    device = 'cpu'
    nms_threshold = nms_threshold
    total_ship = 0   #计算所有测试图片中舰船总量
    total_result = 0 #计算所有图片检测到的舰船
    true_result = 0  #计算真确检测到的舰船数量
    for kk, image_path in enumerate(glob.iglob(os.path.join(image_path, '*.jpg'))):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image/256
        height, width = image.shape[:2]
        image = cv2.resize(image, (image_size, image_size))
        image = image[None, None, :, :]
        width_ratio = float(image_size) / width
        height_ratio = float(image_size) / height
    
        data = Variable(torch.FloatTensor(image), requires_grad=False)      
   
        with torch.no_grad():
           
            logits = model(data)

            predictions = post_processing(logits, image_size, model.anchors, conf_threshold, nms_threshold)

        image_name = image_path.split(os.sep)[-1].strip()[0:-4]
        labels = np.loadtxt(os.path.join(label_path,image_name+'.txt'))
    
        if len(labels.shape) == 1:
            labels = labels[None, :]
        total_ship = total_ship + len(labels)



        if len(predictions) != 0:
            predictions = np.array(predictions[0])
            results = predictions.copy()
            results[:, 0] = np.maximum(predictions[:, 0] / width_ratio, 0)
            results[:, 1] = np.maximum(predictions[:, 1] / height_ratio, 0)
            results[:, 2] = np.minimum((predictions[:, 0] + predictions[:, 2])/ width_ratio, width)
            results[:, 3] = np.minimum((predictions[:, 1] + predictions[:, 3]) / height_ratio, height)
            results = results[:, 0:4]
            # print(results)


            total_result = total_result + len(results)
            res_flag = np.zeros(len(results)).astype(int)
            # print(len(results))
            # print(results.shape)
            # exit()
            for i, label in enumerate(labels):
                iou_list = []
                for j, result in enumerate(results):
                    iou_list.append(bboxes_iou(label, result))
                iou_list = np.array(iou_list)
                # 取没有被匹配过的，与当前标注有最大IOU的结果
                iou_ind = (res_flag==0)       #取出没被分配的检测结果
                iou_list[~iou_ind] = 0        #被分配过的置零
                sort_indx = np.argsort(iou_list)
                sort_indx = sort_indx[::-1]
                if iou_list[sort_indx[0]] >= 0.5:
                    true_result = true_result + 1
                    res_flag[sort_indx[0]] = 1
        else:
            a = 1
    false_alarm = total_result - true_result
    precision = true_result/(true_result+false_alarm+0.00000000001)
    recall = true_result/total_ship
    f1 = 2*precision*recall/(precision+recall+0.00000000001)
    return total_ship, true_result, false_alarm, precision, recall, f1
