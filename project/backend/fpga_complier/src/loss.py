import math
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        # pred_prob = torch.sigmoid(pred)  # prob from logits
        pred_prob = pred  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class YoloLoss(nn.modules.loss._Loss):
    def __init__(self, anchors, reduction=32, coord_scale=1.0, noobject_scale=1.0,
                 object_scale=5.0, thresh=0.7, IOU=None, version='A'):
        super(YoloLoss, self).__init__()
        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.reduction = reduction

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.thresh = thresh
        self.IOU = IOU
        self.version = version
        print('version: ', self.version)

        self.mse = nn.MSELoss(size_average=False)
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.l1_loss = nn.L1Loss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        # self.bce_loss = FocalLoss(nn.BCELoss(reduction='sum'))
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, output, target):

        batch_size = output.data.size(0)
        height = output.data.size(2)
        width = output.data.size(3)

        # Get x,y,w,h,conf,cls
        output = output.view(batch_size, self.num_anchors, -1, height * width)
        coord = torch.zeros_like(output[:, :, :4, :])
        coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()
        coord[:, :, 2:4, :] = output[:, :, 2:4, :]
        conf = output[:, :, 4, :].sigmoid()

        # Create prediction boxes
        pred_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4)
        lin_x = torch.range(0, width - 1).repeat(height, 1).view(height * width)
        lin_y = torch.range(0, height - 1).repeat(width, 1).t().contiguous().view(height * width)
        anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

        if torch.cuda.is_available():
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        # pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
        # pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
        # pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
        # pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
        pred_boxes[:, 0] = (coord[:, :, 0] + lin_x).view(-1)
        pred_boxes[:, 1] = (coord[:, :, 1] + lin_y).view(-1)
        pred_boxes[:, 2] = (coord[:, :, 2].exp() * anchor_w).view(-1)
        pred_boxes[:, 3] = (coord[:, :, 3].exp() * anchor_h).view(-1)
        # pred_boxes = pred_boxes.cpu()

        # Get target values
        coord_mask, conf_mask, tcoord, tconf = self.build_targets(pred_boxes, target, height, width)

        if torch.cuda.is_available():
            tcoord = tcoord.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()

        conf_mask = conf_mask.sqrt()
        coord_mask, conf_mask, tcoord = coord_mask.detach(), conf_mask.detach(), tcoord.detach()

        # Compute losses
        self.loss_iou = 0
        self.loss_coord = 0

        if self.version == 'A':
            self.loss_iou = ((1 - tconf[torch.where(coord_mask[:, :, 0, :] > 0)])).sum() / batch_size
            tconf_d = tconf.detach().clone()
            self.loss_conf = (self.bce_loss(conf, tconf_d) * conf_mask).sum() / batch_size

        elif self.version == 'B':
            self.loss_coord = self.coord_scale * self.mse(coord * coord_mask, tcoord * coord_mask) / batch_size
            self.loss_iou = ((1 - tconf[torch.where(coord_mask[:, :, 0, :] > 0)])).sum() / batch_size
            tconf_d = tconf.detach().clone()
            self.loss_conf = self.mse(conf * conf_mask, tconf_d * conf_mask) / batch_size

        elif self.version == 'C':
            tconf_d = tconf.detach().clone()
            self.loss_coord = self.coord_scale * self.mse(coord * coord_mask, tcoord * coord_mask) / batch_size
            self.loss_conf = self.mse(conf * conf_mask, tconf_d * conf_mask) / batch_size

        else:
            raise ValueError("Wrong loss version")

        # self.loss_coord = self.coord_scale * self.mse(coord * coord_mask, tcoord * coord_mask) / batch_size
        # self.loss_conf = self.mse(conf * conf_mask, tconf_d * conf_mask) / batch_size
        # self.loss_conf = self.bce_loss(conf, tconf_d) / batch_size

        self.loss_tot = self.loss_coord + self.loss_conf + self.loss_iou

        return self.loss_tot, self.loss_coord, self.loss_conf, self.loss_iou

    def build_targets(self, pred_boxes, ground_truth, height, width):
        batch_size = len(ground_truth)

        conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False) * self.noobject_scale
        coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False)
        tcoord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False)
        if torch.cuda.is_available():
            tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).cuda()
        else:
            tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        for b in range(batch_size):
            if len(ground_truth[b]) == 0:
                continue

            # Build up tensors
            cur_pred_boxes = pred_boxes[
                             b * (self.num_anchors * height * width):(b + 1) * (self.num_anchors * height * width)]

            if self.anchor_step == 4:
                anchors = self.anchors
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
            if torch.cuda.is_available():
                gt = torch.zeros(len(ground_truth[b]), 4).cuda()
            else:
            	gt = torch.zeros(len(ground_truth[b]), 4)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction
                gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction
                gt[i, 2] = anno[2] / self.reduction
                gt[i, 3] = anno[3] / self.reduction

            # Set confidence mask of matching detections to 0
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes, IOU=self.IOU)

            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0
            # Find best anchor for each ground truth
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            if torch.cuda.is_available():
                iou_gt_anchors = bbox_iou(gt_wh, anchors.cuda())
            else:
            	iou_gt_anchors = bbox_iou(gt_wh, anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each ground truth
            for i, anno in enumerate(ground_truth[b]):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]
                coord_mask[b][best_n][0][gj * width + gi] = 1
                conf_mask[b][best_n][gj * width + gi] = self.object_scale
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0] + 1e-12)
                tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1] + 1e-12)
                tconf[b][best_n][gj * width + gi] = iou

        return coord_mask, conf_mask, tcoord, tconf


def bbox_iou(boxes1, boxes2):
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / (unions + 1e-12)

def bbox_ious(gt_box1, pr_box2, x1y1x2y2=False, IOU=None):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # pr_box2 = pr_box2.t()
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = gt_box1[:, 0].split(1, 1), gt_box1[:, 1].split(1, 1), gt_box1[:, 2].split(1,1), gt_box1[:,3].split(1, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = pr_box2[:, 0].split(1, 1), pr_box2[:, 1].split(1, 1), pr_box2[:, 2].split(1,1), pr_box2[:,3].split(1, 1)
    else:  # transform from xywh to xyxy
        b1_x1, b1_y1 = (gt_box1[:, :2] - (gt_box1[:, 2:4] / 2)).split(1, 1)
        b1_x2, b1_y2 = (gt_box1[:, :2] + (gt_box1[:, 2:4] / 2)).split(1, 1)
        b2_x1, b2_y1 = (pr_box2[:, :2] - (pr_box2[:, 2:4] / 2)).split(1, 1)
        b2_x2, b2_y2 = (pr_box2[:, :2] + (pr_box2[:, 2:4] / 2)).split(1, 1)
    b2_x1, b2_y1, b2_x2, b2_y2 = b2_x1.t(), b2_y1.t(), b2_x2.t(), b2_y2.t()
    # inter area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if IOU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if IOU == 'GIoU':  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            giou = iou - (c_area - union) / c_area  # GIoU
            return giou
        if IOU == 'DIoU' or IOU == 'CIoU':  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if IOU == 'DIoU':
                diou = iou - rho2 / c2  # DIoU
                return diou
            elif IOU == 'CIoU':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                ciou = iou - (rho2 / c2 + v * alpha)  # CIoU
                return ciou
        else:
            raise ValueError('Srong IOU setting')

    return iou