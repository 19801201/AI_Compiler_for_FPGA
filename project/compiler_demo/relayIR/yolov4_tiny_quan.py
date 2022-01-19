import math
import torch
import torch.nn as nn
from torch.quantization import QuantStub,DeQuantStub


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.1),
        )
        self.activation = nn.LeakyReLU(0.125)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

#--------------Resblock_body-----------------------------
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels, out_channels, 3)
        self.conv3 = BasicConv(out_channels, out_channels, 3)

        self.conv4 = BasicConv(out_channels * 2, out_channels * 2, 1)
        self.conv5 = BasicConv(out_channels, out_channels, 1)

        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

        self.routeq0 = nn.quantized.FloatFunctional()
        self.routeq1 = nn.quantized.FloatFunctional()

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)

        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)

        feat = x
        feat = self.conv5(feat)

        # 主干部分与残差部分进行相接
        x = self.routeq0.cat([x, route1], dim=1)

        # 对相接后的结果进行1x1卷积256 256 64 64 40 1 1 63 57
        x = self.conv4(x)

        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x ):
        x = self.upsample(x)
        return x


# ---------------------------------------------------#
#   最后获得yolov4的输出
# ---------------------------------------------------#
class yolo_head(nn.Module):
    def __init__(self, filters_list, in_filters):
        super(yolo_head, self).__init__()
        self.yolo_head = nn.Sequential(
            BasicConv(in_filters, filters_list[0], 3),
            nn.Conv2d(filters_list[0], filters_list[1], 1),
        )
    def forward(self, x):
        x = self.yolo_head(x)
        return x


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        # self.backbone = darknet53_tiny(None)

        self.conv1 = BasicConv(1, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = Resblock_body(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = Resblock_body(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = Resblock_body(256, 256)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)

        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.conv_for_P5 = BasicConv(512, 256, 1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)], 256)

        self.upsample = Upsample(256, 128)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 384)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.route0 = nn.quantized.FloatFunctional()


    def forward(self, x):
        # ---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        # ---------------------------------------------------#
        # feat1, feat2 = self.backbone(x)

        ##############  backbone start  #########################
        x= self.quant(x)
        # 416x416x1 -> 208x208x32
        x = self.conv1(x)
        # 208x208x32 -> 104x104x64
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)

        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        ##############  backbone end  #########################

        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # # 13,13,256 -> 13,13,512 -> 13,13,255
        # out0 = self.yolo_headP5(P5)

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        P4 = self.route0.cat([P5_Upsample, feat1], 1)

        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out0 = self.yolo_headP5(P5)

        out0 = self.dequant(out0)
        out1 = self.dequant(out1)
        return out0, out1

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m,BasicConv):
                torch.quantization.fuse_modules(m.conv, [['0', '1']],inplace=True)


# model = YoloBody(3, 1)
# print(model.conv1.conv)
# print(model.yolo_headP4.conv2d)
# print(x)
# exit()

#print(YoloBody(3, 20))

