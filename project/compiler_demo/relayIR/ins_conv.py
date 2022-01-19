import torch
from torch.nn.quantized import functional as qF
import numpy as np
def reg6_leaky(s3):
    add_data = []
    data1 = torch.ones(16)
    for index in range(data1.shape[0]):
        data1[index] = -index * 10 - 5
    for index in range(data1.shape[0]):
        q_feature = torch.quantize_per_tensor(data1[index] * s3, scale=float(s3),
                                              zero_point=int(0), dtype=torch.qint8)
        out_leak = qF.leaky_relu(input=q_feature, negative_slope=0.1, inplace=True)
        out = out_leak.int_repr()
        out2 = data1[index] * 0.1

        if np.round(out) > np.round(out2):
            add_data.append(1)
        elif np.round(out) < np.round(out2):
            add_data.append(-1)
        else:
            add_data.append(0)
    out_str=""
    for index in range(data1.shape[0]):
        if add_data[index]==0:
            out_str += '00'
        elif add_data[index]==1:
            out_str += '01'
        elif add_data[index]==-1:
            out_str += '10'
    # a = int(out_str, 2)
    # print(out_str)
    # out_reg6 = '{:08x}'.format(int(a))
    return out_str


def conv33para(m, c, dataSizeW, dataSizeB):
    # m是输出通道数,c是输入通道数,dataSizeW是权重每行多少bit,dataSizeB是bias每行多少bit
    # reg4：高16:1个卷积点的所有通道数的行数,在8:bias行数,剩下都是0

    one_conv_line = int((m * c * 9) / ((dataSizeW / 8) * 9))

    one_conv_line = '{:04x}'.format(one_conv_line)
    # one_conv_line是1个卷积点的所有通道数的行数
    bias_line = int(m / (dataSizeB / 32))
    bias_line = '{:02x}'.format(bias_line)
    all_zero = '00'
    reg4_para = str(one_conv_line) + str(bias_line) + all_zero
    # print(reg4[4])
    out_reg4_para = str('')
    for index in range(len(reg4_para)):
        out_reg4_para += reg4_para[index]

    return out_reg4_para


def conv33compute(m, c, dataSizeW, dataSizeB, inPictureSize, stride=1, padding=0, z1=0, z3=0,s3=0):

    # m是输出通道数,c是输入通道数,dataSizeW是权重每行多少bit,dataSizeB是bias每行多少bit
    # ================================reg4==============================================
    # reg4高十位是输入通道数,接着11位是不带stride(stride=1)卷积后图片宽高,下一位是0,最后十位是输出通道数,stride默认为1
    # channel_in是输入通道数,channel_out是输出通道数
    # outPictureSize_stride1是默认stride=1的图片的宽高,当作11位的
    channel_in = int(c)
    channel_in = '{:010b}'.format(channel_in)
    outPictureSize_stride1 = int((inPictureSize - 3 + 2 * padding) / 1 + 1)
    outPictureSize_stride1 = '{:011b}'.format(outPictureSize_stride1)
    zero = '0'
    channel_out = int(m)
    channel_out = '{:010b}'.format(channel_out)
    # one_conv_line是1个卷积点的所有通道数的行数

    reg4_compute = str(channel_in) + str(outPictureSize_stride1) + zero + str(channel_out)
    # print(reg4[4])
    out_reg4_compute = str('')
    for index in range(len(reg4_compute)):
        out_reg4_compute += reg4_compute[index]
        # if (index + 1) % 4 == 0 and (index + 1) != len(reg4_compute):
        #     out_reg4_compute += '_'
    # return str(int(out_reg4_compute, 2))

    # ================================reg5==============================================
    # compute的reg5
    # 第一位: 是否需要通道补0 (本工程不用，当输入维度是RGB三通道，则需要通道补0)yoloheadconv11在软件补0
    # 第二位:1位  是否需要 padding 的信号
    # is_padding:是否padding,如果padding时候等于1,不padding则等于0
    # 第三位:是否需要  stride  的信号,如果stride不等于1则is_stride是1,如果stride=1,则is_stride是0
    # 第四-六位:3 位   padding 添零的圈数  (针对 5*5 卷积而设计的) ，本工程为1
    # 再往后11位图片输入的宽高数
    # 最后14位补0
    is_padding = 0
    is_stride = 0
    if padding != 0:
        is_padding = 1
    elif padding == 0:
        is_padding = 0
    if stride != 1:
        is_stride = 1
    elif stride == 1:
        is_stride = 0
    inPictureSize = int(inPictureSize)
    inPictureSize = '{:011b}'.format(inPictureSize)
    reg5_compute = '0' + str(is_padding) + str(is_stride) + '001' + '000000000000000' + str(inPictureSize)
    # print(reg4[4])
    out_reg5_compute = str('')
    for index in range(len(reg5_compute)):
        # print(i)
        # exit()
        # print(reg4)
        out_reg5_compute += reg5_compute[index]

    # return str(int(out_reg5_compute, 2)),str(int(out_reg5_compute, 2))

    # ================================reg6==============================================
    # 前16位全部为0,在8位是bias在coe中行数,最后的8位补0(改为leakrelu的reg)
    out_reg6_compute=reg6_leaky(s3)

    # bias_line = int(m / (dataSizeB / 32))
    # bias_line = '{:08b}'.format(bias_line)
    # reg6_compute = '0000000000000000' + str(bias_line) + '00000000'
    # out_reg6_compute = str('')
    # for index in range(len(reg6_compute)):
    #     # print(i)
    #     # exit()
    #     # print(reg4)
    #     out_reg6_compute += reg6_compute[index]
        # if (index + 1) % 4 == 0 and (index + 1) != len(reg6_compute):
        #     out_reg6_compute += '_'

    # ================================reg7==============================================
    # 前8位Padding中填0的值(若Z1不为0，则填0的值就是Z1)
    # 在8位z3的值
    # 最后16位，3 * 3卷积中1个卷积点的所有通道数的行数
    if padding == 0:
        z1 = 0
    z1 = int(z1)
    z1 = '{:08b}'.format(z1)
    z3 = int(z3)
    z3 = '{:08b}'.format(z3)
    one_conv_line = int((m * c * 9) / ((dataSizeW / 8) * 9))
    one_conv_line = '{:016b}'.format(one_conv_line)
    reg7_compute = str(z1) + str(z3) + str(one_conv_line)
    out_reg7_compute = str('')
    for index in range(len(reg7_compute)):
        out_reg7_compute += reg7_compute[index]
        # if (index + 1) % 4 == 0 and (index + 1) != len(reg7_compute):
        #     out_reg7_compute += '_'
    return str(int(out_reg4_compute, 2)), str(int(out_reg5_compute, 2)), str(int(out_reg6_compute, 2)), str(
        int(out_reg7_compute, 2))
def conv11para(m, c, dataSizeW, dataSizeB):
    # m是输出通道数,c是输入通道数,dataSizeW是权重每行多少bit,dataSizeB是bias每行多少bit
    # reg4：高16:一个卷积点的所有通道数的行数,在9:bias在coe中的行数
    one_conv_line = int((m * c) / (dataSizeW / 8))
    one_conv_line = '{:016b}'.format(one_conv_line)
    bias_line = int(m / (dataSizeB / 32))
    if (bias_line <= 255):
        bias_line = '{:08b}'.format(bias_line)
        # print(bias_line)
        # exit()
        out_reg4_para = str(one_conv_line) + str(bias_line) + '00000000'
    else:
        bias_line = '{:09b}'.format(bias_line)
        bias_line = bias_line[1:] + bias_line[0]
        out_reg4_para = str(one_conv_line) + str(bias_line) + '0000000'


    return str(int(out_reg4_para, 2))


def conv11compute(m, c, dataSizeW, dataSizeB, inPictureSize, stride=1, padding=0, z1=0, z3=0, isleakrelu=0,s3=0):
    # m是输出通道数,c是输入通道数,dataSizeW是权重每行多少bit,dataSizeB是bias每行多少bit
    # ================================reg4==============================================
    # reg4高十位是输入通道数,接着11位是卷积后图片宽高,下一位是0,最后十位是输出通道数
    # channel_in是输入通道数,channel_out是输出通道数
    # outPictureSize是的图片的宽高,当作11位的
    channel_in = int(c)

    channel_in = '{:010b}'.format(channel_in)
    outPictureSize = int((inPictureSize - 1 + 2 * padding) / stride + 1)
    # print(outPictureSize)
    # exit()
    outPictureSize = '{:011b}'.format(outPictureSize)
    # print(outPictureSize)
    zero = '0'
    channel_out = int(m)
    channel_out = '{:010b}'.format(channel_out)
    # print(m)
    # print(channel_out)
    # one_conv_line是1个卷积点的所有通道数的行数
    # print(m)
    # exit()
    out_reg4_compute = str(channel_in) + str(outPictureSize) + zero + str(channel_out)
    # print(out_reg4_compute)
    # print('000100000000001010000000010000000')
    # exit()
    # return str(int(out_reg4_compute, 2))

    # ================================reg5==============================================
    # compute的reg5
    # 第一位: 是否需要通道补0 (本工程不用，当输入维度是RGB三通道，则需要通道补0)yoloheadconv11在软件补0
    # 第二位:1位  是否需要 padding 的信号
    # is_padding:是否padding,如果padding时候等于1,不padding则等于0
    # 第三位:是否需要  stride  的信号,如果stride不等于1则is_stride是1,如果stride=1,则is_stride是0
    # 第四-六位:3 位   padding 添零的圈数  (针对 5*5 卷积而设计的) ，本工程为1
    # 第七位:1位   是否需要 leakyrelu 。 1 为不用，0 为用
    # 在往后14位补0
    # 最后11位图片输入的宽高数
    # print(isleakrelu)
    # print('---***//')
    is_padding = 0
    is_stride = 0
    if padding != 0:
        is_padding = 1
    elif padding == 0:
        is_padding = 0
    if stride != 1:
        is_stride = 1
    elif stride == 1:
        is_stride = 0


    inPictureSize = int(inPictureSize)
    inPictureSize = '{:011b}'.format(inPictureSize)
    out_reg5_compute = '0' + str(is_padding) + str(is_stride) + '001' + str(isleakrelu) + '00000000000000' + str(
        inPictureSize)
    # print(is_padding)
    # print(out_reg5_compute)
    # print('******************')
    # print(reg4[4])
    # out_reg5_compute = str('')

    # return str(int(out_reg4_compute, 2)), str(int(out_reg5_compute, 2))

    # ================================reg6==============================================
    # reg_leakrelu
    out_reg6_compute=reg6_leaky(s3)
    # bias_line = int(m / (dataSizeB / 32))
    # if (bias_line <= 255):
    #     bias_line = '{:08b}'.format(bias_line)
    #     print(bias_line)
    #     # exit()
    #     out_reg6_compute = '0000000000000000' + str(bias_line) + '00000000'
    # else:
    #     bias_line = '{:09b}'.format(bias_line)
    #     bias_line = bias_line[1:] + bias_line[0]
    #     out_reg6_compute = '0000000000000000' + str(bias_line) + '0000000'

    # return str(int(out_reg4_compute, 2)), str(int(out_reg5_compute, 2)), str(int(out_reg6_compute, 2))

    # ================================reg7==============================================
    # 前8位为
    # 在8位z3的值
    # 最后16位， 为0

    z3 = int(z3)
    z3 = '{:08b}'.format(z3)
    all_zero = '0000000000000000'
    # one_conv_line = int((m * c) / (dataSizeW / 8))
    # one_conv_line = '{:016b}'.format(one_conv_line)

    out_reg7_compute = '00000000' + str(z3) + all_zero

    # print(out_reg5_compute)
    return str(int(out_reg4_compute, 2)), str(int(out_reg5_compute, 2)), str(int(out_reg6_compute, 2)), str(
        int(out_reg7_compute, 2))
