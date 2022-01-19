# -*- coding: UTF-8 -*-
# encoding=utf-8
import torch
import torch.nn as nn
import numpy as np
from picture_load import *
from ins_conv import *
from get_mod import *
import time
from txt2json import *
import math


def gen_B(S1, S2, S3):
    M = (S1 * S2) / S3
    M = M.numpy()

    daxiao = S2.shape[0]
    SCALE = np.zeros(daxiao, dtype=np.uint32, order='C')
    N_REAL = np.zeros(daxiao, dtype=np.uint32, order='C')
    for i, ii in enumerate(M):

        while not (ii >= 0.5 and ii <= 1.0):
            ii *= 2
        pass
        mmmm = ii * (2 ** 32)

        SCALE[i] = mmmm.astype(np.int32)

    for i, ii in enumerate(M):
        N_REAL[i] = round(math.log(SCALE[i] / ii, 2)) - 32 - 1

    return N_REAL


def gen_M_N(S1, S2, S3):
    daxiao = S2.shape[0]
    M = np.zeros(daxiao, dtype=np.uint32, order='C')
    N_REAL = gen_B(S1, S2, S3)
    M = np.zeros(S2.shape[0])
    for i, ii in enumerate(M):
        M[i] = (torch.round((S1 * S2[i]) / S3 * (2 ** (32 + N_REAL[i] + 1)))).numpy()
    M = M.astype(np.uint32)
    return M, N_REAL


# r_b=s1*s2*q_b
# r_b是量化前的bias,q_b是量化后的bias
def gen_int_bias(s1, s2, bias_float):
    aa = bias_float / s1
    bb = torch.div(aa, s2)
    # for i, m in enumerate(bb):
    #     bb[i] = round(m.item())
    # bias = bb.int()
    return bb


def gen_M(s1, s2, s3):
    aa = s1 * s2
    M = aa / s3
    return M


# def new_bias(z1, q2, bias):
#     q2 = torch.as_tensor(q2, dtype=torch.int32)
#     bias1 = -z1 * q2
#     shape = bias1.shape
#     n_bias = np.zeros(shape[0], dtype=np.int32, order='C')
#     for m in range(shape[0]):
#         n_bias[m] = bias1[m, :, :, :].sum()
#         n_bias[m] = n_bias[m] + bias[m]
#     return n_bias
def new_bias(z1, q2, bias):
    q2 = q2.type(torch.float64)
    bias1 = z1 * q2
    shape = bias1.shape
    n_bias = np.zeros(shape[0], dtype=np.float64, order='C')
    for m in range(shape[0]):
        n_bias[m] = bias1[m, :, :, :].sum()
        # print()
        n_bias[m] = (bias[m] - n_bias[m])
    # print(n_bias)
    # exit()
    daxiao = shape[0]
    SCALE = np.zeros(daxiao, dtype=np.float64, order='C')
    # N_REAL = np.zeros(daxiao, dtype=np.float32, order='C')
    N_REAL = []
    for i, ii in enumerate(n_bias):
        index = 0

        while not (abs(ii) >= (2 ** 23) and abs(ii) <= (2 ** 24)):
            if index >= 16:  # fpga里面最多移动16位,所有成到16就停止了,这样精度也够了
                break
            else:
                ii *= 2
                index = index + 1
        N_REAL.append(index)
        SCALE[i] = round(ii)

    out_bias = []
    for index in range(shape[0]):
        data_integer_old = ('{:024b}'.format(int(SCALE[index]) & 0xffffff))
        n = N_REAL[index]
        symbol = '0'
        if n_bias[index] < 0:
            symbol = '1'
        elif n_bias[index] > 0:
            symbol = '0'
        data_integer = data_integer_old[8:]
        data_decimal = '{:07b}'.format(int(n))
        out_bias1 = symbol + str(data_decimal) + str(data_integer_old)
        a = int(out_bias1, 2)
        out_bias.append(a)
    return out_bias


def get_add_bias(new, shape, old):
    for kernel_num in range(shape):
        new[kernel_num] = old[kernel_num]
    return new


def get_add_SCALE(new, shape, old):
    for kernel_num in range(shape):
        new[kernel_num] = old[kernel_num]
    return new


def get_add_NREAL(new, shape, old):
    for kernel_num in range(shape):
        new[kernel_num] = old[kernel_num]
    return new


def get_weight(new_weight, shape, weight, inchannel,outchannel):
    j = 0
    shift_num_in = 0
    for index in range(inchannel):
        if (inchannel == (1 << index)):
            shift_num_in = index
            break
    shift_num_out = 0
    for index in range(outchannel):
        if (outchannel == (1 << index)):
            shift_num_out = index
            break

    for i in range(shape[2]):  # row
        for ii in range(shape[3]):  # col
            for kernel_times in range(shape[0] >> shift_num_out):  # kernel_num/8   /8代表8个通道
                for channel_in_times in range(shape[1] >> shift_num_in):  # channel_in_num/8
                    for iii in range(outchannel):
                        for iiii in range(inchannel):
                            # print('++++++++++++++++++')
                            weight[j] = new_weight[kernel_times * outchannel + iii][channel_in_times * inchannel + iiii][i][ii]
                            j += 1
    return weight


def add_weight_channel(new_weig, weig, shape):
    for kernel_num in range(shape[0]):
        for channel_in_num in range(shape[1]):
            for row in range(shape[2]):
                for col in range(shape[3]):
                    new_weig[kernel_num][channel_in_num][row][col] = weig[kernel_num][channel_in_num][row][col]
    return new_weig


def get_weight2(new_weight, shape, weight):
    j = 0
    for kernel_times in range(shape[0]):  # row
        for channel_in_times in range(shape[1]):  # col
            for i in range(shape[2]):  # kernel_num/8   /8代表8个通道
                for ii in range(shape[3]):  # channel_in_num/8
                    weight[j] = new_weight[kernel_times][channel_in_times][i][ii]
                    j += 1
    return weight


def Conv2d_Q(weight_int, s1, z1, s2, z2, s3, z3, bias, stride, padding, kernel_size):
    if kernel_size == 1:
        block = (weight_int.shape[0] * weight_int.shape[1]) / limit_size11
        block = math.ceil(block)  # 向上取整
        inchannel = inchannel11
        outchannel = outchannel11
    elif kernel_size == 3:
        block = (weight_int.shape[0] * weight_int.shape[1]) / limit_size33
        block = math.ceil(block)  # 向上取整
        inchannel = inchannel33
        outchannel = outchannel33

    bias= gen_int_bias(s1, s2, bias)
    bias = np.array(bias.data.cpu().numpy(), dtype=np.float64)

    SCALE, N_REAL = gen_M_N(s1, s2, s3)

    bias = new_bias(z1, weight_int, bias)
    # bias = bias.astype(np.uint32)
    q_weight = np.array(weight_int.data.cpu().numpy(), dtype=np.int8)
    shape = q_weight.shape
    if (shape[0] % 8 != 0):
        kernel_num = shape[0] + 8 - shape[0] % 8
    else:
        kernel_num = shape[0]
    channel_in_num = shape[1]
    new_weig = np.zeros((kernel_num, channel_in_num, shape[2], shape[3]))
    new_weight = add_weight_channel(new_weig, q_weight, shape)
    new_shape = new_weight.shape
    daxiao = new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3]
    weight = np.zeros(daxiao, dtype=np.uint8, order='C')
    if (shape[1] % 8 != 0):
        # 第一层先写入256bit的指令
        feature_size = input_shape[2]
        # 计算reg4
        # 输入图片宽高数
        feature_size_new = '{:011b}'.format(int(feature_size))
        # 卷积操作之后宽高数,stride等于1
        outPictureSize = int((feature_size - 3 + 2 * 1) / 1 + 1)
        outPictureSize = '{:011b}'.format(int(outPictureSize))
        # 图片的输出通道数
        out_channel = shape[0]
        out_channel = '{:08b}'.format(int(out_channel))
        out_reg4 = str(feature_size_new) + str(outPictureSize) + str('00') + str(out_channel)
        # 计算reg5
        out_reg5 = str('11001000000000000000000000000000')
        # 计算reg6
        out_reg6 = reg6_leaky(s3)
        # 计算reg7
        out_reg7 = '{:08b}'.format(int(z3))
        out_reg7 = str(out_reg7) + str('000000000000000000000000')
        # 指令前面加128个0凑够256bit
        weight_ins = '0'
        for index_add_zero in range(127):
            weight_ins += '0'
        weight_ins = weight_ins + str(out_reg7) + str(out_reg6) + str(out_reg5) + str(out_reg4)
        weight_ins = int(weight_ins, 2)
        weight_ins = '{:064x}'.format(int(weight_ins))
        out = []
        with open(weight_name, "a+") as fp:
            for r in range(len(weight_ins)):
                out.append(weight_ins[63 - r])
                if len(out) == 8:
                    out.reverse()
                    fp.write('0x')
                    for m in out:
                        # m = m.item()
                        fp.write(m)
                    fp.write('\n')
                    out = []

        a = 0
        new_image_weight = np.zeros(1024, dtype=np.uint8, order='C')
        get_weight2(new_weight, new_shape, weight)
        add_zero = np.zeros(23, dtype=np.uint8)
        shape_new = weight.shape[0] + 1
        # 第一层权重补0补到256bit,一行从72到256
        for index in range(1, shape_new):
            if index != 0 and index % 9 == 0:
                new_image_weight[int(index / 9 - 1) * 32:int(index / 9) * 32] = np.append(weight[index - 9:index],
                                                                                          add_zero)
        weight = new_image_weight
    elif block != 1:
        shape_block = []
        daxiao_new = []
        # 将weight分成4块  例如256 384 3 3 分成四个64 384 3 3 让每个分别进行kkmc生成coe
        for shape_num in range(block):
            # print(int(new_shape[0] / block * (1 + shape_num)))
            shape_block.append(int(new_shape[0] / block * (1 + shape_num)))
            # print(shape_block)
            block_shape = (shape_block[0], new_shape[1], new_shape[2], new_shape[3])
            daxiao_new.append(shape_block[shape_num] * new_shape[1] * new_shape[2] * new_shape[3])
            if shape_num == 0:
                get_weight(new_weight[:daxiao_new[shape_num], :, :, :], block_shape, weight[:daxiao_new[shape_num]],
                           inchannel,outchannel)
            else:
                get_weight(new_weight[shape_block[shape_num - 1]:shape_block[shape_num], :, :, :], block_shape,
                           weight[daxiao_new[shape_num - 1]:daxiao_new[shape_num]], inchannel,outchannel)
        for index in range(block):
            out = []

            with open(weight_name, "a+") as fp:
                for r in weight[index * daxiao_new[0]:daxiao_new[0] * (index + 1)]:
                    out.append(r)
                    if len(out) == 4:
                        out.reverse()
                        fp.write('0x')
                        for m in out:
                            m = m.item()

                            fp.write('%02x' % m)

                        fp.write('\n')
                        out = []
            with open(weight_name, "a+") as fp:
                for r in bias[index * shape_block[0]:shape_block[0] * (index + 1)]:

                    out.append(r)
                    fp.write('0x')
                    if len(out) == 1:
                        out.reverse()
                        for m in out:
                            fp.write('%08x' % int(m))

                        fp.write('\n')
                        out = []
            with open(weight_name, "a+") as fp:
                for r in SCALE[index * shape_block[0]:shape_block[0] * (index + 1)]:

                    out.append(r)
                    fp.write('0x')
                    if len(out) == 1:
                        out.reverse()
                        for m in out:
                            m = m.item()
                            fp.write('%08x' % m)

                        fp.write('\n')
                        out = []
            with open(weight_name, "a+") as fp:
                for r in N_REAL[index * shape_block[0]:shape_block[0] * (index + 1)]:

                    out.append(r)
                    fp.write('0x')
                    if len(out) == 1:
                        out.reverse()
                        for m in out:
                            m = m.item()
                            fp.write('%08x' % m)

                        fp.write('\n')
                        out = []


    else:
        get_weight(new_weight, new_shape, weight, inchannel,outchannel)
    if new_weight.shape[0] != shape[0]:
        new_dimen_bias = np.zeros(kernel_num, dtype=np.uint32)
        new_dimen_SCALE = np.zeros(kernel_num, dtype=np.uint32)
        new_dimen_NREAL = np.zeros(kernel_num, dtype=np.uint32)
        bias = get_add_bias(new_dimen_bias, shape[0], bias)
        SCALE = get_add_SCALE(new_dimen_SCALE, shape[0], SCALE)

        N_REAL = get_add_NREAL(new_dimen_NREAL, shape[0], N_REAL)

    if block == 1:
        out = []
        with open(weight_name, "a+") as fp:
            for r in weight:
                out.append(r)
                if len(out) == 4:
                    out.reverse()
                    fp.write('0x')
                    for m in out:
                        m = m.item()

                        fp.write('%02x' % m)
                    fp.write('\n')

                    out = []
        with open(weight_name, "a+") as fp:
            for r in bias:

                # for index in range(len(bias)):
                out.append(r)
                fp.write('0x')
                if len(out) == 1:

                    out.reverse()
                    for m in out:
                        fp.write('%08x' % int(m))

                    fp.write('\n')
                    out = []

        with open(weight_name, "a+") as fp:
            for r in SCALE:
                fp.write('0x')
                out.append(r)
                if len(out) == 1:
                    out.reverse()
                    for m in out:
                        m = m.item()
                        fp.write('%08x' % m)
                    fp.write('\n')
                    out = []
        with open(weight_name, "a+") as fp:

            for r in N_REAL:
                fp.write('0x')
                out.append(r)
                if len(out) == 1:
                    out.reverse()
                    for m in out:
                        m = m.item()
                        fp.write('%08x' % m)
                    fp.write('\n')
                    out = []


def create_weight(data):

    global input_shape,add_write_address, dataSizeB, dataSizeW, weight_name, limit_size33, limit_size11, inchannel33, inchannel11,outchannel11,outchannel33
    data_dict, model, all_weight_data = get_data(limit_size33, limit_size11, 1, 1,
                                                 1, data, input_shape)
    for k, v in data_dict.items():
        if "conv2d" in v["op"]:
            input_key = v["input_node"]
            if "input" in input_key:
                input_name = "quant"
            else:
                if 'concatenate' in data_dict[input_key]["op"]:
                    input_name = str(data_dict[input_key]["cat_name"])
                else:
                    for index_node in range(len(data)):
                        if "weight_name" in data_dict[input_key].keys():
                            input_name = data_dict[input_key]["weight_name"]
                            break
                        else:
                            input_key = data_dict[input_key]["input_node"]
            str_s1 = input_name + '.scale'
            s1 = model[str_s1]
            str_z1 = input_name + '.zero_point'
            z1 = model[str_z1]
            str_weight = str(v["weight_name"]) + '.weight'
            s2 = model[str_weight].q_per_channel_scales()
            z2 = model[str_weight].q_per_channel_zero_points()
            weight_int = model[str_weight].int_repr()
            str_s3 = str(v["weight_name"]) + '.scale'
            s3 = model[str_s3]
            str_z3 = str(v["weight_name"]) + '.zero_point'
            z3 = model[str_z3]
            str_bias = str(v["weight_name"]) + '.bias'
            conv_bias = model[str_bias]
            Conv2d_Q(weight_int, s1=s1, z1=z1, s2=s2, z2=z2, s3=s3, z3=z3, bias=conv_bias,
                     stride=v["strides"],
                     padding=v["padding"], kernel_size=v["kernel_size"])



if __name__ == "__main__":
    start = time.time()
    # 设置输入文件名称
    weight_name = "all_weight2.dat"
    # 防止多次运行重复写入
    with open(weight_name, 'w+') as fp:
        fp.write('')
    # 设置图片大小
    input_shape = [1, 1, 640, 640]
    data = get_mod(input_shape)
    limit_size33 = 512 * 128  # 设置超过多少开始分块(3*3)
    limit_size11 = 512 * 512  # 设置超过多少开始分块(1*1)
    inchannel33 = 16  #33卷积输入通道是多少
    inchannel11 = 32   #11卷积输入通道是多少
    outchannel33 = 8
    outchannel11 = 8
    # out_api = open('aaa.txt')
    # data = out_api.read().splitlines()
    # 设置权重和图片得起始地址(十进制),每次增加得地址数(十进制)
    create_weight(data)
    print('权重写入成功!!!!!!!!!!!!')
