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
def Conv2d_Q3x3(weight_address, computer_address, write_address, s1, z1, s2, z2, s3, z3, bias, stride, padding,
                kernel_size,
                input_shape,
                output_shape):
    block = (output_shape[1] * input_shape[1]) / limit_size33
    block = math.ceil(block)  # 向上取整
    if int(output_shape[1]%8)!=0:
        xxxxxxxxxxxxxxxx = 1  #no use
        # print('111111111111111111111111111111111')
    if block != 1:
        Conv2d_Q3x3_block(weight_address, computer_address, write_address, s1, z1, s2, z2, s3, z3, bias, stride,
                          padding,
                          kernel_size,
                          input_shape,
                          output_shape)
    # computer_address = int(computer_address[-1])
    # if block==1:
    #    write_address = int(write_address[-1])
    # 计算权重的数量B
    else:
        weight_size = (input_shape[1] * output_shape[1] * kernel_size * kernel_size)
        weight_size += ((output_shape[1]) * 3 * 4)
        # weight_size为权重的数量
        # ----------------conv33权重指令-------------------
        # 计算权重的reg4
        reg4 = conv33para(output_shape[1], input_shape[1], dataSizeW, dataSizeB)
        # 计算权重的reg5
        reg5 = '00000000'
        # 权重第一个指令:读地址
        with open(file_name, 'a+') as f:
            f.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            f.write('%08X' % int(weight_address))
            f.write('\n')
            # 权重第二个指令:读数量
            f.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            f.write('%08X' % weight_size)
            f.write('\n')
            # 权重的第三个指令reg4
            f.write('100000' + ins_address['TJPU_Reg4'])
            f.write(str(reg4))
            f.write('\n')
            # 权重的第四个指令reg5
            f.write('100000' + ins_address['TJPU_Reg5'])
            f.write(str(reg5))
            f.write('\n')
            # 计算的第五个指令,switch
            f.write('100000' + ins_address['TJPU_Switch'])
            f.write('%08X' % int(1))
            f.write('\n')
            # 计算的第六个指令,control
            f.write('100000' + ins_address['TJPU_Control'])
            f.write('%08X' % int(1))
            f.write('\n')
            f.write('110000100000000F')
            f.write('\n')
        # ----------------conv33计算指令-------------------
        # 计算图片的数量,单位是B
        feature_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        # 计算写地址
        computer_write_address = write_address
        # 计算输出图片的大小
        # out_size = int((feature_shape[2] - 3 + 2 * padding) / stride) + 1
        # 计算写地址的数量
        write_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
        # 计算的reg4 计算reg5
        computer_reg4, computer_reg5, computer_reg6, computer_reg7 = conv33compute(output_shape[1], input_shape[1],
                                                                                   dataSizeW,
                                                                                   dataSizeB, input_shape[2],
                                                                                   stride, padding,
                                                                                   z1,
                                                                                   z3,
                                                                                   s3)

        # ---p---------------写入计算的指令----------------------
        with open(file_name, 'a+') as fp:
            # 计算的第一个指令读地址
            fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            fp.write('%08X' % int(computer_address))

            fp.write('\n')
            # 计算的第二个指令读数量
            fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            fp.write('%08X' % int(feature_size))
            fp.write('\n')
            # 计算的第三个指令写地址
            fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
            fp.write('%08X' % int(computer_write_address))
            fp.write('\n')
            # 计算的第四个指令写数量
            fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
            fp.write('%08X' % int(write_size))
            fp.write('\n')
            # 计算的第五个指令reg4
            fp.write('100000' + ins_address['TJPU_Reg4'])
            fp.write('%08X' % int(computer_reg4))
            fp.write('\n')
            # 计算的第六个指令reg5
            fp.write('100000' + ins_address['TJPU_Reg5'])
            fp.write('%08X' % int(computer_reg5))
            fp.write('\n')
            # 计算的第七个指令reg6,33卷积reg6的8位不解析写啥都行

            fp.write('100000' + ins_address['TJPU_Reg6'])
            fp.write('%08X' % int(computer_reg6))
            fp.write('\n')
            # 计算的第八个指令reg7,33卷积reg7的后四位(共八位)没用
            fp.write('100000' + ins_address['TJPU_Reg7'])
            fp.write('%08X' % int(computer_reg7))
            fp.write('\n')
            fp.write('100000' + ins_address['TJPU_Switch'])
            fp.write('%08X' % int(1))
            fp.write('\n')
            # 计算的第十四个指令,control
            fp.write('100000' + ins_address['TJPU_Control'])
            fp.write('%08X' % int(2))
            fp.write('\n')
            fp.write('110000100000000F')
            fp.write('\n')


def Conv2d_Q3x3_block(weight_address, computer_address, write_address, s1, z1, s2, z2, s3, z3, bias, stride, padding,
                      kernel_size,
                      input_shape,
                      output_shape):
    block = (output_shape[1] * input_shape[1]) / limit_size33
    block = math.ceil(block)  # 向上取整
    # 计算权重的数量B
    weight_size = int(input_shape[1] * output_shape[1] * kernel_size * kernel_size / block)
    weight_size += ((output_shape[1] / block) * 3 * 4)
    weight_size = int(weight_size)
    # weight_size为权重的数量
    # ----------------conv33权重指令-------------------
    # 计算权重的reg4
    reg4 = conv33para(int(output_shape[1]/block), input_shape[1], dataSizeW, dataSizeB)
    # 计算权重的reg5
    reg5 = '00000000'
    # ----------------conv33计算指令-------------------
    # 计算图片的数量,单位是B
    feature_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    # 计算写地址
    computer_write_address = write_address
    # 计算输出图片的大小
    # out_size = int((feature_shape[2] - 3 + 2 * padding) / stride) + 1
    # 计算写地址的数量
    write_size = int(output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] / block)
    # 计算的reg4 计算reg5
    computer_reg4, computer_reg5, computer_reg6, computer_reg7 = conv33compute(int(output_shape[1] / block), input_shape[1],
                                                                               dataSizeW,
                                                                               dataSizeB, input_shape[2],
                                                                               stride, padding,
                                                                               z1,
                                                                               z3,
                                                                               s3)

    # ---p---------------写入计算的指令----------------------
    for index in range(block):
        with open(file_name, 'a+') as fp:
            # 权重第一个指令:读地址
            fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            fp.write('%08X' % int(weight_address + index * weight_size))
            fp.write('\n')
            # 权重第二个指令:读数量
            fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            fp.write('%08X' % weight_size)
            fp.write('\n')
            # 权重的第三个指令reg4
            fp.write('100000' + ins_address['TJPU_Reg4'])
            # print(reg4)
            # exit()
            fp.write(str(reg4))
            fp.write('\n')
            # 权重的第四个指令reg5
            fp.write('100000' + ins_address['TJPU_Reg5'])
            fp.write(str(reg5))
            fp.write('\n')
            # 计算的第五个指令,switch
            fp.write('100000' + ins_address['TJPU_Switch'])
            fp.write('%08X' % int(1))
            fp.write('\n')
            # 计算的第六个指令,control
            fp.write('100000' + ins_address['TJPU_Control'])
            fp.write('%08X' % int(1))
            fp.write('\n')
            fp.write('110000100000000F')
            fp.write('\n')
            # 计算的第一个指令读地址
            fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            fp.write('%08X' % int(computer_address))
            # print("dudizhi")
            # print('%08X' % int(computer_address))
            fp.write('\n')
            # 计算的第二个指令读数量
            fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            fp.write('%08X' % int(feature_size))
            fp.write('\n')
            # 计算的第三个指令写地址
            fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
            fp.write('%08X' % int(computer_write_address - abs(index - 2 * block + 2) * add_write_address))
            # print("xiedizhi")
            # print('%08X' % int(computer_write_address - abs(index - 2 * block + 2) * add_write_address))
            fp.write('\n')
            # 计算的第四个指令写数量
            fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
            fp.write('%08X' % int(write_size))
            fp.write('\n')
            # 计算的第五个指令reg4
            fp.write('100000' + ins_address['TJPU_Reg4'])
            fp.write('%08X' % int(computer_reg4))
            fp.write('\n')
            # 计算的第六个指令reg5
            fp.write('100000' + ins_address['TJPU_Reg5'])
            fp.write('%08X' % int(computer_reg5))
            fp.write('\n')
            # 计算的第七个指令reg6,33卷积reg6的8位不解析写啥都行

            fp.write('100000' + ins_address['TJPU_Reg6'])
            fp.write('%08X' % int(computer_reg6))
            fp.write('\n')
            # 计算的第八个指令reg7,33卷积reg7的后四位(共八位)没用
            fp.write('100000' + ins_address['TJPU_Reg7'])
            fp.write('%08X' % int(computer_reg7))
            fp.write('\n')
            fp.write('100000' + ins_address['TJPU_Switch'])
            fp.write('%08X' % int(1))
            fp.write('\n')
            # 计算的第十四个指令,control
            fp.write('100000' + ins_address['TJPU_Control'])
            fp.write('%08X' % int(2))
            fp.write('\n')
            fp.write('110000100000000F')
            fp.write('\n')
    for cat_index in range(int(block - 1)):
        cat2_channel = '{:010b}'.format(int(output_shape[1]/block))  # cat2通道数
        # 下面是二进制
        reg4 = cat2_channel + '0000000000000000000000'
        # 将二进制转成十进制
        reg4 = str(int(reg4, 2))
        # 计算concat的reg5()
        feature_h = '{:011b}'.format(input_shape[2])  # 11 位 输入图片 高  (行数)
        cat1_channel = '{:010b}'.format(int(output_shape[1] / block)*(cat_index+1))  # 10 位，concat 中 cat1 的通道数
        feature_w = '{:011b}'.format(input_shape[2])  # 11 位，输入的图片 宽  (列数)
        # 下面位二进制
        reg5 = feature_h + cat1_channel + feature_w
        # 将二进制转成10进制
        reg5 = str(int(reg5, 2))
        # 计算reg6,reg7,reg8,reg9
        reg6 = 65536
        reg7 = 65536
        reg8 = 0
        reg9 = 0
        if cat_index == 0:
            cat1_address = computer_write_address - abs(2 * block - 2) * add_write_address
        else:
            cat1_address = computer_write_address - abs(block - 1 - cat_index) * add_write_address

        # ----------------concat权重指令-------------------
        # 6个全都是默认的
        with open(file_name, 'a+') as fp:
            # -------------concat的计算指令--------------
            # 第一个指令:读第一个concat地址
            fp.write('100000' + ins_address['Image_Reg0'])
            fp.write('%08X' % int(cat1_address))
            # print("cat1_address")
            # print('%08X' % int(cat1_address))
            fp.write('\n')
            # 第二个指令:读第一个concat数量
            fp.write('100000' + ins_address['Image_Reg1'])
            fp.write('%08X' % int(write_size * (cat_index + 1)))
            # print("cat1size")
            # print(int(write_size * (cat_index + 1)))
            fp.write('\n')
            # 第三个指令:读第二个concat地址
            fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            fp.write(
                '%08X' % int(computer_write_address - abs(cat_index - 2 * block + 3) * add_write_address))
            # print("cat2address")
            # print(
            #     '%08X' % int(computer_write_address - abs(cat_index - 2 * block + 3) * add_write_address))
            fp.write('\n')
            # 第四个指令:读第二个concat数量
            fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            fp.write('%08X' % int(write_size))
            fp.write('\n')
            # 第五个指令:写地址
            fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
            fp.write(
                '%08X' % int(computer_write_address - abs(block - 2 - cat_index) * add_write_address))
            # print('write_address')
            # print('%x'%int(computer_write_address - abs(block - 2 - cat_index) * add_write_address))
            fp.write('\n')
            # 第六个指令:写数量
            fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
            fp.write('%08X' % int(write_size + write_size * (cat_index + 1)))
            fp.write('\n')
            # 第七个指令:reg4
            fp.write('100000' + ins_address['TJPU_Reg4'])
            fp.write('%08X' % int(reg4))
            fp.write('\n')
            # 第八个指令:reg5
            fp.write('100000' + ins_address['TJPU_Reg5'])
            fp.write('%08X' % int(reg5))
            fp.write('\n')
            # 第九个指令:reg6
            fp.write('100000' + ins_address['TJPU_Reg6'])
            fp.write('%08X' % int(reg6))
            fp.write('\n')
            # 第十个指令:reg7
            fp.write('100000' + ins_address['TJPU_Reg7'])
            fp.write('%08X' % int(reg7))
            fp.write('\n')
            # 第十一个指令:reg8
            fp.write('100000' + ins_address['TJPU_Reg8'])
            fp.write('%08X' % int(reg8))
            fp.write('\n')
            # 第十二个指令:reg9
            fp.write('100000' + ins_address['TJPU_Reg9'])
            fp.write('%08X' % int(reg9))
            fp.write('\n')
            # 第十三个指令:switch
            fp.write('100000' + ins_address['TJPU_Switch'])
            fp.write('00000008')
            fp.write('\n')
            # 第十四个指令:control
            fp.write('100000' + ins_address['TJPU_Control'])
            fp.write('00000100')
            fp.write('\n')
            fp.write('1100001000000F00')
            fp.write('\n')


def Conv2d_Q1x1(weight_address, computer_address, write_address, s1, z1, s2, z2, s3, z3, bias, stride, padding,
                kernel_size,
                input_shape,
                output_shape,isleak):
    block = (output_shape[1] * input_shape[1]) / limit_size11
    block = math.ceil(block)  # 向上取整
    if int(output_shape[1]%8)!=0:
        # print('111111111111111111111111111111111')
        new_out_channel =(int(output_shape[1]/8)+1)*8
        output_shape = (output_shape[0],new_out_channel,output_shape[2],output_shape[3])
        # print(new_out_channel)
        # print(output_shape[0])
        # print(output_shape[1])
        # print(output_shape[2])
        # print(output_shape[3])
        # print('111111111111111111111111111111111')
    # computer_address = computer_address
    if block != 1:
        Conv2d_Q1x1_block(weight_address, computer_address, write_address, s1, z1, s2, z2, s3, z3, bias, stride, padding,
                kernel_size,
                input_shape,
                output_shape,isleak)
    else:
        # 计算权重的数量
        weight_size = input_shape[1] * output_shape[1] * kernel_size * kernel_size
        weight_size += output_shape[1] * 3 * 4
        # 权重11conv的reg4
        reg4 = conv11para(int(output_shape[1]/block), input_shape[1], dataSizeW, dataSizeB)
        # 权重reg5的二进制32位全0
        # -------------conv11权重指令--------------
        # 权重第一个指令:读地址
        with open(file_name, 'a+') as f:
            f.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            f.write('%08X' % int(weight_address))
            f.write('\n')
            # 权重第二个指令:读数量
            f.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            f.write('%08X' % weight_size)
            f.write('\n')
            # 权重的第三个指令reg4
            f.write('100000' + ins_address['TJPU_Reg4'])
            f.write('%08X' % int(reg4))
            f.write('\n')
            # 权重的第四个指令reg5
            f.write('100000' + ins_address['TJPU_Reg5'])
            f.write('00000000')
            f.write('\n')
            # 计算的第五个指令,switch
            f.write('100000' + ins_address['TJPU_Switch'])
            f.write('%08X' % int(2))
            f.write('\n')
            # 计算的第六个指令,control
            f.write('100000' + ins_address['TJPU_Control'])
            f.write('%08X' % int(16))
            f.write('\n')
            f.write('11000010000000F0')
            f.write('\n')

        weight_address = weight_address + weight_size

        # 计算读地址数量
        feature_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        # 计算写地址
        computer_write_address = write_address
        # 计算输出图片的大小
        # out_size = int((feature_shape[2] - 1 + 2 * padding) / stride) + 1
        # 计算写地址的数量
        write_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
        # 计算11conv的reg4,reg5,reg6,reg7
        # isleakrelu:是否需要 leakyrelu 。 1 为不用，0 为用

        computer_reg4, computer_reg5, computer_reg6, computer_reg7 = conv11compute(output_shape[1], input_shape[1],
                                                                                   dataSizeW,
                                                                                   dataSizeB, input_shape[2],
                                                                                   stride, padding,
                                                                                   z1,
                                                                                   z3,
                                                                                   isleak,
                                                                                   s3)

        # print(self.coe_name)
        # 无leakrelu则等于0
        #     computer_reg6 = 0

        # -------------conv11计算指令--------------
        with open(file_name, 'a+') as fp:
            # 计算的第一个指令读地址
            fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            fp.write('%08X' % int(computer_address))

            fp.write('\n')
            # 计算的第二个指令读数量
            fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            fp.write('%08X' % int(feature_size))
            fp.write('\n')
            # 计算的第三个指令写地址
            fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
            fp.write('%08X' % int(computer_write_address))
            fp.write('\n')
            # 计算的第四个指令写数量
            fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
            fp.write('%08X' % int(write_size))
            fp.write('\n')
            # 计算的第五个指令reg4
            fp.write('100000' + ins_address['TJPU_Reg4'])
            fp.write('%08X' % int(computer_reg4))
            fp.write('\n')
            # 计算的第六个指令reg5
            fp.write('100000' + ins_address['TJPU_Reg5'])
            fp.write('%08X' % int(computer_reg5))
            fp.write('\n')
            # 计算的第七个指令reg6,reg6的8位不解析写啥都行(所有都没用)

            fp.write('100000' + ins_address['TJPU_Reg6'])
            fp.write('%08X' % int(computer_reg6))
            fp.write('\n')
            # 计算的第八个指令reg7,reg7的第七位没用
            fp.write('100000' + ins_address['TJPU_Reg7'])
            fp.write('%08X' % int(computer_reg7))
            fp.write('\n')
            # 计算的第十三个指令,switch
            fp.write('100000' + ins_address['TJPU_Switch'])
            fp.write('%08X' % int(2))
            fp.write('\n')
            # 计算的第十四个指令,control
            fp.write('100000' + ins_address['TJPU_Control'])
            fp.write('%08X' % int(32))
            fp.write('\n')
            fp.write('11000010000000F0')
            fp.write('\n')
def Conv2d_Q1x1_block(weight_address, computer_address, write_address, s1, z1, s2, z2, s3, z3, bias, stride, padding,
                kernel_size,
                input_shape,
                output_shape,isleak):
    block = (output_shape[1] * input_shape[1]) / limit_size11
    block = math.ceil(block)  # 向上取整
    # 计算权重的数量B
    weight_size = input_shape[1] * output_shape[1] * kernel_size * kernel_size
    weight_size += output_shape[1] * 3 * 4
    # 权重11conv的reg4
    reg4 = conv11para(output_shape[1], input_shape[1], dataSizeW, dataSizeB)
    reg4 = conv11para(output_shape[1], input_shape[1], dataSizeW, dataSizeB)
    # 权重reg5的二进制32位全0
    weight_address = weight_address + weight_size

    # 计算读地址数量
    feature_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    # 计算写地址
    computer_write_address = write_address
    # 计算输出图片的大小
    # out_size = int((feature_shape[2] - 1 + 2 * padding) / stride) + 1
    # 计算写地址的数量
    write_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    # 计算11conv的reg4,reg5,reg6,reg7
    # isleakrelu:是否需要 leakyrelu 。 1 为不用，0 为用

    computer_reg4, computer_reg5, computer_reg6, computer_reg7 = conv11compute(int(output_shape[1]/block), input_shape[1],
                                                                               dataSizeW,
                                                                               dataSizeB, input_shape[2],
                                                                               stride, padding,
                                                                               z1,
                                                                               z3,
                                                                               isleak,
                                                                               s3)


    # ---p---------------写入指令----------------------
    for index in range(block):
        with open(file_name, 'a+') as fp:
            # 权重第一个指令:读地址
            fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            fp.write('%08X' % int(weight_address + index * weight_size))
            fp.write('\n')
            # 权重第二个指令:读数量
            fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            fp.write('%08X' % weight_size)
            fp.write('\n')
            # 权重的第三个指令reg4
            fp.write('100000' + ins_address['TJPU_Reg4'])
            fp.write('%08X' % int(reg4))
            fp.write('\n')
            # 权重的第四个指令reg5
            fp.write('100000' + ins_address['TJPU_Reg5'])
            fp.write('00000000')
            fp.write('\n')
            # 计算的第五个指令,switch
            fp.write('100000' + ins_address['TJPU_Switch'])
            fp.write('%08X' % int(2))
            fp.write('\n')
            # 计算的第六个指令,control
            fp.write('100000' + ins_address['TJPU_Control'])
            fp.write('%08X' % int(16))
            fp.write('\n')
            fp.write('11000014000000F0')
            fp.write('\n')
            # 计算的第一个指令读地址
            fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            fp.write('%08X' % int(computer_address))
            # print("dudizhi")
            # print('%08X' % int(computer_address))
            fp.write('\n')
            # 计算的第二个指令读数量
            fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            fp.write('%08X' % int(feature_size))
            fp.write('\n')
            # 计算的第三个指令写地址
            fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
            fp.write('%08X' % int(computer_write_address - abs(index - 2 * block + 2) * add_write_address))
            # print("xiedizhi")
            # print('%08X' % int(computer_write_address - abs(index - 2 * block + 2) * add_write_address))
            fp.write('\n')
            # 计算的第四个指令写数量
            fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
            fp.write('%08X' % int(write_size))
            fp.write('\n')
            # 计算的第五个指令reg4
            fp.write('100000' + ins_address['TJPU_Reg4'])
            fp.write('%08X' % int(computer_reg4))
            fp.write('\n')
            # 计算的第六个指令reg5
            fp.write('100000' + ins_address['TJPU_Reg5'])
            fp.write('%08X' % int(computer_reg5))
            fp.write('\n')
            # 计算的第七个指令reg6,reg6的8位不解析写啥都行(所有都没用)

            fp.write('100000' + ins_address['TJPU_Reg6'])
            fp.write('%08X' % int(computer_reg6))
            fp.write('\n')
            # 计算的第八个指令reg7,reg7的第七位没用
            fp.write('100000' + ins_address['TJPU_Reg7'])
            fp.write('%08X' % int(computer_reg7))
            fp.write('\n')
            # 计算的第十三个指令,switch
            fp.write('100000' + ins_address['TJPU_Switch'])
            fp.write('%08X' % int(2))
            fp.write('\n')
            # 计算的第十四个指令,control
            fp.write('100000' + ins_address['TJPU_Control'])
            fp.write('%08X' % int(32))
            fp.write('\n')
            fp.write('11000010000000F0')
            fp.write('\n')
    for cat_index in range(int(block - 1)):
        cat2_channel = '{:010b}'.format(int(output_shape[1]/block))  # cat2通道数
        # 下面是二进制
        reg4 = cat2_channel + '0000000000000000000000'
        # 将二进制转成十进制
        reg4 = str(int(reg4, 2))
        # 计算concat的reg5()
        feature_h = '{:011b}'.format(input_shape[2])  # 11 位 输入图片 高  (行数)
        cat1_channel = '{:010b}'.format(int(output_shape[1] / block)*(cat_index+1))  # 10 位，concat 中 cat1 的通道数
        feature_w = '{:011b}'.format(input_shape[2])  # 11 位，输入的图片 宽  (列数)
        # 下面位二进制
        reg5 = feature_h + cat1_channel + feature_w
        # 将二进制转成10进制
        reg5 = str(int(reg5, 2))
        # 计算reg6,reg7,reg8,reg9
        reg6 = 65536
        reg7 = 65536
        reg8 = 0
        reg9 = 0
        if cat_index == 0:
            cat1_address = computer_write_address - abs(2 * block - 2) * add_write_address
        else:
            cat1_address = computer_write_address - abs(block - 1 - cat_index) * add_write_address

        # ----------------concat权重指令-------------------
        # 6个全都是默认的
        with open(file_name, 'a+') as fp:
            # -------------concat的计算指令--------------
            # 第一个指令:读第一个concat地址
            fp.write('100000' + ins_address['Image_Reg0'])
            fp.write('%08X' % int(cat1_address))
            # print("cat1_address")
            # print('%08X' % int(cat1_address))
            fp.write('\n')
            # 第二个指令:读第一个concat数量
            fp.write('100000' + ins_address['Image_Reg1'])
            fp.write('%08X' % int(write_size * (cat_index + 1)))
            # print("cat1size")
            # print(int(write_size * (cat_index + 1)))
            fp.write('\n')
            # 第三个指令:读第二个concat地址
            fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            fp.write(
                '%08X' % int(computer_write_address - abs(cat_index - 2 * block + 3) * add_write_address))
            # print("cat2address")
            # print(
            #     '%08X' % int(computer_write_address - abs(cat_index - 2 * block + 3) * add_write_address))
            fp.write('\n')
            # 第四个指令:读第二个concat数量
            fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            fp.write('%08X' % int(write_size))
            fp.write('\n')
            # 第五个指令:写地址
            fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
            fp.write(
                '%08X' % int(computer_write_address - abs(block - 2 - cat_index) * add_write_address))
            # print('write_address')
            # print('%x' % int(computer_write_address - abs(block - 2 - cat_index) * add_write_address))
            fp.write('\n')
            # 第六个指令:写数量
            fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
            fp.write('%08X' % int(write_size + write_size * (cat_index + 1)))
            fp.write('\n')
            # 第七个指令:reg4
            fp.write('100000' + ins_address['TJPU_Reg4'])
            fp.write('%08X' % int(reg4))
            fp.write('\n')
            # 第八个指令:reg5
            fp.write('100000' + ins_address['TJPU_Reg5'])
            fp.write('%08X' % int(reg5))
            fp.write('\n')
            # 第九个指令:reg6
            fp.write('100000' + ins_address['TJPU_Reg6'])
            fp.write('%08X' % int(reg6))
            fp.write('\n')
            # 第十个指令:reg7
            fp.write('100000' + ins_address['TJPU_Reg7'])
            fp.write('%08X' % int(reg7))
            fp.write('\n')
            # 第十一个指令:reg8
            fp.write('100000' + ins_address['TJPU_Reg8'])
            fp.write('%08X' % int(reg8))
            fp.write('\n')
            # 第十二个指令:reg9
            fp.write('100000' + ins_address['TJPU_Reg9'])
            fp.write('%08X' % int(reg9))
            fp.write('\n')
            # 第十三个指令:switch
            fp.write('100000' + ins_address['TJPU_Switch'])
            fp.write('00000008')
            fp.write('\n')
            # 第十四个指令:control
            fp.write('100000' + ins_address['TJPU_Control'])
            fp.write('00000100')
            fp.write('\n')
            fp.write('1100001000000F00')
            fp.write('\n')


def image_Q(weight_address, computer_address, s1, z1, s2, z2, s3, z3, bias, stride, padding, kernel_size, input_shape,
            output_shape):
    # 计算权重的数量
    weight_size = int((input_shape[1] * output_shape[1] * kernel_size * kernel_size) / 9) * 23 + input_shape[1] * \
                  output_shape[1] * kernel_size * kernel_size + output_shape[1] * 4 * 3 + 32
    # weight_size为权重的数量
    weight_address = weight_address - weight_size
    # ----------------conv33权重指令-------------------
    with open(file_name, 'a+') as f:
        # 权重第一个指令:读地址
        f.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
        f.write('%08X' % int(weight_address))
        f.write('\n')
        # 权重第二个指令:读数量
        f.write('100000' + ins_address['TJPU_DMA_Read_Num'])
        f.write('%08X' % weight_size)
        f.write('\n')
        f.write('')
        f.write('1000000000000001')
        f.write('\n')
        f.write('110000000000000F')
        f.write('\n')
    # ----------------conv33计算指令-------------------
    # 计算图片的数量,单位是B
    feature_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    # 计算写地址
    computer_write_address = computer_address + add_write_address
    # 计算写地址的数量
    write_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    # ------------------写入计算的指令----------------------
    with open(file_name, 'a+') as fp:
        # 新加写入
        # 计算的第一个指令读地址
        fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
        fp.write('%08X' % int(0))
        fp.write('\n')
        # 计算的第二个指令读数量
        fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
        fp.write('%08X' % int(feature_size))
        fp.write('\n')
        # 计算的第三个指令写地址
        fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
        fp.write('%08X' % int(add_write_address))
        fp.write('\n')
        # 计算的第四个指令写数量
        fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
        fp.write('%08X' % int(write_size))
        fp.write('\n')
        fp.write('100000' + '00')
        fp.write('%08X' % int(2))
        fp.write('\n')
        fp.write('110000000000000F')
        fp.write('\n')


def Cat_Q(input_shape, output_shape, write_compute_address, cat1_address, cat2_address, s1, s2, s3, z1, z2, z3):
    cat1_shape = input_shape[0]
    cat2_shape = input_shape[1]
    cat1_address = cat1_address
    cat2_address = cat2_address
    cat1_size = cat1_shape[0] * cat1_shape[1] * cat1_shape[2] * cat1_shape[3]
    cat2_size = cat2_shape[0] * cat2_shape[1] * cat2_shape[2] * cat2_shape[3]
    # 计算写地址
    computer_write_address = write_compute_address
    # 计算concat的reg4
    cat2_channel = '{:010b}'.format(cat2_shape[1])  # cat2通道数
    # 下面是二进制
    reg4 = cat2_channel + '0000000000000000000000'
    # 将二进制转成十进制
    reg4 = str(int(reg4, 2))
    # 计算concat的reg5()
    feature_h = '{:011b}'.format(cat2_shape[2])  # 11 位 输入图片 高  (行数)
    cat1_channel = '{:010b}'.format(cat1_shape[1])  # 10 位，concat 中 cat1 的通道数
    feature_w = '{:011b}'.format(cat2_shape[3])  # 11 位，输入的图片 宽  (列数)
    # 下面位二进制
    reg5 = feature_h + cat1_channel + feature_w
    # 将二进制转成10进制
    reg5 = str(int(reg5, 2))
    # 计算reg6,reg7,reg8,reg9
    reg6, reg7, reg8, reg9 = reg_cat(s1, s2, s3,
                                     z1,
                                     z2, z3)

    # ----------------concat权重指令-------------------
    # 6个全都是默认的
    with open(file_name, 'a+') as fp:
        # -------------concat的计算指令--------------
        # 第一个指令:读第一个concat地址
        # print('%08X' % cat2_address)
        fp.write('100000' + ins_address['Image_Reg0'])
        fp.write('%08X' % cat1_address)
        fp.write('\n')
        # 第二个指令:读第一个concat数量
        fp.write('100000' + ins_address['Image_Reg1'])
        fp.write('%08X' % cat1_size)
        fp.write('\n')
        # 第三个指令:读第二个concat地址
        fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
        fp.write('%08X' % cat2_address)
        fp.write('\n')
        # 第四个指令:读第二个concat数量
        fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
        fp.write('%08X' % cat2_size)
        fp.write('\n')
        # 第五个指令:写地址
        fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
        fp.write('%08X' % computer_write_address)
        fp.write('\n')
        # 第六个指令:写数量
        fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
        fp.write('%08X' % (cat2_size + cat1_size))
        fp.write('\n')
        # 第七个指令:reg4
        fp.write('100000' + ins_address['TJPU_Reg4'])
        fp.write('%08X' % int(reg4))
        fp.write('\n')
        # 第八个指令:reg5
        fp.write('100000' + ins_address['TJPU_Reg5'])
        fp.write('%08X' % int(reg5))
        fp.write('\n')
        # 第九个指令:reg6
        fp.write('100000' + ins_address['TJPU_Reg6'])
        fp.write('%08X' % int(reg6))
        fp.write('\n')
        # 第十个指令:reg7
        fp.write('100000' + ins_address['TJPU_Reg7'])
        fp.write('%08X' % int(reg7))
        fp.write('\n')
        # 第十一个指令:reg8
        fp.write('100000' + ins_address['TJPU_Reg8'])
        fp.write('%08X' % int(reg8))
        fp.write('\n')
        # 第十二个指令:reg9
        fp.write('100000' + ins_address['TJPU_Reg9'])
        fp.write('%08X' % int(reg9))
        fp.write('\n')
        # 第十三个指令:switch
        fp.write('100000' + ins_address['TJPU_Switch'])
        fp.write('00000008')
        fp.write('\n')
        # 第十四个指令:control
        fp.write('100000' + ins_address['TJPU_Control'])
        fp.write('00000100')
        fp.write('\n')
        fp.write('1100001000000F00')
        fp.write('\n')


def reg_cat(cat1_scale, cat2_scale, cat3_scale, cat1_zero_point, cat2_zero_point, cat3_zero_point):
    zero_point_one = (cat3_scale / cat1_scale) * cat3_zero_point - cat1_zero_point
    zero_point_one = (torch.round(zero_point_one * (2 ** 16)))
    zero_point_one = zero_point_one.numpy().astype(np.uint32)
    M1 = (torch.round((cat1_scale / cat3_scale) * (2 ** 16)))
    M1 = M1.numpy().astype(np.uint32)
    zero_point_two = (cat3_scale / cat2_scale) * cat3_zero_point - cat2_zero_point
    zero_point_two = (torch.round(zero_point_two * (2 ** 16)))
    zero_point_two = zero_point_two.numpy().astype(np.uint32)
    M2 = (torch.round((cat2_scale / cat3_scale) * (2 ** 16)))
    M2 = M2.numpy().astype(np.uint32)
    return M1, M2, zero_point_one, zero_point_two


def reshape_maxpool(computer_address, write_address, input_shape):
    # 计算读的数量
    shape = input_shape
    channel_in = shape[1]
    feature_in = shape[2]
    feature_size = shape[0] * channel_in * feature_in * feature_in
    # print(computer_address)
    # 计算写地址
    # computer_write_address = computer_address + add_write_address
    # 计算写数量
    write_size = 0
    write_size = int(feature_size / 4)

    # ----------------reshape权重指令-------------------
    # 6个全都是默认的
    with open(file_name, 'a+') as fp:
        # -------------reshape的计算指令--------------
        # reshape的reg4,reg5,reg6都全是0
        #    reg7:11位   split,maxpool,upsample 没用到,10位输入图片通道, 11 位，输入的图片 宽(高)
        channel_in = '{:010b}'.format(channel_in)
        feature_in = '{:011b}'.format(feature_in)
        reg7 = '00000000000' + str(channel_in) + str(feature_in)
        reg7 = str(int(reg7, 2))
        # 第一个指令:读第一个reshape地址
        fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
        fp.write('%08X' % computer_address)
        fp.write('\n')
        # 第二个指令:读第一个reshape数量
        fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
        fp.write('%08X' % feature_size)
        fp.write('\n')
        # 第三个指令:写地址
        fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
        fp.write('%08X' % int(write_address))
        fp.write('\n')
        # 第六个指令:写数量
        fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
        fp.write('%08X' % (write_size))
        fp.write('\n')
        # 第七个指令:reg4
        fp.write('100000' + ins_address['TJPU_Reg4'])
        fp.write('00000000')
        fp.write('\n')
        # 第八个指令:reg5
        fp.write('100000' + ins_address['TJPU_Reg5'])
        fp.write('00000000')
        fp.write('\n')
        # 第九个指令:reg6
        fp.write('100000' + ins_address['TJPU_Reg6'])
        fp.write('00000000')
        fp.write('\n')
        # 第十个指令:reg7
        fp.write('100000' + ins_address['TJPU_Reg7'])
        fp.write('%08X' % int(reg7))
        fp.write('\n')
        # 第十三个指令:switch
        fp.write('100000' + ins_address['TJPU_Switch'])
        fp.write('00000008')
        fp.write('\n')
        # 第十四个指令:control
        fp.write('100000' + ins_address['TJPU_Control'])
        fp.write('00000400')
        fp.write('\n')
        fp.write('1100001000000F00')
        fp.write('\n')


def reshape_upsample(computer_address, write_address, input_shape):
    # 计算读的数量
    feature_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    # 计算写地址
    # computer_write_address = computer_address + add_write_address
    # 计算写数量
    write_size = 0
    write_size = int(feature_size * 4)

    # ----------------reshape权重指令-------------------
    # 6个全都是默认的
    with open(file_name, 'a+') as fp:
        # -------------reshape的计算指令--------------
        # reshape的reg4,reg5,reg6都全是0
        #    reg7:11位   split,maxpool,upsample 没用到,10位输入图片通道, 11 位，输入的图片 宽(高)
        channel_in = '{:010b}'.format(input_shape[1])
        feature_in = '{:011b}'.format(input_shape[2])
        reg7 = '00000000000' + str(channel_in) + str(feature_in)
        reg7 = str(int(reg7, 2))
        # 第一个指令:读第一个reshape地址
        fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
        fp.write('%08X' % computer_address)
        fp.write('\n')
        # 第二个指令:读第一个reshape数量
        fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
        fp.write('%08X' % feature_size)
        fp.write('\n')
        # 第三个指令:写地址
        fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
        fp.write('%08X' % int(write_address))
        fp.write('\n')
        # 第六个指令:写数量
        fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
        fp.write('%08X' % (write_size))
        fp.write('\n')
        # 第七个指令:reg4
        fp.write('100000' + ins_address['TJPU_Reg4'])
        fp.write('00000000')
        fp.write('\n')
        # 第八个指令:reg5
        fp.write('100000' + ins_address['TJPU_Reg5'])
        fp.write('00000000')
        fp.write('\n')
        # 第九个指令:reg6
        fp.write('100000' + ins_address['TJPU_Reg6'])
        fp.write('00000000')
        fp.write('\n')
        # 第十个指令:reg7
        fp.write('100000' + ins_address['TJPU_Reg7'])
        fp.write('%08X' % int(reg7))
        fp.write('\n')
        # 第十三个指令:switch
        fp.write('100000' + ins_address['TJPU_Switch'])
        fp.write('00000008')
        fp.write('\n')
        # 第十四个指令:control

        fp.write('100000' + ins_address['TJPU_Control'])
        fp.write('00000800')
        fp.write('\n')
        fp.write('1100001000000F00')
        fp.write('\n')


def create_ins(weight_address, computer_address, data, input_shape):
    conv_index = 0
    global add_write_address, dataSizeB, dataSizeW, file_name, limit_size33, limit_size11
    data_dict, model, all_weight_data = get_data(limit_size33, limit_size11, weight_address, computer_address,
                                                 add_write_address, data, input_shape)
    print(data_dict)
    exit()
    for k,v in data_dict.items():

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
            str_s3 = str(v["weight_name"]) + '.scale'
            s3 = model[str_s3]
            str_z3 = str(v["weight_name"]) + '.zero_point'
            z3 = model[str_z3]
            str_bias = str(v["weight_name"]) + '.bias'
            conv_bias = model[str_bias]
            input_key = v["input_node"]
            if "input" in v["input_node"]:
                image_Q(weight_address, computer_address, s1=s1, z1=z1, s2=s2, z2=z2, s3=s3, z3=z3, bias=conv_bias,
                        stride=v["strides"],
                        padding=v["padding"], kernel_size=v["kernel_size"], input_shape=v["input_shape"],
                        output_shape=v["output_shape"])
            else:
                if 'leak' in data_dict[input_key]["op"]:
                    for index_node in range(len(data)):
                        if "weight_name" in data_dict[input_key].keys():
                            new_computer_address = data_dict[input_key]["write_compute_address"]
                            break
                        else:
                            input_key = data_dict[input_key]["input_node"]
                else:
                    if "write_compute_address" in data_dict[input_key].keys():
                        new_computer_address = data_dict[input_key]["write_compute_address"]
                    else:
                        for index_node in range(len(data)):
                            if "write_compute_address" in data_dict[input_key].keys():
                                new_computer_address = data_dict[input_key]["write_compute_address"]
                                break
                            else:
                                input_key = data_dict[input_key]["input_node"]
                write_address = v["write_compute_address"]
                if v["kernel_size"] == 3:
                    # print(v)
                    # print('权重读地址:', '%x' % all_weight_data[conv_index]["address"])
                    # print('计算读地址:', '%x' % new_computer_address)
                    # print('计算写地址:', '%x' % write_address)
                    Conv2d_Q3x3(all_weight_data[conv_index]["address"], new_computer_address, write_address, s1=s1,
                                z1=z1, s2=s2,
                                z2=z2,
                                s3=s3, z3=z3,
                                bias=conv_bias, stride=v["strides"],
                                padding=v["padding"], kernel_size=v["kernel_size"], input_shape=v["input_shape"],
                                output_shape=v["output_shape"])
                elif v["kernel_size"] == 1:

                    # new_computer_address = data_dict[input_key]["write_compute_address"]  # 输入节点的写地址就是这个的读地址
                    # print(v)
                    # print(input_key)
                    # print(all_weight_data[conv_index])
                    # print('权重读地址:', '%x' % all_weight_data[conv_index]["address"])
                    # print('计算读地址:', '%x' % new_computer_address)
                    # print('计算写地址:', '%x' % write_address)
                    Conv2d_Q1x1(all_weight_data[conv_index]["address"], new_computer_address, write_address, s1=s1,
                                z1=z1, s2=s2,
                                z2=z2,
                                s3=s3, z3=z3,
                                bias=conv_bias, stride=v["strides"],
                                padding=v["padding"], kernel_size=v["kernel_size"], input_shape=v["input_shape"],
                                output_shape=v["output_shape"],isleak = v["isleak"])

            conv_index += 1
        elif "concatenate" in v["op"]:
            # exit()
            input_key = v["input_node"]
            cat3_name = v["cat_name"]
            cat1_node = data_dict[input_key]["input_node"][0]
            for index_node in range(len(data)):
                if "weight_name" in data_dict[cat1_node].keys():
                    cat1_name = data_dict[cat1_node]["weight_name"]
                    break
                else:
                    cat1_node = data_dict[cat1_node]["input_node"]
            # print(cat1_node)
            cat1_node = data_dict[input_key]["input_node"][0]
            for index_node in range(len(data)):
                if "weight_name" in data_dict[cat1_node].keys():
                    cat1_address = data_dict[cat1_node]["write_compute_address"]
                    break
                elif "image.resize" in data_dict[cat1_node]["op"]:
                    cat1_address = data_dict[cat1_node]["write_compute_address"]
                    # print('///////////////')
                    break
                elif "max_pool2d" in data_dict[cat1_node]["op"]:
                    cat1_address = data_dict[cat1_node]["write_compute_address"]
                    break
                else:
                    cat1_node = data_dict[cat1_node]["input_node"]
            cat2_node = data_dict[input_key]["input_node"][1]
            # print(cat1_node)
            # print('-----------------------')
            for index_node in range(len(data)):
                if "weight_name" in data_dict[cat2_node].keys():
                    cat2_name = data_dict[cat2_node]["weight_name"]

                    break
                else:
                    cat2_node = data_dict[cat2_node]["input_node"]
            cat2_node = data_dict[input_key]["input_node"][1]
            for index_node in range(len(data)):
                if "weight_name" in data_dict[cat2_node].keys():
                    cat2_address = data_dict[cat2_node]["write_compute_address"]
                    break
                elif "image.resize" in data_dict[cat2_node]["op"]:
                    cat2_address = data_dict[cat2_node]["write_compute_address"]
                    break
                elif "max_pool2d" in data_dict[cat2_node]["op"]:
                    cat2_address = data_dict[cat2_node]["write_compute_address"]
                    break
                else:
                    cat2_node = data_dict[cat2_node]["input_node"]
            str_s1 = cat1_name + '.scale'
            s1 = model[str_s1]
            str_s2 = cat2_name + '.scale'
            s2 = model[str_s2]
            str_s3 = cat3_name + '.scale'
            s3 = model[str_s3]
            str_z1 = cat1_name + '.zero_point'
            z1 = model[str_z1]
            str_z2 = cat2_name + '.zero_point'
            z2 = model[str_z2]
            str_z3 = cat3_name + '.zero_point'
            z3 = model[str_z3]
            # print(v)
            # print('ca1地址:', '%x' % cat1_address)
            # print('ca2地址:', '%x' % cat2_address)
            # print('计算写地址:', '%x' % v["write_compute_address"])
            Cat_Q(input_shape=v["input_shape"], output_shape=v["output_shape"],
                  write_compute_address=v["write_compute_address"], cat1_address=cat1_address,
                  cat2_address=cat2_address,
                  s1=s1, z1=z1, s2=s2, z2=z2, s3=s3, z3=z3)
        elif "max_pool2d" in v["op"]:
            input_key = v["input_node"]

            if 'leak' in data_dict[input_key]["op"]:
                for index_node in range(len(data)):
                    if "weight_name" in data_dict[input_key].keys():
                        break
                    else:
                        input_key = data_dict[input_key]["input_node"]
            computer_address = data_dict[input_key]["write_compute_address"]

            write_address = v["write_compute_address"]
            # print(v)
            # print('计算读地址:', '%x' % computer_address)
            # print('计算写地址:', '%x' % write_address)
            reshape_maxpool(computer_address, write_address, v["input_shape"])
        elif "image.resize" in v["op"]:
            input_key = v["input_node"]
            if 'leak' in data_dict[input_key]["op"]:
                for index_node in range(len(data)):
                    if "weight_name" in data_dict[input_key].keys():
                        computer_address = data_dict[input_key]["write_compute_address"]
                        break
                    else:
                        input_key = data_dict[input_key]["input_node"]
            else:
                if "write_compute_address" in data_dict[input_key].keys():
                    new_computer_address = data_dict[input_key]["write_compute_address"]
                else:
                    for index_node in range(len(data)):
                        if "write_compute_address" in data_dict[input_key].keys():
                            new_computer_address = data_dict[input_key]["write_compute_address"]
                            break
                        else:
                            input_key = data_dict[input_key]["input_node"]
            write_address = v["write_compute_address"]
            # print(v)
            # print('计算读地址:', '%x' % computer_address)
            # print('计算写地址:', '%x' % write_address)
            reshape_upsample(computer_address, write_address, v["input_shape"])


if __name__ == "__main__":
    start = time.time()
    # 设置输入文件名称
    file_name = "test.txt"
    # 防止多次运行重复写入
    with open(file_name, 'w+') as fp:
        fp.write('')
    # 设置图片大小
    input_shape = [1, 1, 640, 640]
    data = get_mod(input_shape)
    # dataSizeW是权重每行多少bit, dataSizeB是bias每行多少bit
    dataSizeW = 64
    dataSizeB = 64
    limit_size33 = 512 * 128  # 设置超过多少开始分块(3*3)
    limit_size11 = 512 * 512  # 设置超过多少开始分块(1*1)
    # 指令地址
    ins_address = {'TJPU_Control': '10', 'TJPU_State': '14',
                   'TJPU_Switch': '18', 'TJPU_DMA_Read_Addr': '1C',
                   'TJPU_DMA_Read_Num': '20', 'TJPU_DMA_Write_Addr': '24',
                   'TJPU_DMA_Write_Num': '28', 'TJPU_Reg4': '2C',
                   'TJPU_Reg5': '30', 'TJPU_Reg6': '34',
                   'TJPU_Reg7': '38', 'TJPU_Reg8': '3C', 'TJPU_Reg9': '40', 'Image_Reg0': '08',
                   'Image_Reg1': '0C'}
    # out_api = open('aaa.txt')
    # data = out_api.read().splitlines()
    # 设置权重和图片得起始地址(十进制),每次增加得地址数(十进制)
    weight_address = 1879048192  # 起始地址:70000000
    computer_address = 16777216  # 起始地址:1000000
    add_write_address = 16777216  # 计算地址每次加的数(1000000)
    create_ins(weight_address, computer_address, data, input_shape)
    print('指令写入成功!!!!!!!!!!!!')
