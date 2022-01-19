import json
import torch
import re
from get_mod import *
import math


def get_cat_info(model):
    cat_name = {}
    cat_name["scale"] = []
    cat_name["zero_point"] = []
    for k, v in model.items():
        if 'route' in k:
            if 'scale' in k:
                cat_name["scale"].append(k)

            elif 'zero_point' in k:
                cat_name["zero_point"].append(k)
    return cat_name


def get_data(limit_size33, limit_size11, weight_address, computer_address, add_write_address, data_list, input_shape):
    model = torch.load('/home/anyilin/compiler_demo/quan_pth/Epoch1-YOLOV4_quantization_post_save.pth', map_location='cpu')
    cat_name = get_cat_info(model)
    data = data_list
    net_index = 0
    data_dict = {}
    cat_index = 0  # 第几个cat
    conv_number = 0  # 第几个conv
    # read_weight_address = weight_address  # 读取权重的地址
    all_weight_data = {}
    all_weight_data[conv_number] = {}
    all_weight_data[conv_number]["address"] = 0
    all_weight_data[conv_number]["weight_size"] = 0
    write_computer_adress = computer_address  # write_computer_adress是这个节点对应的写地址
    all_write_computer_adress = []
    opertor_index = 0  # opertor_index代表第几个操作,为了确定写入地址,把第二层当作起始地址,第一层0开始,
    for m in range(1, len(data)):
        str_data = data[m]
        if 'conv2d' in data[m] and 'channels' in data[m]:
            if ((m+4)<(len(data)-1) and 'leak' in data[m+4]) or ((m+3)<(len(data)-1) and 'leak' in data[m+3]):
                isleak = 0     #0istrue
            else:
                isleak = 1   #1isfalse
            net_index += 1
            conv_index = str_data.find('channels') + len('channels') + 1
            strides_index = str_data.find('strides') + len('strides') + 2
            padding_index = str_data.find('padding') + len('padding') + 2
            kernel_size_index = str_data.find('kernel_size') + len('kernel_size') + 2
            weight_index = str_data.find('weight') - 2
            weight_name = ''
            out_node_conv = ""
            in_node_conv = ''
            in_conv_index = str_data.find('conv2d') + len('conv2d') + 1
            for index in range(len(str_data)):
                if str_data[in_conv_index] != ',':
                    in_node_conv += str_data[in_conv_index]
                    in_conv_index = in_conv_index + 1
                else:
                    break
            for out_conv_index in range(1, len(data)):
                if str_data[out_conv_index] == ' ':
                    continue
                elif str_data[out_conv_index] != '=':
                    out_node_conv += str_data[out_conv_index]
                else:
                    break
            for w_i in range(len(str_data)):
                if str_data[weight_index] != '%':
                    weight_name += str_data[weight_index]
                    weight_index = weight_index - 1
                else:
                    break
            weight_name = weight_name[::-1]
            chanels_num = ''
            kernel_size_num = ''
            strides_num = str_data[strides_index]
            kernel_size_num = str_data[kernel_size_index]
            if str_data.find('strides') == -1:
                strides_num = 1
            for index in range(len(str_data)):
                if str_data[conv_index].isdigit():
                    chanels_num += str_data[conv_index]
                    conv_index = conv_index + 1
                else:
                    break
            if "input" not in in_node_conv:
                input_shape = data_dict[in_node_conv]["output_shape"]

            output_shape_picture = int(
                (input_shape[2] - int(kernel_size_num) + 2 * int(str_data[padding_index])) / int(strides_num)) + 1
            output_shape = [input_shape[0], int(chanels_num), output_shape_picture, output_shape_picture]
            data_dict[out_node_conv] = {}
            if (output_shape[1]%8)!=0:
                output_shape[1] = (int(output_shape[1] / 8) + 1) * 8
            weight_size = (input_shape[1] * output_shape[1] * int(kernel_size_num) * int(kernel_size_num))
            weight_size += ((output_shape[1]) * 3 * 4)
            if int(kernel_size_num) == 3:
                block = (output_shape[1] * input_shape[1]) / limit_size33
                block = math.ceil(block)  # 向上取整
                # print(block)
            elif int(kernel_size_num) == 1:
                block = (output_shape[1] * input_shape[1]) / limit_size11
                block = math.ceil(block)  # 向上取整
            if "input" not in in_node_conv:
                input_shape = data_dict[in_node_conv]["output_shape"]

                if conv_number == 1:
                    all_weight_data[conv_number] = {}
                    all_weight_data[conv_number]["address"] = weight_address
                    all_weight_data[conv_number]["weight_size"] = weight_size
                else:

                    # weight_size为权重的数量
                    all_weight_data[conv_number] = {}
                    all_weight_data[conv_number]["address"] = all_weight_data[conv_number - 1]["address"] + \
                                                              all_weight_data[conv_number - 1]["weight_size"]
                    all_weight_data[conv_number]["weight_size"] = weight_size
                # all_write_computer_adress[opertor_index] = []
                #
                # for block_index in range(block):
                #     all_write_computer_adress[opertor_index].append(computer_address + opertor_index * add_write_address)
                # if block==4:
                #     print('%x'%computer_address)
                #     print('-----------------')
                #     print('%x'%(computer_address + (2 * block - 1) * add_write_address))
                all_write_computer_adress.append(computer_address + (2 * block - 1) * add_write_address)
                computer_address = computer_address + (2 * block - 1) * add_write_address

            else:
                # all_write_computer_adress[opertor_index] = []
                #
                # for block_index in range(block):
                #     all_write_computer_adress[opertor_index].append(computer_address)
                all_write_computer_adress.append(computer_address)
            data_dict[out_node_conv]['op'] = 'conv2d'
            data_dict[out_node_conv]['weight_name'] = weight_name
            data_dict[out_node_conv]['input_node'] = in_node_conv
            data_dict[out_node_conv]['input_shape'] = input_shape
            data_dict[out_node_conv]['output_shape'] = output_shape
            data_dict[out_node_conv]['channels'] = chanels_num
            data_dict[out_node_conv]['strides'] = int(strides_num)
            data_dict[out_node_conv]['padding'] = int(str_data[padding_index])
            data_dict[out_node_conv]['kernel_size'] = int(kernel_size_num)
            data_dict[out_node_conv]['isleak'] = isleak
            data_dict[out_node_conv]['write_compute_address'] = int(all_write_computer_adress[opertor_index])
            opertor_index += 1
            conv_number += 1
        elif 'leaky' in data[m]:
            leak_alpha = ''
            out_node_conv = ""
            in_node_conv = ""
            for out_conv_index in range(1, len(data)):
                if str_data[out_conv_index] == ' ':
                    continue
                elif str_data[out_conv_index] != '=':
                    out_node_conv += str_data[out_conv_index]
                else:
                    break
            leak_index = str_data.find('alpha') + len('alpha') + 1
            for index in range(len(str_data)):
                if leak_index <= len(str_data) - 1 and str_data[leak_index].isalpha():
                    break
                elif leak_index <= len(str_data) - 1 and str_data[leak_index].isalpha() == False:
                    leak_alpha += str_data[leak_index]
                    leak_index = leak_index + 1
            in_conv_index = str_data.find('leaky_relu') + len('leaky_relu') + 1
            for index in range(len(str_data)):
                if str_data[in_conv_index] != ',':
                    in_node_conv += str_data[in_conv_index]
                    in_conv_index = in_conv_index + 1
                else:
                    break

            input_shape = data_dict[in_node_conv]["output_shape"]
            data_dict[out_node_conv] = {}
            data_dict[out_node_conv]['op'] = 'leaky_relu'
            data_dict[out_node_conv]['leak_alpha'] = leak_alpha
            data_dict[out_node_conv]['input_node'] = in_node_conv
            data_dict[out_node_conv]['input_shape'] = input_shape
            data_dict[out_node_conv]['output_shape'] = input_shape

            # data_dict[out_node_conv]['name'] = weight_name
        elif 'bias_add' in data[m]:
            out_node_conv = ""
            in_node_conv = ""

            for out_conv_index in range(1, len(data)):
                if str_data[out_conv_index] == ' ':
                    continue
                elif str_data[out_conv_index] != '=':
                    out_node_conv += str_data[out_conv_index]
                else:
                    break
            in_conv_index = str_data.find('bias_add') + len('bias_add') + 1
            for index in range(len(str_data)):
                if str_data[in_conv_index] != ',':
                    in_node_conv += str_data[in_conv_index]
                    in_conv_index = in_conv_index + 1
                else:
                    break
            input_shape = data_dict[in_node_conv]["output_shape"]
            data_dict[out_node_conv] = {}
            data_dict[out_node_conv]['op'] = 'bias_add'
            data_dict[out_node_conv]['input_node'] = in_node_conv
            data_dict[out_node_conv]['input_shape'] = input_shape
            data_dict[out_node_conv]['output_shape'] = input_shape
        elif 'batch_norm' in data[m]:
            out_node_conv = ""
            in_node_conv = ""
            for out_conv_index in range(1, len(data)):
                if str_data[out_conv_index] == ' ':
                    continue
                elif str_data[out_conv_index] != '=':
                    out_node_conv += str_data[out_conv_index]
                else:
                    break
            in_conv_index = str_data.find('batch_norm') + len('batch_norm') + 1
            for index in range(len(str_data)):
                if str_data[in_conv_index] != ',':
                    in_node_conv += str_data[in_conv_index]
                    in_conv_index = in_conv_index + 1
                else:
                    break
            input_shape = data_dict[in_node_conv]["output_shape"]
            data_dict[out_node_conv] = {}
            data_dict[out_node_conv]['op'] = 'batch_norm'
            data_dict[out_node_conv]['input_node'] = in_node_conv
            data_dict[out_node_conv]['input_shape'] = input_shape
            data_dict[out_node_conv]['output_shape'] = input_shape

        elif re.findall('%(.+?)= %(.+?);', data[m]) != []:
            out_data = re.findall('%(.+?).= %(.+?);', data[m])
            out_node_conv = '%' + out_data[0][0]
            in_node_conv = out_data[0][1]
            data_dict[out_node_conv] = {}
            data_dict[out_node_conv]['op'] = 'and'
            in_node_conv = '%' + in_node_conv
            in_node_conv1 = in_node_conv.split('.', 2)[0]
            in_node_conv2 = in_node_conv.split('.', 2)[1]
            if data_dict[in_node_conv1]["op"] == 'batch_norm':
                data_dict[out_node_conv]['input_node'] = in_node_conv1
                input_shape = data_dict[in_node_conv1]["output_shape"]
                data_dict[out_node_conv]['input_shape'] = input_shape
                data_dict[out_node_conv]['output_shape'] = input_shape
            elif data_dict[in_node_conv1]["op"] == 'add':
                in_node_conv2 = int(in_node_conv2)
                in_node_conv = data_dict[in_node_conv1]["input_node"][in_node_conv2]
                data_dict[out_node_conv]['input_node'] = in_node_conv
                input_shape = data_dict[in_node_conv]["output_shape"]
                data_dict[out_node_conv]['input_shape'] = input_shape
                data_dict[out_node_conv]['output_shape'] = input_shape


        elif re.findall("%(.+?).=.\((.+?)\);", data[m]) != []:  # %110 = (%108, %109);
            out_data = re.findall("%(.+?).=.\((.+?)\);", data[m])
            out_node_conv = '%' + out_data[0][0]
            in_node_conv = out_data[0][1]
            data_dict[out_node_conv] = {}
            data_dict[out_node_conv]['op'] = 'add'
            add_input = in_node_conv.split(', ', 2)
            data_dict[out_node_conv]['input_node'] = add_input
            data_dict[out_node_conv]['input_shape'] = data_dict[add_input[0]]["output_shape"], data_dict[add_input[1]][
                "output_shape"]
            data_dict[out_node_conv]['output_shape'] = data_dict[add_input[0]]["output_shape"], data_dict[add_input[1]][
                "output_shape"]

        elif 'concatenate' in data[m]:
            out_node_conv = ""
            in_node_conv = ""
            for out_conv_index in range(1, len(data)):
                if str_data[out_conv_index] == ' ':
                    continue
                elif str_data[out_conv_index] != '=':
                    out_node_conv += str_data[out_conv_index]
                else:
                    break
            in_conv_index = str_data.find('concatenate') + len('concatenate') + 1
            for index in range(len(str_data)):
                if str_data[in_conv_index] != ',':
                    in_node_conv += str_data[in_conv_index]
                    in_conv_index = in_conv_index + 1
                else:
                    break
            new_cat_name = cat_name["scale"][cat_index]
            cat_index += 1
            input_shape = data_dict[in_node_conv]["output_shape"]
            output_shape = (
                input_shape[0][0], input_shape[0][1] + input_shape[1][1], input_shape[0][2], input_shape[0][3])
            new_cat_name = new_cat_name.split('.scale', 2)[0]
            # all_write_computer_adress[opertor_index] = []
            #
            # all_write_computer_adress[opertor_index].append(computer_address + opertor_index * add_write_address)
            all_write_computer_adress.append(computer_address + add_write_address)
            computer_address = computer_address + add_write_address

            data_dict[out_node_conv] = {}
            data_dict[out_node_conv]['op'] = 'concatenate'
            data_dict[out_node_conv]['input_node'] = in_node_conv
            data_dict[out_node_conv]['cat_name'] = new_cat_name
            data_dict[out_node_conv]['input_shape'] = input_shape
            data_dict[out_node_conv]['output_shape'] = output_shape
            data_dict[out_node_conv]['write_compute_address'] = int(all_write_computer_adress[opertor_index])

            opertor_index += 1

        elif 'max_pool2d' in data[m]:
            out_node_conv = ""
            in_node_conv = ""
            strides_index = str_data.find('strides') + len('strides') + 2
            padding_index = str_data.find('padding') + len('padding') + 2
            strides_num = str_data[strides_index]
            if str_data.find('strides') == -1:
                strides_num = 1

            for out_conv_index in range(1, len(data)):
                if str_data[out_conv_index] == ' ':
                    continue
                elif str_data[out_conv_index] != '=':
                    out_node_conv += str_data[out_conv_index]
                else:
                    break
            in_conv_index = str_data.find('max_pool2d') + len('max_pool2d') + 1
            for index in range(len(str_data)):
                if str_data[in_conv_index] != ',':
                    in_node_conv += str_data[in_conv_index]
                    in_conv_index = in_conv_index + 1
                else:
                    break
            input_shape = data_dict[in_node_conv]["output_shape"]
            output_shape_picture = int(
                (input_shape[2] - int(kernel_size_num) + 2 * int(str_data[padding_index])) / int(strides_num)) + 1
            output_shape = [input_shape[0], int(chanels_num), output_shape_picture, output_shape_picture]
            # all_write_computer_adress[opertor_index] = []
            #
            # all_write_computer_adress[opertor_index].append(computer_address + opertor_index * add_write_address)
            all_write_computer_adress.append(computer_address + add_write_address)
            computer_address = computer_address + add_write_address

            data_dict[out_node_conv] = {}
            data_dict[out_node_conv]['op'] = 'max_pool2d'
            data_dict[out_node_conv]['input_node'] = in_node_conv
            data_dict[out_node_conv]['input_shape'] = input_shape
            data_dict[out_node_conv]['output_shape'] = output_shape
            data_dict[out_node_conv]['strides'] = strides_num
            data_dict[out_node_conv]['padding'] = str_data[padding_index]
            data_dict[out_node_conv]['write_compute_address'] = int(all_write_computer_adress[opertor_index])

            opertor_index += 1

        elif 'image.resize' in data[m]:
            out_node_conv = ""
            in_node_conv = ""
            for out_conv_index in range(1, len(data)):
                if str_data[out_conv_index] == ' ':
                    continue
                elif str_data[out_conv_index] != '=':
                    out_node_conv += str_data[out_conv_index]
                else:
                    break
            in_conv_index = str_data.find('image.resize') + len('image.resize') + 1
            for index in range(len(str_data)):
                if str_data[in_conv_index] != ',':
                    in_node_conv += str_data[in_conv_index]
                    in_conv_index = in_conv_index + 1
                else:
                    break
            input_shape = data_dict[in_node_conv]["output_shape"]
            output_shape_hw = re.findall("size=\[(.+?), .+?],", data[m])
            output_shape = (input_shape[0], input_shape[1], int(output_shape_hw[0]), int(output_shape_hw[0]))
            all_write_computer_adress.append(computer_address + add_write_address)
            computer_address = computer_address + add_write_address

            # all_write_computer_adress[opertor_index] = []

            # all_write_computer_adress[opertor_index].append(computer_address + opertor_index * add_write_address)
            data_dict[out_node_conv] = {}
            data_dict[out_node_conv]['op'] = 'image.resize'
            data_dict[out_node_conv]['input_node'] = in_node_conv
            data_dict[out_node_conv]['input_shape'] = input_shape
            data_dict[out_node_conv]['output_shape'] = output_shape
            data_dict[out_node_conv]['write_compute_address'] = int(all_write_computer_adress[opertor_index])

            opertor_index += 1

    return data_dict, model, all_weight_data
