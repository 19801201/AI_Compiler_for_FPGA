#!/usr/bin/env python
# coding: utf-8

# In[13]:


tmp1 = []
tmp2 = []
def read_para():
    file_name = "para.txt"
    dataset = []
    file = open(file_name,mode='r')
    for line in file:
        line = line.split()
        dataset.append(line)
    file.close()
    print(dataset)
    return dataset
def write_para(): #### 写到这了
    file1 = open('instruction_FPGA.bin','wb')
    weight_addr_list = []
    weight_addr_list.append(to_inst(0))

    dataset = read_para()
    for i,line in enumerate(dataset):
        # for i in range(12):
        channel_in    = int(line[0])
        channel_out   = int(line[1])
        kernal_size   = int(line[2])
        block_ram_addr= int(line[3].rjust(16,'0')) #? 大ram 参数加载到ddr 最大ram的首地址  一直0000_0000
        zero_point1   = int(line[4].rjust(8,'0')) #? 补零的0 00000000
        zero_point2   = int(line[5].rjust(8,'0'))#? 量化加的零点
        en_cin        = int(line[6]) #？判断是不是第一层补成8通道 需要是1 0
        padding       = int(line[7]) 
        stride        = int(line[8])
        zero_num      = int(line[9].rjust(3,'0')) #5*5历史问题 一圈0是1  不添0是0
        row_num_in    = int(line[10])
        row_num_out   = int(line[10])
        real_out = int(line[11])
        # reg1
        reg1 = weight_addr_list[i]
        weight_size2 = weight_size(channel_in,channel_out,kernal_size)
        reg2 = weight_size2
        tmp1.append(weight_addr_list[i])
        tmp2.append(weight_size2)
        para_num = gen_para_weight_bias_num(channel_in,channel_out,kernal_size )
        para_reg4 ,para_reg5 =gen_para_instruction(block_ram_addr,para_num)

        reg3 = para_reg4
        reg4 = para_reg5
        # reg5 shuru daxiao
        reg5 = input_img_size(channel_in,row_num_in)
        # reg6 dizhi
        if i%2 ==0:
            reg6=write_addr
        else:
            reg6=read_addr
        
        # reg7 daxiao
        reg7 = output_img_size(channel_out,real_out)
        # reg8 dizhi
        if i%2 ==0:
            reg8=read_addr
        else:
            reg8=write_addr     
        weight_single_num, bias_num=gen_cu_weight_bias_num(channel_in,channel_out,kernal_size,)
        cu_reg5,cu_reg4 = gen_cu_instruction2(en_cin,padding,stride,zero_num,row_num_in,channel_in,row_num_out,channel_out )

        cu_reg7,cu_reg6 = gen_cu_instruction1(zero_point1,zero_point2,weight_single_num,block_ram_addr,bias_num)

        reg9 = cu_reg4
        reg10 = cu_reg5
        reg11 = cu_reg7
        reg12 = cu_reg6

        temp_next_w_addr =     int(weight_size2,16) + int(weight_addr_list[i],16)
        temp_next_w_addr = to_inst(temp_next_w_addr)
        weight_addr_list.append(temp_next_w_addr)
        reg1,reg2,reg3,reg4,reg5,reg6,reg7,reg8,reg9,reg10,reg11,reg12 = reg1.encode(),reg2.encode(),reg3.encode(),reg4.encode(),reg5.encode(),reg6.encode(),reg7.encode(),reg8.encode(),reg9.encode(),reg10.encode(),reg11.encode(),reg12.encode()
        file1.write(reg1)
        file1.write(reg2)
        file1.write(reg3)
        file1.write(reg4)
        file1.write(reg5)
        file1.write(reg6)
        file1.write(reg7)
        file1.write(reg8)
        file1.write(reg9)
        file1.write(reg10)
        file1.write(reg11)
        file1.write(reg12)
        
        #         file1.write(reg1+'\n'+reg2+'\n'+reg3+'\n'+reg4+'\n'+reg5+'\n'+reg6+'\n'+reg7+'\n'+reg8+'\n'+reg9+'\n'+reg10+'\n'+reg11+'\n'+reg12+'\n')
        print("write ok ")   
        
        
def get_para():
    weight_addr_list = []
    weight_addr_list.append(to_inst(0))

    dataset = read_para()
    for i,line in enumerate(dataset):
        # for i in range(12):
        channel_in    = int(line[0])
        channel_out   = int(line[1])
        kernal_size   = int(line[2])
        block_ram_addr= int(line[3].rjust(16,'0')) #? 大ram 参数加载到ddr 最大ram的首地址  一直0000_0000
        zero_point1   = int(line[4].rjust(8,'0')) #? 补零的0 00000000
        zero_point2   = int(line[5].rjust(8,'0'))#? 量化加的零点
        en_cin        = int(line[6]) #？判断是不是第一层补成8通道 需要是1 0
        padding       = int(line[7]) 
        stride        = int(line[8])
        zero_num      = int(line[9].rjust(3,'0')) #5*5历史问题 一圈0是1  不添0是0
        row_num_in    = int(line[10])
        row_num_out   = int(line[10])
        real_out = int(line[11])
        # reg1
        print(weight_addr_list[i]) #reg1 ok

        weight_size2 = weight_size(channel_in,channel_out,kernal_size)
        print(weight_size2)#reg2
        tmp1.append(weight_addr_list[i])
        tmp2.append(weight_size2)
        para_num = gen_para_weight_bias_num(channel_in,channel_out,kernal_size )
        para_reg4 ,para_reg5 =gen_para_instruction(block_ram_addr,para_num)
        print(para_reg4) # reg3 
        print(para_reg5) # reg4 

        # reg5 shuru daxiao
        reg5 = input_img_size(channel_in,row_num_in)
        print(reg5)
        # reg6 dizhi
        if i%2 ==0:
            reg6=write_addr
        else:
            reg6=read_addr
        print(reg6)
        
        # reg7 daxiao
        reg7 = output_img_size(channel_out,real_out)
        print(reg7)
        # reg8 dizhi
        if i%2 ==0:
            reg8=read_addr
        else:
            reg8=write_addr     
        print(reg8)
        weight_single_num, bias_num=gen_cu_weight_bias_num(channel_in,channel_out,kernal_size,)
        cu_reg5,cu_reg4 = gen_cu_instruction2(en_cin,padding,stride,zero_num,row_num_in,channel_in,row_num_out,channel_out )
        print(cu_reg4)# reg9
        print(cu_reg5)# reg10

        cu_reg7,cu_reg6 = gen_cu_instruction1(zero_point1,zero_point2,weight_single_num,block_ram_addr,bias_num)
        print(cu_reg7)# reg11
        print(cu_reg6)# reg12


        temp_next_w_addr =     int(weight_size2,16) + int(weight_addr_list[i],16)
        temp_next_w_addr = to_inst(temp_next_w_addr)
        weight_addr_list.append(temp_next_w_addr)
        print("------------------")
#========================para_configuration================================
# channel_in = 4                             
# channel_out= 32                                  
# kernal_size = 9                             
# block_ram_addr = 0b0000000000000000         
# #----------------------------------
# zero_point1 = 0b00000000                    
# zero_point2 = 0b00000000                    
# #-----------------------------------
# en_cin      =0b0                            
# padding     =0b1                         
# stride      =0b0                          
# zero_num    =0b001                       
# row_num_in  =13                                                       
# row_num_out=13                                                      
#=========================================================
#--------------------------------para_instru----------------------------------
write_addr = 'write'
read_addr = 'read'
Instruction4 = 0b00000000000000000000000000000000
Instruction5 = 0b00000000000000000000000000000000
def gen_para_weight_bias_num(channel_in,channel_out,kernal_size ):
    weight_num = (channel_in*channel_out*8*kernal_size)>>8
    bias_num = (3*channel_out*32)>>8
    para_num = weight_num+bias_num
    return para_num
def gen_para_instruction(block_ram_addr,para_num):
    Inst1 = Instruction4|para_num<<16
    Inst2 = Inst1|block_ram_addr
    Inst_len = len(str(hex(Inst2)))
    reg_4_temp  = hex(Inst2)[2:Inst_len]
    reg_4 = reg_4_temp.rjust(8,'0')
    Inst_len1 = len(str(hex(Instruction5)))
    reg_5_temp  = hex(Instruction5)[2:Inst_len1]
    reg_5 = reg_5_temp.rjust(8,'0')
    return reg_4,reg_5

#--------------------------------CU_Instru-------------------------------------
Instruction1 = 0b0000000000000000000000000000000000000000000000000000000000000000
Instruction2 = 0b0000000000000000000000000000000000000000000000000000000000000000
def gen_cu_instruction1(zero_point1,zero_point2,weight_single_num,block_ram_addr,bias_num):
    Inst1 = Instruction1|zero_point1<<56
    Inst2 = Inst1|zero_point2<<48
    Inst3 = Inst2|weight_single_num<<32
    Inst4 = Inst3|block_ram_addr<<16
    Inst5 = Inst4|bias_num<<9
    Inst_len = len(str(hex(Inst5)))
    reg_6_temp = str(hex(Inst5))[Inst_len-8:Inst_len]
    reg_7_temp = str(hex(Inst5))[2:Inst_len-8]
    reg_6 =reg_6_temp.rjust(8,'0')
    reg_7 = reg_7_temp.rjust(8,'0')
    return reg_7,reg_6
def gen_cu_instruction2(en_cin,padding,stride,zero_num,row_num_in,channel_in,row_num_out,channel_out ):
    Inst1 = Instruction2|en_cin<<63
    Inst2 = Inst1|padding<<62
    Inst3 = Inst2|stride<<61
    Inst4 = Inst3|zero_num<<58
    Inst5 = Inst4|row_num_in<<37
    Inst6 = Inst5|channel_in<<22
    Inst7 = Inst6|row_num_out<<11
    Inst8 = Inst7|channel_out
    Inst_len = len(str(hex(Inst8)))
    reg_4_temp = str(hex(Inst8))[Inst_len-8:Inst_len]
    reg_5_temp = str(hex(Inst8))[2:Inst_len-8]
    reg_4 =reg_4_temp.rjust(8,'0')
    reg_5 = reg_5_temp.rjust(8,'0')
    return reg_5,reg_4
def gen_cu_weight_bias_num(channel_in,channel_out,kernal_size ):
    weight_single_num = (channel_in*channel_out*8)>>8
    bias_num = (channel_out*32)>>8
    return weight_single_num,bias_num
#=================================PARA_DMA_NUM_ADDR===============================
PARA_DMA_ADDR = 0b00000000000000000000000000000000
PARA_DMA_NUM  = 0b00000000000000000000000000000000
def gen_para_dma_num(channel_in,channel_out,kernal_size):
    weight_dma_num = channel_in*channel_out*kernal_size
    para_dma_num   = weight_dma_num+channel_out*3*4
    Inst1          = PARA_DMA_ADDR|para_dma_num
    Inst_len       = len(str(hex(Inst1)))
    Inst2_temp     = str(hex(Inst1))[2:Inst_len]
    Inst2          = Inst2_temp.rjust(8,'0')
    Inst_len1      =len(str(hex(PARA_DMA_ADDR)))
    Inst3_temp     =str(hex(PARA_DMA_ADDR))[2:Inst_len1]
    Inst3          =Inst3_temp.rjust(8,'0')
    return Inst3
def gen_para_dma_addr(dma_addr_base,dma_num):
    dma_addr_temp = dma_addr_base+dma_num-1
    dma_addr_len  = len(str(hex(dma_addr_temp)))
    dma_addr_temp1=str(hex(dma_addr_temp))[2:dma_addr_len]
    dma_addr      =dma_addr_temp1.rjust(8,'0')
    print(dma_addr)

def to_inst(size): 
    ### 1 - 00000001
    size = hex(size)
    Inst_len = len(size)
    temp = size[2:Inst_len].rjust(8,'0')
    return temp

def weight_size(channel_in,channel_out,kernal_size):
    size = channel_out*kernal_size*channel_in+3*channel_out*4
    size = hex(size)
    Inst_len = len(size)
    size = size[2:Inst_len].rjust(8,'0')

    return size
def input_img_size(channel_in,row_num_in):
    size = channel_in*row_num_in*row_num_in*4
    size = hex(size)
    Inst_len = len(size)
    size = size[2:Inst_len].rjust(8,'0')

    return size
def output_img_size(channel_out,row_num_out):
    size = channel_out*row_num_out*row_num_out*4
    size = hex(size)
    Inst_len = len(size)
    size = size[2:Inst_len].rjust(8,'0')

    return size
def read_add():
    f1 = open('./tmp_addr.txt')
    addr = f1.read()
    a = to_inst(int(addr))
    f1.close()
    return a
def write_add():
    write_addr = int(tmp1[-1],16)+int(tmp2[-1],16)+100
    write_addr = to_inst(write_addr)
    return write_addr
# if kenelsize = 9 do another thing
if __name__ == "__main__":
    get_para() #打印一下 并算出最小写地址
    # 读写地址
    write_addr = write_add()
    read_addr = read_add()
    # 文件写指令
    write_para()


# In[ ]:





# In[ ]:



# def conv_fpga(reg1,reg2,reg3,reg4,reg5,reg6,reg7,reg8,reg9,reg10,reg11,reg12):
#     fpga_memspace.write("reset", None, None) 
#     fpga_memspace.four_to_one(32,reg1)#0xC
#     fpga_memspace.four_to_one(16,reg2)#0x10
#     fpga_memspace.write("pci_cl_ctrl", 0xA8, 0x00000000)
#     fpga_memspace.four_to_one(80,reg3)#0x1C
#     fpga_memspace.four_to_one(96,reg4)#0x20
#     fpga_memspace.write("control_3x3_p", None,  None)
          
#     while (fpga_memspace.read("pci_cl_ctrl", 0xA4)!=0x0000000F):
#         continue
      
#     fpga_memspace.write("clear", None,  None)
    
#     while (fpga_memspace.read("pci_cl_ctrl", 0xA4)!=0x00000000):
#         continue

#     fpga_memspace.four_to_one(32,reg5)#0xC
#     fpga_memspace.four_to_one(16,reg6)#0x10
#     fpga_memspace.four_to_one(64,reg7)#0x14
#     fpga_memspace.four_to_one(48,reg8)#0x18    

#     fpga_memspace.four_to_one(80,reg9)#0x1C
#     fpga_memspace.four_to_one(96,reg10)#0x20
#     fpga_memspace.four_to_one(112,reg11)#0x24
#     fpga_memspace.four_to_one(128,reg12)#0x28
#     fpga_memspace.write("control_3x3_c", None,  None) 
   
          
#     while (fpga_memspace.read("pci_cl_ctrl", 0xA4)!=0x0000000F):
#         continue
    
#     fpga_memspace.write("clear", None,  None)
#     while (fpga_memspace.read("pci_cl_ctrl", 0xA4)!=0x00000000):
#         continue

# for i in range(num_conv):
#     #读取文件中第i坨指令
#     conv_fpga(reg1,reg2,reg3,reg4,reg5,reg6,reg7,reg8,reg9,reg10,reg11,reg12)


# In[4]:





# In[ ]:




