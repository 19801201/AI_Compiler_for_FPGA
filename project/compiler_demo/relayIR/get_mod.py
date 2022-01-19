
from tvm import relay
from yolov4_tiny_quan import YoloBody

import torch
import torchvision

def get_mod(input_shape):
   input_data = torch.randn(input_shape)
   model_float_path = '/home/anyilin/compiler/1.7-LEAKY_RELU_01_Epoch14-Total_Loss0.5940-Val_Loss0.9286_714.pth'
   model = YoloBody(3, 1)
   model.eval()
   state_dict = torch.load(model_float_path, map_location='cpu')
   model.load_state_dict(state_dict)

   model = torch.jit.trace(model, input_data).eval()
   mod, params = relay.frontend.from_pytorch(model, [("input",input_shape)])
   # print(type(mod["main"]))
   print(mod["main"])
   mod = relay.transform.EliminateCommonSubexpr(fskip=None)(mod)   #消除子表达式
   mod = relay.transform.DeadCodeElimination(inline_once=False)(mod) #删除没有任何用户的表达式（死代码）
   # mod = relay.transform.FoldConstant()(mod)#?在Relay程序中折叠常量表达式。
   data_old = str(mod["main"])
   data_list = []
   data = ''
   for index in range(len(data_old)):
      if data_old[index] == '\n':
         data_list.append(data)
         data = ''
      else:
         data += data_old[index]
   # print(data_list)
   # exit()
   return data_list
