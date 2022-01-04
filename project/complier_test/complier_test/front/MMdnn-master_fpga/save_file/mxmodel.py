import mxnet as mx
import numpy as np
import math

# mxnet-cpu only support channel first, default convert the model and weight as channel first

def RefactorModel():

    input           = mx.sym.var('input')
    node121         = mx.sym.Convolution(data=input, kernel=(3, 3), stride=(2, 2), dilate = (1, 1), pad=(1, 1), num_filter = 8, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node121')
    node122         = mx.sym.BatchNorm(data = node121, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node122')
    node123         = mx.sym.Activation(data = node122, act_type = 'relu', name = 'node123')
    node124         = mx.sym.Convolution(data=node123, kernel=(3, 3), stride=(2, 2), dilate = (1, 1), pad=(1, 1), num_filter = 32, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node124')
    node125         = mx.sym.BatchNorm(data = node124, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node125')
    node126         = mx.sym.Activation(data = node125, act_type = 'relu', name = 'node126')
    node127         = mx.sym.Convolution(data=node126, kernel=(3, 3), stride=(2, 2), dilate = (1, 1), pad=(1, 1), num_filter = 64, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node127')
    node128         = mx.sym.BatchNorm(data = node127, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node128')
    node129         = mx.sym.Activation(data = node128, act_type = 'relu', name = 'node129')
    node130         = mx.sym.Convolution(data=node129, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 64, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node130')
    node131         = mx.sym.BatchNorm(data = node130, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node131')
    node132         = mx.sym.Activation(data = node131, act_type = 'relu', name = 'node132')
    node133         = mx.sym.Convolution(data=node132, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 128, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node133')
    node134         = mx.sym.BatchNorm(data = node133, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node134')
    node135         = mx.sym.Activation(data = node134, act_type = 'relu', name = 'node135')
    node136         = mx.sym.Convolution(data=node135, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node136')
    node137         = mx.sym.BatchNorm(data = node136, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node137')
    node138         = mx.sym.Activation(data = node137, act_type = 'relu', name = 'node138')
    node139         = mx.sym.Convolution(data=node138, kernel=(3, 3), stride=(2, 2), dilate = (1, 1), pad=(1, 1), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node139')
    node140         = mx.sym.BatchNorm(data = node139, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node140')
    node141         = mx.sym.Activation(data = node140, act_type = 'relu', name = 'node141')
    node142         = mx.sym.Convolution(data=node141, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node142')
    node143         = mx.sym.BatchNorm(data = node142, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node143')
    node144         = mx.sym.Activation(data = node143, act_type = 'relu', name = 'node144')
    node145         = mx.sym.Convolution(data=node144, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node145')
    node146         = mx.sym.BatchNorm(data = node145, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node146')
    node147         = mx.sym.Activation(data = node146, act_type = 'relu', name = 'node147')
    node148         = mx.sym.Convolution(data=node147, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node148')
    node149         = mx.sym.BatchNorm(data = node148, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node149')
    node150         = mx.sym.Activation(data = node149, act_type = 'relu', name = 'node150')
    node151         = mx.sym.Convolution(data=node150, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node151')
    node152         = mx.sym.BatchNorm(data = node151, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node152')
    node153         = mx.sym.Activation(data = node152, act_type = 'relu', name = 'node153')
    node154         = mx.sym.Convolution(data=node153, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node154')
    node155         = mx.sym.BatchNorm(data = node154, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node155')
    node156         = mx.sym.Activation(data = node155, act_type = 'relu', name = 'node156')
    node157         = mx.sym.Convolution(data=node156, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node157')
    node158         = mx.sym.BatchNorm(data = node157, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node158')
    node159         = mx.sym.Activation(data = node158, act_type = 'relu', name = 'node159')
    node160         = mx.sym.Convolution(data=node159, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 128, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node160')
    node161         = mx.sym.BatchNorm(data = node160, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node161')
    node162         = mx.sym.Activation(data = node161, act_type = 'relu', name = 'node162')
    node163         = mx.sym.Convolution(data=node162, kernel=(3, 3), stride=(2, 2), dilate = (1, 1), pad=(1, 1), num_filter = 128, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node163')
    node164         = mx.sym.BatchNorm(data = node163, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node164')
    node165         = mx.sym.Activation(data = node164, act_type = 'relu', name = 'node165')
    node166         = mx.sym.Convolution(data=node165, kernel=(3, 3), stride=(1, 1), dilate = (1, 1), pad=(1, 1), num_filter = 128, num_group = 1, no_bias = False, layout = 'NCHW', name = 'node166')
    node167         = mx.sym.BatchNorm(data = node166, axis = 1, eps = 9.999999747378752e-06, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'node167')
    node168         = mx.sym.Activation(data = node167, act_type = 'relu', name = 'node168')
    node169         = mx.sym.Convolution(data=node168, kernel=(1, 1), stride=(1, 1), dilate = (1, 1), pad=(0, 0), num_filter = 25, num_group = 1, no_bias = True, layout = 'NCHW', name = 'node169')
    
    # if a GPU is available, change mx.cpu() to mx.gpu()
    model           = mx.mod.Module(symbol = node169, context = mx.cpu(), data_names = ['input'])
    return model

def deploy_weight(model, weight_file):

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    arg_params = dict()
    aux_params = dict()
    for weight_name, weight_data in weights_dict.items():
        weight_name = str(weight_name)
        if "moving" in weight_name:
            aux_params[weight_name] = mx.nd.array(weight_data)
        else:
            arg_params[weight_name] = mx.nd.array(weight_data)

    model.bind(for_training = False, data_shapes = [('input', (1, 1, 800, 800))])
    model.set_params(arg_params = arg_params, aux_params = aux_params, allow_missing = True, allow_extra=True)

    return model


from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


def get_image(url, show=False):
    import cv2
    # download and show the image
    fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def predict(model, labels, url):
    # to show the image, change the argument show into True
    img = get_image(url, show = False)
    # compute the predict probabilities
    model.forward(Batch([mx.nd.array(img)]))
    prob = model.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('prbability = %f, class = %s' %(prob[i], labels[i]))


if __name__ == '__main__':
    model = RefactorModel()
    # remember to adjust params path
    model = deploy_weight(model, 'mxnet_inception_v3-0000.params')

    # # call function predict
    # with open('synset.txt', 'r') as f:
    #     labels = [l.rstrip() for l in f]
    # predict(model, labels, 'http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg')
