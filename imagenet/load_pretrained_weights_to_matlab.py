import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
import theano
import theano.tensor as T
#from theano.sandbox.cuda.dnn import dnn_conv

# from lib import costs
# from lib import inits
# from lib import updates
# from lib import activations
# from lib.vis import color_grid_vis
# from lib.rng import py_rng, np_rng
# from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
# from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX, intX
#from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from sklearn.externals import joblib
#%%
"""
This example loads the 32x32 imagenet model used in the paper,
generates 400 random samples, and sorts them according to the
discriminator's probability of being real and renders them to
the file samples.png
"""
model_path = 'C:/Users/zhanq/OneDrive - Washington University in St. Louis/GitHub/dcgan_code/models/imagenet_gan_pretrain_128f_relu_lrelu_7l_3x3_256z/'
gen_params = [sharedX(p) for p in joblib.load(model_path+'30_gen_params.jl')]
discrim_params = [sharedX(p) for p in joblib.load(model_path+'30_discrim_params.jl')]

#%%
npGANparam = [param.get_value() for param in gen_params]
for paramname in gen_params:
    print(paramname,paramname.container.data.shape)

#%%
# Python3 program to Convert a
# list to dictionary

def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

#%%
import torch.utils.model_zoo as model_zoo
import torch.onnx
from scipy.io import loadmat
dcGANmatW = loadmat(r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\GitHub\Matlab-GAN\DCGAN\DCGAN_paramsGen.mat", struct_as_record=False, squeeze_me=True)["paramsGen"]
#state_dict = model_zoo.load_url(model_url, progress=False)
state_dict = Convert(gen_params)
#%%
for paramname, matpname in zip(state_dict.keys(), dcGANmatW._fieldnames):
    matweight = dcGANmatW.__getattribute__(matpname)
    matweightArray = matweight[0][3].tolist()
    print(matweightArray)
    trcweight = state_dict[paramname]
    trcmatW = torch.from_numpy(matweight)
    if len(trcweight.shape) == 4: # conv2 layer
        trcmatW = trcmatW.permute(3, 2, 0, 1)
        # matlab weight shape [FilterSize(1),FilterSize(2),NumChannels,NumFilters]
        # torch weight shape `[out_channels(NumFilters), in_channels(NumChannels), kernel_size ]`
    elif len(trcweight.shape) == 2: # fc layer matmul weight is 2d
        pass
    assert trcmatW.shape == trcweight.shape
    state_dict[paramname] = trcmatW
    print(paramname, matpname, trcweight.shape, matweight.shape, )

