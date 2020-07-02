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
import torch
import torch.nn as nn
# turn np GAN param into torch and put them in nerg.state_dict
# netg.load_state_dict
# Match the parameters and the input.
# See the result is the iamge look good
#%%
import collections
npGANparam = [param.get_value() for param in gen_params]
torchGANparam = [torch.from_numpy(param) for param in npGANparam]
names = ["0.weight", "1.weight", "1.bias", "3.weight", "4.weight", "4.bias", "6.weight", "7.weight", "7.bias", "9.weight", "10.weight", "10.bias", "12.weight", "13.weight", "13.bias", "15.weight", "16.weight", "16.bias", "18.weight"]
state_dict = {names[name]: param for name, param in enumerate(torchGANparam)}
#%%
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
#%%
nz = 256 # input tensor size
ngf = 128 # Size of feature maps in discriminator
nc = 3 # number of color channels in the input images
img_size = 32 # 32 * 32* 3
netg = nn.Sequential(

    nn.Linear(img_size*nz, nz, bias=False), #8192 * 256
    #Reshape((-1, ngf * 4, 4, 4)),
    nn.BatchNorm2d(4*4*4*ngf, affine=True, track_running_stats=False),
    nn.ReLU(True),



    #added
    nn.ConvTranspose2d(ngf*4, ngf * 4, 3, 1, 0, bias=False),
    nn.BatchNorm2d(ngf * 4,affine=True, track_running_stats=False),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf*4,ngf*2,3,1,0,bias=False),
    nn.BatchNorm2d(ngf*2,affine=True, track_running_stats=False),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf*2,ngf*2,3,2,1,bias=False),
    nn.BatchNorm2d(ngf*2,affine=True, track_running_stats=False),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf*2,ngf,3,2,1,bias=False),
    nn.BatchNorm2d(ngf,affine=True, track_running_stats=False),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf,ngf,3,2,1,bias=False),
    nn.BatchNorm2d(ngf,affine=True, track_running_stats=False),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf,nc,3,2,1,bias=False),
    nn.Tanh()
)
#%%
SD = netg.state_dict()
for name in state_dict.keys():
    print(name, state_dict[name].shape)
for name in SD.keys():
    print(name, SD[name].shape)
#%%
netg.load_state_dict(state_dict)
#%%
import torch as t
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
from torch.optim import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

class Config:
    lr = 0.00005
    image_size = 32
    image_size2 = 32
    nz = 256 # noise dimension
    ngf = 128
    nc = 3
    ndf = 64  # discriminative channel
    beta1 = 0.5
    batch_size = 32
    max_epoch = 50  # =1 when debug
    workers = 2
    gpu = True  # use gpu or not
    clamp_num = 0.01  # WGAN clip gradient


opt = Config()
#%%
#optimizerD = Adam(netd.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
optimizerG = Adam(netg.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))

# criterion
# criterion = nn.BCELoss


fix_noise = Variable(t.FloatTensor(0,1,opt.image_size,8192).normal_(0,1))
if opt.gpu:
    fix_noise = fix_noise.cuda()
   # netd.cuda()
    netg.cuda()
    # criterion.cuda() # it's a good habit

fake_u = netg(fix_noise)
#%%
imgs = make_grid(fake_u.data * 0.5 + 0.5).cpu()  # CHW
plt.imshow(imgs.permute(1, 2, 0).numpy())  # HWC
plt.show()