import sys

sys.path.append('..')

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import costs
from lib import inits
from lib import updates
from lib import activations
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX, intX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from sklearn.externals import joblib
#%%
"""
This example loads the 32x32 imagenet model used in the paper,
generates 400 random samples, and sorts them according to the
discriminator's probability of being real and renders them to
the file samples.png
"""

nz = 256
nc = 3
npx = 32
ngf = 128
ndf = 128

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
#%%
model_path = 'C:/Users/zhanq/OneDrive - Washington University in St. Louis/GitHub/dcgan_code/models/imagenet_gan_pretrain_128f_relu_lrelu_7l_3x3_256z/'
gen_params = [sharedX(p) for p in joblib.load(model_path+'30_gen_params.jl')]

discrim_params = [sharedX(p) for p in joblib.load(model_path+'30_discrim_params.jl')]
#%%
def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6, g6, b6, wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*4, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(1, 1)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(1, 1), border_mode=(1, 1)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(1, 1)), g=g4, b=b4))
    h5 = relu(batchnorm(deconv(h4, w5, subsample=(1, 1), border_mode=(1, 1)), g=g5, b=b5))
    h6 = relu(batchnorm(deconv(h5, w6, subsample=(2, 2), border_mode=(1, 1)), g=g6, b=b6))
    x = tanh(deconv(h6, wx, subsample=(1, 1), border_mode=(1, 1)))
    return x
#%%
def discrim(X, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6, g6, b6, wy):
    h = lrelu(dnn_conv(X, w, subsample=(1, 1), border_mode=(1, 1)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(1, 1)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(1, 1), border_mode=(1, 1)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(1, 1)), g=g4, b=b4))
    h5 = lrelu(batchnorm(dnn_conv(h4, w5, subsample=(1, 1), border_mode=(1, 1)), g=g5, b=b5))
    h6 = lrelu(batchnorm(dnn_conv(h5, w6, subsample=(2, 2), border_mode=(1, 1)), g=g6, b=b6))
    h6 = T.flatten(h6, 2)
    y = sigmoid(T.dot(h6, wy))
    return y

def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+1.)/2.
    return X
#%%
npGANparam = [param.get_value() for param in gen_params]

gen_params32 = [param.get_value().astype('float32') for param in gen_params]
#%%
#%%
Z = T.matrix()
Zz = Z.astype('float32')
X = T.tensor4()
Xx = X.astype('float32')
gX = gen(Zz, *gen_params32)

#%%
dX = discrim(Xx, *discrim_params)

_gen = theano.function([Zz], gX)
_discrim = theano.function([Xx], dX)

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(400, 256)))
samples = _gen(sample_zmb)
scores = _discrim(samples)
sort = np.argsort(scores.flatten())[::-1]
samples = samples[sort]
color_grid_vis(inverse_transform(samples), (20, 20), 'samples.png')
#%%
#%%
def calculate_b_u_b_s(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    if X.ndim == 4:
        if u is not None and s is not None:
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            b_u = np.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            b_s = np.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            b_u = (1. - a)*0. + a*b_u
            b_s = (1. - a)*1. + a*b_s
    elif X.ndim == 2:
        if u is None and s is None:
            b_u = np.mean(X, axis=0)
            b_s = np.mean(T.sqr(X - u), axis=0)
        if a != 1:
            b_u = (1. - a)*0. + a*u
            b_s = (1. - a)*1. + a*s
    return b_u, b_s
#%%
[w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6, g6, b6, wx] = [param for param in gen_params2]
#%%
import tensorflow as tf
Z = np.float32(np.random.rand(256,256))
h = relu(batchnorm(np.matmul(Z, w), g=g, b=b))
h = h.reshape((h.shape[0], ngf*4, 4, 4))
h = h.astype('float32')
#%%
b_u = np.mean(b2, axis=0)
b_s = np.mean(np.square(b2 - b_u), axis=0)