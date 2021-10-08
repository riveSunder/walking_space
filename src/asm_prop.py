import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import time

import skimage
import skimage.io as sio
import skimage.transform

from PIL import Image

def asm_prop(wavefront, length=32.e-3, wavelength=550.e-9, distance=10.e-3):
        
    if len(wavefront.shape) == 2:
        dim_x, dim_y = wavefront.shape
    elif len(wavefront.shape) == 3:
        number_samples, dim_x, dim_y = wavefront.shape
    else:
        print("only 2D wavefronts or array of 2D wavefronts supported")

    assert dim_x == dim_y, "wavefront should be square"
    px = length / dim_x

    l2 = (1/wavelength)**2
    
    fx = np.linspace(-1/(2*px), 1/(2*px) - 1/(dim_x*px), dim_x)
    fxx, fyy = np.meshgrid(fx,fx)

    q = l2 - fxx**2 - fyy**2
    q[q<0] = 0.0

    h = np.fft.fftshift(np.exp(1.j * 2 * np.pi * distance * np.sqrt(q)))
    
    fd_wavefront = np.fft.fft2(np.fft.fftshift(wavefront)) 
    if len(wavefront.shape) == 3:
        fd_new_wavefront = h[np.newaxis,:,:] * fd_wavefront
        new_wavefront=np.fft.ifftshift(np.fft.ifft2(fd_new_wavefront))[:,:dim_x,:dim_x]
    else:
        fd_new_wavefront = h * fd_wavefront
        new_wavefront = np.fft.ifftshift(np.fft.ifft2(fd_new_wavefront))[:dim_x,:dim_x]


    return new_wavefront

def onn_layer(wavefront, phase_objects, d=100.e-3):

    for ii in range(len(phase_objects)):
        wavefront = asm_prop(wavefront * phase_objects[ii], distance=d)

    return wavefront

def get_loss(wavefront, y_tgt, phase_objects, d=100.e-3):

    img = np.abs(onn_layer(wavefront, phase_objects, d=d))**2
    mse_loss = np.mean( (img - y_tgt)**2 + np.abs(img-y_tgt) )

    return mse_loss

get_grad = grad(get_loss, argnum=2)


def save_as_gif(np_array, filename="my_gif.gif", my_cmap=None):
    
    assert (len(np_array.shape) == 3), "expected n by h by w array"
    
    if my_cmap == None:
        my_cmap = plt.get_cmap("magma")
        
    dim_x, dim_y = np_array.shape[-2], np_array.shape[-1]
    
    im = Image.fromarray((my_cmap(np_array[0])*255).astype("uint8"), "RGBA")

    im.save(f"assets/{filename}", save_all=True, duration=3*np_array.shape[0], loop=0, \
            append_images=[Image.fromarray((my_cmap(img)*255).astype("uint8"), "RGBA") for img in np_array[1:]])

def train_run(tgt_img, zero_pad=True):

    dim = 128
    side_length = 32.e-3
    aperture = 8.e-3
    wavelength = 550.e-9
    k0 = 2*np.pi / wavelength
    dist = 50.e-3

    if zero_pad:
        tgt_img = np.pad(tgt_img, (tgt_img.shape[0], tgt_img.shape[1]))
    # resize target image
    tgt_img = skimage.transform.resize(tgt_img, (dim, dim))
    
    px = side_length / dim

    x = np.linspace(-side_length/2, side_length/2-px, dim)

    xx, yy = np.meshgrid(x,x)
    rr = np.sqrt(xx**2 + yy**2)

    wavefront = np.zeros((dim,dim)) * np.exp(1.j*k0*0.0)
    wavefront[rr <= aperture] = 1.0

    y_tgt = 1.0 * tgt_img / np.max(tgt_img)

    lr = 1e-3
    phase_objects = [np.exp(1.j * np.zeros((128,128)) ) \
            for aa in range(32)]
    losses = []

    training_arrays = []
    smooth_slope = 0.0
    
    for step in range(1024):


        my_grad = get_grad(wavefront, y_tgt, phase_objects, d=dist)

        for params, grads in zip(phase_objects, my_grad):
            params -=  lr * np.exp( -1.j * np.angle(grads))

        loss = get_loss(wavefront, y_tgt, phase_objects,d=dist)
        losses.append(loss)
        img = np.abs(onn_layer(wavefront, phase_objects))**2
        
        if step % 16 == 0:
            print("loss at step {} = {:.2e}, lr={:.3e}".format(step, loss, lr))
        
        training_arrays.append(img/2.0)
        
        if len(losses) > 1:
            smooth_slope = 0.95 * smooth_slope + 0.05 * (losses[-2] - losses[-1])
        
        if smooth_slope < 0.0:
            print("stopping training")
            break
    
    return np.array(training_arrays), losses, tgt_img
