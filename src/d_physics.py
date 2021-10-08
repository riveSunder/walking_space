import os


import numpy as np
from autograd import numpy as anp
from autograd import grad

import skimage
import skimage.io as sio
import matplotlib.pyplot as plt


def get_data(mode=0, batch_size=128, stretch=1.):

    
    x = np.random.rand(batch_size, 1) * stretch
    
    if mode == 0:
        y = np.sign(np.random.randn(batch_size))[:, np.newaxis] * np.sqrt(x)
        
    elif mode == 1:
        y = -1.0 * np.sqrt(x)
        
    elif mode == 2:
        y = np.sqrt(x)
        
    return x, y

def forward_nn(x, layers, biases):
    
    #activation = lambda x: x * (x > 0) + x * 0.1 * (x < 0)
    #activation = lambda x: x**2
    activation = lambda x: anp.sin(x**2)
    #activation = lambda x: anp.tanh(x)
    
    for layer, bias in zip(layers, biases):
        
        x = activation(anp.matmul(x, layer)) + bias
        
    return x


def forward_nn_loss(x, y_target, layers, biases):
        
    prediction = forward_nn(x, layers, biases)
    
    loss = anp.mean(anp.abs(y_target - prediction)**2)
    
    return loss


    
def get_layers(dim_x=1, dim_h=32, dim_y=1, number_h=1):
    
    layers = []
    biases = []
    
    layers.append(30 / (dim_x * dim_h) * np.random.randn(dim_x, dim_h))
    biases.append(np.random.randn(dim_h))
    

    for ii in range(number_h):
        layers.append(30 / (dim_h**2) * np.random.randn(dim_h, dim_h))
        biases.append(np.random.randn(dim_h))

    layers.append(30 / (dim_h * dim_y) * np.random.randn(dim_h, dim_y))
    biases.append(np.random.randn(dim_y))
    
    return layers, biases

def train_nn(layers, biases,  x, y, max_steps=2000, lr=3e-4):
    
    for ii in range(max_steps):


        if ii % (max_steps // 10) == 0:

            loss = forward_nn_loss(x, y, layers, biases)

            print(f"loss at step {ii} = {loss:.4}")

        grad_layers, grad_biases = get_nn_grad(x, y, layers, biases)

        for params, grads in zip(layers, grad_layers):
            params -=  lr * grads

        for params, grads in zip(biases, grad_biases):
            params -=  lr * grads
            
    return layers, biases

def forward_dp_loss(x, layers, biases):
        
    prediction = forward_nn(x, layers, biases)
    
    loss = anp.mean(anp.abs(x - prediction**2)**2)
    
    return loss

get_dp_grad = grad(forward_dp_loss, argnum=(1,2))

def train_dp(layers, biases, x, y, max_steps=2000, lr=1e-4):
    
    for ii in range(max_steps):

        if ii % (max_steps // 10) == 0:

            loss = forward_dp_loss(x, layers, biases)

            print(f"loss at step {ii} = {loss:.4}")

        grad_layers, grad_biases = get_dp_grad(x, layers, biases)

        for params, grads in zip(layers, grad_layers):
            params -=  lr * grads

        for params, grads in zip(biases, grad_biases):
            params -=  lr * grads
            
    return layers, biases

def bootstrap(layers_fn, data_fn, num_straps=3):
    
    x, y = data_fn(batch_size=256)
    
    l_layers = []
    l_biases = []
    
    for ii in range(num_straps):
        print(f"begin training bootstrap {ii}")
        l, b = layers_fn()
        
        train_dp(l, b, x, y, max_steps=3000)
        
        l_layers.append(l)
        l_biases.append(b)
    
    val_x, val_y = data_fn(batch_size=512, stretch=2.0)
    
    preds_x = None
    
    for jj in range(num_straps):
        
        preds_x = forward_nn(val_x, l_layers[jj], l_biases[jj]) if preds_x is None \
                else np.append(preds_x, forward_nn(val_x, l_layers[jj], l_biases[jj]), axis=-1)
        
    # The np.sign here is a little bit cheating... ensures that all predictions refer to same mode
    pred_x = np.mean(preds_x * np.sign(preds_x), axis=-1)
    
    std_dev_pred = np.std(preds_x* np.sign(preds_x), axis=-1)
        
    return val_x, val_y, pred_x, std_dev_pred, l_layers, l_biases    

def plot_parabola(x, y, pred, pred_range=None, my_title="Parabola Figure"):
    
    my_cmap = plt.get_cmap("magma")
    fig = plt.figure(figsize=(8,8))
    
    plt.scatter(x, y, color=my_cmap(50), label="target")
    plt.scatter(x, pred, color=my_cmap(150), label="prediction")
    plt.plot([1., 1.0000000001], [-2.50, 2.50], "--", alpha=0.45, \
             color=[0,0,0], label="training boundary")
    
    if pred_range is not None:
        #import pdb; pdb.set_trace()
        x = x.squeeze()
        pred = pred.squeeze()
        pred_range = pred_range.squeeze()
        x_args = np.argsort(x)
        plt.fill_between(x[x_args], (pred+pred_range)[x_args], \
                         (pred-pred_range)[x_args], alpha=0.5, color=my_cmap(150))
        
    plt.legend(fontsize=20)
    plt.title(my_title, fontsize=24)
    plt.show()

get_nn_grad = grad(forward_nn_loss, argnum=(2,3))
