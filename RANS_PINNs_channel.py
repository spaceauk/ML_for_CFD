import os
import math as m
import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn

import scipy.interpolate as int1d
from scipy.sparse import *
from scipy.linalg.lapack  import  dgtsv  as  lapack_dgtsv
from scipy.integrate import cumtrapz, trapz

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import time, sys

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.ticker as ticker

########################################################################################################
#                                               Try PINNs                                              #
########################################################################################################
# Define fluid properties - air taken for this case
rho = 1.225
nu = 0.0000148
# Define NN
numNN = 20      # Number of neurons in each layer
numlayers = 8   # Number of layers in PINN
# Generate NN
class DNN(nn.Module):
    # 1. Initialization
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential()

        # Input (x,y) --> Output (U,V,P,uu,uv,vv)
        # Constants:
        #           - rho (incompressible)
        #           - nu (depends on fluid properties)
        self.net.add_module('Linear_layer_1',nn.Linear(2,numNN))
        self.net.add_module('Tanh_layer_1',nn.Tanh())
        for num in range(2,numlayers):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(numNN,numNN))
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())
        self.net.add_module('Linear_layer_final',nn.Linear(numNN,6))
    # 2. Forward feed
    def forward(self,x):
        return self.net(x)
    # 3. Loss function for PDE
    def loss_pde(self,x):
        y = self.net(x)
        U,V,P,uu,uv,vv = y[:,0:1], y[:,1:2], y[:,2:3], y[:,3:4], y[:,4:5], y[:,5:]
        # Gradients and partial derivatives
        # 1st order derivatives
        dU_g = gradients(U,x)[0]
        U_x, U_y = dU_g[:,:1], dU_g[:,1:]

        dV_g = gradients(V,x)[0]
        V_x, V_y = dV_g[:,:1], dV_g[:,1:]

        dP_g = gradients(P,x)[0]
        P_x, P_y = dP_g[:,:1], dP_g[:,1:]

        duu_g = gradients(uu,x)[0]
        uu_x, uu_y = duu_g[:,:1], duu_g[:,1:]

        duv_g = gradients(uv,x)[0]
        uv_x, uv_y = duv_g[:,:1], duv_g[:,1:]

        dvv_g = gradients(vv,x)[0]
        vv_x, vv_y = dvv_g[:,:1], dvv_g[:,1:]
        # 2nd order derivatives
        dU_xg = gradients(U_x,x)[0]
        U_xx, U_xy = dU_xg[:,:1], dU_xg[:,1:]

        dV_xg = gradients(V_x,x)[0]
        V_xx, V_xy = dV_xg[:,:1], dV_xg[:,1:]

        dU_yg = gradients(U_y,x)[0]
        U_yx, U_yy = dU_yg[:,:1], dU_yg[:,1:]

        dV_yg = gradients(V_y,x)[0]
        V_yx, V_yy = dV_yg[:,:1], dV_yg[:,1:]
        # Loss function for RANS equations
        f = ((U*U_x + V*U_y + (1/rho)*P_x - nu*(U_xx + U_yy) + uu_x + uv_y)**2).mean() + \
            ((U*V_x + V*V_y + (1/rho)*P_y - nu*(V_xx + V_yy) + uv_x + vv_y)**2).mean() + \
            ((U_x + V_y)**2).mean()
        
        return f
    # 4. Loss function for boundary conditions (steady-state)
    def loss_bc(self,inlet_bc,wall_bc,inner_layer,Um):
        y_bc = self.net(inlet_bc)
        U_bc_nn, V_bc_nn = y_bc[:,0:1], y_bc[:,1:2]
        # Loss function for inlet
        f_inlet =  ((U_bc_nn.mean()-Um)**2) + ((V_bc_nn.mean()-0.0)**2) 
        # Loss function for walls (No-slip)
        wall_bc = self.net(wall_bc)
        U_wall_nn, V_wall_nn, P_wall_nn, uu_wall_nn, uv_wall_nn, vv_wall_nn = \
          wall_bc[:,0:1], wall_bc[:,1:2], wall_bc[:,2:3], wall_bc[:,3:4], wall_bc[:,4:5], wall_bc[:,5:6]
        il = self.net(inner_layer)
        U_il, V_il, P_il, uu_il, uv_il, vv_il = \
                il[:,0:1], il[:,1:2], il[:,2:3], il[:,3:4], il[:,4:5], il[:,5:6]
        # No-slip and impermeable walls
        f_wall = ((U_wall_nn + U_il)**2).mean() + ((V_wall_nn + V_il)**2).mean() + \
                 ((uu_wall_nn + uu_il)**2).mean() + ((uv_wall_nn + uv_il)**2).mean() + \
                 ((vv_wall_nn + vv_il)**2).mean()
        f_bcs = f_inlet + f_wall
        return f_bcs

# Calculate gradients using torch.autograd.grad
def gradients(outputs,inputs):
    return torch.autograd.grad(outputs,inputs,grad_outputs=torch.ones_like(outputs),create_graph=True)
# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))

########################################################################################################
#                                               Examples                                               #
########################################################################################################
# 1) Fully-developed channel flow
Um = 1.0
H = 2.0
s = 7.0

# 1 (b) By PINN
L = 10.0
device = torch.device('cpu')
lr = 0.0001
epochs = 10
nx = 400 
ny = 202 
iy = (np.linspace(0,ny-1,ny))/(ny-1) - 0.5
y = H * (1.0 + np.tanh(s*iy)/np.tanh(s/2))/2.0 - 1.0
x = np.linspace(0.,L,nx)
x_grid, y_grid = np.meshgrid(x,y)

w_PDE = 0.4
w_BC = 0.6
loss_pde_track = []
loss_bc_track = []
loss_track = []

Y = y_grid[1:ny-2,:].flatten()[:, None]
X = x_grid[1:ny-2,:].flatten()[:, None]

num_f_train = min(len(Y),11000)
id_f = np.random.choice(len(Y),num_f_train,replace=True)

# At inlet
x_bc = x_grid[1:ny-2,0][:,None]
y_bc = y_grid[1:ny-2,0][:,None]
y_bc_train = np.hstack((x_bc,y_bc))   # Random (x,y) - vectorized
# At wall boundary condition
x_wall = np.hstack((x_grid[0,:],x_grid[ny-1,:]))[:,None]
y_wall = np.hstack((y_grid[0,:],y_grid[ny-1,:]))[:,None]
wall_bc = np.hstack((x_wall,y_wall))
x_inner = np.hstack((x_grid[1,:],x_grid[ny-2,:]))[:,None]
y_inner = np.hstack((y_grid[1,:],y_grid[ny-2,:]))[:,None]
inner_layer = np.hstack((x_inner,y_inner))
plt.figure() # YH to check!!!
plt.plot(X,Y,'.',color='g')
plt.plot(x_wall,y_wall,'x',color='r')
plt.plot(x_bc,y_bc,'.',color='b')
plt.grid(True)
plt.show()

if (len(Y)>num_f_train):
    print('Sample size chosen for training smaller than the number of grid points used.')
    x_int = X[:,0][id_f,None]
    y_int = Y[:,0][id_f,None]
else:
    print(f'All the grid points used for training: {num_f_train}')
    x_int = X[:,0][:,None]
    y_int = Y[:,0][:,None]
y_int_train = np.hstack((x_int,y_int))
y_test = np.hstack((x_grid.flatten()[:,None],y_grid.flatten()[:,None]))
#       Generate tensors
y_bc_train = torch.tensor(y_bc_train,dtype=torch.float32).to(device)
wall_bc_train = torch.tensor(wall_bc,dtype=torch.float32).to(device)
inner_layer_train = torch.tensor(inner_layer,dtype=torch.float32).to(device)
y_int_train = torch.tensor(y_int_train,requires_grad=True,dtype=torch.float32).to(device)
y_test = torch.tensor(y_test,dtype=torch.float32).to(device)

#       Initialize Neural Network
model = DNN().to(device)
#       Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(),lr)
#       Train PINNs
def train(epoch,start_epoch):
    model.train()
    def closure():
        optimizer.zero_grad()
        loss_pde = model.loss_pde(y_int_train)
        loss_bc = model.loss_bc(y_bc_train,wall_bc_train,inner_layer_train,Um)
        loss = w_PDE*loss_pde + w_BC*loss_bc

        loss_bc_track.append(loss_bc.item())
        loss_pde_track.append(loss_pde.item())
        # Compute gradients
        loss.backward()
        return loss
    # Optimize loss function
    loss = optimizer.step(closure)
    loss_value = loss.item() if not isinstance(loss,float) else loss

    # Plot loss
    loss_track.append(loss_value)
    plt.figure(1,figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title('Training')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    epochX = range(start_epoch,len(loss_bc_track)+start_epoch)
    plt.plot(epochX,loss_bc_track,color='r',label='Loss BC')
    plt.plot(epochX,loss_pde_track,color='b',label='Loss PDE')
    plt.plot(epochX,loss_track,color='k',label='Total Loss')
    if (len(loss_track)==1): plt.legend()
    if (len(loss_track)>=100):
        plt.subplot(1,2,2).cla()
        plt.subplot(1,2,2)
        plot_st = len(loss_track)-100
        pcount = range(plot_st+start_epoch,len(loss_track)+start_epoch)
        plt.plot(pcount,np.log10(loss_bc_track[plot_st:]),color='r',label='Loss IC')
        plt.plot(pcount,np.log10(loss_pde_track[plot_st:]),color='b',label='Loss PDE')
        plt.plot(pcount,np.log10(loss_track[plot_st:]),color='k',label='Total Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (log)')
    plt.pause(0.001)



# Load the saved checkpoint
while True:
    contOLD = input('Continue from previous checkpoint? (y/n):')
    try:
        if contOLD in ['y','n']:
            break
        else:
            print('Invalid input. Please choose y or n.')
    except ValueError:
        print('Invalid input type: {}'.format(type(contOLD)))
if (os.path.exists('channel_model_checkpoint.pth') and contOLD == 'y'):
    checkpoint = torch.load('channel_model_checkpoint.pth')
    start_epoch = int(checkpoint['epoch'] + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_bc_track.append(checkpoint['loss_bc'])
    loss_pde_track.append(checkpoint['loss_pde'])
    loss_track.append(checkpoint['loss_overall'])
else:
    print('Restart training.')
    start_epoch = 1
# Start training and compute total time for training
print('Begin training ...')
plt.ion()
tic = time.time()
for epoch in range(start_epoch, epochs+start_epoch):
    train(epoch,start_epoch)

    # Save the model and optimizer state as before
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_bc': loss_bc_track[len(loss_bc_track)-1],
            'loss_pde': loss_pde_track[len(loss_pde_track)-1],
            'loss_overall': loss_track[len(loss_track)-1],
            }
    torch.save(checkpoint,'channel_model_checkpoint.pth')
toc = time.time()
print(f'Total training time for PINNs: {toc - tic}')
plt.ioff()

# Evaluate on the whole computational domain
u_pred = to_numpy(model(y_test))

# 1 (c) Plotting
#       i. From PINNs (contour plots)
plt.figure(figsize=(16,10))
plt.subplot(3,2,1)
plt.contourf(x_grid, y_grid, u_pred[:, 0].reshape(ny,nx), levels=100, cmap='viridis')
plt.colorbar(label='U')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(3,2,2)
plt.contourf(x_grid, y_grid, u_pred[:, 1].reshape(ny,nx), levels=100, cmap='viridis')
plt.colorbar(label='V')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(3,2,3)
plt.contourf(x_grid, y_grid, u_pred[:, 2].reshape(ny,nx), levels=100, cmap='viridis')
plt.colorbar(label='Pressure')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(3,2,4)
plt.contourf(x_grid, y_grid, u_pred[:, 4].reshape(ny,nx), levels=100, cmap='viridis')
plt.colorbar(label='uv')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(3,2,5)
plt.plot(y_grid[:,0],u_pred[:,0].reshape(ny,nx)[:,0],color='g')
plt.plot(y_grid[:,0],u_pred[:,0].reshape(ny,nx)[:,nx//2],color='b')
plt.plot(y_grid[:,0],u_pred[:,0].reshape(ny,nx)[:,nx-1],color='r')
plt.grid(True)

plt.subplot(3,2,6)
plt.plot(y_grid[:,0],u_pred[:,1].reshape(ny,nx)[:,0],color='g')
plt.plot(y_grid[:,0],u_pred[:,1].reshape(ny,nx)[:,nx//2],color='b')
plt.plot(y_grid[:,0],u_pred[:,1].reshape(ny,nx)[:,nx-1],color='r')
plt.grid(True)
plt.savefig('contour_channel_plot.png')

