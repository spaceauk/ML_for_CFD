import os
import math
import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
from scipy.stats import qmc      # Quasi Monte Carlo method
import matplotlib.pyplot as plt
from pyDOE2 import lhs           # Latin hypercube sampling 

# Seeds to initialize random number generator's internal state
torch.manual_seed(123)
np.random.seed(123)

# Constants
gamma = 1.4
num_layers = 7
num_neurons = 30
# Select features for PINNs
while True:
    try:
        ALtype = int(input("Choose Activation layer type (1 to 4): "))
        if ALtype in range(1,5):
            break
        else:
            print("Invalid input. Please choose 1 to 4.")
    except ValueError:
        print("Invalid input. Please enter a number (1 to 4).")
while True:
    try: 
        ICtype = int(input("Choose initial conditions type (1 to 5): "))
        if ICtype in range(1,9):
            break
        else:
            print("Invalid input. Please choose 1 to 5.")
    except ValueError:
        print("Invalid input. Please enter a number between 1 to 5.")
# Weights for different types of loss (W-PINNs)
w_PDE = 0.1
w_IC = 10.0
# Extra features
while True:
    try:
        GPtype = int(input("Choose sampling type for grid points arrangement (1 to 4): "))
        if GPtype in range(1,5):
            break
        else:
            print("Invalid input. Please choose 1 to 4.")
    except ValueError:
        print("Invalid input. Please enter a number (1 to 4).")
discontC_type = int(input("Types of mod to PINN to account for discontinuity (1 to 3) else no mod:  "))
xmax = 1.0
xmin = 0.0
if (discontC_type==1): 
    print("GA-PINNs: based on steep gradient")    
    alphaGA = [1.0,1.0,1.0]
    betaGA = [1.0,1.0,1.0]
elif (discontC_type==2):
    print("PINNs-WE: based on compressibility - div(u)<0")
    eps2 = 1.0
elif (discontC_type==3):
    print("Steep gradient-based magnitude adjustment for compressibility-activated loss weight reduction")
    alphaGA = [1.0,1.0,1.0]
    betaGA = [1.0,1.0,1.0]
else:
    print("No mod selected. Just PINNs with different loss weights.")

# Generate Deep Neural Network
class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential()                                                  # Define neural network
        # Define linear and activation layers
        for num in range(1, num_layers):                                            
            if (num==1):                                                            
                self.net.add_module('Linear_layer_1', nn.Linear(2, num_neurons))    # Input layer (x,t)
            else:
                self.net.add_module('Linear_layer_%d' % (num), nn.Linear(num_neurons, num_neurons))
            if (ALtype==1):
                self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())             
            elif (ALtype==2):
                self.net.add_module('GELU_layer_%d' %(num), nn.GELU())              
            elif (ALtype==3):
                self.net.add_module('Sigmoid_layer_%d' %(num), nn.Sigmoid())
            elif (ALtype==4):
                self.net.add_module('ELU_layer_%d' %(num), nn.ELU())
        self.net.add_module('Linear_layer_final', nn.Linear(num_neurons, 3))        # Output layer (rho,u,p)

    # Forward Feed
    def forward(self, x):
        return self.net(x)

    # Loss function for PDE
    def loss_pde(self, x):
        y = self.net(x)                                                # Forward pass of NN
        rho,u,p = y[:, 0:1], y[:, 1:2], y[:, 2:]                       # Output of NN

        # Gradients and partial derivatives
        drho_grad = gradients(rho, x)[0]                        
        drho_dt, drho_dx = drho_grad[:, :1], drho_grad[:, 1:]          

        du_grad = gradients(u, x)[0]                                   
        du_dt, du_dx = du_grad[:, :1], du_grad[:, 1:]               

        dp_grad = gradients(p, x)[0]                    
        dp_dt, dp_dx = dp_grad[:, :1], dp_grad[:, 1:]   

        # Types of ways to calculate losses for the Euler Eq.
        if (discontC_type==1 or discontC_type==3):
            f_intm1 = ((drho_dt + u*drho_dx + rho*du_dx)**2) + ((rho*(du_dt + (u)*du_dx) + (dp_dx))**2) + \
                ((dp_dt + gamma*p*du_dx + u*dp_dx)**2)
            if (discontC_type==1):
                factor_r = 1/(1+alphaGA[0]*drho_dx**betaGA[0] + alphaGA[1]*du_dx**betaGA[1] + alphaGA[2]*dp_dx**betaGA[2])
            elif (discontC_type==3):                
                int0 = torch.abs(du_dx)-du_dx
                int1 = du_dx*0.0
                non_zero_mask = torch.abs(int0) > 0.
                int1[non_zero_mask] = int0[non_zero_mask]/torch.abs(int0[non_zero_mask])
                factor_r = 1/(1+int1*(alphaGA[0]*drho_dx**betaGA[0] + alphaGA[1]*du_dx**betaGA[1] + alphaGA[2]*dp_dx**betaGA[2]))
            f_intm1 = factor_r*f_intm1
            f = f_intm1.mean()
        elif (discontC_type==2):
            # Differs from PINN only when div(u)<0 where compression occurs
            lambdaWE = 1/(eps2*(torch.abs(du_dx)-du_dx)+1.0)
            f_intm1 = ((drho_dt + u*drho_dx + rho*du_dx)**2) + ((rho*(du_dt + (u)*du_dx) + (dp_dx))**2) + \
                ((dp_dt + gamma*p*du_dx + u*dp_dx)**2)
            f_intm1 = lambdaWE*f_intm1
            f = f_intm1.mean()
        else: 
            f = ((drho_dt + u*drho_dx + rho*du_dx)**2).mean() + ((rho*(du_dt + (u)*du_dx) + (dp_dx))**2).mean() + \
                ((dp_dt + gamma*p*du_dx + u*dp_dx)**2).mean()

        return f

    # Loss function for initial condition
    def loss_ic(self, x_ic, rho_ic, u_ic, p_ic):
        y_ic = self.net(x_ic)                                           
        rho_ic_nn, u_ic_nn,p_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2]            

        # Calculate loss for the initial condition
        loss_ics = ((u_ic_nn - u_ic)**2).mean() + \
               ((rho_ic_nn- rho_ic)**2).mean()  + \
               ((p_ic_nn - p_ic)**2).mean()

        return loss_ics


# Compute gradients using torch.autograd.grad
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

# Convert torch tensor to numpy array for plotting
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Input type not recognized: {}'.format(type(input)))

# Initial conditions
def IC(x):
    N = len(x)
    rho_init = np.zeros((x.shape[0]))
    u_init = np.zeros((x.shape[0]))                                         
    p_init = np.zeros((x.shape[0]))                                                

    # rho, p - initial condition
    for i in range(N):
        if (ICtype==1):
            if (i==1): print("Test case 1: Sod shock tube")
            if (x[i] <= 0.5):
                rho_init[i] = 1.0
                p_init[i] = 1.0
            else:
                rho_init[i] = 0.125
                p_init[i] = 0.1
        elif (ICtype==2):
            if (i==1): print("Test case 2: Double strong rarefactions")
            if (x[i] <= 0.5):
                rho_init[i] = 1.0
                u_init[i] = -2.0
                p_init[i] = 0.4
            else:
                rho_init[i] = 1.0
                u_init[i] = 2.0
                p_init[i] = 0.4
        elif (ICtype==3):
            if (i==1): print("Test case 3: Left half of blast wave problem")
            if (x[i] <= 0.5):
                rho_init[i] = 1.0
                u_init[i] = 0.0
                p_init[i] = 1000.0
            else:
                rho_init[i] = 1.0
                u_init[i] = 0.0
                p_init[i] = 0.01
        elif (ICtype==4):
            if (i==1): print("Test case 4: Right half of Woodward and Collela problem")
            if (x[i] <= 0.5):
                rho_init[i] = 1.0
                u_init[i] = 0.0
                p_init[i] = 0.01
            else:
                rho_init[i] = 1.0
                u_init[i] = 0.0
                p_init[i] = 100.0
        elif (ICtype==5):
            if (i==1): print("Test case 5: Collision of two strong shocks")
            if (x[i] <= 0.5):
                rho_init[i] = 5.99924
                u_init[i] = 19.5975
                p_init[i] = 460.894
            else:
                rho_init[i] = 5.99242
                u_init[i] = -6.19633
                p_init[i] = 46.0950

    return rho_init, u_init, p_init

# Solve Euler equations using PINNs
plt.ion()
loss_track = []
loss_ic_track = []
loss_pde_track = []
# Initialization
device = torch.device('cpu')                                    
lr = 0.0005                                            # Learning rate             
num_x = 2**6                                                        
num_t = num_x                                                   
epochs = 25000                                         # Number of iterations
num_i_train = 1000                                     # Number of samples for training                
num_int_train = 11000                                               
if (ICtype==1):
    tmax = 0.25
elif (ICtype==2):
    tmax = 0.15
elif (ICtype==3):
    tmax = 0.012
elif (ICtype==4 or ICtype ==5):
    tmax = 0.035
else:
    tmax = 0.2
# Grid points arrangement as input for PINN
if (GPtype==1): # Sobol sequence sambler setup
    print("Sobol sequence chosen...")
    sobol_sampler = qmc.Sobol(d=2, scramble=False)
    sobol_sample = sobol_sampler.random_base2(m=int(math.log(num_x*num_t,2)) )
    # Add two more rows of uniform distributionat start and end of t-axis
    t_grid1 = tmax*sobol_sample[:, 0]
    t_grid = np.concatenate((np.zeros(num_x), t_grid1, np.full(num_x, tmax))).reshape((num_x,num_t+2))
    x_grid1 = (xmax-xmin)*sobol_sample[:, 1]+xmin
    x_grid = np.concatenate((np.linspace(0.0,1.0,num_x),x_grid1,np.linspace(0.0,1.0,num_x))).reshape((num_x,num_t+2))
    num_t += 2
elif (GPtype==2):
    print("Random uniform distribution chosen...")
    t_grid = np.random.uniform(0.0, tmax, size=(num_x, num_t))  
    x_grid = np.random.uniform(xmin, xmax, size=(num_x, num_t))  
elif (GPtype==3):
    print("Latin hypercube sampling chosen...")
    samples = lhs(2, samples=num_x*num_t)
    t_grid = tmax*samples[:, 0].reshape((num_x,num_t)) 
    x_grid = (xmax-xmin)*samples[:, 1].reshape((num_x,num_t))+xmin
elif (GPtype==4):
    print("Uniform distribution chosen...")
    x = np.linspace(xmin, xmax, num_x)                               # Partitioned spatial axis
    t = np.linspace(0, tmax, num_t)                                 # Partitioned time axis
    t_grid, x_grid = np.meshgrid(t,x)                                    # (t,x) in [0,0.2]x[a,b]
print("Number of x grid points = ",num_x," and number of t grid points = ",num_t)
x_grid[:,0] = np.linspace(xmin,xmax,num_x)
t_grid[:,0] = 0.0
x_grid[:,num_t-1] = np.linspace(xmin,xmax,num_x)
t_grid[:,num_t-1] = tmax
x = x_grid.reshape(num_x*num_t)
t = t_grid.reshape(num_x*num_t)
plt.figure(figsize=(8,6))
plt.scatter(t_grid,x_grid,marker='.',s=2,c='red')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Grid Points Visualization')
plt.grid(True)
plt.savefig('gridpoints.png')

# Convert from 2D array to 1D array as more efficient for processing
T = t_grid.flatten()[:, None]                                         
X = x_grid.flatten()[:, None]                                         

# Obtain random indices for selecting training samples
id_ic = np.random.choice(num_x, num_i_train, replace=True)           
id_int = np.random.choice(num_x*num_t, num_int_train, replace=True)      

# Obtain IC and interior data for training based on the random indices obtained from above
x_ic = x_grid[id_ic, 0][:, None]                                      
t_ic = t_grid[id_ic, 0][:, None]                                    
x_ic_train = np.hstack((t_ic, x_ic))                                
rho_ic_train, u_ic_train, p_ic_train = IC(x_ic)                

x_int = X[:, 0][id_int, None]                             
t_int = T[:, 0][id_int, None]                             
x_int_train = np.hstack((t_int, x_int))                     
x_test = np.hstack((T, X))                                            

# Converting np.array to torch.tensor as latter can use auto-diff
x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32).to(device)
x_int_train = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)

rho_ic_train = torch.tensor(rho_ic_train, dtype=torch.float32).to(device)
u_ic_train = torch.tensor(u_ic_train, dtype=torch.float32).to(device)
p_ic_train = torch.tensor(p_ic_train, dtype=torch.float32).to(device)

# Initialize neural network
model = DNN().to(device)

# Adam (Adaptive Moment Estimation) optimizer to minimize loss function during training
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Function for training PINNs
def train(epoch,start_epoch):
    model.train()
    def closure():
        # Clear previously computed gradients
        optimizer.zero_grad()                
        # Calculate losses
        loss_pde = model.loss_pde(x_int_train)                                  
        loss_ic = model.loss_ic(x_ic_train, rho_ic_train,u_ic_train,p_ic_train)   
        loss = w_PDE*loss_pde + w_IC*loss_ic                                      
        loss_ic_track.append(loss_ic.item()) 
        loss_pde_track.append(loss_pde.item())
           
        # Compute gradients
        loss.backward()
        return loss

    # Optimize loss function
    loss = optimizer.step(closure)
    loss_value = loss.item() if not isinstance(loss, float) else loss
    # Plot loss
    loss_track.append(loss_value)
    plt.figure(1,figsize=(15,6))
    plt.subplot(1,2,1)
    plt.title('Training')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    epochX = range(start_epoch,len(loss_ic_track)+start_epoch) 
    plt.plot(epochX,loss_ic_track,color='r',label='Loss IC')
    plt.plot(epochX,loss_pde_track,color='b',label='Loss PDE')
    plt.plot(epochX,loss_track,color='k',label='Total Loss')
    if (len(loss_track)==1): plt.legend()
    if (len(loss_track)>=100):
        plt.subplot(1,2,2).cla()
        plt.subplot(1,2,2)
        plot_st = len(loss_track)-100
        pcount = range(plot_st+start_epoch,len(loss_track)+start_epoch)
        plt.plot(pcount,np.log(loss_ic_track[plot_st:]),color='r',label='Loss IC')
        plt.plot(pcount,np.log(loss_pde_track[plot_st:]),color='b',label='Loss PDE')
        plt.plot(pcount,np.log(loss_track[plot_st:]),color='k',label='Total Loss')
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
if (os.path.exists('model_checkpoint.pth') and contOLD == 'y'):
    checkpoint = torch.load('model_checkpoint.pth')
    start_epoch = int(checkpoint['epoch'] + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_ic_track.append(checkpoint['loss_ic'])
    loss_pde_track.append(checkpoint['loss_pde'])
    loss_track.append(checkpoint['loss_overall'])
else:
    print('Restart training.')
    start_epoch = 1
# Start training and compute total time for training
print('Begin training ...')
tic = time.time()
for epoch in range(start_epoch, epochs+start_epoch):
    train(epoch,start_epoch)

    # Save the model and optimizer state as before
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_ic': loss_ic_track[len(loss_ic_track)-1],
            'loss_pde': loss_pde_track[len(loss_pde_track)-1],
            'loss_overall': loss_track[len(loss_track)-1],        
            }
    torch.save(checkpoint,'model_checkpoint.pth')
toc = time.time()
print(f'Overall training time = {toc - tic}')
plt.savefig('train_plot.png')
plt.ioff()

# Evaluate on the whole computational domain
u_pred = to_numpy(model(x_test))

# Create contour plots
plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.contourf(t_grid, x_grid, u_pred[:, 0].reshape(num_x,num_t), levels=100, cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('t')
plt.ylabel('x')

plt.subplot(2,2,2)
plt.contourf(t_grid, x_grid, u_pred[:, 2].reshape(num_x,num_t), levels=100, cmap='viridis')
plt.colorbar(label='U')
plt.xlabel('t')
plt.ylabel('x')

plt.subplot(2,2,3)
plt.contourf(t_grid, x_grid, u_pred[:, 1].reshape(num_x,num_t), levels=100, cmap='viridis')
plt.colorbar(label='Pressure')
plt.xlabel('t')
plt.ylabel('x')

plt.subplot(2,2,4)
ie = u_pred[:,1]/(gamma-1)+0.5*u_pred[:,0]*u_pred[:,2]*u_pred[:,2]
plt.contourf(t_grid, x_grid, ie.reshape(num_x,num_t), levels=100, cmap='viridis')
plt.colorbar(label='Internal Energy')
plt.xlabel('t')
plt.ylabel('x')
plt.savefig('contour_xt_plot.png')

# Create line plots
plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.plot(x_grid[:,num_t-1], u_pred[:, 0].reshape(num_x,num_t)[:,num_t-1])
plt.xlabel('x')
plt.ylabel('Density')

plt.subplot(2,2,2)
plt.plot(x_grid[:,num_t-1], u_pred[:, 2].reshape(num_x,num_t)[:,num_t-1])
plt.xlabel('x')
plt.ylabel('U')

plt.subplot(2,2,3)
plt.plot(x_grid[:,num_t-1], u_pred[:, 1].reshape(num_x,num_t)[:,num_t-1])
plt.xlabel('x')
plt.ylabel('Pressure')

plt.subplot(2,2,4)
plt.plot(x_grid[:,num_t-1], ie.reshape(num_x,num_t)[:,num_t-1])
plt.xlabel('x')
plt.ylabel('Internal Energy')
plt.savefig('line_plot.png')

