# Applying machine learning techniques for CFD

## 1) Physics-Informed Neural Networks (PINNs) for hyperbolic PDEs
Here, PINN works by taking in (x,t) domain as input and produces (\rho,u,p) as outputs. 
This is done by training the model on DNNs using the datasets obtained from each epoch and 
computing the losses from initial conditions and PDEs before applying optimizer to minimize the total loss.



### References
[1]
[2]
