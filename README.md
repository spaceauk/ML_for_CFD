# Applying machine learning techniques for CFD 

## 1) Physics-Informed Neural Networks (PINNs) for hyperbolic PDEs
Here, PINN works by taking in (x,t) domain as input and produces (\rho,u,p) as outputs. 
This is done by training the model on DNNs using the datasets obtained from each epoch and 
computing the losses from initial conditions and PDEs before applying optimizer to minimize the total loss.

The objective of this project is to improve PINNs ability to solve discontinuities as traditional DNNs are limited by their inherent nature of representing function continuously. One way of rectifying this issue is to minimize losses at regions where shocks are expected as follows,
1. GA-PINNs [1]: Minimized loss weight based on the presence of steep gradients
2. PINNs-WE [2]: Reduced loss weight at regions with negative dilatation
3. Compressibility-activated loss weight reduction depending on gradients of primitive variables at localized regions

## 2) Modeless RANS using PINNs
Here, the Reynolds shear stress is estimated using PINNs instead of being modeled through traditional approaches like the Boussinesq hypothesis. This approach is applied to a turbulent channel flow test case.

### References
[1] https://arxiv.org/pdf/2305.08448.pdf <br>
[2] https://arxiv.org/pdf/2206.03864.pdf
