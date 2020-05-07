#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
from jax.config import config 
import jax.numpy as np
config.update('jax_enable_x64', True)
import numpy as onp
#import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from jax import grad, jit, random
from jax.ops import index, index_add, index_update
from Mesh_fd import Mesh
import copy
import scipy
import scipy.stats as ss
from tqdm import tqdm #conda install tqdm - this is needed for plotting a progress bar
import time
date = time.strftime("%d_%m_%Y")



# # Momentum equation (fully developed channel)
# The streamwise momentum equation of the Reynolds-averaged Navier-Stokes equations for a fully developed turbulent channel flow reads, 
# 
# $$ \mu \frac{d^2\left<U\right>}{dy^2} + \frac{d}{dy}\left(\mu_t  \frac{d\left<U\right>}{dy}\right) = -\frac{dP}{dx},$$ 
# 
# 
# with $\left<U\right>$ the streamwise mean velocity, $\mu$ the dynamic visocity, and $\mu_t$ the Eddy viscosity which is obtained by a solving a turbulence model. Linear eddy visosity is used to model the Reynolds stress term here approximate by $\mu_t  \frac{d\left<U\right>}{dy}$. Wall units are used to non-dimensionalize the Navier-Stokes equations, where mean velocity and RS term is normalised by frictional velocity $v_{\tau}$ and $y$ bu viscous length scale $\frac{\nu}{u_{\tau}}$. Mean pressure gradient is given by $-\frac{dp}{dx}=\frac{\tau_w}{\delta}$ which upon normalisation with wall units gives $-1$, considering $\delta=1$ for the channel half height defining the forcing on the right-hand-side. 
# Using the product rule, the momentum equation can also be written as  
# 
# $$\frac{d\mu_t}{dy}\frac{d \left<U\right>}{dy} + (\mu+\mu_t)\frac{d^2 \left<U\right>}{dy^2}=-1$$
# 
# Now it is easy to formulate a linear system $ A u = -1 $ which can be solved using a direct linear solver.
# 
# $$\left[\frac{d\mu_t}{dy}\frac{d}{dy} + (\mu+\mu_t)\frac{d^2}{dy^2}\right] \left<U\right>=-1 $$
#     
# 
# $\mu_t$ can also be taken as parameter to be optimised. The optimsation funtional is of the form $min_{\mu_t}J(\left<U\right>(\mu_t),\mu_t)$, here it can be 
# $$ ||\left<U\right>_{DNS}-\left<U\right>(\mu_t)||^2 \quad \textrm{s.t} \quad \mathcal{F}(\left<U\right>, \mu_t) = 0 $$

# In[3]:


# Function for RANS solution.
def solveRANS(mu_turb):

    n    = mesh.nPoints
    u    = np.zeros(n)          # velocity 

    mut  = mu_turb       # eddy viscosity 
    #Atul: We have scalar value of density and viscoity here

    
    residual   = 1.0e20
    iterations = 0
    #print(mu)
    #print("Start iterating")

    while residual > 1.0e-6 and iterations < 10000:
 
        A = np.einsum('i,ij->ij', mesh.ddy@(mut), mesh.ddy)         + np.einsum('i,ij->ij', mu + mut, mesh.d2dy2)
        # Solve 
        
        #u_old = u.copy()
        
        u_old=copy.deepcopy(u)
        #u[1:n-1]=np.linalg.solve(A[1:n-1, 1:n-1], -np.ones(n-2))
        u=index_update(u,index[1:n-1],np.linalg.solve(A[1:n-1, 1:n-1], -np.ones(n-2)))   
        
        residual = np.linalg.norm(u-u_old)/n

        iterations = iterations + 1

    #print("iteration: ",iterations, ", Residual(u) = ", residual)
    
    return u 


# # Mesh generation and inputs
# Load DNS data and call mesh file

# In[4]:


import linecache

# Wall clustered structured mesh
mesh = Mesh(150, 2, 6, 1)  # mesh point, chaanel height, stretching factor, stencil

DNS_case = ["constProperty.txt",             "constReTauStar.txt",             "gasLike.txt",             "liquidLike.txt"]
file = "DNS_data/"+ DNS_case[1]  # chose case

# get parameters from DNS or set these parameters (they must be defined)
line = linecache.getline(file, 39)[1:].split()       
ReTau  = float(line[0]); print("ReTau  = ", ReTau)   # Reynolds number

# load dns data
DNS = onp.loadtxt(file,skiprows=88)

mu=1.0/ReTau  ## In the DNS data generation mu is varrying with y. For simplicity we take it to be constant
#Extrapolate DNS u to RANS grid
u_DNS=onp.interp(np.minimum(mesh.y, mesh.y[-1]-mesh.y) , DNS[:,0], DNS[:,8])


# # Choosing initial $\mu_t$

# In[5]:




mu_t_RANS=np.array([0.00000000e+00, 7.47051770e-09, 6.55464377e-08, 2.79792381e-07,
       8.44475255e-07, 2.07231617e-06, 4.42491657e-06, 8.53835836e-06,
       1.52349238e-05, 2.55083410e-05, 4.04688599e-05, 6.12399432e-05,
       8.88142654e-05, 1.23901272e-04, 1.66819034e-04, 2.17480068e-04,
       2.75486294e-04, 3.50195460e-04, 4.73238061e-04, 6.17266940e-04,
       7.82504485e-04, 9.69511263e-04, 1.17921502e-03, 1.41287563e-03,
       1.67202631e-03, 1.95841002e-03, 2.27391821e-03, 2.62053294e-03,
       3.00027167e-03, 3.41513354e-03, 3.86704608e-03, 4.35781126e-03,
       4.88905065e-03, 5.46214908e-03, 6.07819739e-03, 6.73793458e-03,
       7.44169093e-03, 8.18933397e-03, 8.98022031e-03, 9.81315746e-03,
       1.06863812e-02, 1.15975558e-02, 1.25438080e-02, 1.35218067e-02,
       1.45278023e-02, 1.55575996e-02, 1.66065667e-02, 1.76696854e-02,
       1.87416455e-02, 1.98169915e-02, 2.08903253e-02, 2.19565758e-02,
       2.30113382e-02, 2.40512905e-02, 2.50746860e-02, 2.60819166e-02,
       2.70761297e-02, 2.80638635e-02, 2.90556388e-02, 3.00664070e-02,
       3.11157045e-02, 3.22272995e-02, 3.34280648e-02, 3.47457761e-02,
       3.62055807e-02, 3.78250504e-02, 3.96080677e-02, 4.15383001e-02,
       4.35735764e-02, 4.56429096e-02, 4.76479316e-02, 4.94699317e-02,
       5.09824842e-02, 5.20680816e-02, 5.26357345e-02, 5.26357345e-02,
       5.20680816e-02, 5.09824842e-02, 4.94699317e-02, 4.76479316e-02,
       4.56429096e-02, 4.35735764e-02, 4.15383001e-02, 3.96080677e-02,
       3.78250504e-02, 3.62055807e-02, 3.47457761e-02, 3.34280648e-02,
       3.22272995e-02, 3.11157045e-02, 3.00664070e-02, 2.90556388e-02,
       2.80638635e-02, 2.70761297e-02, 2.60819166e-02, 2.50746860e-02,
       2.40512905e-02, 2.30113382e-02, 2.19565758e-02, 2.08903253e-02,
       1.98169915e-02, 1.87416455e-02, 1.76696854e-02, 1.66065667e-02,
       1.55575996e-02, 1.45278023e-02, 1.35218067e-02, 1.25438080e-02,
       1.15975558e-02, 1.06863812e-02, 9.81315746e-03, 8.98022031e-03,
       8.18933397e-03, 7.44169093e-03, 6.73793458e-03, 6.07819739e-03,
       5.46214908e-03, 4.88905065e-03, 4.35781126e-03, 3.86704608e-03,
       3.41513354e-03, 3.00027167e-03, 2.62053294e-03, 2.27391821e-03,
       1.95841002e-03, 1.67202631e-03, 1.41287563e-03, 1.17921502e-03,
       9.69511263e-04, 7.82504485e-04, 6.17266940e-04, 4.73238061e-04,
       3.50195460e-04, 2.75486294e-04, 2.17480068e-04, 1.66819034e-04,
       1.23901272e-04, 8.88142654e-05, 6.12399432e-05, 4.04688599e-05,
       2.55083410e-05, 1.52349238e-05, 8.53835836e-06, 4.42491657e-06,
       2.07231617e-06, 8.44475255e-07, 2.79792381e-07, 6.55464377e-08,
       7.47051770e-09, 0.00000000e+00])
       
#Case 1: Starting with mu_t from RANS solution       
noise=0 ## Internsity of noise can be controlled with changing sigma, 0 for no noise, onp.random.normal(0,0.0005,150) for noise
mu_t=mu_t_RANS+noise

'''
#Case 2: If it is randombly choosed from a smooth parabolic function
dim=np.size(u_DNS)
y=np.linspace(0,2,dim)

#mu_t=-(y-1)**2 +y
mu_t=-(y-1)**2 +1
mu_t=0.01*mu_t
#dim=np.size(u_DNS)
#mu_t = onp.random.normal(0.04,0.01,dim)
##mu_t=onp.float32(mu_t)
'''

# Case 3: If its a scalar
#mu_t=0.01



# # Bayesian Implementation
#prior
##zero mean gaussian
class GaussianPrior(object):
    "" #Write Desciption here
    def __init__(self, mean, ssigma, f = None):

        self._mean = mean
        self._ssigma = ssigma  # sigma squared, i.e. variance
        
    @property
    def dim(self):
        return self._mean.size
    @jit    
    def LogEvaluate(self, x):

        return -0.5*self.dim*onp.log(2*onp.pi) - 0.5*self.dim*onp.log(self._ssigma) - 0.5*(1/self._ssigma)*np.dot(x-self._mean, x-self._mean)  #dot by default has sum of all values

##Gaussian Markovian Prior

class GaussianMarkPrior(object):
    "" #Write Desciption here
    def __init__(self, pres, f = None):

        #self._mean = mean
        self._pres = pres  # Q: the precision matrix
        
    @property
    def dim(self):
        return self._pres.size
        
    def LogEvaluate(self, x):

        #return -0.5*self.dim*onp.log(2*onp.pi) - 0.5*onp.log(onp.linalg.det(self._pres)) -0.5 * onp.dot(x, onp.dot(self._pres, x))  #dot by default has sum of all values
        return -0.5 * np.dot(x, np.dot(self._pres, x))
    
class GaussianLikelihood(object):
    
    def __init__(self, data, model, ssigma):
        
        self._data = data     # vector
        self._model = model   # callable, RANS solver
        self._ssigma = ssigma   # variance of observation (noise)
        
    @property
    def dim(self):
        return self._data.size
        
        
    def LogEvaluate(self, x):
        
        U_RANS = self._model(x) #RANS solution
        return -0.5*self.dim*onp.log(2*onp.pi) - 0.5*self.dim*onp.log(self._ssigma) - 0.5*(1/self._ssigma)*np.dot(self._data-U_RANS, self._data-U_RANS)

class Posterior(object):
    
    def __init__(self, prior, likelihood):
        
        self._prior = prior
        self._likelihood = likelihood
        
    def LogEvaluate(self, x):
    
        return self._prior.LogEvaluate(x) + self._likelihood.LogEvaluate(x)
    


# In[7]:


# defining log_h which returns value of funtion and the gradient

dim_mut=np.size(mu_t)
mean_prior=np.zeros(dim_mut)


def log_h(x,ss_prior,ss_likelihood):
#def log_h(x):
    """Return objective value and jacobian value wrt. x"""
    #prior = GaussianPrior(mean_prior, ss_prior)  # zero mean gaussian
    prior=GaussianMarkPrior(ss_prior)
    likelihood = GaussianLikelihood(u_DNS, solveRANS, ss_likelihood)
    posterior = Posterior(prior, likelihood)
    value_posterior=posterior.LogEvaluate(x)
    tmp=grad(posterior.LogEvaluate)   #Implementation using JAX. Can you swotched to TF.
    grad_posterior=tmp(x)
    
    return value_posterior, grad_posterior




# ## Sampling Methods (MCMC)

def mala(x0, log_h, n, dt, args=()): #def mala(x0, n, dt):    #def mala(x0, log_h, n, dt, args=()): ATUL
    """
    Random walk metropolis.
    
    :param x0:     The initial point (numpy array).
    :param log_h:  The logartihm of the function that is proportional to the density you want to sample from (function).
                   Returns also the gradient.
    :param n:      The maximum number of steps you want to take.
    :param dt:     The time step you want to use.
    :param args:   Any parameters to log_h
    
    :returns:  X, acceptance_rate
    """
    x0 = onp.array(x0)
    assert x0.ndim == 1
    # Dimensionality of space
    d = x0.shape[0]
    # A place to store the samples
    X = onp.ndarray((n + 1, d))
    X[0, :] = x0
    # Previous value of log(h(x))
    log_h_p, grad_log_h_p = log_h(x0, *args)
    #log_h_p=posterior.LogEvaluate(x0)  #Atul
    #grad_log_h_p=posterior_grad(x0)    #Atul

    # Keep track of how many samples are accepted
    count_accepted = 0
    # Start the simulation
    for t in tqdm(range(1, n + 1)):
        # Generation
        x = X[t - 1, :] + dt * grad_log_h_p + onp.sqrt(2. * dt) * onp.random.randn(d)
        # Calculation
        log_h_c, grad_log_h_c = log_h(x, *args) # Current value of log(h(x))
        #log_h_c=posterior.LogEvaluate(x)  #Atul, can save output of RANS here and use later for plots
        #grad_log_h_c=posterior_grad(x)    #Atul
        
        log_alpha_1 = log_h_c - log_h_p
        log_T_p_to_c = -onp.sum((x - X[t - 1, :] - dt * grad_log_h_p) ** 2 / (4. * dt))
        log_T_c_to_p = -onp.sum((x + dt * grad_log_h_c - X[t - 1, :]) ** 2 / (4. * dt))
        log_alpha_2 = log_T_c_to_p - log_T_p_to_c
        log_alpha = log_alpha_1 + log_alpha_2
        alpha = min(1, onp.exp(log_alpha))
        # Accept/Reject
        u = onp.random.rand()
        if u <= alpha: # Accept
            X[t, :] = x
            log_h_p = log_h_c
            grad_log_h_p = grad_log_h_c
            count_accepted += 1
        else:          # Reject
            X[t, :] = X[t - 1, :]
    # Empirical acceptance rate
    acceptance_rate = count_accepted / (1. * n)

    return X, acceptance_rate



# In[8]:


# Input for the posterior and prior

# Input for gaussian markovian prior
n_i=2
L=onp.zeros((mesh.y.size,mesh.y.size))
for i in range(mesh.y.size):
    for j in range(mesh.y.size):
        if i==j:
            L[i,j]=n_i
        elif i==j+1 or i==j-1:
            L[i,j]=-1
delta=0.1
h=1/(1+mesh.y.size)
Q=(delta/h**2)*L

#ssigma_prior=10
ssigma_likelihood=8
ssigma_prior=Q


# In[13]:


# Maximising posterior to select initial point (MAP)
def m_log_h(x,ssigma_prior,ssigma_likelihood):
    tmp1,tmp2=log_h(x,ssigma_prior,ssigma_likelihood)
    return -tmp1,-tmp2

#x0_BFGS=onp.zeros(mesh.y.size)
x0_BFGS=mu_t
#res = scipy.optimize.minimize(m_log_h, x0_BFGS, jac=True, args=(ssigma_prior, ssigma_likelihood),options={'disp': True})



# Calling MALA sampler
# Initialiazation:
#x0 = mu_t
x_prev=np.load('Results/Samples_X0_MAP05_05_2020.npy')
x0=x_prev[-1,:]
#x0=res.x
# Parameters of the proposal:
dt = 2e-10 ## Notice here step size is the varience so it will be sqrt. 
#dt=1e-08 ## check
# Number of steps:
n = 30000
#X, acceptance_rate = mala(x0, n, dt)
X, acceptance_rate = mala(x0,log_h, n, dt,args=(ssigma_prior,ssigma_likelihood))


print('acceptanc rate=',acceptance_rate)
np.save('Results/Samples_X0_MAP_v3'+date,X)


# Forward Solve for velocity

u_pred=onp.ndarray((X.shape[0],mesh.y.size))
# posterior samples
for n in range(X.shape[0]):
    u_pred[n,:] = solveRANS(X[-n-1,:])
np.save('Results/U_pred_X0_MAP'+date,u_pred)    


