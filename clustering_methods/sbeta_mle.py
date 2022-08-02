import torch
from torch import Tensor
from typing import Tuple
from torch import digamma as psi
import math
import time
from copy import deepcopy

def inv_psi(y: Tensor, device_name = 'cpu') -> Tensor:
	"""
	Computes the inverse of the digamma function.
	args:
		y: [d,]
	"""
    
	torch_device = torch.device(device_name)
	#y.to(torch_device)
	# Initialize x following https://bariskurt.com/calculating-the-inverse-of-digamma-function/
	mask = (y >= -2.22).float() # [d, ]
	x = mask * (torch.exp(y) + 0.5) + (1 - mask) * - 1 / (y - psi(torch.tensor([1.]).to(torch_device))) # [d, ]

	# Newton iterations
	converged = False
	while not converged:
		x_old = deepcopy(x)
		x = x - (psi(x) - y) / torch.polygamma(1, x) # [d, ]
		converged = (x_old - x).norm() < 1e-6 * x_old.norm()
	return x

 
def multivariate_mle(Y: Tensor, a: float, c:float, alpha_init: float = 0.9, beta_init: float = 1.1, device_name = 'cpu') -> Tuple[float, float]:
	"""
	Estimates alpha and beta of a multivariate scaled beta distribution (assuming independence between coordinates).
	
	args: 
		Y: Scaled observations (following the notations of Wikipedia). Of Shape [N, d]

	returns:
		alpha
		beta
	"""
    
	alpha = alpha_init * torch.ones_like(Y[0]) # [d]
	beta = beta_init * torch.ones_like(Y[0]) # [d]

	const_1 = torch.log(Y - a).mean(0) - torch.log(c-a) # [d,]
	const_2 = torch.log(c - Y).mean(0) - torch.log(c-a)	# [d,]
	converged = False
	i = 0
	while (not converged) and i<500:
		alpha_old = deepcopy(alpha)
		alpha = inv_psi(psi(alpha + beta) + const_1, device_name)
		beta = inv_psi(psi(alpha + beta) + const_2, device_name)
		converged = (alpha - alpha_old).norm() < 1e-6 * alpha_old.norm()
		i +=1 
		#print(i)#if i%1000: print('mle it: ', i) #converged=True
	return alpha, beta

def test():
	device_name = 'cuda:0'
	torch_device = torch.device(device_name)
	N = 1000000
	d = 2
	a = torch.tensor(-0.15).to(torch_device)
	c = torch.tensor(1.15).to(torch_device)
	alpha = 3.0 * torch.ones(d)
	beta = 5.0 * torch.ones(d)
	dist = torch.distributions.Beta(alpha, beta)
	X = (dist.sample(torch.Size([N]))).to(torch_device)
	Y = X * (c - a) + a
    #Y = (X-a)/(c - a)
	t0 = time.time()
	alpha_mle, beta_mle = multivariate_mle(Y=Y,a=a,c=c, device_name=device_name)
	t1 = time.time()
	print(f"True parameters: alpha = {alpha} \t beta = {beta}")
	print(f"MLE took {t1 - t0} and found : alpha = {alpha_mle} \t beta = {beta_mle}")




if __name__ == '__main__':

	test()
