import numpy as np

#x_t: n x d
# z: t x k
# b: k x d

def sample_b_k(x_t, y_t, z, t, k, beta, sigma_beta):
    x_hat = z[t, k] * x_t
    y_hat = y_t - x_t, 

