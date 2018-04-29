import numpy as np

#x_t: n x d
# z: t x k
# B: k x d

def sample_b_k(x_t, y_t, z, t, k, beta, sigma_beta, B):
    #B_k is B without the kth row
    print("task:", t)
    print("k:", k)
    B_k = np.delete(B, (t), axis=0) 
    x_hat = z[t, k] * x_t
    sub = np.matmul(x_t,  np.transpose(B_k))
    z_temp = np.delete(z[t], (k), axis=0)
    # print(np.transpose(z_temp))
    # print(sub)
    sub = np.matmul(sub, np.transpose(z_temp))
    y_hat = y_t - sub
    variance = np.linalg.inv(beta * np.matmul(np.transpose(x_hat), x_hat) + sigma_beta)
    mean = np.matmul(variance, beta * np.matmul(np.transpose(x_hat), y_hat))
    # return mean, variance
    return np.random.normal(mean, variance)

def sample_


n = 3
d = 2
t = 4
k = 5
x_t = np.random.random((n,d))
y_t = np.random.random((n,1))
z = np.random.random((t,k))
B = np.random.random((k,d))
# print(x_t)
# print(z)
# print(B)
sigma_beta = np.random.random((d,d))
sample_b_k(x_t, y_t, z, 2, 2, 1, sigma_beta, B)
