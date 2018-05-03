import numpy as np
from copy import deepcopy

#x_t: n x d
# z: t x k
# B: k x d

def sample_b_k(x_t, y_t, z, t, k, beta, sigma_beta, B):
    #B_k is B without the kth row
    # print("task:", t)
    # print("k:", k)
    B_k = np.delete(B, (t), axis=0)
    # print("x_t", x_t)
    # print("Z", z)
    x_hat = z[t, k] * x_t
    sub = np.matmul(x_t,  np.transpose(B_k))
    z_temp = np.delete(z[t], (k), axis=0)
    # print(np.transpose(z_temp))
    # print(sub)
    sub = np.matmul(sub, np.transpose(z_temp))
    y_hat = y_t - sub
    variance = np.linalg.inv(beta * np.matmul(np.transpose(x_hat), x_hat) + sigma_beta)
    mean = np.transpose(np.matmul(variance, beta * np.matmul(np.transpose(x_hat), y_hat)))[0]
    # return mean, variance
    # min_eig = np.min(np.real(np.linalg.eigvals(variance)))
    # if min_eig < 0:
    #     variance -= 10*min_eig * np.eye(*variance.shape)
    # print("mu",mean)
    # print("sigma",variance)
    return np.random.multivariate_normal(mean, variance)

def sample_z_t(x_t, y_t, z, beta, sigma_z, B):
    x_hat = np.matmul(x_t, np.transpose(B))
    # print(x_hat)
    variance = np.linalg.inv(beta* np.matmul(np.transpose(x_hat), x_hat) + sigma_z)
    mean = np.transpose(np.matmul(variance, beta * np.matmul(np.transpose(x_hat), y_t)))[0]
    # print("mean", mean)
    return np.random.multivariate_normal(mean, variance)
    # print("reached here")


n = 3
d = 2
t = 4
k = 5
x = []
y = []
for t1 in range(t):
    x.append(np.random.random((1,d)))
    y.append(np.random.random((1,1)))
sigma_beta = np.random.random((d,d))
sigma_z = np.random.random((k,k))
samples_B = []
samples_z = []
for iteration in range(100):
    z = np.random.random((t,k))
    B = np.random.random((k,d))
    for t1 in range(t):
        x_t = x[t1]
        y_t = y[t1]
        for k1 in range(k):
            B[k1] = sample_b_k(x_t, y_t, z, t1, k1, 1, sigma_beta, B) # Single input
        z[t1] =  sample_z_t(x_t, y_t, z, 1, sigma_z, B)

    # Append new samples to list
    samples_B.append(deepcopy(B))
    samples_z.append(deepcopy(z))

#print(len(samples_B))
#print(len(samples_z))

# Now for each t, set mu_t and sigma_t using the monte carlo samples from above
w = B.T.dot(z.T)
#print(.shape)
x_pred = np.random.random((1,d))
#print(w[:, 1].reshape((d,1)).T.dot(x_pred.T))
#print(w.shape)
for tno in range(t):
    cum_mu = 0
    for s in range(len(samples_B)):
        w = samples_B[s].T.dot(samples_z[s].T)
        cum_mu += w[:, 1].reshape((d,1)).T.dot(x_pred.T)[0]
    print(cum_mu)
#w_3 = B.T.dot(z[3])
#print(w_3.dot(np.random.random((1,d)).T))
