import numpy as np

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
z = np.random.random((t,k))
B = np.random.random((k,d))
# print(x_t)
# print(z)
# print(B)
sigma_beta = np.random.random((d,d))
sigma_z = np.random.random((k,k))
# print("b samples", sample_b_k(x_t, y_t, z, 2, 2, 1, sigma_beta, B))
# print("z samples", sample_z_t(x_t, y_t, z, 2, 2, 1, sigma_z, B))
for iteration in range(100):
    for t1 in range(t):
        x_t = x[t1]
        y_t = y[t1]
        for k1 in range(k):
            B[k1] = sample_b_k(x_t, y_t, z, t1, k1, 1, sigma_beta, B)
            # print(iteration)
        z[t1] =  sample_z_t(x_t, y_t, z, 1, sigma_z, B)

# B[k1] = 
