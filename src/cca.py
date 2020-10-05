# ======================================================================================
#
# Update cca.py written by by Weiran Wang and Qingming Tang in 2019 for Tensorflow2.0.  
#
# ======================================================================================

#%%
import numpy as np
import tensorflow as tf
#%%
eps_eig = 1e-6
def cca(h1, h2, dim, rcov1, rcov2):
    n = h1.shape[0]
    d1 = h1.shape[1]
    d2 = h2.shape[1]
    # Normalization
    m1 = tf.math.reduce_mean(h1, axis=0, keepdims=True)
    h1 = tf.subtract(h1, m1)

    m2 = tf.math.reduce_mean(h2, axis=0, keepdims=True)
    h2 = tf.subtract(h2, m2)

    # Correlation
    s11 = tf.linalg.matmul(h1, h1, transpose_a=True) / (n-1) + rcov1 * tf.eye(d1)
    s22 = tf.linalg.matmul(h2, h2, transpose_a=True) / (n-1) + rcov2 * tf.eye(d2)
    s12 = tf.linalg.matmul(h1, h2, transpose_a=True) / (n-1)
    
    # Eigen decompostion of adjoint matrix
    e1, v1 = tf.linalg.eigh(s11)
    e2, v2 = tf.linalg.eigh(s22)

    # For numerical stability.
    idx1 = tf.where(e1>eps_eig)[:,0]
    e1 = tf.gather(e1, idx1)
    v1 = tf.gather(v1, idx1, axis=1)

    idx2 = tf.where(e2>eps_eig)[:,0]
    e2 = tf.gather(e2, idx2)
    v2 = tf.gather(v2, idx2, axis=1)

    k11 = tf.linalg.matmul(tf.linalg.matmul(v1, tf.linalg.diag(tf.math.reciprocal(tf.math.sqrt(e1)))), v1, transpose_b=True)
    k22 = tf.linalg.matmul(tf.linalg.matmul(v2, tf.linalg.diag(tf.math.reciprocal(tf.math.sqrt(e2)))), v2, transpose_b=True)
    T1 = tf.linalg.matmul(tf.linalg.matmul(k11, s12), k22)
    T2 = tf.linalg.matmul(tf.linalg.matmul(k22, s12, transpose_b=True), k11)
    
    
    # Eigenvalues are sorted in increasing order.
    e3, u = tf.linalg.eigh(tf.linalg.matmul(T1, T1, transpose_b=True))
    _, v = tf.linalg.eigh(tf.linalg.matmul(T2, T2, transpose_b=True))
    idx3 = tf.where(e3 > eps_eig)[:, 0]
    # This is the thresholded rank.
    dim_svd = tf.cond(tf.size(idx3) < dim, lambda: tf.size(idx3), lambda: dim)

    return tf.math.reduce_sum(tf.sqrt(e3[-dim_svd:])), e3, tf.linalg.matmul(u, k11, transpose_a=True), tf.linalg.matmul(v, k22, transpose_a=True), dim_svd
# %%
