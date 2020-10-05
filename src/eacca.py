# =================================================================================
#
# 2020 by SungChul Hong (chulhongsung@gmail.com or hsc0526@uos.ac.kr) 
#
# =================================================================================
# #%%
import os
import xclib
from xclib.data import data_utils
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import preprocessing

from cca import cca
#%%
os.getcwd()
os.chdir('./dataset/Eurlex/')
#%%
features, tabels, num_samples, num_features, num_labels = data_utils.read_data('eurlex_train.txt')
features.todense().shape
tabels.todense().shape
#%%
label = tabels.todense()
rows, cols = np.where(label != 0)
label_lst = list(zip(rows, cols))
values = set(rows.tolist())
label_list = [[y[1]+1 for y in label_lst if y[0]==x] for x in values]
#%%
MAX_PAD_LENGTH =  max(map(lambda x: len(x), label_list))
#%%
padded_label = preprocessing.sequence.pad_sequences(label_list,
                                     maxlen=MAX_PAD_LENGTH,
                                     padding='post')
#%%
class Attention(K.layers.Layer):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.wq = K.layers.Dense(d_model)
        self.wk = K.layers.Dense(d_model)
        self.wv = K.layers.Dense(d_model)   
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def attention(self, q, k, v):
        matmul_qk = tf.linalg.matmul(q, k, transpose_b=True)
    
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
    
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
        attention_weights = tf.nn.softmax(tf.reduce_mean(scaled_attention_logits, axis=-1), axis=-1)[:, tf.newaxis, :]
        
        output = tf.matmul(attention_weights, v)

        return output, attention_weights
    
    def call(self, q, k, v):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        self_attention, _ = self.attention(q, k, v)
        attn_output = self.layernorm(self_attention)
        
        return attn_output
#%%
class Encoder(K.layers.Layer):
    def __init__(self, dim_seq, activation, l2_penalty, bias=True, constraint=None):
        super(Encoder,  self).__init__()
        self.n_layer = len(dim_seq)
        self.dense = [K.layers.Dense(dim_seq[i],
                                        activation=activation[i],
                                        use_bias=bias,
                                        kernel_constraint=constraint,
                                        kernel_regularizer=K.regularizers.L2(l2_penalty)) for i in range(self.n_layer)]
    
    def call(self, x):
        for i in range(self.n_layer):
            x = self.dense[i](x)
        return x
#%%
class DCCA(K.layers.Layer):
    def __init__(self, x_dim_seq, label_dim_seq, activation_f, activation_g, l2_penalty=0.0):
        super(DCCA, self).__init__()
        self.f = Encoder(x_dim_seq, activation_f, l2_penalty)
        self.g = Encoder(label_dim_seq, activation_g, l2_penalty, bias=False, constraint=None) #K.constraints.unit_norm()
    
    def call(self, x, y, dim, rcov1, rcov2):
        fx = self.f(x)
        gy = self.g(y)
        canonical_corr, e, u, v, __ = cca(fx, gy, dim, rcov1, rcov2)
        
        return canonical_corr, e, u, v, fx, gy
#%%
class Attention_Classifier(K.layers.Layer):
    def __init__(self, label_hot, d_model):
        super(Attention_Classifier, self).__init__()
        self.label_hot = label_hot
        self.wq = K.layers.Dense(d_model)
        self.wk = K.layers.Dense(d_model)
        self.wv = K.layers.Dense(d_model)   
        self.dense = K.layers.Dense(1, activation='sigmoid')
   
    def broadcast(self, q, k):
        n = q.shape[0]
        n_class = k.shape[0] #one_hot
        n_emb = q.shape[1]
        
        broadcast_q = tf.broadcast_to(q[:, tf.newaxis, :], [n, n_class, n_emb])
        multiply_mat = tf.math.multiply(broadcast_q, k)

        return multiply_mat
    
    def attention(self, q, k, v):
        matmul_qk = tf.linalg.matmul(q, k, transpose_b=True)
    
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
    
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights
    
    def call(self, q, k, v_mat):
        y = tf.linalg.matmul(k, v_mat, transpose_b=True)
        x = self.broadcast(q, y)    
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(self.label_hot)
        
        prob, _ = self.attention(q, k, v)
        output = self.dense(prob)
        
        return output
#%% EACCA
class EACCA(K.models.Model):
    
    def __init__(self, em_dim, d_model, x_dim_seq, label_dim_seq, activation_f, activation_g, num_labels, l2_penalty=0.0, rho=0.2):
        super(EACCA, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(num_labels+1, em_dim)
        self.self_attention = Attention(d_model)
        self.dcca = DCCA(x_dim_seq, label_dim_seq, activation_f, activation_g, l2_penalty)
        self.emb_total_label = self.embedding_layer(np.arange(num_labels))
        self.acls = Attention_Classifier(self.emb_total_label, d_model)
        self.rho = rho
        
    def call(self, x, y):
        emb_label = self.embedding_layer(y)
        sa_output = self.self_attention(emb_label, emb_label, emb_label)
        cc, e, u, v, fx, gy = self.dcca(x, tf.squeeze(sa_output), 30, 1e-6, 1e-6)
        output = self.acls(tf.linalg.matmul(fx, u, transpose_b=True), dcca.g(self.emb_total_label), v)
        self.add_loss(-self.rho * cc)
        
        return output
# %%
eacca = EACCA(50,
              50,
              [1000, 800, 500, 400, 200, 100],
              [300, 200, 150, 100],
              ['elu', 'elu', 'elu', 'elu', 'relu', 'relu'],
              ['elu', 'elu', 'relu', 'relu'],
              num_labels)
#%%
bce = tf.losses.BinaryCrossentropy()
optimizer = tf.optimizers.Adam(0.001)
#%%
max_iter = 10

for i in range(max_iter):
    with tf.GradientTape() as tape:    
        prob = eacca(features.todense()[0:1000], padded_label[0:1000])
        loss = bce(tf.squeeze(prob), label[0:1000])
    grad_ = tape.gradient(loss, eacca.weights)
    optimizer.apply_gradients(zip(grad_, eacca.weights))
    
    if (i+1) % 10 == 0:
        print("iteration {:03d}: Loss is {:.04f}.".format(i+1, loss.numpy()))
#%%
# features.todense()[0:100].shape
# self_attention = Attention(100)
# sa_output = self_attention(emb_list[0:100], emb_list[0:100], emb_list[0:100])
# dcca = DCCA([4000, 3000, 2000, 1500, 1000, 500, 400, 200, 100],
#      [300, 200, 150, 100],
#      ['elu', 'elu', 'elu', 'elu', 'elu', 'elu', 'elu', 'relu', 'relu'],
#      ['elu', 'elu', 'relu', 'relu'])
# cc, e, u, v, fx, gy = dcca(features.todense()[0:100], tf.squeeze(sa_output), 30, 1e-6, 1e-6)
# total_label = embedding_layer(np.arange(num_labels))
# embedding_layer = tf.keras.layers.Embedding(num_labels+1, 100)
# emb_list = embedding_layer(padded_label)
# acls = Attention_Classifier(total_label, 100)
# prob = acls(tf.linalg.matmul(fx, u, transpose_b=True), dcca.g(total_label), v)
# test = eacca(features.todense()[0:100], padded_label[0:100])