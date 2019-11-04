import numpy as np
import time
from sys import path
path.append('..')
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional, Dropout, Flatten
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import os
from src.logger import Logger
import sys


element_number = 1848
embedding_dim=200
num_words = 50000
embedding_matrix = np.zeros((num_words, embedding_dim))
TIME_STEPS=4
max_length=48
lstm_units=32
# first way attention
def attention_3d_block(inputs):
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    print(output_attention_mul.shape)
    return output_attention_mul

# 开始进行模型构建
model = Sequential()
# 模型第一层为Embedding,不需要进行训练
model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_length,
                    trainable=False))

# 模型第二层为BiLSTM,也可以尝试一下GRU，但是BiLSTM效果较好
bilstm_layer=Bidirectional(LSTM(units=32, return_sequences=True))
model.add(bilstm_layer)
# # 模型第三层为LSTM
# model.add(LSTM(units=16, return_sequences=False))

a = Permute((2, 1))(bilstm_layer)
a = Dense(TIME_STEPS, activation='softmax')(a)
a_probs = Permute((2, 1), name='attention_vec')(a)
# output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
output_attention_mul = multiply([bilstm_layer, a_probs], name='attention_mul')
print(output_attention_mul.shape)
model.add(output_attention_mul)

# attention_mul = attention_3d_block(Bidirectional(LSTM(units=32, return_sequences=True)))
# attention_flatten = Flatten()(attention_mul)
# drop2 = Dropout(0.3)(attention_flatten)
# model.add(attention_mul)
# model.add(attention_flatten)
# model.add(drop2)

# 通过sigmoid函数进行分类
model.add(Dense(1, activation='sigmoid'))


# 使用adam以1e-4的learning rate进行优化
optimizer = Adam(lr=1e-4)

# 编译模型
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# input_layer = Input(shape = (max_length, ))
# embedding_layer = Embedding(500000, output_dim = embedding_dim, mask_zero = True)
# bi_lstm_layer = Bidirectional(LSTM(64, return_sequences = True))
# bi_lstm_drop_layer = Dropout(0.5)
# dense_layer = TimeDistributed(Dense(1))
# # crf_layer = CRF(1, sparse_target = True)
#
# input = input_layer
# embedding = embedding_layer(input)
# bi_lstm = bi_lstm_layer(embedding)
# # bi_lstm_drop = bi_lstm_drop_layer(bi_lstm)
# attention_mul = attention_3d_block(bi_lstm)
# attention_flatten = Flatten()(attention_mul)
# dense = dense_layer(attention_flatten)
# # crf = crf_layer(dense)

model = Model(input = [input], output = [dense])
model.summary()
# 查看模型结构
print(model.summary())

if __name__=='__main__':
    pass
