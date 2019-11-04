'''进行Bi-lstm模型训练'''
import numpy as np
import time
from sys import path

path.append('..')
from preProc.preProcess import preporcess_elements
from src.util import get_all_elements, ele_dir
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




# 准备训练集
# 加载数据集句子
pico_elements = get_all_elements(ele_dir)
presplited_elements = []
for i in range(len(pico_elements)):
    presplited_elements.append(pico_elements[i].sentence)

# 文本预处理
tokenized_elements, embedding_matrix, num_words, embedding_dim, max_length = preporcess_elements(presplited_elements)
# 准备target向量，大小为样本数量,标签为PIO粗粒度
train_target_P = np.zeros(len(pico_elements))
train_target_I = np.zeros(len(pico_elements))
train_target_O = np.zeros(len(pico_elements))
for i in range(len(pico_elements)):
    if pico_elements[i].label[0].lower() == 'p':
        train_target_P[i] = 1
    elif pico_elements[i].label[0].lower() == 'i':
        train_target_I[i] = 1
    elif pico_elements[i].label[0].lower() == 'o':
        train_target_O[i] = 1
    else:
        print("no such class")

'''创建lstm模型'''

# first way attention
# def attention_3d_block(inputs):
#     # input_dim = int(inputs.shape[2])
#     a = Permute((2, 1))(inputs)
#     a = Dense(TIME_STEPS, activation='softmax')(a)
#     a_probs = Permute((2, 1), name='attention_vec')(a)
#     # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
#     return output_attention_mul


def create_lstm_model():
    # 开始进行模型构建
    model = Sequential()
    # 模型第一层为Embedding,不需要进行训练
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_length,
                        trainable=False))

    # 模型第二层为BiLSTM,也可以尝试一下GRU，但是BiLSTM效果较好
    model.add(Bidirectional(LSTM(units=32, return_sequences=True)))

    # 模型第三层为LSTM
    model.add(LSTM(units=16, return_sequences=False))

    # #若只使用bilstm则需要flatten
    # model.add(Flatten())

    # 通过sigmoid函数进行分类
    model.add(Dense(1, activation='sigmoid'))

    # # 通过tahn函数进行分类
    # model.add(Dense(1, activation='tanh'))

    # 使用adam以1e-4的learning rate进行优化
    optimizer = Adam(lr=1e-4)

    # 编译模型
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


    #加入注意力机制的模型构建
    inputs=Input(shape=(num_words,))

    # 查看模型结构
    print(model.summary())
    return model


'''训练lstm模型'''


def train_lstm_model(train_target, model, path_checkpoint):
    # 交叉验证，90%训练，10%验证
    X_train, X_test, y_train, y_test = train_test_split(tokenized_elements,
                                                        train_target,
                                                        test_size=0.1,
                                                        random_state=12)

    # 建立一个权重存储点
    checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_weights_only=True,
                                 save_best_only=True)

    # 尝试加载已有的模型权重
    # try:
    #     model.load_weights(path_checkpoint)
    # except Exception as e:
    #     print('No existed weights,train a raw model')

    # 定义early stopping 三个epoch内validation loss没有明显变化则停止
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # 自动降低learning rate
    lr_reduction = ReduceLROnPlateau(monitor='val_loss'
                                     , factor=0.1,
                                     min_lr=0,
                                     patience=0,
                                     verbose=1)

    # 定义callback函数
    callbacks = [
        earlystopping,
        checkpoint,
        lr_reduction
    ]

    # callbacks = [
    #     checkpoint,
    #     lr_reduction
    # ]

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用编号为2号的GPU,好像有点问题

    # 开始训练
    history = model.fit(X_train, y_train,
                        validation_split=0.1,
                        epochs=20,
                        batch_size=32,
                        callbacks=callbacks)

    # 查看训练结果
    result = model.evaluate(X_test, y_test)
    print('Accuracy:{0:.2%}'.format(result[1]))

    return history


def show_loss(trained_model, file_path, model_name):
    pyplot.plot(trained_model.history['loss'])
    pyplot.plot(trained_model.history['val_loss'])
    pyplot.title('model train vs validation loss of ' + model_name)
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.xlim((0, 20))
    pyplot.ylim((0,1))
    pyplot.legend(['train', 'validation'], loc='upper right')
    file_name = 'trained_loss_' + model_name + '.png'
    file_full_path = os.path.join(file_path, file_name)
    pyplot.savefig(file_full_path)
    pyplot.close()


# 创建三个分类的model
model_P = create_lstm_model()
model_I = create_lstm_model()
model_O = create_lstm_model()
# 一会儿重新训练一下
checkpoint_path_P = 'pico_checkpoint_Label_P.keras'
checkpoint_path_I = 'pico_checkpoint_Label_I.keras'
checkpoint_path_O = 'pico_checkpoint_Label_O.keras'

if __name__ == '__main__':
    logger_path = "../log/logger_train-{}.txt".format(time.strftime('%m%d-%H%M', time.localtime(time.time())))
    sys.stdout = Logger(logger_path)  # 打印控制台内容作为日志

    abspath = os.path.abspath('.')
    output_path = os.path.join(abspath, 'outputs')

    # 训练模型并保存loss为图片
    history1 = train_lstm_model(train_target_P, model_P, checkpoint_path_P)
    show_loss(history1, output_path, 'Label_P')

    history2 = train_lstm_model(train_target_I, model_I, checkpoint_path_I)
    show_loss(history1, output_path, 'Label_I')

    history3 = train_lstm_model(train_target_O, model_O, checkpoint_path_O)
    show_loss(history1, output_path, 'Label_O')
    pass
