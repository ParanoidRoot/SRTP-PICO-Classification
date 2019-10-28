from pathlib import Path
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers as optimizers
import time
from keras import callbacks as kcbs
from keras.models import load_model
import os
from fast_bert.prediction import BertClassificationPredictor
import datetime


class Record(object):
    '''记录对象.'''
    p_pos = 9
    i_pos = 0
    o_pos = 6
    labels_sequence = None
    # 设置默认参数
    p_threshold = 0.346
    i_threshold = 0.243
    o_threshold = 0.281

    def __init__(self, _as: dict, _ps: dict, _vs: dict):
        self.attributes = _as.copy()  # 就是 id, text, file_path
        self.predicts = _ps.copy()
        self.values = _vs.copy()
        self.predicted_tensor = None
        self.value_tensor = None

    def __str__(self):
        return (
            str(self.attributes) + '\r\n' +
            str(self.predicts) + '\r\n' +
            str(self.values) + '\r\n'
        )

    def init_label_tensor(self):
        '''获取预测以及真实情况的两个 tensor.'''
        predicted_tensor = torch.zeros(len(Record.labels_sequence))
        value_tensor = torch.zeros(len(Record.labels_sequence))
        for index, label_name in enumerate(Record.labels_sequence):
            predicted_tensor[index] = self.predicts[label_name]
            value_tensor[index] = self.values[label_name]
        self.predicted_tensor, self.value_tensor = (
            predicted_tensor, value_tensor
        )

    @property
    def id(self):
        return self.attributes['id']

    @property
    def text(self):
        return self.attributes['text']

    @property
    def file_path(self):
        return self.attributes['file_path']

    @classmethod
    def compare_tensors(cls, *tensors):
        '''比较两个向量.'''
        format_string = ['{0:3d} {1:15s}']
        parts = [('{%d:8.5f}' % (i + 2)) for i in range(len(tensors))]
        format_string.extend(parts)
        format_string = ' '.join(format_string)
        for i, v in enumerate(cls.labels_sequence):
            temp = [i, v]
            temp.extend([tensor[i] for tensor in tensors])
            print(
                format_string.format(
                    *(temp)
                )
            )

    def get_final_tensor_1(self):
        '''运用一些处理的方法获取到需要的 tensor.'''
        # 方法一找出句子的类别的每个类别中的最大值
        ans_tensor = self.predicted_tensor.clone()
        if ans_tensor[Record.i_pos] > Record.i_threshold:
            ans_tensor[Record.i_pos] = 1.0
            i_tensor = ans_tensor[Record.i_pos + 1: Record.o_pos]
            i_max = i_tensor.max()
            i_tensor[i_tensor != i_max] = 0.0
            i_tensor[i_tensor == i_max] = 1.0
        else:
            ans_tensor[Record.i_pos: Record.o_pos] = 0.0
        if ans_tensor[Record.o_pos] > Record.o_threshold:
            ans_tensor[Record.o_pos] = 1.0
            o_tensor = ans_tensor[Record.o_pos + 1: Record.p_pos]
            o_max = o_tensor.max()
            o_tensor[o_tensor != o_max] = 0.0
            o_tensor[o_tensor == o_max] = 1.0
        else:
            ans_tensor[Record.o_pos: Record.p_pos] = 0.0
        if ans_tensor[Record.p_pos] > Record.p_threshold:
            ans_tensor[Record.p_pos] = 1.0
            p_tensor = ans_tensor[Record.p_pos + 1:]
            p_max = p_tensor.max()
            p_tensor[p_tensor != p_max] = 0.0
            p_tensor[p_tensor == p_max] = 1.0
        else:
            ans_tensor[Record.p_pos:] = 0.0
        return ans_tensor

    def cal_final_rate_1(self):
        '''计算算法一是否正确.'''
        final_tensor = self.get_final_tensor_1()
        temp_value_tensor = self.value_tensor.clone()
        temp_value_tensor[Record.p_pos] = 0.0
        temp_value_tensor[Record.o_pos] = 0.0
        temp_value_tensor[Record.i_pos] = 0.0
        final_tensor = final_tensor * temp_value_tensor
        return int(final_tensor.sum()), int(temp_value_tensor.sum())

    def get_final_tensor_2(self):
        '''用深度学习再来学.'''
        predicted_array = np.array(self.predicted_tensor).copy()
        if predicted_array[Record.i_pos] > Record.i_threshold:
            predicted_array[Record.i_pos] = (
                (predicted_array[Record.i_pos] - Record.i_threshold) /
                (1 - Record.i_threshold)
            )
        else:
            predicted_array[Record.i_pos: Record.o_pos] = 0.0
        if predicted_array[Record.o_pos] > Record.o_threshold:
            predicted_array[Record.o_pos] = (
                (predicted_array[Record.o_pos] - Record.o_threshold) /
                (1 - Record.o_threshold)
            )
        else:
            predicted_array[Record.o_pos: Record.p_pos] = 0.0
        if predicted_array[Record.p_pos] > Record.p_threshold:
            predicted_array[Record.p_pos] = (
                (predicted_array[Record.p_pos] - Record.p_threshold) /
                (1 - Record.p_threshold)
            )
        else:
            predicted_array[Record.p_pos:] = 0.0
        _value_array = np.array(self.value_tensor).copy()
        _value_array[Record.i_pos] = 0.0
        _value_array[Record.o_pos] = 0.0
        _value_array[Record.p_pos] = 0.0
        return predicted_array, _value_array


def get_records():
    # 读入数据
    file_path = Path('./data/final/train.csv')
    original_data = pd.read_csv(str(file_path.absolute()))
    # 处理列名分为三类
    attribute_column_names = set()
    labels = set()
    column_names = original_data.columns.values
    for column_name in column_names:
        if column_name.startswith('predicted_'):
            labels.add(column_name.replace('predicted_', ''))
        elif not column_name.startswith('value_'):
            attribute_column_names.add(column_name)
        else:
            pass
    # 把每一行记录转成一个对象.
    records = []
    for row_tuple in original_data.itertuples():
        # 首先获取一个属性集合
        _as = dict()
        for attribute_column_name in attribute_column_names:
            _as[attribute_column_name] = getattr(
                row_tuple, attribute_column_name
            )
        _ps = dict()
        _vs = dict()
        for label in labels:
            _ps[label] = getattr(
                row_tuple, 'predicted_' + label
            )
            _vs[label] = getattr(
                row_tuple, 'value_' + label
            )
        records.append(Record(_as, _ps, _vs))
    return list(labels), records


labels, records = get_records()
labels = sorted(labels)
records = records[:-2]
Record.labels_sequence = labels.copy()


def final_main():
    # 训练或者使用某种算法获取最终的分类.
    
    for record in records:
        record.init_label_tensor()

    # 计算算法-1-正确率
    def function_1():
        right_num = 0
        total_num = 0
        for record in records:
            current_right, current_total = record.cal_final_rate_1()
            right_num += current_right
            total_num += current_total
        print(right_num, total_num)
        print(float(right_num) / float(total_num))

    # 构建数据集
    x_train, y_train = [], []
    # for record in records:
    # x_train.append(
    #     np.array(record.predicted_tensor)
    # )
    # y_train.append(
    #     np.array(record.value_tensor)
    # )
    for record in records:
        a1, a2 = record.get_final_tensor_2()
        x_train.append(a1)
        y_train.append(a2)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    original_data_x = x_train.copy()
    original_data_y = y_train.copy()
    # 分隔数据集与测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train,
        test_size=0.20,
        random_state=31
    )

    # 计算算法-2-正确率
    def function_2():
        '''使用 keras 模块来学习.'''
        # 开始训练的时间
        what_time_str = time.strftime(
            '%Y%m%d-%H%M%S',
            time.localtime(time.time())
        )
        # 保存模型的路径
        save_model_path = './models/final/model-%s.h5' % (what_time_str)
        # 搭建模型
        model = Sequential(
            [
                Dense(128, input_dim=23, activation='relu'),
                # Dense(256, activation='relu'),
                # Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(23, activation='sigmoid')
            ]
        )
        # 优化器, 减少学习时间
        adam = optimizers.Adam(lr=0.00001, epsilon=1e-08, decay=0.0)
        # 回调函数
        callbacks = [
            kcbs.EarlyStopping(
                monitor='loss',  # 监控模型的验证精度
                patience=10  # 如果精度在多于一轮的时间（即两轮）内不再改善，就中断训练
            ),
            # ModelCheckpoint用于在每轮过后保存当前权重
            kcbs.ModelCheckpoint(
                filepath=save_model_path,  # 目标文件的保存路径
                monitor='loss',
                save_best_only=True
            )
        ]
        # 编译模型
        model.compile(
            optimizer=adam,
            loss='mean_squared_logarithmic_error',
            metrics=['acc']
        )
        # 训练模型
        model.fit(
            x_train, y_train,
            epochs=60, batch_size=4,
            callbacks=callbacks,
            validation_split=0.2
        )
        loss, accur = model.evaluate(x_test, y_test)
        print()
        print()
        print(''.center(31, '*'))
        print('final:', 'loss =', loss, 'accur =', accur)
        return save_model_path

    def function_2_load(model_base_name, _x_test, _y_test):
        model_path = './models/final/' + model_base_name
        # 加载模型
        model = load_model(model_path)
        # 预测结果
        print(''.center(31, '*'))
        loss, acc = model.evaluate(
            _x_test, _y_test
        )
        print(loss, acc)

    function_2()
    # function_2_load('model-20191027-185710.h5')
    # function_2_load('new1.h5', original_data_x, original_data_y)


def get_inputted_sentences() -> list:
    return [
        'patients with psoriasis n 15 '
        'treated identically and healthy subjects not receiving any therapy '
        'n 15 served as controls'
    ]


# 清空 cache
torch.cuda.empty_cache()

# 配置最长显示长度以及当前日期
pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

# 提示开始工作
os.system('cls')
os.system('cls')
print('Start'.center(25, '*'))


def fuck():

    def get_predictor(train_for):
        # 开始构建预测模型
        output_dir = Path('./models/%s/output/model_out' % train_for)
        label_dir = Path('./labels/%s/' % train_for)
        predictor = BertClassificationPredictor(
            model_path=output_dir,
            label_path=label_dir,
            multi_label=True,
            model_type='bert',
            do_lower_case=True
        )
        return predictor

    # 获取到句子的预测
    sentence_predictor = get_predictor('sentence')
    # 在获取细粒度的预测模型
    combined_predictor = get_predictor('combined')
    # 最后获取到最后的一层分析模型
    final_predictor = load_model('./models/final/model-20191027-204628.h5')

    # 将句子分析为一个 array
    def parse_sentence(sentence):
        # final dict
        input_dict = dict()
        # p, i, o dict
        pios = sentence_predictor.predict(sentence)
        for t in pios:
            input_dict[t[0]] = t[1]
        # 细粒度的 dict
        parts = combined_predictor.predict(sentence)
        for part in parts:
            input_dict[part[0]] = part[1]
        # 构建出一个 array
        input_array = np.zeros(len(Record.labels_sequence))
        for i, v in enumerate(Record.labels_sequence):
            input_array[i] = input_dict[v]
        # 处理一个 array
        if input_array[Record.i_pos] > Record.i_threshold:
            input_array[Record.i_pos] = 0.0
        else:
            input_array[Record.i_pos:Record.o_pos] = 0.0
        if input_array[Record.o_pos] > Record.o_threshold:
            input_array[Record.o_pos] = 0.0
        else:
            input_array[Record.o_pos:Record.p_pos] = 0.0
        if input_array[Record.p_pos] > Record.p_threshold:
            input_array[Record.p_pos] = 0.0
        else:
            input_array[Record.p_pos:] = 0.0
        # 然后预测结果
        output_array = final_predictor.predict(
            np.array(
                [input_array]
            )
        )
        # 处理分析输出结果
        temp = output_array.reshape((23,)).copy()
        temp[input_array == 0.0] = 0.0
        # 最多两个 标签
        first_big_index = temp.argmax()
        first_big_value = temp[first_big_index]
        temp[first_big_index] = 0.0
        second_big_index = temp.argmax()
        second_big_value = temp[second_big_index]
        # 返回输出标签的列表
        return (
            (
                first_big_index, first_big_value,
                Record.labels_sequence[first_big_index]
            ), (
                second_big_index, second_big_value,
                Record.labels_sequence[second_big_index]
            )
        )

    print(parse_sentence(get_inputted_sentences()[0]))

    # 获取到今天训练的直接预测
    fine_grained_sentence_predictor = get_predictor('final')
    print()


if __name__ == "__main__":
    fuck()
