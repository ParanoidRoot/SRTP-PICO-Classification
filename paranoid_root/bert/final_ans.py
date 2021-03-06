from pathlib import Path
import pandas as pd
# import torch
import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense
# import keras.optimizers as optimizers
# import time
# from keras.models import load_model
# import os
from fast_bert.prediction import BertClassificationPredictor
# import datetime
from matplotlib import pyplot as plt


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


# 设置阈值
thresholds = dict(
    {
        'p': 0.45,
        'i': 0.53,
        'o': 0.45
    }
)
fine_grained_labels = [
    'posize',
    'podisease',
    'prdisease',
    'pogender',
    'potreatment',
    'poage',
    'prss',
    'poss',
    'poprocedure',
    'pophyconditon',
    'poclinical',
    'prbehavior',
    'pomedhistory',
    'iprocedure',
    'iss',
    'idiagnostic',
    'idisease',
    'idiagnostictest',
    'opatient',
    'otreatment'
]
pio_labels = ['p', 'i', 'o']
fine_grained_label_position_dict = dict()
fine_grained_ps = set(range(13))
fine_grained_is = set(range(13, 18))
fine_grained_os = set(range(18, 20))
fine_grained_label_position_dict['p'] = fine_grained_ps
fine_grained_label_position_dict['i'] = fine_grained_is
fine_grained_label_position_dict['o'] = fine_grained_os


def get_ans_array(temp_pd: pd.DataFrame, index_name_list: list):
    '''输入一个 pd, 转成一个 array, 按照细粒度的索引位置.'''
    ans_list = []
    for row_tuple in temp_pd.itertuples():
        current_list = []
        for index, fine_grained_class_name in enumerate(index_name_list):
            current_list.append(getattr(row_tuple, fine_grained_class_name))
        ans_list.append(current_list)
    return np.array(ans_list)


def get_correct_fine_grained_ans_array():
    original = pd.read_csv('./ans/final/correct.csv')
    return get_ans_array(original, fine_grained_labels)


def get_predicted_fine_grained_ans_array():
    original = pd.read_csv('./ans/final/predicted.csv')
    return get_ans_array(original, fine_grained_labels)


def get_predicted_pio_ans_array():
    original = pd.read_csv('./ans/sentence/predicted.csv')
    return get_ans_array(original, pio_labels)


def get_correct_pio_ans_array():
    original = pd.read_csv('./ans/sentence/correct.csv')
    return get_ans_array(original, pio_labels)


def get_correct_fine_grained_index(label, curr_cfg_array):
    '''输入一个数组, 输出一个标签下细粒度正确的位置.'''
    positions = []
    for i in fine_grained_label_position_dict[label]:
        if float(curr_cfg_array[i]) == 1.0:
            positions.append(i)
    return positions


def get_max_index(a: np.array, _from, _included_to):
    '''从一个数组中的某个范围找到最大值的下标.'''
    max_index, max_value = None, -1.0
    for i in range(_from, _included_to + 1):
        if a[i] > max_value:
            max_value = a[i]
            max_index = i
    return max_index


def judge_right(
    label: str, threshold: float,
    curr_ppio_array: np.array, curr_cpio_array: np.array,
    curr_pfg_array: np.array, curr_cfg_array: np.array
):
    index = pio_labels.index(label)
    predicted_is_in_label = (curr_ppio_array[index] > threshold)
    cpio_indexes = get_correct_fine_grained_index(label, curr_cfg_array)
    if len(cpio_indexes) <= 0:
        # 一个也没有, 不是这个类别
        return not predicted_is_in_label
    else:
        # 在这个类别中, 有回答的结果
        if not predicted_is_in_label:
            return False
        else:
            temp = list(fine_grained_label_position_dict[label])
            biggest_index = curr_pfg_array[
                min(temp): max(temp) + 1
            ].argmax() + min(temp)
            return biggest_index in cpio_indexes


def top_1_correctness(label, threshold, cfg, pfg, cpio, ppio):
    '''
    1. 首先对某一个大类的标签设置一个阈值.
    2. 计算在采用这个阈值的时候 top1 方法有多少的准确率.
    '''
    right_count = 0
    total_count = 0
    for i in range(len(cfg)):
        judge_res = judge_right(
            label, threshold,
            ppio[i], cpio[i],
            pfg[i], cfg[i]
        )
        total_count += 1
        right_count += (1 if judge_res else 0)
    return right_count / total_count


def get_height_line_index(_ys, _correctness=0.9):
    '''找出所有大于0.9的值中方差最小的值.'''
    temp = []
    for _y in _ys:
        if _y > _correctness:
            temp.append(_y)
    target_ys = np.array(temp)
    temp_ys = np.zeros(target_ys.shape[0])
    current_min_val = 1000000000.0
    current_min_index = None
    for pos, target in enumerate(target_ys):
        temp_ys[pos] = np.sum(np.sqrt((target_ys - target) ** 2))
        if temp_ys[pos] < current_min_val:
            current_min_val = temp_ys[pos]
            current_min_index = pos
    current_min_val = target_ys[current_min_index]
    for i, v in enumerate(_ys):
        if v == current_min_val:
            return i


def top_1_with_thresholds(label, cpio, ppio, cfg, pfg, _from, _to, _num):
    '''枚举每种可能的情况, 获取到曲线.'''
    xs = []
    ys = []
    for threshold in np.linspace(_from, _to, _num):
        xs.append(threshold)
        ys.append(
            top_1_correctness(
                label, threshold,
                cfg, pfg,
                cpio, ppio
            )
        )
    # 画图
    plt.figure()
    plt.xlabel('threshold')
    plt.ylabel('correctness of top 1')
    plt.title('%s: threshold - top1 correctness' % label)
    plt.plot(xs, ys, color='green')
    index = get_height_line_index(ys, 0.4)
    plt.plot(xs, [ys[index]] * len(xs), linestyle='-.')
    save_path = Path(
        './ans/save_for_bset_model/%s-%.2f-%.2f.png' %
        (label, xs[index], ys[index])
    )
    save_path = str(
        save_path.absolute()
    )
    plt.savefig(save_path)
    plt.close()
    return xs[index], ys[index]


def main():
    print('start'.center(31, '*'))
    cfg = get_correct_fine_grained_ans_array()
    pfg = get_predicted_fine_grained_ans_array()
    cpio = get_correct_pio_ans_array()
    ppio = get_predicted_pio_ans_array()
    # 测试
    top_1_with_thresholds(
        'p', cpio, ppio, cfg, pfg, 0.0, 0.90, 100
    )
    top_1_with_thresholds(
        'i', cpio, ppio, cfg, pfg, 0.0, 0.90, 100
    )
    top_1_with_thresholds(
        'o', cpio, ppio, cfg, pfg, 0.0, 0.90, 100
    )
    print('finish'.center(31, '*'))


def get_sentence_label_array(
    text,
    label_list: list,
    predictor: BertClassificationPredictor
):
    '''输入一个句子, 输出 pio 的分类.'''
    ans_list = predictor.predict(text)
    ans_array = np.zeros(len(label_list))
    for ans in ans_list:
        label = ans[0]
        index = label_list.index(label)
        ans_array[index] = ans[1]
    return ans_array


def get_pio_labels(pio_array):
    ans_list = []
    for i, v in enumerate(pio_labels):
        if pio_array[i] > thresholds[v]:
            ans_list.append(v)
    return ans_list


def get_fg_labels(curr_pio_labels, fg_array):
    '''获取到分类分出的结果.'''
    fg_labels = []
    for pio_label in curr_pio_labels:
        fg_label_index_set = fine_grained_label_position_dict[pio_label]
        min_index = min(fg_label_index_set)
        max_index = max(fg_label_index_set)
        temp = fg_array[min_index: max_index + 1]
        where = temp.argmax() + min_index
        fg_labels.append(
            fine_grained_labels[where]
        )
    return fg_labels


def get_sentences():
    return [
        "this prospective double blind study was undertaken to compare "
        "the safety and efﬁcacy of oral vs vaginal misoprostol in equivalent doses "
        "50 mg for induction of labour",
        "A total of 128 term pregnancies with indication for induction of "
        "labour were allocated to two groups to receive 50 mg misoprostol "
        "orally or vaginally, every 4 h until adequate contractions were achieved "
        "or a maximum of 200 mg dose",
        "Induction to delivery interval was signiﬁcantly shorter in the "
        "vaginal group compared with the oral group 14.6 h vs 22.5 h p50.001",
        "There was no signiﬁcant difference between the groups with respect "
        "to mode of delivery neonatal outcome and maternal side effects"
    ]


def test():
    print('start'.center(31, '*'))
    sentence_predictor = get_predictor('sentence')
    fine_grained_predicor = get_predictor('final')

    sentences = get_sentences()
    for i, sentence in enumerate(sentences):
        print((' %d ' % i).center(31, '*'))
        print(sentence)
        pio_array = get_sentence_label_array(
            sentence, pio_labels, sentence_predictor
        )
        fg_array = get_sentence_label_array(
            sentence, fine_grained_labels, fine_grained_predicor
        )
        current_pio_labels = get_pio_labels(pio_array)
        print('pios: ')
        print(pio_array)
        print(current_pio_labels)
        print('fgs: ')
        print(fg_array)
        print(get_fg_labels(current_pio_labels, fg_array))
        print()


if __name__ == "__main__":
    main()
    # test()
