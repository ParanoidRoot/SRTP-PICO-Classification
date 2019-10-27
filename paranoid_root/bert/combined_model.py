from pathlib import Path
import torch

import pandas as pd
import os
import datetime


from fast_bert.prediction import BertClassificationPredictor

# 清空 cache
torch.cuda.empty_cache()

# 配置最长显示长度以及当前日期
pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

# 提示开始工作
os.system('cls')
os.system('cls')
print('Start'.center(25, '*'))


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


def get_inputted_sentences() -> list:
    return [
        'patients with psoriasis n 15 '
        'treated identically and healthy subjects not receiving any therapy '
        'n 15 served as controls'
    ]


def combined_model_main():
    # 获取到两个 predictor
    sentence_predictor = get_predictor('sentence')
    fine_grained_predictor = get_predictor('fine_grained')

    # 配置三个常量
    p_threshold = 0.346
    i_threshold = 0.243
    o_threshold = 0.281

    # 获取句子
    sentences = get_inputted_sentences()

    # 对每个句子处理
    def get_one_sentence_classification(sentence):
        temp_anses = sentence_predictor.predict(sentence)
        ans_list = list()
        for temp_ans in temp_anses:
            if temp_ans[0] == 'p':
                # 属于 p 的
                if temp_ans[1] > p_threshold:
                    ans_list.append(
                        (
                            'p',
                            (temp_ans[1] - p_threshold) / (1 - p_threshold)
                        )
                    )
            elif temp_ans[0] == 'i':
                if temp_ans[1] > i_threshold:
                    ans_list.append(
                        (
                            'i',
                            (temp_ans[1] - i_threshold) / (1 - i_threshold)
                        )
                    )
            elif temp_ans[0] == 'o':
                if temp_ans[1] > o_threshold:
                    ans_list.append(
                        (
                            'o',
                            (temp_ans[1] - o_threshold) / (1 - o_threshold)
                        )
                    )
            else:
                raise KeyError
        ans_list = sorted(
            ans_list,
            key=lambda ans_tuple: ans_tuple[1],
            reverse=True
        )
        return ans_list

    # # 对一个小分句子进行分类
    # def get_fine_grained_clsf(fine_grained_content, sentence_clsf):
    #     '''注意这个小部分是固定长度的.'''
    #     fine_grained_predicted_ans = (
    #         fine_grained_predictor.predict(fine_grained_content)
    #     )
    #     # 汇聚结果
    


if __name__ == '__main__':
    combined_model_main()
