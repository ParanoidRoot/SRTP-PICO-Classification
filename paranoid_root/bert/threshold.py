# 在这个文件中计算出需要设计的阈值
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PredictedResult(object):
    '''构造结果记录对象.'''

    threshold = None

    def __init__(self, *args, **kwargs):
        self.text = kwargs['text']
        self.predicted_p = kwargs['predicted_p']
        self.predicted_i = kwargs['predicted_i']
        self.predicted_o = kwargs['predicted_o']
        self.value_p = kwargs['value_p']
        self.value_i = kwargs['value_i']
        self.value_o = kwargs['value_o']

    def __str__(self):
        temp = '%.4f %.4f'
        return (
            self.text[: min(10, len(self.text))],
            temp % (self.predicted_p, self.value_p),
            temp % (self.predicted_i, self.value_i),
            temp % (self.predicted_o, self.value_o)
        ).__str__()

    def verify_prediction(self, label_name):
        predicted_ans = (
            getattr(self, 'predicted_%s' % label_name) >=
            PredictedResult.threshold
        )
        value_ans = (float(getattr(self, 'value_%s' % label_name)) == 1.0)
        return predicted_ans == value_ans


def main_4_whole_sentence():
    '''计算出整个句子的 p, i, o 分类的阈值.'''
    file_path = './ans/output_bert_00.csv'
    final_pd = pd.read_csv(file_path)
    results = dict()
    for row_tuple in final_pd.itertuples():
        results[row_tuple.index] = PredictedResult(
            text=row_tuple.text,
            predicted_p=row_tuple.predicted_p,
            predicted_i=row_tuple.predicted_i,
            predicted_o=row_tuple.predicted_o,
            value_p=row_tuple.value_p,
            value_i=row_tuple.value_i,
            value_o=row_tuple.value_o
        )
    xs = np.linspace(0.05, 0.5, num=70, endpoint=False)

    def calculate_correctness(_results: dict, label_name, _xs):
        ys = np.zeros(xs.shape[0])
        for pos, x in enumerate(_xs):
            PredictedResult.threshold = x
            total_num = 0
            correct_num = 0
            for index, predicted_res in _results.items():
                total_num += 1
                correct_num += (
                    1 if predicted_res.verify_prediction(label_name) else 0
                )
            ys[pos] = correct_num / total_num
        return ys

    pys = calculate_correctness(results, 'p', xs)
    iys = calculate_correctness(results, 'i', xs)
    oys = calculate_correctness(results, 'o', xs)

    def draw_picture(_xs, _ys, label_name, pic_name):
        plt.figure()
        plt.xlabel('thresholds')
        plt.ylabel('correctness')
        plt.title(label_name)
        plt.plot(_xs, _ys)
        plt.savefig(pic_name)
        plt.close()

    # fp = np.poly1d(np.polyfit(xs, pys, 5))

    # draw_picture(xs, pys, 'p', './ans/reshold-p_correctness')
    # draw_picture(xs, iys, 'i', './ans/reshold-i_correctness')
    # draw_picture(xs, oys, 'o', './ans/reshold-o_correctness')
    # draw_picture(xs, fp(xs), 'plot_fit_p', './ans/reshold-plot_fit_p')

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

    p_height_index = get_height_line_index(pys, 0.9)
    i_height_index = get_height_line_index(iys, 0.9)
    o_height_index = get_height_line_index(oys, 0.9)

    def draw_picture_with_line(
        _xs, _ys, label_name, pic_path, height_line_cross_index
    ):
        plt.figure()
        plt.xlabel('thresholds')
        plt.ylabel('correctness')
        plt.title(label_name)
        plt.plot(_xs, _ys)
        line_xs = np.linspace(0.05, 0.5, num=10)
        temp = np.ones(line_xs.shape[0]) * _ys[height_line_cross_index]
        plt.plot(line_xs, temp, linestyle='-.')
        plt.savefig(pic_path)
        plt.close()

    draw_picture_with_line(
        xs, pys, 'p', './ans/reshold-p_correctness',
        p_height_index
    )
    draw_picture_with_line(
        xs, iys, 'i', './ans/reshold-i_correctness',
        i_height_index
    )
    draw_picture_with_line(
        xs, oys, 'o', './ans/reshold-o_correctness',
        o_height_index
    )
    # print(
    #     '%s threshold = %.3f correctness = %.3f' % (
    #         'p', xs[p_height_index], pys[p_height_index]
    #     )
    # )
    # print(
    #     '%s threshold = %.3f correctness = %.3f' % (
    #         'i', xs[i_height_index], iys[i_height_index]
    #     )
    # )
    # print(
    #     '%s threshold = %.3f correctness = %.3f' % (
    #         'o', xs[o_height_index], oys[o_height_index]
    #     )
    # )


if __name__ == '__main__':
    main_4_whole_sentence()
