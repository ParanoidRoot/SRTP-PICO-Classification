import os
import pandas as pd
from entity import PICOElement
import numpy as np
import matplotlib.pyplot as plt


class __CSVReader(object):
    '''在这个类中完成使用 pd 读入data的功能.'''
    @classmethod
    def read_one_csv_2_pds(cls, csv_path):
        '''读入一个csv返回生成一个pandas的data frame'''
        return pd.read_csv(csv_path)

    @classmethod
    def get_csv_paths(cls, dir):
        '''获取一个文件夹下的所有csv名.'''
        csv_basenames = [
            basename for basename in os.listdir(dir) if '.csv' in basename
        ]
        csv_abspaths = [
            os.path.join(dir, basename) for basename in csv_basenames
        ]
        return csv_abspaths

    @classmethod
    def read_all_csvs_2_pds(cls, dir):
        '''返回一个目录下的所有csv文件变成一个(abspath, data_frame)的tuple.
        返回一个list.
        '''
        abspaths = cls.get_csv_paths(dir)
        ans = []
        for abspath in abspaths:
            t = (abspath, cls.read_one_csv_2_pds(abspath))
            ans.append(t)
        return ans


def get_elements_by_one_csv_path(csv_path: str):
    '''获取一个csv文件中的所有elements.'''
    path, pd = csv_path, __CSVReader.read_one_csv_2_pds(csv_path)
    ans = []
    for row_pds in pd.itertuples():
        ans.append(PICOElement(path, row_pds=row_pds))
    return ans


def get_all_elements(ele_dir):
    '''收集ele_dir中的所有elements对象，返回一个list'''
    paths = __CSVReader.get_csv_paths(ele_dir)
    ans = []
    for path in paths:
        ans.extend(get_elements_by_one_csv_path(path))
    return ans


def check_one_csv_elements(csv_path):
    '''检查一个csv文件中的元素是否无重复的句子被分到多个标签.'''
    one_csv_elements = get_elements_by_one_csv_path(csv_path)
    ans_dict = dict()
    is_ok = True
    for element in one_csv_elements:
        if element.sentence not in ans_dict.keys():
            ans_dict[element.sentence] = 1
        else:
            ans_dict[element.sentence] += 1
            is_ok = False
    return one_csv_elements[0].pubmed_id, is_ok, '' if is_ok else (
        one_csv_elements[0].pubmed_id, is_ok, one_csv_elements[0].file_path
    )


def check_all_elements(ele_dir):
    '''检查所有的elements是否都仅被分到一个标签中.'''
    paths = __CSVReader.get_csv_paths(ele_dir)
    total = 0
    for path in paths:
        pubmed_id, is_ok, path = check_one_csv_elements(path)
        if not is_ok:
            total += 1
            print(total, path)


class FastBertCSVWriter(object):
    '''生成指定样式的训练集, 以及测试集.'''

    @classmethod
    def get_all_sentence_2_labels(cls, pathes):
        '''将数据元素目录下的所有元素, 转成一个dict.'''
        ans_dict = dict()
        ans_dict['text'] = []
        ans_dict['file_path'] = []
        ans_dict['p'] = []
        ans_dict['i'] = []
        ans_dict['o'] = []
        for path in pathes:
            one_csv_elements = get_elements_by_one_csv_path(path)
            sentence_2_position = dict()
            for element in one_csv_elements:
                sentence = element.sentence
                label = element.label.lower()[0]
                if sentence not in sentence_2_position.keys():
                    sentence_2_position[sentence] = len(ans_dict['text'])
                    ans_dict['text'].append(sentence)
                    ans_dict['file_path'].append(element.file_path)
                    ans_dict['p'].append(0)
                    ans_dict['i'].append(0)
                    ans_dict['o'].append(0)
                if label == 'p':
                    ans_dict['p'][sentence_2_position[sentence]] = 1
                elif label == 'i':
                    ans_dict['i'][sentence_2_position[sentence]] = 1
                elif label == 'o':
                    ans_dict['o'][sentence_2_position[sentence]] = 1
                else:
                    raise KeyError
            sentence_2_position.clear()
            one_csv_elements.clear()
        return ans_dict


    @classmethod
    def count_word_num_of_a_string(cls, input_str: str):
        '''计算一个字符串中的单词个数.'''
        return input_str.count(' ') + 1


    @classmethod
    def count_seq_len(cls, pathes):
        sentence_2_labels = cls.get_all_sentence_2_labels(pathes)
        texts = sentence_2_labels['text']
        seq_len_counter = []
        for text in texts:
            seq_len_counter.append(cls.count_word_num_of_a_string(text))
        # from matplotlib import pyplot as plt
        plt.xlabel('sentence length')
        plt.ylabel('sentence length rates')
        plt.title('pre-processed sentence length distribution')
        plt.hist(seq_len_counter, 50, normed=1, facecolor='blue', alpha=0.5)
        plt.show()
        # seq_len = 60
        
        def count_where_smaller(seq, thred):
            ans = 0
            for what in seq:
                if what < thred:
                    ans += 1
            return ans / len(seq)
        # print(count_where_smaller(seq_len_counter, seq_len))


    @classmethod
    def write_to_csvs(cls, csv_pathes, train_set_rate, tag):
        '''
        将原始数据按比例分成数据集与测试集,
        按所要求的格式写入.
        '''
        import time
        import datetime
        sentence_2_labels = cls.get_all_sentence_2_labels(csv_pathes)
        total_rows = len(sentence_2_labels['text'])
        train_set_number = int(total_rows * train_set_rate)
        original_dataframe = pd.DataFrame(sentence_2_labels, columns=list(sentence_2_labels.keys()))
        train_set_dataframe = original_dataframe.sample(n=train_set_number, random_state=None)
        train_set_dataframe.to_csv('train_set_%s.csv' % tag)
        test_set_dataframe = original_dataframe.sample(
            n=total_rows - train_set_number, random_state=None
        )
        test_set_dataframe.to_csv('val_set_%s.csv' % tag)


def fine_grained_data_preparation():
    '''在这个方法中完成对细粒度的分类的数据准备工作.'''
    paths = __CSVReader.get_csv_paths(
        r'F:\PythonProjects\SRTP-PICO-Classification\data\PICOElement'
    )
    all_elements = get_all_elements(
        r'F:\PythonProjects\SRTP-PICO-Classification\data\PICOElement'
    )
    labels_dict = dict()
    for element in all_elements:
        if element.fine_grained_label not in labels_dict.keys():
            labels_dict[element.fine_grained_label] = []
    labels_dict['content'] = []
    labels_dict['file_path'] = []
    for path in paths:
        elements = get_elements_by_one_csv_path(path)
        current_csv_contents = dict()
        for element in elements:
            current_content = element.content
            current_label = element.fine_grained_label
            current_file_path = element.file_path
            if current_content not in current_csv_contents.keys():
                current_csv_contents[current_content] = (
                    len(labels_dict['content'])
                )
                for key in labels_dict.keys():
                    if key == 'content':
                        labels_dict[key].append(current_content)
                    elif key == current_label:
                        labels_dict[key].append(1)
                    elif key == 'file_path':
                        labels_dict[key].append(current_file_path)
                    else:
                        labels_dict[key].append(0)
            else:
                # 当前的短语已经出现过了
                index = current_csv_contents[current_content]
                labels_dict[current_label][index] = 1


def method3():
    # 构建出每一个句子可能的分类的属性
    all_elements = get_all_elements(
        r'F:\PythonProjects\SRTP-PICO-Classification\data\PICOElement'
    )
    labels_dict = dict()
    for element in all_elements:
        if element.fine_grained_label not in labels_dict.keys():
            labels_dict[element.fine_grained_label] = []
    # 添加两列
    labels_dict['text'] = []
    labels_dict['file_path'] = []
    # 获取所有的路径
    paths = __CSVReader.get_csv_paths(
        r'F:\PythonProjects\SRTP-PICO-Classification\data\PICOElement'
    )
    # 遍历每一个文件
    for path in paths:
        elements = get_elements_by_one_csv_path(path)
        current_csv_texts = dict()
        for element in elements:
            current_text = element.sentence
            current_label = element.fine_grained_label
            current_file_path = element.file_path
            # current_masked_sentence = element.get_sentence_without_content()
            # word_num = FastBertCSVWriter.count_word_num_of_a_string(
            #     current_masked_sentence
            # )
            if current_text not in current_csv_texts.keys():
                # 首先先加入到字典中去.
                current_csv_texts[current_text] = (
                    len(labels_dict['text'])
                )
                # 添加一条正确的句子的标签记录
                for key in labels_dict.keys():
                    if key == 'text':
                        labels_dict[key].append(current_text)
                    elif key == current_label:
                        labels_dict[key].append(1)
                    elif key == 'file_path':
                        labels_dict[key].append(current_file_path)
                    else:
                        labels_dict[key].append(0)
                # 添加一条没有关键子句的标签记录
                # if (
                #     current_masked_sentence is not None and
                #     len(current_masked_sentence) > 5 and
                #     word_num > 1
                # ):
                #     # 添加到字典中去, 防止重复子句
                #     current_csv_texts[current_masked_sentence] = (
                #         len(labels_dict['text'])
                #     )
                #     for key in labels_dict.keys():
                #         if key == 'text':
                #             labels_dict[key].append(current_masked_sentence)
                #         elif key == 'file_path':
                #             labels_dict[key].append(current_file_path)
                #         else:
                #             # 其余所有标签都为 0
                #             labels_dict[key].append(0)
            else:
                # 当前的短语已经出现过了
                index = current_csv_texts[current_text]
                labels_dict[current_label][index] = 1
                # if (
                #     current_masked_sentence is not None and
                #     len(current_masked_sentence) > 4 and
                #     word_num > 1 and
                #     current_masked_sentence not in labels_dict.keys()
                # ):
                #     # 将去掉短句的句子试图再加入到字典中去
                #     current_csv_texts[current_masked_sentence] = (
                #         len(labels_dict['text'])
                #     )
                #     for key in labels_dict.keys():
                #         if key == 'text':
                #             labels_dict[key].append(current_masked_sentence)
                #         elif key == 'file_path':
                #             labels_dict[key].append(current_file_path)
                #         else:
                #             # 其余所有标签都为 0
                #             labels_dict[key].append(0)
        ans_pd = pd.DataFrame(labels_dict)
        ans_pd.to_csv('./ans/combined/result.csv')


def parse_fine_grained_data():
    '''分析细粒度的分类.'''
    original_data = pd.read_csv('./data/final/train.csv')
    sample = original_data.sample(frac=0.2, random_state=None)
    test = original_data.sample(frac=1.0, random_state=None)
    sample.to_csv('./data/final/val.csv')
    test.to_csv('./data/final/test.csv')
    # lengths = []
    # for text in original_data['text']:
    #     lengths.append(FastBertCSVWriter.count_word_num_of_a_string(text))
    # lengths = np.array(lengths)

    # def get_overlap_rate(_lengths, target_len):
    #     temp = _lengths.copy()
    #     temp[_lengths < target_len] = 1
    #     temp[_lengths >= target_len] = 0
    #     return np.sum(temp) / temp.shape[0]

    # xs = np.array(list(range(20, 70)))
    # ys = []
    # for target_len in xs:
    #     overlap_rate = get_overlap_rate(lengths, target_len)
    #     ys.append(overlap_rate)
    # ys = np.array(ys)
    # plt.figure()
    # plt.xlabel('length')
    # plt.ylabel('overlap rate')
    # plt.title('fine grained labels')
    # plt.bar(xs, ys, color='green', width=0.5)
    # plt.savefig('./ans/combined/overlap_rates')
    # plt.close()


if __name__ == '__main__':
    # FastBertCSVWriter.write_to_csvs(
    #     __CSVReader.get_csv_paths(r'F:\PythonProjects\TestFastBert\PICOElement'),
    #     1.0,
    #     tag='2'
    # )
    FastBertCSVWriter.count_seq_len(
        __CSVReader.get_csv_paths(
            r'F:\PythonProjects\SRTP-PICO-Classification\data\PICOElement'
        )
    )
    # print('hello world')
    # fine_grained_data_preparation()
    # method3()
    # parse_fine_grained_data()
    # get_all_elements('')
