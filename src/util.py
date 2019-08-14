import os
import pandas as pd
from src.entity import PICOElement

cur_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(cur_dir)
data_dir = os.path.join(project_dir, 'data')
ele_dir = os.path.join(data_dir, 'PICOElement')
pre_model_dir = os.path.join(data_dir, 'preModel')
pre_model_path = os.path.join(pre_model_dir, 'PubMed-shuffle-win-30.bin')


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


if __name__ == '__main__':
    # 检查一下是不是所有句子都是只分到一个类别中去
    check_all_elements(ele_dir)
