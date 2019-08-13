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
    def get_csv_abspaths(cls, dir):
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
        abspaths = cls.get_csv_abspaths(dir)
        ans = []
        for abspath in abspaths:
            t = (abspath, cls.read_one_csv_2_pds(abspath))
            ans.append(t)
        return ans


def get_elements(ele_dir):
    '''收集ele_dir中的所有elements对象，返回一个list'''
    paths_with_pds = __CSVReader.read_all_csvs_2_pds(ele_dir)
    ans = []
    for path_with_pd in paths_with_pds:
        path, pd = path_with_pd
        for row_pds in pd.itertuples():
            ans.append(PICOElement(path, row_pds=row_pds))
    return ans
