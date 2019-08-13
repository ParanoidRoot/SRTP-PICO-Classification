#@Time  : 2019/6/3 22:01
#@Author: ParanoidRoot
#@File  : model.py

from gensim.models.keyedvectors import KeyedVectors
from paranoid_root.model_training.data_utility import PRE_TRAINED_MODEL_PATH
from paranoid_root.model_training.data_utility import pico_elements


def prompt(s : str):
    """
    用于模型构建过程中的提示语句.
    :param s:
    :return:
    """
    print(s.center(25, "*"))
    return None


class Pretrainer(object):
    """
    导入预训练词向量库.
    """
    pre_trained_model = None
    # def __init__(self, ):




if __name__ == "__main__":
    pass
    # prompt("正在读入预训练词向量库")
    # prompt("正在载入预训练词向量语料库")
    # temp = KeyedVectors.load_word2vec_format(
    #     r"C:\Users\57879\Desktop\71118123\My-Projects\srtps\PICO-Text-Classification\data\PubMed-shuffle-win-30.bin",
    #     binary=True
    # )
    # print(temp['medicine'])
    # print(type(temp['medicine']))
    # print(temp['hello'].shape)
    #
    # print(temp.similarity("medicine", "medical"))
