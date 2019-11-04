from src.util import get_all_elements, ele_dir, pre_model_path
from gensim.models.keyedvectors import KeyedVectors
import random

print('加载预训练模型...稍等一会儿...')
# 加载预训练模型
pre_model = KeyedVectors.load_word2vec_format(
    pre_model_path, binary=True
)
if __name__ == '__main__':
    # 把训练集中的所有句子提取出来
    pico_elements = get_all_elements(ele_dir)
    element_number = len(pico_elements)
    # 随机找一个句子输出测试
    index = random.randint(1, element_number)
    element = pico_elements[index]
    print(index, '\r\n', element)
    print(pre_model['medicine'])  # 输入一个词, 输出一个矩阵

    print("test illegal word 'millar'")
    try:
        print(pre_model['millar'])  # 输入一个不存在的词, 测试结果
    except BaseException:
        print("Base millar' does not in vovabulary")
        pass
    else:
        print("'millar' does not in vovabulary")

    print(type(pre_model['medicine']))  # 输出矩阵为200维的array
    print(pre_model['hello'].shape)
    print(pre_model.similarity("medicine", "medical"))  # 比较两个词的相似度