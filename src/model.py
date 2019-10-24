from util import get_all_elements, ele_dir, pre_model_path
from gensim.models.keyedvectors import KeyedVectors
import random




if __name__ == '__main__':
    pico_elements = get_all_elements(ele_dir)
    element_number = len(pico_elements)
    index = random.randint(1, element_number)
    element = pico_elements[index]
    print(index, '\r\n', element)
    print('这里有点慢...稍等一会儿...')
    pre_model = KeyedVectors.load_word2vec_format(
        pre_model_path, binary=True
    )
    print(pre_model['medicine'])  # 输入一个词, 输出一个矩阵
    print(type(pre_model['medicine']))  # 输出矩阵为200为的array
    print(pre_model['hello'].shape)
    print(pre_model.similarity("medicine", "medical"))  # 比较两个词的相似度
