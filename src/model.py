from src.util import get_all_elements, ele_dir, pre_model_path
from gensim.models.keyedvectors import KeyedVectors
import random


if __name__ == '__main__':
    pico_elements = get_all_elements(ele_dir)
    element_number = len(pico_elements)
    # pre_model = KeyedVectors.load_word2vec_format(
    #     pre_model_path, binary=True
    # )
    index = random.randint(1, element_number)
    element = pico_elements[index]
    print(index, '\r\n', element)
