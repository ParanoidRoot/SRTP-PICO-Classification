'''
进行文本的预处理，包括：
分词
去掉停用词和还原词形(未实现）
每个句子剪裁或填充
句子词向量化
'''
from sys import path
import nltk

path.append('..')
from src.model import pre_model
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''
将每个句子分割为单词
splited_elements为一个大list，每个元素为一个小list，包含该句子的所有单词
'''

'''剪裁elements'''


def split_elements(presplited_elements: list):
    # 句子分割成单词
    #  去掉停用词
    # 还原词形
    splited_elements = []
    num_elements_len = []
    wnl = WordNetLemmatizer()
    for i in range(len(presplited_elements)):
        # 去掉停用词
        filter_elements = [w for w in presplited_elements[i].split(' ')
                           if w not in stopwords.words('english')]
        # 还原词形
        lem_elements = [wnl.lemmatize(word) for word in filter_elements]
        splited_elements.append(lem_elements)
        num_elements_len.append(len(lem_elements))

    # 获得所有句子的分布情况并选择合适剪裁长度
    num_elements_len = np.array(num_elements_len)
    # 句子长度分布可视化
    plt.hist(num_elements_len, bins=100)
    plt.xlim((0, 100))
    plt.ylabel('num of each length')
    plt.xlabel('length size')
    plt.title('length distribution')
    plt.savefig('../lstm/outputs/elements_length_distribution.png')
    plt.close()





    # 裁剪长度为均值+2倍标准差
    max_length = np.mean(num_elements_len) + 2 * np.std(num_elements_len)
    max_length = int(max_length)
    print("max len = " + str(max_length))
    print("accuracy = " + str(np.sum(num_elements_len < max_length) / (len(num_elements_len))))

    # '''分割elements'''
    # splited_elements = []
    # for i in range(len(presplited_elements)):
    #     # 分割elements
    #     # 去掉停用词
    #     filter_elements = [w for w in presplited_elements[i].split(' ') if w not in \
    #                         stopwords.words('english')]
    #     #词干提取
    #     lem_elements=[SnowballStemmer(filter_elements[k]) for k in filter_elements]
    #     splited_elements.append(lem_elements)
    #     # if (len(presplited_elements[i]) > max_length):
    #     #     splited_elements[i] = splited_elements[i][:max_length]
    #     # else:
    #     #     while (len(splited_elements[i]) < max_length):
    #     #         splited_elements[i].insert(0, 0)
    return splited_elements, max_length


'''句子索引化'''


def tokenize_elements(splited_elements: list):
    tokenized_elements = []
    # if splited_elements==False:
    #     print("The sentences have not been tokenized")
    #     raise KeyError('Not tokenize')
    for i in range(len(splited_elements)):
        temp = []
        for j in range(len(splited_elements[i])):  # 训练集长度应该等于400左右
            word = splited_elements[i][j]
            if (word != 0):
                try:
                    pre_model.vocab[word].index
                except BaseException:
                    # 预处理词向量中没有的单词索引为0
                    print("Exception: '" + word + "' does not exit in model")
                    temp.append(0)
                    pass
                else:
                    temp.append(pre_model.vocab[word].index)
            else:
                temp.append(0)
        tokenized_elements.append(temp)
    tokenized_elements = np.array(tokenized_elements)
    return tokenized_elements


'''词向量化'''


def embedding_elements(tokenized_elements: list, max_length):
    num_words = 50000  # 只使用50000个常用词
    embedding_dim = 200  # 维度200
    embedding_matrix = np.zeros((num_words, embedding_dim))  # 初始化词向量化矩阵，维度50000*300
    for i in range(num_words):
        embedding_matrix[i, :] = pre_model[pre_model.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    tokenized_elements = pad_sequences(tokenized_elements, maxlen=max_length)
    tokenized_elements[tokenized_elements >= num_words] = 0
    return embedding_matrix, tokenized_elements, num_words, embedding_dim


def preporcess_elements(_presplited_elements):
    _splited_elements, _max_length = split_elements(_presplited_elements)
    _tokenized_elements = tokenize_elements(_splited_elements)
    _embedding_matrix, _tokenized_elements, _num_words, _embedding_dim = embedding_elements(_tokenized_elements,
                                                                                            _max_length)
    return _tokenized_elements, _embedding_matrix, _num_words, _embedding_dim, _max_length


def preporcess_predict_elements(_presplited_elements, defined_max_len):
    _splited_elements, _max_length = split_elements(_presplited_elements)
    _max_length = defined_max_len  # 421，根据训练模型决定
    _tokenized_elements = tokenize_elements(_splited_elements)
    _embedding_matrix, _tokenized_elements, _num_words, _embedding_dim = embedding_elements(_tokenized_elements,
                                                                                            _max_length)
    return _tokenized_elements, _embedding_matrix, _num_words, _embedding_dim, _max_length


if __name__ == '__main__':
    # 测试预处理效果
    test_str1 = "this is a big apple and i eat it"
    test_str2 = " This study shows that vaginal route of administration of " \
                "misoprostol is preferable to oral route for induction " \
                "of labour when used in equivalent dosage of 50 mcg 6 hourly"
    test_elements = []
    test_elements.append(test_str1)
    test_elements.append(test_str2)
    print(split_elements(test_elements))
