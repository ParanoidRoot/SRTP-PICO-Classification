'''
加载已经训练好的模型用来预测句子
每个句子用三个标签模型分别进行预测
'''
from lstm.bi_lstm_model import model_O, model_I, model_P, \
    checkpoint_path_O, checkpoint_path_I, checkpoint_path_P
from sys import path
import sys
import time

path.append('..')
from preProc.preProcess import preporcess_predict_elements
from src.logger import Logger

# 加载训练模型
try:
    model_P.load_weights(checkpoint_path_P)
    model_I.load_weights(checkpoint_path_I)
    model_O.load_weights(checkpoint_path_O)
except Exception as e:
    print('No existed trained models,please train models before use')
'''预测句子标签'''


# 假设句子已经去除标点
def predict_element_label(sentence: str):
    sentence_to_list = [sentence]
    tokenized_elements, \
    embedding_matrix, \
    num_words, \
    embedding_dim, \
    max_length = preporcess_predict_elements(sentence_to_list, 48)  # 未经过去除停用词未421，去除后未48
    # 进行预测
    result_P = model_P.predict(x=tokenized_elements)
    result_I = model_I.predict(x=tokenized_elements)
    result_O = model_I.predict(x=tokenized_elements)
    # 存储结果
    coef = [result_P[0][0], result_I[0][0], result_O[0][0]]
    print("sentence = " + sentence)
    print("Label_P = " + str(coef[0]))
    print("Label_I = " + str(coef[1]))
    print("Label_O = " + str(coef[2]))

if __name__ == '__main__':
    logger_path = "../log/logger_predict-{}.txt".format(time.strftime('%m%d-%H%M', time.localtime(time.time())))
    sys.stdout = Logger(logger_path)  # 打印控制台内容作为日志
    # O*
    test_sentence1 = "This study shows that vaginal route of administration of " \
                     "misoprostol is preferable to oral route for induction " \
                     "of labour when used in equivalent dosage of 50 mcg 6 hourly"
    # I*
    test_sentence2 = "In the misoprostol group " \
                     "a tablet of 200 µg was dissolved in 200 cc of water" \
                     " and 25 cc was administered every " \
                     "2 hours until adequate uterine contractions were achieved"
    # P
    test_sentence3 = "Between 2008 and 2010  " \
                     "a total of 285 term pregnant women " \
                     "whom were candidate for vaginal delivery " \
                     "were assessed for eligibility to enter the study"
    # P I
    test_sentence4 = "A total of 160 women were enrolled between " \
                     "January 2011 and July 2012 of whom 80 " \
                     "were randomized to the misoprostol and 80 to " \
                     "the dinoprostone vaginal insert groups Figure"
    # P I
    test_sentence5 = "Seventy term pregnant women 37 42 weeks gestation" \
                     " were randomized to Group A or B " \
                     "after informed written consent and excluding the following " \
                     "cervix favorable for amniotomy Bishop score 6" \
                     " non vertex presentation " \
                     "intrauterine demise previous uterine scar" \
                     " oligohydramnios intrauterine growth retardation" \
                     " multifetal pregnancy" \
                     " clinical evidence of cardiopulmonaryyhepaticyrenal disease " \
                     "electrolyte abnormalities pre eclampsiayeclampsia"
    # P*
    test_sentence6 = "this prospective double blind study was undertaken to compare " \
                     "the safety and efﬁcacy of oral vs vaginal misoprostol in equivalent doses" \
                     "50 mg for induction of labour"
    # P I*
    test_sentence7 = "A total of 128 term pregnancies with indication for induction of " \
                     "labour were allocated to two groups to receive 50 mg misoprostol " \
                     "orally or vaginally, every 4 h until adequate contractions were achieved " \
                     "or a maximum of 200 mg dose"
    # I/O*
    test_sentence8 = "Induction to delivery interval was signiﬁcantly shorter in the " \
                     "vaginal group compared with the oral group 14.6 h vs 22.5 h p50.001"
    # O*
    test_sentence9 = "There was no signiﬁcant difference between the groups with respect " \
                     "to mode of delivery neonatal outcome and maternal side effects"
    # 1，2，6,7,8,9句子不在训练集中出现
    # #测试训练结果
    predict_element_label(test_sentence1.lower())  # O
    predict_element_label(test_sentence2.lower())  # I
    predict_element_label(test_sentence3.lower())  # P
    predict_element_label(test_sentence4.lower())  # P I
    predict_element_label(test_sentence5.lower())  # P I
    predict_element_label(test_sentence6.lower())  # P
    predict_element_label(test_sentence7.lower())  # P I
    predict_element_label(test_sentence8.lower())  # I/O
    predict_element_label(test_sentence9.lower())  # O

    pass
