#@Time  : 2019/3/9 15:30
#@Author: Root
#@File  : NLP.py

import re
import numpy as np
import jieba  #结巴分词
import os
from gensim.models import KeyedVectors  # gensim用来加载预训练word vector
import warnings
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from jieba import posseg
from collections import defaultdict
import codecs
import math
import random
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


class TextHelper(object) :

    positiveTextNumber = 0
    allPosTexts = []
    negativeTextNumber = 0
    allNegTexts = []
    allNumber = 0
    allTexts = []

    def readInEntireFileTexts(self, filePath) :
        """读取一整个文件夹中的所有文本到一个list中"""
        ansList = []
        txtNames = os.listdir(filePath)
        for txtName in txtNames :
            tempStr = filePath + "\\" + txtName
            ansList += self.readInEntireTrainText(tempStr)
        return ansList

    def readInEntireTrainText(self, path) :
        """读取文本"""
        texts = []
        with open(path, "r", errors="ignore") as file :
            text = file.read().strip()
            texts.append(text)
            file.close()
        return texts

    def readInTextLines(self, path, isUtf8=False) :
        """读取文件并把它们分行，加到一个列表中"""
        ansList = []
        if isUtf8 == False :
            with open(path, "r") as file :
                ansList = file.read().splitlines(False)
                file.close()
            tempList = []
            i = 0
        else :
            with open(path, "r", encoding="utf-8") as file :
                ansList = file.read().splitlines(False)
                file.close()
            tempList = []
            i = 0
        while i < len(ansList) :
            if len(ansList[i]) != 0 :
                tempList.append(ansList[i])
            i += 1
        return tempList

    def writeLineToText(self, path, line) :
        """追加一行文字到指定文件"""
        with open(path, "a+", encoding="utf8") as file :
            file.write(line + "\r\n")
            file.close()

    def removePunctuation(self, text) :
        """去除句子中的标点"""
        pStr = r"[^\w\u4e00-\u9fff]+"
        string = re.sub(pStr, "", text)
        pStr = "[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+"
        string = re.sub(pStr, "", string)
        pStr = r"[0-9a-zA-Z]+"
        string = re.sub(pStr, "", string)
        return string.strip()

    def segmentSentence(self, sentence) :
        """分词, 返回一个list"""
        phrasesIterator = jieba.cut(sentence)
        phrasesList = [phrase for phrase in phrasesIterator]
        return phrasesList

    def rmvAndSeg(self, _sentence) :
        """移除标点并且切分成list"""
        sentence = self.removePunctuation(_sentence)
        return self.segmentSentence(sentence)

    def getLength(self, sentence) :
        """
        返回句子被移除标点并且分词后的长度大小
        :param sentence:
        :return:
        """
        tempStr = self.rmvAndSeg(sentence)
        return len(tempStr)



class NLPBiLSTMModel(object) :
    preTrainedChineseVectors = None
    rootPath = ""
    preTrainedPath = ""
    selectedCorpus = ""
    myTrainPath = ""
    trainContentPath = ""
    originalTexts = []  # 一个list, 里面存放所有4000个训练文本中的内容
    positiveTextNumber = 0  # 为1的文本是多少个
    negativeTextNumber = 0  # 为0的文本是多少个
    originalTextPad = []  #每个元素是一个列表，列表中的元素代表分词后的索引
    suitableSize = 0
    suitableTextPad = None
    embeddingMatrix = None
    totalChineseNumber = 0
    embeddingDimension = 300
    verifyVector = None
    trainSetX = None
    testSetX = None
    trainResultY = None
    testResultY = None
    model = None
    callBackFunctions = []
    saveFlag = False
    extraPosCounter = 0
    extraNegCounter = 0

    def __init__(self, textHelper , loadFlag = True,
                 selectedCorpus = r"zhihu", trainContentPath = r"hotel",
                 totalChineseNumber = 50000, testSize = 0.15, epochs = 50) :

        self.rootPath = os.path.dirname(os.path.realpath(__file__))  #获取当前的路径
        self.preTrainedPath = self.rootPath + r"\texts\preTrainText"  #找到预训练的路径
        self.selectedCorpus = selectedCorpus  #选择使用的预料类型
        self.readInPretrainedCorpus()  #读入预训练语料库
        self.trainContentPath = trainContentPath
        self.myTrainPath = self.rootPath + "\\texts\\myTrainText\\" + self.trainContentPath
        self.totalChineseNumber = totalChineseNumber  # 设置总的中文数量


        '''
        if loadFlag == False :
            """自主建立模型"""
            self.readInMytrainTexts(textHelper)  # 读入训练文本
            self.buildOriginalTextPad(textHelper)  # 建立textPad
            self.suitableSize = self.calSuitableSize()  # 确定适合的长度
            self.suitableTextPad = pad_sequences(self.originalTextPad,
                                                 maxlen=self.suitableSize,
                                                 padding="pre",
                                                 truncating="pre")  # 把所有的都trim到一个size
            print("处理我们的训练文本结束".center(25, "*"), "\r\n")
            self.buildEmbeddingMatrix()  # 建立embedding矩阵
            self.suitableTextPad[self.suitableTextPad >= totalChineseNumber] = 0  # 去除五万个词之外的词的索引
            self.verifyVector = np.concatenate((np.ones(self.positiveTextNumber),  # 前面多少个1
                                                np.zeros(self.negativeTextNumber)))  # 后面多少个0
            # 0.9为训练集, 0.1为测试集, 保证shuffle的概率
            # 训练集x -> 训练结果集y; 测试集x -> 测试结果集y;
            self.trainSetX, self.testSetX, self.trainResultY, self.testResultY \
                = train_test_split(self.suitableTextPad, self.verifyVector, test_size=testSize, random_state=12)
            self.model = self.buildModel()
            self.buildCallBackFunctions()
            print("开始训练".center(25, "$"))
            self.model.fit(self.trainSetX,  # 训练集x
                           self.trainResultY,  # 训练结果集y
                           validation_split=0.1,  # 再分出的0.9个样本中在抽0.1来作为  预测样本
                           epochs=epochs,  # 训练将在达到20(epochs) 时停止, 实际因为调用  earlyStopping  会不到20即停止
                           batch_size=128,  # 固定的
                           callbacks=self.callBackFunctions)
            print("结束训练".center(25, "$"))
        else :
        '''
        self.loadNLPModel()

        extraNegPath = self.myTrainPath + "\\extraNegCounter.txt"
        extraPosPath = self.myTrainPath + "\\extraPosCounter.txt"
        with open(extraNegPath, "r", errors="ignore") as file :
            self.extraNegCounter = int(file.readline().strip())
            file.close()
        with open(extraPosPath, "r", errors="ignore") as file :
            self.extraPosCounter = int(file.readline().strip())
            file.close()
        #print("end init...")

    def readInPretrainedCorpus(self) :
        """读取预训练词向量库"""
        print("正在读入预训练词向量库".center(25, "*"))
        contentString = "sgns." + self.selectedCorpus + ".bigram"
        self.preTrainedPath = self.preTrainedPath + "\\" + contentString
        self.preTrainedChineseVectors = KeyedVectors.load_word2vec_format(self.preTrainedPath,
                                                                          binary = False)  #从文件中读入预训练中文词向量
        #self.testReadInPreTrain()
        print("已读入预训练词向量库完成".center(25, "*"))
        print()

    def readInMytrainTexts(self, textHelper) :
        """读取训练文本"""
        #print("正在读取我们的训练文本".center(25, "*"))
        self.positiveTextNumber = textHelper.positiveTextNumber
        self.negativeTextNumber = textHelper.negativeTextNumber
        self.originalTexts = textHelper.allTexts
        #print("读取我们的训练文本完毕".center(25, "*"))

    def transferToPhrasesList(self, textHelper, text) :
        """把一个sentence转成分词索引"""
        phrasesList = textHelper.rmvAndSeg(text)
        # 索引化
        for i, phrase in enumerate(phrasesList):
            try:
                phrasesList[i] = self.preTrainedChineseVectors.vocab[phrase].index
            except KeyError:
                phrasesList[i] = 0
        return phrasesList

    def buildOriginalTextPad(self, textHelper) :
        """把original的所有文本都索引化"""
        #print("处理读进来的训练文本中".center(25, "*"))
        for text in self.originalTexts:
            self.originalTextPad.append(self.transferToPhrasesList(textHelper, text))
        #print("".center(25, "*"))

    def reverseToSentence(self, phrasesList):
        "把phraseList中的索引变回string"
        text = ""
        for i in phrasesList:
            if i != 0:
                text = text + self.preTrainedChineseVectors.index2word[i]
        return text

    def calSuitableSize(self) :
        """计算textPad的大范围长度"""
        sizes = [len(i) for i in self.originalTextPad]
        sizes = np.array(sizes)
        ans = int(np.mean(sizes) + 2 * np.std(sizes))
        #print("可以覆盖的样本数量为%.3f" % (np.sum(sizes < ans) / len(sizes)))
        return ans

    def trimPhrasesList(self, phrasesList) :
        """把phrasesList中内容trim to suitable size, 返回一个pad(1 * suitableSize的矩阵)"""
        return pad_sequences([phrasesList],
                             maxlen = self.suitableSize,
                             padding = "pre",
                             truncating = "pre")

    def buildEmbeddingMatrix(self) :
        #print("开始构建embedding 矩阵".center(25, "*"))
        self.embeddingMatrix = np.zeros((self.totalChineseNumber,
                                         self.embeddingDimension))  # 大小为50000 * 300
        for i in range(self.totalChineseNumber) :
            # mat的 第i行 所有元素赋值为词组库中对应 ： i号索引 -> 中文 -> 词向量
            self.embeddingMatrix[i, :] = self.preTrainedChineseVectors[self.preTrainedChineseVectors.index2word[i]]
        self.embeddingMatrix = self.embeddingMatrix.astype("float32")  # 转换元素的数据类型为float32
        #print("结束构建embedding 矩阵".center(25, "*"))
        #print()

    def buildModel(self) :
        """构建模型"""
        #print("开始进入神经网络模型的构建".center(25, "*"))
        model = Sequential()  # 构建模型
        tempLayer = Embedding(input_dim = self.totalChineseNumber,
                              output_dim= self.embeddingDimension,
                              weights=[self.embeddingMatrix],
                              input_length=self.suitableSize,
                              trainable=False)
        model.add(tempLayer)

        # embedding 之后会得到一个  (文本数量, 规格化长度, 嵌入维度)  的output
        # 在这里就是(4000, suitableSize, 300)

        # units 代表BiLSTM中的神经元个数为32, 并且每一步都会返回一个sequence
        model.add(Bidirectional(LSTM(units = 32, return_sequences = True)))
        # 其中的sequence就是suitableSize 这个中间维度, 而sequence维度就是suitableSize
        # 双向LSTM, 会返回64个

        # 把这些返回的sequence 导入到新的一个LSTM层, 不返回新的sequence, 即取消suitableSize
        model.add(Bidirectional(LSTM(units = 16, return_sequences = True)))

        #####
        model.add(LSTM(units = 16, return_sequences = False))
        model.add(Dropout(0.5))

        # 最后进入一个全连接层, 的到最后的结果
        model.add(Dense(1, activation = "sigmoid"))  # sigmoid 激活输出

        # 以0.001的学习来深度学习, 使用了Adam这个优化器
        optimizer = Adam(lr = 1e-3)

        model.compile(loss = "binary_crossentropy",  #
                      optimizer = optimizer,  # 加载优化器
                      metrics = ["accuracy"])  # 准确度衡量选择accuracy

        model.summary()  # 看一下模型结构
        #print("结束神经网络模型的构建".center(25, "*"))
        #print()
        return model

    def buildCallBackFunctions(self) :
        """构建回调函数列表, 返回一个列表"""
        #print("开始构建回调函数".center(25, "*"))
        tempString = str(self.trainContentPath[0 : ])  #动态的生成kerase文件
        checkpointPath = tempString + r"SentimentCheckpoint.keras"  #一个存储文件的路径
        checkpoint = ModelCheckpoint(filepath = checkpointPath,
                                     monitor = "val_loss",  #当有value-loss有改善的时候进行新的权重储存
                                     verbose = 1,
                                     save_weights_only = True,
                                     save_best_only = True)
        # 尝试加载之前已经训练的模型
        try:
            self.model.load_weights(checkpointPath)
        except Exception as e:
            print(e)

        # 当尝试了3个训练后, val-loss没有改善则停止训练
        earlyStopping = EarlyStopping(monitor = "val_loss", patience = 3, verbose = 1)

        # 自动降低学习速率
        learningRateReduction = ReduceLROnPlateau(monitor = "val_loss",
                                                  factor = 0.1,  #再分出0.1作为验证集
                                                  min_lr = 1e-5,
                                                  patience = 0,
                                                  verbose = 1)
        self.callBackFunctions.append(checkpoint)
        self.callBackFunctions.append(earlyStopping)
        self.callBackFunctions.append(learningRateReduction)
        #print("结束构建回调函数".center(25, "*"))
        return self.callBackFunctions

    def predictSentenceFeeling(self, textHelper, sentence) :
        phrasesList = self.transferToPhrasesList(textHelper, sentence)
        tempPad = self.trimPhrasesList(phrasesList)
        tempPad[tempPad >= self.totalChineseNumber] = 0
        result = self.model.predict(x = tempPad)
        ans = result[0][0]
        if ans >= 0.5:
            print("review :\" %s \" is a good review,  ans = %.2f" % (sentence, ans))
        else:
            print("review :\" %s \" is a evil review,  ans = %.2f" % (sentence, ans))

    def saveNLPModel(self) :
        """
        保存模型以及suitable size
        :return:
        """
        #print("start saving...")
        content = self.trainContentPath + "Model"
        savePath = self.rootPath + "\\saves\\" + content + ".h5"
        self.model.save(savePath)
        textPath = self.rootPath + "\\saves\\" + content + "SuitableSize" + ".txt"
        with open(textPath, "w+") as f:  # 打开test.txt   如果文件不存在，创建该文件。
            f.write(str(self.suitableSize))  # 把suitablesize写入txt
        #print("successful saving...")

    def loadNLPModel(self) :
        """
        读取模型以及suitable size
        :param path:
        :return:
        """
        #print("start loading...")
        content = self.trainContentPath + "Model"
        savePath = self.rootPath + "\\saves\\" + content + ".h5"
        self.model = load_model(savePath)
        textPath = self.rootPath + "\\saves\\" + content + "SuitableSize" + ".txt"
        with open(textPath, "r") as file :
            self.suitableSize = int(file.read().strip())
            file.close()
        #print("suitableSize after loading =", self.suitableSize)
        #print("successful loading...")

    def userUpdateTrainSet(self, text, review) :
        """补充评价"""
        negPath = self.myTrainPath + "\\neg\\" + "user_neg_" + str(self.extraNegCounter) + ".txt"
        posPath = self.myTrainPath + "\\pos\\" + "user_pos_" + str(self.extraPosCounter) + ".txt"
        if review == "neg" :  #是负面评论
            with open(negPath, "w+") as file :
                file.write(text)
                file.close()
            self.extraNegCounter += 1

        if review == "pos" :  #是正面评论
            with open(posPath, "w+") as file :
                file.write(text)
                file.close()
            self.extraPosCounter += 1

    def saveExtraInfo(self) :
        """
        保存模型以及数据文档
        :return:
        """
        extraNegPath = self.myTrainPath + "\\extraNegCounter.txt"
        extraPosPath = self.myTrainPath + "\\extraPosCounter.txt"
        with open(extraNegPath, "w+", errors="ignore") as file :
            file.write(str(self.extraNegCounter))
            file.close()
        with open(extraPosPath, "w+", errors="ignore") as file :
            file.write(str(self.extraPosCounter))
            file.close()

    def getAns(self, textHelper, sentence) :
        """
        返回分数
        :param textHelper:
        :param sentence:
        :return:
        """
        phrasesList = self.transferToPhrasesList(textHelper, sentence)
        tempPad = self.trimPhrasesList(phrasesList)
        tempPad[tempPad >= self.totalChineseNumber] = 0
        result = self.model.predict(x=tempPad)
        ans = result[0][0]
        return ans

    def getFinalAns(self, textHelper, _sentence) :
        """
        获取最后的分数
        :param textHelper:
        :param _sentence:
        :return:
        """
        return self.getAns(textHelper=textHelper, sentence=_sentence)



class NLPWordProcessor(object) :

    stopWordPath = ""
    notWordPath = ""
    degreeWordPath = ""
    emotionalWordPath = ""
    rootPath = ""
    stopWordsList = set()
    notWordsList = set()
    degreeWordsDictionary = defaultdict()  #形式是(key = "极其", value = 2)
    emotionalWordsDictionary = defaultdict()  #形式是(key = "最尼玛", value = -6.70400012637)

    def __init__(self, textHelper) :
        self.rootPath = os.path.dirname(os.path.realpath(__file__))  #获取当前的路径
        self.emotionalWordPath = self.rootPath + "\\words\\" + "BosonNLPLib.txt"
        self.stopWordPath = self.rootPath + "\\words\\" + "stopWords.txt"
        self.degreeWordPath = self.rootPath + "\\words\\" + "degreeWords.txt"
        self.notWordPath = self.rootPath + "\\words\\" + "notWords.txt"

        #入读停顿词
        self.stopWordsList = self.readInWords2Set(self.stopWordPath)
        #读入否定词
        self.notWordsList = self.readInWords2Set(self.notWordPath)
        #测试上面两个读入
        #self.testReadInWrods2Set("stopWords", self.stopWordsList)
        #self.testReadInWrods2Set("notWords", self.notWordsList)
        #读入两个词典
        self.degreeWordsDictionary = self.buildDictionary(textHelper, self.degreeWordPath, isUtf8=False)
        self.emotionalWordsDictionary = self.buildDictionary(textHelper, self.emotionalWordPath, isUtf8=True)

    def readInWords2Set(self, path) :
        """读入一个文件中的词到一个set中，并返回集合"""
        wordsSet = set()
        file = codecs.open(path, "r")
        for word in file :
            if len(word.strip()) != 0 :
                wordsSet.add(word.strip())
        file.close()
        return wordsSet

    def removeStopWords(self, phrasesList) :
        """去除一个分词list中的停用词, 返回去除后的list"""
        return list(filter(lambda phrase : phrase not in self.stopWordsList, phrasesList))

    def buildDictionary(self, textHelper, path, isUtf8 = False) :
        """读入一个文本中的情感分, 前面是词语, 后面是分数"""
        dictionary = defaultdict()
        tempList = textHelper.readInTextLines(path, isUtf8=isUtf8)
        for tempStr in tempList:
            tempStrs = tempStr.split(' ')
            dictionary[tempStrs[0]] = float(tempStrs[1])
        return dictionary

    def isInDictionary(self, word, dictionary) :
        """确定word是不是在字典中"""
        return word in dictionary.keys()

    def list2Dictionary(self, phrasesList) :
        """
        把分完词的列表转换成一个dict,
        "你就不要想起我" -> "你", "就", "不要", "想起", "我"
        变成{<"你", 0>, <"就", 1>, <"不要", 2>, <"想起", 3>, <"我", 4>}
        """
        ansDic = dict()
        for index in range(len(phrasesList)) :
            ansDic[phrasesList[index]] = index
        return ansDic

    def classifyPhrases(self, phrasesList) :
        """对一个分完词并且删除停用此后的phrasesList进行查找"""
        phrasesDictionary = self.list2Dictionary(phrasesList)
        tempEmotionalWordDic = dict()
        tempDegreeWordDic = dict()
        tempNotWordDic = dict()
        for phrase in phrasesDictionary.keys() :
            isEmotioanlWord = self.isInDictionary(phrase, self.emotionalWordsDictionary)
            isDegreeWord = self.isInDictionary(phrase, self.degreeWordsDictionary)
            isNotWord = phrase in self.notWordsList
            if isEmotioanlWord :
                #index作为key, 词语的分值作为value
                #比如phrase = "想起", tempEmotionalDic[3] = -0.283691721554
                tempEmotionalWordDic[phrasesDictionary[phrase]] = self.emotionalWordsDictionary[phrase]
            if isDegreeWord :
                # inde作为key, 词语的分值作为value
                # 比如phrase = "极其", tempEmotionalDic[index] = 2.0
                tempDegreeWordDic[phrasesDictionary[phrase]] = self.degreeWordsDictionary[phrase]
            if isNotWord :
                #这个否定词的赋分为  -1.0
                tempNotWordDic[phrasesDictionary[phrase]] = -1.0
        return tempEmotionalWordDic, tempDegreeWordDic, tempNotWordDic
    #注意这个classify返回的dic顺序

    def initializeWeight(self, tempEmoWordDic, tempDegWordDic, tempNotWordDic) :
        """初始化权重"""
        weight = 1
        emotionalWordIndexes = list(tempEmoWordDic.keys())
        if len(emotionalWordIndexes) == 0 :
            #一个情感词也没有
            return weight
        for index in range(0, emotionalWordIndexes[0]) :
            #找到第一个情感词之前的 __否定词__ 和 __程度词__
            #比如 "不是"  "特别的"  "喜欢"  "这家酒店"
            if index in tempNotWordDic.keys() :
                weight *= -1
            elif index in tempDegWordDic.keys() :
                weight *= float(tempDegWordDic[index])
        return weight

    def scorePhrasesList(self, phrasesList, tempEmoWordDic, tempDegWordDic, tempNotWordDic) :
        """计算切分词之后的list的得分"""
        weight = self.initializeWeight(tempEmoWordDic, tempDegWordDic, tempNotWordDic)
        score = 0

        emotionalWordIndexes = list(tempEmoWordDic.keys())
        nextEmoWord = 0  #这个表示emotionList的下表进行到哪里了

        for phraseIndex in range(0, len(phrasesList)) :

            if phraseIndex in emotionalWordIndexes :
                #如果下标是情感词
                score += weight * float(tempEmoWordDic[phraseIndex])
                nextEmoWord += 1
                #开始计算下一次的权重 ??
                weight = 1
                if nextEmoWord < len(emotionalWordIndexes) :
                    #直到最后倒数第二个情感词
                    for j in  range(phraseIndex + 1, emotionalWordIndexes[nextEmoWord]) :
                        if j in tempNotWordDic.keys() :
                            weight *= -1.625
                        elif j in tempDegWordDic.keys() :
                            weight *= float(tempDegWordDic[j])
            #找到下一个情感词
            if nextEmoWord < len(emotionalWordIndexes) :
                phraseIndex = emotionalWordIndexes[nextEmoWord]
        return score

    def predictSentenceFeeling(self, textHelper, _sentence) :
        """预测句子的分数"""
        #分词
        phrasesList = textHelper.rmvAndSeg(_sentence=_sentence)
        #去停顿词
        phrasesList = self.removeStopWords(phrasesList)
        #print("after remove : ", phrasesList)
        #获得三个dic
        tempEmoWordDic, tempDegWordDic, tempNotWordDic = self.classifyPhrases(phrasesList)
        #print("emodic : ", tempEmoWordDic)
        #print("degdic : ", tempDegWordDic)
        #print("notdic : ", tempNotWordDic)
        ans = self.scorePhrasesList(phrasesList, tempEmoWordDic, tempDegWordDic, tempNotWordDic)
        if ans >= 0:
            print("review :\" %s \" is a good review,  ans = %.2f" % (_sentence, ans))
        else :
            print("review :\" %s \" is a evil review,  ans = %.2f" % (_sentence, ans))
        return ans

    def getWordDetails(self, word) :
        """返回一个的细节"""
        ansList = []
        if word in self.emotionalWordsDictionary.keys() :
            ansList.append(self.emotionalWordsDictionary[word])
        else :
            ansList.append(0.0)
        if word in self.degreeWordsDictionary.keys() :
            ansList.append(self.degreeWordsDictionary[word])
        else :
            ansList.append(0.0)
        if word in self.notWordsList :
            ansList.append(-1)
        else :
            ansList.append(1)
        return ansList

    def calculateFre(self, textHelper, texts):
        """计算一个列表中去除停用词后的分词频率"""
        myDic = {}
        counter = 0
        for text in texts:
            tempList = textHelper.rmvAndSeg(text)
            tempList = self.removeStopWords(tempList)
            for phrase in tempList:

                if phrase in myDic.keys():
                    myDic[phrase] += 1
                else:
                    myDic[phrase] = 1
                    counter += 1

        for key in myDic.keys():
            myDic[key] = float(myDic[key] / counter) * 100.0

        ansList = sorted(myDic.items(), key=lambda e: e[1], reverse=True)
        # 返回一个全是元组的list
        index = 0
        for tempTuple in ansList :
            if tempTuple[1] < 0.05 :
                index = ansList.index(tempTuple)
                break
        pieceList = ansList[0 : index]

        print("total counter = ", counter)
        i = 0
        print("index".rjust(12), "word".rjust(12), "freq".rjust(15), "emotion".rjust(15),
              "degree".rjust(15), "not".rjust(15))
        for item in pieceList :
            word = item[0]
            details = self.getWordDetails(word)
            print(
                str(i).rjust(12),
                word.rjust(12),
                "%.6f".rjust(15) % item[1],
                "%.6f".rjust(15) % details[0],
                "%.2f".rjust(15) % details[1],
                "%d".rjust(15) % details[2]
            )
            i += 1

    def trimLib(self, textHelper, savePath) :
        """剔除情感词库的否定词、程度词"""
        print("size =", len(self.emotionalWordsDictionary))
        counter = 0
        for key, value in self.emotionalWordsDictionary.items() :
            if key not in self.degreeWordsDictionary.keys() and key not in self.notWordsList :
                counter += 1
                line = key + " " + str(value)
                textHelper.writeLineToText(path=savePath, line= line)
        print("size =", counter)

    def getAns(self, textHelper, _sentence) :
        """
        返回分数
        :param textHelper:
        :param _sentence:
        :return:
        """
        phrasesList = textHelper.rmvAndSeg(_sentence=_sentence)
        # 去停顿词
        phrasesList = self.removeStopWords(phrasesList)
        # print("after remove : ", phrasesList)
        # 获得三个dic
        tempEmoWordDic, tempDegWordDic, tempNotWordDic = self.classifyPhrases(phrasesList)
        # print("emodic : ", tempEmoWordDic)
        # print("degdic : ", tempDegWordDic)
        # print("notdic : ", tempNotWordDic)
        ans = self.scorePhrasesList(phrasesList, tempEmoWordDic, tempDegWordDic, tempNotWordDic)
        return ans

    def getFinalAns(self, textHelper, _sentence, threshod = 100) :
        """
        返回最终的句子评价分数
        设定阈值为100
        :param textHelper:
        :param _sentence:
        :return:
        """
        theAns = self.getAns(textHelper=textHelper, _sentence=_sentence)
        if theAns > threshod :
            theAns = threshod
        elif theAns < -threshod :
            theAns = -threshod
        finalAns = theAns + 100.0
        return finalAns / 200.0



class Tester(object) :
    """
    测试者类，用来测试两个类的工作情况
    """

    def testTextsPerformance(self, myModel, myProcessor, state, textHelper) :
        """
        用来测试对某一类文本的两种方法的表现情况
        :param myModel:
        :param myProcessor:
        :param textHelper:
        :return:
        """
        texts = []
        if state == "pos" :
            #测试对负面文本的表现情况
            texts = textHelper.allPosTexts
        elif state == "neg" :
            texts = textHelper.allNegTexts
        modelAnsList = [0, 0]  #答错次数, 答对次数
        processorAnsList = [0, 0]  #答错次数, 答对次数
        totalPrediction = 0
        details = [0, [], 0, [], 0, []]  #model错, processor对; model对, processor错; 同时答错

        if state == "pos" :
            for text in texts :
                modelAns = myModel.getAns(textHelper, text)
                processorAns = myProcessor.getAns(textHelper, text)
                totalPrediction += 1
                print(totalPrediction)
                modelRight = True
                processorRight = True
                if modelAns >= 0.5 :
                    modelAnsList[1] += 1
                    modelRight = True
                else :
                    modelAnsList[0] += 1
                    modelRight = False
                if processorAns >= 0.0 :
                    processorAnsList[1] += 1
                    processorRight = True
                else :
                    processorAnsList[0] += 1
                    processorRight = False
                if not modelRight and processorRight :
                    details[0] += 1
                    details[1].append(text)
                elif modelRight and not processorRight :
                    details[2] += 1
                    details[3].append(text)
                elif not modelRight and not processorRight :
                    details[4] += 1
                    details[5].append(text)
        elif state == "neg" :
            for text in texts :
                modelAns = myModel.getAns(textHelper, text)
                processorAns = myProcessor.getAns(textHelper, text)
                totalPrediction += 1
                print(totalPrediction)
                modelRight = True
                processorRight = True
                if modelAns >= 0.5 :
                    modelAnsList[0] += 1
                    modelRight = False
                else :
                    modelAnsList[1] += 1
                    modelRight = True
                if processorAns >= 0.0 :
                    processorAnsList[0] += 1
                    processorRight = False
                else :
                    processorAnsList[1] += 1
                    processorRight = True
                if not modelRight and processorRight :
                    details[0] += 1
                    details[1].append(text)
                elif modelRight and not processorRight :
                    details[2] += 1
                    details[3].append(text)
                elif not modelRight and not processorRight :
                    details[4] += 1
                    details[5].append(text)

        modelAnsList = [float(number / totalPrediction) * 100.0 for number in modelAnsList]
        processorAnsList = [float(number / totalPrediction) * 100.0 for number in processorAnsList]
        details[0] = float(details[0] / totalPrediction) * 100.0
        details[2] = float(details[2] / totalPrediction) * 100.0
        details[4] = float(details[4] / totalPrediction) * 100.0

        print("".center(25, "*"))
        print("model      错误率: %.2f %%, 正确率: %.2f %%" % (modelAnsList[0], modelAnsList[1]))
        print("processor  错误率: %.2f %%, 正确率: %.2f %%" % (processorAnsList[0], processorAnsList[1]))
        print()

        print("".center(25, "*"))
        print("model 答错, processor 答对的概率为 : %.2f %%" % details[0])
        #for sentence in details[1] :
            #print(textHelper.removePunctuation(sentence))
        print()

        print("".center(25, "*"))
        print("model 答对, processor 答错的概率为 : %.2f %%" % details[2])
        #for sentence in details[3] :
            #print(textHelper.removePunctuation(sentence))
        print()

        print("".center(25, "*"))
        print("model 答错, processor 答错的概率为 : %.2f %%" % details[4])
        for sentence in details[5] :
            print(textHelper.removePunctuation(sentence))
        print()

    def getModel0Processor1Texts(self, myModel, myProcessor, textHelper, flag = "pos") :
        """
        返回一个dic{子路径 : 句子}
        :param myModel:
        :param myProcessor:
        :param texts:
        :param textHelper:
        :return:
        """
        allTexts = []
        filePath = r"C:\Users\57879\Desktop\71118123\竞赛\计算机程序设计竞赛before 4.1\NLPDemo_prnd\texts\myTrainText\hotel"
        filePath += "\\" + flag + "\\"
        txtNames = os.listdir(filePath)
        for txtName in txtNames :
            fileName = filePath + txtName
            with open(fileName, "r", errors="ignore") as inFile :
                string = inFile.read().strip()
                temp = [txtName, string]
                allTexts.append(temp)
                inFile.close()
        ansList = []
        counter = 0
        if flag == "pos" :
            for text in allTexts :
                if myModel.getAns(textHelper, text[1]) < 0.5 and myProcessor.getAns(textHelper, text[1]) > 0.0 :
                    counter += 1
                    print("counter =", counter, text[0])
                    ansList.append(text)
        elif flag == "neg" :
            for text in allTexts :
                if myModel.getAns(textHelper, text[1]) > 0.5 and myProcessor.getAns(textHelper, text[1]) < 0.0 :
                    counter += 1
                    print("counter =", counter, text[0])
                    ansList.append(text)
        for item in ansList :
            textHelper.writeLineToText(
                r"C:\Users\57879\Desktop\71118123\竞赛\计算机程序设计竞赛before 4.1\NLPDemo_prnd\texts\myTrainText\hotel\check.txt",
                item[0]
            )
        print("ending finding...")
        return ansList

    def pickWrongTexts(self, myModel, textHelper, flag = "pos") :
        """
        pick wrong texts
        :param myModel:
        :param flag:
        :return:
        """
        counter = 0
        while True :
            rootPath = r"C:\Users\57879\Desktop\71118123\竞赛\计算机程序设计竞赛before 4.1\NLPDemo_prnd\texts\myTrainText\hotel"
            checkPath = rootPath + "\\check.txt"
            fileNames = textHelper.readInTextLines(checkPath, True)
            currentFileName = fileNames[0]
            fileNames = fileNames[1 : ]
            filePath = rootPath + "\\" + flag + "\\" + currentFileName
            text = ""
            with open(filePath, "r", errors="ignore") as inFile :
                text = inFile.read().strip()
                inFile.close()
            print("the sentence is : ")
            print("\" %s \"" % text)
            choice = input("your choice[-1(负面), 0(unsure), 1(正面)] : ")
            myFlag = 0
            try :
                myFlag = int(choice)
            except Exception :
                print("exception")
                myModel.store()
                with open(checkPath, "w+") as outFile :
                    for fileName in fileNames :
                        outFile.write(fileName + "\r\n")
                    outFile.close()
                break
            temp = ""
            if myFlag == -1 :
                temp = "neg"
            elif myFlag == 1:
                temp = "pos"
            else :
                temp = "unsure"
            if temp == "unsure" :
                print("正在移入unsure文件夹")
                tempPath = r"C:\Users\57879\Desktop\71118123\竞赛\计算机程序设计竞赛before 4.1\NLPDemo_prnd\texts\myTrainText\hotel\unsure\\" + currentFileName
                with open(tempPath, "w+") as outFile :
                    outFile.write(text+"\r\n")
                    outFile.close()
                os.remove(filePath)
                print("移入成功")
            elif temp != flag :
                myModel.userUpdateTrainSet(text, temp)
                os.remove(filePath)
                myModel.store()
            with open(checkPath, "w+") as outFile:
                for fileName in fileNames:
                    outFile.write(fileName + "\r\n")
                outFile.close()
            counter += 1
            print("完成%d" % counter, "rest = %d" % len(fileNames))

    def counter(self, textHelper, myProcessor) :
        """

        :param textHelper:
        :param myProcessor:
        :return:
        """
        posList = []
        tooBig = []
        for text in textHelper.allTexts :
            ans = myProcessor.getAns(textHelper, text)
            if ans < 0.0 and ans >= -100.0:
                posList.append(-ans)
                print(ans)
            elif ans < -100.0 :
                tooBig.append(ans)
                print("****************************", ans)
        posArray = np.array(posList)
        print(np.mean(posArray), "  ", np.sum(posArray) / len(posArray), "     ", tooBig)
        temp = np.mean(posArray) + 2 * np.std(posArray)
        temp = float(temp)
        print(temp)
        print(np.sum(posArray < temp) / len(posArray))

    def getLengthRateDict(self, textHelper, my) :
        """
        返回一个dict 其中放的是长度以及这种算法对其准确率的计算
        :param textHelper:
        :param my: 使用的某种计算方法
        :return:
        """
        ansDict = {}
        counter = 1
        for posText in textHelper.allPosTexts :
            print(counter)
            counter += 1
            finalAns = my.getFinalAns(textHelper, posText)
            length = textHelper.getLength(posText)
            if length in ansDict.keys() :
                tempList = ansDict[length]
                tempList[1] += 1
            else :
                ansDict[length] = [0, 1]
            if finalAns > 0.5 :
                ansDict[length][0] += 1
        for negText in textHelper.allNegTexts :
            print(counter)
            counter += 1
            finalAns = my.getFinalAns(textHelper, negText)
            length = textHelper.getLength(negText)
            if length in ansDict.keys() :
                ansDict[length][1] += 1
            else :
                ansDict[length] = [0, 1]
            if finalAns < 0.5 :
                ansDict[length][0] += 1
        for key in ansDict.keys() :
            if ansDict[key][1] <= 0:
                ansDict[key] = None
            else :
                ansDict[key] = ansDict[key][0] / ansDict[key][1]
        tempDict = {}
        for key, value in ansDict.items() :
            if value != None :
                tempDict[key] = value
        print("ending...")
        return tempDict

    def getAccuracy(self, textHelper, my, a, b):
        """
        返回在a, b之间的正确率
        :param textHelper:
        :param my:
        :param a:
        :param b:
        :return:
        """
        times = 0
        total = 0
        print("working...")
        counter = 0
        for posText in textHelper.allPosTexts :
            print(len(textHelper.allTexts) - counter)
            counter += 1
            finalAns = my.getFinalAns(textHelper, posText)
            if finalAns > a and finalAns < b :
                total += 1
                if finalAns > 0.5:
                    times += 1
        for negText in textHelper.allNegTexts :
            print(len(textHelper.allTexts) - counter)
            counter += 1
            finalAns = my.getFinalAns(textHelper, negText)
            if finalAns > a and finalAns < b :
                total += 1
                if finalAns < 0.5 :
                    times += 1
        print("times = ", times)
        print("rates = %.4f %%" % ((times / total) * 100))
        return ("times = %d" % times) + "\r\n" +  ("rates = %.4f %%" % ((times / total) * 100))



class Solution(object) :
    """
    最终包装好的类
    """
    textHelper = None
    myModel = None
    modelPercentages = [0.977, 0.851, 0.792, 0.711, 0.605, 0.484, 0.622, 0.745, 0.872, 0.975]
    modelGoodIndexes = [0, 1, 2, 8, 9]
    myProcessor = None
    processorPercentages = [0.400, 0.525, 0.727, 0.773, 0.801, 0.640, 0.832, 0.699, 0.510, 0.677]
    processorGoodIndexes = [3, 4, 5, 6]
    unsureIndexes = [7]

    def __init__(self, textTrainPath = "hotel",
        loadFlag = True, selectedCorpus = r"zhihu", trainContentPath = r"hotel",
        totalChineseNumber = 50000, testSize = 0.15, epochs = 50,
    ) :
        self.textHelper = TextHelper()
        self.myModel = NLPBiLSTMModel(
            textHelper=self.textHelper,
            loadFlag=loadFlag, selectedCorpus=selectedCorpus, trainContentPath=trainContentPath,
            totalChineseNumber=totalChineseNumber, testSize=testSize, epochs= epochs
        )
        self.myProcessor = NLPWordProcessor(textHelper=self.textHelper)

    def getModelAns(self, sentence) :

        return self.myModel.getFinalAns(self.textHelper, sentence)

    def getProcessorAns(self, sentence) :
        """
        返回情感词库的最终答案0~1
        :param sentence:
        :return:
        """
        return self.myProcessor.getFinalAns(self.textHelper, _sentence=sentence, threshod=100)

    # 3.23 HQ：将返回值改成成绩加极性str
    def getFinalScoreWithStr(self, sentence, modelGoodRate, processorGoodRate, unsureModelRate, maxLength) :
        """
        返回两个算法的综合打分,
        :return: 返回 score + str : "pos" 或者 "neg" 或者 中性 返回"neu"
        """
        positiveString = "pos"
        negativeString = "neg"
        modelAns = self.getModelAns(sentence)
        processorAns = self.getProcessorAns(sentence)
        #print("modelAns =", modelAns, "processorAns =", processorAns)
        # 当极性判断相同时，返回基于模型所计算出的得分
        if (modelAns - 0.5) * (processorAns - 0.5) > 0.0 :
            if modelAns > 0.5 :
                return modelAns, positiveString
            else :
                return modelAns, negativeString
        modelLeftIndex = int(modelAns * 10)
        processorLeftIndex = int(processorAns * 10)
        isGoodInModel = (modelLeftIndex in self.modelGoodIndexes)
        isGoodInProcessor = (processorLeftIndex in self.processorGoodIndexes)
        temp = 0.0
        if isGoodInModel and not isGoodInProcessor :
            temp = modelGoodRate * modelAns + (1 - modelGoodRate) * processorAns
        if not isGoodInModel and isGoodInProcessor :
            temp = (1 - processorGoodRate) * modelAns + processorGoodRate * processorAns
        if not isGoodInModel and not isGoodInProcessor :
            temp = unsureModelRate * modelAns + (1 - unsureModelRate) * processorAns
        if isGoodInModel and isGoodInProcessor :
            sentenceLength = self.textHelper.getLength(sentence)
            if sentenceLength >= maxLength :
                sentenceLength = maxLength - 1
            lengthRate = sentenceLength / maxLength
            k = math.sqrt(1 - lengthRate * lengthRate)  #会超过一
            #print("k =", k)
            temp = (1 - k) * modelAns + k * processorAns
        if temp > 0.5:
            return temp, "pos"
        elif temp < 0.5:
            return temp, "neg"
        else:
            return temp, "neu"

    def testFinalScoreString(self, size, modelGoodRate, processorGoodRate, unsureModelRate, tempLength) :
        """
        调试各个参数
        :param modelGoodRate:
        :param processorGoodRate:
        :param unsureModelRate:
        :param tempLength:
        :return:
        """
        countRight = 0
        countNeu = 0
        for i in range(size) :
            print(i)
            k = random.randrange(0, 2)
            sentence = ""
            if k == 0 :
                sentence = random.choice(self.textHelper.allNegTexts)
            else :
                sentence = random.choice(self.textHelper.allPosTexts)
            finalScoreStr = self.getFinalScoreWithStr(
                sentence, modelGoodRate, processorGoodRate, unsureModelRate, tempLength
            )[1]
            t = 0
            if finalScoreStr == "pos" :
                t = 1
            elif finalScoreStr == "neg" :
                t = 0
            else :
                print("中性评论%d" % countNeu)
                countNeu += 1
            if t == k :
                countRight += 1
        print(
            "一共进行了随机实验%d组, 正确%d次, 中性%d次, 正确率%.2f %%" % (size, countRight, countNeu, ((countRight / size) * 100))
        )
        return "一共进行了随机实验%d组, 正确%d次, 中性%d次, 正确率%.2f %%" % (size, countRight, countNeu, ((countRight / size) * 100))

    def getScoreStrWithFlag(self, sentence, modelGoodRate, processorGoodRate, unsureModelRate, maxLength) :
        """

        :param sentence:
        :param modelGoodRate:
        :param processorGoodRate:
        :param unsureModelRate:
        :param maxLength:
        :return:
        """

        positiveString = "pos"
        negativeString = "neg"
        modelAns = self.getModelAns(sentence)
        processorAns = self.getProcessorAns(sentence)
        if (modelAns - 0.5) * (processorAns - 0.5) > 0.0:
            if modelAns > 0.5:
                return positiveString, 0
            else:
                return negativeString, 0
        modelLeftIndex = int(modelAns * 10)
        processorLeftIndex = int(processorAns * 10)
        isGoodInModel = (modelLeftIndex in self.modelGoodIndexes)
        isGoodInProcessor = (processorLeftIndex in self.processorGoodIndexes)
        temp = 0.0
        flag = 0
        if isGoodInModel and not isGoodInProcessor:
            temp = modelGoodRate * modelAns + (1 - modelGoodRate) * processorAns
            flag = 1
        if not isGoodInModel and isGoodInProcessor:
            temp = (1 - processorGoodRate) * modelAns + processorGoodRate * processorAns
            flag = 2
        if not isGoodInModel and not isGoodInProcessor:
            temp = unsureModelRate * modelAns + (1 - unsureModelRate) * processorAns
            flag = 3
        if isGoodInModel and isGoodInProcessor:
            flag = 4
            sentenceLength = self.textHelper.getLength(sentence)
            if sentenceLength >= maxLength:
                sentenceLength = maxLength - 1
            lengthRate = sentenceLength / maxLength
            k = math.sqrt(1 - lengthRate * lengthRate)  # 会超过一
            temp = (1 - k) * modelAns + k * processorAns
        if temp > 0.5:
            return "pos", flag
        elif temp < 0.5:
            return "neg", flag
        else:
            return "neu", flag

    def getScoreFloatWithFlag(self, sentence, modelGoodRate, processorGoodRate, unsureModelRate, maxLength) :
        """

        :param sentence:
        :param modelGoodRate:
        :param processorGoodRate:
        :param unsureModelRate:
        :param maxLength:
        :return:
        """
        modelAns = self.getModelAns(sentence)
        processorAns = self.getProcessorAns(sentence)
        if (modelAns - 0.5) * (processorAns - 0.5) > 0.0:
            if modelAns > 0.5:
                return 1, 0
            else:
                return 0, 0
        modelLeftIndex = int(modelAns * 10)
        processorLeftIndex = int(processorAns * 10)
        isGoodInModel = (modelLeftIndex in self.modelGoodIndexes)
        isGoodInProcessor = (processorLeftIndex in self.processorGoodIndexes)
        temp = 0.0
        flag = 0
        if isGoodInModel and not isGoodInProcessor:
            temp = modelGoodRate * modelAns + (1 - modelGoodRate) * processorAns
            flag = 1
        if not isGoodInModel and isGoodInProcessor:
            temp = (1 - processorGoodRate) * modelAns + processorGoodRate * processorAns
            flag = 2
        if not isGoodInModel and not isGoodInProcessor:
            temp = unsureModelRate * modelAns + (1 - unsureModelRate) * processorAns
            flag = 3
        if isGoodInModel and isGoodInProcessor:
            flag = 4
            sentenceLength = self.textHelper.getLength(sentence)
            if sentenceLength >= maxLength:
                sentenceLength = maxLength - 1
            lengthRate = sentenceLength / maxLength
            k = math.sqrt(1 - lengthRate * lengthRate)  # 会超过一
            temp = (1 - k) * modelAns + k * processorAns
        return temp, flag

    def classifiedTests(self, size, modelGoodRate, processorGoodRate, unsureModelRate, maxLength) :
        """

        :param size:
        :param modelGoodRate:
        :param processorGoodRate:
        :param unsureModelRate:
        :param maxLength:
        :return:
        """
        ansList = [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]]
        neuList = []
        countTotalRight = 0
        for i in range(size) :
            print(i)
            k = random.randrange(0, 2)
            sentence = ""
            if k == 0:
                sentence = random.choice(self.textHelper.allNegTexts)
            else:
                sentence = random.choice(self.textHelper.allPosTexts)
            finalScoreStr, flag = self.getScoreStrWithFlag(
                sentence, modelGoodRate, processorGoodRate, unsureModelRate, maxLength
            )
            ansList[flag][1] += 1
            t = 0
            if finalScoreStr == "pos" :
                t = 1
            elif finalScoreStr == "neg" :
                t = 0
            else :
                neuList.append(sentence)
                continue
            if t == k :
                ansList[flag][0] += 1
                countTotalRight += 1
        for i in ansList :
            if i[1] != 0 :
                i[2] = i[0] / i[1]
        print("".center(25, "*"))
        print("总共答对了%d次, 共进行实验%d次, acc = %.2f %%" % (countTotalRight, size, (countTotalRight / size) * 100.0))
        print("两者判定同正同负时 : ")
        print("总共答对了%d次, 共有%d次这种情况, acc = %.2f %%" % (int(ansList[0][0]), int(ansList[0][1]), float(ansList[0][2]) * 100.0))
        print("两者判定model占主导时 : ")
        print("总共答对了%d次, 共有%d次这种情况, acc = %.2f %%" % (int(ansList[1][0]), int(ansList[1][1]), float(ansList[1][2]) * 100.0))
        print("两者processor占主导时 : ")
        print("总共答对了%d次, 共有%d次这种情况, acc = %.2f %%" % (int(ansList[2][0]), int(ansList[2][1]), float(ansList[2][2]) * 100.0))
        print("两者都不占主导时 : ")
        print("总共答对了%d次, 共有%d次这种情况, acc = %.2f %%" % (int(ansList[3][0]), int(ansList[3][1]), float(ansList[3][2]) * 100.0))
        print("两者都占主导时 : ")
        print("总共答对了%d次, 共有%d次这种情况, acc = %.2f %%" % (int(ansList[4][0]), int(ansList[4][1]), float(ansList[4][2]) * 100.0))
        print("".center(25, "*"))
        print("neu list : ")
        for i in neuList :
            print(i)
        print("ending...")

    def testPart(self, size, modelGoodRate, processorGoodRate, unsureModelRate, maxLength, part) :
        """

        :param size:
        :param modelGoodRate:
        :param processorGoodRate:
        :param unsureModelRate:
        :param maxLength:
        :return:
        """
        myScores = []
        theScores = []
        i = 0
        posPointer = 0
        negPointer = 0
        while i < size :
            print(i)
            sentence = ""
            k = -1
            if posPointer < len(self.textHelper.allPosTexts) :
                k = 1
                sentence = self.textHelper.allPosTexts[posPointer]
                posPointer += 1
            elif negPointer < len(self.textHelper.allNegTexts) :
                k = 0
                sentence = self.textHelper.allNegTexts[negPointer]
                negPointer += 1
            else :
                break
            finalScore, f = self.getScoreFloatWithFlag(
                sentence, modelGoodRate, processorGoodRate, unsureModelRate, maxLength
            )
            if f != part :
                continue
            else :
                theScores.append(k)
                myScores.append(finalScore)
                i += 1
        indexes = range(1, size + 1)
        plt.scatter(indexes, theScores, color = "b")
        plt.scatter(indexes, myScores, color = "r")
        plt.show()
        plt.close()  # 关闭matplotlib




































