class PICOElement(object):
    '''在这个类中完成对PICOElement的构建.'''

    def __init__(
        self,
        file_path,
        sentence="",
        pubmed_id=0,
        content="",
        label="",
        labeler="",
        row_pds=None
    ):
        '''对PICOElement实现构建.'''
        if row_pds is not None:
            sentence = row_pds.sentence
            pubmed_id = row_pds.pubmed_id
            content = row_pds.content
            label = row_pds.label
            labeler = row_pds.labeler
        self.__file_path = file_path
        self.__sentence = self.__parse_sentence(sentence)
        self.__pubmed_id = pubmed_id
        self.__content = self.__parse_sentence(content)
        self.__label = label
        self.__labeler = labeler

    @property
    def file_path(self):
        return self.__file_path

    @property
    def sentence(self):
        return self.__sentence

    @property
    def pubmed_id(self):
        return self.__pubmed_id

    @property
    def content(self):
        return self.__content

    @property
    def label(self):
        return self.__label

    @property
    def labeler(self):
        return self.__labeler

    def __parse_sentence(self, sentence):
        '''将原文的句子进行 re 分析'''
        import re
        ans = sentence.split(" ||| ")[-1].strip()
        ans = re.sub('[^A-Za-z 0-9]', ' ', ans).lower()
        ans = " ".join([part for part in ans.split() if part])
        return ans

    def __str__(self):
        ans = ''
        ans = (
            'file_path = %s\r\n' % self.file_path +
            'sentence = %s\r\n' % self.sentence +
            'pubmed_id = %d\r\n' % self.pubmed_id +
            'content = %s\r\n' % self.content +
            'label = %s\r\n' % self.label +
            'labeler = %s\r\n' % self.labeler
        )
        return ans

    @property
    def fine_grained_label(self):
        ans = (
            self.label.lower().replace(' ', '').
            replace('diagnositc', 'diagnostic').replace('.', '')
        )
        return ans
