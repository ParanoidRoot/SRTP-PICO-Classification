'''用于打印控制台输出内容'''
import sys
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__=='__main__':
    sys.stdout = Logger("../log/2.txt")
    print("测试控制台的输出")
