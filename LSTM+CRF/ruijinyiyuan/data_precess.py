import pandas as pd
import numpy as np
import os
import codecs
# ['Drug', 'Duration', 'Test_Value', 'Symptom', 'Reason', 'Test', 'Anatomy', 'Amount', 'Disease', 'Treatment', 'Frequency', 'Method', 'SideEff', 'Level', 'Operation']

#自己更改为我们需要的语料
entities_dict={
    'Level':'LEV',
    'Test_Value':'TSV',
    'Test':'TES',
    'Anatomy':'ANT',
    'Amount':'AMO',
    'Disease':'DIS',
    'Drug':'DRU',
    'Treatment':'TRE',
    'Reason':'REA',
    'Method':'MET',
    'Duration':'DUR',
    'Operation':'OPE',
    'Frequency':'FRE',
    'Symptom':'SYM',
    'SideEff':'SID'
}
train_dir='datas/ruijin_round1_train2_20181022'
train_file='datas/example1.train'
dev_file='datas/example1.dev'
test_file='datas/example1.test'
stopwords=[' ','\n',' ']

def readfile(filepath):
    """
    该函数负责读取文章并生成由文章中每个字符构成的列表
    :param filepath: txt文件路径
    :return: 返回传入文本文件中文章每个字构成的列表
    """
    sentence=[]
    for line in codecs.open(filepath, 'r', 'utf8'):
        for i in range(len(line)):
            sentence.append(line[i])
    return sentence
    # with open(filepath,'rb') as f:
    #         for line in f.readlines():



def get_entities(dir):
    entities = []
    files = os.listdir(dir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(files)):
        if files[i].split(".")[1] == 'ann':
            path = os.path.join(dir, files[i])
            for line in codecs.open(path, 'r', 'utf8'):
                parts = line.split('\t')
                entities.append(parts[1].split(' ')[0])
    return list(set(entities))

def getNotations(filepath):
    """

    :param filepath: ann标注文件路径
    :return: 返回标注文件中每个标注信息的三元组【实体类别，起始位置，终止位置】，返回值为这些三元组构成的列表
    """
    pairs=[]
    for line in codecs.open(filepath, 'r', 'utf8'):
        parts=line.split('\t')[1].split(' ')
        # print(filepath,parts)
        pair=[]
        pair.append(parts[0])
        pair.append(int(parts[1]))
        pair.append(int(parts[-1]))
        pairs.append(pair)
    return pairs


def getPairs(senc_file, label_file):
    """
    该函数接受一对文件 txt和ann，并生成单个字标注（BIOES）好的文件
    :param senc_file:对应文本文件路径
    :param label_file: 对应ann标注文件路径
    :return:
    """
    sentence = readfile(senc_file)
    sentence = [[w, 'O'] for w in sentence]
    labels = getNotations(label_file)

    def replace_l(index, char):  # 替换字符串第一个字符
        s = list(sentence[index][1])
        s[0] = char
        sentence[index][1] = "".join(s)

    def check_single(idnex, lab):  # 检查单个标注时所在索引位置是否已经标注并做处理
        if idnex < 2 or idnex > len(sentence) - 2:
            return
        if (sentence[idnex][1] != 'O'):
            sentence[idnex][1] = 'S-' + lab
            if sentence[idnex - 1][1] != 'O' and sentence[idnex - 2][1] != 'O':
                replace_l(idnex - 1, 'E')
            elif sentence[idnex - 1][1] != 'O':
                replace_l(idnex - 1, 'S')

            if sentence[idnex + 1][1] != 'O' and sentence[idnex + 2][1] != 'O':
                replace_l(idnex + 1, 'B')
            elif sentence[idnex + 1][1] != 'O':
                replace_l(idnex + 1, 'S')
        else:
            sentence[idnex][1] = 'S-' + lab

    def check(start, end, lab):  # 检查多个字标注时开始位置和结束位置是否已经标注并处理
        if sentence[start][1] != 'O' and start > 2:
            if sentence[start - 1][1] != 'O' and sentence[start - 2][1] != 'O':
                replace_l(start - 1, 'E')
            elif sentence[start - 1][1] != 'O':
                replace_l(start - 1, 'S')
        if sentence[end][1] != 'O' and end < len(sentence) - 2:
            if sentence[end + 1][1] != 'O' and sentence[end + 2][1] != 'O':
                replace_l(end + 1, 'B')
            elif sentence[end + 1][1] != 'O':
                replace_l(end + 1, 'S')

    for label in labels:
        lab = entities_dict[label[0]]
        start = label[1]
        end = label[2]
        if end - start == 1:
            check_single(start, lab)
            continue
        check(start, end - 1, lab)
        sentence[start][1] = 'B-' + lab
        sentence[end - 1][1] = 'E-' + lab
        for i in range(start + 1, end - 1):
            sentence[i][1] = 'I-' + lab
    sentence = [w for w in sentence if  w[0] not in stopwords]
    return sentence



def writefile(filename,list,sep=' '):
    """
    该函数负责将处理好的标注数据文件保存
    :param filename: 保存的文件名
    :param list: 要保存的列表
    :param sep: 字符和标注之间的分隔符
    :return:
    """
    with open(filename,'w',encoding='utf-8') as f:
        for item in list:
            if item=='\n':
                f.write( '\n')
            else:
                f.write(sep.join(item) + '\n')





def get_files(dir):
    """
    获取所有文件名
    :param dir: 目录
    :return: 目录下所有去重文件名的列表
    """
    file_list=[]
    for roots, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(file.split('.')[0])
    return list(set(file_list))

def due_files(dir,files,filename,sep=' '):
    """
    批量处理文件
    :param dir: 目录
    :param filename: 处理后保存的文件名
    :param sep: 分隔符
    :return:
    """

    sentences=[]
    for name in files:
        t_file=os.path.join(dir,name+'.txt')
        a_file=os.path.join(dir,name+'.ann')
        sentence=getPairs(t_file,a_file)
        for word in sentence:
            sentences.append(word)
            if word[0]=='。' and word[1]=='O':
                sentences.append('\n')

    writefile(filename,sentences)

def split_data(dir,train_name,dev_name,test_name,split_ratio=[0.8,0.1],sep=' '):
    file_list=get_files(dir)
    total=len(file_list)
    p1=int(total*split_ratio[0])
    p2=int(total*(split_ratio[0]+split_ratio[1]))
    due_files(dir,file_list[:p1],train_name)
    due_files(dir, file_list[p1:p2], dev_name)
    due_files(dir, file_list[p2:], test_name)
if __name__ == '__main__':
    split_data(train_dir,train_name=train_file,dev_name=dev_file,test_name=test_file,)
    # print(get_entities(train_dir))
