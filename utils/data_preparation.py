import json
import os

import jieba
from collections import Counter
from input_file import read
#stopwords = [line.strip() for line in open('../stopwords/stopwords.txt', 'r', encoding='utf-8').readlines()]
def get_token(all_doc_text):
    number = 10
    all_word_list = []
    all_word_num_list = []
    for i, doc in enumerate(all_doc_text):
        doc_list_before = [word for word in jieba.cut(doc)]
        doc_list = []
        doc_keywords_list = []
        doc_keywords_num_list = []
        for word in doc_list_before:
            if (word != '\t') & (word != ' ') & (word != '\n') & ('_' not in word):
                if word not in stopwords:
                    doc_list.append(word)
        '''根据词频抽取关键词'''
        for j, key in enumerate(sorted(Counter(doc_list).items(), key=lambda __key: __key[1], reverse=True)):
            if j >= number:
                break
            doc_keywords_list.append(key[0])
            doc_keywords_num_list.extend([key[0] for i in range(0, key[1])])
        all_word_list.append(doc_keywords_list)
        all_word_num_list.append(doc_keywords_num_list)

def read_name(_path):
    """读取文档路径下的文档名列表
    :param _path: 文档路径
    :return: 文档名列表
    """
    name_list_temp = []
    files = os.listdir(_path)  # 得到文件夹下的所有一级子路径名称
    for file in files:  # 遍历路径筛选文件
        name_list_temp.append(file)
        #if '.' in file:

    return name_list_temp
#TODO:解决txt、doc读取
if __name__=='__main__':
    data_path = "F:\lab\项目实训-多级分类标签\数据\项目实训_下载数据\\"
    sections = read_name(data_path)
    subsections_tmp = []
    print(sections)
    doc_num = 1
    for index, section in enumerate(sections):
        for file_name in read_name(data_path + section):
            print(file_name.split('-'))
            subsections_tmp.append(file_name.split('-')[1])
    subsections = list(set(subsections_tmp))
    print(len(sections), len(subsections))
    subsections.sort(key=subsections_tmp.index)
    for index, section_one in enumerate(sections):
        for file_name in read_name(data_path + section_one):
            try:
                context, image = read(data_path + section_one + '\\', file_name)
            except:
                continue
            abstract = "".join(context)
            if abstract.strip(" ") == "":
                continue
            abstract = [abstract]
            id = doc_num
            title = [file_name.split('-')[2].split('.')[0]]
            doc_num = doc_num + 1
            section = [index]
            subsection = [subsections.index(file_name.split('-')[1])]
            labels = [section[0], subsection[0] + len(sections)]
            data = json.dumps({
                'id': id,
                'title': title,
                'abstract': abstract,
                'section': section,
                'subsection': subsection,
                'labels': labels
            }, ensure_ascii=False)
            print(data)
            with open("../data/our_data.json", 'a') as write_f:
                write_f.write(data)
                write_f.write('\n')

    print(subsections)