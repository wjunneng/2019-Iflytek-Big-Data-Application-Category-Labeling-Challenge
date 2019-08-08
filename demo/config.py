# -*- coding: utf-8 -*-
"""
    配置文件
"""
import os


class DefaultConfig(object):
    """
    参数配置
    """
    def __init__(self):
        pass

    # 次数
    k = 5

    # 项目路径
    project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    # 停用词文件路径
    stopwords_path = project_path + '/data/stopwords/stopwords.txt'
    # app_desc.dat 路径
    app_desc_path = project_path + '/data/original/app_desc.dat'
    # apptype_id_name.txt 路径
    apptype_id_name_path = project_path + '/data/original/apptype_id_name.txt'
    # apptype_train.dat 路径
    apptype_train_path = project_path + '/data/original/apptype_train.dat'
    # apptype_train_term_doc.h5文件保存路径
    apptype_train_term_doc_path = project_path + '/data/cache/apptype_train_term_doc.h5'
    # app_desc_term_doc.h5文件路径
    app_desc_term_doc_path = project_path + '/data/cache/app_desc_term_doc.h5'


# if __name__ == '__main__':
#     config = DefaultConfig()
