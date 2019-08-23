# -*- coding: utf-8 -*-
"""
    配置文件
"""
import os
# 集成方法分类器
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# 高斯过程分类器
from sklearn.gaussian_process import GaussianProcessClassifier

# 广义线性分类器
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifierCV

# K近邻分类器
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.neighbors import KNeighborsClassifier

# 朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB

# 神经网络分类器
from sklearn.neural_network import MLPClassifier

# 决策树分类器
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# 支持向量机分类器
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# 提升树
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier


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
    # app_desc_apptype 对app_desc进行预判断
    app_desc_apptype_path = project_path + '/data/cache/app_desc_apptype.h5'

    # apptype_train_classification.h5文件路径
    apptype_train_classification_path = project_path + '/data/cache/apptype_train_classification.h5'
    # app_desc_classification.h5文件路径
    app_desc_classification_path = project_path + '/data/cache/app_desc_classification.h5'

    # apptype_train_word_index.h5
    apptype_train_word_index_path = project_path + '/data/cache/apptype_train_word_index.h5'
    # app_desc_word_index.h5
    app_desc_word_index_path = project_path + '/data/cache/app_desc_word_index.h5'


    # 单模型
    AdaBoostClassifier_model = AdaBoostClassifier()
    BaggingClassifier_model = BaggingClassifier()
    ExtraTreesClassifier_model = ExtraTreesClassifier()
    GradientBoostingClassifier_model = GradientBoostingClassifier()
    RandomForestClassifier_model = RandomForestClassifier()
    GaussianProcessClassifier_model = GaussianProcessClassifier()
    PassiveAggressiveClassifier_model = PassiveAggressiveClassifier()
    RidgeClassifier_model = RidgeClassifier(alpha=0.8, tol=0.1, solver="sag", normalize=True, max_iter=1000, random_state=2019)
    SGDClassifier_model = SGDClassifier()
    KNeighborsClassifier_model = KNeighborsClassifier()
    GaussianNB_model = GaussianNB()
    MLPClassifier_model = MLPClassifier()
    DecisionTreeClassifier_model = DecisionTreeClassifier()
    ExtraTreeClassifier_model = ExtraTreeClassifier()
    SVC_model = SVC()
    LinearSVC_model = LinearSVC()
    # XGBClassifier_model = XGBClassifier()
    # LGBMClassifier_model = LGBMClassifier()
    LinearClassifierMixin_model = LinearClassifierMixin()
    RidgeClassifierCV_model = RidgeClassifierCV()
    SparseCoefMixin_model = SparseCoefMixin()

    # 选中的模型
    select_model = RidgeClassifier_model
    # select_model = 'lgb'
    # select_model = 'fast_text'

    # replace 是否进行替换
    not_replace = False


# 0.608/0.744  76.43278
# model = RidgeClassifier(random_state=2019)
# 0.552/0.672
# model = PassiveAggressiveClassifier(random_state=2019)
# 0.607/0.725  74.81213
# model = SGDClassifier(random_state=2019)
