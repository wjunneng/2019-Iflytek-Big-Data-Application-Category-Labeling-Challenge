import pandas as pd
from demo.config import DefaultConfig


def get_app_desc(**params):
    """
    返回 app_desc 文件内容
    :param params:
    :return:
    """
    app_desc_path = DefaultConfig.app_desc_path
    app_desc_data = pd.read_table(app_desc_path, header=None, sep='\t', names=['id', 'conment'])

    return app_desc_data


def get_apptype_id_name(**params):
    """
    返回 apptype_id_name 文件内容
    :param params:
    :return:
    """
    apptype_id_name_path = DefaultConfig.apptype_id_name_path
    apptype_id_name_data = pd.read_table(apptype_id_name_path, header=None, sep='\t', names=['label_code', 'label'])

    return apptype_id_name_data


def get_apptype_train(**params):
    """
    返回 apptype_train 文件内容
    :param params:
    :return:
    """
    apptype_train_data = pd.read_csv(DefaultConfig.apptype_train_path, header=None, encoding='utf8', delimiter=' ')
    # 以tab键分割，不知道为啥delimiter='\t'会报错，所以先读入再分割。
    apptype_train_data = pd.DataFrame(apptype_train_data[0].apply(lambda x: x.split('\t')).tolist(),
                                      columns=['id', 'label', 'conment'])

    return apptype_train_data


def get_stopwords(**params):
    """
    获取停用词文件数据
    :param params:
    :return:
    """
    with open(DefaultConfig.stopwords_path, 'r') as file:
        return file.readlines()


def get_label1_label2(df, **params):
    """
    构造label1和label2 特征
    :param df:
    :param params:
    :return:
    """
    df['label1'] = df['label'].apply(lambda x: x.split('|')[0])
    df['label2'] = df['label'].apply(lambda x: x.split('|')[1] if '|' in x else 0)

    return df


def delete_counts_less_than_k(df, k, **params):
    """
    删除出现次数少于k次的 数据
    :param df:
    :param k:
    :param params:
    :return:
    """
    counts_less_than_k = []

    for key, value in df['label1'].value_counts().items():
        if value < k:
            counts_less_than_k.append(key)

    df = df[~df['label1'].isin(counts_less_than_k)].reset_index(drop=True)

    return df


def reduce_mem_usage(df, verbose=True):
    """
    减少内存消耗
    :param df:
    :param verbose:
    :return:
    """
    import numpy as np

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def store_sparse_mat(M, name, filename='store.h5'):
    """
    Store a csr matrix in HDF5

    Parameters
    ----------
    M : scipy.sparse.csr.csr_matrix
     sparse matrix to be stored

    name: str
     node prefix in HDF5 hierarchy

    filename: str
     HDF5 filename
    """
    import numpy as np
    from scipy import sparse
    import tables

    assert (M.__class__ == sparse.csr.csr_matrix), 'M must be a csr matrix'
    with tables.open_file(filename, 'a') as f:
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            full_name = f'{name}_{attribute}'

            # remove existing nodes
            try:
                n = getattr(f.root, full_name)
                n._f_remove()
            except AttributeError:
                pass

                # add nodes
            arr = np.array(getattr(M, attribute))
            atom = tables.Atom.from_dtype(arr.dtype)
            ds = f.create_carray(f.root, full_name, atom, arr.shape)
            ds[:] = arr


def load_sparse_mat(name, filename='store.h5'):
    """
    Load a csr matrix from HDF5

    Parameters
    ----------
    name: str
     node prefix in HDF5 hierarchy

    filename: str
     HDF5 filename

    Returns
    ----------
    M : scipy.sparse.csr.csr_matrix
     loaded sparse matrix
    """
    from scipy import sparse
    import tables

    with tables.open_file(filename) as f:
        # get nodes
        attributes = []
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            attributes.append(getattr(f.root, f'{name}_{attribute}').read())

            # construct sparse matrix
    M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])
    return M


def get_term_doc(apptype_train, app_desc, save=True, **params):
    """
    获取train/test TF-IDF矩阵
    :param df:
    :param params:
    :return:
    """
    import os
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer

    apptype_train_term_doc_path = DefaultConfig.apptype_train_term_doc_path
    app_desc_term_doc_path = DefaultConfig.app_desc_term_doc_path

    if os.path.exists(apptype_train_term_doc_path) and os.path.exists(app_desc_term_doc_path):
        apptype_train_term_doc = load_sparse_mat(name='apptype_train_term_doc', filename=apptype_train_term_doc_path)
        app_desc_term_doc = load_sparse_mat(name='app_desc_term_doc', filename=app_desc_term_doc_path)
    else:
        stopwords = get_stopwords()

        print('stopwords length:', len(stopwords))
        apptype_train['conment'] = apptype_train['conment'].apply(lambda x: ' '.join(jieba.cut(x)))
        app_desc['conment'] = app_desc['conment'].apply(lambda x: ' '.join(jieba.cut(x)))

        vec = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.8, use_idf=1, smooth_idf=1, sublinear_tf=1,
                              stop_words=stopwords)  # 这里参数可以改
        apptype_train_term_doc = vec.fit_transform(apptype_train['conment'])
        app_desc_term_doc = vec.transform(app_desc['conment'])

        print(type(apptype_train_term_doc))
        if save:
            store_sparse_mat(M=apptype_train_term_doc, name='apptype_train_term_doc',
                             filename=apptype_train_term_doc_path)
            store_sparse_mat(M=app_desc_term_doc, name='app_desc_term_doc', filename=app_desc_term_doc_path)

    return apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc


def get_label_encoder(apptype_train, columns: list = None, **params):
    """
    返回经过labelEncoder后的数据
    :param apptype_train:
    :param params:
    :return:
    """
    from sklearn import preprocessing

    for column in columns:
        # 构造标签属性
        lbl = preprocessing.LabelEncoder()

        # 训练
        lbl.fit(apptype_train[column].values)
        apptype_train[column] = lbl.transform(apptype_train[column].values)

    return apptype_train, lbl


def cross_validation(apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc, **params):
    import time
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
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

    # K近邻分类器
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

    from xgboost import XGBClassifier
    from sklearn import metrics

    # 类别数 122
    num_class = apptype_train['label1'].max() + 1
    # 类别
    label = apptype_train['label1']

    n_splits = 5
    stack_train = np.zeros((apptype_train.shape[0], num_class))
    stack_test = np.zeros((app_desc.shape[0], num_class))

    for i, (tr, va) in enumerate(
            StratifiedKFold(n_splits=n_splits, random_state=2019).split(apptype_train_term_doc, label)):
        print('stack:%d/%d' % ((i + 1), n_splits))
        start = time.clock()

        # 0.608/0.744
        # model = RidgeClassifier(random_state=2019)
        # 0.552/0.672
        # model = PassiveAggressiveClassifier(random_state=2019)
        # 0.607/0.725
        model = SGDClassifier(random_state=2019)

        model.fit(apptype_train_term_doc[tr], label[tr])
        score_va = model._predict_proba_lr(apptype_train_term_doc[va])
        score_te = model._predict_proba_lr(app_desc_term_doc)

        stack_train[va] += score_va
        stack_test += score_te
        print('consuming time:', time.clock() - start)

    print("model acc_score:",
          metrics.accuracy_score(label, np.argmax(stack_train, axis=1), normalize=True, sample_weight=None))

    return stack_train, stack_test


def get_offline_accuracy(apptype_train, stack_train, **params):
    """
    返回线下准确率
    :return:
    """
    import numpy as np

    label = apptype_train['label1']

    # 获取第一第二个标签：取概率最大的前两个即可：
    m = pd.DataFrame(stack_train)
    first = []
    second = []
    for j, row in m.iterrows():
        zz = list(np.argsort(row))
        # 第一个标签
        first.append(row.index[zz[-1]])
        # 第二个标签
        second.append(row.index[zz[-2]])
    m['label1'] = first
    m['label2'] = second

    # 计算准确率，只要命中一个就算正确：
    k = 0
    for i in range(len(label)):
        if label[i] in [m.loc[i, 'label1'], m.loc[i, 'label2']]:
            k += 1
        else:
            pass
    print('线下准确率：%f' % (k / len(label)))


def get_prediction(apptype_train, app_desc, stack_test, lbl, **params):
    """
    获取结果
    :param stack_test:
    :param params:
    :return:
    """
    import numpy as np

    # 准备测试集结果：
    results = pd.DataFrame(stack_test)
    first = []
    second = []
    for j, row in results.iterrows():
        zz = list(np.argsort(row))
        # 第一个标签
        first.append(row.index[zz[-1]])
        # 第二个标签
        second.append(row.index[zz[-2]])
    results['label1'] = first
    results['label2'] = second

    print("len(list(train['label1'].values): ", len(list(apptype_train['label1'].values)))
    print(results.head())

    # 之前编码，最后逆编码回来：
    results['label1'] = lbl.inverse_transform(results['label1'].apply(lambda x: int(x)).values)
    results['label2'] = lbl.inverse_transform(results['label2'].apply(lambda x: int(x)).values)

    # 结合id列，保存：

    # pd.concat([app_desc[['id']], results[['label1', 'label2']]], axis=1).to_csv('../data/submit/baseline_ridge.csv',
    #                                                                             index=None,
    #                                                                             encoding='utf8')

    # pd.concat([app_desc[['id']], results[['label1', 'label2']]], axis=1).to_csv(
    #     '../data/submit/baseline_passiveAggressive.csv',
    #     index=None,
    #     encoding='utf8')

    pd.concat([app_desc[['id']], results[['label1', 'label2']]], axis=1).to_csv(
        '../data/submit/baseline_sgd.csv',
        index=None,
        encoding='utf8')

# if __name__ == '__main__':
#     # app_desc
#     result = get_app_desc()
#     print(result.head())
#
#     # apptype_id_name
#     result = get_apptype_id_name()
#     print(result.head())
#
#     # apptype_train
#     result = get_apptype_train()
#     print(result.head())
