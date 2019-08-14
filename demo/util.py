import pandas as pd

from config import DefaultConfig


def get_app_desc(**params):
    """
    返回 app_desc 文件内容
    :param params:
    :return:
    """
    app_desc_data = pd.read_csv(DefaultConfig.app_desc_path, header=None, encoding='utf8', delimiter=' ')
    # 以tab键分割，不知道为啥delimiter='\t'会报错，所以先读入再分割。
    app_desc_data = pd.DataFrame(app_desc_data[0].apply(lambda x: x.split('\t')).tolist(), columns=['id', 'conment'])

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


def get_app_desc_apptype(app_desc, apptype_id_name, save=True, **params):
    """
    预判断app_desc的label, 根据apptype_id_name中的label是否存在于conment中，来标注label_id
    :param app_desc:
    :param apptype_id_name:
    :param params:
    :return:
    """
    import os

    if os.path.exists(DefaultConfig.app_desc_apptype_path) and DefaultConfig.not_replace:
        app_desc = reduce_mem_usage(
            pd.read_hdf(path_or_buf=DefaultConfig.app_desc_apptype_path, mode='r', key='app_desc_apptype'))
    else:
        index = 0
        for i in range(apptype_id_name.shape[0]):
            if len(str(apptype_id_name.iloc[i, 0])) < 6:
                continue
            else:
                index = i
                break

        result = []
        for raw_app_desc in range(app_desc.shape[0]):
            if raw_app_desc % 10000 == 1:
                print(raw_app_desc)

            counts_dict = {}
            for raw_apptype_id_name in range(index, apptype_id_name.shape[0]):
                # 父字符串
                father = app_desc.iloc[raw_app_desc, 1]
                # 子字符串
                son = apptype_id_name.iloc[raw_apptype_id_name, 1]
                # 出现次数
                counts = father.count(son)
                # 如果出现次数大于等于8次， 则记录
                if counts >= 8:
                    counts_dict[str(apptype_id_name.iloc[raw_apptype_id_name, 0])] = counts

            counts_dict_sorted = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
            result.append([i[0] for i in counts_dict_sorted])

        app_desc['label'] = result

        if save:
            app_desc.to_hdf(path_or_buf=DefaultConfig.app_desc_apptype_path, key='app_desc_apptype')

    return app_desc


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


def add_new_apptype_train_data(df, **params):
    """
    实现将具备两个label的数据转化为两条记录
    :param df:
    :param params:
    :return:
    """
    df_new = df[df['label2'] != 0]
    df_new['label1'] = df['label2']
    # 合并数据
    df = pd.concat([df, df_new], axis=0)
    # label2列清空
    df['label2'] = 0

    return df


def get_classification(df, **params):
    """
    返回 类别
    :param df:
    :param params:
    :return:
    """
    result = {}

    # 寻找大类的类别编号
    for index in range(df.shape[0]):
        key = df.ix[index, 'label_code']
        value = df.ix[index, 'label']

        # 处理大类
        if len(str(key)) == 4:
            result[str(key)] = [value]
            # 处理小类
        else:
            # 获取前4个字符
            key_4 = str(key)[:4]
            # 如果小类属于的大类已经在结果中
            if key_4 in result.keys():
                # 如果小类对应的值未存在结果中
                if value not in result[key_4]:
                    # 添加
                    result[key_4].append(value)
            # 如果小类属于的大类未包含在结果中
            else:
                result[key_4] = [value]

    # 将list 转化为 dict
    for key in result.keys():
        result[key] = dict.fromkeys(result[key], True)

    return result


def deal_label_code(df, apptype_id_name, type, save=True, **params):
    """
    处理类别特征
    :param df:
    :param apptype_id_name:
    :param params:
    :return:
    """
    from scipy.sparse import csr_matrix
    import numpy as np
    import os

    assert list(df.columns) == ['id', 'conment']

    result = None
    if os.path.exists(DefaultConfig.apptype_train_classification_path) and os.path.exists(
            DefaultConfig.app_desc_classification_path):
        if type is 'train':
            result = load_sparse_mat(name='apptype_train_classification',
                                     filename=DefaultConfig.apptype_train_classification_path)
        if type is 'test':
            result = load_sparse_mat(name='app_desc_classification',
                                     filename=DefaultConfig.app_desc_classification_path)
    else:
        classification = get_classification(apptype_id_name)

        # 内容
        conments = df['conment']
        # 将list 转化为 dict Python 字典中使用了 hash table，因此查找操作的复杂度为 O(1)，
        # 而 list 实际是个数组，在 list 中，查找需要遍历整个 list，其复杂度为 O(n)，因此对成员的查找访问等操作字典要比 list 更快。
        keys = dict.fromkeys(classification.keys(), True)
        # 添加多列
        df = pd.concat([df, pd.DataFrame(columns=classification.keys())], sort=False)
        # 删除 ‘id’, 'conment' 列
        del df['id']
        del df['conment']

        counts_all = []
        for index in range(df.shape[0]):
            if index % 1000 == 1:
                print(index)

            # 获取内容
            conment = conments[index]

            # 每行的数目
            counts = []
            for column in keys:
                # 计算出现次数
                counts.append(sum([conment.count(label) for label in classification[column]]))

            counts_all.append(counts)

        # 为df赋值
        df = pd.DataFrame(data=np.array(counts_all), columns=df.columns)

        # 标准化
        df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        result = csr_matrix(df.values)

        if type is 'train' and save:
            store_sparse_mat(M=result, name='apptype_train_classification',
                             filename=DefaultConfig.apptype_train_classification_path)
        if type is 'test' and save:
            store_sparse_mat(M=result, name='app_desc_classification',
                             filename=DefaultConfig.app_desc_classification_path)
    return result


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

    if os.path.exists(apptype_train_term_doc_path) and os.path.exists(
            app_desc_term_doc_path) and DefaultConfig.not_replace:
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

    lbl = None
    for column in columns:
        # 构造标签属性
        lbl = preprocessing.LabelEncoder()

        # 训练
        lbl.fit(apptype_train[column].values)
        apptype_train[column] = lbl.transform(apptype_train[column].values)

    return apptype_train, lbl


def cross_validation(apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc, **params):
    """
    k折交叉验证
    :param apptype_train:
    :param app_desc:
    :param apptype_train_term_doc:
    :param app_desc_term_doc:
    :param params:
    :return:
    """
    import time
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
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

        model = DefaultConfig.select_model.fit(apptype_train_term_doc[tr], label[tr])
        score_va = model._predict_proba_lr(apptype_train_term_doc[va])
        score_te = model._predict_proba_lr(app_desc_term_doc)

        stack_train[va] += score_va
        stack_test += score_te
        print('consuming time:', time.clock() - start)

    print("model acc_score:",
          metrics.accuracy_score(label, np.argmax(stack_train, axis=1), normalize=True, sample_weight=None))

    return stack_train, stack_test


def lgb_model(apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc, **params):
    """
    lgb模型
    :param apptype_train:
    :param app_desc:
    :param apptype_train_term_doc:
    :param app_desc_term_doc:
    :param params:
    :return:
    """
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn import metrics

    # 类别数 122
    num_class = apptype_train['label1'].max() + 1
    # 类别
    label = apptype_train['label1']

    n_splits = 5

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'nthread': -1,
        'silent': True,  # 是否打印信息，默认False
        'learning_rate': 0.01,
        'num_leaves': 1000,
        'max_depth': 7,  # 第二次交叉验证得到的参数
        'max_bin': 127,
        'subsample_for_bin': 1000,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
    }

    oof_lgb = np.zeros((apptype_train.shape[0], num_class))
    prediction_lgb = np.zeros((app_desc.shape[0], num_class))
    for i, (tr, va) in enumerate(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019).split(apptype_train_term_doc, label)):
        print('fold:', i + 1, 'training')
        # 训练：
        bst = LGBMClassifier(**params).fit(X=apptype_train_term_doc[tr], y=label[tr])
        # 预测验证集：
        oof_lgb[va] += bst.predict_proba(apptype_train_term_doc[va], num_iteration=bst.best_iteration_)
        # 预测测试集：
        prediction_lgb += bst.predict_proba(app_desc_term_doc, num_iteration=bst.best_iteration_)

    print("model acc_score:",
          metrics.accuracy_score(label, np.argmax(oof_lgb, axis=1), normalize=True, sample_weight=None))

    return oof_lgb, prediction_lgb


def lgb_model_hyperparameter(apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc, **params):
    """
    参数设置
    :param apptype_train:
    :param app_desc:
    :param apptype_train_term_doc:
    :param app_desc_term_doc:
    :param params:
    :return:
    """
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    # 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置
    X, val_X, y, val_y = train_test_split(
        apptype_train_term_doc,
        apptype_train['label1'],
        test_size=0.2,
        random_state=1,
        # 这里保证分割后y的比例分布与原数据一致
        stratify=apptype_train['label1']
    )

    X_train = X
    y_train = y
    X_test = val_X
    y_test = val_y

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 9,
        'metric': 'multi_error',
        'num_leaves': 300,
        'min_data_in_leaf': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.4,
        'lambda_l2': 0.5,
        'min_gain_to_split': 0.2,
        'verbose': 5,
        'is_unbalance': True
    }

    # train
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=500)

    print('Start predicting...')
    preds = gbm.predict(app_desc_term_doc, num_iteration=gbm.best_iteration)  # 输出的是概率结果

    # 导出结果
    for pred in preds:
        prediction = int(np.argmax(pred))

    # 导出特征重要性
    importance = gbm.feature_importance()
    names = gbm.feature_name()
    with open('../data/feature_importance.txt', 'w+') as file:
        for index, im in enumerate(importance):
            string = names[index] + ', ' + str(im) + '\n'
            file.write(string)

    return prediction


def xgb_model(apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc, **params):
    """
    xgb模型
    :param apptype_train:
    :param app_desc:
    :param apptype_train_term_doc:
    :param app_desc_term_doc:
    :param params:
    :return:
    """
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn import metrics

    # 类别数 122
    num_class = apptype_train['label1'].max() + 1
    # 类别
    label = apptype_train['label1']

    n_splits = 5

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'nthread': -1,
        'silent': True,  # 是否打印信息，默认False
        'learning_rate': 0.01,
        'num_leaves': 1000,
        'max_depth': 7,  # 第二次交叉验证得到的参数
        'max_bin': 127,
        'subsample_for_bin': 1000,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
    }

    oof_xgb = np.zeros((apptype_train.shape[0], num_class))
    prediction_xgb = np.zeros((app_desc.shape[0], num_class))
    for i, (tr, va) in enumerate(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019).split(apptype_train_term_doc, label)):
        print('fold:', i + 1, 'training')
        # 训练：
        bst = XGBClassifier(**params)
        bst.fit(X=apptype_train_term_doc[tr], y=label[tr])
        # 预测验证集：
        oof_xgb[va] += bst.predict_proba(apptype_train_term_doc[va])
        # 预测测试集：
        prediction_xgb += bst.predict_proba(app_desc_term_doc)

    print("model acc_score:",
          metrics.accuracy_score(label, np.argmax(oof_xgb, axis=1), normalize=True, sample_weight=None))

    return oof_xgb, prediction_xgb


def get_offline_accuracy(apptype_train, app_desc, stack_train, stack_test, lbl, **params):
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
    print(DefaultConfig.select_model.__str__().split('(')[0])

    import time
    start = time.clock()

    label1_list = []
    label2_list = []
    for i in range(app_desc.shape[0]):
        if i % 10000 == 1:
            print(i)

        original = app_desc.ix[i, 'label']
        original.append(results.ix[i, 'label1'])
        original.append(results.ix[i, 'label2'])
        label1_list.append(original[0])
        label2_list.append(original[1])

    app_desc['label1'] = label1_list
    app_desc['label2'] = label2_list

    print(time.clock() - start)

    pd.concat([app_desc[['id', 'label1', 'label2']]], axis=1).to_csv(
        '../data/submit/baseline_' + DefaultConfig.select_model.__str__().split('(')[0] + '.csv',
        index=None,
        encoding='utf8')
