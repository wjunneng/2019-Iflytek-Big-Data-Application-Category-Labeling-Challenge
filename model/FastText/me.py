import time
from scipy import sparse
from demo.util import *
from demo.config import DefaultConfig
import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.preprocessing import sequence

from keras import Input, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense


class FastText(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen,))

        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
        x = GlobalAveragePooling1D()(embedding)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model


def main():
    start = time.clock()

    # 获取数据
    app_desc = get_app_desc()
    apptype_id_name = get_apptype_id_name()
    apptype_train = get_apptype_train()
    print('获取数据集 耗时： %s \n' % str(time.clock() - start))

    # 预判断
    app_desc = get_app_desc_apptype(app_desc, apptype_id_name)
    print('预判断 耗时： %s \n' % str(time.clock() - start))

    # 获取label1/label2特征列
    apptype_train = get_label1_label2(apptype_train)
    print('获取label1/label2特征列 耗时： %s \n' % str(time.clock() - start))

    # 若label2中存在数据则新添加
    apptype_train = add_new_apptype_train_data(apptype_train)
    print('若label2中存在数据则新添加 耗时： %s \n' % str(time.clock() - start))

    # 删除出现次数少于5次的数据
    k = DefaultConfig.k
    apptype_train = delete_counts_less_than_k(apptype_train, k)
    print('删除出现次数少于5次的数据 耗时： %s \n' % str(time.clock() - start))

    return apptype_train, app_desc


def transform_word_to_index(train, test, max_features, topK, save=True, **params):
    """
    将word转化为index
    :param train:
    :param params:
    :return:
    """
    import os
    import jieba.analyse

    if os.path.exists(DefaultConfig.apptype_train_word_index_path) and os.path.exists(
            DefaultConfig.app_desc_word_index_path) and DefaultConfig.not_replace:
        train = reduce_mem_usage(
            pd.read_hdf(path_or_buf=DefaultConfig.apptype_train_word_index_path, mode='r', key='train'))
        test = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.app_desc_word_index_path, mode='r', key='test'))
    else:
        # 获取topK个词语
        train['conment'] = train['conment'].apply(lambda x: jieba.analyse.extract_tags(x, topK=topK))
        test['conment'] = test['conment'].apply(lambda x: jieba.analyse.extract_tags(x, topK=topK))

        # 停用词
        stopwords = get_stopwords()

        words = set()
        for conment in train['conment'].values:
            # 合并
            words = words | set([i for i in conment if len(i) > 1 and len(i) < 7])

        # 去除停用词
        words = words - stopwords

        # 长度限制
        if len(words) > max_features:
            words = list(words)[:max_features]
        else:
            words = list(words)
            max_features = len(words)

        # words 转 index
        train['conment'] = train['conment'].apply(lambda x: [words.index(i) for i in x if i in words])
        # words 转 index
        test['conment'] = test['conment'].apply(lambda x: [words.index(i) for i in x if i in words])

        if save:
            train.to_hdf(path_or_buf=DefaultConfig.apptype_train_word_index_path, mode='w', type='train')
            test.to_hdf(path_or_buf=DefaultConfig.app_desc_word_index_path, mode='w', type='test')

    return train, test, max_features


if __name__ == '__main__':
    # train = pd.DataFrame({'label1': [0, 1], 'conment': [
    #     '《游戏王座》使用说明书成分由怪兽卡、魔法卡、陷阱卡合计数千张卡牌以及刺激性、耐久性玩法组成。性状本游戏为真正集换式卡牌游戏TCG，随着玩家的深入而不断释放独特魅力。功能主治锻炼思维，形成头脑风暴。对智商105以上者的智商有明显促进作用，对无脑游戏厌恶症、收集强迫症患者有治愈之疗效。不良反应经临床实验，对智商90以下者的智商和自信有降低和灭杀反应。注意事项使用本游戏时请务必仔细阅读游戏内的新手入门，同时请参考不良反应。试用后请自行判断自己是否适合继续使用。更新内容《游戏王座》使用说明书成分由怪兽卡、魔法卡、陷阱卡合计数千张卡牌以及刺激性、耐久性玩法组成。性状本游戏为真正集换式卡牌游戏TCG，随着玩家的深入而不断释放独特魅力。功能主治锻炼思维，形成头脑风暴。对智商105以上者的智商有明显促进作用，对无脑游戏厌恶症、收集强迫症患者有治愈之疗效。不良反应经临床实验，对智商90以下者的智商和自信有降低和灭杀反应。注意事项使用本游戏时请务必仔细阅读游戏内的新手入门，同时请参考不良反应。试用后请自行判断自己是否适合继续使用。',
    #     '《小钱袋》是一款免费网络版记帐软件，适用于个人记帐、家庭记帐、团队记帐，全程帮您安全记录您财富的增长过程理财不分大小，从记账做起，从《小钱袋》开始更新内容修复账单明细列表显示错误问题您的反馈意见是我们前进的动力']})
    # test = pd.DataFrame({'label1': [0, 1], 'conment': [
    #     '《游戏王座》使用说明书成分由怪兽卡、魔法卡、陷阱卡合计数千张卡牌以及刺激性、耐久性玩法组成。性状本游戏为真正集换式卡牌游戏TCG，随着玩家的深入而不断释放独特魅力。功能主治锻炼思维，形成头脑风暴。对智商105以上者的智商有明显促进作用，对无脑游戏厌恶症、收集强迫症患者有治愈之疗效。不良反应经临床实验，对智商90以下者的智商和自信有降低和灭杀反应。注意事项使用本游戏时请务必仔细阅读游戏内的新手入门，同时请参考不良反应。试用后请自行判断自己是否适合继续使用。更新内容《游戏王座》使用说明书成分由怪兽卡、魔法卡、陷阱卡合计数千张卡牌以及刺激性、耐久性玩法组成。性状本游戏为真正集换式卡牌游戏TCG，随着玩家的深入而不断释放独特魅力。功能主治锻炼思维，形成头脑风暴。对智商105以上者的智商有明显促进作用，对无脑游戏厌恶症、收集强迫症患者有治愈之疗效。不良反应经临床实验，对智商90以下者的智商和自信有降低和灭杀反应。注意事项使用本游戏时请务必仔细阅读游戏内的新手入门，同时请参考不良反应。试用后请自行判断自己是否适合继续使用。',
    #     '《小钱袋》是一款免费网络版记帐软件，适用于个人记帐、家庭记帐、团队记帐，全程帮您安全记录您财富的增长过程理财不分大小，从记账做起，从《小钱袋》开始更新内容修复账单明细列表显示错误问题您的反馈意见是我们前进的动力']})
    # train, test, max_features = transform_word_to_index(train, test, max_features=10000, topK=50)
    #

    maxlen = 400
    epochs = 100
    embedding_dims = 50
    batch_size = 128

    train, test = main()
    train, test, max_features = transform_word_to_index(train, test, max_features=100000, topK=50)

    x_train = train['conment'].values
    y_train = train['label1'].values

    x_test = test['conment'].values

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = FastText(maxlen=maxlen, max_features=max_features, embedding_dims=embedding_dims,
                     class_num=x_train['label1'].max() + 1,
                     last_activation='softmax').get_model()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')

    early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[early_stopping])

    print('Test...')
    result = model.predict(x_test)
    print(result)
