import time
from scipy import sparse
from demo.util import *
from demo.config import DefaultConfig
from imblearn.over_sampling import SMOTE


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

    # # 删除出现次数少于5次的数据
    # k = DefaultConfig.k
    # apptype_train = delete_counts_less_than_k(apptype_train, k)
    # print('删除出现次数少于5次的数据 耗时： %s \n' % str(time.clock() - start))

    # 获取TF-IDF矩阵
    apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc = get_term_doc(apptype_train, app_desc)
    print('获取TF-IDF矩阵 耗时： %s \n' % str(time.clock() - start))

    # 对label1进行labelEncoder
    apptype_train, lbl = get_label_encoder(apptype_train, columns=['label1'])
    print('对label1进行labelEncoder 耗时： %s \n' % str(time.clock() - start))

    # 对apptype_train/app_desc 添加类别特征  合并csr_matrix
    # apptype_train_classification = deal_label_code(apptype_train[['id', 'conment']], apptype_id_name, type='train')
    # app_desc_classification = deal_label_code(app_desc[['id', 'conment']], apptype_id_name, type='test')
    # apptype_train_term_doc = sparse.hstack((apptype_train_term_doc, apptype_train_classification), format='csr')
    # app_desc_term_doc = sparse.hstack((app_desc_term_doc, app_desc_classification), format='csr')
    # print('添加了类别特征 耗时： %s \n' % str(time.clock() - start))

    print('apptype_train_term_doc.shape: ', apptype_train_term_doc.shape)
    print('\n')

    if DefaultConfig.select_model is 'lgb':
        # 交叉验证
        stack_train, stack_test = lgb_model(apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc)
        print('lgb交叉验证 耗时： %s \n' % str(time.clock() - start))
        # 线下准确率+测试结果
        get_offline_accuracy(apptype_train, app_desc, stack_train, stack_test, lbl)
        print('线下准确率+测试结果 耗时： %s \n' % str(time.clock() - start))

    elif DefaultConfig.select_model is 'xgb':
        # 交叉验证
        stack_train, stack_test = xgb_model(apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc)
        print('xgb交叉验证 耗时： %s \n' % str(time.clock() - start))
        # 线下准确率+测试结果
        get_offline_accuracy(apptype_train, app_desc, stack_train, stack_test, lbl)
        print('线下准确率+测试结果 耗时： %s \n' % str(time.clock() - start))

    else:
        # 交叉验证
        stack_train, stack_test = cross_validation(apptype_train, app_desc, apptype_train_term_doc, app_desc_term_doc)
        print('交叉验证 耗时： %s \n' % str(time.clock() - start))

        # 线下准确率+测试结果
        get_offline_accuracy(apptype_train, app_desc, stack_train, stack_test, lbl)
        print('线下准确率+测试结果 耗时： %s \n' % str(time.clock() - start))


if __name__ == '__main__':
    main()
