import os
from demo.config import DefaultConfig
from demo import util


def get_stopwords():
    """
    合并各个不同的停用词表
    :return:
    """
    try:
        stopwords_list = [u'中文停用词表.txt', u'哈工大停用词表.txt', u'四川大学机器智能实验室停用词库.txt', u'百度停用词表.txt']

        # apptype_id_name
        apptype_id_name = util.get_apptype_id_name()

        result = set()
        for stopwords_file in stopwords_list:
            print(stopwords_file)
            with open(DefaultConfig().project_path + '/data/stopwords/' + stopwords_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line not in result and line not in apptype_id_name[u'label']:
                        result.add(line)

        with open(DefaultConfig().stopwords_path, 'w') as file:
            for words in result:
                file.write(words)

        return True

    except Exception as e:
        print(e.args)

        return False


if __name__ == '__main__':
    print(get_stopwords())
