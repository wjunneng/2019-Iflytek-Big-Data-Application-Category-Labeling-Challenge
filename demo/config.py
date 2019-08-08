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

    # 项目路径
    project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])




if __name__ == '__main__':
    config = DefaultConfig()
