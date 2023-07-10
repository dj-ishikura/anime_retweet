# -*- coding: utf-8 -*-
'''
Utilities
'''

def get_logger(name):
    '''いいかんじのloggerを構築して返す'''
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger