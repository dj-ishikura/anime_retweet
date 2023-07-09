# -*- coding: utf-8 -*-

import json

from log import get_logger
logger = get_logger(__name__)

def parse_tweet(inputfile):
    '''ツイートJSONをパースしてdictを返すジェネレータ'''
    for no, line in enumerate(inputfile):
        try:
            obj = json.loads(line.rstrip())
            yield obj
        except ValueError as e:
            logger.debug('ValueError: %s', e)
            logger.info('Found broken JSON at line %d', no)