import sys
import io
import argparse
import os
import jieba
import tokenizer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN = {
    "source": "train.src",
    "target": "train.dst",
}

DEV = {
    "source": "valid.src",
    "target": "valid.dst",
}

TEST = {
    "source": "test.src",
    "target": "test.dst",
}

DATA_DIR = "data/src_dst/"
RAW_DIR = "data_raw/"


def prepare_dataset(data_dir, raw_dir, dataset_config, tokenize=True):
    """ tokenize, copy files to data_dir """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    for _file in ["source", "target"]:

        print(_file)
        _tmp = dataset_config[_file]
        _data = dataset_config[_file]

        # skip if data file exists. 
        data_filepath = os.path.join(data_dir, _data)
        if os.path.isfile(data_filepath):
            logger.info("Found file: %s" % data_filepath)
            continue

        # get raw file
        tmp_filepath = os.path.join(raw_dir, _tmp)

        if tokenize:
            logger.info("tokenizing: %s" % tmp_filepath)
            tokenized = tokenizer.tokenize_file(tmp_filepath)
            logger.info("...done. writing to: %s" % data_filepath)
            f = open(data_filepath, 'w')
            f.write(tokenized)
            f.close()
        else:
            logger.info("tokenize=False, copying to %s" % data_filepath)
            os.rename(tmp_filepath, data_filepath)


if __name__ == '__main__':
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

    for ds in [TRAIN, DEV, TEST]:
        prepare_dataset(DATA_DIR, RAW_DIR, ds)
