import argparse
import os
import jieba

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# init
jieba.initialize()


def tokenize(line, delim=' '):
    _line = line
    # replace non-breaking whitespace
    _line = _line.replace("\xa0", " ").strip()
    # tokenize
    _tok = jieba.cut(_line.rstrip('\r\n'))
    _tokenized = delim.join(_tok)
    return _tokenized


def tokenize_file(filepath, delim=' '):
    filename = os.path.basename(filepath)

    tokenized = ''
    f = open(filepath, 'rb')
    for i, line in enumerate(f):
        line = line.decode('utf-8')  # decode
        if i % 3000 == 0:
            _tokenizer_name = "jieba"
            logger.info("     [%d] %s: %s" % (i, _tokenizer_name, line))

        # tokenize  
        _tokenized = tokenize(line, delim)

        # append
        tokenized += "%s\n" % _tokenized
    f.close()
    return tokenized


parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='input filepath')
parser.add_argument('--output', required=True, help='output filepath')
parser.add_argument('--delim', required=False, default=" ", help='delimiter, default=" "')

if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    tokenized = tokenize(opt.input, delim=opt.delim)
    fo = open(opt.output, 'w')
    fo.write(tokenized)
    fo.close()
