#!/bin/sh

DOMAIN=
TEXT=
DATADIR=

mkdir -p $DATADIR

# build subword vocab
SUBWORD_NMT=subword-nmt/subword_nmt
NUM_OPS=32000

# learn codes and encode separately
CODES=codes.${NUM_OPS}.bpe
echo "Encoding subword with BPE using ops=${NUM_OPS}"
$SUBWORD_NMT/learn_bpe.py --num-workers 8 -s ${NUM_OPS} < $TEXT/train.src > $TEXT/${CODES}.src
$SUBWORD_NMT/learn_bpe.py --num-workers 8 -s ${NUM_OPS} < $TEXT/train.dst > $TEXT/${CODES}.dst

echo "Applying vocab to training"
$SUBWORD_NMT/apply_bpe.py --num-workers 8 -c $TEXT/${CODES}.src < $TEXT/train.src > $TEXT/train.${NUM_OPS}.bpe.src
$SUBWORD_NMT/apply_bpe.py --num-workers 8 -c $TEXT/${CODES}.dst < $TEXT/train.dst > $TEXT/train.${NUM_OPS}.bpe.dst

VOCAB=vocab.${NUM_OPS}.bpe
echo "Generating vocab: ${VOCAB}.src"
cat $TEXT/train.${NUM_OPS}.bpe.src | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.src

echo "Generating vocab: ${VOCAB}.dst"
cat $TEXT/train.${NUM_OPS}.bpe.dst | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.dst

# encode validation
echo "Applying vocab to valid"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.src --vocabulary $TEXT/${VOCAB}.src < $TEXT/valid.${DOMAIN}.src > $TEXT/valid.${NUM_OPS}.bpe.src
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.dst --vocabulary $TEXT/${VOCAB}.dst < $TEXT/valid.${DOMAIN}.dst > $TEXT/valid.${NUM_OPS}.bpe.dst

# encode test
echo "Applying vocab to test"
$SUBWORD_NMT/apply_bpe.py --num-workers 8 -c $TEXT/${CODES}.src --vocabulary $TEXT/${VOCAB}.src < $TEXT/test.${DOMAIN}.src > $TEXT/test.${NUM_OPS}.bpe.src
$SUBWORD_NMT/apply_bpe.py --num-workers 8 -c $TEXT/${CODES}.dst --vocabulary $TEXT/${VOCAB}.dst < $TEXT/test.${DOMAIN}.dst > $TEXT/test.${NUM_OPS}.bpe.dst

# generate preprocessed data
echo "Preprocessing datasets..."
fairseq-preprocess --source-lang src --target-lang dst \
    --trainpref $TEXT/train.${NUM_OPS}.bpe --validpref $TEXT/valid.${NUM_OPS}.bpe --testpref $TEXT/test.${NUM_OPS}.bpe \
    --thresholdtgt 0 --thresholdsrc 0 --workers 8 --destdir $DATADIR
