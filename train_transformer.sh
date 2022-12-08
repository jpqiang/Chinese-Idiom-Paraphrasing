#!/bin/sh

EXPERIMENT=
LOG=log/$EXPERIMENT

DATADIR=
TRAIN=model/$EXPERIMENT

mkdir -p $TRAIN
mkdir -p $LOG

CUDA_VISIBLE_DEVICES=1 fairseq-train $DATADIR \
    --arch transformer --dropout 0.2 --max-tokens 9600 --optimizer adam --lr 3e-4 --min-lr 1e-09 --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --weight-decay 0.0 --eval-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu --eval-bleu-print-samples --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu --no-progress-bar --log-interval 100 --keep-last-epochs 5 --patience 5 \
    --max-epoch 100 --save-dir $TRAIN --tensorboard-logdir $LOG
