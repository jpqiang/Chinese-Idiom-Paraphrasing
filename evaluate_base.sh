# modify for every experiments
EXPERIMENT=transformer_in_domain
DOMAIN=in_domain

OUTPUT_DIR=output/$EXPERIMENT
BIN_DIR=data-bin/$DOMAIN

MODEL=$OUTPUT_DIR/averaged_model.pt
RESULT_DIR=result/$EXPERIMENT

# avg checkpoint for last 5
python fairseq/scripts/average_checkpoints.py \
--inputs output/$EXPERIMENT/ \
--num-epoch-checkpoints  5 \
--output output/$EXPERIMENT/averaged_model.pt \

num_beam=5
mkdir -p $RESULT_DIR/result_beam_$num_beam
fairseq-generate $BIN_DIR \
    --path $MODEL \
    --beam $num_beam --batch-size 128  --remove-bpe > $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.out

# get pair
#grep ^T $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.ref
#grep ^H $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.out | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.sys

# get pair and resort
grep ^H $RESULR_FILE | sed 's/^H-//g' | sort -k1,1n | cut -f 3 | sed 's/ ##//g' > $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.sys
grep ^T $RESULR_FILE | sed 's/^T-//g' | sort -k1,1n | cut -f 2 | sed 's/ ##//g' > $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.ref

# detok
python preprocess/detokenizer.py --input $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.sys --output $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.sys
python preprocess/detokenizer.py --input $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.ref --output $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.ref

# evaluate
sacrebleu $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.ref -i $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.sys --tokenize zh -w 2 -m bleu
bert-score -r $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.ref -c $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.sys --lang zh
python evaluate_rough.py --sys $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.sys --ref $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.ref
