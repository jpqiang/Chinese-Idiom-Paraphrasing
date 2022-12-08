# modify for every experiments
EXPERIMENT=
DOMAIN=

OUTPUT_DIR=model/$EXPERIMENT
BIN_DIR=data_bin/$DOMAIN

MODEL=$OUTPUT_DIR/checkpoint_best.pt
RESULT_DIR=result/$EXPERIMENT

echo "generate..."
num_beam=5
mkdir -p $RESULT_DIR/result_beam_$num_beam
fairseq-generate $BIN_DIR \
    --path $MODEL \
    --beam $num_beam --batch-size 128  --remove-bpe > $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.out

# get pair and resort
grep ^H $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.out | sed 's/^H-//g' | sort -k1,1n | cut -f 3 | sed 's/ ##//g' > $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.sys
grep ^T $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.out | sed 's/^T-//g' | sort -k1,1n | cut -f 2 | sed 's/ ##//g' > $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.ref

# detok
python preprocess/detokenizer.py --input $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.sys --output $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.sys
python preprocess/detokenizer.py --input $RESULT_DIR/result_beam_$num_beam/generate_beam_$num_beam.ref --output $RESULT_DIR/result_beam_$num_beam/generate_beam_detok_$num_beam.ref
