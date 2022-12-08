EXPERIMENT=
LOG=log/$EXPERIMENT
OUTPUT_DIR=model/$EXPERIMENT

TRAIN=data/train.fill.csv
VALID=data/in_domain/valid.in.csv

mkdir -p $OUTPUT_DIR
mkdir -p $LOG

python train_infill.py \
      --model_name model/ \
      --train_file $TRAIN \
      --validation_file $VALID \
      --idioms idioms.txt \
      --output_dir $OUTPUT_DIR \
      --max_source_length 128 \
      --max_target_length 128 \
      --preprocessing_num_workers 128 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --learning_rate 3e-4 \
      --num_train_epochs 20 \
      --log_file $OUTPUT_DIR/log.txt
