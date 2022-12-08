EXPERIMENT=
LOG=log/$EXPERIMENT
OUTPUT_DIR=model/$EXPERIMENT

VALID=data/in_domain/valid.in.csv

mkdir -p $OUTPUT_DIR
mkdir -p $LOG

python train_t5_pegasus.py \
      --model_name model/t5_pagesus \
      --train_file data/train.csv \
      --validation_file $VALID \
      --output_dir $OUTPUT_DIR \
      --max_source_length 128 \
      --max_target_length 128 \
      --preprocessing_num_workers 128 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --learning_rate 3e-4 \
      --num_train_epochs 20 \
      --log_file $OUTPUT_DIR/log.txt
