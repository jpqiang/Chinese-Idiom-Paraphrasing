TRAIN_FILE=train.csv
VALID_FILE=valid.csv

OUTPUT_MODEL=model/
mkdir -p $OUTPUT_MODEL

LOG=$OUTPUT_MODEL/log.txt
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path mt5-base \
    --train_file $TRAIN_FILE \
    --valid_file $VALID_FILE \
    --end_token="[NULL]" \
    --num_train_epochs 30 \
    --preprocessing_num_workers 8 \
    --output_dir $OUTPUT_MODEL \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=8 \
    --num_warmup_steps 8000 \
    --seed 42 \
    --early_stop 5 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --num_beams 5 \
    --max_length 256 \
    --max_source_length 256 \
    --max_target_length 256 \
    --val_max_target_length 256 \
    --log_file $LOG \
    --pad_to_max_length \
    --idioms idioms.txt \
