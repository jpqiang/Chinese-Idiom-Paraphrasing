TRAIN_FILE=train.csv
VALID_FILE=valid.csv
OUTPUT_MODEL=in_domain_model

mkdir -p $OUTPUT_MODEL
LOG=$OUTPUT_MODEL/log.txt

python train.py \
    --model_name_or_path mt5-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --num_train_epochs 20 \
    --text_column src \
    --simplification_column dst \
    --source_prefix "simplify: " \
    --output_dir $OUTPUT_MODEL \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --num_warmup_steps 0 \
    --seed 42 \
    --early_stop 5 \
    --learning_rate 3e-4 \
    --weight_decay 0.0 \
    --gradient_accumulation_steps 1 \
    --max_length 256 \
    --max_source_length 256 \
    --max_target_length 256 \
    --val_max_target_length 256 \
    --log_file $LOG
