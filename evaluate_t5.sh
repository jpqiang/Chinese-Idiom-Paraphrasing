MODEL_DIR=model_path
TEST_FILE=test_file

CUDA_VISIBLE_DEVICES=0 python evaluate_t5.py \
    --model_name_or_path $MODEL_DIR \
    --validation_file $TEST_FILE \
    --text_column src \
    --simplification_column dst \
    --num_beams 5 \
    --source_prefix "simplify: " \
    --per_device_eval_batch_size=32 \
    --val_max_target_length 256 \
