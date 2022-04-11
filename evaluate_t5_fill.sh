MODEL_DIR=model_path
TEST_FILE=test_file

CUDA_VISIBLE_DEVICES=0 python generate_t5_fill.py \
    --model_path $MODEL_DIR \
    --idioms 'idioms.txt' \
    --test_path $TEST_FILE \
    --max_length 256 \
    --beam 5
