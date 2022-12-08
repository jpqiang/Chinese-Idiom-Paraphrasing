MODEL_DIR=model/
TEST_FILE=data/
OUT_DIR=result/

mkdir -p $OUT_DIR

CUDA_VISIBLE_DEVICES=0 python generate_t5_pegasus.py \
    --model_name $MODEL_DIR \
    --validation_file $TEST_FILE \
    --num_beams 5 \
    --per_device_eval_batch_size=32 \
    --max_target_length 128 \
    --output_path $OUT_DIR/test.t5_pagesus_in.sys \
