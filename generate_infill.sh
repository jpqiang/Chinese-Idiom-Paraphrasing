MODEL_DIR=model/
TEST_FILE=data/
OUT_DIR=result/

mkdir -p $OUT_DIR

python generate_infill.py \
    --model_path $MODEL_DIR \
    --test_path $TEST_FILE \
    --beam 5 \
    --output_path $OUT_DIR \
