export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

DATASET=news
LABEL_NAME_FILE=news_category.txt
TRAIN_CORPUS=news_train.txt
TEST_CORPUS=news_train.txt
TEST_LABEL=news_train_labels.txt
MAX_LEN=200
TRAIN_BATCH=32
ACCUM_STEP=2
EVAL_BATCH=128
GPUS=2
MCP_EPOCH=30
SELF_TRAIN_EPOCH=0

python src/train.py --dataset_dir datasets/${DATASET}/ --label_names_file ${LABEL_NAME_FILE} \
                    --train_file ${TRAIN_CORPUS} \
                    --test_file ${TEST_CORPUS} --test_label_file ${TEST_LABEL} \
                    --max_len ${MAX_LEN} \
                    --train_batch_size ${TRAIN_BATCH} --accum_steps ${ACCUM_STEP} --eval_batch_size ${EVAL_BATCH} \
                    --gpus ${GPUS} \
                    --mcp_epochs ${MCP_EPOCH} --self_train_epochs ${SELF_TRAIN_EPOCH} \