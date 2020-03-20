#!/bin/bash

MODEL_PATH=/mnt/efs/models/bertsum/
BERT_DATA_PATH=/mnt/data/bert_data_cnndm_final/

python train.py -task ext -mode train -bert_data_path $BERT_DATA_PATH -ext_dropout 0.1 -model_path $MODEL_PATH -lr 2e-3 -visible_gpus -1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512
