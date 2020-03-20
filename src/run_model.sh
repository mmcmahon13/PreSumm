#!/bin/bash

export MODEL_DIR=mnt/data/models
export BERT_DATA=/mnt/data/bert_data_cnndm_final

python train.py -mode test -task ext -mode validate -batch_size 3000 -test_batch_size 500 -log_file ../logs/val_abs_bert_cnndm -model_path $MODEL_DIR -sep_optim true -use_interval true -visible_gpus -1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm -bert_data_path $BERT_DATA
