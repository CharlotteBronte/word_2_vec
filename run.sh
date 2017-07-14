#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

#####################################################################
##测试数据
##test data
#####################################################################

SCRIPT="./script"
DATA="./data_dir"
CONFIG="./configs"
#cd $DATA/test
#sh build_dict.sh  seg_qa test
#cd -
#python $SCRIPT/word_2_vec.py $CONFIG/word_embedding_test.ini

#####################################################################
##全量qa训练数据
##whole train data
#####################################################################
python $SCRIPT/word_2_vec.py $CONFIG/word_embedding.ini

