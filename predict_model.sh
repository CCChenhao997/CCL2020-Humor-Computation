#!/bin/bash


# en
python predict_model.py --model_name bert_spc --dataset en --fold_n 0 --pseudo True --cuda 2
python predict_model.py --model_name bert_spc --dataset en --fold_n 1 --pseudo True --cuda 2
python predict_model.py --model_name bert_spc --dataset en --fold_n 2 --pseudo True --cuda 2
python predict_model.py --model_name bert_spc --dataset en --fold_n 3 --pseudo True --cuda 2
python predict_model.py --model_name bert_spc --dataset en --fold_n 4 --pseudo True --cuda 2