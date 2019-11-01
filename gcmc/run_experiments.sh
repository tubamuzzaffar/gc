#!/bin/bash

for i in 1000, 750, 500, 250, 125
do
# Movielens 100K on official split with features
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden 10 --testing > ml_100k_feat_testing.txt  2>&1
done

