#!/usr/bin/env bash

for i in {0..5};
    #do sh ./ww.sh $i &
    python run_101_learning_split_2samp_R1_randomization.py $i &
done

wait
