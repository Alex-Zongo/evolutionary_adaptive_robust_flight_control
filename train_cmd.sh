#!/bin/bash
SEED1=0
SEED2=7

echo "Training for random seed: $SEED1"
python3 ES_TD3Buffers.py --seed=$SEED1 --elitism --cem_with_adapt --should_log

echo "Training for random seed: $SEED2"
python3 ES_TD3Buffers.py --seed=$SEED2 --elitism --cem_with_adapt --should_log