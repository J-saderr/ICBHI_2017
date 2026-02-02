#!/bin/bash

# Fix temporary directory issue for torchvision
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"

MODEL="hftt"
SEED="2"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="seed${s}_hftt"
        CUDA_VISIBLE_DEVICES=0 python ./test.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --from_sl_official \
                                        --method projectors_loss \
                                        --norm_type ln \
                                        --output_dim 768 \
                                        --nospec
    done
done