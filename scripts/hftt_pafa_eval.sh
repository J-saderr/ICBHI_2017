#!/bin/bash
# File: scripts/hftt_pafa_eval.sh

MODEL="hftt"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="seed${s}_best"
        CUDA_VISIBLE_DEVICES=0 python ./main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 20 \
                                        --batch_size 32 \
                                        --desired_length 5 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --from_sl_official \
                                        --method pafa \
                                        --w_ce 1.0 \
                                        --w_pafa 1.0 \
                                        --lambda_pcsl 50.0 \
                                        --lambda_gpal 0.0005 \
                                        --norm_type ln \
                                        --output_dim 768 \
                                        --nospec \
                                        --eval \
                                        --pretrained \
                                        --pretrained_ckpt ./save/icbhi_hftt_pafa_seed${s}_best/best.pth

    done
done