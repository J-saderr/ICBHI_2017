#!/bin/bash

MODEL="hftt"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="seed${s}_test_enhanced"
        CUDA_VISIBLE_DEVICES=0 python ./main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 5 \
                                        --batch_size 8 \
                                        --desired_length 5 \
                                        --optimizer adamw \
                                        --learning_rate 3e-5 \
                                        --weight_decay 1e-4 \
                                        --cosine \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.999 \
                                        --from_sl_official \
                                        --method pafa \
                                        --audioset_pretrained \
                                        --w_ce 1.0 \
                                        --w_pafa 1.0 \
                                        --lambda_pcsl 100.0 \
                                        --lambda_gpal 0.001 \
                                        --norm_type ln \
                                        --output_dim 768 \
                                        --freeze_encoder \
                                        --gradient_accumulation_steps 1 \
                                        --warmup_epochs 1 \
                                        --label_smoothing 0.1
    done
done
