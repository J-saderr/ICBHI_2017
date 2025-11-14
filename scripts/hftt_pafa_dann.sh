# #!/bin/bash

# MODEL="hftt"
# SEED="2"

# for s in $SEED
# do
#     for m in $MODEL
#     do
#         TAG="seed${s}_dann"
#         CUDA_VISIBLE_DEVICES=0 python ./main.py --tag $TAG \
#                                         --dataset icbhi \
#                                         --seed $s \
#                                         --class_split lungsound \
#                                         --n_cls 4 \
#                                         --epochs 20 \
#                                         --batch_size 32 \
#                                         --desired_length 5 \
#                                         --optimizer adam \
#                                         --learning_rate 3e-5 \
#                                         --weight_decay 1e-6 \
#                                         --cosine \
#                                         --model $m \
#                                         --test_fold official \
#                                         --pad_types repeat \
#                                         --resz 1 \
#                                         --n_mels 128 \
#                                         --ma_update \
#                                         --ma_beta 0.5 \
#                                         --from_sl_official \
#                                         --method pafa \
#                                         --audioset_pretrained \
#                                         --use_dann \
#                                         --w_ce 1.0 \
#                                         --w_pafa 1.0 \
#                                         --lambda_pcsl 53.0 \
#                                         --lambda_dann 0.3 \
#                                         --norm_type ln \
#                                         --output_dim 768 \
#                                         --nospec
#     done
# done

#!/bin/bash

#!/bin/bash
Checkpoint Name: icbhi_hftt_pafa_seed2_pcsl50_dann0.3_center0.0001_max20
Training for 20 epochs on icbhi dataset
GRL lambda: 0.2449
Train: [1][100/129]     BT 1.180 (1.175)        DT 0.026 (0.033)        Loss 2.541 (3.072)     Acc@1 68.750 (55.312)
Train epoch 1, total time 150.75, accuracy:56.88
 * S_p: 65.55, S_e: 49.11, Score: 57.33 (Best S_p: 65.55, S_e: 49.11, Score: 57.33)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 58.53
Best ckpt is modified with Score = 57.33 when Epoch = 1
GRL lambda: 0.4621
Train: [2][100/129]     BT 1.188 (1.153)        DT 0.023 (0.036)        Loss 2.534 (2.477)     Acc@1 53.125 (64.219)
Train epoch 2, total time 149.20, accuracy:64.34
 * S_p: 44.27, S_e: 58.54, Score: 51.40 (Best S_p: 65.55, S_e: 49.11, Score: 57.33)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 50.36
GRL lambda: 0.6351
Train: [3][100/129]     BT 1.033 (1.152)        DT 0.025 (0.035)        Loss 2.560 (2.327)     Acc@1 62.500 (70.531)
Train epoch 3, total time 148.13, accuracy:70.08
 * S_p: 52.75, S_e: 58.45, Score: 55.60 (Best S_p: 65.55, S_e: 49.11, Score: 57.33)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 55.19
GRL lambda: 0.7616
Train: [4][100/129]     BT 1.181 (1.148)        DT 0.027 (0.034)        Loss 2.193 (2.282)     Acc@1 84.375 (71.656)
Train epoch 4, total time 147.32, accuracy:72.02
 * S_p: 62.51, S_e: 53.53, Score: 58.02 (Best S_p: 62.51, S_e: 53.53, Score: 58.02)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 58.67
Best ckpt is modified with Score = 58.02 when Epoch = 4
GRL lambda: 0.8483
Train: [5][100/129]     BT 1.151 (1.161)        DT 0.027 (0.036)        Loss 2.240 (2.187)     Acc@1 75.000 (76.188)
Train epoch 5, total time 149.19, accuracy:75.48
 * S_p: 51.55, S_e: 58.62, Score: 55.09 (Best S_p: 62.51, S_e: 53.53, Score: 58.02)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 54.57
GRL lambda: 0.9051
Train: [6][100/129]     BT 1.192 (1.161)        DT 0.027 (0.034)        Loss 2.085 (2.131)     Acc@1 81.250 (77.719)
Train epoch 6, total time 150.10, accuracy:77.40
 * S_p: 70.74, S_e: 50.13, Score: 60.43 (Best S_p: 70.74, S_e: 50.13, Score: 60.43)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 61.94
Best ckpt is modified with Score = 60.43 when Epoch = 6
GRL lambda: 0.9414
Train: [7][100/129]     BT 1.129 (1.160)        DT 0.021 (0.036)        Loss 2.201 (2.082)     Acc@1 84.375 (79.531)
Train epoch 7, total time 149.54, accuracy:79.82
 * S_p: 70.42, S_e: 51.06, Score: 60.74 (Best S_p: 70.42, S_e: 51.06, Score: 60.74)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 62.16
Best ckpt is modified with Score = 60.74 when Epoch = 7
GRL lambda: 0.9640
Train: [8][100/129]     BT 1.198 (1.164)        DT 0.027 (0.036)        Loss 2.041 (2.039)     Acc@1 84.375 (81.812)
Train epoch 8, total time 149.52, accuracy:81.42
 * S_p: 74.16, S_e: 50.98, Score: 62.57 (Best S_p: 74.16, S_e: 50.98, Score: 62.57)
 * F1 Score: 0.31 (F1 Score: 0.31)
 * Acc@1 64.26
Best ckpt is modified with Score = 62.57 when Epoch = 8
GRL lambda: 0.9780
Train: [9][100/129]     BT 1.122 (1.160)        DT 0.027 (0.036)        Loss 2.039 (1.985)     Acc@1 78.125 (83.844)
Train epoch 9, total time 148.84, accuracy:83.75
 * S_p: 71.37, S_e: 49.45, Score: 60.41 (Best S_p: 74.16, S_e: 50.98, Score: 62.57)
 * F1 Score: 0.30 (F1 Score: 0.31)
 * Acc@1 62.01
GRL lambda: 0.9866
Train: [10][100/129]    BT 1.126 (1.163)        DT 0.023 (0.035)        Loss 2.004 (1.971)     Acc@1 87.500 (84.875)
Train epoch 10, total time 150.44, accuracy:84.74
 * S_p: 71.18, S_e: 51.83, Score: 61.51 (Best S_p: 74.16, S_e: 50.98, Score: 62.57)
 * F1 Score: 0.30 (F1 Score: 0.31)
 * Acc@1 62.92
GRL lambda: 0.9919
Train: [11][100/129]    BT 1.208 (1.160)        DT 0.028 (0.034)        Loss 1.921 (1.940)     Acc@1 87.500 (86.750)
Train epoch 11, total time 149.24, accuracy:86.31
 * S_p: 79.04, S_e: 47.41, Score: 63.22 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 65.53
Best ckpt is modified with Score = 63.22 when Epoch = 11
GRL lambda: 0.9951
Train: [12][100/129]    BT 1.111 (1.167)        DT 0.027 (0.036)        Loss 1.818 (1.906)     Acc@1 93.750 (87.312)
Train epoch 12, total time 149.88, accuracy:87.62
 * S_p: 66.88, S_e: 54.04, Score: 60.46 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 61.39
GRL lambda: 0.9970
Train: [13][100/129]    BT 1.051 (1.163)        DT 0.027 (0.036)        Loss 1.897 (1.889)     Acc@1 81.250 (89.062)
Train epoch 13, total time 148.86, accuracy:89.07
 * S_p: 81.25, S_e: 43.59, Score: 62.42 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 65.17
GRL lambda: 0.9982
Train: [14][100/129]    BT 1.117 (1.151)        DT 0.022 (0.035)        Loss 1.741 (1.879)     Acc@1 100.000 (88.875)
Train epoch 14, total time 148.62, accuracy:89.05
 * S_p: 71.25, S_e: 51.66, Score: 61.45 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 62.88
GRL lambda: 0.9989
Train: [15][100/129]    BT 1.169 (1.159)        DT 0.025 (0.034)        Loss 1.789 (1.873)     Acc@1 96.875 (90.000)
Train epoch 15, total time 149.62, accuracy:90.09
 * S_p: 73.40, S_e: 49.70, Score: 61.55 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 63.28
GRL lambda: 0.9993
Train: [16][100/129]    BT 0.937 (1.148)        DT 0.025 (0.035)        Loss 2.529 (1.843)     Acc@1 68.750 (90.750)
Train epoch 16, total time 148.41, accuracy:90.48
 * S_p: 71.06, S_e: 52.34, Score: 61.70 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 63.06
GRL lambda: 0.9996
Train: [17][100/129]    BT 1.078 (1.147)        DT 0.025 (0.036)        Loss 2.311 (1.850)     Acc@1 78.125 (91.000)
Train epoch 17, total time 148.01, accuracy:91.04
 * S_p: 76.69, S_e: 46.90, Score: 61.80 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 63.97
GRL lambda: 0.9998
Train: [18][100/129]    BT 1.173 (1.161)        DT 0.023 (0.036)        Loss 1.623 (1.842)     Acc@1 96.875 (91.312)
Train epoch 18, total time 149.14, accuracy:91.62
 * S_p: 72.51, S_e: 50.21, Score: 61.36 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 62.99
GRL lambda: 0.9999
Train: [19][100/129]    BT 1.153 (1.155)        DT 0.027 (0.034)        Loss 1.787 (1.862)     Acc@1 93.750 (90.906)
Train epoch 19, total time 148.95, accuracy:91.01
 * S_p: 74.79, S_e: 47.92, Score: 61.36 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 63.32
GRL lambda: 0.9999
Train: [20][100/129]    BT 1.175 (1.160)        DT 0.027 (0.035)        Loss 1.709 (1.826)     Acc@1 93.750 (91.688)
Train epoch 20, total time 149.79, accuracy:91.76
 * S_p: 75.24, S_e: 47.75, Score: 61.49 (Best S_p: 79.04, S_e: 47.41, Score: 63.22)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 63.50
best Score: [79.04, 47.41, 63.22, 0.3] (sp, se, sc)
results updated to ./save/results.json
Checkpoint icbhi_hftt_pafa_seed2_pcsl50_dann0.3_center0.0001_max20 finished
Running TAG=seed2_pcsl50_dann0.3_center0.0005_max20
SpecAugment using
********************
Train and test 60-40% split with test_fold official
File number in train dataset: 539
********************
Extracting individual breathing cycles..
[Preprocessed train dataset information]
total number of audio data: 4142
Class 0 (normal) : 2063 (49.8%)
Class 1 (crackle): 1215 (29.3%)
Class 2 (wheeze) : 501  (12.1%)
Class 3 (both)   : 363  (8.8%)
********************
Train and test 60-40% split with test_fold official
File number in test dataset: 381
********************
Extracting individual breathing cycles..
[Preprocessed test dataset information]
total number of audio data: 2756
Class 0 (normal) : 1579 (57.3%)
Class 1 (crackle): 649  (23.5%)
Class 2 (wheeze) : 385  (14.0%)
Class 3 (both)   : 143  (5.2%)
********************
Checkpoint Name: icbhi_hftt_pafa_seed2_pcsl50_dann0.3_center0.0005_max20
Training for 20 epochs on icbhi dataset
GRL lambda: 0.2449
Train: [1][100/129]     BT 1.172 (1.178)        DT 0.025 (0.035)        Loss 2.543 (3.070)     Acc@1 65.625 (55.344)
Train epoch 1, total time 151.24, accuracy:57.10
 * S_p: 61.11, S_e: 48.77, Score: 54.94 (Best S_p: 61.11, S_e: 48.77, Score: 54.94)
 * F1 Score: 0.29 (F1 Score: 0.29)
 * Acc@1 55.84
Best ckpt is modified with Score = 54.94 when Epoch = 1
GRL lambda: 0.4621
Train: [2][100/129]     BT 1.186 (1.160)        DT 0.023 (0.036)        Loss 2.589 (2.494)     Acc@1 56.250 (65.000)
Train epoch 2, total time 149.81, accuracy:65.55
 * S_p: 48.13, S_e: 57.77, Score: 52.95 (Best S_p: 61.11, S_e: 48.77, Score: 54.94)
 * F1 Score: 0.30 (F1 Score: 0.29)
 * Acc@1 52.25
GRL lambda: 0.6351
Train: [3][100/129]     BT 1.046 (1.157)        DT 0.025 (0.035)        Loss 2.529 (2.356)     Acc@1 62.500 (69.656)
Train epoch 3, total time 148.84, accuracy:69.48
 * S_p: 64.85, S_e: 52.51, Score: 58.68 (Best S_p: 64.85, S_e: 52.51, Score: 58.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 59.58
Best ckpt is modified with Score = 58.68 when Epoch = 3
GRL lambda: 0.7616
Train: [4][100/129]     BT 1.175 (1.157)        DT 0.027 (0.036)        Loss 2.038 (2.253)     Acc@1 84.375 (72.375)
Train epoch 4, total time 148.14, accuracy:72.63
 * S_p: 67.64, S_e: 51.32, Score: 59.48 (Best S_p: 67.64, S_e: 51.32, Score: 59.48)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 60.67
Best ckpt is modified with Score = 59.48 when Epoch = 4
GRL lambda: 0.8483
Train: [5][100/129]     BT 1.083 (1.157)        DT 0.022 (0.037)        Loss 2.216 (2.176)     Acc@1 84.375 (76.438)
Train epoch 5, total time 149.18, accuracy:75.82
 * S_p: 49.21, S_e: 57.86, Score: 53.53 (Best S_p: 67.64, S_e: 51.32, Score: 59.48)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 52.90
GRL lambda: 0.9051
Train: [6][100/129]     BT 1.173 (1.164)        DT 0.023 (0.035)        Loss 2.094 (2.117)     Acc@1 81.250 (78.156)
Train epoch 6, total time 150.57, accuracy:78.08
 * S_p: 61.43, S_e: 54.29, Score: 57.86 (Best S_p: 67.64, S_e: 51.32, Score: 59.48)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 58.38
GRL lambda: 0.9414
Train: [7][100/129]     BT 1.106 (1.177)        DT 0.022 (0.037)        Loss 2.164 (2.080)     Acc@1 81.250 (79.531)
Train epoch 7, total time 151.44, accuracy:79.84
 * S_p: 68.59, S_e: 51.57, Score: 60.08 (Best S_p: 68.59, S_e: 51.57, Score: 60.08)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 61.32
Best ckpt is modified with Score = 60.08 when Epoch = 7
GRL lambda: 0.9640
Train: [8][100/129]     BT 1.182 (1.183)        DT 0.027 (0.037)        Loss 1.999 (2.030)     Acc@1 84.375 (82.500)
Train epoch 8, total time 151.50, accuracy:81.86
 * S_p: 71.82, S_e: 51.23, Score: 61.52 (Best S_p: 71.82, S_e: 51.23, Score: 61.52)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 63.03
Best ckpt is modified with Score = 61.52 when Epoch = 8
GRL lambda: 0.9780
Train: [9][100/129]     BT 1.131 (1.164)        DT 0.024 (0.036)        Loss 1.961 (1.982)     Acc@1 81.250 (83.812)
Train epoch 9, total time 149.19, accuracy:83.62
 * S_p: 67.95, S_e: 51.23, Score: 59.59 (Best S_p: 71.82, S_e: 51.23, Score: 61.52)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 60.81
GRL lambda: 0.9866
Train: [10][100/129]    BT 1.118 (1.171)        DT 0.023 (0.036)        Loss 2.013 (1.965)     Acc@1 87.500 (85.344)
Train epoch 10, total time 150.57, accuracy:85.13
 * S_p: 70.11, S_e: 50.98, Score: 60.54 (Best S_p: 71.82, S_e: 51.23, Score: 61.52)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 61.94
GRL lambda: 0.9919
Train: [11][100/129]    BT 1.179 (1.169)        DT 0.027 (0.036)        Loss 1.882 (1.937)     Acc@1 87.500 (86.844)
Train epoch 11, total time 150.19, accuracy:86.22
 * S_p: 77.52, S_e: 47.83, Score: 62.68 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 64.84
Best ckpt is modified with Score = 62.68 when Epoch = 11
GRL lambda: 0.9951
Train: [12][100/129]    BT 1.116 (1.171)        DT 0.027 (0.037)        Loss 1.799 (1.899)     Acc@1 96.875 (87.938)
Train epoch 12, total time 150.60, accuracy:87.94
 * S_p: 64.72, S_e: 54.46, Score: 59.59 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 60.34
GRL lambda: 0.9970
Train: [13][100/129]    BT 1.062 (1.157)        DT 0.027 (0.036)        Loss 1.943 (1.886)     Acc@1 78.125 (89.062)
Train epoch 13, total time 148.83, accuracy:88.86
 * S_p: 80.11, S_e: 44.94, Score: 62.53 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 65.09
GRL lambda: 0.9982
Train: [14][100/129]    BT 1.114 (1.158)        DT 0.022 (0.036)        Loss 1.743 (1.876)     Acc@1 100.000 (89.000)
Train epoch 14, total time 150.00, accuracy:89.15
 * S_p: 69.47, S_e: 51.57, Score: 60.52 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 61.83
GRL lambda: 0.9989
Train: [15][100/129]    BT 1.156 (1.157)        DT 0.025 (0.036)        Loss 1.779 (1.865)     Acc@1 96.875 (89.844)
Train epoch 15, total time 149.45, accuracy:90.02
 * S_p: 72.58, S_e: 49.28, Score: 60.93 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 62.63
GRL lambda: 0.9993
Train: [16][100/129]    BT 0.964 (1.159)        DT 0.027 (0.035)        Loss 2.514 (1.838)     Acc@1 62.500 (90.594)
Train epoch 16, total time 149.62, accuracy:90.53
 * S_p: 70.04, S_e: 52.00, Score: 61.02 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 62.34
GRL lambda: 0.9996
Train: [17][100/129]    BT 1.079 (1.153)        DT 0.025 (0.034)        Loss 2.358 (1.843)     Acc@1 75.000 (91.094)
Train epoch 17, total time 148.45, accuracy:91.01
 * S_p: 74.92, S_e: 48.43, Score: 61.67 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 63.61
GRL lambda: 0.9998
Train: [18][100/129]    BT 1.167 (1.166)        DT 0.028 (0.036)        Loss 1.613 (1.840)     Acc@1 96.875 (91.031)
Train epoch 18, total time 149.57, accuracy:91.55
 * S_p: 71.37, S_e: 50.47, Score: 60.92 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 62.45
GRL lambda: 0.9999
Train: [19][100/129]    BT 1.149 (1.156)        DT 0.027 (0.035)        Loss 1.752 (1.856)     Acc@1 96.875 (91.031)
Train epoch 19, total time 149.41, accuracy:91.13
 * S_p: 73.46, S_e: 49.28, Score: 61.37 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 63.13
GRL lambda: 0.9999
Train: [20][100/129]    BT 1.180 (1.159)        DT 0.027 (0.035)        Loss 1.693 (1.821)     Acc@1 96.875 (91.844)
Train epoch 20, total time 149.83, accuracy:91.93
 * S_p: 73.72, S_e: 49.28, Score: 61.50 (Best S_p: 77.52, S_e: 47.83, Score: 62.68)
 * F1 Score: 0.30 (F1 Score: 0.30)
 * Acc@1 63.28
best Score: [77.52, 47.83, 62.68, 0.3] (sp, se, sc)
results updated to ./save/results.json
Checkpoint icbhi_hftt_pafa_seed2_pcsl50_dann0.3_center0.0005_max20 finished
MODEL="hftt"
SEED="2"
LAMBDA_PCSL_LIST="50"
LAMBDA_DANN_LIST="0.3"
LAMBDA_CENTER_LIST="0.0001 0.0005"
DANN_MAX_EPOCH_LIST="20"
SUMMARY="./save/tuning_summary.csv"

echo "tag,lambda_pcsl,lambda_dann,lambda_center,dann_max_epochs,score,sp,se,f1" > "$SUMMARY"

for s in $SEED; do
  for m in $MODEL; do
    for lambda_pcsl in $LAMBDA_PCSL_LIST; do
      for lambda_dann in $LAMBDA_DANN_LIST; do
        for lambda_center in $LAMBDA_CENTER_LIST; do
          for dann_max_epochs in $DANN_MAX_EPOCH_LIST; do
          TAG="seed${s}_pcsl${lambda_pcsl}_dann${lambda_dann}_center${lambda_center}_max${dann_max_epochs}"
          echo "Running TAG=${TAG}"

          CUDA_VISIBLE_DEVICES=0 python ./main.py --tag "$TAG" \
            --dataset icbhi \
            --seed $s \
            --class_split lungsound \
            --n_cls 4 \
            --epochs 20 \
            --batch_size 32 \
            --desired_length 5 \
            --optimizer adam \
            --learning_rate 3e-5 \
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
            --audioset_pretrained \
            --use_dann \
            --w_ce 1.0 \
            --w_pafa 1.0 \
            --lambda_pcsl $lambda_pcsl \
            --lambda_dann $lambda_dann \
            --lambda_center $lambda_center \
            --norm_type ln \
            --output_dim 768 \
            --nospec

          RESULT_DIR="./save/icbhi_hftt_pafa_${TAG}"
          if [ -f "${RESULT_DIR}/results.json" ]; then
            score=$(python - <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
print(data.get("Score", data.get("score", "NA")))
PY
"$RESULT_DIR/results.json")
          else
            score="NA"
          fi

          echo "${TAG},${lambda_pcsl},${lambda_dann},${lambda_center},${dann_max_epochs},${score},NA,NA,NA" >> "$SUMMARY"
          done
        done
      done
    done
  done
done