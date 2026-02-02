# ICBHI 2017 Lung Sound Classification

This repository contains the implementation for lung sound classification using the ICBHI 2017 dataset. The project supports multiple models including HFTT (Hybrid Frequency-Temporal Transformer) and BEATs, with advanced training methods such as PCSL (Patient Contrastive Self-supervised Learning) and DANN (Domain Adversarial Neural Network).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Important Parameters](#important-parameters)
- [Troubleshooting](#troubleshooting)

## Requirements

### System Requirements

- **Python**: 3.7 or higher
- **CUDA**: 11.0 or higher (for GPU training)
- **RAM**: At least 16GB recommended
- **Storage**: At least 50GB free space for dataset and checkpoints

### Hardware Recommendations

- **GPU**: NVIDIA GPU with at least 8GB VRAM (e.g., RTX 3080, V100, A100)
- **CPU**: Multi-core processor (8+ cores recommended for data loading)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/J-saderr/ICBHI_2017.git
cd ICBHI_2017
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Step 1: Download ICBHI 2017 Dataset

Extract the dataset in drive to the `data/` directory

### Step 2: Organize Dataset Structure

The expected directory structure should be:

```
data/
└── icbhi_dataset/
    ├── audio_test_data/
    │   ├── *.wav (audio files)
    │   └── *.txt (annotation files)
    ├── official_split.txt
    ├── patient_list_foldwise.txt
    ├── patient_diagnosis.txt
    └── metadata.txt
```

### Step 3: Verify Dataset

Ensure that:
- All `.wav` files have corresponding `.txt` annotation files
- The `official_split.txt` file contains train/test split information
- The `metadata.txt` file contains patient metadata

### Step 4: Download Pretrained Models

```bash
# The pretrained model should be placed in:
pretrained_models/BEATs_iter3_plus_AS2M.pt
```

## Project Structure

```
ICBHI_2017/
├── BEATs/                    # BEATs model implementation
│   ├── BEATs.py
│   ├── backbone.py
│   ├── modules.py
│   └── quantizer.py
├── data/                      # Dataset directory
│   └── icbhi_dataset/
├── method/                    # Training methods
│   └── dann.py               # DANN and PCSL implementation
├── models/                    # Model definitions
│   ├── __init__.py
│   └── hftt.py               # HFTT model
├── pretrained_models/        # Pretrained model checkpoints
├── scripts/                   # Training and evaluation scripts
│   ├── hftt_pcsl_dann.sh
│   └── hftt_pcsl_dann_eval.sh
├── util/                      # Utility functions
│   ├── icbhi_dataset.py       # Dataset loader
│   ├── icbhi_util.py         # ICBHI utilities
│   ├── augmentation.py       # Data augmentation
│   └── misc.py               # Miscellaneous utilities
├── main.py                    # Main training script
├── test.py                    # Evaluation script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Training Process

### Overview

The training process involves:
1. **Data Loading**: Loading and preprocessing ICBHI audio files
2. **Model Initialization**: Setting up the backbone model (HFTT or BEATs)
3. **Training Loop**: Iterative training with loss optimization
4. **Validation**: Periodic evaluation on validation set
5. **Checkpointing**: Saving best models based on validation metrics

### Step 1: Basic Training Command

The simplest way to start training is using the provided script:

```bash
bash scripts/hftt_pcsl_dann.sh
```

Or run directly with Python:

```bash
python main.py \
    --dataset icbhi \
    --model hftt \
    --class_split lungsound \
    --n_cls 4 \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 7e-5 \
    --method projectors_loss \
    --seed 2
```

### Step 2: Understanding Training Stages

#### Stage 1: Data Preprocessing
- Audio files are loaded and converted to spectrograms or processed by the model's internal preprocessing
- Individual breathing cycles are extracted from recordings
- Data augmentation is applied (if enabled)

#### Stage 2: Model Setup
- Backbone encoder (HFTT or BEATs) is initialized
- Classification head is added
- Projection head is added (for PCSL method)
- Loss functions are configured

#### Stage 3: Training Loop
For each epoch:
1. **Forward Pass**: 
   - Audio samples → Model features → Classifier output
   - Features → Projection head (for PCSL)
   
2. **Loss Calculation**:
   - Classification loss (CrossEntropy)
   - PCSL loss (patient contrastive learning)
   - DANN loss (domain adversarial, if enabled)
   - Combined loss: `w_ce * L_ce + w_projectors * L_pcsl`

3. **Backward Pass**:
   - Gradient computation
   - Optimizer step
   - Moving average update (if enabled)

4. **Validation**:
   - Evaluate on validation set
   - Calculate metrics: Specificity (Sp), Sensitivity (Se), Score (Sc), F1
   - Save best model if metrics improve

### Step 3: Training with Custom Configuration

#### Example 1: Training with BEATs Model

```bash
python main.py \
    --dataset icbhi \
    --model beats \
    --class_split lungsound \
    --n_cls 4 \
    --epochs 400 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --method ce \
    --pretrained \
    --pretrained_ckpt pretrained_models/BEATs_iter3_plus_AS2M.pt \
    --seed 0
```

#### Example 2: Training with PCSL + DANN

```bash
python main.py \
    --dataset icbhi \
    --model hftt \
    --class_split lungsound \
    --n_cls 4 \
    --epochs 20 \
    --batch_size 32 \
    --desired_length 5 \
    --optimizer adam \
    --learning_rate 7e-5 \
    --weight_decay 1e-6 \
    --cosine \
    --method projectors_loss \
    --w_ce 1.0 \
    --w_projectors 1.0 \
    --lambda_pcsl 22.0 \
    --lambda_dann 0.58 \
    --norm_type ln \
    --output_dim 768 \
    --ma_update \
    --ma_beta 0.5 \
    --from_sl_official \
    --nospec \
    --seed 2 \
    --tag seed2
```

#### Example 3: Resume Training from Checkpoint

```bash
python main.py \
    --resume save/icbhi_hftt_projectors_loss_seed2/best_epoch_15.pth \
    --dataset icbhi \
    --model hftt \
    --epochs 20 \
    --batch_size 32 \
    --seed 2
```

### Step 4: Monitoring Training

During training, you will see output like:

```
Train: [1][100/500]    BT 0.123 (0.120)    DT 0.045 (0.050)    Loss 1.234 (1.250)    Acc@1 45.67 (44.23)
Test: [100/200]    Time 0.089 (0.092)    Loss 1.123 (1.145)    Acc@1 52.34 (51.23)    S_p 0.456    S_e 0.567    Score 0.512    F1 Score 0.489
 * S_p: 45.60, S_e: 56.70, Score: 51.20 (Best S_p: 45.60, S_e: 56.70, Score: 51.20)
```

**Metrics Explanation**:
- **S_p (Specificity)**: True negative rate
- **S_e (Sensitivity)**: True positive rate  
- **Score**: Average of S_p and S_e
- **F1 Score**: Harmonic mean of precision and recall

### Step 5: Training Outputs

Training generates the following outputs in `save/<model_name>/`:

- `best.pth`: Best model checkpoint based on validation score
- `best_epoch_<N>.pth`: Checkpoint saved when best score is updated
- `epoch_<N>.pth`: Periodic checkpoints (every `save_freq` epochs)
- `train_args.json`: Training configuration
- `patient_evaluation.csv`: Patient-level evaluation metrics (if generated)

## Evaluation

### Step 1: Evaluate Trained Model

Use the evaluation script:

```bash
bash scripts/hftt_pcsl_dann_eval.sh
```

Or run directly:

```bash
python test.py \
    --tag seed2_hftt \
    --dataset icbhi \
    --model hftt \
    --class_split lungsound \
    --n_cls 4 \
    --test_fold official \
    --method projectors_loss \
    --norm_type ln \
    --output_dim 768 \
    --from_sl_official \
    --nospec
```

### Step 2: Evaluation with Specific Checkpoint

```bash
python test.py \
    --resume save/icbhi_hftt_projectors_loss_seed2/best.pth \
    --dataset icbhi \
    --model hftt \
    --eval \
    --class_split lungsound \
    --n_cls 4
```

### Step 3: Two-Class Evaluation

For binary classification (normal vs abnormal):

```bash
python test.py \
    --resume save/icbhi_hftt_projectors_loss_seed2/best.pth \
    --dataset icbhi \
    --model hftt \
    --class_split lungsound \
    --n_cls 2 \
    --two_cls_eval \
    --eval
```

## Important Parameters

### Dataset Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--dataset` | Dataset name | `icbhi` | `icbhi` |
| `--data_folder` | Path to dataset | `./data` | Any valid path |
| `--class_split` | Classification type | `lungsound` | `lungsound`, `diagnosis` |
| `--n_cls` | Number of classes | `0` | `2`, `4` (for lungsound) |
| `--test_fold` | Test fold selection | `official` | `official`, `0-4` |
| `--sample_rate` | Audio sampling rate | `16000` | Typically `16000` |
| `--desired_length` | Cycle length in seconds | `8` | `5`, `8`, etc. |
| `--n_mels` | Mel filter banks | `128` | `64`, `128`, `256` |
| `--pad_types` | Padding strategy | `repeat` | `zero`, `repeat`, `aug` |

### Model Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model` | Backbone model | `beats` | `beats`, `hftt` |
| `--pretrained` | Use pretrained weights | `False` | Flag |
| `--pretrained_ckpt` | Pretrained checkpoint path | `None` | File path |
| `--from_sl_official` | Load official PyTorch weights | `False` | Flag |
| `--nospec` | Disable spectrogram preprocessing | `False` | Flag |

### Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--epochs` | Number of training epochs | `400` | Any positive integer |
| `--batch_size` | Batch size | `128` | Power of 2 recommended |
| `--learning_rate` | Initial learning rate | `1e-3` | Typically `1e-5` to `1e-2` |
| `--optimizer` | Optimizer type | `adam` | `adam`, `sgd` |
| `--weight_decay` | Weight decay (L2) | `1e-4` | `1e-6` to `1e-3` |
| `--cosine` | Use cosine annealing | `False` | Flag |
| `--warm` | Use warmup | `False` | Flag |
| `--warm_epochs` | Warmup epochs | `0` | Positive integer |

### PCSL + DANN Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--method` | Training method | `ce` | `ce`, `projectors_loss` |
| `--w_ce` | Classification loss weight | `1.0` | Positive float |
| `--w_projectors` | Projector loss weight | `0.5` | Positive float |
| `--lambda_pcsl` | PCSL loss coefficient | `0.1` | Positive float |
| `--lambda_dann` | DANN loss coefficient | `0.1` | Positive float |
| `--norm_type` | Normalization type | `bn` | `bn`, `ln` |
| `--output_dim` | Projection output dimension | `128` | Positive integer |
| `--hidden_dim` | Projection hidden dimension | `None` | Positive integer or `None` |

### Moving Average Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--ma_update` | Enable moving average | `False` | Flag |
| `--ma_beta` | Moving average coefficient | `0` | `0.0` to `1.0` |

### Other Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--seed` | Random seed | `0` | Any integer |
| `--num_workers` | DataLoader workers | `8` | Positive integer |
| `--print_freq` | Print frequency | `100` | Positive integer |
| `--save_freq` | Save checkpoint frequency | `100` | Positive integer |
| `--save_dir` | Save directory | `./save` | Any valid path |
| `--tag` | Experiment tag | `''` | Any string |

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Reduce number of workers: `--num_workers 4`
- Use gradient accumulation (modify code)
- Use mixed precision training (already enabled by default)

#### 2. Dataset Loading Errors

**Error**: `FileNotFoundError` or dataset-related errors

**Solutions**:
- Verify dataset structure matches expected format
- Check file paths in `data/icbhi_dataset/`
- Ensure all `.wav` files have corresponding `.txt` files
- Check `official_split.txt` format

#### 3. Import Errors

**Error**: `ModuleNotFoundError`

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Ensure you're in the correct directory
cd ICBHI_2017

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 4. Torchvision Temporary Directory Issue

**Error**: Issues with torchvision transforms

**Solution**:
```bash
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"
```

Or add to your script:
```bash
#!/bin/bash
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"
# ... rest of script
```

#### 5. Slow Training

**Solutions**:
- Increase `--num_workers` (but not more than CPU cores)
- Use `--pin_memory True` (already enabled in code)
- Ensure data is on fast storage (SSD)
- Use smaller `--desired_length` to reduce processing time
- Enable `--nospec` if using models with internal preprocessing

#### 6. Poor Validation Performance

**Solutions**:
- Adjust learning rate (try `1e-4` to `1e-5`)
- Increase training epochs
- Tune loss weights (`--w_ce`, `--w_projectors`)
- Adjust `--lambda_pcsl` and `--lambda_dann`
- Try different `--ma_beta` values (0.5-0.99)
- Use data augmentation (ensure `--raw_augment` is set)

#### 7. Checkpoint Loading Errors

**Error**: `KeyError` or state dict mismatch

**Solutions**:
- Ensure model architecture matches checkpoint
- Check `--model` parameter matches training
- Verify `--method` and other parameters match
- Use `--strict False` in code (if modifying)

### Performance Optimization Tips

1. **Data Loading**:
   - Use appropriate `--num_workers` (typically 4-8)
   - Enable caching (dataset creates `.pt` cache files)
   - Use SSD for dataset storage

2. **Training Speed**:
   - Use mixed precision (automatic in code)
   - Reduce `--desired_length` if possible
   - Use `--nospec` for models with internal preprocessing
   - Batch size: larger is better (if memory allows)

3. **Memory Optimization**:
   - Reduce batch size
   - Use gradient checkpointing (requires code modification)
   - Clear cache: `torch.cuda.empty_cache()`

## Additional Resources

### Model Architecture

- **HFTT**: Hybrid Frequency-Temporal Transformer with multi-scale attention
- **BEATs**: Bidirectional Encoder representation from Audio Transformers
- **PCSL**: Patient Contrastive Self-supervised Learning for domain generalization
- **DANN**: Domain Adversarial Neural Network for device-invariant features

### Citation

If you use this code, please cite the relevant papers:
- ICBHI 2017 Challenge dataset
- BEATs paper
- HFTT paper (if applicable)
- PCSL/DANN methodology

### License

[Specify your license here]

### Contact

[Add contact information if needed]

---

**Last Updated**: [Current Date]
**Version**: 1.0

