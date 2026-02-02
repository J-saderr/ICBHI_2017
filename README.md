# ICBHI 2017 Lung Sound Classification

This repository contains the implementation for lung sound classification using the ICBHI 2017 Challenge dataset. The project employs advanced deep learning models (BEATs, HFTT) combined with domain adaptation techniques (PAFA, DANN) to improve classification performance across different patients and recording devices.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Pretrained Models](#pretrained-models)
- [Model Architectures](#model-architectures)
- [Training Methods](#training-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
- [Citation](#citation)

## Overview

This project addresses the challenge of lung sound classification in the presence of domain shift caused by:
- **Patient variability**: Different patients have different lung characteristics
- **Device variability**: Recordings from different devices (Littmann, AKG, Meditron, 3M)
- **Recording conditions**: Variations in recording environments

The solution combines:
- **Self-supervised learning models** (BEATs, HFTT) for robust feature extraction
- **Patient-Aware Feature Alignment (PAFA)** for handling patient-specific variations
- **Domain Adversarial Neural Networks (DANN)** for device domain adaptation
- **Center Loss** for intra-class compactness

## Features

- **Multiple Model Architectures**: BEATs and HFTT (Hierarchical Feature Transformer)
- **Domain Adaptation Methods**: 
  - PAFA (Patient-Aware Feature Alignment) with PCSL and GPAL losses
  - DANN (Domain Adversarial Neural Network) with Gradient Reversal Layer
  - Center Loss for feature learning
- **Flexible Training**: Support for various training configurations and hyperparameters
- **Comprehensive Evaluation**: Patient-level and cycle-level evaluation metrics
- **ICBHI 2017 Dataset**: Official 60-40 split and custom fold-wise splits

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.8+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/J-saderr/ICBHI_2017.git
cd ICBHI_2017
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scipy
pip install librosa soundfile
pip install scikit-learn matplotlib seaborn
```

## Dataset Setup

### ICBHI 2017 Challenge Dataset

1. Download the ICBHI 2017 Challenge dataset from the [official website](https://bhichallenge.med.auth.gr/)

2. Organize the dataset structure:
```
data/
├── icbhi_dataset/
│   ├── audio_test_data/          # Audio files
│   ├── official_split.txt        # Official 60-40 split
│   ├── patient_list_foldwise.txt # Fold-wise patient splits
│   ├── metadata.txt              # Dataset metadata
│   └── patient_diagnosis.txt     # Patient diagnosis information
```

3. Place the audio files in `data/icbhi_dataset/audio_test_data/`

### Dataset Information

- **Total Patients**: 126
- **Total Audio Files**: 920
- **Classes**: 
  - Normal
  - Crackle
  - Wheeze
  - Both (crackle + wheeze)
- **Recording Devices**: Littmann (L), AKG (A), Meditron (M), 3M (3)
- **Sample Rate**: 16 kHz (configurable)
- **Cycle Length**: 5-8 seconds (configurable)

## Usage

### Pretrained Models

- `BEATs_iter3_plus_AS2M.pt`: BEATs pretrained on AudioSet. Downloaded on "https://github.com/microsoft/unilm/tree/master/beats"

### Basic Training

Train a model with default settings:

```bash
python main.py \
    --dataset icbhi \
    --model beats \
    --method pafa \
    --class_split lungsound \
    --n_cls 4 \
    --epochs 400 \
    --batch_size 128 \
    --learning_rate 1e-3
```

### Training with DANN

Train with Domain Adversarial Neural Network:

```bash
python main.py \
    --dataset icbhi \
    --model hftt \
    --method pafa \
    --use_dann \
    --lambda_dann 0.3 \
    --lambda_pcsl 50.0 \
    --class_split lungsound \
    --n_cls 4 \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --cosine \
    --ma_update \
    --ma_beta 0.5
```

### Evaluation Only

Evaluate a pretrained model:

```bash
python main.py \
    --eval \
    --resume ./save/icbhi_hftt_pafa_seed2_dann/best.pth \
    --dataset icbhi \
    --model hftt \
    --method pafa \
    --class_split lungsound \
    --n_cls 4
```

### Using Training Scripts

The repository includes several training scripts in the `scripts/` directory:

```bash
# Train HFTT with PAFA and DANN
bash scripts/hftt_pafa_dann.sh

# Train with different configurations
bash scripts/hftt_enhanced.sh
bash scripts/hftt_optimized.sh
```

## Model Architectures

### BEATs (Bidirectional Encoder representation from Audio Transformers)

- Self-supervised audio representation learning
- Pretrained on large-scale audio datasets
- Supports fine-tuning for lung sound classification

### HFTT (Hierarchical Feature Transformer)

- Hierarchical feature extraction for audio
- Transformer-based architecture
- Supports both supervised and self-supervised learning

## Training Methods

### 1. Cross-Entropy (CE)

Standard supervised learning with cross-entropy loss:

```bash
--method ce
```

### 2. PAFA (Patient-Aware Feature Alignment)

PAFA addresses patient-specific variations using:
- **PCSL (Patient-Centered Similarity Loss)**: Encourages similar features for the same patient
- **GPAL (Global Patient Alignment Loss)**: Aligns features across different patients

```bash
--method pafa \
--lambda_pcsl 0.1 \
--lambda_gpal 0.1 \
--w_ce 1.0 \
--w_pafa 0.5
```

### 3. PAFA with DANN

Combines PAFA with Domain Adversarial Neural Network for device domain adaptation:

```bash
--method pafa \
--use_dann \
--lambda_dann 0.3 \
--lambda_pcsl 50.0
```

## Evaluation Metrics

The project uses ICBHI 2017 Challenge evaluation metrics:

- **Specificity (S_p)**: True negative rate
- **Sensitivity (S_e)**: True positive rate  
- **Score**: Average of S_p and S_e: `Score = (S_p + S_e) / 2`
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy

### Patient-Level Evaluation

Generate patient-level evaluation metrics:

```python
# Uncomment in main.py evaluation section
evaluate_patient_level(val_loader, model, classifier, projector, args)
```

This generates a CSV file with per-patient metrics.

## Project Structure

```
ICBHI_2017/
├── BEATs/                 # BEATs model implementation
│   ├── backbone.py
│   ├── BEATs.py
│   ├── modules.py
│   └── Tokenizers.py
├── data/                  # Dataset files and metadata
│   ├── icbhi_dataset/
│   ├── official_split.txt
│   └── patient_list_foldwise.txt
├── method/                # Training methods
│   ├── pafa.py           # PAFA implementation
│   ├── dann.py           # DANN implementation
│   ├── center_loss.py    # Center loss
│   ├── analysis.py       # Analysis utilities
│   └── visualization.py  # Visualization tools
├── model/                 # Model architectures
│   ├── beats.py
│   └── hftt.py
├── scripts/               # Training scripts
│   ├── hftt_pafa_dann.sh
│   ├── hftt_enhanced.sh
│   └── ...
├── util/                  # Utilities
│   ├── icbhi_dataset.py  # Dataset loader
│   ├── icbhi_util.py     # ICBHI utilities
│   └── misc.py           # Miscellaneous utilities
├── pretrained_models/     # Pretrained model checkpoints
├── save/                  # Training outputs and checkpoints
└── main.py               # Main training script
```

## Scripts

### Available Training Scripts

- `hftt_pafa_dann.sh`: Train HFTT with PAFA and DANN
- `hftt_pafa.sh`: Train HFTT with PAFA only
- `hftt_enhanced.sh`: Enhanced HFTT training configuration
- `hftt_optimized.sh`: Optimized HFTT training
- `hftt_pafa_dann_eval.sh`: Evaluation script for DANN models


## Key Hyperparameters

### Model Selection
- `--model`: `beats` or `hftt`
- `--pretrained`: Use pretrained weights
- `--pretrained_ckpt`: Path to pretrained checkpoint

### Training Configuration
- `--epochs`: Number of training epochs (default: 400)
- `--batch_size`: Batch size (default: 128)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--optimizer`: `adam` or `sgd`
- `--cosine`: Use cosine annealing scheduler
- `--warm`: Enable warm-up training

### PAFA Parameters
- `--lambda_pcsl`: Weight for Patient-Centered Similarity Loss (default: 0.1)
- `--lambda_gpal`: Weight for Global Patient Alignment Loss (default: 0.1)
- `--w_ce`: Weight for classification loss (default: 1.0)
- `--w_pafa`: Weight for PAFA loss (default: 0.5)
- `--output_dim`: Projection output dimension (default: 128)

### DANN Parameters
- `--use_dann`: Enable DANN
- `--lambda_dann`: Weight for DANN loss (default: 0.1)
- `--lambda_center`: Weight for center loss (default: 0.0)

### Dataset Configuration
- `--class_split`: `lungsound` or `diagnosis`
- `--n_cls`: Number of classes (2 or 4 for lungsound)
- `--test_fold`: `official` or `0-4` for fold-wise splits
- `--desired_length`: Cycle length in seconds (default: 8)
- `--sample_rate`: Audio sample rate (default: 16000)
- `--n_mels`: Number of mel filter banks (default: 128)

## Pretrained Models

Pretrained models are stored in `pretrained_models/` directory. Due to their large size (>2GB), they are not included in the repository. 

**Note**: Pretrained model files are excluded from Git via `.gitignore`. To use pretrained models:

1. Download pretrained checkpoints separately
2. Place them in `pretrained_models/` directory
3. Specify the path using `--pretrained_ckpt` argument

## Results

Final training results with HFTT + PSCL + DANN:

``` 
Best Score: [82.58, 46.9, 64.74] (S_p, S_e, Score)
F1 Score: 0.31
Accuracy: 67.34%
```

Results are saved in:
- `save/{model_name}/best.pth`: Best model checkpoint
- `save/{model_name}/results.json`: Evaluation results
- `save/results.json`: Aggregated results

## Analysis Tools

The project includes analysis and visualization tools:

- **Patient Centroids**: Analyze patient-level feature representations
- **Visualization**: Visualize training/test distributions
- **Patient Evaluation**: Generate per-patient performance metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{icbhi2017,
  title={ICBHI 2017 Lung Sound Classification},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/J-saderr/ICBHI_2017}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- ICBHI 2017 Challenge organizers for providing the dataset
- HFTT model developers
- Contributors to the open-source audio processing libraries

---

**Note**: This is a research project. Results may vary depending on hardware, dataset preprocessing, and hyperparameter settings.

