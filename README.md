# ICBHI 2017 Lung Sound Classification

This repository contains the implementation for lung sound classification using the ICBHI 2017 dataset. The project supports multiple models including HFTT (Hybrid Frequency-Temporal Transformer) and BEATs, with advanced training methods such as PCSL (Patient Contrastive Self-supervised Learning) and DANN (Domain Adversarial Neural Network).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
  
## Requirements

### System Requirements

- **Python**: 3.7 or higher
- **CUDA**: 11.0 or higher (for GPU training)
- **RAM**: At least 16GB recommended
- **Storage**: At least 50GB free space for dataset and checkpoints

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/J-saderr/ICBHI_2017.git
cd ICBHI_2017
```

### Step 2: Create Virtual Environment 

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

### Step 2: Verify Dataset

Ensure that:
- All `.wav` files have corresponding `.txt` annotation files
- The `official_split.txt` file contains train/test split information
- The `metadata.txt` file contains patient metadata

### Step 3: Download Pretrained Models

```bash
# The pretrained model should be placed in:
pretrained_models/BEATs_iter3_plus_AS2M.pt
``````

## Training Process

### Overview

The training process involves:
1. **Data Loading**: Loading and preprocessing ICBHI audio files
2. **Model Initialization**: Setting up the backbone model (HFTT)
3. **Training Loop**: Iterative training with loss optimization
4. **Validation**: Periodic evaluation on validation set
5. **Checkpointing**: Saving best models based on validation metrics

### Step 1: Basic Training Command

The simplest way to start training is using the provided script:

```bash
bash scripts/hftt_pcsl_dann.sh
```

### Step 2: Understanding Training Stages

#### Stage 1: Data Preprocessing
- Audio files are loaded and converted to spectrograms or processed by the model's internal preprocessing
- Individual breathing cycles are extracted from recordings

#### Stage 2: Model Setup
- Backbone encoder (HFTT) is initialized
- Classification head is added
- Projection head is added (for PCSL + DANNLighter method)
- Loss functions are configured

#### Stage 3: Training Loop
For each epoch:
1. **Forward Pass**: 
   - Audio samples → Model features → Classifier output
   - Features → Projection head (for PCSL + DANNLighter method)
   
2. **Loss Calculation**:
   - Classification loss (CrossEntropy)
   - PCSL loss (patient contrastive learning)
   - DANN loss (domain adversarial, if enabled)
   - Combined loss: `w_ce * L_ce + w_projectors * ( L_pcsl`+ L_dannlighter

3. **Backward Pass**:
   - Gradient computation
   - Optimizer step
   - Moving average update (if enabled)

4. **Validation**:
   - Evaluate on validation set
   - Calculate metrics: Specificity (Sp), Sensitivity (Se), Score (Sc), F1
   - Save best model if metrics improve

**Metrics Explanation**:
- **S_p (Specificity)**: True negative rate
- **S_e (Sensitivity)**: True positive rate  
- **Score**: Average of S_p and S_e
- **F1 Score**: Harmonic mean of precision and recall

### Step 4: Training Outputs

Training generates the following outputs in `save/…`:

- `best.pth`: Best model checkpoint based on validation score
- `best_epoch_<N>.pth`: Checkpoint saved when best score is updated
- `epoch_<N>.pth`: Periodic checkpoints (every `save_freq` epochs)
- `train_args.json`: Training configuration
- `patient_evaluation.csv`: Patient-level evaluation metrics (if generated)

## Evaluation
Use the evaluation script:

```bash
bash scripts/hftt_pcsl_dann_eval.sh
```
<img width="468" height="645" alt="image" src="https://github.com/user-attachments/assets/757ce740-c1f7-43a6-83bf-af90a86f19d0" />
