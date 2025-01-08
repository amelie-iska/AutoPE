# AutoProteinEngine (AutoPE)

AutoProteinEngine (AutoPE) is an innovative framework that leverages large language models (LLMs) for multimodal automated machine learning (AutoML) in protein engineering tasks. It enables biologists without deep learning backgrounds to interact with deep learning models using natural language, significantly lowering the entry barrier for protein engineering tasks. Our paper is at https://arxiv.org/pdf/2411.04440.

## Overview

AutoPE uniquely integrates:
- LLM-driven model selection for both protein sequence and graph modalities
- Automatic hyperparameter optimization
- Automated data retrieval from protein databases
- Natural language interface for non-experts

## Installation

```bash
# Clone the repository
git clone https://github.com/tsynbio/AutoPE.git
cd AutoPE

## Key Dependencies

- PyTorch
- ESM
- Ray/Ray Tune
- Pandas
- NumPy
- Scikit-learn

## Usage

### 1. Classification Tasks

1. Upload a file;
2. Enter similar content: 'Execute classification task, read uploaded file, analyze sequence AAAAAA/pdb with id 1asc, column A is data, column B is label, set lr to 0.01, dropout to 0.2'

### 2. Regression Tasks

1. Upload a file;
2. Enter similar content: 'Execute regression task, read uploaded file, analyze sequence AAAAAA/pdb with id 1asc, column A is data, column B is label, set lr to 0.01, dropout to 0.2'


## Model Architecture

The framework includes:

1. **Base Model Selection**
   - ESM2 series models for different complexities
   - Automatic model selection based on task requirements

2. **Custom Layers**
   - MaskedAveragePooling for sequence handling
   - Self-attention mechanisms for enhanced feature extraction

3. **Training Components**
   - Gradient accumulation
   - Mixed precision training
   - Advanced learning rate scheduling

## Experimental Results

### Brazzein Protein Sweetness Classification

| Method | F1-score | SRCC | Accuracy |
|--------|----------|------|-----------|
| Zero-Shot | 0.4764 | 0.3769 | 0.6917 |
| Manual Fine-Tuning | 0.5709 | 0.3098 | 0.9137 |
| AutoPE (w/o HPO) | 0.6396 | 0.4405 | 0.7988 |
| AutoPE (w/ HPO) | 0.7306 | 0.4621 | 0.8908 |

### STM1221 Enzyme Activity Regression

| Method | RMSE | MAE | R²_score |
|--------|------|-----|-----------|
| Zero-Shot | 0.4862 | 0.2766 | 0.5663 |
| Manual Fine-Tuning | 0.3579 | 0.2236 | 0.5965 |
| AutoPE (w/o HPO) | 0.4029 | 0.2164 | 0.6153 |
| AutoPE (w/ HPO) | 0.3488 | 0.1999 | 0.6805 |

## Features

1. **Natural Language Interface**
   - Task specification through conversation
   - Automated model selection and configuration
   - Interactive feedback and result interpretation

2. **Automated Model Selection**
   - Task complexity analysis
   - Resource-aware model recommendations
   - Multi-modality support

3. **Hyperparameter Optimization**
   - Ray Tune integration
   - ASHA scheduler for efficient searching
   - Resource-aware trial allocation

4. **Data Management**
   - Automated data retrieval from PDB/UniProt
   - Intelligent data preprocessing
   - Multi-modal data handling

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Citation

If you use AutoPE in your research, please cite:

```bibtex
@article{autope2024,
  title={AutoProteinEngine: A Large Language Model Driven Agent Framework for Multimodal AutoML in Protein Engineering},
  author={Liu, Yungeng and Chen, Zan and Zhao, Xinqing and Wang, Yu Guang and Shen, Yiqing},
  year={2024}
}
```
