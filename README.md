我来帮您编写一个完整的README文档：

```markdown
# AutoProteinEngine (AutoPE)

AutoProteinEngine (AutoPE) is an innovative framework that leverages large language models (LLMs) for multimodal automated machine learning (AutoML) in protein engineering tasks. It enables biologists without deep learning backgrounds to interact with deep learning models using natural language, significantly lowering the entry barrier for protein engineering tasks.

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

# Install dependencies
pip install -r requirements.txt
```

## Key Dependencies

- PyTorch
- ESM
- Ray/Ray Tune
- Pandas
- NumPy
- Scikit-learn

## Usage

### 1. Classification Tasks

```python
from classification import main_sweet

# Configure your task
config = {
    'save_path': './results', 
    'cpu_per_trial': '4',
    'gpus_per_trial': '1',
    'num_samples': 20,
    'lr': tune.loguniform(1e-6, 1e-3),
    'dropout': tune.uniform(0.001, 0.3),
    'num_epochs': 30,
    'batch_size': tune.choice([2,4]),
    'accumulation_steps': 4
}

# Run classification
main_sweet()
```

### 2. Regression Tasks

```python
from regression import main

# Run regression with custom parameters
main(num_samples=5, max_num_epochs=10, gpus_per_trial=0.3)
```

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

这个README包含了项目的主要组成部分、使用方法、实验结果等关键信息。您觉得需要补充或修改哪些内容吗？
