# Semantic-level multimodal molecular learning inspired by biological concept formation via soft matching

Fig. 1 | From instance-level to semantic-level alignment in multimodal molecular representation learning. 
a, Instance-level (left) and semantic-level (right) alignment paradigm. b, An example of human perception, where the simultaneous presence of bread, meat, and vegetables naturally leads to the recognition of a hamburger (left); an illustration of the proposed semantic-level soft matching mechanism for molecular understanding (right). c, Classification performance comparison across multiple benchmark datasets for representative molecular representation models.
![WPS图片(1)](https://github.com/user-attachments/assets/eec1ebf2-230a-4315-9b4d-dcf797eb9de6)
Fig. 2 | Conceptual framework of the semantic-level multimodal molecular representation learning of SemMol. 
a, DCL construction. Initial semantic centers for 2D and 3D modalities are constructed from 1D, 2D, and 3D representations via intra-batch clustering and refined using mini-batch K-means with Exponential Moving Average (EMA) updates in the DCL. b, ACSM mechanism. b1, Positive sample generation: retrieve the nearest semantic centers from 2D/3D center libraries relative to the anchor and construct positive samples using similarity-weighted fusion. b2, Negative sample generation and debiasing: filter out easily distinguishable negatives using a similarity threshold while retaining hard negatives to improve discriminative ability. c, Semantic alignment training by using the constructed positive and negative samples.
<img width="1493" height="1310" alt="图片1" src="https://github.com/user-attachments/assets/1152bc6a-7c20-4239-8d26-ef0139749594" />


##
The most important supplementary file is provided at the following link: https://github.com/3505675604/SemMol/blob/main/Supplementary_Materials/Supplementary%20Materials.pdf

## 🚀 Features
- Semantic-level learning: By introducing the anchor-center soft matching (ACSM) mechanism and the dynamic center library (DCL), cross-modal semantic alignment is achieved, avoiding the risk of overfitting caused by one-to-one instance matching. This method can simultaneously capture the global skeleton structure and local functional group characteristics of the molecule, improving the model's generalization ability and interpretability.
- **Multi-Modal Learning**: Integrates chemical structure, text, and other modalities for improved prediction.
- **Flexible Task Support**: Handles both classification and regression tasks with dynamic configuration.
- **Advanced Pseudo-Pair Generation**: Supports hard negative mining, adaptive temperature, and memory bank for contrastive learning.
- **Streaming & Incremental Clustering**: Online center library with streaming K-means and FAISS acceleration.
- **Configurable & Reproducible**: All settings managed via a single JSON config; supports experiment reproducibility.
- **Extensible Architecture**: Modular codebase for easy extension of models, data pipelines, and loss functions.
- **Pretrained Model Integration**: Easy download and usage of state-of-the-art pretrained models.

---


## ⚙️ Installation & Environment

1. Clone the repository
   ```bash
   git clone https://github.com/3505675604/SemMol.git
   ```
3. Create environment (recommended)
   ```bash
   conda env create -f environment.yml
   conda activate A
   ```
🏃‍♂️ Training, Fine-tuning, and Resources
Training :Run the following commands to start pre-training:
```bash
   chmod +x /data/FL/Semol/scripts/start_Pre_DPP.sh
   /data/FL/Semol/scripts/start_Pre_DPP.sh
```
Fine-tuning :Fine-tune the pretrained model with:
```bash
   python finetune.py
```
Pre-trained Models and Datasets
We provide the pre-trained SemMol model (trained on 1M molecules) and the datasets used for pre-training and downstream fine-tuning:
- Pre-trained Models (1M molecules): https://huggingface.co/Lin-Glory/SemMol_model
- Datasets (for pre-training and fine-tuning): https://huggingface.co/datasets/Lin-Glory/SemMol_datasets/tree/main/dataset

Note: Please download and extract all files before starting training or fine-tuning.

Outputs：All model checkpoints, logs, and training histories are saved in:
```bash
   Save_model/
```
---

## 📊 Datasets
-Input format: CSV files containing a smiles column and corresponding target column(s)
   -Supported benchmarks:
   -BBBP
   -ESOL
   -Lipophilicity
   -Tox21
   -Data split strategies:
   -Scaffold split (default, chemically-aware)
   -Random split
---

## 🧩 Configuration

All experiment, model, and data settings are managed in `config/config.json`:

- **Data**: File paths, target columns, split type, normalization
- **Model**: Architecture, fusion strategy, dropout, pretrained paths
- **Training**: Epochs, batch size, learning rate, scheduler, seed
- **Loss**: Loss function, pseudo-pair and alignment weights
- **Pseudo-Pair**: Hard negative mining, memory bank, temperature
- **Early Stopping**: Patience, monitored metric

See in-file comments and descriptions for all options.

---


📈 Reproducibility
-Fixed random seeds for all experiments
-Centralized configuration management
-Modular and extensible codebase for easy customization

📚 Citation
If you use SemMol in your research, please cite the original authors:
```bash
@article{semmol,
  title={SemMol:Semantic-level multimodal molecular learning inspired by biological concept formation via soft matching},
  author={Anonymous},
  journal={arXiv preprint},
  year={202X}
}
```








