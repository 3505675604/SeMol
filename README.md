<img width="1493" height="1310" alt="图片1" src="https://github.com/user-attachments/assets/8df2ce67-0b8f-471e-8914-c4e604c92ac6" />
# Semantic-level multimodal molecular learning inspired by biological concept formation via soft matching.

Figure 1：This figure illustrates the transition from instance-level to semantic-level alignment in multimodal molecular representation learning, where semantic-level alignment captures higher-level cross-modal semantics beyond one-to-one matching. Inspired by human holistic perception, a semantic-level soft matching mechanism is introduced, resulting in improved classification performance across multiple molecular benchmarks.
![WPS图片(1)](https://github.com/user-attachments/assets/eec1ebf2-230a-4315-9b4d-dcf797eb9de6)
Figure 2：This figure illustrates the semantic-level multimodal molecular representation learning framework of SemMol. It constructs and updates 2D and 3D semantic centers via intra-batch clustering and EMA-based mini-batch K-means, then employs the ACSM mechanism to generate similarity-weighted positives and informative hard negatives. These samples are used for semantic alignment training to achieve cross-modal semantic consistency.
<img width="1493" height="1310" alt="图片1" src="https://github.com/user-attachments/assets/1152bc6a-7c20-4239-8d26-ef0139749594" />


##
The most important supplementary file is provided at the following link: https://github.com/3505675604/SemMol/blob/main/Supplementary_Materials/Supplementary%20Materials.pdf

## Pre-trained Models and Datasets
- **Our pre-trained SemMol mode can all be downloaded via Baidu Netdisk:
- **Models:Download link: https://huggingface.co/Lin-Glory/SemMol_model
- **Datasets:Download link: https://huggingface.co/datasets/Lin-Glory/SemMol_datasets/tree/main/dataset
- **Please download and extract the files before training or fine-tuning.
---

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

1. **Clone the repository**
2. **Create environment** (recommended)
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
  title={SemMol: Semantic-Level Multi-Modal Molecular Representation Learning},
  author={Anonymous},
  journal={arXiv preprint},
  year={202X}
}
```








