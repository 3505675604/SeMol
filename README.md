
# Semantic-level multimodal molecular learning inspired by biological concept formation via soft matching.

Figure 1：Based on biomimetic principles, the SemMol model utilizes a "Dynamic Center Library" (DCL) and a soft matching mechanism to deeply learn molecular structural and semantic features from large-scale data. Experiments demonstrate that the model consistently outperforms existing state-of-the-art methods in both molecular representation learning and property prediction.
![10](https://github.com/user-attachments/assets/47f17e40-5fad-4882-b452-f8f165edbedd)
Figure 2：The SemMol model projects multi-dimensional molecular representations into a unified space and constructs a Dynamic Center Library (DCL), utilizing the soft matching mechanism (ACSM) to achieve one-to-many associations between molecules and semantic centers. Compared to traditional one-to-one matching, this method enables learning from a broader knowledge distribution, significantly enhancing the model's generalization ability and discriminative power.
![9](https://github.com/user-attachments/assets/6eb4750e-b7a4-4ebf-9494-86e2d59c1fc6)



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

## 🏃‍♂️ Training & Finetune

**Training:**
```bash
chmod +x /data/FL/Semol/scripts/start_Pre_DPP.sh
/data/FL/Semol/scripts/start_Pre_DPP.sh
```
**Finetune:**
```bash
python finetune.py
```
**All model checkpoints, logs, and training histories are saved in:**
```bash
Save_model/
```

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








