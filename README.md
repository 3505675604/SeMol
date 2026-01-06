
# Semantic-level multimodal molecular learning inspired by biological concept formation via soft matching.

Figure 1：Based on biomimetic principles, the SemMol model utilizes a "Dynamic Center Library" (DCL) and a soft matching mechanism to deeply learn molecular structural and semantic features from large-scale data. Experiments demonstrate that the model consistently outperforms existing state-of-the-art methods in both molecular representation learning and property prediction.
![10](https://github.com/user-attachments/assets/47f17e40-5fad-4882-b452-f8f165edbedd)
Figure 2：The SemMol model projects multi-dimensional molecular representations into a unified space and constructs a Dynamic Center Library (DCL), utilizing the soft matching mechanism (ACSM) to achieve one-to-many associations between molecules and semantic centers. Compared to traditional one-to-one matching, this method enables learning from a broader knowledge distribution, significantly enhancing the model's generalization ability and discriminative power.
![9](https://github.com/user-attachments/assets/6eb4750e-b7a4-4ebf-9494-86e2d59c1fc6)



## Pre-trained Models
Our pre-trained SemMol mode can all be downloaded via Baidu Netdisk:
Download link: https://pan.baidu.com/s/1nIcHZoealZG2kRBem0949w Extraction code: abcd
Please download and extract the files before training or fine-tuning.
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

## 🗂️ Project Structure

```
├── config/           # Centralized configuration (config.json)
├── core/             # Core algorithms: center library, clustering, pseudo-pair logic
├── data/             # Example datasets (CSV, SMILES, targets)
├── datasets/         # Data loading, splitting (scaffold/random), and processing
├── img/              # Images and figures for reports or publications
├── model/            # Model components: embedding, fusion, projector, pseudo-pair
├── model_config/     # Model download and usage instructions
├── weight/           # Pretrained weights and training history
├── train.py          # Main training & evaluation script
├── environment.yml   # Conda environment for full reproducibility
└── README.md         # This documentation
```

---

## ⚙️ Installation & Environment

1. **Clone the repository**
2. **Create environment** (recommended)
   ```bash
   conda env create -f environment.yml
   conda activate A
   ```
3. **Install additional dependencies** (if needed)
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Datasets

- Place your CSV datasets in `data/`. Each file should contain a `smiles` column and the appropriate target column (see `config/config.json`).
- Supported datasets: BBBP, ESOL, Lipophilicity, Tox21, etc.
- Data splitting: Scaffold split (chemically-aware) and random split are both supported.

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

## 🏃‍♂️ Training & Evaluation

**Basic usage:**
```bash
python train.py \
    --task-type classification \
    --data-path path \
    --target-column Class/reg \
    --batch-size 32 \
    --lr 5e-4 \
    --epochs 50 \
    --hard-negative-k -1
```

**Advanced examples:** (see `config/config.json` for more)
```bash
python train \
    --task-type regression \
    --data-path path \
    --target-column   -- \
    --normalize-targets \
    --batch-size 64 \
    --lr 5e-4 \
    --epochs 50 \
    --hard-negative-k 32 \
    --hard-negative-ratio 0.3
```

**Model weights and training history** are saved in `weight/` after each run.

---

## 🏗️ Extending the Platform

- **Add new datasets**: Place in `data/` and update `config/config.json`.
- **Custom models**: Implement in `model/` and reference in config.
- **New data splits or augmentations**: Add to `datasets/`.
- **Custom loss or metrics**: Extend in `core/` or `train.py`.

---

## 📥 Pretrained Models

See `model_config/File Description` for download links and usage instructions for pretrained models.

---

## 🧪 Reproducibility & Best Practices

- All random seeds, splits, and hyperparameters are controlled via config.
- Use `environment.yml` for full environment reproducibility.
- For large-scale or production runs, see the `production_mode` and `debug_mode` settings in config.

---

## 📚 References & Citation

If you use this platform in your research, please cite the original authors and relevant papers.

---

## 🤝 Contributing & Support

- Pull requests and issues are welcome!
- For questions, suggestions, or bug reports, please open an issue.

---

**Contact:** For collaboration or consulting, please reach out via GitHub or email.












