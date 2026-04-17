# 🫁 Chest X-Ray Pneumonia Detection — Deep Learning Classifier

> **Graduate Project** · Faculty of Computer Science & Artificial Intelligence  
> Kaggle Notebook: [chest-Abdoul]([https://www.kaggle.com/code/abdelwhabradi/chest-abdoul/notebook](https://www.kaggle.com/code/abdelwhabradi/chest-abdoul)

---

## 📌 Project Overview

Pneumonia is a life-threatening respiratory infection that causes hundreds of thousands of deaths annually, particularly among children and the elderly. Early and accurate diagnosis from chest X-rays is critical — yet it remains a challenging task even for trained radiologists.

This project presents a **deep learning-based binary classification system** that automatically detects pneumonia from chest X-ray images with near-human-level accuracy. Using a multi-stage transfer learning strategy and Test Time Augmentation (TTA), our model achieves a remarkable **99.23% accuracy** on the held-out test set — demonstrating that AI-assisted diagnosis has the potential to meaningfully support clinical decision-making.

---

## 🎯 Objectives

- Automate the detection of pneumonia from chest X-ray scans.
- Build a robust, generalizable model using state-of-the-art deep learning techniques.
- Minimize both false positives and false negatives to ensure clinical reliability.
- Deliver reproducible, well-documented research-grade results.

---

## 📂 Dataset

| Property        | Details                                  |
|-----------------|------------------------------------------|
| **Source**      | [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Classes**     | `NORMAL` · `PNEUMONIA`                   |
| **Format**      | JPEG grayscale radiograph images         |
| **Split**       | Train / Validation / Test                |

The dataset contains thousands of validated chest X-ray images sourced from pediatric patients, with labels verified by expert physicians.

---

## 🏗️ Model Architecture & Training Strategy

Our approach uses **transfer learning** with a carefully designed multi-stage training pipeline — a proven method that maximizes performance while preventing overfitting on medical imaging data.

### Training Pipeline

The training is divided into three progressive stages:

```
Stage 1 → Warm-up: Train only the classifier head (backbone frozen)
Stage 2 → Frozen Consolidation: Fine-tune the head with frozen backbone (5 epochs)
Stage 3 → Full Fine-tuning: Unfreeze all layers and train end-to-end (25 epochs)
```

This staged approach allows the model to:
1. Learn task-specific features without disrupting pre-trained weights early on.
2. Gradually adapt the backbone to the chest X-ray domain.
3. Converge to a stable, high-accuracy solution.

### Key Techniques

| Technique | Description |
|---|---|
| **Transfer Learning** | Pre-trained CNN backbone adapted to chest X-ray classification |
| **Multi-Stage Training** | Progressive unfreezing of layers for stable convergence |
| **Test Time Augmentation (TTA)** | Multiple augmented inference passes averaged for robust predictions |
| **Best-Checkpoint Saving** | Automatically retains the model with the highest validation accuracy |
| **Learning Curve Monitoring** | Tracks train/validation loss across all ~6,400 steps |

---

## 📊 Results

### Stage 2 — Frozen Consolidation (5 Epochs)

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 Score |
|-------|-----------|-----------|----------|-----------|--------|----------|
| 0     | 0.3126    | 0.1105    | 97.51%   | 97.50%    | 96.25% | 96.85%   |
| 1     | 0.3321    | 0.1407    | 94.63%   | 92.04%    | 95.75% | 93.60%   |
| 2     | 0.3479    | 0.1050    | 96.36%   | 94.45%    | 96.95% | 95.58%   |
| 3     | 0.3136    | 0.0872    | 97.70%   | 96.61%    | 97.77% | 97.16%   |
| **4** | **0.2853**| **0.0657**| **98.27%**| **97.39%**| **98.38%**| **97.87%**|

✅ **Global best after Stage 2: 98.27% accuracy**

---

### Stage 3 — Full Fine-tuning (25 Epochs)

The model continued to improve significantly through full fine-tuning, converging to a final top accuracy of **99.23%** (saved at best checkpoint).

| Epoch | Train Loss | Valid Loss | Accuracy | Precision | Recall | F1 Score |
|-------|-----------|-----------|----------|-----------|--------|----------|
| 15    | 0.2413    | 0.0627    | 99.04%   | 98.70%    | 98.91% | 98.81%   |
| 16    | 0.2270    | 0.0583    | 99.23%   | 99.15%    | 98.94% | 99.04%   |
| 17    | 0.2332    | 0.0480    | 99.14%   | 98.58%    | 99.30% | 98.93%   |
| ...   | ...       | ...       | ...      | ...       | ...    | ...      |
| **24**| **0.2183**| **0.0588**| **99.23%**| **99.04%**| **99.04%**| **99.04%**|

✅ **Global best after Stage 3: 99.23% accuracy**

---

### Final Evaluation (with TTA)

#### Confusion Matrix

```
                 Predicted
                 NORMAL    PNEUMONIA
Actual  NORMAL    285          4
        PNEUMONIA   4        750
```

| Metric        | Score   |
|---------------|---------|
| **Accuracy**  | **99.23%** |
| **Precision** | 99.04%  |
| **Recall**    | 99.04%  |
| **F1 Score**  | 99.04%  |

Out of **1,043 test samples**, the model made only **8 total errors** — 4 false positives and 4 false negatives — an exceptionally strong result for a binary medical classification task.

---

### Learning Curve

The training and validation loss curves demonstrate stable and consistent convergence across ~6,400 steps, with the validation loss settling well below 0.10 — indicating excellent generalization with no signs of overfitting.

![Learning Curve](assets/plot_losses.png)

---

### Confusion Matrix Visualization

![Confusion Matrix](assets/plot_confusion_matrix.png)

---

## 🗂️ Repository Structure

```
chest-abdoul/
│
├── notebook.ipynb              # Main Kaggle notebook (full pipeline)
│
├── artifacts/
│   ├── best_model.pth          # Saved best model weights (PyTorch)
│   ├── export.pkl              # Exported model for inference
│   ├── metrics.json            # Final evaluation metrics (JSON)
│   ├── plot_losses.png         # Training & validation loss curves
│   └── plot_confusion_matrix.png  # Confusion matrix visualization
│
└── assets/
    ├── plot_losses.png
    └── plot_confusion_matrix.png
```

---

## ⚙️ How to Reproduce

### 1. Clone or open the Kaggle Notebook

Visit the notebook directly:  
👉 https://www.kaggle.com/code/abdelwhabradi/chest-abdoul/notebook

### 2. Add the Dataset

In the Kaggle notebook interface, add the following dataset as input:

```
Chest X-Ray Images (Pneumonia)
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

### 3. Run All Cells

The notebook is fully self-contained. Running all cells will:
- Load and preprocess the dataset
- Train the model through all 3 stages
- Save the best checkpoint automatically
- Generate the learning curve and confusion matrix
- Output `metrics.json` with final evaluation scores

### 4. Inference

Load the saved model for inference on new X-ray images:

```python
import torch
from pathlib import Path

model = torch.load('artifacts/best_model.pth')
model.eval()

# Or use the exported fastai learner:
from fastai.vision.all import load_learner
learn = load_learner('artifacts/export.pkl')
pred, idx, probs = learn.predict(img)
```

---

## 🧰 Tech Stack

| Tool / Library | Purpose |
|---|---|
| **Python 3** | Core programming language |
| **PyTorch** | Deep learning framework |
| **fastai** | High-level training API (multi-stage, TTA, callbacks) |
| **torchvision** | Pre-trained model backbone & transforms |
| **scikit-learn** | Metrics: precision, recall, F1, confusion matrix |
| **matplotlib** | Visualization (loss curves, confusion matrix) |
| **Kaggle Notebooks** | Cloud GPU training environment (GPU accelerated) |

---

## 👨‍💻 Team

This project was developed as a **graduation project**, representing months of research, experimentation, and engineering effort.

| Name | Role |
|---|---|
| **Abdelwhab Radi** | Model Architecture · Training Pipeline · Evaluation |
| *(Add teammates)* | *(Add roles)* |

> We are proud to have built a system that pushes toward real-world applicability in AI-assisted medical diagnosis.

---

## 🔭 Future Work

- [ ] Extend to multi-class classification (COVID-19, Tuberculosis, Pleural Effusion).
- [ ] Deploy as a web application with a DICOM file upload interface.
- [ ] Integrate Grad-CAM visualizations to highlight pathological regions for radiologist review.
- [ ] Evaluate on external clinical datasets to validate cross-site generalization.
- [ ] Explore lightweight architectures (MobileNet, EfficientNet-Lite) for edge deployment.

---

## 📄 License

This project is released for academic and research purposes.  
Dataset credit: [Paul Mooney — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
Original dataset sourced from: *Kermany et al., Cell, 2018.*

---

<p align="center">
  <i>Built with dedication as a Computer Science Graduate Project.</i><br/>
  <i>Because better tools save lives. 🩺</i>
</p>
