***

# 🫁 Chest X-Ray Pneumonia Detection — Deep Learning Classifier

> **Graduate Project** · Faculty of Computer Science & Artificial Intelligence  
> Kaggle Notebook: [chest-Abdoul](https://www.kaggle.com/code/abdelwhabradi/chest-abdoul/notebook)

---

## 📌 Project Overview

Pneumonia is a life-threatening respiratory infection that causes hundreds of thousands of deaths annually, particularly among children and the elderly. Early and accurate diagnosis from chest X-rays is critical — yet it remains a challenging task even for trained radiologists.

This project presents a **deep learning-based binary classification system** that automatically detects pneumonia from chest X-ray images with near-human-level accuracy. Using a multi-stage transfer learning strategy and Test Time Augmentation (TTA), our model achieves a remarkable **99.23% accuracy** on the held-out test set — demonstrating that AI-assisted diagnosis has the potential to meaningfully support clinical decision-making.

---

## 📥 Model & Artifacts

Due to file size limitations, the trained model weights and full training logs are hosted on Google Drive.

> [!IMPORTANT]  
> **Download trained weights and stage-wise logs here:** > 📂 **[Google Drive: Project Artifacts](https://drive.google.com/drive/folders/105SihTlkFIkITXoB0ZJh8bSEcO_STPUb?usp=sharing)**

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
| **Source** | [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Classes** | `NORMAL` · `PNEUMONIA`                    |
| **Format** | JPEG grayscale radiograph images          |
| **Split** | Train / Validation / Test                |

---

## 🏗️ Model Architecture & Training Strategy

Our approach uses **transfer learning** with a carefully designed multi-stage training pipeline — a proven method that maximizes performance while preventing overfitting on medical imaging data.

### Training Pipeline

The training is divided into three progressive stages:

```
Stage 1 → Warm-up: Train only the classifier head (backbone frozen)
Stage 2 → Frozen Consolidation: Fine-tune the head with frozen backbone
Stage 3 → Full Fine-tuning: Unfreeze all layers and train end-to-end
```

### Key Techniques

| Technique | Description |
|---|---|
| **Transfer Learning** | Pre-trained CNN backbone adapted to chest X-ray classification |
| **Multi-Stage Training** | Progressive unfreezing of layers for stable convergence |
| **Test Time Augmentation (TTA)** | Multiple augmented inference passes averaged for robust predictions |
| **Checkpointing** | Automatic retention of the model with the highest validation accuracy |

---

## 📊 Results

### Stage 2 — Frozen Consolidation
✅ **Best Accuracy: 98.27%**

### Stage 3 — Full Fine-tuning
✅ **Global Best Accuracy: 99.23%** (Saved at best checkpoint)

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
| **Accuracy** | **99.23%** |
| **Precision** | 99.04%  |
| **Recall** | 99.04%  |
| **F1 Score** | 99.04%  |

---

## 🗂️ Repository Structure

The following files represent the core project outputs (available via the Google Drive link):

```
📂 Project Artifacts/
│
├── 🧠 Modle.pth                # PyTorch model weights (Best Checkpoint)
├── 📦 Modle.pkl                # Exported fastai Learner for easy inference
│
├── 📈 Stage-1.jpg              # Training logs for Initial Warm-up
├── 📈 Stage-2.jpg              # Training logs for Frozen Consolidation
├── 📈 Stage-3.jpg              # Training logs for Full Fine-tuning
│
├── 📊 Learning-Curve.jpg       # Train/Valid Loss visualization
└── 🎯 Confusion matrix.jpg     # Final evaluation performance visualization
```

---

## ⚙️ How to Reproduce

1. **Open the Notebook:** [Kaggle Notebook](https://www.kaggle.com/code/abdelwhabradi/chest-abdoul/notebook)
2. **Dataset:** Ensure the `paultimothymooney/chest-xray-pneumonia` dataset is attached.
3. **Inference:** To use the pre-trained weights, download `Modle.pkl` from the Drive link and load it:

```python
from fastai.vision.all import load_learner
learn = load_learner('Modle.pkl')
pred, idx, probs = learn.predict('path_to_xray.jpg')
```

---

## 👨‍💻 Team

| Name | Role |
|---|---|
| **Abdelwhab Radi** | Model Architecture · Training Pipeline · Evaluation |


---

## 🔭 Future Work

- [ ] Extend to multi-class classification (COVID-19, Tuberculosis).
- [ ] Deploy as a web application with a DICOM file upload interface.
- [ ] Integrate Grad-CAM visualizations to highlight pathological regions.

---

<p align="center">
  <i>Built with dedication as a Computer Science Graduate Project.</i><br/>
  <i>Because better tools save lives. 🩺</i>
</p>
