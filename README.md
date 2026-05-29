# 🫁 Lung X-Ray Disease Classifier (CNN + SE Block + Grad-CAM)

A deep learning model to classify chest X-rays into **Covid**, **Normal**, and **Viral Pneumonia** 
using a custom CNN with Squeeze-and-Excitation (SE) attention blocks and Grad-CAM interpretability.

---

## 📌 What This Project Does

- Splits chest X-ray dataset into train/val/test sets
- Applies data augmentation to handle small dataset size
- Trains a custom CNN with **SE (channel attention) blocks**
- Uses **5-Fold Stratified Cross-Validation** for robust evaluation
- Visualizes model decisions using **Grad-CAM heatmaps**

---

## 🧠 Model Architecture

Input (224×224×3)
→ Data Augmentation (flip, rotation, zoom)
→ Conv2D(32) + MaxPooling
→ Conv2D(64) + MaxPooling
→ Conv2D(128) + MaxPooling
→ SE Block (channel attention)
→ Flatten → Dense(128) → Dropout(0.5)
→ Softmax (3 classes)

---

## 📊 Results

### 5-Fold Cross-Validation

| Fold | Val Accuracy |
|------|-------------|
| 1    | 86.27%      |
| 2    | 84.31%      |
| 3    | 76.47%      |
| 4    | 88.00%      |
| 5    | 86.00%      |

### Test Accuracy: **87.50%**

### Confusion Matrix

|  | Covid | Normal | Viral Pneumonia |
|--|-------|--------|-----------------|
| **Covid** | 13 | 0 | 1 |
| **Normal** | 1 | 7 | 1 |
| **Viral Pneumonia** | 0 | 1 | 8 |

---

## 🔍 Grad-CAM Visualization

Grad-CAM overlays highlight which regions of the X-ray the model 
focuses on when making a prediction — making the model's decisions 
interpretable and transparent.

---

## 🗂️ Repository Structure

---

classifier-cnn-gradcam/
│
├── README.md
└── report/
    └── image_processing_project_report.pdf

> The full implementation including data pipeline, model training,
> evaluation, and Grad-CAM is documented in the project report.

---

## ⚠️ Limitations

- Small dataset (~253 training samples)
- Some Grad-CAM heatmaps attend to non-anatomical regions
- Not intended for real clinical diagnosis

---

## 🔮 Future Improvements

- [ ] Transfer learning with ResNet50 / EfficientNet
- [ ] Larger and more balanced dataset
- [ ] Per-class precision, recall, F1-score
- [ ] Streamlit app for live inference

---

## 🙋 Author

Tarunpreet Singh 
[GitHub](https://github.com/tarunzkaynaat) · [Kaggle](https://www.kaggle.com/tarunarora6029) · [LinkedIn](https://linkedin.com/in/tarunzkaynaat)
