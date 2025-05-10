# 🧠 Brain Tumor Segmentation using Vision Transformer (ViT)

This project implements a semantic segmentation model for detecting brain tumors from MRI scans using a Vision Transformer (ViT) as the encoder and a CNN-based decoder.

---

## 📂 Dataset

- **Source**: Kaggle Brain Tumor Segmentation Dataset (COCO format)
- **Content**:
  - MRI images
  - COCO-style annotations
  - Binary masks (tumor vs. background)
- **Splits**: Training / Validation / Test

---

## ⚙️ Preprocessing

- Parsed annotations using `pycocotools`
- Generated binary masks
- Resized images and masks to `256x256`
- Normalized pixel values to `[0, 1]`
- Loaded using custom `tf.data.Dataset` pipelines

---

## 🧠 Model Architecture

### Encoder: Vision Transformer (ViT)
- Patch Embedding via `Conv2D`
- Positional Embeddings
- Stacked Transformer Blocks (Multi-Head Attention + MLP with GELU)
- Layer Normalization and residual connections

### Decoder: U-Net-style CNN
- 4 Upsampling Blocks with:
  - Bilinear Upsampling
  - Conv2D → BatchNorm → ReLU
- Final 1×1 Conv layer with Sigmoid activation

---

## 📉 Loss Function & Metrics

- **Loss**: Combined Dice + Binary Crossentropy  
  \[
  \text{Loss} = 0.6 \times \text{Dice Loss} + 0.4 \times \text{Binary Crossentropy}
  \]
- **Metrics**:
  - Accuracy
  - Dice Coefficient

---

## 🚀 Training Configuration

- **Framework**: TensorFlow / Keras
- **Batch Size**: 16  
- **Epochs**: 50  
- **Optimizer**: Adam (`lr = 1e-4`)
- **Callbacks**:
  - `ReduceLROnPlateau`
  - `EarlyStopping`

---

## 📈 Results

- **Dice Score**: 0.47  
- **Test Loss**: 0.37  
- **Test Accuracy**: 0.96  

> Note: Moderate Dice score due to class imbalance and boundary segmentation difficulty.

---

## 🔮 Future Improvements

- Use pre-trained ViTs (e.g. DeiT, Swin)
- Try Tversky or Focal Loss
- Incorporate data augmentation
- Use hybrid ViT + CNN models

---

## 📌 How to Run

1. Clone the repo and open in a Jupyter/Colab notebook
2. Ensure you have `pycocotools`, `tensorflow`, `matplotlib`, `PIL`, etc.
3. Modify paths to point to your dataset directory
4. Run the cells in sequence

---

## 👥 Authors

- Dhruv Patel  
*Ahmedabad University – CSE602: Deep Learning*
