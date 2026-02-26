# cryospheric-image-classification
Code for a controlled comparison of CNN, transformer, and hybrid architectures for buried lake image classification over the Greenland Ice Sheet.
# Cryospheric Image Classification  
## 

This repository contains the implementation:

**"Assessing Convolutional and Attention-Based Architectures for Buried Lake Image Classification"**

The study presents a controlled comparison of four deep learning architectures for cryospheric image classification:

- AlexNet (classical CNN)
- ConvNeXt-Tiny (modern CNN)
- Swin-Tiny (attention-based)
- CoAtNet-0 (hybrid convolution–attention)

All models were trained using identical:
- Data splits
- Optimization settings (AdamW, lr = 5e-4)
- Early stopping based on validation loss
- Evaluation metrics (Accuracy, Macro-F1, Macro-AUC)

---

## Repository Structure
