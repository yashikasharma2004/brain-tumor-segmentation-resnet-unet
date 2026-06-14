# Brain Tumor Segmentation using ResNet34-UNet

Pixel-level brain tumor segmentation using 
transfer learning on LGG MRI dataset.

## Results
| Metric | Score |
|--------|-------|
| Dice Score | 90.45% |
| IoU (Jaccard) | 82.56% |
| Sensitivity | 90.28% |
| Specificity | 99.86% |
| ROC-AUC | 99.64% |

## Tech Stack
- PyTorch
- ResNet34 (Pretrained ImageNet)
- Transfer Learning
- Data Augmentation
- LGG MRI Dataset (2746 images)

## Architecture
ResNet34 encoder + U-Net decoder with skip 
connections for pixel-level segmentation.

## Dataset
LGG MRI Segmentation Dataset — 110 patients, 
Brain MRI images with FLAIR abnormality masks.
