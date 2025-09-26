- # SCDW: Semi-supervised Cervical Abnormal Cell Detection via Whole Slide Image Labeling

  > Official implementation of **SCDW**, a novel semi-supervised framework for cervical abnormal cell detection leveraging WSI-level diagnostic labels.  
  > Built upon  [MMDetection](https://github.com/open-mmlab/mmdetection).

  ---

  ## 📌 Introduction

  Early and accurate detection of cervical abnormal cells is critical for cervical cancer prevention.  
  However, annotating large-scale cervical cytology datasets at the cell level is expensive and time-consuming.  

  We propose **SCDW**, a semi-supervised detection framework that incorporates **Whole Slide Image (WSI) diagnostic labels** into the teacher–student paradigm, effectively bridging the gap between WSI-level supervision and cell-level detection.  

  ![Framework](assets/framework.png) 

  **Key components of SCDW:**

  1. **Fusion WSI Label Class Balanced Resampling (FWL-CBR):**  
     Balances training data by fusing WSI-level labels into sampling, mitigating class imbalance.  

  2. **Complexity-Aware Augmentation Strategy (CAAS):**  
     Dynamically adjusts augmentation strength based on sample complexity estimated from pseudo labels and WSI labels.  

  3. **WSI-Guided Adaptive Threshold Pseudo-Labeling (WATPL):**  
     Refines teacher-generated pseudo labels via adaptive thresholds optimized with a Genetic Algorithm (GA).  

  Together, these designs enhance representation learning, improve pseudo-label quality, and boost detection performance under limited annotations.

  ---

  ## 🚀 Framework Overview

  SCDW follows a **teacher–student detector paradigm** based on RetinaNet with ResNet-50 backbone:

  - Teacher generates pseudo labels on weakly augmented unlabeled patches.
  - WATPL refines pseudo labels using WSI-level priors and GA-optimized thresholds.
  - Student learns from both labeled and unlabeled (strongly augmented) data.
  - Teacher updated via EMA of student weights.

  ---

  ## 📊 Dataset Preparation

  We follow **COCO format**.

  - **Labeled set:** normal/abnormal cells with bounding boxes.

  - **Unlabeled set:** only `images` field is required, no annotations needed.

  - Each image in `images` can include custom fields for WSI info:

    ```
    {
      "id": 1001,
      "file_name": "slide001_patch_0001.png",
      "height": 512,
      "width": 512,
      "wsi_id": "slide001",
      "wsi_label": 1
    }
    ```

  Directory example:

  ```
  data/cervical/
  ├── annotations/
  │   ├── instances_train_labeled.json
  │   └── instances_train_unlabeled.json
  ├── train_labeled/
  └── train_unlabeled/
  ```

## 🙌 Acknowledgements

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [SoftTeacher](https://github.com/microsoft/SoftTeacher)
- [MixPL](https://github.com/Czm369/MixPL)
- [ConsistentTeacher](https://github.com/Adamdad/ConsistentTeacher)
- [SCAC](https://github.com/Lewisonez/cc_detection)
