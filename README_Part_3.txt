
# README - Part 3 Folder

This folder contains scripts and datasets used for training and evaluating a classifier model for tumor and tumor budding tissue detection.

## Folder Structure:
- `real_test_train.py`: 
  - This Python script is used to evaluate a test folder containing real images of tumor and tumor budding tissues. 
  - The pre-trained model is loaded from the `models` folder, which contains models trained in **Part 1**.
  
- `comp_eval.py`: 
  - This script is used to evaluate a classifier model that has been trained using synthetic images. 
  - The synthetic images used for training are stored in the `synth_dir` folder, which consists of two subfolders:
    - `class0`: Contains synthetic images for class 0 (tumor budding).
    - `class1`: Contains synthetic images for class 1 (tumor).
  - Once trained, the model is saved in the `models` folder for future evaluation.

- `synth_dir `: 
  - A copy of the synthetic dataset used for training the classifier. It contains:
    - `class0`: Synthetic images representing tumor budding.
    - `class1`: Synthetic images representing tumor tissue.

- `models `: 
  - A copy of the folder where the trained models are saved. Each model corresponds to different training epochs and can be loaded for evaluation.

- `master `: 
  - This folder contains real images that are used as the test dataset for evaluating both real and synthetic models. This folder is vital for assessing the performance of the classifier on unseen real-world data.
