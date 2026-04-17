# Brightness Perceiving for Recursive Low-Light Image Enhancement


This README summarizes how to organize data, train the three stages, and run inference for the paper-aligned code.

## 1. Overview

The full pipeline is split into three stages:

- **Step 1:** pre-train **ACT-Net** only.
- **Step 2:** pre-train **BP-Net** with the frozen ACT-Net.
- **Step 3:** jointly fine-tune **ACT-Net + BP-Net**.

In the paper, Step 1 trains ACT-Net on **Level 4** images only and uses **one enhancement pass**. BP-Net predicts the recursive factor from the **histogram of the V channel in HSV**, and its output is constrained to **[rho_min, rho_max] = [1, 10]**. The joint training stage uses all brightness levels. 

## 2. Important prerequisite

Before running **Step 2**, **Step 3**, or **final inference**, make sure that:

- `model.light_class()` has been rewritten to accept a **256-bin histogram** rather than the raw V-channel image.
- The BP-Net output corresponds to a recursive factor in **[1, 10]**.

If `light_class()` is still the old convolutional image-input version, Step 2 / Step 3 / final test will fail with an input-shape mismatch.

## 3. Recommended folder structure

Put the project files in a root directory like this:

```text
project_root/
├─ dataloader.py
├─ model.py
├─ loss_function.py
├─ psnr_ssim.py
├─ step1_train.py
├─ step2_train.py
├─ step3_train.py
├─ test.py
└─ data/
   ├─ step_1/
   │  ├─ train/
   │  ├─ val/
   │  ├─ val_gt/
   │  ├─ results/
   │  └─ save_model/
   ├─ step_2/
   │  ├─ train/
   │  ├─ val/
   │  ├─ val_gt/
   │  ├─ results/
   │  └─ save_model/
   ├─ step_3/
   │  ├─ train/
   │  ├─ val/
   │  ├─ val_gt/
   │  ├─ results/
   │  └─ save_model/
   ├─ test_in/
   ├─ test_out/
   └─ test_gt/
```

## 4. What data should be placed in each folder?

### Step 1: `data/step_1/`

- `train/`: **Level 4 low-light images only**
- `val/`: validation low-light images
- `val_gt/`: paired GT images for validation

This matches the paper's Step 1, where ACT-Net is pre-trained using **Level 4** images only and the recursive enhancement times are manually fixed to **1**.

### Step 2: `data/step_2/`

- `train/`: low-light training images from **all levels**
- `val/`: validation low-light images
- `val_gt/`: paired GT images for validation

Step 2 freezes ACT-Net and trains BP-Net with pseudo labels produced by recursively enhancing each image until the brightness approaches the target value.

### Step 3: `data/step_3/`

- `train/`: low-light training images from **all levels**
- `val/`: validation low-light images
- `val_gt/`: paired GT images for validation

Step 3 jointly fine-tunes ACT-Net and BP-Net on the full mixed-brightness training set.

### Test folders

- `data/test_in/`: images to be enhanced
- `data/test_out/`: output images saved by the test script
- `data/test_gt/`: optional paired GT images for PSNR/SSIM evaluation

## 5. Training order

Run the three stages in order.

### 5.1 Step 1 — train ACT-Net only

```bash
python step1_train_actnet_paper_aligned.py
```

Default paths used by the script:

- training images: `./data/step_1/train/`
- validation images: `./data/step_1/val/`
- validation GT: `./data/step_1/val_gt/`
- results: `./data/step_1/results/`
- checkpoints: `./data/step_1/save_model/`

Expected output checkpoint:

```text
./data/step_1/save_model/ACTNet_best.pth
```

### 5.2 Step 2 — train BP-Net

```bash
python step2_train_bpnet_paper_aligned.py
```

Default paths used by the script:

- training images: `./data/step_2/train/`
- validation images: `./data/step_2/val/`
- validation GT: `./data/step_2/val_gt/`
- results: `./data/step_2/results/`
- checkpoints: `./data/step_2/save_model/`
- ACT-Net pretrained weight: `./data/step_1/save_model/ACTNet_best.pth`

Expected output checkpoint:

```text
./data/step_2/save_model/BPNet_best.pth
```

### 5.3 Step 3 — joint fine-tuning

```bash
python step3_train_joint_paper_aligned.py
```

Default paths used by the script:

- training images: `./data/step_3/train/`
- validation images: `./data/step_3/val/`
- validation GT: `./data/step_3/val_gt/`
- results: `./data/step_3/results/`
- checkpoints: `./data/step_3/save_model/`
- ACT-Net pretrained weight: `./data/step_1/save_model/ACTNet_best.pth`
- BP-Net pretrained weight: `./data/step_2/save_model/BPNet_best.pth`

Expected output checkpoints:

```text
./data/step_3/save_model/ACTNet_best.pth
./data/step_3/save_model/BPNet_best.pth
```

## 6. Inference

### 6.1 Test Step 1 ACT-Net only

This mode runs **one enhancement pass** and is suitable for checking the Step 1 model.

```bash
python test_recursive_llie_paper_aligned.py \
  --mode step1 \
  --test_images_path ./data/test_in/ \
  --results_folder ./data/test_out/ \
  --pretrain_dir ./data/step_1/save_model/ACTNet_best.pth \
  --gt_path ./data/test_gt/
```

### 6.2 Test the final recursive model

This mode loads both ACT-Net and BP-Net. BP-Net predicts the iteration number, and ACT-Net is applied recursively.

```bash
python test_recursive_llie_paper_aligned.py \
  --mode final \
  --test_images_path ./data/test_in/ \
  --results_folder ./data/test_out/ \
  --pretrain_dir ./data/step_3/save_model/ACTNet_best.pth \
  --class_pretrain_dir ./data/step_3/save_model/BPNet_best.pth \
  --gt_path ./data/test_gt/
```

The script will print, for each image:

- the file name
- the number of enhancement iterations actually used
- the mean V value of the output

If `--gt_path` is valid, it also reports **PSNR** and **SSIM**.

## 7. Key training settings in the current paper-aligned scripts

### Step 1

- learning rate: `1e-4`
- batch size: `8`
- epochs: `200`
- exposure target: `0.6`
- loss weights: `lambda_exp=1.0`, `lambda_col=0.5`, `lambda_tv=200.0`

### Step 2

- learning rate: `1e-4`
- batch size: `8`
- epochs: `200`
- target brightness: `0.6`
- BP-Net range: `rho_min=1`, `rho_max=10`

### Step 3

- learning rate: `1e-4`
- batch size: `8`
- epochs: `200`
- target brightness: `0.6`
- total loss weights: `lambda_exp=1.0`, `lambda_col=0.5`, `lambda_tvm=200.0`, `lambda_p=0.001`

## 8. Notes

1. **Step 1 training data must be Level 4 only.** If you put mixed-brightness images into `data/step_1/train/`, the behavior no longer matches the paper.
2. **Step 2 / Step 3 / final test require the histogram-input BP-Net.**
3. The recursive factor is limited to **1–10** in the paper-aligned implementation.
4. Validation and testing with PSNR/SSIM require paired GT images with the **same filenames**.
5. If your images are stored elsewhere, you can override the default paths with command-line arguments.

## 9. Minimal full workflow

```bash
# 1) Train ACT-Net
python step1_train_actnet_paper_aligned.py

# 2) Train BP-Net
python step2_train_bpnet_paper_aligned.py

# 3) Joint fine-tuning
python step3_train_joint_paper_aligned.py

# 4) Final inference
python test_recursive_llie_paper_aligned.py \
  --mode final \
  --test_images_path ./data/test_in/ \
  --results_folder ./data/test_out/ \
  --pretrain_dir ./data/step_3/save_model/ACTNet_best.pth \
  --class_pretrain_dir ./data/step_3/save_model/BPNet_best.pth
```

## Copyright Notice

This code is provided for academic research and educational use only.  
If you use this code, in whole or in part, in your research or any resulting publication, please cite the following paper:

> Wang H, Peng L, Sun Y, et al. Brightness perceiving for recursive low-light image enhancement[J]. *IEEE Transactions on Artificial Intelligence*, 2023, 5(6): 3034-3045.

```bibtex
@article{wang2023brightness,
  title={Brightness perceiving for recursive low-light image enhancement},
  author={Wang, Haodian and Peng, Long and Sun, Yuejin and et al.},
  journal={IEEE Transactions on Artificial Intelligence},
  volume={5},
  number={6},
  pages={3034--3045},
  year={2023}
}
