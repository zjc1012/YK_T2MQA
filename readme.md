# YK_T2MQA

## Environment Setup
```bash
conda env create -f environment.yml
conda activate YK_T2MQA
```

## Training & Inference Pipeline
### 1. Train model0
```bash
bash train.sh  # Only execute the first command for model0 training
```

### 2. Infer validation set (generate pseudo-labels)
- Modify `infer.sh`:
  - Update checkpoint path to model0's ckpt
  - Set data path to `val_img` and `val_total.csv`
- Run inference:
```bash
bash infer.sh
```
- Merge results (Average of k fold) with original training set (with GT) to generate:  
  `data/train_gt_val_pseudo.csv` 

### 3. Train model1
```bash
bash train.sh  # Only execute the second command for model1 training
```

### 4. Final inference (competition submission)
- Infer model1
```bash
bash infer.sh
```
  - Results saved to: `data/model1_result`

- Infer model0
  - Modify `infer.sh`:
    - Update checkpoint path to model0's ckpt
```bash
bash infer.sh
```
  - Results saved to: `data/model0_result`

- Obtain Final Result
```bash
bash average.sh
```
  - Results saved to: `data/final_result`

## Network Architecture
- Dual encoder structure (visual + text) for AIGC image quality assessment
- Self-distillation training strategy across 2 model stages (model0 → model1)

## File Paths
| File/Directory               | Path                                          |
|------------------------------|-----------------------------------------------|
| Environment config           | `environment.yml`                             |
| Model0/1 training script     | `train.sh`                                    |
| Inference script             | `infer.sh`                                    |
| Average script               | `average.sh`                                    |
| Model0 training data         | `data/train_total.csv`                        |
| Model1 training data         | `data/train_gt_val_pseudo.csv`                |