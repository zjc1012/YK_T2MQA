import os, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from models.ELHnet import ELHnet
from utils.ELH_Dataset import ELH_Dataset
from utils.loss import L2RankLoss
import math
from scipy.stats import pearsonr, spearmanr, kendalltau
from torch.nn import functional as F
from torch.utils.data import Subset
import copy
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   # fix the random seed

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    """Logistic function for fitting"""
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    """Fit logistic function and return transformed predictions"""
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    try:
        popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
        y_output_logistic = logistic_func(y_output, *popt)
    except Exception as e:
        print(f"Curve fitting failed, using raw predictions: {e}")
        y_output_logistic = y_output
    return y_output_logistic

def calculate_metrics(y_true, y_pre):
    """Calculate evaluation metrics"""
    # Apply logistic fitting
    y_output_logistic = fit_function(y_true, y_pre)
    
    # Calculate metrics
    test_PLCC = scipy.stats.pearsonr(y_output_logistic, y_true)[0]
    test_SROCC = scipy.stats.spearmanr(y_pre, y_true)[0]
    plcc_srocc_avg = (test_PLCC + test_SROCC) / 2
    
    return {
        'PLCC': test_PLCC,
        'SROCC': test_SROCC,
        'PLCC_SROCC_AVG': plcc_srocc_avg
    }

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="training with k-fold cross validation (stratified by model_name)")
    parser.add_argument('--num_epochs',  help='Maximum number of training epochs.', default=5, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=24, type=int)
    parser.add_argument('--learning_rate', default=0.000005, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--data_dir_2d', default='', type=str, help = 'path to the train set images')
    parser.add_argument('--data_dir_2d_dev', default='', type=str, help = 'path to the val set images')
    parser.add_argument('--img_length_read', default=6, type=int, help = 'number of the using images')
    parser.add_argument('--ckpt_name', default='ntire', type=str)
    parser.add_argument('--k_folds', default=20, type=int, help='Number of folds')
    parser.add_argument('--train_filename_list', default='./data/train_total.csv', type=str, help='CSV file containing training data')
    parser.add_argument('--split_output_dir', default='', type=str, 
                        help='Directory to save stratified split CSV files')
    args = parser.parse_args()
    return args

def delete_old_checkpoints(ckpt_dir, fold_idx, prefix, keep_latest=True):
    fold_num = fold_idx + 1
    pattern = f'fold_{fold_num}_{prefix}*'
    old_files = []
    for file in os.listdir(ckpt_dir):
        if file.startswith(f'fold_{fold_num}_{prefix}') and file.endswith('.pth'):
            old_files.append(os.path.join(ckpt_dir, file))
    
    if keep_latest and len(old_files) > 1:
        old_files.sort(key=lambda x: os.path.getmtime(x))
        if keep_latest:
            old_files = old_files[:-1]
    
    for file_path in old_files:
        try:
            os.remove(file_path)
            print(f'Fold {fold_num} - Deleted old checkpoint: {os.path.basename(file_path)}')
        except Exception as e:
            print(f'Fold {fold_num} - Failed to delete {os.path.basename(file_path)}: {e}')

def train_single_fold(fold_idx, train_indices, val_indices, full_dataset, args, device, score_list, quality_classes):
    print(f"\n{'='*80}")
    print(f"Starting training for Fold {fold_idx + 1}/{args.k_folds}")
    print(f"Fold {fold_idx + 1} - Validation set: {len(val_indices)} samples (1/20 of total)")
    print(f"Fold {fold_idx + 1} - Training set: {len(train_indices)} samples (9/20 of total)")
    print(f"{'='*80}")
    
    train_dataset = Subset(copy.deepcopy(full_dataset), train_indices)
    val_dataset = Subset(copy.deepcopy(full_dataset), val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    model = ELHnet(device, args, score_list, quality_classes)
    model = model.cpu()

    criterion = L2RankLoss().to(device)
    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = True  # 视觉部分可训练
        else:
            param.requires_grad = False  # 文本部分冻结
        
    if torch.cuda.device_count() > 1:
        print(f"Fold {fold_idx + 1} - Using {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p : p.requires_grad, model.parameters()), 
        lr = args.learning_rate, 
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    min_training_loss = float('inf')
    best_avg_score = -float('inf')
    best_metrics = None
    best_epoch = -1

    for epoch in range(args.num_epochs):
        model.train()
        train_start = time.time()
        batch_losses = []

        for i, (img_name, imgs, mos) in enumerate(train_loader):
            imgs = imgs.to(device)
            mos = mos[:, np.newaxis]
            mos = mos.to(device)
            quality_score = model(img_name, imgs)
            
            loss = criterion(quality_score, mos)
            batch_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(batch_losses)
        train_time = time.time() - train_start
        
        scheduler.step()
        lr_current = scheduler.get_last_lr()
        
        model.eval()
        val_start = time.time()
        all_val_preds = []
        all_val_trues = []
        
        with torch.no_grad():
            for i, (img_name, imgs, mos) in enumerate(val_loader):
                imgs = imgs.to(device)
                mos = mos.cpu().numpy()
                quality_score = model(img_name, imgs)
                preds = quality_score.cpu().numpy()
                all_val_preds.extend(preds.flatten())
                all_val_trues.extend(mos.flatten())
        y_true = np.array(all_val_trues)
        y_pre = np.array(all_val_preds)
        val_metrics = calculate_metrics(y_true, y_pre)
        val_time = time.time() - val_start
        
        print(f"\n{'-'*80}")
        print(f'Fold {fold_idx + 1} - Epoch {epoch + 1}/{args.num_epochs}, Current LR: {lr_current[0]:.10f}')
        print(f'Fold {fold_idx + 1} - Training - Loss: {avg_train_loss:.4f}, Time: {train_time:.2f}s')
        print(f'Fold {fold_idx + 1} - Validation - Time: {val_time:.2f}s')
        print(f'Fold {fold_idx + 1} -   PLCC: {val_metrics["PLCC"]:.4f}')
        print(f'Fold {fold_idx + 1} -   SROCC: {val_metrics["SROCC"]:.4f}')
        print(f'Fold {fold_idx + 1} - 🌟 PLCC+SROCC Average: {val_metrics["PLCC_SROCC_AVG"]:.4f}')
        
        ckpt_dir = f'ckpts/{args.ckpt_name}'
        os.makedirs(ckpt_dir, exist_ok=True)
        
        current_avg_score = val_metrics['PLCC_SROCC_AVG']
        if current_avg_score > best_avg_score:
            best_avg_score = current_avg_score
            best_metrics = val_metrics
            best_epoch = epoch + 1
            delete_old_checkpoints(ckpt_dir, fold_idx, 'best_epoch', keep_latest=True)
            best_ckpt_path = f'{ckpt_dir}/fold_{fold_idx + 1}_best_epoch{best_epoch}_avg_{best_avg_score:.4f}.pth'
            torch.save(model.state_dict(), best_ckpt_path)
            print(f'Fold {fold_idx + 1} - Best model updated (epoch {best_epoch}, Avg: {best_avg_score:.4f}): {best_ckpt_path}')
    
    return {
        'fold_idx': fold_idx + 1,
        'best_epoch': best_epoch,
        'best_avg_score': best_avg_score,
        'best_metrics': best_metrics,
        'val_set_size': len(val_indices),
        'train_set_size': len(train_indices)
    }

def create_stratified_kfold_split(csv_path, k_folds=20, random_state=1998, output_dir=None):
    df = pd.read_csv(csv_path, header=0, index_col=False, encoding="utf-8-sig")
    assert 'model_name' in df.columns, "CSV文件必须包含model_name列"
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    labels = df['model_name'].values
    
    fold_splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        fold_splits.append((train_idx, val_idx))
        
        if output_dir is not None:
            fold_dir = os.path.join(output_dir, f'fold{fold_idx+1}')
            os.makedirs(fold_dir, exist_ok=True)
            train_df = df.iloc[train_idx].reset_index(drop=True)
            train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False, encoding='utf-8-sig')
            val_df = df.iloc[val_idx].reset_index(drop=True)
            val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False, encoding='utf-8-sig')
            
            print(f"\nFold {fold_idx+1} 分层统计:")
            print("训练集各模型样本数:")
            print(train_df['model_name'].value_counts())
            print("验证集各模型样本数:")
            print(val_df['model_name'].value_counts())
    
    return fold_splits, df

if __name__=='__main__':
    print('*************************************************************************************************************************')
    
    args = parse_args()
    set_rand_seed()
    cudnn.enabled = True
    ckpt_name = args.ckpt_name
    img_length_read = args.img_length_read
    data_dir_2d = args.data_dir_2d
    data_dir_2d_dev = args.data_dir_2d_dev

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    quality_classes =['bad', 'poor', 'fair', 'good', 'perfect'] 
    train_filename_list = args.train_filename_list
    score_list = [0.0, 1.25, 2.5, 3.75, 5.0]

    print('Training set: ' + train_filename_list)
    
    fold_splits, df = create_stratified_kfold_split(
        csv_path=train_filename_list,
        k_folds=args.k_folds,
        random_state=1998,
        output_dir=args.split_output_dir
    )
    
    full_dataset = ELH_Dataset(
        data_dir_2d = data_dir_2d, 
        data_dir_2d_dev = data_dir_2d_dev,
        datainfo_path = train_filename_list, 
        img_length_read = img_length_read
    )
    
    total_samples = len(full_dataset)
    val_sizes = [len(val_idx) for (train_idx, val_idx) in fold_splits]
    assert len(set(val_sizes)) <= 2, "验证集大小不一致（仅允许最后一折多1个样本）"
    
    all_folds_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        fold_result = train_single_fold(
            fold_idx, train_indices, val_indices,
            full_dataset, args, device, score_list, quality_classes
        )
        all_folds_results.append(fold_result)
    
    for result in all_folds_results:
        print(f'\nFold {result["fold_idx"]}:')
        print(f'  Validation set size: {result["val_set_size"]} samples')
        print(f'  Training set size: {result["train_set_size"]} samples')
        print(f'  Best Epoch: {result["best_epoch"]}')
        print(f'  Best PLCC+SROCC Average: {result["best_avg_score"]:.4f}')
        print(f'  Detailed Metrics:')
        print(f'    PLCC: {result["best_metrics"]["PLCC"]:.4f}')
        print(f'    SROCC: {result["best_metrics"]["SROCC"]:.4f}')
    
    avg_plcc = np.mean([r['best_metrics']['PLCC'] for r in all_folds_results])
    avg_srocc = np.mean([r['best_metrics']['SROCC'] for r in all_folds_results])
    avg_total_score = np.mean([r['best_avg_score'] for r in all_folds_results])
    
    print('\n' + '='*80)
    print('k-Fold Average Metrics (Stratified):')
    print(f'  Average PLCC: {avg_plcc:.4f}')
    print(f'  Average SROCC: {avg_srocc:.4f}')
    print(f'  Average PLCC+SROCC Score: {avg_total_score:.4f}')
    print('='*80)