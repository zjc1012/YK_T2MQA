import os
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def min_max_scaling(series, new_min=0, new_max=5):
    series = pd.to_numeric(series, errors='coerce').fillna(0)
    
    original_min = series.min()
    original_max = series.max()
    
    if original_max == original_min:
        return pd.Series([new_min] * len(series))
    scaled_series = (series - original_min) / (original_max - original_min) * (new_max - new_min) + new_min
    
    return scaled_series

def load_and_scale_csv(file_path, file_suffix):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['Image'] = df['Image'].astype(str).str.strip('"')
    df['Scaled_Score'] = min_max_scaling(df['Score'])
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    col_name = f'Score_{file_name}_{file_suffix}'
    
    result_df = df[['Image', 'Scaled_Score']].rename(columns={'Scaled_Score': col_name})
    
    return result_df, col_name

def merge_two_csvs(csv1_path, csv2_path, output_path):
    input_files = [csv1_path, csv2_path]
    valid_files = []
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"错误：文件 {file_path} 不存在！")
        else:
            valid_files.append(file_path)
    
    if len(valid_files) != 2:
        print("错误：必须提供两个有效的CSV文件！")
        return
    
    all_dfs = []
    score_columns = []
    
    print(f"开始处理两个CSV文件：")
    print(f"1. {os.path.basename(valid_files[0])}")
    print(f"2. {os.path.basename(valid_files[1])}")
    
    df1, col1_name = load_and_scale_csv(valid_files[0], "1")
    all_dfs.append(df1)
    score_columns.append(col1_name)
    
    df2, col2_name = load_and_scale_csv(valid_files[1], "2")
    all_dfs.append(df2)
    score_columns.append(col2_name)
    
    merged_df = all_dfs[0].merge(all_dfs[1], on='Image', how='inner')
    
    missing_cols = [col for col in score_columns if col not in merged_df.columns]
    if missing_cols:
        print(f"错误：合并后缺失列 {missing_cols}")
        return
    
    merged_df['Average_Score'] = merged_df[score_columns].mean(axis=1)
    final_df = merged_df[['Image', 'Average_Score']].rename(columns={'Average_Score': 'Score'})
    final_df['Score'] = final_df['Score'].round(6)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig', quotechar='"')
    print(f"\n合并完成！结果已保存到：{output_path}")
    
    return final_df

def parse_args():
    parser = argparse.ArgumentParser(description="合并两个CSV文件，归一化Score列并计算平均分")
    parser.add_argument('--csv1', required=True, type=str, 
                        help='第一个CSV文件的完整路径')
    parser.add_argument('--csv2', required=True, type=str, 
                        help='第二个CSV文件的完整路径')
    parser.add_argument('--output', required=True, type=str, 
                        help='输出CSV文件的完整路径（如：./merged_result.csv）')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    merge_two_csvs(args.csv1, args.csv2, args.output)

if __name__ == '__main__':
    main()