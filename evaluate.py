import os
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
import pandas as pd

def compute_iou(pred, gt):
    # 将图像转换为灰度图像（单通道）
    if pred.mode != 'L':
        pred = pred.convert('L')
    if gt.mode != 'L':
        gt = gt.convert('L')

    # 将图像转换为 numpy 数组并进行二值化
    pred_arr = np.array(pred) / 255.0
    gt_arr = np.array(gt) / 255.0

    # 二值化处理
    pred_bin = (pred_arr > 0.5).astype(np.uint8)
    gt_bin = (gt_arr > 0.5).astype(np.uint8)

    # 计算交集和并集
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 0.0
    return intersection / union

def main():
    test_s_dir = '/18018998051/Ref-AVS/results/EEMC/test_s'
    gt_mask_dir = '/18018998051/Ref-AVS/data/REFAVS/gt_mask'
    output_csv = '/18018998051/Ref-AVS/data/REFAVS/failure_cases.csv'
    failure_samples = []
    threshold = 0.5  # IoU 阈值

    all_items = os.listdir(test_s_dir)
    total_samples = sum(1 for item in all_items if os.path.isdir(os.path.join(test_s_dir, item)))

    # 创建进度条
    pbar = tqdm(total=total_samples, desc="Evaluating samples", unit="sample",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for pred_subdir in all_items:
        pred_subdir_path = os.path.join(test_s_dir, pred_subdir)
        if not os.path.isdir(pred_subdir_path):
            continue

        parts = pred_subdir.split('_')
        if len(parts) < 5:
            print(f"Skipping invalid folder: {pred_subdir} (needs at least 5 parts)")
            continue

        # 构建 GT 子文件夹名
        gt_subdir_name = '_'.join(parts[:-2])
        gt_subdir_path = os.path.join(gt_mask_dir, gt_subdir_name)

        # 查找预测文件夹下的所有 fid_* 子文件夹
        pred_subfolders = [d for d in os.listdir(pred_subdir_path)
                           if os.path.isdir(os.path.join(pred_subdir_path, d)) and d.startswith('fid_')]

        total_iou = 0.0
        total_images = 0
        # 处理该样本下的所有 fid_* 子文件夹
        for subfolder in pred_subfolders:
            pred_images_dir = os.path.join(pred_subdir_path, subfolder)
            # 构建 GT 子文件夹路径（包含相同子文件夹名）
            gt_subfolder_path = os.path.join(gt_subdir_path, subfolder)

            if not os.path.exists(gt_subfolder_path) or not os.path.isdir(gt_subfolder_path):
                print(f"GT folder not found: {gt_subfolder_path}")
                continue

            # 遍历 10 个图像
            for i in range(10):
                pred_file = f"{i:05d}.png"
                pred_file_path = os.path.join(pred_images_dir, pred_file)
                gt_file_path = os.path.join(gt_subfolder_path, pred_file)

                if not os.path.exists(pred_file_path) or not os.path.exists(gt_file_path):
                    continue  # 跳过缺失的图像

                try:
                    pred_img = Image.open(pred_file_path)
                    gt_img = Image.open(gt_file_path)
                except Exception as e:
                    print(f"Error opening images for {pred_file}: {e}")
                    continue

                # 调整 GT 图像大小以匹配预测图像
                if pred_img.size != gt_img.size:
                    gt_img = gt_img.resize(pred_img.size, Image.NEAREST)

                # 计算 IoU
                iou = compute_iou(pred_img, gt_img)
                total_iou += iou
                total_images += 1

        # 计算样本平均 IoU
        if total_images == 0:
            print(f"Skipping sample {pred_subdir} (no valid images found)")
            continue

        average_iou = total_iou / total_images

        # 如果平均 IoU 低于阈值，记录样本
        if average_iou < threshold:
            failure_samples.append({
                'sample_name': pred_subdir,  # 直接使用文件夹名
                'average_iou': average_iou
            })
        if average_iou < threshold:
            pbar.set_postfix(sample=pred_subdir, avg_iou=f"{average_iou:.4f} (FAIL)", color="red")
        else:
            pbar.set_postfix(sample=pred_subdir, avg_iou=f"{average_iou:.4f}")

        pbar.update(1)

    pbar.close()  # 关闭进度条
        # 导出 CSV 报告
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sample_name', 'average_iou']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_samples)

        # 输出结果
    print(f"\nTotal failing samples (average IoU < {threshold}): {len(failure_samples)}")
    print(f"Report saved to: {os.path.abspath(output_csv)}")

    # 打印失败样本的详细信息
    if failure_samples:
        print("\nFailing samples (top 5):")
        for idx, sample in enumerate(failure_samples[:5], 1):
            print(f"{idx}. Sample: {sample['sample_name']}, Avg IoU: {sample['average_iou']:.4f}")




def add_vid_exp_to_csv(file1_path, file2_path, output_path):
    """
    将第二个CSV中的vid和exp添加到第一个CSV，输出vid, uid, exp, iou

    Args:
        file1_path: 第一个CSV文件路径（包含uid, average_iou）
        file2_path: 第二个CSV文件路径（包含vid, uid, exp）
        output_path: 输出文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        raise FileNotFoundError(f"File not found: {file1_path}")
    if not os.path.exists(file2_path):
        raise FileNotFoundError(f"File not found: {file2_path}")

    # 读取第一个文件（包含uid, average_iou）
    df1 = pd.read_csv(file1_path)

    # 读取第二个文件（包含vid, uid, exp）
    df2 = pd.read_csv(file2_path)

    # 确保列名正确（处理大小写问题）
    uid_col1 = [col for col in df1.columns if 'sample_name' in col.lower()][0]
    uid_col2 = [col for col in df2.columns if 'uid' in col.lower()][0]
    exp_col = [col for col in df2.columns if 'exp' in col.lower()][0]
    vid_col = [col for col in df2.columns if 'vid' in col.lower()][0]
    fid_col = [col for col in df2.columns if 'fid' in col.lower()][0]

    # 重命名第一个文件的average_iou为iou
    if 'average_iou' in df1.columns:
        df1 = df1.rename(columns={'average_iou': 'iou'})

    # 确保uid列是字符串类型（避免类型不匹配）
    df1[uid_col1] = df1[uid_col1].astype(str).str.strip()
    df2[uid_col2] = df2[uid_col2].astype(str).str.strip()

    # 合并数据（左连接）
    merged_df = pd.merge(
        df1,
        df2[[vid_col, uid_col2, exp_col, fid_col]],
        left_on=uid_col1,
        right_on=uid_col2,
        how='left'
    )

    # 重命名列（移除重复的uid列）
    merged_df = merged_df.drop(columns=[uid_col2])
    merged_df = merged_df.rename(columns={vid_col: 'vid', exp_col: 'exp'})

    # 重新排列列顺序
    final_columns = ['vid', 'sample_name', 'fid', 'exp', 'iou']

    # 确保列存在（如果不存在则创建空列）
    for col in final_columns:
        if col not in merged_df.columns:
            merged_df[col] = pd.NA

    # 选择并排序列
    merged_df = merged_df[final_columns]

    # 保存结果
    merged_df.to_csv(output_path, index=False)

    # 打印统计信息
    print(f"Successfully added 'vid' and 'exp' columns. Output saved to: {os.path.abspath(output_path)}")
    print(f"Total records: {len(merged_df)}")
    print(f"Missing exp values: {merged_df['exp'].isna().sum()}")
    print(f"Missing vid values: {merged_df['vid'].isna().sum()}")

if __name__ == "__main__":
    # main()
    FILE1 = '/18018998051/Ref-AVS/data/REFAVS/failure_cases.csv'  # 第一个CSV文件（uid, average_iou）
    FILE2 = '/18018998051/Ref-AVS/data/metadata.csv'  # 第二个CSV文件（vid, uid, split, fid, exp）
    OUTPUT = 'file1_with_exp.csv'

    add_vid_exp_to_csv(FILE1, FILE2, OUTPUT)