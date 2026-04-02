#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ===== 手动在这里设置参数 =====
# CPAC 数据集结果根目录（对应之前的 base_output_dir）
BASE_OUTPUT_DIR = "/mnt/sda1/zhangyan/cpac_output/output/ds002748-default"

# 汇总 CSV 输出路径
OUTPUT_CSV_PATH = "/mnt/sda1/zhangyan/cpac_output/质量控制分析/ds002748-default.csv"

# CPAC pipeline 名称、session、task（如有变化在这里改）
PIPELINE_NAME = "pipeline_cpac-default-pipeline"
SESSION = "ses-1"
TASK = None


def find_subjects(base_output_dir):
    """在 base_output_dir 下自动发现所有 sub-* 目录。"""
    subjects = []
    for name in os.listdir(base_output_dir):
        sub_path = os.path.join(base_output_dir, name)
        if name.startswith("sub-") and os.path.isdir(sub_path):
            subjects.append(name)
    subjects.sort()
    return subjects


def extract_metrics_for_subject(base_output_dir, sub_id, pipeline_name, session, task):
    """从单个被试目录中提取 MeanFD_Power, MeanDVARS, boldSnr。"""
    func_dir = os.path.join(
        base_output_dir,
        sub_id,
        "output",
        pipeline_name,
        sub_id,
        session,
        "func",
    )

    motion_tsv_path = None

    if os.path.isdir(func_dir):
        # 如果指定了 task，优先按 task 构造文件名
        if task:
            candidate = os.path.join(
                func_dir,
                f"{sub_id}_{session}_task-{task}_desc-summary_motion.tsv",
            )
            if os.path.exists(candidate):
                motion_tsv_path = candidate
            else:
                print(
                    f"[{sub_id}] 按 task='{task}' 未找到 motion 文件, 尝试自动搜索 *_desc-summary_motion.tsv"
                )

        # 若没有指定 task，或按 task 没找到文件，则在 func 目录中自动搜索
        if motion_tsv_path is None:
            try:
                candidates = []
                prefix = f"{sub_id}_{session}"
                for filename in os.listdir(func_dir):
                    if filename.endswith("_desc-summary_motion.tsv") and filename.startswith(
                        prefix
                    ):
                        candidates.append(filename)

                if len(candidates) == 1:
                    motion_tsv_path = os.path.join(func_dir, candidates[0])
                elif len(candidates) > 1:
                    candidates.sort()
                    motion_tsv_path = os.path.join(func_dir, candidates[0])
                    print(
                        f"[{sub_id}] 找到多个 motion 文件, 使用 {candidates[0]}: {candidates}"
                    )
                else:
                    print(
                        f"[{sub_id}] func 目录下未找到 *_desc-summary_motion.tsv: {func_dir}"
                    )
            except Exception as e:
                print(f"[{sub_id}] 搜索 motion 文件时出错: {e}")
    else:
        print(f"[{sub_id}] 未找到 func 目录: {func_dir}")

    mean_fd = None
    mean_dvars = None
    bold_snr = None

    # 读取 motion.tsv
    if motion_tsv_path and os.path.exists(motion_tsv_path):
        try:
            motion_df = pd.read_csv(motion_tsv_path, sep="\t")
            if "MeanFD_Power" in motion_df.columns:
                mean_fd = motion_df["MeanFD_Power"].iloc[0]
            if "MeanDVARS" in motion_df.columns:
                mean_dvars = motion_df["MeanDVARS"].iloc[0]
        except Exception as e:
            print(f"[{sub_id}] 读取 {motion_tsv_path} 失败: {e}")

    # 查找 boldSnr txt
    if os.path.isdir(func_dir):
        try:
            snr_file_found = False
            for filename in os.listdir(func_dir):
                if "boldSnr_quality" in filename and filename.endswith(".txt"):
                    snr_txt_path = os.path.join(func_dir, filename)
                    with open(snr_txt_path, "r") as f:
                        content = f.read().strip()
                    try:
                        bold_snr = float(content)
                    except ValueError:
                        print(
                            f"[{sub_id}] 无法解析 boldSnr 数值: {snr_txt_path} 内容='{content}'"
                        )
                    snr_file_found = True
                    break
            if not snr_file_found:
                print(f"[{sub_id}] func 目录下未找到 boldSnr_quality*.txt: {func_dir}")
        except Exception as e:
            print(f"[{sub_id}] 处理 SNR 文件出错: {e}")
    else:
        print(f"[{sub_id}] 未找到 func 目录: {func_dir}")

    return {
        "SubjectID": sub_id,
        "MeanFD_Power": mean_fd,
        "MeanDVARS": mean_dvars,
        "boldSnr": bold_snr,
    }


def main():
    base_output_dir = BASE_OUTPUT_DIR
    output_csv_path = OUTPUT_CSV_PATH

    print(f"Base directory: {base_output_dir}")
    print(f"Output CSV   : {output_csv_path}")

    subjects = find_subjects(base_output_dir)
    print(f"发现被试数量: {len(subjects)}")
    if not subjects:
        print("未发现任何 sub-* 目录，请检查 BASE_OUTPUT_DIR 是否正确。")
        return

    all_subjects_data = []

    for sub_id in subjects:
        print(f"处理: {sub_id}")
        metrics = extract_metrics_for_subject(
            base_output_dir=base_output_dir,
            sub_id=sub_id,
            pipeline_name=PIPELINE_NAME,
            session=SESSION,
            task=TASK,
        )
        all_subjects_data.append(metrics)

    results_df = pd.DataFrame(all_subjects_data)
    results_df.to_csv(output_csv_path, index=False)

    print("\n--- 提取完成 ---")
    print(results_df)
    print(f"\n结果已保存到: {output_csv_path}")


if __name__ == "__main__":
    main()
