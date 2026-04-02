#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepPrep 批处理脚本
支持双GPU并发处理，自动管理待处理/已处理列表
"""

import os
import sys
import time
import subprocess
import threading
from queue import Queue
from datetime import datetime
from pathlib import Path
import yaml

from qc_utils import generate_qc_report, collect_html_reports

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ============================================================================
# 配置区域 - 在此处修改参数
# ============================================================================

cfg = load_config()

# 路径配置 (宿主机路径)
BIDS_DIR = cfg['deepprep']['bids_dir']
OUTPUT_DIR = cfg['deepprep']['output_dir']
WORK_ROOT = cfg['deepprep']['work_root']
FS_LICENSE = cfg['deepprep']['fs_license']

# Docker镜像配置
DOCKER_IMAGE = cfg['deepprep']['docker_image']

# GPU与资源配置
GPU_IDS = cfg['deepprep']['gpu_ids']              # 单GPU运行，避免显存竞争 (如测试通过可改为[0,1])
CPUS_PER_SUB = cfg['deepprep']['cpus_per_subject']          # 每个被试CPU数
MEM_PER_SUB = cfg['deepprep']['memory_per_subject_gb']           # 每个被试内存(GB)

# DeepPrep参数
TEMPLATE_SPACE = cfg['deepprep']['template_space']

# 列表文件
TO_PROCESS_FILE = cfg['deepprep']['to_process_file']
PROCESSED_FILE = cfg['deepprep']['processed_file']

# 日志与状态
LOG_DIR = cfg['deepprep']['log_dir']
STATUS_CSV = cfg['deepprep']['status_csv']
QC_CSV = cfg['deepprep']['qc_csv']

# ============================================================================
# 以下为脚本逻辑，一般无需修改
# ============================================================================


class GPUManager:
    """GPU资源队列管理器"""
    
    def __init__(self, gpu_ids: list):
        self.queue = Queue()
        for gpu_id in gpu_ids:
            self.queue.put(gpu_id)
        self.lock = threading.Lock()
    
    def acquire(self) -> int:
        """获取一个可用GPU，阻塞直到有可用"""
        return self.queue.get()
    
    def release(self, gpu_id: int):
        """释放GPU回队列"""
        self.queue.put(gpu_id)


def load_subject_list(filepath: str) -> list:
    """从文件加载被试ID列表，返回带sub-前缀的完整ID（如sub-01, sub-02）"""
    if not os.path.exists(filepath):
        return []
    
    subjects = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and line != '.':
                # 统一转换为带sub-前缀的格式
                if not line.startswith('sub-'):
                    line = f"sub-{line}"  # 添加 "sub-" 前缀
                subjects.append(line)
    return subjects


def save_subject_list(filepath: str, subjects: list):
    """保存被试ID列表到文件"""
    with open(filepath, 'w') as f:
        for sub in subjects:
            f.write(f"{sub}\n")


def append_to_processed(filepath: str, subject_id: str):
    """追加被试ID到已处理列表"""
    with open(filepath, 'a') as f:
        f.write(f"{subject_id}\n")


def build_deepprep_command(subject_id: str, gpu_id: int) -> list:
    """构建DeepPrep docker run命令，subject_id格式为sub-01"""
    work_dir = os.path.join(WORK_ROOT, subject_id)
    
    # 容器内路径
    container_bids = "/input"
    container_output = "/output"
    container_work = "/work"
    container_license = "/license/license.txt"
    
    # 获取license文件所在目录
    license_dir = os.path.dirname(FS_LICENSE)

    # 挂载 TemplateFlow 缓存
    templateflow_host = os.path.expanduser("~/.cache/templateflow")

    cmd = [
        "docker", "run", "--rm",
        # GPU配置
        "--gpus", f'device={gpu_id}',
        # TensorFlow显存限制：避免OOM
        "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true",
        "-e", "TF_GPU_ALLOCATOR=cuda_malloc_async",
        "-e", "TF_MEMORY_ALLOCATION=0.8",
        "-e", "TF_PER_PROCESS_GPU_MEMORY_FRACTION=0.8",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        # 资源限制：CPUs/内存
        "--cpus", str(CPUS_PER_SUB),
        "--memory", f"{MEM_PER_SUB}g",
        # 挂载卷
        "-v", f"{BIDS_DIR}:{container_bids}:ro",
        "-v", f"{OUTPUT_DIR}:{container_output}",
        "-v", f"{work_dir}:{container_work}",
        "-v", f"{license_dir}:/license:ro",
        "-v", f"{templateflow_host}:/home/deepprep/.cache/templateflow",  # 挂载 TemplateFlow 缓存
        # 镜像
        DOCKER_IMAGE,
        # DeepPrep参数
        container_bids,
        container_output,
        "participant",
        "--participant_label", subject_id,
        "--fs_license_file", container_license,
        "--work_dir", container_work,
        "--device", "auto",  # 使用 'auto' 自动检测GPU，避免硬编码
        "--skip_bids_validation",
        "--bold_task_type", "rest",  # 根据需要，可以更改任务类型
        "--bold_sdc", "on",  # 确保启用 SDC (如果需要)
        "--bold_confounds", "acompcor",  # 可根据需求调整
    ]
    
    return cmd



def run_subject(subject_id: str, gpu_id: int, log_dir: str) -> dict:
    """
    运行单个被试的DeepPrep处理
    
    Returns:
        dict: 包含subject, gpu_id, exit_code, runtime, start_time, end_time
    """
    start_time = datetime.now()
    
    # 创建日志文件
    log_file = os.path.join(log_dir, f"{subject_id}.log")
    
    cmd = build_deepprep_command(subject_id, gpu_id)
    cmd_str = ' '.join(cmd)
    
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] "
          f"开始处理 {subject_id} (GPU {gpu_id})")
    print(f"  命令: {cmd_str}")
    
    try:
        with open(log_file, 'w') as f:
            f.write(f"Command: {cmd_str}\n")
            f.write(f"Start: {start_time}\n")
            f.write("=" * 60 + "\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            exit_code = process.wait()
            
    except Exception as e:
        exit_code = -1
        with open(log_file, 'a') as f:
            f.write(f"\n[ERROR] 执行异常: {e}\n")
    
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    
    status = "成功" if exit_code == 0 else "失败"
    print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] "
          f"{subject_id} 处理{status} (耗时: {runtime/3600:.2f}h, 退出码: {exit_code})")
    
    return {
        'subject': subject_id,
        'gpu_id': gpu_id,
        'exit_code': exit_code,
        'runtime_sec': runtime,
        'runtime_hours': runtime / 3600,
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'log_file': log_file
    }


def worker(subject_id: str, gpu_manager: GPUManager, results: list, 
           results_lock: threading.Lock, log_dir: str, processed_file: str):
    """工作线程：获取GPU -> 处理被试 -> 释放GPU"""
    gpu_id = gpu_manager.acquire()
    
    try:
        result = run_subject(subject_id, gpu_id, log_dir)
        
        with results_lock:
            results.append(result)
        
        # 处理成功则追加到已处理列表
        if result['exit_code'] == 0:
            append_to_processed(processed_file, subject_id)
            
    finally:
        gpu_manager.release(gpu_id)


def save_status_csv(results: list, filepath: str):
    """保存处理状态到CSV"""
    import csv
    
    if not results:
        return
    
    fieldnames = ['subject', 'gpu_id', 'exit_code', 'runtime_hours', 
                  'start_time', 'end_time', 'log_file']
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    print(f"[INFO] 状态表已保存至: {filepath}")


def check_docker_gpu():
    """检查Docker GPU是否可用"""
    print("[INFO] 检查Docker GPU环境...")
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", 
             DOCKER_IMAGE, "nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print("[INFO] Docker GPU环境正常")
            return True
        else:
            print(f"[ERROR] Docker GPU检查失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Docker GPU检查异常: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("DeepPrep 批处理脚本")
    print("=" * 60)
    
    # 创建必要目录
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WORK_ROOT, exist_ok=True)
    
    # 检查Docker GPU环境
    if not check_docker_gpu():
        print("[ERROR] Docker GPU环境检查失败，请检查配置")
        sys.exit(1)
    
    # 加载待处理列表
    to_process = load_subject_list(TO_PROCESS_FILE)
    if not to_process:
        print(f"[ERROR] 待处理列表为空或文件不存在: {TO_PROCESS_FILE}")
        sys.exit(1)
    
    # 加载已处理列表
    processed = set(load_subject_list(PROCESSED_FILE))
    
    # 过滤掉已处理的被试
    pending = [s for s in to_process if s not in processed]
    
    print(f"\n[INFO] 配置信息:")
    print(f"  BIDS目录: {BIDS_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  工作目录: {WORK_ROOT}")
    print(f"  GPU列表: {GPU_IDS}")
    print(f"  每被试资源: {CPUS_PER_SUB} CPU, {MEM_PER_SUB} GB内存")
    print(f"\n[INFO] 被试信息:")
    print(f"  待处理列表总数: {len(to_process)}")
    print(f"  已处理数量: {len(processed)}")
    print(f"  本次待处理: {len(pending)}")
    
    if not pending:
        print("\n[INFO] 所有被试已处理完成！")
        return
    
    print(f"\n[INFO] 本次处理被试: {pending}")
    print("=" * 60)
    
    # 初始化GPU管理器
    gpu_manager = GPUManager(GPU_IDS)
    
    # 结果收集
    results = []
    results_lock = threading.Lock()
    
    # 启动工作线程
    threads = []
    for subject_id in pending:
        t = threading.Thread(
            target=worker,
            args=(subject_id, gpu_manager, results, results_lock, 
                  LOG_DIR, PROCESSED_FILE)
        )
        t.start()
        threads.append(t)
        # 稍微延迟启动，避免同时抢占资源
        time.sleep(2)
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    
    # 保存状态CSV
    save_status_csv(results, STATUS_CSV)
    
    # 统计结果
    success = [r for r in results if r['exit_code'] == 0]
    failed = [r for r in results if r['exit_code'] != 0]
    
    print(f"\n[统计]")
    print(f"  成功: {len(success)}")
    print(f"  失败: {len(failed)}")
    
    if failed:
        print(f"\n[失败被试]")
        for r in failed:
            print(f"  {r['subject']} (退出码: {r['exit_code']}, 日志: {r['log_file']})")
    
    # 生成QC报告（仅对成功的被试）
    if success:
        print(f"\n[INFO] 生成QC汇总报告...")
        success_ids = [r['subject'] for r in success]
        generate_qc_report(OUTPUT_DIR, success_ids, QC_CSV)
        
        # 收集HTML报告
        html_reports = collect_html_reports(OUTPUT_DIR, success_ids)
        print(f"\n[HTML报告]")
        for sub_id, path in html_reports.items():
            if path:
                print(f"  {sub_id}: {path}")


if __name__ == "__main__":
    main()
