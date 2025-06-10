import os
import sys
# import pexpect
import subprocess
import shutil
import glob
import logging
from datetime import datetime
import json
# import image_description

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/a001/zhangyan/cpac/log/cpac_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('cpac_pipeline')


# 生成批处理的参与者 ID
def generate_participant_batches(start=1, end=1700, batch_size=50):
    batches = []
    for i in range(start, end + 1, batch_size):
        batch = [f"sub-{str(j).zfill(8)}" for j in range(i, min(i + batch_size, end + 1))]
        batches.append(batch)
    return batches


# 根据participant_id插入script之后运行cpac_script
def run_cpac_script(script_path, dataset_id, participant_id):
    output_dir = f"/home/a001/zhangyan/cpac/output/{dataset_id}/{participant_id}"
    os.makedirs(output_dir, exist_ok=True)
    timing_file = os.path.join(output_dir, "cpac_timing.txt")
    start_time = datetime.now()
    
    logger.info(f"Starting CPAC processing for participant {participant_id}")
    logger.info(f"Using script: {script_path}")
    
    with open(timing_file, "w") as f:
        f.write(f"Start time: {start_time}\n")

    try:
        with subprocess.Popen(script_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            # Create log files for stdout and stderr
            stdout_file = os.path.join(output_dir, "stdout.log")
            stderr_file = os.path.join(output_dir, "stderr.log")
            
            with open(stdout_file, 'w') as out_f, open(stderr_file, 'w') as err_f:
                for line in proc.stdout:
                    print(line, end='')
                    out_f.write(line)
                    logger.debug(line.strip())
                for line in proc.stderr:
                    print(line, end='', file=sys.stderr)
                    err_f.write(line)
                    logger.error(line.strip())
            
            proc.wait()

            if proc.returncode != 0:
                logger.error(f"CPAC processing failed for participant {participant_id} with return code {proc.returncode}")
                raise subprocess.CalledProcessError(proc.returncode, script_path)
            
        logger.info(f"CPAC processing completed successfully for participant {participant_id}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: CPAC processing failed for participant {participant_id} with return code {e.returncode}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred for participant {participant_id}: {str(e)}")
        raise
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        with open(timing_file, "a") as f:
            f.write(f"End time: {end_time}\n")
            f.write(f"Duration: {duration}\n")
        
        logger.info(f"Processing time for participant {participant_id}: {duration}")

    return output_dir


def create_cpac_script(bids_root,dataset_id,participant_id, config_path):
    """Create CPAC processing script for a participant
    
    Args:
        participant_id: 参与者ID
        config_path: YAML配置文件的完整路径
    """
    try:
        # 定义输出目录和脚本路径
        output_dir = f"/home/a001/zhangyan/cpac/output/{dataset_id}/{participant_id}"
        script_path = os.path.join(output_dir, "cpac.sh")
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 获取配置文件的目录和文件名
        pipeline_config_dir = os.path.dirname(config_path)
        config_filename = os.path.basename(config_path)
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            logger.error(f"Pipeline config file not found: {config_path}")
            return None
        
        # 生成脚本内容
        script_content = f"""#!/bin/bash

docker run -i --rm \\
-v {bids_root}:/bids_dataset:ro \\
-v {output_dir}:/outputs \\
-v /home/a001/zhangyan/cpac/temp:/scratch \\
-v {pipeline_config_dir}:/pipeline_config \\
fcpindi/c-pac:latest /bids_dataset /outputs participant \\
--participant_label {participant_id.replace('sub-', '')} \\
--pipeline_file /pipeline_config/{config_filename} \\
--save_working_dir
"""

        # 将脚本内容写入文件
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 设置脚本为可执行
        os.chmod(script_path, 0o755)
        
        # 记录日志
        logger.info(f"Script file created at {script_path}")
        logger.info(f"Using config file: {config_path}")
        return script_path

    except Exception as e:
        # 记录错误日志
        logger.error(f"Error creating CPAC script: {str(e)}")
        return None

def generate_config_id(brain_extraction, slice_timing_correction, motion_correction, coregistration, boundary_based_registration):
    # 字符串映射字典
    mapping = {
        '3dSkullStrip': '1',
        'BET': '2',
        'On': '1',
        'Off': '2',
        '3dvolreg': '1',
        'mcflirt': '2',
        'FSL': '1',
        'ABCD': '2'
    }
    
    # 生成 config_id
    config_id = (
        mapping[brain_extraction] +
        mapping[slice_timing_correction] +
        mapping[motion_correction] +
        mapping[coregistration] +
        mapping[boundary_based_registration]
    )
    
    return config_id

def copy_feature_file(participant_id):
     # 定义文件搜索路径和目标路径
    search_path = f"/home/user/zhangyan/cpac/outputs/{participant_id}/output/pipeline_cpac-default-pipeline/{participant_id}/ses-001/func"
    target_path = f"/home/user/zhangyan/cpac/outputs/{participant_id}"
    
    # 检查并遍历搜索路径中的文件
    if not os.path.exists(search_path):
        logger.info(f"Search path {search_path} does not exist.")
        return
    
    # 定义要查找的文件模式
    file_patterns = [
        {"pattern": "PearsonNilearn_correlations", "extension": ".tsv"},
        {"pattern": "task-rest_space-MNI152NLin6ASym_reg-default_desc-smZstd_alff", "extension": ".gz"},
        {"pattern": "task-rest_space-MNI152NLin6ASym_reg-default_desc-smZstd_falff", "extension": ".gz"},
        {"pattern": "task-rest_space-MNI152NLin6ASym_reg-default_desc-smZstd_reho", "extension": ".gz"}
    ]
    
    found_files = []
    for root, dirs, files in os.walk(search_path):
        for file in files:
            for file_pattern in file_patterns:
                if file_pattern["pattern"] in file and file.endswith(file_pattern["extension"]):
                    found_files.append(os.path.join(root, file))
                    break  # 一旦找到匹配的文件，就不再继续检查其他模式
    
    if not found_files:
        logger.info("No matching files found in the directory.")
        return
    
    # 确保目标路径存在
    os.makedirs(target_path, exist_ok=True)

    # 复制找到的文件到目标路径
    for file in found_files:
        shutil.copy(file, target_path)
        logger.info(f"Copied {file} to {target_path}")

def remove_working_directory(participant_id):
    """
    删除指定参与者ID的working目录
    Args:
        participant_id: 参与者ID
    """
    working_dir = f"/home/user/zhangyan/cpac/outputs/{participant_id}/working"
    if os.path.exists(working_dir):
        try:
            logger.info(f"Removing working directory for {participant_id}...")
            shutil.rmtree(working_dir)
            logger.info(f"Successfully removed working directory for {participant_id}")
        except Exception as e:
            logger.error(f"Error removing working directory for {participant_id}: {str(e)}")
    else:
        logger.info(f"Working directory for {participant_id} does not exist")

def read_ids_from_file(file_path):
    """
    从文件中读取ID列表
    Args:
        file_path: ID文件的路径
    Returns:
        list: ID列表
    """
    if not os.path.exists(file_path):
        logger.info(f"Warning: File {file_path} does not exist")
        return []
    
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def update_processed_ids(processed_file, participant_id, status="completed"):
    """更新已处理的参与者ID列表
    
    Args:
        processed_file: 已处理ID文件路径
        participant_id: 参与者ID
        status: 处理状态，可以是 'completed' 或 'invalid_data'
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        
        # 读取现有记录
        processed_data = {}
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        pid, pstatus = line.strip().split(':')
                        processed_data[pid] = pstatus
                    else:
                        processed_data[line.strip()] = 'completed'  # 兼容旧格式
        
        # 更新记录
        processed_data[participant_id] = status
        
        # 写入更新后的记录
        with open(processed_file, 'w') as f:
            for pid, pstatus in processed_data.items():
                f.write(f"{pid}:{pstatus}\n")
                
    except Exception as e:
        logger.error(f"更新已处理ID文件时出错: {str(e)}")

def get_next_batch(to_process_file, processed_file, batch_size=2):
    """
    获取下一批要处理的ID
    Args:
        to_process_file: 待处理ID文件路径
        processed_file: 已处理ID文件路径
        batch_size: 每批处理的数量
    Returns:
        list: 下一批要处理的ID列表
    """
    # 读取待处理ID
    to_process_ids = []
    if os.path.exists(to_process_file):
        with open(to_process_file, 'r') as f:
            to_process_ids = f.read().splitlines()
    
    # 读取已处理ID
    processed_ids = set()
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed_ids = set(f.read().splitlines())
    
    # 找出未处理的ID
    unprocessed_ids = [id for id in to_process_ids if id not in processed_ids]
    
    # 返回下一批要处理的ID
    next_batch = unprocessed_ids[:batch_size]
    logger.info(f"Selected next batch of {len(next_batch)} IDs to process")
    return next_batch

def initialize_id_files(source_id_file, to_process_file, processed_file):
    """
    初始化待处理和已处理ID文件
    Args:
        source_id_file: 源ID文件路径
        to_process_file: 待处理ID文件路径
        processed_file: 已处理ID文件路径
    """
    # 读取源ID文件
    all_ids = read_ids_from_file(source_id_file)
    
    # 读取已处理的ID（如果文件存在）
    processed_ids = set(read_ids_from_file(processed_file))
    
    # 确定待处理的ID
    to_process_ids = [id for id in all_ids if id not in processed_ids]
    
    # 写入待处理ID文件
    with open(to_process_file, 'w') as f:
        for id in to_process_ids:
            f.write(f"{id}\n")
    
    logger.info(f"Initialized ID files:")
    logger.info(f"Total IDs: {len(all_ids)}")
    logger.info(f"Already processed: {len(processed_ids)}")
    logger.info(f"To process: {len(to_process_ids)}")
    
    return to_process_ids

def prepare_bids_data(participant_id):
    """Prepare BIDS format data for a participant"""
    try:
        logger.info(f"Preparing BIDS data for {participant_id}")
        
        # Base paths
        bids_root = "/home/user/zhangyan/dpabi/data/shenyang"
        participant_dir = os.path.join(bids_root, participant_id)
        func_dir = os.path.join(participant_dir, "func")
        anat_dir = os.path.join(participant_dir, "anat")
        
        # Create directories if they don't exist
        os.makedirs(func_dir, exist_ok=True)
        os.makedirs(anat_dir, exist_ok=True)

        # Prepare dataset_description.json
        dataset_desc = {
            "Name": "CPAC BIDS Dataset",
            "BIDSVersion": "1.4.0",
            "DatasetType": "raw",
            "Authors": ["CPAC Team"]
        }
        with open(os.path.join(bids_root, "dataset_description.json"), "w") as f:
            json.dump(dataset_desc, f, indent=4)

        # Create participants.tsv if it doesn't exist
        participants_file = os.path.join(bids_root, "participants.tsv")
        if not os.path.exists(participants_file):
            with open(participants_file, "w") as f:
                f.write("participant_id\tsex\tage\n")
                f.write(f"{participant_id}\tM\t25\n")

        # Find and rename functional and anatomical files
        source_dir = os.path.join(bids_root, participant_id)
        if not os.path.exists(source_dir):
            logger.error(f"Source directory not found: {source_dir}")
            return False

        # Handle functional data
        func_source_dir = os.path.join(source_dir, "func")
        func_files = glob.glob(os.path.join(func_source_dir, "*bold.nii*"))
        if not func_files:
            logger.error(f"No functional files found in {func_source_dir}")
            return False

        for func_file in func_files:
            # New BIDS compliant filename with task-rest
            new_name = os.path.join(
                func_dir,
                f"{participant_id}_task-rest_bold.nii.gz"
            )
            
            # If file is already in the correct location with correct name, skip copying
            if os.path.abspath(func_file) == os.path.abspath(new_name):
                logger.info(f"Functional file already exists at correct location: {new_name}")
            else:
                # Copy file only if source and destination are different
                if os.path.exists(func_file):
                    shutil.copy2(func_file, new_name)
                    logger.info(f"Copied functional file from {func_file} to {new_name}")
                    
                    # Create corresponding JSON sidecar
                    json_sidecar = {
                        "TaskName": "rest",
                        "RepetitionTime": 2.0,
                        "EchoTime": 0.03,
                        "FlipAngle": 90,
                        "PhaseEncodingDirection": "j-",
                        "EffectiveEchoSpacing": 0.0004,
                        "SliceTiming": [0.0, 1.0],
                        "TaskDescription": "resting state scan",
                        "Manufacturer": "Siemens",
                        "ManufacturersModelName": "TrioTim",
                        "MagneticFieldStrength": 3,
                        "ScanningSequence": "EP"
                    }
                    with open(new_name.replace(".nii.gz", ".json"), "w") as f:
                        json.dump(json_sidecar, f, indent=4)

        # Handle anatomical data
        anat_source_dir = os.path.join(source_dir, "anat")
        anat_files = glob.glob(os.path.join(anat_source_dir, "*T1w.nii*"))
        if not anat_files:
            logger.error(f"No T1w files found in {anat_source_dir}")
            return False
            
        for anat_file in anat_files:
            # New BIDS compliant filename
            new_name = os.path.join(
                anat_dir,
                f"{participant_id}_T1w.nii.gz"
            )
            
            # If file is already in the correct location with correct name, skip copying
            if os.path.abspath(anat_file) == os.path.abspath(new_name):
                logger.info(f"Anatomical file already exists at correct location: {new_name}")
            else:
                # Copy file only if source and destination are different
                if os.path.exists(anat_file):
                    shutil.copy2(anat_file, new_name)
                    logger.info(f"Copied anatomical file from {anat_file} to {new_name}")
                    
                    # Create corresponding JSON sidecar
                    json_sidecar = {
                        "Type": "T1w",
                        "ScanningSequence": "MPRAGE",
                        "EchoTime": 0.00293,
                        "RepetitionTime": 2.3,
                        "FlipAngle": 8,
                        "Manufacturer": "Siemens",
                        "ManufacturersModelName": "TrioTim",
                        "MagneticFieldStrength": 3
                    }
                    with open(new_name.replace(".nii.gz", ".json"), "w") as f:
                        json.dump(json_sidecar, f, indent=4)

        logger.info(f"Successfully prepared BIDS data for {participant_id}")
        return True

    except Exception as e:
        logger.error(f"Error preparing BIDS data for {participant_id}: {str(e)}")
        return False

def check_bids_data(bids_path):
    """检查BIDS数据目录是否有效，支持有会话(ses-*)和无会话两种组织形式
    
    Args:
        bids_path: BIDS数据目录路径（被试级别，如sub-01）
        
    Returns:
        tuple: (是否有效, 错误信息)
            - 如果目录有效，返回 (True, None)
            - 如果目录无效，返回 (False, 错误信息)
    """
    import os
    
    try:
        # 检查是否存在会话目录
        session_dirs = [d for d in os.listdir(bids_path) if d.startswith('ses-') and os.path.isdir(os.path.join(bids_path, d))]
        
        # 判断是否是有会话的组织形式
        if session_dirs:
            # 有会话目录的情况
            found_anat = False
            found_func = False
            
            for session in session_dirs:
                session_path = os.path.join(bids_path, session)
                
                # 检查每个会话目录中的anat和func
                for dir_name in ['anat', 'func']:
                    dir_path = os.path.join(session_path, dir_name)
                    
                    # 检查目录是否存在
                    if not os.path.exists(dir_path):
                        continue  # 不是所有会话都需要所有模态
                    
                    # 检查目录是否为空（不包括隐藏文件）
                    has_files = False
                    for item in os.listdir(dir_path):
                        if not item.startswith('.'):  # 忽略隐藏文件
                            item_path = os.path.join(dir_path, item)
                            if os.path.isfile(item_path):
                                has_files = True
                                break
                    
                    if not has_files:
                        return False, f"目录为空: {os.path.join(session, dir_name)}"
                    
                    # 记录找到的模态
                    if dir_name == 'anat':
                        found_anat = True
                    elif dir_name == 'func':
                        found_func = True
            
            # 确保至少在一个会话中找到所需的模态
            if not found_anat:
                return False, "在所有会话中均未找到anat目录"
            if not found_func:
                return False, "在所有会话中均未找到func目录"
                
        else:
            # 无会话目录的情况，直接检查被试级别的anat和func目录
            found_anat = False
            found_func = False
            
            for dir_name in ['anat', 'func']:
                dir_path = os.path.join(bids_path, dir_name)
                
                # 检查目录是否存在
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    # 检查目录是否为空（不包括隐藏文件）
                    has_files = False
                    for item in os.listdir(dir_path):
                        if not item.startswith('.'):  # 忽略隐藏文件
                            item_path = os.path.join(dir_path, item)
                            if os.path.isfile(item_path):
                                has_files = True
                                break
                    
                    if has_files:
                        if dir_name == 'anat':
                            found_anat = True
                        elif dir_name == 'func':
                            found_func = True
                    else:
                        return False, f"目录为空: {dir_name}"
            
            # 确保找到了所需的模态
            if not found_anat:
                return False, "未找到anat目录或该目录为空"
            if not found_func:
                return False, "未找到func目录或该目录为空"
        
        return True, None
        
    except Exception as e:
        return False, f"检查BIDS数据时出错: {str(e)}"

def get_unprocessed_ids(processed_file, participant_file):
    """获取未处理的参与者ID列表
    
    Args:
        processed_file: 已处理ID文件路径
        participant_file: 所有参与者ID文件路径
        
    Returns:
        list: 未处理的参与者ID列表
    """
    try:
        # 读取所有参与者ID
        all_ids = set(read_ids_from_file(participant_file))
        
        # 读取已处理的ID
        processed_ids = set()
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        # 新格式：ID:状态
                        pid, status = line.strip().split(':')
                        # 只有完成处理的才算作已处理
                        if status == 'completed':
                            processed_ids.add(pid)
                    else:
                        # 旧格式：只有ID
                        processed_ids.add(line.strip())
        
        # 返回未处理的ID列表
        unprocessed_ids = list(all_ids - processed_ids)
        unprocessed_ids.sort()  # 保持顺序一致
        
        logger.info(f"Found {len(unprocessed_ids)} unprocessed participants")
        return unprocessed_ids
        
    except Exception as e:
        logger.error(f"获取未处理ID列表时出错: {str(e)}")
        return []
    

# Generate an HTML report
def generate_subject_report(dataset_id, participant_id):
    """
    Generate an HTML report displaying all PNG images for a specific subject.
    Images are embedded directly in the HTML using base64 encoding.
    The report is saved in the subject's output directory.
    
    Args:
        subject_id (str): Subject ID (e.g., 'sub-00011')
    """
    import base64
    # Define paths
    base_output_dir = f"/home/a001/zhangyan/cpac/output/{dataset_id}"
    subject_output_dir = os.path.join(base_output_dir, participant_id)
    pipeline_dir = os.path.join(subject_output_dir, "output", "pipeline_cpac-default-pipeline",
                             participant_id, "ses-1")
    
    # Create HTML content
    html_content = []
    html_content.append("<!DOCTYPE html>")
    html_content.append("<html>")
    html_content.append("<head>")
    html_content.append(f"<title>Image Report for {participant_id}</title>")
    html_content.append("""
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px 40px;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            text-align: center;
            padding: 20px;
            margin-bottom: 30px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            padding: 10px 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin: 20px 0;
        }
        .image-container {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .image-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .filename {
            font-family: 'Consolas', monospace;
            color: #2c3e50;
            margin: 10px 0;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 5px;
            font-size: 0.9em;
            word-break: break-all;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            display: block;
        }
        .section {
            margin-bottom: 40px;
        }
        .timestamp {
            color: #666;
            font-size: 0.9em;
            text-align: center;
            margin-top: 10px;
        }
                        
        .description {
        margin-top: 10px;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 5px;
        font-size: 0.9em;
        color: #444;
        line-height: 1.4;
        }

    </style>
    """)
    html_content.append("</head>")
    html_content.append("<body>")
    html_content.append("<div class='header'>")
    html_content.append(f"<h1>Image Report for {participant_id}</h1>")
    html_content.append(f"<p class='timestamp'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html_content.append("</div>")
    
    # Process anat and func directories
    for folder in ['anat', 'func']:
        folder_path = os.path.join(pipeline_dir, folder)
        if os.path.exists(folder_path):
            html_content.append("<div class='section'>")
            html_content.append(f"<h2>{folder.upper()} Images</h2>")
            html_content.append("<div class='image-grid'>")
            
            # Find all PNG files
            png_files = glob.glob(os.path.join(folder_path, "*.png"))
            
            # # 第一阶段：收集所有需要解释的文件名,批量获取所有图片解释（假设你的函数支持批量处理）
            # all_filenames = [os.path.basename(png_file) for png_file in sorted(png_files)]
            # descriptions_dict = image_description.Interpret_preprocessed_images(all_filenames)  # 返回 {filename: description} 的字典


            # 第二阶段：生成HTML内容
            for png_file in sorted(png_files):
                filename = os.path.basename(png_file)
                
                # Read and encode image
                with open(png_file, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                
                # 从预先生成的解释字典中获取描述
                # image_description = descriptions_dict.get(filename, "No description available")

                html_content.append("<div class='image-container'>")
                html_content.append(f"<p class='filename'>{filename}</p>")
                html_content.append(f"<img src='data:image/png;base64,{image_data}' alt='{filename}'>")
                
                # # Add description below the image
                # html_content.append("<div class='description'>")
                # html_content.append(f"<p>{image_description}</p>")
                # html_content.append("</div>")
                
                html_content.append("</div>")
            
            html_content.append("</div>")  # Close image-grid
            html_content.append("</div>")  # Close section
    
    html_content.append("</body>")
    html_content.append("</html>")
    
    # Write HTML file in the subject's output directory
    output_html = os.path.join(subject_output_dir, f"{participant_id}_report.html")
    with open(output_html, 'w') as f:
        f.write("\n".join(html_content))
    
    logger.info(f"Generated image report: {output_html}")
    return output_html