import os
import sys
import json
import logging
import Tools
import glob
from datetime import datetime
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, '..', 'configs', 'config.yaml')

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# # Import intelligent parameter configuration
# sys.path.append('/home/user/zhangyan/cpac/workflow/intelliparam')
# from param_advisor import get_parameter_suggestions, update_yml_config, generate_summary



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

def load_research_questions():
    """从JSON文件加载研究问题"""
    try:
        with open("/home/user/zhangyan/cpac/workflow/intelliparam/research_question.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载研究问题文件时出错: {e}")
        return None

def load_default_config():
    """从JSON文件加载默认配置"""
    try:
        with open("/home/user/zhangyan/cpac/workflow/intelliparam/default_config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载默认配置文件时出错: {e}")
        return None

def main():
    try:
        # 从配置文件读取参数
        cfg = load_config()
        
        # 配置参数
        dataset_id = cfg['dataset']['id']   # 数据集ID
        batch_size = cfg['batch']['size']   # 批处理大小
        bids_root = cfg['dataset']['bids_root']  # BIDS根目录
        
        # 获取智能参数配置文件路径
        config_path = cfg['paths']['cpac_config']
        # config_path = get_intelligent_config(dataset_id)
        if not config_path:
            logger.error("无法获取预处理参数配置")
            return

        # 配置批处理参数
        config = {
            "batch_size": batch_size,
            "bids_root": bids_root, 
            "source_id_file": cfg['paths']['source_id_file'],
            "to_process_file": cfg['paths']['to_process_file'],
            "processed_file": cfg['paths']['processed_file']
        }
        
        logger.info("配置信息:")
        logger.info(f"配置文件路径: {config_path}")
        logger.info(f"批处理大小: {config['batch_size']}")
        logger.info(f"BIDS根目录: {config['bids_root']}")

        # Get next batch of IDs to process
        batch = Tools.get_next_batch(
            config["to_process_file"],
            config["processed_file"],
            config["batch_size"]
        )
        
        if not batch:
            logger.info("No more IDs to process.")
            return

        logger.info(f"Processing batch of {len(batch)} participants: {batch}")

        # Process each participant
        for participant_id in batch:
            try:
                bids_path = os.path.join(config["bids_root"], participant_id)
                logger.info(f"Processing participant {participant_id}")
                logger.info(f"bids_root: {bids_path}")

                if not os.path.exists(bids_path):
                    logger.warning(f"Path not found for participant {participant_id}: {bids_path}")
                    continue 
                
                # 检查 BIDS 数据目录是否有效
                print("---------检查 BIDS 数据目录是否有效---------")
                is_valid, error_msg = Tools.check_bids_data(bids_path)
                if not is_valid:
                    logger.error(f"参与者 {participant_id} 的数据无效: {error_msg}")
                    # 记录到已处理文件中，但标记为无效数据
                    Tools.update_processed_ids(config["processed_file"], participant_id, status="invalid_data")
                    continue
                
                # Create and run CPAC script
                script_path = Tools.create_cpac_script(bids_root,dataset_id,participant_id, config_path)
                if not script_path:
                    logger.error(f"Failed to create CPAC script for {participant_id}")
                    continue
                    
                logger.info(f"script_path: {script_path}")
                

                # 运行生成的CPAC脚本
                output_dir = Tools.run_cpac_script(script_path, dataset_id, participant_id)
                
                # Generate image report for the processed subject
                try:
                    report_path = Tools.generate_subject_report(dataset_id, participant_id)
                    logger.info(f"Generated image report for {participant_id}: {report_path}")
                except Exception as e:
                    logger.error(f"Error generating image report for {participant_id}: {str(e)}")
                
                # Update processed IDs file
                Tools.update_processed_ids(config["processed_file"], participant_id)
                logger.info(f"Successfully completed processing for {participant_id}")
                logger.info(f"participant results in {output_dir}")

            except Exception as e:
                logger.error(f"Error processing participant {participant_id}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Critical error in main workflow: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    print(f"日志文件已保存到: /home/a001/zhangyan/cpac/log/cpac_pipeline.log")