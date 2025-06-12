import requests
import json
import os
from pathlib import Path
from ollama_client import OllamaClient
from openai_api import OpenAIClient

class FMRIConfigGenerator:
    """
    fMRI配置参数生成器
    """
    def __init__(self, client):
        self.client = client
        
    def load_template_and_dataset(self, template_file="surface_analysis.txt", dataset_file="dataset_description.txt"):
        """
        加载模板和数据集描述
        """
        try:
            # 读取模板文件
            if os.path.exists(template_file):
                with open(template_file, 'r', encoding='utf-8') as f:
                    template = f.read()
            else:
                # 如果文件不存在
                print(f"模板文件不存在")
                return None
            
            # 读取数据集描述
            dataset_description = ""
            if os.path.exists(dataset_file):
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    dataset_description = f.read()
            else:
                print(f"警告: 未找到数据集描述文件 {dataset_file}")
                dataset_description = "No dataset description provided."
            
            # 将数据集描述插入模板
            prompt = template.replace("{PASTE THE CONTENT OF YOUR dataset_description.txt HERE}", dataset_description)
            
            return prompt
            
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return None

    def extract_json_from_response(self, response_text):
        """
        从响应中提取JSON对象
        """
        try:
            # 尝试直接解析整个响应
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 如果失败，尝试找到JSON部分
            start_markers = ['{', '```json\n{', '```\n{']
            end_markers = ['}', '}\n```', '}\n```']
            
            for start_marker, end_marker in zip(start_markers, end_markers):
                start_idx = response_text.find(start_marker)
                if start_idx != -1:
                    end_idx = response_text.rfind(end_marker)
                    if end_idx != -1:
                        json_str = response_text[start_idx:end_idx + len(end_marker.split('\n')[0])]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            continue
            
            # 如果还是失败，尝试更灵活的方式
            lines = response_text.split('\n')
            json_lines = []
            in_json = False
            brace_count = 0
            
            for line in lines:
                if '{' in line and not in_json:
                    in_json = True
                    brace_count += line.count('{') - line.count('}')
                    json_lines.append(line)
                elif in_json:
                    brace_count += line.count('{') - line.count('}')
                    json_lines.append(line)
                    if brace_count == 0:
                        break
            
            if json_lines:
                json_str = '\n'.join(json_lines)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            print("无法从响应中提取有效的JSON")
            return None
    
    def generate_config(self, model_name, template_file, dataset_file, output_file="fmri_config.json", temperature=0.3, max_tokens=2000):
        """
        生成fMRI配置参数，使用OpenAI API
        
        Args:
            model_name: OpenAI模型名称，默认gpt-4o
            template_file: 模板文件路径
            dataset_file: 数据集描述文件路径
            output_file: 输出配置文件路径
            temperature: 生成温度
            max_tokens: 最大token数
        """
        print("正在加载模板和数据集描述...")
        prompt = self.load_template_and_dataset(template_file, dataset_file)
        
        if prompt is None:
            print("加载模板失败")
            return None
        
        # 优化提示词长度
        print(f"原始提示词长度: {len(prompt)} 字符")
        print(f"正在使用模型 {model_name} 生成配置...")
        
        # 构建消息
        messages = [
            {
                "role": "system", 
                "content": "你是一个专业的fMRI数据分析专家，能够根据模板和数据集描述生成准确的配置参数。请返回标准的JSON格式配置。"
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        try:
            # 调用OpenAI API
            response_text = self.client.chat(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            print(f"\n模型响应长度: {len(response_text)} 字符")
            
            # 提取JSON配置
            config_json = self.extract_json_from_response(response_text)
            
            if config_json is None:
                print("无法解析模型响应为JSON格式")
                return None
    
            # 保存配置到文件
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(config_json, f, indent=2, ensure_ascii=False)
                print(f"配置已保存到 {output_file}")
            except Exception as e:
                print(f"保存配置文件时出错: {e}")
            
            return config_json
            
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return None

def get_file_list(directory, extension='.txt'):
    """
    获取指定目录下的所有指定扩展名文件
    
    Args:
        directory (str): 目录路径
        extension (str): 文件扩展名，默认.txt
        
    Returns:
        list: 文件路径列表
    """
    try:
        path = Path(directory)
        if not path.exists():
            print(f"目录不存在: {directory}")
            return []
        
        files = list(path.glob(f'*{extension}'))
        return sorted(files)
    except Exception as e:
        print(f"读取目录失败 {directory}: {e}")
        return []


def create_output_filename(dataset_file, parameter_file, output_dir):
    """
    创建输出文件名
    
    Args:
        dataset_file (Path): 数据集文件路径
        parameter_file (Path): 参数文件路径
        output_dir (str): 输出目录
        
    Returns:
        str: 输出文件完整路径
    """
    dataset_name = dataset_file.stem  # 不包含扩展名的文件名
    parameter_name = parameter_file.stem
    
    output_filename = f"{dataset_name}_{parameter_name}.json"
    return os.path.join(output_dir, output_filename)



def main():
    """
    主函数
    """
    # ----------------调用本地ollama-----------------------------
    # # 创建Ollama客户端
    # client = OllamaClient()
    
    # # 检查模型是否可用
    # models = client.list_models()
    # available_models = [model['name'] for model in models.get('models', [])]
    
    # print("可用的模型:")
    # for model in available_models:
    #     print(f"- {model}")
    
    # model_name = "gemma3:27b"
    # if model_name not in available_models:
    #     print(f"警告: 模型 {model_name} 不在可用模型列表中")
    #     if available_models:
    #         print("请确保模型已正确安装，或选择其他可用模型")
    #     else:
    #         print("未检测到任何可用模型，请检查Ollama服务是否正常运行")
    #     return
    
    # # 创建配置生成器
    # generator = FMRIConfigGenerator(client)
    
    # # 生成配置
    # template_file = "/home/a001/zhangyan/cpac/llm_parameters/parameters/surface_analysis.txt"
    # dataset_file = "/home/a001/zhangyan/cpac/llm_parameters/dataset_description.txt"
    # output_file = "/home/a001/zhangyan/cpac/llm_parameters/fmri_config.json"
    # config = generator.generate_config(
    #     model_name=model_name,
    #     template_file=template_file,
    #     dataset_file=dataset_file,
    #     output_file=output_file,
    #     use_stream=False
    # )
    
    # if config:
    #     print("\n生成的配置:")
    #     print(json.dumps(config, indent=2, ensure_ascii=False))
    # else:
    #     print("配置生成失败")


    """
    批量处理主函数
    """
    # 创建OpenAI客户端
    OPENAI_API_KEY = "xxx"
    OPENAI_BASE_URL = "xxx"
    
    try:
        client = OpenAIClient(
            api_key=OPENAI_API_KEY,  
            base_url=OPENAI_BASE_URL        
        )
        print("OpenAI客户端初始化成功")
    except Exception as e:
        print(f"OpenAI客户端初始化失败: {e}")
        return
    
    # 设置路径
    parameters_dir = "/home/a001/zhangyan/cpac/llm_parameters/parameters"
    dataset_dir = "/home/a001/zhangyan/cpac/llm_parameters/dataset_description"
    output_dir = "/home/a001/zhangyan/cpac/llm_parameters/dataset_parameters"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有参数文件和数据集文件
    parameter_files = get_file_list(parameters_dir)
    dataset_files = get_file_list(dataset_dir)
    
    print(f"发现 {len(parameter_files)} 个参数文件:")
    for pf in parameter_files:
        print(f"  - {pf.name}")
    
    print(f"\n发现 {len(dataset_files)} 个数据集文件:")
    for df in dataset_files:
        print(f"  - {df.name}")
    
    if not parameter_files:
        print("未找到参数文件！")
        return
    
    if not dataset_files:
        print("未找到数据集文件！")
        return
    
    # 选择模型
    model_name = "Qwen/Qwen3-32B"
    print(f"\n使用模型: {model_name}")
    
    # 创建配置生成器
    generator = FMRIConfigGenerator(client)
    
    # 批量处理
    total_tasks = len(parameter_files) * len(dataset_files)
    current_task = 0
    success_count = 0
    failed_tasks = []
    
    print(f"\n开始批量处理，总共 {total_tasks} 个任务...")
    print("=" * 60)
    
    for dataset_file in dataset_files:
        for parameter_file in parameter_files:
            current_task += 1
            
            # 创建输出文件名
            output_file = create_output_filename(dataset_file, parameter_file, output_dir)
            
            print(f"\n[{current_task}/{total_tasks}] 处理任务:")
            print(f"  数据集: {dataset_file.name}")
            print(f"  参数模块: {parameter_file.name}")
            print(f"  输出文件: {os.path.basename(output_file)}")
            
            # 检查文件是否已存在
            if os.path.exists(output_file):
                print(f"  ⚠️  文件已存在，跳过: {output_file}")
                continue
            
            # 生成配置
            try:
                config = generator.generate_config(
                    model_name=model_name,
                    template_file=str(parameter_file),
                    dataset_file=str(dataset_file),
                    output_file=output_file,
                    temperature=0.3,
                    max_tokens=2000
                )
                
                if config:
                    success_count += 1
                    print(f"  ✓ 成功生成配置")
                else:
                    failed_tasks.append({
                        'dataset': dataset_file.name,
                        'parameter': parameter_file.name,
                        'output': output_file,
                        'reason': '配置生成失败'
                    })
                    print(f"  ✗ 配置生成失败")
                    
            except Exception as e:
                failed_tasks.append({
                    'dataset': dataset_file.name,
                    'parameter': parameter_file.name,
                    'output': output_file,
                    'reason': str(e)
                })
                print(f"  ✗ 处理出错: {e}")
    
    # 输出总结
    print("\n" + "=" * 60)
    print("批量处理完成！")
    print(f"总任务数: {total_tasks}")
    print(f"成功完成: {success_count}")
    print(f"失败任务: {len(failed_tasks)}")
    
    if failed_tasks:
        print("\n失败任务列表:")
        for i, task in enumerate(failed_tasks, 1):
            print(f"{i}. {task['dataset']} + {task['parameter']}")
            print(f"   原因: {task['reason']}")
    
    print(f"\n所有生成的配置文件保存在: {output_dir}")


def test_single_combination():
    """
    测试单个组合的函数（调试用）
    """
    OPENAI_API_KEY = "xxx"
    OPENAI_BASE_URL = "xxx"
    
    client = OpenAIClient(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    generator = FMRIConfigGenerator(client)
    
    # 测试单个组合
    dataset_file = "/home/a001/zhangyan/cpac/llm_parameters/dataset_description/your_dataset.txt"
    parameter_file = "/home/a001/zhangyan/cpac/llm_parameters/parameters/surface_analysis.txt"
    output_file = "/home/a001/zhangyan/cpac/llm_parameters/dataset_parameters/test_output.json"
    
    config = generator.generate_config(
        model_name="Qwen/Qwen3-32B",
        template_file=parameter_file,
        dataset_file=dataset_file,
        output_file=output_file
    )
    
    if config:
        print("测试成功！")
        print(json.dumps(config, indent=2, ensure_ascii=False))
    else:
        print("测试失败！")


if __name__ == "__main__":
    main()
    
    # 如果需要测试单个组合，取消下面注释
    # test_single_combination()