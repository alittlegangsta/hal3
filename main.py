# main.py (最终可配置版)

import argparse
import sys
import os
import time

# 这个检查确保用户在正确的项目根目录下运行脚本
if not os.path.exists('src') or not os.path.exists('config.py'):
    print("错误: 未在当前目录下找到 'src' 文件夹或 'config.py' 文件。")
    print("请确保您在项目的根目录下运行此脚本，并且 config.py 也在根目录。")
    sys.exit(1)

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_config # 导入新的配置函数

def main():
    """项目主入口函数。"""
    parser = argparse.ArgumentParser(
        description="统一的深度学习声波测井分析项目入口。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- 新增 --array 参数 ---
    parser.add_argument(
        '--array',
        type=str,
        default='03',
        help="指定要使用的声波接收器阵列编号 (例如: '03', '07', '11')。"
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='可执行的命令')

    # 定义命令
    subparsers.add_parser('preprocess', help='阶段一: 将原始.mat文件处理成对齐的波形数据')
    subparsers.add_parser('cwt', help='阶段二: 对波形数据进行小波变换，生成尺度图')
    subparsers.add_parser('split', help='阶段三: 划分训练/验证集，并计算标准化统计数据')
    subparsers.add_parser('normalize', help='阶段四: 创建标准化的训练就绪数据文件')
    subparsers.add_parser('tfrecord', help='阶段五: 将数据转换为TFRecord格式')
    subparsers.add_parser('train', help='阶段六: 训练模型')
    subparsers.add_parser('analyze', help='阶段七: 评估模型性能并生成分析图')
    subparsers.add_parser('all', help='一键按顺序执行从 preprocess 到 analyze 的所有阶段')

    args = parser.parse_args()
    
    # --- 使用 array 参数生成配置 ---
    config = get_config(args.array)
    print_step_header(0, f"实验配置 (阵列: {args.array})")
    print(f"声波数据源: {config['paths']['sonic']}")
    print(f"处理后数据将保存至: {config['paths']['base_processed_dir']}")
    print(f"模型和图表将保存至: {os.path.dirname(config['paths']['plot_dir'])}")


    # 命令分发中心
    command_functions = {
        'preprocess': (run_preprocess_command, "数据预处理"),
        'cwt': (run_cwt_command, "小波变换"),
        'split': (run_split_command, "数据集划分"),
        'normalize': (run_normalize_command, "创建并标准化训练就绪数据"),
        'tfrecord': (run_tfrecord_command, "转换为TFRecord格式"),
        'train': (run_train_command, "模型训练"),
        'analyze': (run_analyze_command, "模型评估与分析"),
    }
    
    if args.command == 'all':
        run_all_command(config, command_functions)
    else:
        func, name = command_functions.get(args.command)
        if func:
            print_step_header(list(command_functions.keys()).index(args.command) + 1, name)
            func(config)

def run_preprocess_command(config):
    from src.data_processing.main_preprocess import run_stage1_preprocessing
    run_stage1_preprocessing(config)

def run_cwt_command(config):
    from src.cwt_transformation.main_transform import run_cwt_transformation
    run_cwt_transformation(config)

def run_split_command(config):
    from src.data_processing.main_preprocess import run_stage2_split_and_normalize
    run_stage2_split_and_normalize(config)

def run_normalize_command(config):
    from src.data_processing.main_preprocess import run_stage3_normalize_data
    run_stage3_normalize_data(config)
    
def run_tfrecord_command(config):
    from src.data_processing.create_tfrecords import run_tfrecord_conversion
    run_tfrecord_conversion(config)

def run_train_command(config):
    from src.modeling.train import train_model
    train_model(config)

def run_analyze_command(config):
    from src.interpretation.run_analysis import run_analysis
    run_analysis(config)

def run_all_command(config, commands):
    """按顺序执行完整的流水线"""
    print_step_header('ALL', "开始执行完整流水线")
    start_time = time.time()
    try:
        step = 1
        for command, (func, name) in commands.items():
            print_step_header(step, name)
            func(config)
            step += 1
    except Exception as e:
        print(f"\n流水线在步骤 '{name}' 执行出错: {e}")
        import traceback
        traceback.print_exc()
        print("请检查错误信息并修正后重试。")
        return
    end_time = time.time()
    print_pipeline_summary(start_time, end_time)

def print_step_header(step, name):
    """用于在终端打印格式化的步骤标题"""
    print("\n" + "="*80)
    print(f"| [步骤 {step}] {name}")
    print("="*80)

def print_pipeline_summary(start_time, end_time):
    """打印流水线执行总耗时"""
    total_duration = end_time - start_time
    print("\n" + "#"*80)
    print("### 完整流水线执行完毕！ ###")
    print(f"### 总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟) ###")
    print("#"*80)


if __name__ == '__main__':
    main()