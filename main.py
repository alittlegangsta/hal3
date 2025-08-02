# main.py

import argparse
import sys
import os
import time

# 将 src 目录添加到 Python 的模块搜索路径中
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """项目主入口函数。"""
    parser = argparse.ArgumentParser(
        description="统一的深度学习声波测井分析项目入口。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='可执行的命令')

    # --- 定义命令 ---
    subparsers.add_parser('preprocess', help='阶段一: 将原始.mat文件处理成对齐的波形数据')
    subparsers.add_parser('cwt', help='阶段二: 对波形数据进行小波变换，生成尺度图')
    subparsers.add_parser('split', help='阶段三: 划分训练/验证集，并计算标准化统计数据')
    subparsers.add_parser('train', help='阶段四: 训练CNN模型')
    subparsers.add_parser('analyze', help='阶段五: 评估模型性能并生成分析图 (待实现)')
    subparsers.add_parser('all', help='一键按顺序执行从 preprocess 到 train 的所有阶段')

    args = parser.parse_args()

    # --- 命令分发中心 ---
    command_functions = {
        'preprocess': run_preprocess_command,
        'cwt': run_cwt_command,
        'split': run_split_command,
        'train': run_train_command,
        'analyze': run_analyze_command,
        'all': run_all_command
    }
    
    command_to_run = command_functions.get(args.command)
    if command_to_run:
        command_to_run()

def run_preprocess_command():
    print_step_header(1, "数据预处理")
    from data_processing.main_preprocess import run_stage1_preprocessing
    run_stage1_preprocessing()

def run_cwt_command():
    print_step_header(2, "小波变换")
    from cwt_transformation.main_transform import run_cwt_transformation
    run_cwt_transformation()

def run_split_command():
    print_step_header(3, "数据集划分与标准化")
    from data_processing.main_preprocess import run_stage2_split_and_normalize
    run_stage2_split_and_normalize()

def run_train_command():
    print_step_header(4, "模型训练")
    from modeling.train import train_model
    train_model()

def run_analyze_command():
    print_step_header(5, "模型评估与分析")
    print("注意：分析功能 (analyze) 尚未完全实现。")
    # from analysis.advanced_analysis import run_analysis
    # run_analysis()

def run_all_command():
    print("--- 开始执行完整流水线 (preprocess -> cwt -> split -> train) ---")
    start_time = time.time()
    try:
        run_preprocess_command()
        run_cwt_command()
        run_split_command()
        run_train_command()
    except Exception as e:
        print(f"\n流水线执行出错: {e}")
        print("请检查错误信息并修正后重试。")
        return
    end_time = time.time()
    print_pipeline_summary(start_time, end_time)

def print_step_header(step, name):
    print("\n" + "="*80)
    print(f"| [步骤 {step}] {name}")
    print("="*80)

def print_pipeline_summary(start_time, end_time):
    total_duration = end_time - start_time
    print("\n" + "#"*80)
    print("### 完整流水线执行完毕！ ###")
    print(f"### 总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟) ###")
    print("#"*80)


if __name__ == '__main__':
    if not os.path.exists('src'):
        print("错误: 未在当前目录下找到 'src' 文件夹。")
        print("请确保您在项目的根目录下运行此脚本。")
        sys.exit(1)
    main()