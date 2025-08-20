# src/interpretation/visualize_model.py

import os
import sys
import argparse
import tensorflow as tf

# --- 确保能找到项目模块 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config import get_config
from src.modeling.model import build_cnn_regressor

def visualize_model_architecture(config):
    """
    构建模型并将其架构图保存为图片。
    
    Args:
        config (dict): 从 get_config() 函数生成的配置字典。
    """
    array_id = config['array_id']
    print(f"--- [开始为阵列 {array_id} 可视化模型架构] ---")

    # 1. 从配置中获取模型参数和输出路径
    model_params = config['modeling']
    paths = config['paths']
    
    # 2. 构建模型实例
    # 我们不需要训练好的模型，只需要其结构，所以直接构建即可
    print("正在构建 SE-ResNet 模型...")
    model = build_cnn_regressor(input_shape=model_params['input_shape'])
    
    # 打印模型的文本摘要
    model.summary()

    # 3. 定义输出路径
    output_path = os.path.join(paths['plot_dir'], f"model_architecture_array_{array_id}.png")
    print(f"\n模型架构图将被保存至: {output_path}")

    # 4. 使用 TensorFlow 工具生成并保存可视化图
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=output_path,
            show_shapes=True,          # 在图中显示输入/输出的形状
            show_layer_names=True,     # 显示层的名称
            show_layer_activations=True, # 显示激活函数
            expand_nested=True,        # 展开嵌套的模型（对我们的残差块很重要）
            dpi=96                     # 图像分辨率
        )
        print("\n--- [模型可视化成功！] ---")
    except ImportError as e:
        print("\n--- [错误] ---")
        print("生成模型图失败。这通常是因为缺少必要的绘图库。")
        print(f"错误信息: {e}")
        print("\n请尝试在您的conda环境中安装 pydot 和 graphviz:")
        print("conda install pydot")
        print("conda install graphviz")
        print("如果上述命令无效，您可能还需要在系统层面安装 graphviz。")

if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="为当前项目模型生成并保存一个可视化的架构图。")
    parser.add_argument(
        '--array',
        type=str,
        default='03',
        help="指定要使用的配置对应的阵列编号 (例如: '03', '07', '11')。这会影响图片保存的位置。"
    )
    args = parser.parse_args()

    # 根据命令行参数获取配置
    config = get_config(args.array)
    
    # 运行主函数
    visualize_model_architecture(config)