import argparse
import time

from src.data_processing.main_preprocess import run_preprocessing
from src.cwt_transformation.main_transform import run_cwt_transformation
from src.modeling.train import run_training
from src.interpretation.explain import run_explanation
from src.interpretation.advanced_analysis import run_advanced_analysis # 导入新函数

def main():
    parser = argparse.ArgumentParser(description="Run stages of the well log channeling detection project.")
    parser.add_argument(
        'stage',
        nargs='+',
        choices=['preprocess', 'transform', 'train', 'explain', 'analysis'], # 添加 'analysis' 选项
        help="The stage(s) to run."
    )
    parser.add_argument(
        '--index',
        type=int,
        default=100,
        help="The sample index to explain (for 'explain' stage)."
    )
    args = parser.parse_args()
    
    start_time = time.time()
    
    # ... (preprocess, transform, train, explain 逻辑保持不变) ...
    if 'preprocess' in args.stage:
        print("--- Running Stage: Preprocessing ---")
        run_preprocessing()
        
    if 'transform' in args.stage:
        print("\n--- Running Stage: CWT Transformation ---")
        run_cwt_transformation()

    if 'train' in args.stage:
        print("\n--- Running Stage: Model Training ---")
        run_training()
        
    if 'explain' in args.stage:
        print("\n--- Running Stage: Model Explanation (Grad-CAM) ---")
        run_explanation(sample_index=args.index)
        
    if 'analysis' in args.stage:
        print("\n--- Running Stage: Advanced Analysis & Visualization ---")
        run_advanced_analysis()
        
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()