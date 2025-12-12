# -*- coding: utf-8 -*-
"""
主运行脚本 - 一键生成所有可视化图片
"""

import os
import sys
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_step(step_num, total, text):
    print(f"\n[{step_num}/{total}] {text}")
    print("-" * 40)

def main():
    start_time = time.time()
    print_header("IRIS CLASSIFICATION & VISUALIZATION PROJECT")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_steps = 7

    # Step 1: 特征分布
    print_step(1, total_steps, "Feature Distribution Visualization")
    try:
        from importlib import import_module
        module = import_module('src.analysis.1_feature_distribution')
        df = module.load_iris_dataframe()
        module.draw_feature_distribution_combined(df)
        module.draw_feature_distribution_by_species(df)
        module.draw_boxplot_grid(df)
        print("  [OK] Feature distribution plots completed!")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Step 2: 特征分析
    print_step(2, total_steps, "Feature Analysis Visualization")
    try:
        module = import_module('src.analysis.2_feature_analysis')
        df = module.load_iris_dataframe()
        module.draw_pairplot(df)
        module.draw_correlation_heatmap(df)
        module.draw_scatter_2d_advanced(df)
        print("  [OK] Feature analysis plots completed!")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Step 3: 2D决策边界
    print_step(3, total_steps, "2D Decision Boundary Visualization")
    try:
        module = import_module('src.visualization.3_2d_decision_boundary')
        X, y, feature_names = module.load_data_2features()
        module.draw_classifier_comparison(X, y, feature_names)
        module.draw_probability_contours(X, y, feature_names)
        print("  [OK] 2D decision boundary plots completed!")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Step 4: 2D概率图
    print_step(4, total_steps, "2D Probability Map Visualization")
    try:
        module = import_module('src.visualization.4_2d_probability_map')
        X, y, feature_names = module.load_data_2features()
        module.draw_uncertainty_map(X, y, feature_names)
        module.draw_probability_comparison_by_class(X, y, feature_names)
        module.draw_decision_regions_detailed(X, y, feature_names)
        print("  [OK] 2D probability map plots completed!")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Step 5: 3D二分类边界
    print_step(5, total_steps, "3D Binary Boundary Visualization")
    try:
        module = import_module('src.visualization.5_3d_boundary_binary')
        X, y, feature_names = module.load_data_3features()

        # Setosa vs Others
        X1, y1, n1, c1, t1 = module.prepare_binary_data(X, y, 'setosa_vs_others')
        module.draw_3d_boundary_single(X1, y1, n1, c1, t1, feature_names)
        module.draw_3d_boundary_with_slices(X1, y1, n1, c1, t1, feature_names)

        # Versicolor vs Virginica
        X2, y2, n2, c2, t2 = module.prepare_binary_data(X, y, 'versicolor_vs_virginica')
        module.draw_3d_boundary_single(X2, y2, n2, c2, t2, feature_names)
        module.draw_3d_boundary_with_slices(X2, y2, n2, c2, t2, feature_names)
        print("  [OK] 3D binary boundary plots completed!")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Step 6: 3D二分类概率图
    print_step(6, total_steps, "3D Binary Probability Visualization")
    try:
        module = import_module('src.visualization.6_3d_probability_binary')
        X, y, feature_names = module.load_data_3features()

        X1, y1, n1, c1, t1 = module.prepare_binary_data(X, y, 'setosa_vs_others')
        module.draw_3d_probability_surface(X1, y1, n1, c1, t1, feature_names)

        X2, y2, n2, c2, t2 = module.prepare_binary_data(X, y, 'versicolor_vs_virginica')
        module.draw_3d_probability_surface(X2, y2, n2, c2, t2, feature_names)
        print("  [OK] 3D binary probability plots completed!")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Step 7: 3D三分类
    print_step(7, total_steps, "3D Multiclass Visualization")
    try:
        module = import_module('src.visualization.7_3d_boundary_multiclass')
        X, y, feature_names = module.load_data_3features()
        module.draw_3d_multiclass_scatter(X, y, feature_names)
        module.draw_3d_multiclass_boundaries(X, y, feature_names)
        module.draw_3d_multiclass_probability_slices(X, y, feature_names)
        module.draw_3d_combined_view(X, y, feature_names)
        print("  [OK] 3D multiclass plots completed!")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # 完成
    elapsed = time.time() - start_time
    print_header("ALL VISUALIZATIONS COMPLETED!")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Output directory: outputs/figures/")

    # 列出生成的文件
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        print(f"\nGenerated {len(files)} figures:")
        for f in sorted(files):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
