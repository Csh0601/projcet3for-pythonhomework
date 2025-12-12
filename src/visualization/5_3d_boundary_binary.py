# -*- coding: utf-8 -*-
"""
3D决策边界可视化 - 鸢尾花数据集 (任务二)
两分类问题：
1. Setosa vs 其他 (Versicolor + Virginica)
2. Versicolor vs Virginica
特征: Sepal Width (x1) + Petal Length (x2) + Petal Width (x3)
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage import measure

from config import (
    set_dark_style, reset_style, save_figure, style_3d_axes,
    COLORS, SPECIES_COLORS, SPECIES_NAMES, FIGURE_DIR
)


def load_data_3features():
    """加载数据，使用Sepal Width, Petal Length, Petal Width三个特征"""
    iris = load_iris()
    # 使用特征1,2,3 (Sepal Width, Petal Length, Petal Width)
    X = iris.data[:, 1:4]
    y = iris.target
    feature_names = ['Sepal Width', 'Petal Length', 'Petal Width']
    return X, y, feature_names


def prepare_binary_data(X, y, mode='setosa_vs_others'):
    """
    准备二分类数据
    mode: 'setosa_vs_others' 或 'versicolor_vs_virginica'
    """
    if mode == 'setosa_vs_others':
        # Setosa (0) vs Others (1)
        y_binary = (y != 0).astype(int)
        class_names = ['Setosa', 'Others']
        class_colors = [SPECIES_COLORS[0], '#888888']
        title_suffix = 'Setosa vs Others'
    else:
        # Versicolor (0) vs Virginica (1) - 只使用这两类数据
        mask = y != 0
        X_binary = X[mask]
        y_binary = (y[mask] == 2).astype(int)
        class_names = ['Versicolor', 'Virginica']
        class_colors = [SPECIES_COLORS[1], SPECIES_COLORS[2]]
        title_suffix = 'Versicolor vs Virginica'
        return X_binary, y_binary, class_names, class_colors, title_suffix

    return X, y_binary, class_names, class_colors, title_suffix


def compute_decision_boundary_surface(clf, X, resolution=30):
    """
    计算3D决策边界曲面
    返回等值面的顶点和面
    """
    # 创建3D网格
    padding = 0.5
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    z_min, z_max = X[:, 2].min() - padding, X[:, 2].max() + padding

    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
        np.linspace(z_min, z_max, resolution)
    )

    # 获取决策函数值或概率
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    if hasattr(clf, 'decision_function'):
        decision_values = clf.decision_function(grid_points)
    else:
        probs = clf.predict_proba(grid_points)
        decision_values = probs[:, 1] - probs[:, 0]

    decision_values = decision_values.reshape(xx.shape)

    # 使用 marching cubes 提取等值面 (decision_value = 0)
    try:
        verts, faces, _, _ = measure.marching_cubes(decision_values, level=0)

        # 转换回原始坐标
        verts[:, 0] = verts[:, 0] / resolution * (x_max - x_min) + x_min
        verts[:, 1] = verts[:, 1] / resolution * (y_max - y_min) + y_min
        verts[:, 2] = verts[:, 2] / resolution * (z_max - z_min) + z_min

        return verts, faces, (xx, yy, zz, decision_values)
    except:
        return None, None, (xx, yy, zz, decision_values)


def draw_3d_boundary_single(X, y, class_names, class_colors, title_suffix, feature_names):
    """绘制单个3D决策边界图（多角度）"""
    set_dark_style()

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练SVM分类器
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(X_scaled, y)

    # 计算决策边界
    verts, faces, grid_data = compute_decision_boundary_surface(clf, X_scaled, resolution=35)

    # 创建多视角图
    fig = plt.figure(figsize=(20, 10))

    angles = [(30, 45), (30, 135), (60, 45), (15, 225)]
    titles = ['View 1 (Default)', 'View 2 (Rotated 90°)', 'View 3 (Top-Down)', 'View 4 (Back)']

    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        style_3d_axes(ax)

        # 绘制数据点
        for class_id in range(2):
            mask = y == class_id
            ax.scatter(
                X_scaled[mask, 0], X_scaled[mask, 1], X_scaled[mask, 2],
                c=class_colors[class_id], s=60, alpha=0.8,
                edgecolors='white', linewidths=0.5,
                label=class_names[class_id], depthshade=True
            )

        # 绘制决策边界曲面
        if verts is not None and faces is not None:
            mesh = Poly3DCollection(
                verts[faces],
                alpha=0.25,
                facecolor='gray',
                edgecolor='white',
                linewidths=0.1
            )
            ax.add_collection3d(mesh)

        # 设置视角
        ax.view_init(elev=elev, azim=azim)

        # 标签
        ax.set_xlabel(feature_names[0], fontsize=10, color='white', labelpad=8)
        ax.set_ylabel(feature_names[1], fontsize=10, color='white', labelpad=8)
        ax.set_zlabel(feature_names[2], fontsize=10, color='white', labelpad=8)
        ax.set_title(titles[idx], fontsize=12, fontweight='bold', color='white', pad=10)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i],
               markersize=12, label=class_names[i], linestyle='None', markeredgecolor='white')
        for i in range(2)
    ]
    legend_elements.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=12, label='Decision Boundary', linestyle='None', alpha=0.5)
    )
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12,
              framealpha=0.95, facecolor=COLORS['background_alt'],
              edgecolor=COLORS['accent'])

    fig.suptitle(f'3D DECISION BOUNDARY: {title_suffix.upper()}',
                fontsize=18, fontweight='bold', color='white', y=0.98)
    fig.text(0.5, 0.94, f'Features: {", ".join(feature_names)} | Classifier: SVM (RBF)',
            fontsize=11, color=COLORS['text_secondary'], ha='center')

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    filename = f'3d_boundary_{title_suffix.lower().replace(" ", "_")}.png'
    save_figure(fig, filename, tight=False)
    plt.show()

    reset_style()


def draw_3d_boundary_with_slices(X, y, class_names, class_colors, title_suffix, feature_names):
    """绘制3D决策边界 + 2D切片图"""
    set_dark_style()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    clf.fit(X_scaled, y)

    fig = plt.figure(figsize=(18, 14))

    # 3D主图
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    style_3d_axes(ax_3d)

    # 数据点
    for class_id in range(2):
        mask = y == class_id
        ax_3d.scatter(
            X_scaled[mask, 0], X_scaled[mask, 1], X_scaled[mask, 2],
            c=class_colors[class_id], s=80, alpha=0.85,
            edgecolors='white', linewidths=0.8,
            label=class_names[class_id], depthshade=True
        )

    # 决策边界
    verts, faces, _ = compute_decision_boundary_surface(clf, X_scaled, resolution=30)
    if verts is not None:
        mesh = Poly3DCollection(verts[faces], alpha=0.3, facecolor='#888888',
                               edgecolor='white', linewidths=0.1)
        ax_3d.add_collection3d(mesh)

    ax_3d.view_init(elev=25, azim=45)
    ax_3d.set_xlabel(feature_names[0], fontsize=10, color='white')
    ax_3d.set_ylabel(feature_names[1], fontsize=10, color='white')
    ax_3d.set_zlabel(feature_names[2], fontsize=10, color='white')
    ax_3d.set_title('3D Decision Boundary', fontsize=13, fontweight='bold', color='white')

    # 2D切片图
    slice_axes = [(0, 1), (0, 2), (1, 2)]
    slice_titles = [
        f'{feature_names[0]} vs {feature_names[1]}',
        f'{feature_names[0]} vs {feature_names[2]}',
        f'{feature_names[1]} vs {feature_names[2]}'
    ]
    slice_positions = [(2, 2, 2), (2, 2, 3), (2, 2, 4)]

    for (dim1, dim2), title, pos in zip(slice_axes, slice_titles, slice_positions):
        ax = fig.add_subplot(*pos)

        # 创建2D网格
        padding = 0.5
        x_range = np.linspace(X_scaled[:, dim1].min() - padding,
                             X_scaled[:, dim1].max() + padding, 150)
        y_range = np.linspace(X_scaled[:, dim2].min() - padding,
                             X_scaled[:, dim2].max() + padding, 150)
        xx, yy = np.meshgrid(x_range, y_range)

        # 第三维度取中值
        third_dim = [d for d in range(3) if d not in [dim1, dim2]][0]
        third_val = np.median(X_scaled[:, third_dim])

        grid = np.zeros((xx.size, 3))
        grid[:, dim1] = xx.ravel()
        grid[:, dim2] = yy.ravel()
        grid[:, third_dim] = third_val

        probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

        # 概率热图
        cmap = LinearSegmentedColormap.from_list('binary',
            [class_colors[0], '#FFFFFF', class_colors[1]], N=256)
        ax.contourf(xx, yy, probs, levels=20, cmap=cmap, alpha=0.8)
        ax.contour(xx, yy, probs, levels=[0.5], colors='white', linewidths=2)

        # 数据点
        for class_id in range(2):
            mask = y == class_id
            ax.scatter(X_scaled[mask, dim1], X_scaled[mask, dim2],
                      c=class_colors[class_id], s=40, edgecolors='white',
                      linewidths=0.5, alpha=0.9, zorder=5)

        ax.set_xlabel(feature_names[dim1], fontsize=10, color='white')
        ax.set_ylabel(feature_names[dim2], fontsize=10, color='white')
        ax.set_title(f'Slice: {title}\n(3rd dim @ median)', fontsize=11,
                    fontweight='bold', color='white')
        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors='white')

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i],
               markersize=12, label=class_names[i], linestyle='None', markeredgecolor='white')
        for i in range(2)
    ]
    fig.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(0.98, 0.98), fontsize=11,
              framealpha=0.95, facecolor=COLORS['background_alt'],
              edgecolor=COLORS['accent'])

    fig.suptitle(f'3D BOUNDARY WITH 2D SLICES: {title_suffix.upper()}',
                fontsize=18, fontweight='bold', color='white', y=0.99)

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filename = f'3d_boundary_slices_{title_suffix.lower().replace(" ", "_")}.png'
    save_figure(fig, filename)
    plt.show()

    reset_style()


def draw_3d_scatter_with_projections(X, y, class_names, class_colors, title_suffix, feature_names):
    """绘制3D散点图 + 各平面投影"""
    set_dark_style()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    style_3d_axes(ax)

    # 获取坐标范围
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    z_min, z_max = X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5

    # 绘制主数据点
    for class_id in range(2):
        mask = y == class_id
        ax.scatter(
            X_scaled[mask, 0], X_scaled[mask, 1], X_scaled[mask, 2],
            c=class_colors[class_id], s=100, alpha=0.9,
            edgecolors='white', linewidths=1,
            label=class_names[class_id], depthshade=True
        )

    # XY平面投影 (z=z_min)
    for class_id in range(2):
        mask = y == class_id
        ax.scatter(
            X_scaled[mask, 0], X_scaled[mask, 1],
            np.full(mask.sum(), z_min),
            c=class_colors[class_id], s=30, alpha=0.3,
            marker='o', depthshade=False
        )

    # XZ平面投影 (y=y_max)
    for class_id in range(2):
        mask = y == class_id
        ax.scatter(
            X_scaled[mask, 0],
            np.full(mask.sum(), y_max),
            X_scaled[mask, 2],
            c=class_colors[class_id], s=30, alpha=0.3,
            marker='o', depthshade=False
        )

    # YZ平面投影 (x=x_min)
    for class_id in range(2):
        mask = y == class_id
        ax.scatter(
            np.full(mask.sum(), x_min),
            X_scaled[mask, 1], X_scaled[mask, 2],
            c=class_colors[class_id], s=30, alpha=0.3,
            marker='o', depthshade=False
        )

    # 连接线（采样部分点）
    sample_idx = np.random.choice(len(X_scaled), min(30, len(X_scaled)), replace=False)
    for idx in sample_idx:
        color = class_colors[y[idx]]
        # 到XY平面
        ax.plot([X_scaled[idx, 0], X_scaled[idx, 0]],
               [X_scaled[idx, 1], X_scaled[idx, 1]],
               [X_scaled[idx, 2], z_min],
               color=color, alpha=0.2, linewidth=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.view_init(elev=20, azim=45)
    ax.set_xlabel(feature_names[0], fontsize=12, color='white', labelpad=10)
    ax.set_ylabel(feature_names[1], fontsize=12, color='white', labelpad=10)
    ax.set_zlabel(feature_names[2], fontsize=12, color='white', labelpad=10)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i],
               markersize=14, label=class_names[i], linestyle='None', markeredgecolor='white')
        for i in range(2)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
             framealpha=0.95, facecolor=COLORS['background_alt'],
             edgecolor=COLORS['accent'])

    ax.set_title(f'3D SCATTER WITH PROJECTIONS: {title_suffix.upper()}',
                fontsize=16, fontweight='bold', color='white', pad=20)

    fig.patch.set_facecolor(COLORS['background'])

    filename = f'3d_scatter_projections_{title_suffix.lower().replace(" ", "_")}.png'
    save_figure(fig, filename)
    plt.show()

    reset_style()


if __name__ == "__main__":
    print("Loading Iris dataset (3 features)...")
    X, y, feature_names = load_data_3features()
    print(f"Data shape: {X.shape}")
    print(f"Features: {feature_names}")

    # 场景1: Setosa vs Others
    print("\n=== Scenario 1: Setosa vs Others ===")
    X1, y1, names1, colors1, title1 = prepare_binary_data(X, y, 'setosa_vs_others')
    print(f"Class distribution: {np.bincount(y1)}")

    print("1.1 Drawing 3D boundary (multiple views)...")
    draw_3d_boundary_single(X1, y1, names1, colors1, title1, feature_names)

    print("1.2 Drawing 3D boundary with slices...")
    draw_3d_boundary_with_slices(X1, y1, names1, colors1, title1, feature_names)

    print("1.3 Drawing 3D scatter with projections...")
    draw_3d_scatter_with_projections(X1, y1, names1, colors1, title1, feature_names)

    # 场景2: Versicolor vs Virginica
    print("\n=== Scenario 2: Versicolor vs Virginica ===")
    X2, y2, names2, colors2, title2 = prepare_binary_data(X, y, 'versicolor_vs_virginica')
    print(f"Data shape: {X2.shape}")
    print(f"Class distribution: {np.bincount(y2)}")

    print("2.1 Drawing 3D boundary (multiple views)...")
    draw_3d_boundary_single(X2, y2, names2, colors2, title2, feature_names)

    print("2.2 Drawing 3D boundary with slices...")
    draw_3d_boundary_with_slices(X2, y2, names2, colors2, title2, feature_names)

    print("2.3 Drawing 3D scatter with projections...")
    draw_3d_scatter_with_projections(X2, y2, names2, colors2, title2, feature_names)

    print("\nAll 3D binary boundary plots completed!")
