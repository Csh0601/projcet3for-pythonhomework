# -*- coding: utf-8 -*-
"""
3D三分类可视化 - 鸢尾花数据集 (任务四 - 加分项)
三分类: Setosa, Versicolor, Virginica
3D决策边界 + 概率分布
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
    iris = load_iris()
    X = iris.data[:, 1:4]
    y = iris.target
    return X, y, ['Sepal Width', 'Petal Length', 'Petal Width']


def draw_3d_multiclass_scatter(X, y, feature_names):
    """绘制三分类3D散点图（多视角）"""
    set_dark_style()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig = plt.figure(figsize=(20, 10))
    angles = [(30, 45), (30, 135), (60, 45), (15, 225)]
    titles = ['View 1', 'View 2', 'View 3', 'View 4']

    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        style_3d_axes(ax)

        for class_id in range(3):
            mask = y == class_id
            ax.scatter(
                X_scaled[mask, 0], X_scaled[mask, 1], X_scaled[mask, 2],
                c=SPECIES_COLORS[class_id], s=80, alpha=0.85,
                edgecolors='white', linewidths=0.8,
                label=SPECIES_NAMES[class_id], depthshade=True
            )

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(feature_names[0], fontsize=10, color='white')
        ax.set_ylabel(feature_names[1], fontsize=10, color='white')
        ax.set_zlabel(feature_names[2], fontsize=10, color='white')
        ax.set_title(titles[idx], fontsize=12, fontweight='bold', color='white')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SPECIES_COLORS[i],
               markersize=12, label=SPECIES_NAMES[i], linestyle='None', markeredgecolor='white')
        for i in range(3)
    ]
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=12,
              framealpha=0.95, facecolor=COLORS['background_alt'],
              edgecolor=COLORS['accent'])

    fig.suptitle('3D MULTICLASS SCATTER: IRIS DATASET',
                fontsize=18, fontweight='bold', color='white', y=0.98)

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    save_figure(fig, '3d_multiclass_scatter.png', tight=False)
    plt.show()
    reset_style()


def draw_3d_multiclass_boundaries(X, y, feature_names):
    """绘制三分类3D决策边界"""
    set_dark_style()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=1, max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_scaled, y)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    style_3d_axes(ax)

    # 数据点
    for class_id in range(3):
        mask = y == class_id
        ax.scatter(
            X_scaled[mask, 0], X_scaled[mask, 1], X_scaled[mask, 2],
            c=SPECIES_COLORS[class_id], s=100, alpha=0.9,
            edgecolors='white', linewidths=1,
            label=SPECIES_NAMES[class_id], depthshade=True
        )

    # 计算决策边界（多分类）
    resolution = 25
    padding = 0.5
    x_range = np.linspace(X_scaled[:, 0].min() - padding, X_scaled[:, 0].max() + padding, resolution)
    y_range = np.linspace(X_scaled[:, 1].min() - padding, X_scaled[:, 1].max() + padding, resolution)
    z_range = np.linspace(X_scaled[:, 2].min() - padding, X_scaled[:, 2].max() + padding, resolution)

    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    predictions = clf.predict(grid).reshape(xx.shape)

    # 绘制类别之间的边界面
    boundary_pairs = [(0, 1), (1, 2), (0, 2)]
    boundary_colors = ['#00CED1', '#FFD700', '#FF6B6B']

    for (c1, c2), color in zip(boundary_pairs, boundary_colors):
        boundary_mask = np.zeros_like(predictions, dtype=float)
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                for k in range(resolution - 1):
                    neighbors = [
                        predictions[i, j, k], predictions[i+1, j, k],
                        predictions[i, j+1, k], predictions[i, j, k+1]
                    ]
                    if c1 in neighbors and c2 in neighbors:
                        boundary_mask[i, j, k] = 1

        if boundary_mask.sum() > 0:
            try:
                verts, faces, _, _ = measure.marching_cubes(boundary_mask, level=0.5)
                verts[:, 0] = verts[:, 0] / resolution * (x_range[-1] - x_range[0]) + x_range[0]
                verts[:, 1] = verts[:, 1] / resolution * (y_range[-1] - y_range[0]) + y_range[0]
                verts[:, 2] = verts[:, 2] / resolution * (z_range[-1] - z_range[0]) + z_range[0]

                mesh = Poly3DCollection(verts[faces], alpha=0.2, facecolor=color,
                                       edgecolor='white', linewidths=0.1)
                ax.add_collection3d(mesh)
            except:
                pass

    ax.view_init(elev=25, azim=45)
    ax.set_xlabel(feature_names[0], fontsize=12, color='white', labelpad=10)
    ax.set_ylabel(feature_names[1], fontsize=12, color='white', labelpad=10)
    ax.set_zlabel(feature_names[2], fontsize=12, color='white', labelpad=10)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SPECIES_COLORS[i],
               markersize=14, label=SPECIES_NAMES[i], linestyle='None', markeredgecolor='white')
        for i in range(3)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
             framealpha=0.95, facecolor=COLORS['background_alt'],
             edgecolor=COLORS['accent'])

    ax.set_title('3D MULTICLASS DECISION BOUNDARIES',
                fontsize=16, fontweight='bold', color='white', pad=20)

    fig.patch.set_facecolor(COLORS['background'])
    save_figure(fig, '3d_multiclass_boundaries.png')
    plt.show()
    reset_style()


def draw_3d_multiclass_probability_slices(X, y, feature_names):
    """绘制三分类概率切片"""
    set_dark_style()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=1, max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_scaled, y)

    fig = plt.figure(figsize=(20, 6))

    d1, d2, d_fixed = 1, 2, 0
    fixed_val = np.median(X_scaled[:, d_fixed])

    padding = 0.5
    x_range = np.linspace(X_scaled[:, d1].min() - padding, X_scaled[:, d1].max() + padding, 100)
    y_range = np.linspace(X_scaled[:, d2].min() - padding, X_scaled[:, d2].max() + padding, 100)
    xx, yy = np.meshgrid(x_range, y_range)

    grid = np.zeros((xx.size, 3))
    grid[:, d1] = xx.ravel()
    grid[:, d2] = yy.ravel()
    grid[:, d_fixed] = fixed_val

    probs = clf.predict_proba(grid)

    for class_id in range(3):
        ax = fig.add_subplot(1, 3, class_id + 1)

        prob_map = probs[:, class_id].reshape(xx.shape)
        cmap = LinearSegmentedColormap.from_list('prob',
            [COLORS['background'], SPECIES_COLORS[class_id]], N=256)

        im = ax.contourf(xx, yy, prob_map, levels=20, cmap=cmap, alpha=0.9)
        ax.contour(xx, yy, prob_map, levels=[0.33, 0.5, 0.67], colors='white',
                  linewidths=[1, 2, 1], linestyles=['--', '-', '--'])

        for i in range(3):
            mask = y == i
            ax.scatter(X_scaled[mask, d1], X_scaled[mask, d2],
                      c=SPECIES_COLORS[i], s=40, edgecolors='white',
                      linewidths=0.5, alpha=0.8, zorder=5)

        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label('Probability', fontsize=10, color='white')
        cbar.ax.tick_params(colors='white')

        ax.set_xlabel(feature_names[d1], fontsize=11, color='white')
        ax.set_ylabel(feature_names[d2], fontsize=11, color='white')
        ax.set_title(f'{SPECIES_NAMES[class_id]} Probability',
                    fontsize=13, fontweight='bold', color='white')
        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors='white')

    legend_elements = [
        Patch(facecolor=SPECIES_COLORS[i], edgecolor='white', label=SPECIES_NAMES[i])
        for i in range(3)
    ]
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=11,
              framealpha=0.95, facecolor=COLORS['background_alt'],
              edgecolor=COLORS['accent'])

    fig.suptitle('3D MULTICLASS PROBABILITY MAPS (SLICE VIEW)',
                fontsize=18, fontweight='bold', color='white', y=0.98)

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    save_figure(fig, '3d_multiclass_probability_slices.png', tight=False)
    plt.show()
    reset_style()


def draw_3d_combined_view(X, y, feature_names):
    """综合视图：3D散点 + 2D切片"""
    set_dark_style()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=1, max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_scaled, y)

    fig = plt.figure(figsize=(18, 14))

    # 3D主图
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    style_3d_axes(ax_3d)

    for class_id in range(3):
        mask = y == class_id
        ax_3d.scatter(
            X_scaled[mask, 0], X_scaled[mask, 1], X_scaled[mask, 2],
            c=SPECIES_COLORS[class_id], s=80, alpha=0.9,
            edgecolors='white', linewidths=0.8,
            label=SPECIES_NAMES[class_id], depthshade=True
        )

    ax_3d.view_init(elev=25, azim=45)
    ax_3d.set_xlabel(feature_names[0], fontsize=10, color='white')
    ax_3d.set_ylabel(feature_names[1], fontsize=10, color='white')
    ax_3d.set_zlabel(feature_names[2], fontsize=10, color='white')
    ax_3d.set_title('3D Scatter Plot', fontsize=13, fontweight='bold', color='white')
    ax_3d.legend(loc='upper left', fontsize=9, framealpha=0.9,
                facecolor=COLORS['background_alt'], edgecolor=COLORS['accent'])

    # 2D切片
    slice_configs = [
        (0, 1, 2, 'XY Plane'),
        (0, 2, 1, 'XZ Plane'),
        (1, 2, 0, 'YZ Plane')
    ]

    for idx, (d1, d2, d_fixed, title) in enumerate(slice_configs):
        ax = fig.add_subplot(2, 2, idx + 2)

        fixed_val = np.median(X_scaled[:, d_fixed])
        padding = 0.5
        x_range = np.linspace(X_scaled[:, d1].min() - padding, X_scaled[:, d1].max() + padding, 80)
        y_range = np.linspace(X_scaled[:, d2].min() - padding, X_scaled[:, d2].max() + padding, 80)
        xx, yy = np.meshgrid(x_range, y_range)

        grid = np.zeros((xx.size, 3))
        grid[:, d1] = xx.ravel()
        grid[:, d2] = yy.ravel()
        grid[:, d_fixed] = fixed_val

        predictions = clf.predict(grid).reshape(xx.shape)

        cmap = ListedColormap([f'{c}80' for c in SPECIES_COLORS])
        ax.contourf(xx, yy, predictions, alpha=0.5, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5])
        ax.contour(xx, yy, predictions, colors='white', linewidths=1.5, levels=[0.5, 1.5])

        for i in range(3):
            mask = y == i
            ax.scatter(X_scaled[mask, d1], X_scaled[mask, d2],
                      c=SPECIES_COLORS[i], s=50, edgecolors='white',
                      linewidths=0.5, alpha=0.9, zorder=5)

        ax.set_xlabel(feature_names[d1], fontsize=10, color='white')
        ax.set_ylabel(feature_names[d2], fontsize=10, color='white')
        ax.set_title(f'{title} Slice', fontsize=12, fontweight='bold', color='white')
        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors='white')

    fig.suptitle('3D MULTICLASS CLASSIFICATION: COMBINED VIEW',
                fontsize=18, fontweight='bold', color='white', y=0.99)

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_figure(fig, '3d_multiclass_combined.png')
    plt.show()
    reset_style()


if __name__ == "__main__":
    print("Loading Iris dataset...")
    X, y, feature_names = load_data_3features()

    print("\n1. Drawing 3D multiclass scatter...")
    draw_3d_multiclass_scatter(X, y, feature_names)

    print("\n2. Drawing 3D multiclass boundaries...")
    draw_3d_multiclass_boundaries(X, y, feature_names)

    print("\n3. Drawing 3D multiclass probability slices...")
    draw_3d_multiclass_probability_slices(X, y, feature_names)

    print("\n4. Drawing combined view...")
    draw_3d_combined_view(X, y, feature_names)

    print("\nAll 3D multiclass plots completed!")
