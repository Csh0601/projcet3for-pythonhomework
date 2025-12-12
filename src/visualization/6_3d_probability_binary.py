# -*- coding: utf-8 -*-
"""
3D概率图可视化 - 鸢尾花数据集 (任务三)
两分类问题的3D概率曲面
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib import cm
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import (
    set_dark_style, reset_style, save_figure, style_3d_axes,
    COLORS, SPECIES_COLORS, FIGURE_DIR
)


def load_data_3features():
    iris = load_iris()
    X = iris.data[:, 1:4]
    y = iris.target
    return X, y, ['Sepal Width', 'Petal Length', 'Petal Width']


def prepare_binary_data(X, y, mode='setosa_vs_others'):
    if mode == 'setosa_vs_others':
        y_binary = (y != 0).astype(int)
        class_names = ['Setosa', 'Others']
        class_colors = [SPECIES_COLORS[0], '#888888']
        title_suffix = 'Setosa vs Others'
        return X, y_binary, class_names, class_colors, title_suffix
    else:
        mask = y != 0
        X_binary = X[mask]
        y_binary = (y[mask] == 2).astype(int)
        class_names = ['Versicolor', 'Virginica']
        class_colors = [SPECIES_COLORS[1], SPECIES_COLORS[2]]
        title_suffix = 'Versicolor vs Virginica'
        return X_binary, y_binary, class_names, class_colors, title_suffix


def draw_3d_probability_surface(X, y, class_names, class_colors, title_suffix, feature_names):
    """绘制3D概率曲面 - 固定一个维度，显示概率曲面"""
    set_dark_style()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=1, max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    fig = plt.figure(figsize=(20, 10))

    # 使用特征0和1，固定特征2
    dim_pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    titles = [
        f'{feature_names[0]} vs {feature_names[1]}',
        f'{feature_names[0]} vs {feature_names[2]}',
        f'{feature_names[1]} vs {feature_names[2]}'
    ]

    for idx, ((d1, d2, d_fixed), title) in enumerate(zip(dim_pairs[:2], titles[:2])):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        style_3d_axes(ax)

        # 网格
        padding = 0.5
        x_range = np.linspace(X_scaled[:, d1].min() - padding,
                             X_scaled[:, d1].max() + padding, 80)
        y_range = np.linspace(X_scaled[:, d2].min() - padding,
                             X_scaled[:, d2].max() + padding, 80)
        xx, yy = np.meshgrid(x_range, y_range)

        fixed_val = np.median(X_scaled[:, d_fixed])
        grid = np.zeros((xx.size, 3))
        grid[:, d1] = xx.ravel()
        grid[:, d2] = yy.ravel()
        grid[:, d_fixed] = fixed_val

        probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

        # 概率曲面
        cmap = LinearSegmentedColormap.from_list('prob',
            [class_colors[0], '#FFFFFF', class_colors[1]], N=256)
        surf = ax.plot_surface(xx, yy, probs, cmap=cmap, alpha=0.8,
                              linewidth=0, antialiased=True,
                              vmin=0, vmax=1)

        # 决策边界线 (p=0.5)
        ax.contour(xx, yy, probs, levels=[0.5], colors='white',
                  linewidths=3, linestyles='-', offset=0.5)

        # 底部投影
        ax.contourf(xx, yy, probs, levels=15, cmap=cmap, alpha=0.3,
                   offset=0, zdir='z')

        # 数据点投影到底部
        for class_id in range(2):
            mask = y == class_id
            ax.scatter(X_scaled[mask, d1], X_scaled[mask, d2],
                      np.zeros(mask.sum()), c=class_colors[class_id],
                      s=40, alpha=0.7, edgecolors='white', linewidths=0.5)

        ax.set_xlabel(feature_names[d1], fontsize=10, color='white')
        ax.set_ylabel(feature_names[d2], fontsize=10, color='white')
        ax.set_zlabel('Probability', fontsize=10, color='white')
        ax.set_zlim(0, 1)
        ax.set_title(f'{title}\n(fixed: {feature_names[d_fixed]} @ median)',
                    fontsize=12, fontweight='bold', color='white')
        ax.view_init(elev=25, azim=45)

    # 颜色条
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label(f'P({class_names[1]})', fontsize=12, color='white')
    cbar.ax.tick_params(colors='white')

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i],
               markersize=12, label=class_names[i], linestyle='None', markeredgecolor='white')
        for i in range(2)
    ]
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.08), ncol=2, fontsize=12,
              framealpha=0.95, facecolor=COLORS['background_alt'],
              edgecolor=COLORS['accent'])

    fig.suptitle(f'3D PROBABILITY SURFACE: {title_suffix.upper()}',
                fontsize=18, fontweight='bold', color='white', y=0.98)

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout(rect=[0, 0.1, 0.9, 0.95])

    filename = f'3d_probability_surface_{title_suffix.lower().replace(" ", "_")}.png'
    save_figure(fig, filename, tight=False)
    plt.show()
    reset_style()


def draw_3d_probability_dual_surface(X, y, class_names, class_colors, title_suffix, feature_names):
    """绘制双概率曲面（两个类别的概率）"""
    set_dark_style()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=1, max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    style_3d_axes(ax)

    # 使用前两个特征
    d1, d2 = 0, 1
    padding = 0.5
    x_range = np.linspace(X_scaled[:, d1].min() - padding,
                         X_scaled[:, d1].max() + padding, 60)
    y_range = np.linspace(X_scaled[:, d2].min() - padding,
                         X_scaled[:, d2].max() + padding, 60)
    xx, yy = np.meshgrid(x_range, y_range)

    fixed_val = np.median(X_scaled[:, 2])
    grid = np.zeros((xx.size, 3))
    grid[:, d1] = xx.ravel()
    grid[:, d2] = yy.ravel()
    grid[:, 2] = fixed_val

    probs = clf.predict_proba(grid)
    prob_0 = probs[:, 0].reshape(xx.shape)
    prob_1 = probs[:, 1].reshape(xx.shape)

    # Class 0 概率曲面
    cmap_0 = LinearSegmentedColormap.from_list('c0', ['#0d0d1a', class_colors[0]], N=256)
    surf_0 = ax.plot_surface(xx, yy, prob_0, cmap=cmap_0, alpha=0.6,
                            linewidth=0, antialiased=True)

    # Class 1 概率曲面
    cmap_1 = LinearSegmentedColormap.from_list('c1', ['#0d0d1a', class_colors[1]], N=256)
    surf_1 = ax.plot_surface(xx, yy, prob_1, cmap=cmap_1, alpha=0.6,
                            linewidth=0, antialiased=True)

    # 交界线 (p=0.5)
    ax.contour(xx, yy, prob_0, levels=[0.5], colors='white',
              linewidths=3, offset=0.5)

    # 数据点
    for class_id in range(2):
        mask = y == class_id
        prob_vals = clf.predict_proba(X_scaled[mask])[:, class_id]
        ax.scatter(X_scaled[mask, d1], X_scaled[mask, d2], prob_vals,
                  c=class_colors[class_id], s=80, alpha=0.9,
                  edgecolors='white', linewidths=1,
                  label=class_names[class_id])

    ax.set_xlabel(feature_names[d1], fontsize=12, color='white', labelpad=10)
    ax.set_ylabel(feature_names[d2], fontsize=12, color='white', labelpad=10)
    ax.set_zlabel('Probability', fontsize=12, color='white', labelpad=10)
    ax.set_zlim(0, 1)
    ax.view_init(elev=20, azim=45)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i],
               markersize=14, label=f'{class_names[i]} (P surface)',
               linestyle='None', markeredgecolor='white')
        for i in range(2)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
             framealpha=0.95, facecolor=COLORS['background_alt'],
             edgecolor=COLORS['accent'])

    ax.set_title(f'DUAL PROBABILITY SURFACES: {title_suffix.upper()}',
                fontsize=16, fontweight='bold', color='white', pad=20)

    fig.patch.set_facecolor(COLORS['background'])

    filename = f'3d_probability_dual_{title_suffix.lower().replace(" ", "_")}.png'
    save_figure(fig, filename)
    plt.show()
    reset_style()


def draw_3d_probability_contour_slices(X, y, class_names, class_colors, title_suffix, feature_names):
    """绘制多个z切片的概率等高线"""
    set_dark_style()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=1, max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    style_3d_axes(ax)

    d1, d2, d3 = 0, 1, 2
    padding = 0.3
    x_range = np.linspace(X_scaled[:, d1].min() - padding,
                         X_scaled[:, d1].max() + padding, 50)
    y_range = np.linspace(X_scaled[:, d2].min() - padding,
                         X_scaled[:, d2].max() + padding, 50)
    xx, yy = np.meshgrid(x_range, y_range)

    # 多个z切片
    z_slices = np.percentile(X_scaled[:, d3], [20, 40, 60, 80])

    cmap = LinearSegmentedColormap.from_list('prob',
        [class_colors[0], '#FFFFFF', class_colors[1]], N=256)

    for z_val in z_slices:
        grid = np.zeros((xx.size, 3))
        grid[:, d1] = xx.ravel()
        grid[:, d2] = yy.ravel()
        grid[:, d3] = z_val

        probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

        # 等高线填充
        ax.contourf(xx, yy, probs, levels=10, cmap=cmap, alpha=0.4,
                   offset=z_val, zdir='z')
        # 决策边界
        ax.contour(xx, yy, probs, levels=[0.5], colors='white',
                  linewidths=2, offset=z_val, zdir='z')

    # 数据点
    for class_id in range(2):
        mask = y == class_id
        ax.scatter(X_scaled[mask, d1], X_scaled[mask, d2], X_scaled[mask, d3],
                  c=class_colors[class_id], s=60, alpha=0.9,
                  edgecolors='white', linewidths=0.8,
                  label=class_names[class_id])

    ax.set_xlabel(feature_names[d1], fontsize=11, color='white', labelpad=8)
    ax.set_ylabel(feature_names[d2], fontsize=11, color='white', labelpad=8)
    ax.set_zlabel(feature_names[d3], fontsize=11, color='white', labelpad=8)
    ax.view_init(elev=25, azim=45)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i],
               markersize=12, label=class_names[i], linestyle='None', markeredgecolor='white')
        for i in range(2)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
             framealpha=0.95, facecolor=COLORS['background_alt'],
             edgecolor=COLORS['accent'])

    ax.set_title(f'3D PROBABILITY CONTOUR SLICES: {title_suffix.upper()}',
                fontsize=16, fontweight='bold', color='white', pad=15)

    fig.patch.set_facecolor(COLORS['background'])

    filename = f'3d_probability_slices_{title_suffix.lower().replace(" ", "_")}.png'
    save_figure(fig, filename)
    plt.show()
    reset_style()


if __name__ == "__main__":
    print("Loading Iris dataset...")
    X, y, feature_names = load_data_3features()

    # Setosa vs Others
    print("\n=== Scenario 1: Setosa vs Others ===")
    X1, y1, names1, colors1, title1 = prepare_binary_data(X, y, 'setosa_vs_others')

    print("1.1 Drawing 3D probability surface...")
    draw_3d_probability_surface(X1, y1, names1, colors1, title1, feature_names)

    print("1.2 Drawing dual probability surfaces...")
    draw_3d_probability_dual_surface(X1, y1, names1, colors1, title1, feature_names)

    print("1.3 Drawing probability contour slices...")
    draw_3d_probability_contour_slices(X1, y1, names1, colors1, title1, feature_names)

    # Versicolor vs Virginica
    print("\n=== Scenario 2: Versicolor vs Virginica ===")
    X2, y2, names2, colors2, title2 = prepare_binary_data(X, y, 'versicolor_vs_virginica')

    print("2.1 Drawing 3D probability surface...")
    draw_3d_probability_surface(X2, y2, names2, colors2, title2, feature_names)

    print("2.2 Drawing dual probability surfaces...")
    draw_3d_probability_dual_surface(X2, y2, names2, colors2, title2, feature_names)

    print("2.3 Drawing probability contour slices...")
    draw_3d_probability_contour_slices(X2, y2, names2, colors2, title2, feature_names)

    print("\nAll 3D probability plots completed!")
