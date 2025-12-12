# -*- coding: utf-8 -*-
"""
2D概率图可视化 - 鸢尾花数据集 (任务一补充)
详细的概率热图、不确定性区域、多分类器对比
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

from config import (
    set_dark_style, reset_style, save_figure,
    COLORS, SPECIES_COLORS, SPECIES_NAMES, FIGURE_DIR
)


def load_data_2features():
    """加载数据，使用Petal Length和Petal Width"""
    iris = load_iris()
    X = iris.data[:, 2:4]
    y = iris.target
    return X, y, ['Petal Length', 'Petal Width']


def create_mesh(X, resolution=250, padding=0.5):
    """创建网格"""
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    return xx, yy


def draw_uncertainty_map(X, y, feature_names):
    """
    绘制不确定性区域图
    显示分类器最不确定的区域（熵最高的区域）
    """
    set_dark_style()

    clf = LogisticRegression(C=1, max_iter=1000, random_state=42)
    clf.fit(X, y)

    xx, yy = create_mesh(X, resolution=300)
    probs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # 计算熵（不确定性）
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    entropy = entropy.reshape(xx.shape)
    max_entropy = np.log(3)  # 三分类的最大熵

    # 获取预测类别
    predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 决策边界 + 不确定性叠加
    ax = axes[0]
    cmap_decision = ListedColormap(SPECIES_COLORS)
    ax.contourf(xx, yy, predictions, alpha=0.3, cmap=cmap_decision, levels=[-0.5, 0.5, 1.5, 2.5])

    # 不确定性等高线
    cs = ax.contour(xx, yy, entropy, levels=10, cmap='Reds', linewidths=1.5, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

    for i in range(3):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=60,
                  edgecolors='white', linewidths=1, label=SPECIES_NAMES[i], zorder=5)

    ax.set_xlabel(feature_names[0], fontsize=12, color='white')
    ax.set_ylabel(feature_names[1], fontsize=12, color='white')
    ax.set_title('Decision Boundary + Uncertainty Contours', fontsize=13,
                fontweight='bold', color='white')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
             facecolor=COLORS['background_alt'], edgecolor=COLORS['accent'])
    ax.set_facecolor(COLORS['background'])
    ax.tick_params(colors='white')

    # 2. 纯不确定性热图
    ax = axes[1]
    cmap_entropy = LinearSegmentedColormap.from_list(
        'entropy', ['#0d0d1a', '#1a1a4e', '#4a2c7a', '#8B0000', '#FF4500'], N=256
    )
    im = ax.contourf(xx, yy, entropy, levels=20, cmap=cmap_entropy)

    # 高不确定区域边界
    ax.contour(xx, yy, entropy, levels=[max_entropy * 0.7], colors='white',
              linewidths=2, linestyles='--')

    for i in range(3):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=50,
                  edgecolors='white', linewidths=0.8, zorder=5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Entropy (Uncertainty)', fontsize=11, color='white')
    cbar.ax.tick_params(colors='white')

    ax.set_xlabel(feature_names[0], fontsize=12, color='white')
    ax.set_ylabel(feature_names[1], fontsize=12, color='white')
    ax.set_title('Uncertainty Heatmap\n(Shannon Entropy)', fontsize=13,
                fontweight='bold', color='white')
    ax.set_facecolor(COLORS['background'])
    ax.tick_params(colors='white')

    # 3. 最高概率热图（置信度）
    ax = axes[2]
    max_probs = np.max(probs, axis=1).reshape(xx.shape)

    cmap_conf = LinearSegmentedColormap.from_list(
        'confidence', ['#FF4500', '#FFD700', '#00FF88'], N=256
    )
    im = ax.contourf(xx, yy, max_probs, levels=20, cmap=cmap_conf)

    # 低置信度区域（< 0.5）
    ax.contour(xx, yy, max_probs, levels=[0.5], colors='white',
              linewidths=2.5, linestyles='-')
    ax.contour(xx, yy, max_probs, levels=[0.7], colors='white',
              linewidths=1.5, linestyles='--')

    for i in range(3):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=50,
                  edgecolors='white', linewidths=0.8, zorder=5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Max Probability (Confidence)', fontsize=11, color='white')
    cbar.ax.tick_params(colors='white')

    ax.set_xlabel(feature_names[0], fontsize=12, color='white')
    ax.set_ylabel(feature_names[1], fontsize=12, color='white')
    ax.set_title('Confidence Heatmap\n(Maximum Class Probability)', fontsize=13,
                fontweight='bold', color='white')
    ax.set_facecolor(COLORS['background'])
    ax.tick_params(colors='white')

    fig.suptitle('CLASSIFICATION UNCERTAINTY ANALYSIS',
                fontsize=18, fontweight='bold', color='white', y=0.98)

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, 'uncertainty_analysis_2d.png')
    plt.show()

    reset_style()


def draw_probability_comparison_by_class(X, y, feature_names):
    """
    对比不同分类器对每个类别的概率预测
    """
    set_dark_style()

    classifiers = {
        'Logistic\nRegression': LogisticRegression(C=1, max_iter=1000, random_state=42),
        'Gradient\nBoosting': HistGradientBoostingClassifier(random_state=42),
        'SVM\n(RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random\nForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    xx, yy = create_mesh(X, resolution=200)

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.25, wspace=0.2)

    for class_id in range(3):
        for clf_idx, (clf_name, clf) in enumerate(classifiers.items()):
            ax = fig.add_subplot(gs[class_id, clf_idx])

            clf.fit(X, y)
            probs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            prob_map = probs[:, class_id].reshape(xx.shape)

            cmap = LinearSegmentedColormap.from_list(
                f'prob_{class_id}',
                [COLORS['background'], SPECIES_COLORS[class_id]],
                N=256
            )

            im = ax.contourf(xx, yy, prob_map, levels=15, cmap=cmap, alpha=0.9)
            ax.contour(xx, yy, prob_map, levels=[0.5], colors='white',
                      linewidths=2, linestyles='-')
            ax.contour(xx, yy, prob_map, levels=[0.3, 0.7], colors='white',
                      linewidths=1, linestyles='--', alpha=0.7)

            for i in range(3):
                mask = y == i
                ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=25,
                          edgecolors='white', linewidths=0.5, alpha=0.8, zorder=5)

            ax.set_facecolor(COLORS['background'])
            ax.tick_params(colors='white', labelsize=8)

            if class_id == 0:
                ax.set_title(clf_name, fontsize=11, fontweight='bold', color='white', pad=10)

            if clf_idx == 0:
                ax.set_ylabel(f'{SPECIES_NAMES[class_id]}\n(Class {class_id})\n\n{feature_names[1]}',
                             fontsize=10, color='white')

            if class_id == 2:
                ax.set_xlabel(feature_names[0], fontsize=10, color='white')

    # 添加颜色条
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Class Probability', fontsize=12, color='white', labelpad=15)
    cbar.ax.tick_params(colors='white', labelsize=10)

    # 图例
    legend_elements = [
        Patch(facecolor=SPECIES_COLORS[i], edgecolor='white', label=SPECIES_NAMES[i])
        for i in range(3)
    ]
    legend_elements.extend([
        Line2D([0], [0], color='white', linewidth=2, label='p = 0.5'),
        Line2D([0], [0], color='white', linewidth=1, linestyle='--', label='p = 0.3, 0.7'),
    ])
    fig.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.45, 0.02), ncol=5, fontsize=10,
              framealpha=0.95, facecolor=COLORS['background_alt'],
              edgecolor=COLORS['accent'])

    fig.suptitle('CLASS PROBABILITY COMPARISON ACROSS CLASSIFIERS',
                fontsize=20, fontweight='bold', color='white', y=0.97)
    fig.text(0.45, 0.935, 'Rows: Classes | Columns: Classifiers',
            fontsize=12, color=COLORS['text_secondary'], ha='center')

    fig.patch.set_facecolor(COLORS['background'])

    save_figure(fig, 'probability_comparison_by_class.png', tight=False)
    plt.show()

    reset_style()


def draw_decision_regions_detailed(X, y, feature_names):
    """
    绘制详细的决策区域图
    包含：决策边界、等概率线、支持向量标注
    """
    set_dark_style()

    clf = SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
    clf.fit(X, y)

    xx, yy = create_mesh(X, resolution=300)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    probs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots(figsize=(12, 10))

    # 决策区域（半透明）
    cmap_decision = ListedColormap([
        f'{SPECIES_COLORS[0]}80',  # 带透明度
        f'{SPECIES_COLORS[1]}80',
        f'{SPECIES_COLORS[2]}80'
    ])

    # 使用概率最大值作为颜色强度
    max_probs = np.max(probs, axis=1).reshape(xx.shape)

    # 绘制决策区域
    for class_id in range(3):
        class_mask = Z == class_id
        prob_class = np.zeros_like(max_probs)
        prob_class[class_mask] = max_probs[class_mask]

        cmap = LinearSegmentedColormap.from_list(
            f'region_{class_id}',
            ['#0d0d1a00', SPECIES_COLORS[class_id]],  # 透明到颜色
            N=256
        )
        ax.contourf(xx, yy, np.where(class_mask, prob_class, np.nan),
                   levels=10, cmap=cmap, alpha=0.7)

    # 决策边界（粗白线）
    ax.contour(xx, yy, Z, colors='white', linewidths=3, levels=[0.5, 1.5])

    # 等概率线
    for class_id in range(3):
        prob_map = probs[:, class_id].reshape(xx.shape)
        ax.contour(xx, yy, prob_map, levels=[0.5], colors=SPECIES_COLORS[class_id],
                  linewidths=2, linestyles='--')

    # 支持向量标注
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors='none',
              edgecolors='white', linewidths=2, label='Support Vectors', zorder=6)

    # 数据点
    for i in range(3):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=80,
                  edgecolors='white', linewidths=1.2, label=SPECIES_NAMES[i], zorder=5)

    # 质心标注
    for i in range(3):
        mask = y == i
        cx, cy = X[mask, 0].mean(), X[mask, 1].mean()
        ax.scatter(cx, cy, marker='X', s=200, c='white',
                  edgecolors=SPECIES_COLORS[i], linewidths=2, zorder=10)
        ax.annotate(f'{SPECIES_NAMES[i]}\nCentroid', xy=(cx, cy),
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=SPECIES_COLORS[i],
                            edgecolor='white', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color='white', linewidth=1.5))

    ax.set_xlabel(feature_names[0], fontsize=14, color='white', labelpad=10)
    ax.set_ylabel(feature_names[1], fontsize=14, color='white', labelpad=10)
    ax.set_title('SVM (RBF) DECISION REGIONS\nwith Support Vectors & Class Boundaries',
                fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_facecolor(COLORS['background'])
    ax.tick_params(colors='white', labelsize=11)
    ax.grid(True, alpha=0.15, linestyle='--', color='white')

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SPECIES_COLORS[i],
               markersize=12, label=SPECIES_NAMES[i], linestyle='None', markeredgecolor='white')
        for i in range(3)
    ]
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=12, label='Support Vectors', linestyle='None',
               markeredgecolor='white', markeredgewidth=2),
        Line2D([0], [0], color='white', linewidth=3, label='Decision Boundary'),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='p = 0.5 Contour'),
    ])
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             framealpha=0.95, facecolor=COLORS['background_alt'],
             edgecolor=COLORS['accent'])

    # 添加信息框
    n_sv = len(sv)
    info_text = f'Support Vectors: {n_sv}\nKernel: RBF\nC: 1.0'
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
           fontsize=10, color='white', ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background_alt'],
                    edgecolor=COLORS['accent'], alpha=0.95), family='monospace')

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout()

    save_figure(fig, 'decision_regions_detailed_svm.png')
    plt.show()

    reset_style()


if __name__ == "__main__":
    print("Loading Iris dataset (2 features)...")
    X, y, feature_names = load_data_2features()

    print("\n1. Drawing uncertainty analysis...")
    draw_uncertainty_map(X, y, feature_names)

    print("\n2. Drawing probability comparison by class...")
    draw_probability_comparison_by_class(X, y, feature_names)

    print("\n3. Drawing detailed decision regions (SVM)...")
    draw_decision_regions_detailed(X, y, feature_names)

    print("\nAll 2D probability map plots completed!")
