# -*- coding: utf-8 -*-
"""
2D决策边界可视化 - 鸢尾花数据集 (任务一)
多分类器对比：Logistic Regression, Gradient Boosting, SVM, Random Forest
特征: Petal Length (x2) + Petal Width (x3)
分类: 三分类
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from config import (
    set_dark_style, reset_style, save_figure,
    COLORS, SPECIES_COLORS, SPECIES_NAMES, MAIN_CLASSIFIERS, FIGURE_DIR
)


def load_data_2features():
    """加载数据，只使用Petal Length和Petal Width两个特征"""
    iris = load_iris()
    X = iris.data[:, 2:4]  # Petal Length, Petal Width
    y = iris.target
    feature_names = ['Petal Length', 'Petal Width']
    return X, y, feature_names


def create_mesh(X, resolution=200, padding=0.5):
    """创建网格用于绘制决策边界"""
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    return xx, yy


def draw_decision_boundary_single(X, y, classifier, clf_name, feature_names):
    """绘制单个分类器的决策边界"""
    set_dark_style()

    # 训练模型
    clf = classifier
    clf.fit(X, y)
    accuracy = accuracy_score(y, clf.predict(X))

    # 创建网格
    xx, yy = create_mesh(X)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 获取概率
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        probs = probs.reshape(xx.shape[0], xx.shape[1], 3)
    else:
        probs = None

    # 创建图形
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # 颜色映射
    cmap_decision = ListedColormap(SPECIES_COLORS)

    # 1. 决策边界图
    ax = axes[0]
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_decision, levels=[-0.5, 0.5, 1.5, 2.5])
    ax.contour(xx, yy, Z, colors='white', linewidths=1.5, levels=[0.5, 1.5])

    for i in range(3):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=60,
                  edgecolors='white', linewidths=1, label=SPECIES_NAMES[i], zorder=5)

    ax.set_xlabel(feature_names[0], fontsize=12, color='white')
    ax.set_ylabel(feature_names[1], fontsize=12, color='white')
    ax.set_title(f'Decision Boundaries\nAccuracy: {accuracy:.2%}', fontsize=13,
                fontweight='bold', color='white')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
             facecolor=COLORS['background_alt'], edgecolor=COLORS['accent'])
    ax.set_facecolor(COLORS['background'])
    ax.tick_params(colors='white')

    # 2-4. 各类别概率图
    if probs is not None:
        for class_id in range(3):
            ax = axes[class_id + 1]
            prob_map = probs[:, :, class_id]

            # 创建渐变色图
            cmap = LinearSegmentedColormap.from_list(
                f'prob_{class_id}',
                [COLORS['background'], SPECIES_COLORS[class_id]],
                N=256
            )

            # 绘制概率热图
            im = ax.contourf(xx, yy, prob_map, levels=20, cmap=cmap, alpha=0.85)
            ax.contour(xx, yy, prob_map, levels=[0.5], colors='white',
                      linewidths=2, linestyles='--')

            # 数据点
            for i in range(3):
                mask = y == i
                ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=40,
                          edgecolors='white', linewidths=0.8, alpha=0.9, zorder=5)

            # 颜色条
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Probability', fontsize=10, color='white')
            cbar.ax.tick_params(colors='white', labelsize=9)

            ax.set_xlabel(feature_names[0], fontsize=12, color='white')
            ax.set_ylabel(feature_names[1], fontsize=12, color='white')
            ax.set_title(f'{SPECIES_NAMES[class_id]} Probability (Class {class_id})',
                        fontsize=13, fontweight='bold', color='white')
            ax.set_facecolor(COLORS['background'])
            ax.tick_params(colors='white')

    # 主标题
    fig.suptitle(f'{clf_name.upper()}: DECISION BOUNDARY & PROBABILITY MAPS',
                fontsize=18, fontweight='bold', color='white', y=1.02)

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout()

    filename = f'decision_boundary_2d_{clf_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    save_figure(fig, filename)
    plt.show()

    reset_style()
    return accuracy


def draw_classifier_comparison(X, y, feature_names):
    """绘制多分类器对比图"""
    set_dark_style()

    classifiers = {
        'Logistic Regression': LogisticRegression(C=1, max_iter=1000, random_state=42),
        'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    xx, yy = create_mesh(X)
    cmap_decision = ListedColormap(SPECIES_COLORS)

    for row_idx, (clf_name, clf) in enumerate(classifiers.items()):
        # 训练
        clf.fit(X, y)
        accuracy = accuracy_score(y, clf.predict(X))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            probs = probs.reshape(xx.shape[0], xx.shape[1], 3)
        else:
            probs = None

        # 各类别概率图
        for class_id in range(3):
            ax = axes[row_idx, class_id]

            if probs is not None:
                prob_map = probs[:, :, class_id]
                cmap = LinearSegmentedColormap.from_list(
                    f'prob_{class_id}',
                    [COLORS['background'], SPECIES_COLORS[class_id]],
                    N=256
                )
                ax.contourf(xx, yy, prob_map, levels=15, cmap=cmap, alpha=0.85)
                ax.contour(xx, yy, prob_map, levels=[0.5], colors='white',
                          linewidths=1.5, linestyles='--')

            for i in range(3):
                mask = y == i
                ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=25,
                          edgecolors='white', linewidths=0.5, alpha=0.9, zorder=5)

            ax.set_facecolor(COLORS['background'])
            ax.tick_params(colors='white', labelsize=8)

            if row_idx == 0:
                ax.set_title(f'{SPECIES_NAMES[class_id]}\nProbability',
                            fontsize=11, fontweight='bold', color='white')
            if class_id == 0:
                ax.set_ylabel(f'{clf_name}\n\n{feature_names[1]}',
                             fontsize=10, color='white')
            if row_idx == 3:
                ax.set_xlabel(feature_names[0], fontsize=10, color='white')

        # 决策边界图
        ax = axes[row_idx, 3]
        ax.contourf(xx, yy, Z, alpha=0.5, cmap=cmap_decision, levels=[-0.5, 0.5, 1.5, 2.5])
        ax.contour(xx, yy, Z, colors='white', linewidths=1.5, levels=[0.5, 1.5])

        for i in range(3):
            mask = y == i
            ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=25,
                      edgecolors='white', linewidths=0.5, zorder=5)

        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors='white', labelsize=8)

        if row_idx == 0:
            ax.set_title('Decision\nBoundary', fontsize=11, fontweight='bold', color='white')
        if row_idx == 3:
            ax.set_xlabel(feature_names[0], fontsize=10, color='white')

        # 准确率标注
        ax.text(0.98, 0.02, f'Acc: {accuracy:.1%}', transform=ax.transAxes,
               fontsize=10, fontweight='bold', color='white', ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['background_alt'],
                        edgecolor=COLORS['accent'], alpha=0.95))

    # 添加颜色条（概率）
    cax = fig.add_axes([0.125, 0.02, 0.5, 0.012])
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Probability', fontsize=11, color='white')
    cbar.ax.tick_params(colors='white', labelsize=9)

    # 图例
    legend_elements = [
        Patch(facecolor=SPECIES_COLORS[i], edgecolor='white', label=SPECIES_NAMES[i])
        for i in range(3)
    ]
    fig.legend(handles=legend_elements, loc='lower right',
              bbox_to_anchor=(0.98, 0.02), fontsize=11, ncol=3,
              framealpha=0.95, facecolor=COLORS['background_alt'],
              edgecolor=COLORS['accent'])

    fig.suptitle('CLASSIFIER COMPARISON: DECISION BOUNDARIES & PROBABILITY MAPS',
                fontsize=20, fontweight='bold', color='white', y=0.995)
    fig.text(0.5, 0.965, 'Features: Petal Length vs Petal Width | 3-Class Classification',
            fontsize=12, color=COLORS['text_secondary'], ha='center')

    fig.patch.set_facecolor(COLORS['background'])
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.15, wspace=0.15)

    save_figure(fig, 'decision_boundary_2d_comparison.png', tight=False)
    plt.show()

    reset_style()


def draw_probability_contours(X, y, feature_names):
    """绘制概率等高线图（更详细的概率分布）"""
    set_dark_style()

    clf = LogisticRegression(C=1, max_iter=1000, random_state=42)
    clf.fit(X, y)

    xx, yy = create_mesh(X, resolution=300)
    probs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    probs = probs.reshape(xx.shape[0], xx.shape[1], 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for class_id in range(3):
        ax = axes[class_id]
        prob_map = probs[:, :, class_id]

        cmap = LinearSegmentedColormap.from_list(
            f'prob_{class_id}',
            ['#0d0d1a', SPECIES_COLORS[class_id]],
            N=256
        )

        # 填充等高线
        cf = ax.contourf(xx, yy, prob_map, levels=20, cmap=cmap, alpha=0.9)

        # 等高线标签
        contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        cs = ax.contour(xx, yy, prob_map, levels=contour_levels,
                       colors='white', linewidths=1, alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f', colors='white')

        # 决策边界（0.5）
        ax.contour(xx, yy, prob_map, levels=[0.5], colors='white',
                  linewidths=3, linestyles='-')

        # 数据点
        for i in range(3):
            mask = y == i
            ax.scatter(X[mask, 0], X[mask, 1], c=SPECIES_COLORS[i], s=50,
                      edgecolors='white', linewidths=1, alpha=0.9, zorder=5)

        # 颜色条
        cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
        cbar.set_label('Probability', fontsize=11, color='white')
        cbar.ax.tick_params(colors='white', labelsize=10)

        ax.set_xlabel(feature_names[0], fontsize=12, color='white')
        ax.set_ylabel(feature_names[1], fontsize=12, color='white')
        ax.set_title(f'{SPECIES_NAMES[class_id]} (Class {class_id})\nProbability Distribution',
                    fontsize=14, fontweight='bold', color='white', pad=10)
        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors='white', labelsize=10)
        ax.grid(True, alpha=0.1, linestyle='--', color='white')

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SPECIES_COLORS[i],
               markersize=12, label=SPECIES_NAMES[i], linestyle='None', markeredgecolor='white')
        for i in range(3)
    ]
    legend_elements.append(
        Line2D([0], [0], color='white', linewidth=3, label='Decision Boundary (p=0.5)')
    )
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=11,
              framealpha=0.95, facecolor=COLORS['background_alt'],
              edgecolor=COLORS['accent'])

    fig.suptitle('PROBABILITY DISTRIBUTION MAPS (LOGISTIC REGRESSION)',
                fontsize=18, fontweight='bold', color='white', y=0.98)

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    save_figure(fig, 'probability_maps_2d_detailed.png')
    plt.show()

    reset_style()


if __name__ == "__main__":
    print("Loading Iris dataset (2 features)...")
    X, y, feature_names = load_data_2features()
    print(f"Data shape: {X.shape}")
    print(f"Features: {feature_names}")

    print("\n1. Drawing classifier comparison...")
    draw_classifier_comparison(X, y, feature_names)

    print("\n2. Drawing individual classifiers...")
    for clf_name, clf in MAIN_CLASSIFIERS.items():
        print(f"   - {clf_name}")
        draw_decision_boundary_single(X, y, clf, clf_name, feature_names)

    print("\n3. Drawing detailed probability contours...")
    draw_probability_contours(X, y, feature_names)

    print("\nAll 2D decision boundary plots completed!")
