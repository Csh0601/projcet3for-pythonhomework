# -*- coding: utf-8 -*-
"""
特征分布可视化 - 鸢尾花数据集
包含：小提琴图 + 箱线图 + 散点图组合，统计标注
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import load_iris

from config import (
    set_dark_style, reset_style, save_figure, add_title_with_subtitle,
    COLORS, SPECIES_COLORS, SPECIES_NAMES, FEATURE_COLORS, FIGURE_DIR
)


def load_iris_dataframe():
    """加载鸢尾花数据并转换为DataFrame"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width',
                                           'Petal Length', 'Petal Width'])
    df['Species'] = iris.target
    df['Species_Name'] = df['Species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df


def draw_feature_distribution_combined(df):
    """
    绘制高级特征分布分析图
    结合小提琴图 + 箱线图 + 散点图 + 统计标注
    """
    set_dark_style()

    # 准备数据 - 将宽格式转为长格式
    feature_cols = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    df_melted = df[feature_cols].melt(var_name='Feature', value_name='Value')

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 10))

    # 定义配色
    palette = dict(zip(feature_cols, FEATURE_COLORS))

    # 1. 绘制小提琴图（底层 - 显示密度分布）
    sns.violinplot(
        data=df_melted,
        x='Feature',
        y='Value',
        hue='Feature',
        palette=palette,
        inner=None,
        alpha=0.5,
        linewidth=0,
        legend=False,
        ax=ax
    )

    # 2. 绘制箱线图（中层 - 显示四分位数）
    sns.boxplot(
        data=df_melted,
        x='Feature',
        y='Value',
        hue='Feature',
        palette=palette,
        width=0.15,
        boxprops={'alpha': 0.8, 'edgecolor': 'white', 'linewidth': 1.5},
        whiskerprops={'color': 'white', 'linewidth': 1.5},
        capprops={'color': 'white', 'linewidth': 1.5},
        medianprops={'color': '#FF6B6B', 'linewidth': 2.5},
        fliersize=0,
        legend=False,
        ax=ax
    )

    # 3. 绘制散点图（顶层 - 显示每个数据点）
    np.random.seed(42)
    sns.stripplot(
        data=df_melted,
        x='Feature',
        y='Value',
        hue='Feature',
        palette=palette,
        size=5,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,
        jitter=0.25,
        legend=False,
        ax=ax
    )

    # 4. 添加均值点（菱形标记）
    df_features = df[feature_cols]
    means = df_features.mean()
    for i, (col, mean_val) in enumerate(means.items()):
        ax.scatter(i, mean_val, marker='D', s=120, c='white',
                   edgecolors=FEATURE_COLORS[i], linewidths=2, zorder=10)

    # 5. 添加统计标注
    stats = df_features.agg(['mean', 'std', 'min', 'max'])
    for i, col in enumerate(feature_cols):
        mu = stats.loc['mean', col]
        sigma = stats.loc['std', col]
        min_val = stats.loc['min', col]
        max_val = stats.loc['max', col]

        y_max = df_features[col].max()

        bbox_props = dict(
            boxstyle='round,pad=0.4',
            facecolor=FEATURE_COLORS[i],
            edgecolor='white',
            alpha=0.95,
            linewidth=1.5
        )
        ax.annotate(
            f'$\\mu$={mu:.2f}\n$\\sigma$={sigma:.2f}',
            xy=(i, y_max + 0.4),
            fontsize=11,
            fontweight='bold',
            color='white',
            ha='center',
            va='bottom',
            bbox=bbox_props
        )

    # 6. 设置标题和标签
    ax.set_title(
        'IRIS FEATURE DISTRIBUTION ANALYSIS',
        fontsize=22,
        fontweight='bold',
        color='white',
        pad=25
    )
    ax.text(
        0.5, 1.02,
        'Violin Plot + Box Plot + Strip Points | Statistical Overview',
        transform=ax.transAxes,
        fontsize=12,
        color=COLORS['text_secondary'],
        ha='center',
        va='bottom'
    )

    ax.set_xlabel('Features', fontsize=14, color='white', labelpad=12)
    ax.set_ylabel('Value (cm)', fontsize=14, color='white', labelpad=12)

    # 7. 设置刻度标签
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, fontsize=12, color='white')
    ax.tick_params(axis='y', colors='white', labelsize=11)

    # 8. 添加图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=FEATURE_COLORS[i],
               markersize=10, label=feature_cols[i], linestyle='None', markeredgecolor='white')
        for i in range(4)
    ]
    legend_elements.extend([
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markersize=10, label='Mean ($\\mu$)', linestyle='None', markeredgecolor='gray'),
        Line2D([0], [0], color='gray', linewidth=12, alpha=0.4, label='Density (Violin)'),
        Line2D([0], [0], color='#FF6B6B', linewidth=3, label='Median')
    ])
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.95,
        facecolor=COLORS['background_alt'],
        edgecolor=COLORS['accent']
    )

    # 9. 设置背景和网格
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])
    ax.grid(axis='y', alpha=0.25, linestyle='--', color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['border'])
    ax.spines['bottom'].set_color(COLORS['border'])

    plt.tight_layout()
    save_figure(fig, 'feature_distribution_combined.png')
    plt.show()

    reset_style()


def draw_feature_distribution_by_species(df):
    """
    按物种分组的特征分布图
    每个特征一行，显示三个物种的分布对比
    """
    set_dark_style()

    feature_cols = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]

        # 为每个物种绘制小提琴图和箱线图
        for species_id in range(3):
            subset = df[df['Species'] == species_id][feature]

            # 小提琴图
            parts = ax.violinplot(
                subset,
                positions=[species_id],
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.8
            )
            for pc in parts['bodies']:
                pc.set_facecolor(SPECIES_COLORS[species_id])
                pc.set_edgecolor('white')
                pc.set_alpha(0.4)
                pc.set_linewidth(1)

            # 箱线图
            bp = ax.boxplot(
                subset,
                positions=[species_id],
                widths=0.15,
                patch_artist=True,
                showfliers=False
            )
            for patch in bp['boxes']:
                patch.set_facecolor(SPECIES_COLORS[species_id])
                patch.set_edgecolor('white')
                patch.set_alpha(0.9)
            for element in ['whiskers', 'caps']:
                for line in bp[element]:
                    line.set_color('white')
                    line.set_linewidth(1.5)
            for median in bp['medians']:
                median.set_color('#FF6B6B')
                median.set_linewidth(2)

            # 散点
            jitter = np.random.uniform(-0.15, 0.15, len(subset))
            ax.scatter(
                species_id + jitter, subset,
                c=SPECIES_COLORS[species_id],
                s=25,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.3,
                zorder=5
            )

            # 均值标记
            mean_val = subset.mean()
            ax.scatter(species_id, mean_val, marker='D', s=80, c='white',
                      edgecolors=SPECIES_COLORS[species_id], linewidths=2, zorder=10)

        # 设置轴
        ax.set_title(feature, fontsize=14, fontweight='bold', color='white', pad=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels(SPECIES_NAMES, fontsize=11, color='white')
        ax.set_ylabel('Value (cm)', fontsize=11, color='white')
        ax.tick_params(axis='y', colors='white', labelsize=10)
        ax.grid(axis='y', alpha=0.2, linestyle='--', color='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['border'])
        ax.spines['bottom'].set_color(COLORS['border'])
        ax.set_facecolor(COLORS['background'])

    # 主标题
    fig.suptitle(
        'FEATURE DISTRIBUTION BY SPECIES',
        fontsize=20,
        fontweight='bold',
        color='white',
        y=1.02
    )
    fig.text(
        0.5, 0.99,
        'Violin + Box + Strip | Mean Markers (Diamond)',
        fontsize=11,
        color=COLORS['text_secondary'],
        ha='center'
    )

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SPECIES_COLORS[i],
               markersize=12, label=SPECIES_NAMES[i], linestyle='None', markeredgecolor='white')
        for i in range(3)
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        fontsize=11,
        framealpha=0.95,
        facecolor=COLORS['background_alt'],
        edgecolor=COLORS['accent']
    )

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout()
    save_figure(fig, 'feature_distribution_by_species.png')
    plt.show()

    reset_style()


def draw_boxplot_grid(df):
    """
    绘制4x1箱线图网格 (类似作业要求中的基础图)
    但增强为科研级别质量
    """
    set_dark_style()

    feature_cols = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]

        # 使用seaborn绘制增强箱线图
        sns.boxplot(
            data=df,
            x='Species',
            y=feature,
            hue='Species',
            palette=SPECIES_COLORS,
            width=0.5,
            linewidth=2,
            flierprops=dict(marker='o', markerfacecolor='white', markersize=5, alpha=0.6),
            ax=ax,
            legend=False
        )

        # 叠加散点
        sns.stripplot(
            data=df,
            x='Species',
            y=feature,
            hue='Species',
            palette=SPECIES_COLORS,
            size=4,
            alpha=0.5,
            jitter=0.2,
            ax=ax,
            legend=False
        )

        ax.set_title(f'{feature} by Species', fontsize=13, fontweight='bold',
                    color='white', pad=10)
        ax.set_xlabel('Species', fontsize=11, color='white')
        ax.set_ylabel(f'{feature} (cm)', fontsize=11, color='white')
        ax.set_xticklabels(SPECIES_NAMES, fontsize=10, color='white')
        ax.tick_params(axis='y', colors='white', labelsize=10)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['border'])
        ax.spines['bottom'].set_color(COLORS['border'])
        ax.set_facecolor(COLORS['background'])

    fig.suptitle('IRIS FEATURES: BOXPLOT ANALYSIS', fontsize=18,
                fontweight='bold', color='white', y=1.01)

    fig.patch.set_facecolor(COLORS['background'])
    plt.tight_layout()
    save_figure(fig, 'feature_boxplot_grid.png')
    plt.show()

    reset_style()


if __name__ == "__main__":
    print("Loading Iris dataset...")
    df = load_iris_dataframe()
    print(f"Dataset shape: {df.shape}")
    print(f"Species distribution:\n{df['Species_Name'].value_counts()}")

    print("\n1. Drawing combined feature distribution...")
    draw_feature_distribution_combined(df)

    print("\n2. Drawing feature distribution by species...")
    draw_feature_distribution_by_species(df)

    print("\n3. Drawing boxplot grid...")
    draw_boxplot_grid(df)

    print("\nAll feature distribution plots completed!")
