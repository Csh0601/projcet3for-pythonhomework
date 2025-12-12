# -*- coding: utf-8 -*-
"""
特征关系分析可视化 - 鸢尾花数据集
包含：Pairplot配对图、相关性热力图、高级2D散点图
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
from matplotlib.patches import Patch, Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as transforms
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.spatial import ConvexHull
from sklearn.datasets import load_iris

from config import (
    set_dark_style, reset_style, save_figure,
    COLORS, SPECIES_COLORS, SPECIES_NAMES, FIGURE_DIR, CMAP_DIVERGING
)


def load_iris_dataframe():
    """加载鸢尾花数据并转换为DataFrame"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width',
                                           'Petal Length', 'Petal Width'])
    df['Species'] = iris.target
    return df


def draw_pairplot(df):
    """
    绘制高级特征配对分析图
    对角线: KDE密度曲线
    下三角: 散点图
    上三角: KDE等高线 + 相关系数
    """
    set_dark_style()

    palette = dict(zip(range(3), SPECIES_COLORS))

    # 创建PairGrid
    g = sns.PairGrid(
        df,
        hue='Species',
        palette=palette,
        diag_sharey=False,
        corner=False,
        height=2.8,
        aspect=1
    )

    # 设置背景
    g.figure.patch.set_facecolor(COLORS['background'])
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_facecolor(COLORS['background'])

    # 对角线: KDE密度曲线
    def diag_kde(x, **kwargs):
        color = kwargs.get('color', SPECIES_COLORS[0])
        sns.kdeplot(x, color=color, fill=True, alpha=0.3, linewidth=2,
                   **{k: v for k, v in kwargs.items() if k not in ['color']})
        sns.kdeplot(x, color=color, fill=False, linewidth=2.5,
                   **{k: v for k, v in kwargs.items() if k not in ['color']})

    g.map_diag(diag_kde)

    # 下三角: 散点图
    def lower_scatter(x, y, **kwargs):
        color = kwargs.get('color', SPECIES_COLORS[0])
        plt.scatter(x, y, color=color, s=40, alpha=0.7,
                   edgecolor='white', linewidth=0.5)

    g.map_lower(lower_scatter)

    # 上三角: KDE等高线
    def upper_kde(x, y, **kwargs):
        color = kwargs.get('color', SPECIES_COLORS[0])
        try:
            sns.kdeplot(x=x, y=y, color=color, levels=4, linewidths=1.5, alpha=0.8)
        except:
            plt.scatter(x, y, color=color, s=20, alpha=0.5)

    g.map_upper(upper_kde)

    # 添加相关系数
    feature_cols = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    for i, row_var in enumerate(feature_cols):
        for j, col_var in enumerate(feature_cols):
            if i < j:
                ax = g.axes[i, j]
                corr, p_value = stats.pearsonr(df[row_var], df[col_var])
                if abs(corr) > 0.7:
                    corr_color = COLORS['positive']
                elif abs(corr) > 0.4:
                    corr_color = COLORS['neutral']
                else:
                    corr_color = COLORS['text_secondary']

                ax.annotate(
                    f'r = {corr:.2f}',
                    xy=(0.95, 0.95),
                    xycoords='axes fraction',
                    fontsize=10,
                    fontweight='bold',
                    color=corr_color,
                    ha='right',
                    va='top',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor=COLORS['background_alt'],
                        edgecolor=corr_color,
                        alpha=0.95
                    )
                )

    # 美化轴
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(COLORS['border'])
            ax.spines['bottom'].set_color(COLORS['border'])
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)

    # 标题
    g.figure.suptitle(
        'PAIRWISE FEATURE ANALYSIS',
        fontsize=22,
        fontweight='bold',
        color='white',
        y=1.02
    )
    g.figure.text(
        0.5, 0.99,
        'Scatter Matrix | KDE Contours | Correlation Coefficients',
        fontsize=12,
        color=COLORS['text_secondary'],
        ha='center'
    )

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SPECIES_COLORS[i],
               markersize=10, label=SPECIES_NAMES[i], linestyle='None', markeredgecolor='white')
        for i in range(3)
    ]
    legend_elements.extend([
        Line2D([0], [0], color=COLORS['positive'], linewidth=3, label='Strong (|r|>0.7)'),
        Line2D([0], [0], color=COLORS['neutral'], linewidth=3, label='Moderate (|r|>0.4)'),
        Line2D([0], [0], color=COLORS['text_secondary'], linewidth=3, label='Weak'),
    ])

    g.figure.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        fontsize=10,
        framealpha=0.95,
        facecolor=COLORS['background_alt'],
        edgecolor=COLORS['accent'],
        title='Species & Correlation',
        title_fontsize=11
    )
    g.figure.legends[0].get_title().set_color('white')

    plt.tight_layout()
    save_figure(g.figure, 'pairplot_analysis.png')
    plt.show()

    reset_style()


def draw_correlation_heatmap(df):
    """
    绘制相关性热力图 + 层次聚类
    """
    set_dark_style()

    # 计算相关矩阵
    feature_cols = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    df_features = df[feature_cols]
    corr = df_features.corr()

    # 计算p值
    n = len(df_features)
    p_matrix = np.zeros((4, 4))
    for i, col1 in enumerate(feature_cols):
        for j, col2 in enumerate(feature_cols):
            if i != j:
                _, p_val = stats.pearsonr(df_features[col1], df_features[col2])
                p_matrix[i, j] = p_val

    # 层次聚类
    dissimilarity = 1 - np.abs(corr.values)
    np.fill_diagonal(dissimilarity, 0)
    dissimilarity = (dissimilarity + dissimilarity.T) / 2
    condensed_dist = squareform(dissimilarity)
    linkage = hierarchy.linkage(condensed_dist, method='average')
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    order = dendro['leaves']

    corr_ordered = corr.iloc[order, order]
    p_matrix_ordered = p_matrix[np.ix_(order, order)]

    # 创建布局
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(10, 12, figure=fig, hspace=0.3, wspace=0.3)

    ax_main = fig.add_subplot(gs[1:9, 2:10])
    ax_dendro_top = fig.add_subplot(gs[0:1, 2:10])
    ax_dendro_left = fig.add_subplot(gs[1:9, 0:1])
    ax_cbar = fig.add_subplot(gs[1:9, 11:12])

    # 树状图
    ax_dendro_top.set_facecolor(COLORS['background'])
    hierarchy.dendrogram(linkage, ax=ax_dendro_top, orientation='top',
                        color_threshold=0, above_threshold_color=COLORS['accent'])
    ax_dendro_top.axis('off')

    ax_dendro_left.set_facecolor(COLORS['background'])
    hierarchy.dendrogram(linkage, ax=ax_dendro_left, orientation='left',
                        color_threshold=0, above_threshold_color=COLORS['accent'])
    ax_dendro_left.axis('off')

    # 热力图
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8',
         '#FFFFFF',
         '#FFCCD5', '#FF8FA3', '#FF4D6D', '#C9184A'],
        N=256
    )

    ax_main.set_facecolor(COLORS['background'])
    im = ax_main.imshow(corr_ordered.values, cmap=custom_cmap, vmin=-1, vmax=1, aspect='auto')

    # 网格线
    for i in range(len(corr_ordered) + 1):
        ax_main.axhline(i - 0.5, color=COLORS['grid'], linewidth=0.5)
        ax_main.axvline(i - 0.5, color=COLORS['grid'], linewidth=0.5)

    # 标注
    for i in range(len(corr_ordered)):
        for j in range(len(corr_ordered)):
            value = corr_ordered.iloc[i, j]
            p_val = p_matrix_ordered[i, j]

            text_color = 'white' if abs(value) > 0.5 else COLORS['text_secondary']

            if i != j:
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = ''
            else:
                sig = ''

            ax_main.text(j, i, f'{value:.2f}', ha='center', va='center',
                        fontsize=14, fontweight='bold', color=text_color)
            if sig:
                ax_main.text(j, i + 0.28, sig, ha='center', va='center',
                            fontsize=10, fontweight='bold', color=COLORS['neutral'])

    # 刻度
    ordered_labels = [corr_ordered.columns[i] for i in range(len(corr_ordered))]
    ax_main.set_xticks(range(len(corr_ordered)))
    ax_main.set_yticks(range(len(corr_ordered)))
    ax_main.set_xticklabels(ordered_labels, fontsize=11, color='white', rotation=45, ha='right')
    ax_main.set_yticklabels(ordered_labels, fontsize=11, color='white')

    # 颜色条
    ax_cbar.set_facecolor(COLORS['background'])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label('Pearson Correlation (r)', fontsize=12, color='white', labelpad=15)
    cbar.ax.tick_params(colors='white', labelsize=10)
    cbar.outline.set_edgecolor(COLORS['border'])
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # 标题
    fig.suptitle('CORRELATION MATRIX WITH HIERARCHICAL CLUSTERING',
                fontsize=18, fontweight='bold', color='white', y=0.98)
    fig.text(0.5, 0.94, 'Pearson Coefficients | * p<0.05, ** p<0.01, *** p<0.001',
            fontsize=11, color=COLORS['text_secondary'], ha='center')

    # 图例
    legend_elements = [
        Patch(facecolor='#C9184A', edgecolor='white', label='Strong Positive'),
        Patch(facecolor='#FFFFFF', edgecolor=COLORS['border'], label='No Correlation'),
        Patch(facecolor='#0077B6', edgecolor='white', label='Strong Negative'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=10,
              framealpha=0.95, facecolor=COLORS['background_alt'],
              edgecolor=COLORS['accent'])

    fig.patch.set_facecolor(COLORS['background'])
    save_figure(fig, 'correlation_heatmap.png')
    plt.show()

    reset_style()


def draw_scatter_2d_advanced(df):
    """
    高级2D散点图：置信椭圆 + 凸包 + 边缘分布 + 质心标注
    """
    set_dark_style()

    x_col = "Petal Length"
    y_col = "Petal Width"

    # 创建布局
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    for ax in [ax_main, ax_top, ax_right]:
        ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])

    # KDE等高线背景
    for species_id in range(3):
        subset = df[df['Species'] == species_id]
        try:
            sns.kdeplot(x=subset[x_col], y=subset[y_col], ax=ax_main,
                       levels=3, color=SPECIES_COLORS[species_id],
                       alpha=0.15, fill=True, linewidths=0)
        except:
            pass

    # 凸包
    for species_id in range(3):
        subset = df[df['Species'] == species_id]
        points = subset[[x_col, y_col]].values
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_points = np.append(hull.vertices, hull.vertices[0])
                ax_main.plot(points[hull_points, 0], points[hull_points, 1],
                           color=SPECIES_COLORS[species_id], linestyle='--',
                           linewidth=1.5, alpha=0.6)
                ax_main.fill(points[hull_points, 0], points[hull_points, 1],
                           color=SPECIES_COLORS[species_id], alpha=0.05)
            except:
                pass

    # 置信椭圆
    def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
        if len(x) < 2:
            return None
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                         facecolor='none', **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_x, mean_y = np.mean(x), np.mean(y)
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    for species_id in range(3):
        subset = df[df['Species'] == species_id]
        confidence_ellipse(subset[x_col], subset[y_col], ax_main,
                          n_std=2.0, edgecolor=SPECIES_COLORS[species_id],
                          linewidth=2.5, alpha=0.8)

    # 散点图
    for species_id in range(3):
        subset = df[df['Species'] == species_id]
        ax_main.scatter(subset[x_col], subset[y_col], c=SPECIES_COLORS[species_id],
                       s=80, alpha=0.85, edgecolors='white', linewidths=0.8,
                       label=SPECIES_NAMES[species_id], zorder=5)
        ax_main.scatter(subset[x_col], subset[y_col], c=SPECIES_COLORS[species_id],
                       s=200, alpha=0.15, edgecolors='none', zorder=4)

    # 质心
    label_offsets = {0: (-60, -40), 1: (50, -50), 2: (50, 40)}
    for species_id in range(3):
        subset = df[df['Species'] == species_id]
        cx, cy = subset[x_col].mean(), subset[y_col].mean()
        ax_main.scatter(cx, cy, marker='X', s=250, c=SPECIES_COLORS[species_id],
                       edgecolors='white', linewidths=2, zorder=10)
        offset = label_offsets[species_id]
        ax_main.annotate(f'{SPECIES_NAMES[species_id]}\nCentroid',
                        xy=(cx, cy), xytext=offset, textcoords='offset points',
                        fontsize=9, fontweight='bold', color='white', ha='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor=SPECIES_COLORS[species_id],
                                 edgecolor='white', linewidth=1.5, alpha=0.95),
                        arrowprops=dict(arrowstyle='->', color='white', linewidth=1.5,
                                       connectionstyle='arc3,rad=0.2'), zorder=15)

    # 统计信息
    stats_text = []
    for species_id in range(3):
        subset = df[df['Species'] == species_id]
        corr, _ = stats.pearsonr(subset[x_col], subset[y_col])
        stats_text.append(f"{SPECIES_NAMES[species_id]}: n={len(subset)}, r={corr:.3f}")
    total_corr, _ = stats.pearsonr(df[x_col], df[y_col])
    stats_text.append(f"Overall: r={total_corr:.3f}")
    ax_main.text(0.02, 0.98, '\n'.join(stats_text), transform=ax_main.transAxes,
                fontsize=10, color='white', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background_alt'],
                         edgecolor=COLORS['accent'], alpha=0.95), family='monospace')

    # 边缘分布
    for species_id in range(3):
        subset = df[df['Species'] == species_id]
        sns.kdeplot(subset[x_col], ax=ax_top, color=SPECIES_COLORS[species_id],
                   fill=True, alpha=0.4, linewidth=2)
        sns.kdeplot(y=subset[y_col], ax=ax_right, color=SPECIES_COLORS[species_id],
                   fill=True, alpha=0.4, linewidth=2)

    for ax in [ax_top, ax_right]:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 主图美化
    ax_main.set_xlabel('Petal Length (cm)', fontsize=14, color='white', labelpad=10)
    ax_main.set_ylabel('Petal Width (cm)', fontsize=14, color='white', labelpad=10)
    ax_main.tick_params(colors='white', labelsize=11)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['left'].set_color(COLORS['border'])
    ax_main.spines['bottom'].set_color(COLORS['border'])
    ax_main.grid(True, alpha=0.15, linestyle='--', color='white')

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SPECIES_COLORS[i],
               markersize=12, label=SPECIES_NAMES[i], linestyle='None', markeredgecolor='white')
        for i in range(3)
    ]
    legend_elements.extend([
        Line2D([0], [0], marker='X', color='w', markerfacecolor='white',
               markersize=12, label='Centroid', linestyle='None'),
        Patch(facecolor='none', edgecolor='white', linewidth=2, label='95% Confidence'),
        Line2D([0], [0], color='white', linestyle='--', linewidth=1.5, label='Convex Hull'),
    ])
    ax_main.legend(handles=legend_elements, loc='lower right', fontsize=10,
                  framealpha=0.95, facecolor=COLORS['background_alt'],
                  edgecolor=COLORS['accent'])

    fig.suptitle('FEATURE SPACE ANALYSIS: PETAL DIMENSIONS',
                fontsize=20, fontweight='bold', color='white', y=0.98)
    fig.text(0.45, 0.94, 'Scatter | 95% Confidence Ellipse | Marginal KDE | Centroids',
            fontsize=11, color=COLORS['text_secondary'], ha='center')

    save_figure(fig, 'scatter_2d_advanced.png')
    plt.show()

    reset_style()


if __name__ == "__main__":
    print("Loading Iris dataset...")
    df = load_iris_dataframe()

    print("\n1. Drawing pairplot...")
    draw_pairplot(df)

    print("\n2. Drawing correlation heatmap...")
    draw_correlation_heatmap(df)

    print("\n3. Drawing advanced 2D scatter...")
    draw_scatter_2d_advanced(df)

    print("\nAll feature analysis plots completed!")
