# -*- coding: utf-8 -*-
"""
全局配置文件 - 鸢尾花数据分类与可视化项目
科研级可视化配置：配色方案、字体、样式、分类器等
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')

# 确保输出目录存在
os.makedirs(FIGURE_DIR, exist_ok=True)

# ============================================================
# 科研级配色方案 (深色主题)
# ============================================================
COLORS = {
    # 物种颜色
    'setosa': '#00CED1',       # 青色 (Dark Turquoise)
    'versicolor': '#FFD700',   # 金色 (Gold)
    'virginica': '#FF6B6B',    # 珊瑚红 (Coral Red)

    # 背景和界面颜色
    'background': '#0d0d1a',   # 深蓝黑色
    'background_alt': '#1a1a2e', # 稍浅的背景
    'text': '#FFFFFF',         # 白色文字
    'text_secondary': '#AAAAAA', # 次要文字
    'grid': '#2a2a4a',         # 网格线
    'border': '#444444',       # 边框

    # 强调色
    'accent': '#00CED1',       # 主强调色
    'positive': '#00FF88',     # 正向/强相关
    'neutral': '#FFD700',      # 中性
    'negative': '#FF4D6D',     # 负向/强负相关
}

# 物种颜色列表 (按索引顺序)
SPECIES_COLORS = [COLORS['setosa'], COLORS['versicolor'], COLORS['virginica']]
SPECIES_NAMES = ['Setosa', 'Versicolor', 'Virginica']

# 特征颜色
FEATURE_COLORS = ['#00CED1', '#2E8B57', '#DAA520', '#CD5C5C']
FEATURE_NAMES = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# ============================================================
# 自定义颜色映射
# ============================================================
def create_species_cmap(species_color, reverse=False):
    """创建从白色到物种颜色的渐变色图"""
    if reverse:
        return LinearSegmentedColormap.from_list('custom', [species_color, 'white'], N=256)
    return LinearSegmentedColormap.from_list('custom', ['white', species_color], N=256)

def create_diverging_cmap():
    """创建蓝-白-红发散色图"""
    return LinearSegmentedColormap.from_list(
        'diverging',
        ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8',
         '#FFFFFF',
         '#FFCCD5', '#FF8FA3', '#FF4D6D', '#C9184A'],
        N=256
    )

def create_probability_cmap(color):
    """创建概率热图的渐变色图"""
    return LinearSegmentedColormap.from_list(
        'probability',
        ['#0d0d1a', color],
        N=256
    )

# 预定义的颜色映射
CMAP_SETOSA = create_species_cmap(COLORS['setosa'])
CMAP_VERSICOLOR = create_species_cmap(COLORS['versicolor'])
CMAP_VIRGINICA = create_species_cmap(COLORS['virginica'])
CMAP_DIVERGING = create_diverging_cmap()

# ============================================================
# 分类器配置
# ============================================================
CLASSIFIERS = {
    'Logistic Regression\n(C=0.1)': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    'Logistic Regression\n(C=1)': LogisticRegression(C=1, max_iter=1000, random_state=42),
    'Logistic Regression\n(C=100)': LogisticRegression(C=100, max_iter=1000, random_state=42),
    'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

# 用于展示的主要分类器
MAIN_CLASSIFIERS = {
    'Logistic Regression': LogisticRegression(C=1, max_iter=1000, random_state=42),
    'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

# ============================================================
# 图片输出设置
# ============================================================
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
FIGURE_FACECOLOR = COLORS['background']

# ============================================================
# 样式设置函数
# ============================================================
def set_dark_style():
    """设置深色科研风格"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['background'],
        'axes.edgecolor': COLORS['border'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'legend.facecolor': COLORS['background_alt'],
        'legend.edgecolor': COLORS['accent'],
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 100,
        'savefig.dpi': FIGURE_DPI,
        'savefig.facecolor': COLORS['background'],
        'savefig.edgecolor': 'none',
    })

def reset_style():
    """重置为默认样式"""
    plt.style.use('default')
    plt.rcParams.update(plt.rcParamsDefault)

def save_figure(fig, filename, tight=True):
    """保存图片到输出目录"""
    filepath = os.path.join(FIGURE_DIR, filename)
    if tight:
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor=COLORS['background'], edgecolor='none')
    else:
        fig.savefig(filepath, dpi=FIGURE_DPI,
                   facecolor=COLORS['background'], edgecolor='none')
    print(f"Saved: {filepath}")
    return filepath

def style_3d_axes(ax):
    """设置3D坐标轴样式"""
    ax.set_facecolor(COLORS['background'])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(COLORS['grid'])
    ax.yaxis.pane.set_edgecolor(COLORS['grid'])
    ax.zaxis.pane.set_edgecolor(COLORS['grid'])
    ax.xaxis.line.set_color(COLORS['border'])
    ax.yaxis.line.set_color(COLORS['border'])
    ax.zaxis.line.set_color(COLORS['border'])
    ax.tick_params(colors=COLORS['text'])
    ax.xaxis.label.set_color(COLORS['text'])
    ax.yaxis.label.set_color(COLORS['text'])
    ax.zaxis.label.set_color(COLORS['text'])
    ax.grid(True, alpha=0.2, color=COLORS['grid'])

# ============================================================
# 辅助函数
# ============================================================
def add_title_with_subtitle(fig, title, subtitle, y_title=0.98, y_subtitle=0.94):
    """添加主标题和副标题"""
    fig.suptitle(title, fontsize=20, fontweight='bold', color=COLORS['text'], y=y_title)
    fig.text(0.5, y_subtitle, subtitle, fontsize=11, color=COLORS['text_secondary'], ha='center')

def create_legend_elements(include_stats=False):
    """创建标准图例元素"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SPECIES_COLORS[i],
               markersize=10, label=SPECIES_NAMES[i], linestyle='None', markeredgecolor='white')
        for i in range(3)
    ]

    if include_stats:
        elements.extend([
            Line2D([0], [0], marker='X', color='w', markerfacecolor='white',
                   markersize=10, label='Centroid', linestyle='None'),
            Patch(facecolor='none', edgecolor='white', linewidth=2, label='95% Confidence'),
        ])

    return elements

# ============================================================
# 测试配置
# ============================================================
if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print(f"Available classifiers: {list(CLASSIFIERS.keys())}")

    # 测试深色样式
    set_dark_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Test Dark Style")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.plot([1, 2, 3], [1, 4, 9], color=COLORS['accent'], linewidth=2)
    plt.show()
    reset_style()
