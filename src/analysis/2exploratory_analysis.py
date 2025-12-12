import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from src.utils.dataloading import load_and_process

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
#对中文字体设置支持避免乱码

def load_DataFrame():
    X_train , X_test , Y_train , Y_test  , feature_names , target_names= load_and_process()
     
    df = pd.DataFrame(X_train  ,columns = feature_names)
    df["species"] = Y_train
    return df , feature_names

def draw_basic_info(df):
    print(df.info())
    print(df.head())
    print(df.tail())
    print(df.describe())

def draw_boxplot(df):
    """
    绘制高级特征分布分析图
    结合小提琴图 + 箱线图 + 蜂群图 + 统计标注
    """
    # 设置深色科研风格
    plt.style.use('dark_background')
    
    # 准备数据 - 将宽格式转为长格式
    df_plot = df.drop(columns=['species'])
    df_melted = df_plot.melt(var_name='Features', value_name='Value')
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 定义专业配色方案
    colors = ['#00CED1', '#2E8B57', '#DAA520', '#CD5C5C']  # 青色、绿色、金色、珊瑚红
    palette = dict(zip(df_plot.columns, colors))
    
    # 1. 绘制小提琴图（底层 - 显示密度分布）
    violin = sns.violinplot(
        data=df_melted, 
        x='Features', 
        y='Value',
        hue='Features',
        palette=palette,
        inner=None,  # 不显示内部
        alpha=0.6,
        linewidth=0,
        legend=False,
        ax=ax
    )
    
    # 2. 绘制箱线图（中层 - 显示四分位数）
    box = sns.boxplot(
        data=df_melted,
        x='Features',
        y='Value',
        hue='Features',
        palette=palette,
        width=0.15,
        boxprops={'alpha': 0.7, 'edgecolor': 'white', 'linewidth': 1.5},
        whiskerprops={'color': 'white', 'linewidth': 1.5},
        capprops={'color': 'white', 'linewidth': 1.5},
        medianprops={'color': 'white', 'linewidth': 2},
        fliersize=0,  # 隐藏离群点（蜂群图会显示）
        legend=False,
        ax=ax
    )
    
    # 3. 绘制散点图（顶层 - 显示每个数据点，带自然抖动）
    import numpy as np
    np.random.seed(42)  # 保证可重复性
    
    # 使用stripplot替代swarmplot，添加更自然的抖动
    strip = sns.stripplot(
        data=df_melted,
        x='Features',
        y='Value',
        hue='Features',
        palette=palette,
        size=5,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.85,
        jitter=0.25,  # 添加水平抖动，让点分布更自然
        legend=False,
        ax=ax
    )
    
    # 4. 添加均值点（菱形标记）
    means = df_plot.mean()
    for i, (col, mean_val) in enumerate(means.items()):
        ax.scatter(i, mean_val, marker='D', s=80, c='white', 
                   edgecolors='white', linewidths=1, zorder=10)
    
    # 5. 添加统计标注（均值μ和标准差σ）
    stats = df_plot.agg(['mean', 'std'])
    for i, col in enumerate(df_plot.columns):
        mu = stats.loc['mean', col]
        sigma = stats.loc['std', col]
        
        # 计算标注位置（在数据最大值上方）
        y_max = df_plot[col].max()
        
        # 创建标注框
        bbox_props = dict(
            boxstyle='round,pad=0.3',
            facecolor=colors[i],
            edgecolor='none',
            alpha=0.9
        )
        ax.annotate(
            f'μ={mu:.2f}\nσ={sigma:.2f}',
            xy=(i, y_max + 0.3),
            fontsize=10,
            fontweight='bold',
            color='white',
            ha='center',
            va='bottom',
            bbox=bbox_props
        )
    
    # 6. 设置标题和标签
    ax.set_title(
        'FEATURE DISTRIBUTION ANALYSIS',
        fontsize=20,
        fontweight='bold',
        color='white',
        pad=20
    )
    ax.text(
        0.5, 1.02,
        'Violin Plot + Box Plot + Swarm Points | Statistical Overview',
        transform=ax.transAxes,
        fontsize=11,
        color='#AAAAAA',
        ha='center',
        va='bottom'
    )
    
    ax.set_xlabel('Features', fontsize=14, color='white', labelpad=10)
    ax.set_ylabel('Value (cm)', fontsize=14, color='white', labelpad=10)
    
    # 7. 设置刻度标签
    ax.set_xticks(range(len(df_plot.columns)))
    ax.set_xticklabels(
        [f'{col}' for col in df_plot.columns],
        fontsize=11,
        color='white'
    )
    ax.tick_params(axis='y', colors='white', labelsize=10)
    
    # 8. 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00CED1', 
               markersize=8, label='Data Points', linestyle='None'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markersize=8, label='Mean (μ)', linestyle='None'),
        Line2D([0], [0], color='gray', linewidth=10, alpha=0.5, label='Density (Violin)'),
        Line2D([0], [0], color='white', linewidth=2, label='IQR (Box)')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.8,
        facecolor='#1a1a2e',
        edgecolor='#00CED1'
    )
    
    # 9. 设置背景和网格
    ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')
    ax.grid(axis='y', alpha=0.2, linestyle='--', color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    plt.tight_layout()
    plt.show()
    
    # 恢复默认样式
    plt.style.use('default')

def draw_pairplot(df):
    """
    绘制高级特征配对分析图
    展示特征间的相关性和类别可分性
    """
    import numpy as np
    from scipy import stats
    
    # 设置深色科研风格
    plt.style.use('dark_background')
    
    # 定义专业配色方案（三个物种）
    species_colors = ['#00CED1', '#FFD700', '#FF6B6B']  # 青色、金色、珊瑚红
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    palette = dict(zip(range(3), species_colors))
    
    # 创建自定义 PairGrid
    g = sns.PairGrid(
        df, 
        hue='species',
        palette=palette,
        diag_sharey=False,
        corner=False,  # 显示完整矩阵
        height=2.5,
        aspect=1
    )
    
    # 设置背景颜色
    g.figure.patch.set_facecolor('#0d0d1a')
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_facecolor('#0d0d1a')
    
    # 对角线：绘制 KDE 密度曲线 + 直方图
    def diag_kde(x, **kwargs):
        color = kwargs.get('color', '#00CED1')
        ax = plt.gca()
        # 绘制填充的 KDE
        sns.kdeplot(x, color=color, fill=True, alpha=0.3, linewidth=2, **{k:v for k,v in kwargs.items() if k not in ['color']})
        # 绘制 KDE 线条
        sns.kdeplot(x, color=color, fill=False, linewidth=2.5, **{k:v for k,v in kwargs.items() if k not in ['color']})
    
    g.map_diag(diag_kde)
    
    # 下三角：绘制散点图 + 置信椭圆
    def lower_scatter(x, y, **kwargs):
        color = kwargs.get('color', '#00CED1')
        ax = plt.gca()
        ax.scatter(x, y, color=color, s=35, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    g.map_lower(lower_scatter)
    
    # 上三角：绘制 KDE 等高线图
    def upper_kde(x, y, **kwargs):
        color = kwargs.get('color', '#00CED1')
        ax = plt.gca()
        try:
            sns.kdeplot(x=x, y=y, color=color, levels=5, linewidths=1.5, alpha=0.8)
        except:
            ax.scatter(x, y, color=color, s=20, alpha=0.5)
    
    g.map_upper(upper_kde)
    
    # 添加相关系数到上三角
    feature_cols = [col for col in df.columns if col != 'species']
    for i, row_var in enumerate(feature_cols):
        for j, col_var in enumerate(feature_cols):
            if i < j:  # 上三角
                ax = g.axes[i, j]
                # 计算皮尔逊相关系数
                corr, p_value = stats.pearsonr(df[row_var], df[col_var])
                # 根据相关性强度选择颜色
                if abs(corr) > 0.7:
                    corr_color = '#00FF88'  # 强相关 - 绿色
                elif abs(corr) > 0.4:
                    corr_color = '#FFD700'  # 中等相关 - 金色
                else:
                    corr_color = '#888888'  # 弱相关 - 灰色
                
                # 添加相关系数标注
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
                        facecolor='#1a1a2e',
                        edgecolor=corr_color,
                        alpha=0.9
                    )
                )
    
    # 设置轴标签样式
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#444444')
            ax.spines['bottom'].set_color('#444444')
            # 设置轴标签颜色
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
    
    # 添加主标题
    g.figure.suptitle(
        'PAIRWISE FEATURE ANALYSIS',
        fontsize=22,
        fontweight='bold',
        color='white',
        y=1.02
    )
    
    # 添加副标题
    g.figure.text(
        0.5, 0.99,
        'Scatter Matrix | KDE Contours | Correlation Coefficients',
        fontsize=12,
        color='#AAAAAA',
        ha='center',
        va='top'
    )
    
    # 创建自定义图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=species_colors[0],
               markersize=10, label=species_names[0], linestyle='None', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=species_colors[1],
               markersize=10, label=species_names[1], linestyle='None', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=species_colors[2],
               markersize=10, label=species_names[2], linestyle='None', markeredgecolor='white'),
    ]
    
    # 添加相关性图例
    legend_elements.extend([
        Line2D([0], [0], color='#00FF88', linewidth=3, label='Strong (|r|>0.7)'),
        Line2D([0], [0], color='#FFD700', linewidth=3, label='Moderate (|r|>0.4)'),
        Line2D([0], [0], color='#888888', linewidth=3, label='Weak (|r|≤0.4)'),
    ])
    
    g.figure.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        fontsize=10,
        framealpha=0.9,
        facecolor='#1a1a2e',
        edgecolor='#00CED1',
        title='Species & Correlation',
        title_fontsize=11
    )
    # 设置图例标题颜色
    g.figure.legends[0].get_title().set_color('white')
    
    plt.tight_layout()
    plt.savefig("plot_pairplot002.png", dpi=300, bbox_inches='tight', 
                facecolor='#0d0d1a', edgecolor='none')
    plt.show()
    
    # 恢复默认样式
    plt.style.use('default')

def draw_correlation_heatmap(df):
    """
    绘制顶刊级别的相关性热力图
    包含相关系数矩阵、显著性标注、层次聚类
    """
    import numpy as np
    from scipy import stats
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    
    # 设置深色科研风格
    plt.style.use('dark_background')
    
    # 计算相关系数矩阵
    df_features = df.drop(columns=["species"])
    corr = df_features.corr()
    
    # 计算 p 值矩阵（显著性检验）
    n = len(df_features)
    p_matrix = np.zeros_like(corr)
    for i, col1 in enumerate(df_features.columns):
        for j, col2 in enumerate(df_features.columns):
            if i != j:
                _, p_val = stats.pearsonr(df_features[col1], df_features[col2])
                p_matrix[i, j] = p_val
            else:
                p_matrix[i, j] = 0
    
    # 创建画布 - 使用 GridSpec 实现复杂布局
    fig = plt.figure(figsize=(12, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(10, 12, figure=fig, hspace=0.3, wspace=0.3)
    
    # 主热力图区域
    ax_main = fig.add_subplot(gs[1:9, 2:10])
    # 顶部树状图
    ax_dendro_top = fig.add_subplot(gs[0:1, 2:10])
    # 左侧树状图
    ax_dendro_left = fig.add_subplot(gs[1:9, 0:1])
    # 颜色条
    ax_cbar = fig.add_subplot(gs[1:9, 11:12])
    
    # 计算层次聚类
    # 将相关系数转换为距离矩阵
    dissimilarity = 1 - np.abs(corr.values)
    np.fill_diagonal(dissimilarity, 0)
    
    # 确保对称性
    dissimilarity = (dissimilarity + dissimilarity.T) / 2
    
    # 转换为压缩形式并进行层次聚类
    condensed_dist = squareform(dissimilarity)
    linkage = hierarchy.linkage(condensed_dist, method='average')
    
    # 获取聚类顺序
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    order = dendro['leaves']
    
    # 重新排序相关矩阵
    corr_ordered = corr.iloc[order, order]
    p_matrix_ordered = p_matrix[np.ix_(order, order)]
    
    # 绘制顶部树状图
    ax_dendro_top.set_facecolor('#0d0d1a')
    hierarchy.dendrogram(
        linkage,
        ax=ax_dendro_top,
        orientation='top',
        color_threshold=0,
        above_threshold_color='#00CED1',
        leaf_rotation=0,
        leaf_font_size=0
    )
    ax_dendro_top.axis('off')
    
    # 绘制左侧树状图
    ax_dendro_left.set_facecolor('#0d0d1a')
    hierarchy.dendrogram(
        linkage,
        ax=ax_dendro_left,
        orientation='left',
        color_threshold=0,
        above_threshold_color='#00CED1',
        leaf_rotation=0,
        leaf_font_size=0
    )
    ax_dendro_left.axis('off')
    
    # 创建自定义颜色映射（深色主题优化）
    from matplotlib.colors import LinearSegmentedColormap
    colors_neg = ['#1a0a2e', '#2d1b4e', '#4a2c7a', '#6b3fa0']  # 深紫到紫
    colors_zero = ['#1a1a2e']  # 深蓝灰
    colors_pos = ['#2e4a1a', '#4a7a2c', '#6ba03f', '#8fce52']  # 深绿到亮绿
    
    # 更专业的配色：蓝-白-红
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8',  # 蓝色系
         '#FFFFFF',  # 白色中心
         '#FFCCD5', '#FF8FA3', '#FF4D6D', '#C9184A'],  # 红色系
        N=256
    )
    
    # 绘制主热力图
    ax_main.set_facecolor('#0d0d1a')
    
    # 创建掩码（只显示下三角，避免重复）
    mask = np.triu(np.ones_like(corr_ordered, dtype=bool), k=0)
    mask = np.zeros_like(corr_ordered, dtype=bool)  # 显示完整矩阵
    
    # 绘制热力图
    im = ax_main.imshow(
        corr_ordered.values,
        cmap=custom_cmap,
        vmin=-1, vmax=1,
        aspect='auto'
    )
    
    # 添加网格线
    for i in range(len(corr_ordered) + 1):
        ax_main.axhline(i - 0.5, color='#2a2a4a', linewidth=0.5)
        ax_main.axvline(i - 0.5, color='#2a2a4a', linewidth=0.5)
    
    # 添加相关系数和显著性标注
    for i in range(len(corr_ordered)):
        for j in range(len(corr_ordered)):
            value = corr_ordered.iloc[i, j]
            p_val = p_matrix_ordered[i, j]
            
            # 根据背景色选择文字颜色
            text_color = 'white' if abs(value) > 0.5 else '#CCCCCC'
            
            # 添加显著性星号
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
            
            # 绘制相关系数
            ax_main.text(
                j, i, f'{value:.2f}',
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color=text_color
            )
            # 绘制显著性标注
            if sig:
                ax_main.text(
                    j, i + 0.25, sig,
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='#FFD700'
                )
    
    # 设置刻度标签
    ordered_labels = [corr_ordered.columns[i].replace(' (cm)', '') for i in range(len(corr_ordered))]
    ax_main.set_xticks(range(len(corr_ordered)))
    ax_main.set_yticks(range(len(corr_ordered)))
    ax_main.set_xticklabels(ordered_labels, fontsize=11, color='white', rotation=45, ha='right')
    ax_main.set_yticklabels(ordered_labels, fontsize=11, color='white')
    
    # 添加颜色条
    ax_cbar.set_facecolor('#0d0d1a')
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label('Pearson Correlation (r)', fontsize=12, color='white', labelpad=15)
    cbar.ax.tick_params(colors='white', labelsize=10)
    cbar.outline.set_edgecolor('#444444')
    
    # 在颜色条上添加关键标记
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(['-1.0\n(Perfect\nNegative)', '-0.5', '0\n(None)', '0.5', '1.0\n(Perfect\nPositive)'])
    
    # 添加主标题
    fig.suptitle(
        'CORRELATION MATRIX WITH HIERARCHICAL CLUSTERING',
        fontsize=20,
        fontweight='bold',
        color='white',
        y=0.98
    )
    
    # 添加副标题
    fig.text(
        0.5, 0.94,
        'Pearson Coefficients | Significance Levels: * p<0.05, ** p<0.01, *** p<0.001',
        fontsize=11,
        color='#AAAAAA',
        ha='center'
    )
    
    # 添加解释性图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C9184A', edgecolor='white', label='Strong Positive (r > 0.7)'),
        Patch(facecolor='#FF8FA3', edgecolor='white', label='Moderate Positive'),
        Patch(facecolor='#FFFFFF', edgecolor='#444444', label='No Correlation'),
        Patch(facecolor='#90E0EF', edgecolor='white', label='Moderate Negative'),
        Patch(facecolor='#0077B6', edgecolor='white', label='Strong Negative (r < -0.7)'),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=9,
        framealpha=0.9,
        facecolor='#1a1a2e',
        edgecolor='#00CED1'
    )
    
    # 设置背景
    fig.patch.set_facecolor('#0d0d1a')
    
    plt.savefig("plot_correlation_heatmap003.png", dpi=300, bbox_inches='tight',
                facecolor='#0d0d1a', edgecolor='none')
    plt.show()
    
    # 恢复默认样式
    plt.style.use('default')

def draw_scatter_2d(df):
    """
    绘制顶刊级别的2D散点图
    包含：置信椭圆、边缘分布、类别中心、决策区域暗示、统计标注
    """
    import numpy as np
    from scipy import stats
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    from scipy.spatial import ConvexHull
    
    # 设置深色科研风格
    plt.style.use('dark_background')
    
    # 特征选择
    x_col = "petal length (cm)"
    y_col = "petal width (cm)"
    
    # 定义专业配色方案
    species_colors = {0: '#00CED1', 1: '#FFD700', 2: '#FF6B6B'}
    species_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    
    # 创建复杂布局：主图 + 边缘分布
    fig = plt.figure(figsize=(12, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    
    # 主散点图
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    # 顶部边缘分布
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    # 右侧边缘分布
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # 设置背景
    for ax in [ax_main, ax_top, ax_right]:
        ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')
    
    # ========== 绘制主散点图 ==========
    
    # 1. 先绘制 KDE 等高线作为背景（决策区域暗示）
    for species_id in df['species'].unique():
        subset = df[df['species'] == species_id]
        x_data = subset[x_col].values
        y_data = subset[y_col].values
        
        try:
            # 绘制 KDE 等高线
            sns.kdeplot(
                x=x_data, y=y_data,
                ax=ax_main,
                levels=3,
                color=species_colors[species_id],
                alpha=0.15,
                fill=True,
                linewidths=0
            )
        except:
            pass
    
    # 2. 绘制凸包（Convex Hull）- 类别边界
    for species_id in df['species'].unique():
        subset = df[df['species'] == species_id]
        points = subset[[x_col, y_col]].values
        
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_points = np.append(hull.vertices, hull.vertices[0])
                ax_main.plot(
                    points[hull_points, 0], points[hull_points, 1],
                    color=species_colors[species_id],
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.6
                )
                ax_main.fill(
                    points[hull_points, 0], points[hull_points, 1],
                    color=species_colors[species_id],
                    alpha=0.05
                )
            except:
                pass
    
    # 3. 绘制置信椭圆（95% 置信区间）
    def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
        """绘制协方差置信椭圆"""
        if len(x) < 2:
            return None
        
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor,
            **kwargs
        )
        
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_x, mean_y = np.mean(x), np.mean(y)
        
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    
    for species_id in df['species'].unique():
        subset = df[df['species'] == species_id]
        confidence_ellipse(
            subset[x_col], subset[y_col], ax_main,
            n_std=2.0,
            edgecolor=species_colors[species_id],
            linewidth=2.5,
            linestyle='-',
            alpha=0.8
        )
    
    # 4. 绘制散点（带渐变效果）
    for species_id in df['species'].unique():
        subset = df[df['species'] == species_id]
        
        # 主散点
        ax_main.scatter(
            subset[x_col], subset[y_col],
            c=species_colors[species_id],
            s=80,
            alpha=0.85,
            edgecolors='white',
            linewidths=0.8,
            label=species_names[species_id],
            zorder=5
        )
        
        # 添加发光效果（大的半透明点）
        ax_main.scatter(
            subset[x_col], subset[y_col],
            c=species_colors[species_id],
            s=200,
            alpha=0.15,
            edgecolors='none',
            zorder=4
        )
    
    # 5. 绘制类别中心点（质心）
    # 为每个类别设置不同的标签偏移方向，避免遮挡
    label_offsets = {
        0: (-60, -40),   # Setosa: 左下方
        1: (50, -50),    # Versicolor: 右下方  
        2: (50, 40)      # Virginica: 右上方
    }
    
    for species_id in df['species'].unique():
        subset = df[df['species'] == species_id]
        centroid_x = subset[x_col].mean()
        centroid_y = subset[y_col].mean()
        
        # 中心点标记
        ax_main.scatter(
            centroid_x, centroid_y,
            marker='X',
            s=250,
            c=species_colors[species_id],
            edgecolors='white',
            linewidths=2,
            zorder=10
        )
        
        # 获取该类别的标签偏移
        offset = label_offsets.get(species_id, (40, 40))
        
        # 中心点标签 - 提高 zorder 确保在最上层
        ax_main.annotate(
            f'{species_names[species_id]}\nCentroid',
            xy=(centroid_x, centroid_y),
            xytext=offset,
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='white',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor=species_colors[species_id],
                edgecolor='white',
                linewidth=1.5,
                alpha=0.95
            ),
            arrowprops=dict(
                arrowstyle='->',
                color='white',
                linewidth=1.5,
                connectionstyle='arc3,rad=0.2'
            ),
            zorder=15  # 确保标签在所有元素之上
        )
    
    # 6. 添加统计信息面板
    stats_text = []
    for species_id in df['species'].unique():
        subset = df[df['species'] == species_id]
        n = len(subset)
        corr, p_val = stats.pearsonr(subset[x_col], subset[y_col])
        stats_text.append(f"{species_names[species_id]}: n={n}, r={corr:.3f}")
    
    # 计算总体相关性
    total_corr, total_p = stats.pearsonr(df[x_col], df[y_col])
    stats_text.append(f"Overall: r={total_corr:.3f}, p<0.001")
    
    ax_main.text(
        0.02, 0.98,
        '\n'.join(stats_text),
        transform=ax_main.transAxes,
        fontsize=10,
        color='white',
        va='top',
        ha='left',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='#1a1a2e',
            edgecolor='#00CED1',
            alpha=0.9
        ),
        family='monospace'
    )
    
    # ========== 绘制边缘分布 ==========
    
    # 顶部：X轴边缘分布
    for species_id in df['species'].unique():
        subset = df[df['species'] == species_id]
        sns.kdeplot(
            subset[x_col],
            ax=ax_top,
            color=species_colors[species_id],
            fill=True,
            alpha=0.4,
            linewidth=2
        )
    ax_top.set_ylabel('')
    ax_top.set_xlabel('')
    ax_top.tick_params(labelbottom=False, labelleft=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.spines['bottom'].set_color('#444444')
    
    # 右侧：Y轴边缘分布
    for species_id in df['species'].unique():
        subset = df[df['species'] == species_id]
        sns.kdeplot(
            y=subset[y_col],
            ax=ax_right,
            color=species_colors[species_id],
            fill=True,
            alpha=0.4,
            linewidth=2
        )
    ax_right.set_ylabel('')
    ax_right.set_xlabel('')
    ax_right.tick_params(labelbottom=False, labelleft=False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.spines['left'].set_color('#444444')
    
    # ========== 美化主图 ==========
    
    ax_main.set_xlabel('Petal Length (cm)', fontsize=14, color='white', labelpad=10)
    ax_main.set_ylabel('Petal Width (cm)', fontsize=14, color='white', labelpad=10)
    ax_main.tick_params(colors='white', labelsize=11)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['left'].set_color('#444444')
    ax_main.spines['bottom'].set_color('#444444')
    ax_main.grid(True, alpha=0.15, linestyle='--', color='white')
    
    # 添加图例
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=species_colors[0],
               markersize=12, label=species_names[0], linestyle='None', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=species_colors[1],
               markersize=12, label=species_names[1], linestyle='None', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=species_colors[2],
               markersize=12, label=species_names[2], linestyle='None', markeredgecolor='white'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='white',
               markersize=12, label='Centroid', linestyle='None', markeredgecolor='gray'),
        Patch(facecolor='none', edgecolor='white', linewidth=2, label='95% Confidence'),
        Line2D([0], [0], color='white', linestyle='--', linewidth=1.5, label='Convex Hull'),
    ]
    
    ax_main.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=10,
        framealpha=0.9,
        facecolor='#1a1a2e',
        edgecolor='#00CED1'
    )
    
    # 添加主标题
    fig.suptitle(
        'FEATURE SPACE ANALYSIS: PETAL DIMENSIONS',
        fontsize=20,
        fontweight='bold',
        color='white',
        y=0.98
    )
    
    # 添加副标题
    fig.text(
        0.45, 0.94,
        'Scatter Plot | 95% Confidence Ellipse | Marginal Distributions | Class Centroids',
        fontsize=11,
        color='#AAAAAA',
        ha='center'
    )
    
    plt.savefig("plot_scatter_advanced.png", dpi=300, bbox_inches='tight',
                facecolor='#0d0d1a', edgecolor='none')
    plt.show()
    
    # 恢复默认样式
    plt.style.use('default')

if __name__ == "__main__":
    df , features = load_DataFrame()
    draw_basic_info(df)
    draw_boxplot(df)
    draw_pairplot(df)
    draw_correlation_heatmap(df)
    draw_scatter_2d(df)