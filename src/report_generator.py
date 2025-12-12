# -*- coding: utf-8 -*-
"""
数据分析报告生成器 - 鸢尾花数据集分类与可视化
生成详细的Markdown格式报告
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)
from scipy import stats

from config import OUTPUT_DIR, FIGURE_DIR


def generate_report():
    """生成完整的数据分析报告"""

    # 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y
    df['species_name'] = [target_names[i] for i in y]

    # 开始生成报告
    report = []

    # 标题
    report.append("# 鸢尾花数据集分类与可视化分析报告")
    report.append(f"\n**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report.append("\n---\n")

    # 1. 数据集概述
    report.append("## 1. 数据集概述\n")
    report.append("### 1.1 基本信息\n")
    report.append(f"- **数据集名称**: Iris (鸢尾花数据集)")
    report.append(f"- **样本数量**: {len(df)}")
    report.append(f"- **特征数量**: {len(feature_names)}")
    report.append(f"- **类别数量**: {len(target_names)}")
    report.append(f"- **类别名称**: {', '.join(target_names)}")
    report.append("")

    report.append("### 1.2 特征说明\n")
    report.append("| 特征编号 | 特征名称 | 描述 |")
    report.append("|:--------:|:---------|:-----|")
    report.append("| x₀ | Sepal Length | 花萼长度 (cm) |")
    report.append("| x₁ | Sepal Width | 花萼宽度 (cm) |")
    report.append("| x₂ | Petal Length | 花瓣长度 (cm) |")
    report.append("| x₃ | Petal Width | 花瓣宽度 (cm) |")
    report.append("")

    report.append("### 1.3 类别分布\n")
    report.append("| 类别 | 名称 | 样本数 | 比例 |")
    report.append("|:----:|:-----|:------:|:----:|")
    for i, name in enumerate(target_names):
        count = (y == i).sum()
        ratio = count / len(y) * 100
        report.append(f"| {i} | {name} | {count} | {ratio:.1f}% |")
    report.append("")

    # 2. 描述性统计
    report.append("## 2. 描述性统计分析\n")
    report.append("### 2.1 总体统计量\n")
    stats_df = df[feature_names].describe()
    report.append("| 统计量 | " + " | ".join([f.replace(' (cm)', '') for f in feature_names]) + " |")
    report.append("|:------:|" + ":------:|" * len(feature_names))
    for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        values = [f"{stats_df.loc[stat, f]:.2f}" for f in feature_names]
        report.append(f"| {stat} | " + " | ".join(values) + " |")
    report.append("")

    report.append("### 2.2 按类别统计\n")
    for i, name in enumerate(target_names):
        subset = df[df['species'] == i][feature_names]
        report.append(f"\n**{name}**:\n")
        report.append("| 特征 | 均值 | 标准差 | 最小值 | 最大值 |")
        report.append("|:-----|:----:|:------:|:------:|:------:|")
        for f in feature_names:
            fname = f.replace(' (cm)', '')
            report.append(f"| {fname} | {subset[f].mean():.2f} | {subset[f].std():.2f} | {subset[f].min():.2f} | {subset[f].max():.2f} |")
    report.append("")

    # 3. 相关性分析
    report.append("## 3. 相关性分析\n")
    report.append("### 3.1 Pearson相关系数矩阵\n")
    corr = df[feature_names].corr()
    report.append("| | " + " | ".join([f.replace(' (cm)', '').split()[0][:5] for f in feature_names]) + " |")
    report.append("|:--|" + ":--:|" * len(feature_names))
    for i, f1 in enumerate(feature_names):
        f1_short = f1.replace(' (cm)', '').split()[0][:5]
        values = [f"{corr.iloc[i, j]:.3f}" for j in range(len(feature_names))]
        report.append(f"| {f1_short} | " + " | ".join(values) + " |")
    report.append("")

    report.append("### 3.2 显著相关特征对\n")
    report.append("| 特征对 | 相关系数 | p值 | 显著性 |")
    report.append("|:-------|:--------:|:---:|:------:|")
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            r, p = stats.pearsonr(df[feature_names[i]], df[feature_names[j]])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            f1 = feature_names[i].replace(' (cm)', '').split()[0]
            f2 = feature_names[j].replace(' (cm)', '').split()[0]
            report.append(f"| {f1} - {f2} | {r:.3f} | {p:.4f} | {sig} |")
    report.append("\n> 显著性标记: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著\n")

    # 4. 分类器性能
    report.append("## 4. 分类器性能评估\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        'Logistic Regression': LogisticRegression(C=1, max_iter=1000, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    }

    report.append("### 4.1 分类器对比 (测试集)\n")
    report.append("| 分类器 | 准确率 | 精确率 | 召回率 | F1分数 |")
    report.append("|:-------|:------:|:------:|:------:|:------:|")

    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report.append(f"| {name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |")
    report.append("")

    report.append("### 4.2 交叉验证结果 (5-fold)\n")
    report.append("| 分类器 | 平均准确率 | 标准差 |")
    report.append("|:-------|:----------:|:------:|")
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=5)
        report.append(f"| {name} | {scores.mean():.4f} | {scores.std():.4f} |")
    report.append("")

    # 5. 可视化结果
    report.append("## 5. 可视化结果\n")
    report.append("### 5.1 生成的图片列表\n")

    figures = [
        ("feature_distribution_combined.png", "特征分布组合图", "展示四个特征的小提琴图+箱线图+散点图"),
        ("feature_distribution_by_species.png", "按物种特征分布", "展示不同物种在各特征上的分布差异"),
        ("feature_boxplot_grid.png", "箱线图网格", "传统箱线图展示"),
        ("pairplot_analysis.png", "特征配对分析", "散点矩阵+KDE+相关系数"),
        ("correlation_heatmap.png", "相关性热力图", "带层次聚类的相关系数矩阵"),
        ("scatter_2d_advanced.png", "高级2D散点图", "置信椭圆+凸包+边缘分布"),
        ("decision_boundary_2d_comparison.png", "2D决策边界对比", "多分类器决策边界对比"),
        ("probability_maps_2d_detailed.png", "2D概率图", "各类别概率分布"),
        ("uncertainty_analysis_2d.png", "不确定性分析", "分类不确定性区域可视化"),
        ("3d_boundary_setosa_vs_others.png", "3D边界(Setosa vs Others)", "两分类3D决策边界"),
        ("3d_boundary_versicolor_vs_virginica.png", "3D边界(Versi. vs Virgi.)", "两分类3D决策边界"),
        ("3d_probability_surface_setosa_vs_others.png", "3D概率曲面", "两分类概率分布"),
        ("3d_multiclass_scatter.png", "3D三分类散点图", "三类数据的3D分布"),
        ("3d_multiclass_boundaries.png", "3D三分类边界", "三分类决策边界"),
        ("3d_multiclass_combined.png", "3D综合视图", "3D散点+2D切片"),
    ]

    report.append("| 序号 | 文件名 | 描述 |")
    report.append("|:----:|:-------|:-----|")
    for i, (fname, title, desc) in enumerate(figures, 1):
        report.append(f"| {i} | {fname} | {title}: {desc} |")
    report.append("")

    # 6. 主要发现
    report.append("## 6. 主要发现与结论\n")

    report.append("### 6.1 数据特征分析\n")
    report.append("1. **类别可分性**: Setosa类与其他两类在花瓣特征(Petal Length/Width)上具有明显差异，容易区分")
    report.append("2. **特征相关性**: Petal Length与Petal Width高度正相关(r>0.9)，说明花瓣的长宽比较一致")
    report.append("3. **特征重要性**: 花瓣特征比花萼特征更能区分不同物种")
    report.append("")

    report.append("### 6.2 分类器性能\n")
    report.append("1. 所有测试的分类器在该数据集上都能达到较高准确率(>95%)")
    report.append("2. SVM和随机森林在处理Versicolor与Virginica的边界时表现略好")
    report.append("3. Logistic Regression提供了良好的可解释性和概率输出")
    report.append("")

    report.append("### 6.3 可视化洞察\n")
    report.append("1. **2D可视化**: 使用Petal Length和Petal Width两个特征即可实现较好的三分类")
    report.append("2. **3D可视化**: 增加第三个特征可以更清晰地展示决策边界")
    report.append("3. **概率图**: 揭示了分类器的不确定性区域主要集中在Versicolor和Virginica交界处")
    report.append("")

    report.append("---\n")
    report.append("*本报告由自动化脚本生成*")

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, 'report.md')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_report()
