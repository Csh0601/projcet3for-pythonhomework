# 鸢尾花数据集分类与可视化项目

## 项目简介

本项目是一个完整的鸢尾花（Iris）数据集分类与可视化分析系统，包含数据探索、特征分析、模型训练、决策边界可视化和3D概率分布展示等功能。

## 项目结构

```
python作业3/
├── src/                          # 源代码目录
│   ├── analysis/                 # 数据分析模块
│   │   ├── 1_feature_distribution.py      # 特征分布可视化
│   │   ├── 2_feature_analysis.py          # 特征关系分析
│   │   └── 2exploratory_analysis.py       # 探索性数据分析
│   ├── visualization/            # 数据可视化模块
│   │   ├── 3_2d_decision_boundary.py      # 2D决策边界
│   │   ├── 4_2d_probability_map.py        # 2D概率图
│   │   ├── 5_3d_boundary_binary.py        # 3D二分类边界
│   │   ├── 6_3d_probability_binary.py     # 3D二分类概率
│   │   ├── 7_3d_boundary_multiclass.py    # 3D三分类边界
│   │   ├── 3d_boundary_plot.py            # 3D边界绘图工具
│   │   ├── decision_boundary_plot.py      # 决策边界绘图工具
│   │   └── probability_map_plot.py         # 概率图绘图工具
│   ├── utils/                    # 工具函数模块
│   │   ├── dataloading.py                 # 数据加载
│   │   ├── model_build_train.py            # 模型构建与训练
│   │   ├── evaluation_mertircs.py         # 评估指标
│   │   └── utils.py                       # 通用工具函数
│   ├── 8_main_runner.py          # 主运行脚本（一键生成所有可视化）
│   └── report_generator.py        # 报告生成器
├── config.py                     # 全局配置文件
├── references/                   # 参考资料
│   ├── classifier2d.py           # 老师提供的2D分类器示例
│   └── data_preview.py           # 老师提供的数据预览示例
├── docs/                         # 文档目录
│   └── homeworkrequest.pdf       # 作业要求文档
├── outputs/                      # 输出目录
│   ├── figures/                  # 生成的图片
│   ├── report.md                 # Markdown格式报告
│   ├── report.pdf                # PDF格式报告
│   └── report.tex                # LaTeX格式报告
├── requirements.txt              # 依赖包列表
└── readme.md                     # 本文件
```

## 功能特性

### 1. 数据分析模块 (`src/analysis/`)
- **特征分布可视化**: 小提琴图、箱线图、散点图组合
- **特征关系分析**: 配对图、相关性热力图、高级2D散点图
- **探索性数据分析**: 基础统计信息、数据预览

### 2. 可视化模块 (`src/visualization/`)
- **2D可视化**:
  - 多分类器决策边界对比
  - 概率热图与不确定性分析
  - 详细的决策区域展示
- **3D可视化**:
  - 二分类问题的3D决策边界
  - 3D概率曲面与切片
  - 三分类问题的综合可视化

### 3. 工具模块 (`src/utils/`)
- 数据加载与预处理
- 模型构建与训练
- 评估指标计算
- 通用工具函数

## 安装与使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行主程序

一键生成所有可视化图片：

```bash
python src/8_main_runner.py
```

### 3. 生成报告

生成完整的分析报告：

```bash
python src/report_generator.py
```

### 4. 单独运行模块

也可以单独运行各个模块：

```python
# 特征分布分析
python src/analysis/1_feature_distribution.py

# 2D决策边界可视化
python src/visualization/3_2d_decision_boundary.py

# 3D三分类可视化
python src/visualization/7_3d_boundary_multiclass.py
```

## 输出说明

所有生成的图片保存在 `outputs/figures/` 目录下，包括：

- 特征分布图（小提琴图、箱线图等）
- 相关性热力图
- 2D决策边界对比图
- 2D概率分布图
- 3D决策边界图
- 3D概率曲面图
- 三分类综合可视化图

报告文件保存在 `outputs/` 目录下：
- `report.md`: Markdown格式报告
- `report.pdf`: PDF格式报告（如果配置了LaTeX）

## 配置说明

项目配置集中在 `config.py` 文件中，包括：

- 路径配置（输出目录等）
- 配色方案（科研级深色主题）
- 分类器配置
- 图片输出设置（DPI、格式等）

## 技术栈

- **数据处理**: NumPy, Pandas
- **机器学习**: scikit-learn
- **可视化**: Matplotlib, Seaborn
- **统计分析**: SciPy
- **报告生成**: Markdown, LaTeX (可选)

## 分类器

项目支持多种分类器：

- Logistic Regression（逻辑回归）
- SVM with RBF kernel（支持向量机）
- Random Forest（随机森林）
- Gradient Boosting（梯度提升）

## 注意事项

1. 运行前确保已安装所有依赖包
2. 某些3D可视化功能需要 `scikit-image` 库
3. 生成PDF报告需要安装LaTeX（可选）
4. 所有路径配置在 `config.py` 中，可根据需要修改

## 作者

Python作业3 - 鸢尾花数据集分类与可视化

## 许可证

本项目仅用于学习和研究目的。

