import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_process(test_size = 0.2 , random_state = 42):
    iris = load_iris()
    X = iris.data  #特征矩阵
    y = iris.target #向量矩阵
    feature_names = iris.feature_names
    target_names = iris.target_names
    print(feature_names)

    df = pd.DataFrame(X , columns = feature_names)
    #将numpy数组转为DataFrame，每个栏目的名转为特征名字
    df["species"] = y
    #添加新的一列并且命名为种类，种类共有0 1 2 三个类别
    
    #特征标准化，将特征转为符合0为均值，1为标准差的正态分布，方便MLP进行训练
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #划分test_iter 和train_iter
    X_train , X_test , Y_train , Y_test = train_test_split(X_scaled , y , test_size = test_size , random_state = random_state , stratify = y)
    #stratify = y ,这个是很重要的参数，保证在类别不平衡的情况下，test_iter和train_iter的类别比例与原始数据集相同
    return X_train ,X_test ,  Y_train , Y_test , feature_names , target_names

# 调用函数查看
if __name__ == "__main__":
    load_and_process()