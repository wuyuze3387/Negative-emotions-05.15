# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:19:41 2025

@author: 86185
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
import joblib

# 加载数据集
df = pd.read_excel(r'D:\OneDrive\桌面\ML数据集1部署APP.xlsx')

# 划分特征和目标变量
X = df.drop(['Negative Emotions'], axis=1)
y = df['Negative Emotions']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['Negative Emotions']
)

# 显示数据集的前几行
print(df.head())

# 初始化CatBoost分类模型，使用默认参数
model_cat = CatBoostClassifier(
    random_state=42,  # 设置随机种子以确保结果可重现
)

# 训练模型
model_cat.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = model_cat.predict(X_test)

# 输出模型报告，查看评价指标
print(classification_report(y_test, y_pred))

# 保存模型到文件
joblib.dump(model_cat, r'D:\OneDrive\桌面\CatBoost.pkl')
