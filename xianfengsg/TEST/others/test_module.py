# -*- coding: utf-8 -*-
# @Time    : 2019/11/25 10:35
# @Author  : Ye Jinyu__jimmy
# @File    : test_module
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

# 构造训练样本
n_samples = 200  # 样本总数
outliers_fraction = 0.25  # 异常样本比例
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)


#构建样本
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(n_inliers // 2, 2)
X_train = np.r_[X + 2, X - 2]  # 正常样本
X_train = np.r_[X_train, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]  # 正常样本加上异常样本


