# --encoding:utf-8 --
"""下采样操作"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 给定随机种子
np.random.seed(28)


def lower_sample_data(df, sample_number):
    """
      进行下采样操作, 最终返回抽样数据形成的DataFrame以及抽取剩下的数据
      df: DataFrame对象，进行上采样过程中的原始数据集
      sample_number： 需要采样的数量
    """
    # 1. 获取总的数据量
    rows = len(df)

    # 2. 进行判断操作, 当DataFrame中的数据量少于需要的数据量的时候，直接返回
    if rows <= sample_number:
        return df

    # 3. 进行下标索引生成操作
    row_index = set()
    while len(row_index) != sample_number:
        index = np.random.randint(0, rows, 1)[0]
        row_index.add(index)

    # 4. 进行数据获取
    sample_df = df.iloc[list(row_index)]
    other_row_index = [i for i in range(rows) if i not in row_index]
    after_sampled_df = df.iloc[other_row_index]

    # 5. 返回最终结果
    return (sample_df, after_sampled_df)

# EasyEnsemble方式
if __name__ == '__main__':
    # 1. 模拟数据创建
    category1 = np.random.randint(0, 10, [10000, 5]).astype(np.float)
    label1 = np.array([1] * 10000).reshape(-1, 1)
    data1 = np.concatenate((category1, label1), axis=1)
    category2 = np.random.randint(8, 18, [10, 5]).astype(np.float)
    label2 = np.array([0] * 10).reshape(-1, 1)
    data2 = np.concatenate((category2, label2), axis=1)

    name = ['A', 'B', 'C', 'D', 'E', 'Label']
    data = np.concatenate((data1, data2), axis=0)
    df = pd.DataFrame(data, columns=name)
    print(df.head())

    # 2. 查看各个类别的数据
    print("=" * 100)
    print(df.Label.value_counts())

    # 3. 获取大众类别的数据
    small_category = df[df.Label != 1.0]
    big_category = df[df.Label == 1.0]
    sample_number = 200  # 需要下抽样的数量
    (sample_category_data, big_category) = lower_sample_data(big_category, sample_number)
    print("=" * 100)
    print(sample_category_data.head())
    print(sample_category_data.shape)

    # 4. 合并数据得到最终数据
    final_df = pd.concat([small_category, sample_category_data], ignore_index=True)
    print("=" * 100)
    print(final_df.head())
    print("=" * 100)
    print(final_df.Label.value_counts())
    print("=" * 100)
    print(final_df.describe())

    # 5. 删除已经被抽取的数据
    print("=" * 100)
    print(big_category.head())
    print("去掉抽样样本后大众样本数据量:", end="")
    print(big_category.shape)

    # TODO: 大家可以自己讲X的维度设置为2，然后进行画图展示