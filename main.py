from random import randint

import numpy as np
from tqdm import tqdm


class AllData:
    def __init__(self, data_list: list, label_list: list, toler, c, k=390):  # 乳腺癌测试集线性可分，k应设置较大
        self.data_mat = np.mat(data_list)
        self.label_mat = np.mat(label_list).T  # 行向量
        self.toler = toler
        self.c = c
        self.nums = np.shape(self.data_mat)[0]  # 样本总数
        self.attribute = np.shape(self.data_mat)[1]  # 属性总数
        self.k = k  # 径向基核函数中的超参数
        self.inner_product = self.findinner_p()  # 用核函数求所有样本之间的内积
        self.alphas = np.mat(np.zeros(shape=(self.nums, 1), dtype=np.float64))  # alpha乘子
        self.b = 0  # 标量参数b

    def findinner_p(self) -> np.matrix:  # 径向基核函数 封装在类里
        inner_product = np.mat(np.zeros(shape=(self.nums, self.nums), dtype=np.float64))  # 初始化核函数求出的内积
        for i in range(self.nums):  # i j都是行索引，一行一个样本
            for j in range(self.nums):
                inner_product[i, j] = np.exp(-((np.linalg.norm(self.data_mat[i, :] - self.data_mat[j, :])) ** 2)
                                             / (self.k ** 2))
        return inner_product


def load_data():
    data_list = []
    label_list = []
    with open('wdbc.data', 'r') as f:
        for line in f.readlines():
            data_list.append([float(linedata) for linedata in line.split(',')[2:]])  # 去除第一个编号和第二个标签
            label_list.append(-1 if line.split(',')[1] == 'M' else 1)  # M-malignant恶性_-1   B-benign 良性_1
    print(f'良性样本有{label_list.count(1)}个，恶性样本有{label_list.count(-1)}个')
    return data_list, label_list


def select_j(i):
    # 需优化
    j = randint(0, alldata.nums - 1)  # 不能永远从0开始  randint范围是左右闭区间
    while j == i:
        j = randint(0, alldata.nums - 1)
    return j


def smo(max_iter: int):
    iternum = 0  # 当前迭代次数
    alpha_updated = 0  # alpha更新次数
    while iternum < max_iter:  # 是否加入其他条件？
        # 遍历全数据集
        for i in range(alldata.nums):
            j = select_j(i)  # 随机选择一个不一样的j
            # 1 计算误差
            Ei = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[i, :].T
                       - alldata.label_mat[i])
            Ej = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[j, :].T
                       - alldata.label_mat[j])
            # 松弛变量范围限定
            if ((alldata.label_mat[i] * Ei < -alldata.toler) and (alldata.alphas[i] < alldata.c)) or (
                    (alldata.label_mat[i] * Ei > alldata.toler) and (alldata.alphas[i] > 0)):
                # 2 计算上下界
                if alldata.label_mat[i] != alldata.label_mat[j]:
                    L = max(0, alldata.alphas[j] - alldata.alphas[i])
                    H = min(alldata.c, alldata.c + alldata.alphas[j] - alldata.alphas[i])
                else:
                    L = max(0, alldata.alphas[j] + alldata.alphas[i] - alldata.c)
                    H = min(alldata.c, alldata.alphas[j] + alldata.alphas[i])
                if L == H:   # 此时alpha为确定值0，无法更新
                    continue
                # 3 学习速率
                eta = alldata.data_mat[i] * alldata.data_mat[i].T + alldata.alphas[j] * alldata.alphas[j].T \
                      - 2 * alldata.data_mat[i] * alldata.data_mat[j].T
                if eta <= 0:  # 学习速率为负，无法有效更新
                    continue
                # 保存旧值
                alphai_old = alldata.alphas[i]
                alphaj_old = alldata.alphas[j]
                # 4 更新aj
                alldata.alphas[j] += alldata.label_mat[j] * (Ei - Ej) / eta
                # 5 根据取值范围修剪aj
                if alldata.alphas[j] > H:
                    alldata.alphas[j] = H
                elif alldata.alphas[j] < L:
                    alldata.alphas[j] = L
                # 6 更新ai
                alldata.alphas[i] += alldata.label_mat[i] * alldata.label_mat[j] * (alphaj_old - alldata.alphas[j])
                # 7 计算b1、b2
                b1 = alldata.b - Ei - alldata.label_mat[i] * (alldata.alphas[i] - alphai_old) * alldata.data_mat[i] * alldata.data_mat[i].T\
                    - alldata.label_mat[j] * (alldata.alphas[j] - alphaj_old) * alldata.data_mat[j] * alldata.data_mat[i].T
                b2 = alldata.b - Ej - alldata.label_mat[i] * (alldata.alphas[i] - alphai_old) * alldata.data_mat[i] * alldata.data_mat[j].T\
                    - alldata.label_mat[j] * (alldata.alphas[j] - alphaj_old) * alldata.data_mat[j] * alldata.data_mat[j].T
                # 8 更新b
                if 0 < alldata.alphas[i] < alldata.c:
                    alldata.b = b1
                elif 0 < alldata.alphas[j] < alldata.c:
                    alldata.b = b2
                else:
                    alldata.b = (b1 + b2) / 2
                alpha_updated += 1

            else:
                continue
        iternum += 1
    print()


if __name__ == '__main__':
    data_list, label_list = load_data()
    alldata = AllData(data_list, label_list, toler=0.001, c=20)  # k默认1.5
    smo(max_iter=5000)
