from random import randint
from time import sleep

import numpy as np
from copy import deepcopy
from sklearn import metrics


class AllData:
    def __init__(self, data_list: list, label_list: list, toler, c, k):
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
    data_list = []  # 训练集
    label_list = []
    t_data_list = []  # 测试集
    t_label_list = []
    with open('wdbc.data', 'r') as f:
        for line in f.readlines():
            data_list.append([float(linedata) for linedata in line.split(',')[2:]])  # 去除第一个编号和第二个标签
            label_list.append(-1 if line.split(',')[1] == 'M' else 1)  # M-malignant恶性_-1   B-benign 良性_1
    with open('wdbctest.data', 'r') as f:
        for line in f.readlines():
            t_data_list.append([float(linedata) for linedata in line.split(',')[2:]])  # 去除第一个编号和第二个标签
            t_label_list.append(-1 if line.split(',')[1] == 'M' else 1)  # M-malignant恶性_-1   B-benign 良性_1
    print(
        f'良性样本有{label_list.count(1) + t_label_list.count(1)}个，恶性样本有{label_list.count(-1) + t_label_list.count(-1)}个')
    return data_list, label_list, t_data_list, t_label_list


def select_j(i):
    # 需优化
    j = randint(0, alldata.nums - 1)  # 不能永远从0开始  randint范围是左右闭区间
    while j == i:
        j = randint(0, alldata.nums - 1)
    return j


def smo(max_iter: int):
    iternum = 0  # 当前迭代次数
    while iternum < max_iter:  # 是否加入其他条件？
        alpha_updated = 0  # alpha更新次数
        # 遍历全数据集
        for i in range(alldata.nums):
            j = select_j(i)  # 随机选择一个不一样的j
            # 1 计算误差
            Ei = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[i, :].T + alldata.b
                       - alldata.label_mat[i])
            Ej = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[j, :].T + alldata.b
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
                if L == H:  # 此时alpha为确定值0，无法更新
                    continue
                # 3 学习速率
                eta = alldata.inner_product[i, i] + alldata.inner_product[j, j] - 2 * alldata.inner_product[i, j]
                if eta <= 0:  # 学习速率为负，无法有效更新
                    continue
                # 保存旧值
                alphai_old = deepcopy(alldata.alphas[i])  # 深拷贝，否则numpy底层指针指向的是同一个地址
                alphaj_old = deepcopy(alldata.alphas[j])
                # 4 更新aj
                alldata.alphas[j] += alldata.label_mat[j] * (Ei - Ej) / eta
                # 5 根据取值范围修剪aj
                if alldata.alphas[j] > H:
                    alldata.alphas[j] = H
                elif alldata.alphas[j] < L:
                    alldata.alphas[j] = L
                # 6 更新ai
                alldata.alphas[i] += alldata.label_mat[i] * alldata.label_mat[j] * (alphaj_old - alldata.alphas[j])
                # 评估a更新幅度
                # if np.abs(alldata.alphas[i] - alphai_old < 0.0001) and np.abs(alldata.alphas[j] - alphaj_old) < 0.0001:
                #     continue
                # 7 计算b1、b2
                b1 = alldata.b - Ei - alldata.label_mat[i] * (alldata.alphas[i] - alphai_old) * alldata.data_mat[i] * \
                     alldata.data_mat[i].T \
                     - alldata.label_mat[j] * (alldata.alphas[j] - alphaj_old) * alldata.data_mat[j] * alldata.data_mat[
                         i].T
                b2 = alldata.b - Ej - alldata.label_mat[i] * (alldata.alphas[i] - alphai_old) * alldata.data_mat[i] * \
                     alldata.data_mat[j].T \
                     - alldata.label_mat[j] * (alldata.alphas[j] - alphaj_old) * alldata.data_mat[j] * alldata.data_mat[
                         j].T
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
        if alpha_updated != 0:
            iternum += 1
            if iternum % 1 == 0:
                evaluate()
            print(f'第{iternum}次迭代')
        else:
            print(f'alpha停止更新，训练结束，已训练{iternum}代')
            evaluate()
            # break


def evaluate():
    predicts1 = []
    for i in range(alldata.nums):
        predicts1.append(np.sign(
            float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[i, :].T + alldata.b)))
    accuracy_score = metrics.accuracy_score(alldata.label_mat, predicts1)
    print(f'训练集准确率{accuracy_score * 100: .2f}%')

    predicts2 = []
    t_data_mat = np.mat(t_data_list)
    t_label_mat = np.mat(t_label_list).T  # 转行矩阵
    for i in range(len(t_label_mat)):
        predicts2.append(np.sign(float(np.multiply(alldata.alphas, alldata.label_mat).T
                                       * alldata.data_mat * t_data_mat[i].T + alldata.b)))
    accuracy_score_t = metrics.accuracy_score(t_label_mat, predicts2)
    print(f'测试集准确率{accuracy_score_t * 100: .2f}%')
    sumresult.append((accuracy_score, accuracy_score_t))
    predicts.append((predicts1, predicts2))


if __name__ == '__main__':
    data_list, label_list, t_data_list, t_label_list = load_data()  # 训练集和测试集
    sumresult = []  # 正确率
    predicts = []  # 预测结果
    # for knum in range(1000, 2000, 50):
    # print(f'当前k取值为{knum}')
    # sleep(2)
    alldata = AllData(data_list, label_list, toler=0.1, c=200, k=50)  # k为径向基核函数中的超参数 50-100
    smo(max_iter=150)
    evaluate()
    print()
