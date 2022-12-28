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
        self.inner_product = self.findinner_p(self.data_mat, self.data_mat, self.k)  # 用核函数求所有样本之间的内积
        self.alphas = np.mat(np.zeros(shape=(self.nums, 1), dtype=np.float64))  # alpha乘子
        self.b = 0  # 标量参数b
        self.E_cache = np.mat(np.zeros((self.nums, 2)))  # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。

    @staticmethod
    def findinner_p(data_mat1: np.matrix, data_mat2: np.matrix, k) -> np.matrix:  # 径向基核函数 封装在类里
        nums1 = data_mat1.shape[0]
        nums2 = data_mat2.shape[0]
        inner_product = np.mat(np.zeros(shape=(nums1, nums2)), dtype=np.float64)  # 初始化核函数求出的内积
        for i in range(nums1):  # i j都是行索引，一行一个样本
            for j in range(nums2):
                if nums1 > 1 and nums2 > 1:
                    inner_product[i, j] = np.exp(-((np.linalg.norm(data_mat1[i, :] - data_mat2[j, :])) ** 2)
                                                 / (k ** 2))
                elif nums1 == 1 and nums2 > 1:
                    inner_product[i, j] = np.exp(-((np.linalg.norm(data_mat1[i] - data_mat2[j, :])) ** 2)
                                                 / (k ** 2))
                elif nums1 > 1 and nums2 == 1:
                    inner_product[i, j] = np.exp(-((np.linalg.norm(data_mat1[i, :] - data_mat2[j])) ** 2)
                                                 / (k ** 2))
                else:
                    inner_product[i, j] = np.exp(-((np.linalg.norm(data_mat1[i] - data_mat2[j])) ** 2)
                                                 / (k ** 2))
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


def rand_j(i):
    # 启发式选择
    j = randint(0, alldata.nums - 1)  # 不能永远从0开始  randint范围是左右闭区间
    while j == i:
        j = randint(0, alldata.nums - 1)
    return j


def select_j(i, alldata, Ei):
    max_k = -1
    max_delta_e = 0
    Ej = 0  # 初始化
    alldata.E_cache[i] = [1, Ei]  # 根据Ei更新误差缓存
    valid_cache_index = np.nonzero(alldata.E_cache[:, 0].A)[0]  # 返回误差不为0的数据的索引值
    # 优先更新误差不为0的点
    if len(valid_cache_index) > 1:  # 有不为0的误差
        for k in valid_cache_index:  # 找到最大的Ek
            if k == i:  # 不计算i,浪费时间
                continue
            Ek = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[k, :].T + alldata.b
                       - alldata.label_mat[k])  # 计算Ek
            delta_e = abs(Ei - Ek)  # 计算|Ei-Ek|
            if delta_e > max_delta_e:  # 找到最大误差之差
                max_k = k
                max_delta_e = delta_e
                Ej = Ek
        return max_k, Ej  # 返回maxK,Ej
    else:  # 没有不为0的误差
        j = rand_j(i)  # 随机选择alpha_j的索引值
        Ej = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[j, :].T + alldata.b
                   - alldata.label_mat[j])  # 计算Ej
    return j, Ej  # j,Ej


def update_e(k):
    Ek = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[k, :].T + alldata.b
               - alldata.label_mat[k])  # 计算Ek
    alldata.E_cache[k] = [1, Ek]  # 更新误差缓存


def smo(max_iter: int):
    iternum = 0  # 当前迭代次数
    entire_data = True
    alpha_updated = 0
    while (iternum < max_iter and alpha_updated > 0) or entire_data:  # 是否加入其他条件？
        alpha_updated = 0  # alpha更新次数
        i_range = []  # 或range对象 表示i的迭代范围

        if entire_data:  # 遍历全数据集
            i_range = range(alldata.nums)
        else:  # 遍历非边界值
            i_range = np.nonzero((alldata.alphas.A != 0) * (alldata.alphas.A != alldata.c))[0]  # 对应位置求and 直接用and会引发报错

        for i in i_range:
            # 1 计算误差
            Ei = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[i, :].T + alldata.b
                       - alldata.label_mat[i])
            # 松弛变量范围限定                                        受软间隔影响此时alpha可能小于0，因此只需满足一侧，另一边同理
            if ((alldata.label_mat[i] * Ei < -alldata.toler) and (alldata.alphas[i] < alldata.c)) or (
                    (alldata.label_mat[i] * Ei > alldata.toler) and (alldata.alphas[i] > 0)):
                j, Ej = select_j(i, alldata, Ei)
                # j = select_j(i)  # 随机选择一个不一样的j
                # Ej = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[j, :].T + alldata.b
                #            - alldata.label_mat[j])

                # 2 计算上下界
                if alldata.label_mat[i] != alldata.label_mat[j]:
                    L = max(0, alldata.alphas[j] - alldata.alphas[i])
                    H = min(alldata.c, alldata.c + alldata.alphas[j] - alldata.alphas[i])
                else:
                    L = max(0, alldata.alphas[j] + alldata.alphas[i] - alldata.c)
                    H = min(alldata.c, alldata.alphas[j] + alldata.alphas[i])
                if L == H:  # 此时alpha为确定值0，无法更新
                    print(f'上下界相等，alpha已更新{alpha_updated}次')
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
                # 更新ej到缓存
                update_e(j)
                # 6 更新ai
                alldata.alphas[i] += alldata.label_mat[i] * alldata.label_mat[j] * (alphaj_old - alldata.alphas[j])
                update_e(i)
                # 评估a更新幅度
                # if np.abs(alldata.alphas[i] - alphai_old < 0.0001) and np.abs(alldata.alphas[j] - alphaj_old) < 0.0001:
                #     continue
                # 7 计算b1、b2
                b1 = alldata.b - Ei - alldata.label_mat[i] * (alldata.alphas[i] - alphai_old) * alldata.inner_product[
                    i, i] \
                     - alldata.label_mat[j] * (alldata.alphas[j] - alphaj_old) * alldata.inner_product[j, i]
                b2 = alldata.b - Ej - alldata.label_mat[i] * (alldata.alphas[i] - alphai_old) * alldata.inner_product[
                    i, j] \
                     - alldata.label_mat[j] * (alldata.alphas[j] - alphaj_old) * alldata.inner_product[j, j]
                # 8 更新b
                if 0 < alldata.alphas[i] < alldata.c:
                    alldata.b = b1
                elif 0 < alldata.alphas[j] < alldata.c:
                    alldata.b = b2
                else:
                    alldata.b = (b1 + b2) / 2
                alpha_updated += 2

            else:
                continue

        iternum += 1
        print(f'第{iternum}次迭代，alpha更新{alpha_updated}次')
        if iternum % 1 == 0:
            evaluate()
        if entire_data:  # 全样本更新后标志位置False
            print('全样本更新')
            entire_data = False
            if alpha_updated == 0:  # alpha全样本都不更新即结束
                print(f'alpha停止更新，共迭代{iternum}次')
                break
        elif alpha_updated == 0:  # 非全样本 alpha无更新,更换为全样本模式
            entire_data = True


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
                                       * alldata.findinner_p(alldata.data_mat, t_data_mat[i, :],
                                                             alldata.k) + alldata.b)))
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
    # k为径向基核函数中的超参数 50-100  C最开始默认200 toler 0.0001
    alldata = AllData(data_list, label_list, toler=0.0001, c=1, k=1597430)
    smo(max_iter=550)
    evaluate()
    print()
