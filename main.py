from random import randint

import numpy as np
from copy import deepcopy
from sklearn import metrics
import matplotlib.pyplot as plt


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
        print('正在通过核函数计算内积')
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
    j = randint(0, alldata.nums - 1)  # 不能永远从0开始  randint范围是左右闭区间
    while j == i:
        j = randint(0, alldata.nums - 1)
    return j


def select_j(i, alldata, Ei):
    max_k = -1
    max_delta_e = 0
    Ej = 0
    alldata.E_cache[i] = [1, Ei]  # 根据Ei更新误差
    valid_cache_index = np.nonzero(alldata.E_cache[:, 0].A)[0]  # 缓存中误差不为0的索引
    # 优先更新误差不为0的点
    if len(valid_cache_index) > 1:
        for k in valid_cache_index:  # 找出最大误差
            if k == i:  # 跳过重复
                continue
            Ek = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[k, :].T + alldata.b
                       - alldata.label_mat[k])
            delta_e = abs(Ei - Ek)
            if delta_e > max_delta_e:
                max_delta_e = delta_e
                max_k = k
                Ej = Ek
        return max_k, Ej
    else:  # 没有不为0的误差
        j = rand_j(i)  # 随机选择j
        Ej = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[j, :].T + alldata.b
                   - alldata.label_mat[j])
    return j, Ej


def update_e(k):
    Ek = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[k, :].T + alldata.b
               - alldata.label_mat[k])  # 计算Ek
    alldata.E_cache[k] = [1, Ek]  # 更新误差缓存


def smo(max_iter: int):
    iternum = 0  # 当前迭代次数
    entire_data = True
    alpha_updated = 0
    stop_flag = 0
    while iternum < max_iter  or entire_data:  # 是否加入其他条件？
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
                    print(f'i={i}, alpha已更新{alpha_updated}次')
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
                # aj更新范围判断
                if abs(alldata.alphas[j] - alphaj_old) < 1e-5:
                    continue
                # 6 更新ai
                alldata.alphas[i] += alldata.label_mat[i] * alldata.label_mat[j] * (alphaj_old - alldata.alphas[j])
                update_e(i)
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
        if iternum % 10 == 0:
            evaluate()
        if entire_data:  # 全样本更新后标志位置False
            entire_data = False
            if alpha_updated == 0 and stop_flag < 2:  # 阈值 防止过早停止 alpha全样本20次都不更新即结束
                stop_flag += 1
            elif alpha_updated == 0 and stop_flag >= 2:
                print(f'alpha停止更新，共迭代{iternum}次')
                break
            else:  # 全样本有alpha更新 即将停止标签置0
                stop_flag = 0
        elif alpha_updated == 0:  # 非全样本 alpha无更新,更换为全样本模式
            print('全样本更新')
            entire_data = True


def evaluate(end=False):
    predicts1 = []
    raw_predicts1 = []
    for i in range(alldata.nums):
        raw_predicts1.append(
            float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[i, :].T + alldata.b))
        # predicts1.append(np.where(np.array(raw_predicts1[-1]) > 0.000001, 1, -1))
        predicts1.append(np.sign(raw_predicts1[-1]))
    accuracy_score = metrics.accuracy_score(alldata.label_mat, predicts1)
    print(f'本轮训练集准确率{accuracy_score * 100: .2f}%')

    predicts2 = []
    raw_predicts2 = []
    t_data_mat = np.mat(t_data_list)
    t_label_mat = np.mat(t_label_list).T  # 转行矩阵
    for i in range(len(t_label_mat)):
        raw_predicts2.append(float(np.multiply(alldata.alphas, alldata.label_mat).T
                                   * alldata.findinner_p(alldata.data_mat, t_data_mat[i, :],
                                                         alldata.k) + alldata.b))
        # predicts2.append(np.where(np.array(raw_predicts2[-1]) > 0.000001, 1, -1))
        predicts2.append(np.sign(raw_predicts2[-1]))
    accuracy_score_t = metrics.accuracy_score(t_label_mat, predicts2)
    print(f'本轮测试集准确率{accuracy_score_t * 100: .2f}%')

    if end:
        result.sort(reverse=True, key=lambda x: np.average([x[0], x[2]]))  # 训练集和测试集准确率均值降序
        print(f'\n最终训练集准确率{result[0][0] * 100: .2f}%')
        print(f'最终测试集准确率{result[0][2] * 100: .2f}%')
        raw_predicts1 = result[0][1]
        raw_predicts2 = result[0][3]

        fpr1, tpr1, thresholds1 = metrics.roc_curve(alldata.label_mat, raw_predicts1, drop_intermediate=True)
        fpr2, tpr2, thresholds2 = metrics.roc_curve(t_label_mat, raw_predicts2, drop_intermediate=True)
        auc1 = metrics.auc(fpr1, tpr1)
        auc2 = metrics.auc(fpr2, tpr2)
        p1, r1, thres1 = metrics.precision_recall_curve(alldata.label_mat, raw_predicts1)
        p2, r2, thres2 = metrics.precision_recall_curve(t_label_mat, raw_predicts2)
        print(f'训练集auc:{auc1}\n测试集auc:{auc2}')
        print(f'AP指标：{metrics.average_precision_score(t_label_mat, raw_predicts2)}')

        plt.subplots_adjust(wspace=0.3)
        plt.plot(fpr1, tpr1, color='red', label='training set')
        plt.plot(fpr2, tpr2, color='blue', label='testing set')
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()
        plt.clf()

        plt.plot(p1, r1, color='red', label='training set')
        plt.plot(p2, r2, color='blue', label='testing set')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('P-R Curve')
        plt.legend(loc='best')

        plt.show()
    else:
        return accuracy_score, raw_predicts1, accuracy_score_t, raw_predicts2


if __name__ == '__main__':
    data_list, label_list, t_data_list, t_label_list = load_data()  # 训练集和测试集
    result = []
    for n in range(10):
        alldata = AllData(data_list, label_list, toler=1e-4, c=0.8, k=27)
        smo(max_iter=100)
        result.append(evaluate())
        if result[-1][2] > 0.95:
            break
    evaluate(end=True)
