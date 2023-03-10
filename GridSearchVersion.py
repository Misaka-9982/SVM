from random import randint

import numpy as np
from copy import deepcopy
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV


class AllData(BaseEstimator, ClassifierMixin):
    def __init__(self, toler=0.0001, c=1, k=1600000):
        self.data_mat = np.mat(data_list)
        self.label_mat = np.mat(label_list).T  # 行向量
        self.toler = toler
        self.c = c
        self.nums = np.shape(self.data_mat)[0]  # 样本总数
        self.attribute = np.shape(self.data_mat)[1]  # 属性总数
        self.k = k  # 径向基核函数中的超参数
        self.inner_product = None  # 用核函数求所有样本之间的内积
        self.alphas = np.mat(np.zeros(shape=(self.nums, 1), dtype=np.float64))  # alpha乘子
        self.b = 0  # 标量参数b
        self.E_cache = np.mat(np.zeros((self.nums, 2)))  # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.accuracy = 0

    def fit(self, data, label):
        self.smo(100, data, label)   # 自动调用训练集，初始化的时候即读入
        self.accuracy = self.evaluate()  # 自动调用测试集

        return self

    def score(self, X, y, sample_weight=None):
        return self.accuracy

    def smo(self, max_iter: int, data, label):
        self.inner_product = self.findinner_p(self.data_mat, self.data_mat, self.k)
        iternum = 0  # 当前迭代次数
        entire_data = True
        alpha_updated = 0
        while (iternum < max_iter and alpha_updated > 0) or entire_data:  # 是否加入其他条件？
            alpha_updated = 0  # alpha更新次数
            i_range = []  # 或range对象 表示i的迭代范围

            if entire_data:  # 遍历全数据集
                i_range = range(self.nums)
            else:  # 遍历非边界值
                i_range = np.nonzero((self.alphas.A != 0) * (self.alphas.A != self.c))[0]  # 对应位置求and 直接用and会引发报错

            for i in i_range:
                # 1 计算误差
                Ei = float(np.multiply(self.alphas, self.label_mat).T * self.inner_product[i, :].T + self.b
                           - self.label_mat[i])
                # 松弛变量范围限定                                        受软间隔影响此时alpha可能小于0，因此只需满足一侧，另一边同理
                if ((self.label_mat[i] * Ei < -self.toler) and (self.alphas[i] < self.c)) or (
                        (self.label_mat[i] * Ei > self.toler) and (self.alphas[i] > 0)):
                    j, Ej = self.select_j(i, self, Ei)
                    # j = select_j(i)  # 随机选择一个不一样的j
                    # Ej = float(np.multiply(self.alphas, self.label_mat).T * self.inner_product[j, :].T + self.b
                    #            - self.label_mat[j])

                    # 2 计算上下界
                    if self.label_mat[i] != self.label_mat[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.c, self.c + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[j] + self.alphas[i] - self.c)
                        H = min(self.c, self.alphas[j] + self.alphas[i])
                    if L == H:  # 此时alpha为确定值0，无法更新
                        # print(f'上下界相等，alpha已更新{alpha_updated}次')
                        continue
                    # 3 学习速率
                    eta = self.inner_product[i, i] + self.inner_product[j, j] - 2 * self.inner_product[i, j]
                    if eta <= 0:  # 学习速率为负，无法有效更新
                        continue
                    # 保存旧值
                    alphai_old = deepcopy(self.alphas[i])  # 深拷贝，否则numpy底层指针指向的是同一个地址
                    alphaj_old = deepcopy(self.alphas[j])
                    # 4 更新aj
                    self.alphas[j] += self.label_mat[j] * (Ei - Ej) / eta
                    # 5 根据取值范围修剪aj
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif self.alphas[j] < L:
                        self.alphas[j] = L
                    # 更新ej到缓存
                    self.update_e(j)
                    # aj更新范围判断
                    if abs(self.alphas[j] - alphaj_old) < 1e-5:
                        continue
                    # 6 更新ai
                    self.alphas[i] += self.label_mat[i] * self.label_mat[j] * (alphaj_old - self.alphas[j])
                    self.update_e(i)
                    # 7 计算b1、b2
                    b1 = self.b - Ei - self.label_mat[i] * (self.alphas[i] - alphai_old) * self.inner_product[
                        i, i] \
                         - self.label_mat[j] * (self.alphas[j] - alphaj_old) * self.inner_product[j, i]
                    b2 = self.b - Ej - self.label_mat[i] * (self.alphas[i] - alphai_old) * self.inner_product[
                        i, j] \
                         - self.label_mat[j] * (self.alphas[j] - alphaj_old) * self.inner_product[j, j]
                    # 8 更新b
                    if 0 < self.alphas[i] < self.c:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.c:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    alpha_updated += 2

                else:
                    continue

            iternum += 1
            print(f'第{iternum}次迭代，alpha更新{alpha_updated}次')
            if iternum % 10 == 0:
                self.evaluate()
            if entire_data:  # 全样本更新后标志位置False
                print('全样本更新')
                entire_data = False
                if alpha_updated == 0:  # alpha全样本都不更新即结束
                    print(f'alpha停止更新，共迭代{iternum}次')
                    break
            elif alpha_updated == 0:  # 非全样本 alpha无更新,更换为全样本模式
                entire_data = True

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

    def rand_j(self, i):
        # 启发式选择
        j = randint(0, self.nums - 1)  # 不能永远从0开始  randint范围是左右闭区间
        while j == i:
            j = randint(0, self.nums - 1)
        return j

    def update_e(self, k):
        Ek = float(np.multiply(self.alphas, self.label_mat).T * self.inner_product[k, :].T + self.b
                   - self.label_mat[k])  # 计算Ek
        self.E_cache[k] = [1, Ek]  # 更新误差缓存

    def select_j(self, i, alldata, Ei):
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
                    max_delta_e = delta_e
                    max_k = k
                    Ej = Ek
            return max_k, Ej  # 返回maxK,Ej
        else:  # 没有不为0的误差
            j = self.rand_j(i)  # 随机选择alpha_j的索引值
            Ej = float(np.multiply(alldata.alphas, alldata.label_mat).T * alldata.inner_product[j, :].T + alldata.b
                       - alldata.label_mat[j])  # 计算Ej
        return j, Ej  # j,Ej

    def evaluate(self):
        predicts1 = []
        for i in range(self.nums):
            predicts1.append(np.sign(
                float(np.multiply(self.alphas, self.label_mat).T * self.inner_product[i, :].T + self.b)))
        accuracy_score = metrics.accuracy_score(self.label_mat, predicts1)
        print(f'训练集准确率{accuracy_score * 100: .2f}%')

        predicts2 = []
        t_data_mat = np.mat(t_data_list)
        t_label_mat = np.mat(t_label_list).T  # 转行矩阵
        for i in range(len(t_label_mat)):
            predicts2.append(np.sign(float(np.multiply(self.alphas, self.label_mat).T
                                           * self.findinner_p(self.data_mat, t_data_mat[i, :],
                                                              self.k) + self.b)))
        accuracy_score_t = metrics.accuracy_score(t_label_mat, predicts2)
        print(f'测试集准确率{accuracy_score_t * 100: .2f}%')

        return min(accuracy_score_t, accuracy_score)


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


if __name__ == '__main__':
    data_list, label_list, t_data_list, t_label_list = load_data()  # 训练集和测试集
    # for knum in range(1000, 2000, 50):
    # print(f'当前k取值为{knum}')
    # sleep(2)
    # k为径向基核函数中的超参数 50-100  C最开始默认200 toler 0.0001
    # alldata = AllData(toler=0.0001, c=1, k=1600000)
    # smo(max_iter=100)
    classifier = AllData()
    parameters = {'toler': np.linspace(1e-4, 1e-4, 1), 'c': np.linspace(0.8, 0.8, 1),
                  'k': np.linspace(26.5, 26.5, 1)}
    gs = GridSearchCV(classifier, parameters, n_jobs=-1, verbose=2)
    res = gs.fit(data_list, label_list)
    print(f'best_params_:{gs.best_params_}')
    print(f'best_score_: {gs.best_score_}')
    print(f'cv_results_:{gs.cv_results_}')
