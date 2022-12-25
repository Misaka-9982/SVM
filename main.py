import numpy as np


class AllData:
    def __init__(self, data_list: list, label_list: list, toler, c, k=20):  # 参数有严重问题
        self.data_mat = np.mat(data_list)
        self.label_mat = np.mat(label_list)
        self.toler = toler
        self.c = c
        self.nums = np.shape(self.data_mat)[0]  # 样本总数
        self.attribute = np.shape(self.data_mat)[1]  # 属性总数
        self.k = k  # 径向基核函数中的超参数
        self.inner_product = self.findinner_p()  # 用核函数求所有样本之间的内积

    def findinner_p(self) -> np.matrix:  # 核函数封装在类里
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


if __name__ == '__main__':
    data_list, label_list = load_data()
    alldata = AllData(data_list, label_list, toler=0.001, c=20)  # k默认1.5
