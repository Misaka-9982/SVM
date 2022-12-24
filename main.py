import numpy as np


class AllData:
    def __init__(self, data_list: list, label_list: list, toler, c):
        self.data_mat = np.mat(data_list)
        self.label_mat = np.mat(label_list)
        self.toler = toler
        self.c = c
        self.nums = np.shape(self.data_mat)[0]  # 样本总数
        self.inner_product = np.mat(np.zeros(shape=self.nums, dtype=np.float64))  # 初始化核函数求出的内积


    def abc(self):  # 把核函数封装在类里？
        pass

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
    alldata = AllData(data_list, label_list, toler=0.001, c=20)
