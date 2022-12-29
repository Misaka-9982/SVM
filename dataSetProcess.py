num = 0
with open('wdbc.data', 'r') as f:
    data_list = [line for line in f.readlines()]
    with open('wdbc2.data', 'w') as f2:
        for i in data_list:
            if i.split(',')[1] == 'M':
                f2.write(i)
            elif num <= 182:
                f2.write(i)
                num += 1
