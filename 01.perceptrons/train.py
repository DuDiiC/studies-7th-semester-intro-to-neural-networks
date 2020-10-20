import numpy as np

# loading data
text_data = np.empty([10, 35, 5], dtype='float32')
for i in range(10):
    text_data[i] = np.loadtxt('digits/' + str(i) + '.txt')
numbers_separated = np.empty([10, 5, 7, 5], dtype='float32')
for i in range(10):
    numbers_separated[i] = np.array_split(text_data[i], 5)
numbers = np.empty([10, 5, 35], dtype='float32')
for i in range(10):
    for j in range(5):
        numbers[i][j] = numbers_separated[i][j].flatten()

