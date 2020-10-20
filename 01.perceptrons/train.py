import numpy as np
from SPLA import SPLA
from PLA import PLA

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

# prepare data
E = numbers
T = np.full((10, 10 * 5), -1, dtype='int')
for i in range(10):
    T[i][i*5 : i*5 + 5] = 1

# perceptrons init
SPLAs = []
PLAs = []
for i in range(10):
    SPLAs.append(SPLA(i, 5 * 7, E, T[i]))
    PLAs.append(PLA(i, 5 * 7, E, T[i]))

# learning
for i in range(10): # for each digit
    SPLAs[i].train(10000)
    PLAs[i].train(10000)

# checking on teaching examples
SPLAs_predict = np.zeros(10)
PLAs_predict = np.zeros(10)
for digit in range(10):
    for i in range(10):
        for j in range(5):
            if(SPLAs[digit].predict(E[i][j]) == T[i][j]):
                SPLAs_predict[digit] += 1
            if(PLAs[digit].predict(E[i][j]) == T[i][j]):
                PLAs_predict[digit] += 1

print(SPLAs_predict)
print(PLAs_predict)
print()
print(f'{round(np.sum(SPLAs_predict) / 500 * 100, 1)} % poprawnych odpowiedzi na danych uczacych dla perceptronow bez ulepszen.\n')
print(f'{round(np.sum(PLAs_predict) / 500 * 100, 1)} % poprawnych odpowiedzi na danych uczacych dla perceptronow z kieszonka.\n')