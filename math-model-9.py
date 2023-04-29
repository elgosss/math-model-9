import numpy as np  # для работы с матрицами
import pandas as pd  # для вывода матрицы
import matplotlib.pyplot as plt  # для построения графиков
# pip install matplotlib

a, b = 4, 2
n = 76
y = [47, 47, 48, 49, 49, 50, 50, 51, 51, 52, 53, 55, 56, 57, 57, 57, 57, 58, 58, 59, 60, 62, 63, 64, 64, 65, 66, 66, 66, 67, 67, 67, 67, 68, 68, 69, 69, 69, 69, 70, 70, 70, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 
     73, 73, 73, 74, 75, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 79, 79, 80]
x1 = [3.0, 4.9, 2.6, 2.3, 2.9, 2.0, 6.5, 5.7, 7.6, 2.4, 4.5, 4.3, 4.2, 3.4, 7.5, 7.8, 9.0, 5.1, 5.1, 7.4, 8.3, 5.2, 14.2, 14.1, 18.6, 12.4, 10.6, 12.4, 14.0, 7.0, 9.7, 20.0, 20.7, 7.4, 13.5, 10.8, 15.6, 18.5, 
      28.0, 15.6, 19.6, 20.0, 14.3, 29.3, 33.4, 49.0, 12.1, 13.4, 23.7, 31.9, 35.3, 42.4, 22.2, 24.6, 30.8, 13.1, 78.7, 60.6, 65.8, 84.4, 53.8, 58.1, 61.1, 70.2, 78.8, 80.3, 100.0, 43.4, 70.0, 73.7, 73.9, 
      78.3, 95.9, 68.7, 85.1, 82.0]
x2 = [2.6, 3.1, 2.5, 2.6, 2.8, 2.9, 2.9, 2.5, 2.9, 3.1, 2.9, 2.5, 3.0, 2.0, 2.4, 2.9, 2.3, 1.6, 2.7, 2.8, 2.9, 1.8, 2.0, 1.6, 2.2, 2.0, 2.2, 2.9, 2.0, 3.0, 2.2, 1.5, 1.7, 3.1, 2.7, 1.1, 2.2, 1.9, 0.9, 0.2, 2.2, 
      0.3, 1.9, 2.3, 2.4, 1.3, 1.3, 0.3, 1.9, 0.8, 1.5, 0.9, 1.7, 0.6, 1.3, 1.0, 0.3, 1.4, 0.5, 2.0, 0.2, 0.5, 3.5, 1.1, 0.8, 0.4, 1.0, 0.6, 0.5, 0.2, 0.7, 1.3, 1.0, 0.6, 1.6, 0.3]
x3 = [2.4, 2.8, 2.5, 2.7, 2.1, 2.7, 2.5, 2.7, 2.6, 3.1, 2.8, 2.4, 2.8, 1.7, 2.2, 3.1, 2.3, 2.1, 2.7, 2.7, 3.3, 2.0, 2.7, 2.5, 2.4, 2.6, 2.7, 3.5, 3.1, 3.8, 3.4, 1.6, 2.1, 4.0, 2.9, 1.1, 3.2, 3.0, 1.3, 0.2, 4.1, 
      0.6, 2.6, 3.0, 2.7, 1.8, 2.0, 0.7, 2.8, 1.8, 2.1, 1.9, 2.4, 1.0, 2.0, 1.8, 0.1, 1.5, 0.1, 1.7, 1.0, 1.7, 3.5, 1.4, 0.5, 0.5, 1.1, 0.9, 0.8, 0.4, 0.6, 1.0, 0.8, 0.3, 1.3, 0.6]
x4 = [113, 124, 117, 98, 99, 123, 95, 96, 90, 89, 80, 91, 88, 72, 55, 56, 64, 79, 58, 73, 90, 68, 56, 51, 50, 55, 39, 44, 47, 45, 36, 44, 48, 46, 41, 34, 36, 39, 35, 13, 34, 14, 37, 23, 12, 16, 16, 11, 33, 13, 12, 
      10, 23, 18, 22, 13, 6, 7, 5, 4, 7, 6, 8, 6, 6, 8, 8, 8, 6, 7, 6, 6, 6, 4, 5, 4]
for i in range(n):
    y[i] = y[i] + a
    x4[i] = x4[i] + b


# а) оценить коэффициенты линейной регрессионной модели (y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4)
x_1 = [1] * n
matrix_x = np.array([x_1, x1, x2, x3, x4])
matrix_y = np.array([y])
def koeff_model(matrix_x, matrix_y):
    matrix_xT = matrix_x.transpose()
    matrix_yT = matrix_y.transpose()
    matrix_x_xy_xT = np.dot(matrix_x, matrix_xT)
    matrix_x_xy_yT = np.dot(matrix_x, matrix_yT)
    inverse_matrix_x_xy_xT = np.linalg.inv(matrix_x_xy_xT)  # обратная матрица
    matrix_coeff = np.dot(inverse_matrix_x_xy_xT, matrix_x_xy_yT)
    return matrix_coeff
print('а) Оценить коэффициенты линейной регрессионной модели')
print('y = ' + str(round(koeff_model(matrix_x, matrix_y)[0, 0], 3)) + ' + ' + str(round(koeff_model(matrix_x, matrix_y)[1, 0], 3)) + '*x1 + ' + 
      str(round(koeff_model(matrix_x, matrix_y)[2, 0], 3)) + '*x2 + ' + str(round(koeff_model(matrix_x, matrix_y)[3, 0], 3)) + '*x3 + ' + 
      str(round(koeff_model(matrix_x, matrix_y)[4, 0], 3)) + '*x4\n')


# в) построить матрицу частных коэфф. корреляции. Установить, какие факторы коллениарны. Построить уравнение множественной регрессии, обосновав отбор факторов.
arr = [y, x1, x2, x3, x4]
av_sum = [0]*5  # средние суммы y, х1, х2, х3, х4
for i in range(5):
    for j in range(n):
        av_sum[i] += arr[i][j]
    av_sum[i] /= n
av_x, av_y = [0]*10, [0]*10   # средние x, средние y
av_xy = [0]*10    # средние умножения пар
s_x, s_y = [0]*10, [0]*10   # среднеквадратические отклонения
j1, t = 0, -1
for i in range(4):
    j1 += 1
    for j in range(j1, 5):
        t += 1
        for k in range(n):
            av_x[t] += arr[j][k]
            av_y[t] += arr[i][k]
            av_xy[t] += arr[i][k] * arr[j][k]
            s_x[t] += arr[j][k]**2
            s_y[t] += arr[i][k]**2
        av_x[t] /= n
        av_y[t] /= n
        av_xy[t] /= n
        s_x[t] = (s_x[t] / n - av_x[t]**2)**0.5
        s_y[t] = (s_y[t] / n - av_y[t]**2)**0.5

pair_coeff = [[0]*5 for i in range(5)]  # матрица парных коэфф. корреляции
k = -1
for i in range(5):
    for j in range(5):
        if (i == j):
            pair_coeff[i][j] = 1
        elif (j < i):
            pair_coeff[i][j] = pair_coeff[j][i]
        else:
            k += 1
            pair_coeff[i][j] = (av_xy[k] - av_x[k] * av_y[k]) / (s_x[k] * s_y[k])

M = [pair_coeff]*25  # миноры
k = -1
for i in range(5):
    for j in range(5):
        k += 1
        M[k] = np.delete(M[k], i, 0)    # удаляем строку
        M[k] = np.delete(M[k], j, 1)    # удаляем столбец

algebr_add = [[0]*5 for i in range(5)]  # алгебраическое дополнение
k = -1
for i in range(5):
    for j in range(5):
        k += 1
        algebr_add[i][j] = (-1)**(i + j) * np.linalg.det(M[k])

partial_coeff = [[0]*5 for i in range(5)]  # матрица частных коэфф. корреляции
for i in range(5):
    for j in range(5):
        partial_coeff[i][j] = - algebr_add[i][j] / (algebr_add[i][i] * algebr_add[j][j])**0.5

# обоснование факторов
coll = [0]*2    # коллинеарные факторы
k = -1
for i in range(1, 5):
    for j in range(1, 5):
        if (0.8 < abs(partial_coeff[i][j]) < 1):
            k += 1
            coll[k] = j

if (abs(pair_coeff[0][coll[0]]) < abs(pair_coeff[0][coll[1]])):
    ex_f = coll[0]
else:
    ex_f = coll[1]
partial_coeff = pd.DataFrame(partial_coeff)
partial_coeff.index = ['y', 'x1', 'x2', 'x3', 'x4']
partial_coeff.columns = ['y', 'x1', 'x2', 'x3', 'x4']
print('в) Построить матрицу частных коэфф. корреляции. Установить, какие факторы коллениарны. Построить уравнение множественной регрессии, обосновав отбор факторов.')
print('Матрица частных коэффициентов корреляции:')
print(round(partial_coeff, 3))
print('Факторы x' + str(coll[1]) + ' и x' + str(coll[0]) + ' - коллинеарны, теснота связи между ними высока')
print('Из модели исключим фактор х' + str(ex_f))

matrix_x_1 = np.delete(matrix_x, ex_f, 0) # исключили фактор

xIndex = [1, 2, 3, 4]
xIndex.remove(ex_f)

print('Уравнение множественной регресии:')
print('y = ' + str(round(koeff_model(matrix_x_1, matrix_y)[0, 0], 3)) + ' + ' + str(round(koeff_model(matrix_x_1, matrix_y)[1, 0], 3)) + '*x' + str(xIndex[0]) + ' + ' + 
      str(round(koeff_model(matrix_x_1, matrix_y)[2, 0], 3)) + '*x' + str(xIndex[1]) + ' + ' + str(round(koeff_model(matrix_x_1, matrix_y)[3, 0], 3)) + '*x' + str(xIndex[2]) + '\n')


# г) построить графики остатков. Провести тестирование ошибок уравнения множественной регрессии на гетероскедастичность, применив тест Готфельда-Квандта.
y_x = [0]*n
for i in range(n):
    y_x[i] = koeff_model(matrix_x, matrix_y)[0, 0] + koeff_model(matrix_x, matrix_y)[1, 0] * x1[i] + koeff_model(matrix_x, matrix_y)[2, 0] * x2[i] + \
        koeff_model(matrix_x, matrix_y)[3, 0] * x3[i] + koeff_model(matrix_x, matrix_y)[4, 0] * x4[i] # y от x (предсказанные)

e = [0]*n   # остаток
e2 = [0]*n  # квадрат остатка
for i in range(n):
    e[i] = y[i] - y_x[i]
    e2[i] = e[i]**2

# графики остатков
plt.subplot(2, 4, 1)
plt.scatter(x1, e)
plt.title('Фактор X1')

plt.subplot(2, 4, 2)
plt.scatter(x2, e)
plt.title('Фактор X2')

plt.subplot(2, 4, 3)
plt.scatter(x3, e)
plt.title('Фактор X3')

plt.subplot(2, 4, 4)
plt.scatter(x4, e)
plt.title('Фактор X4')

plt.subplot(2, 4, 5)
plt.scatter(x1, e2)
plt.title('Квадрат остатков от X1')

plt.subplot(2, 4, 6)
plt.scatter(x2, e2)
plt.title('Квадрат остатков от X2')

plt.subplot(2, 4, 7)
plt.scatter(x3, e2)
plt.title('Квадрат остатков от X3')

plt.subplot(2, 4, 8)
plt.scatter(x4, e2)
plt.title('Квадрат остатков от X4')

plt.show()

D = [0]*4   # дисперсия
for i in range(4):
    for j in range(n):
        D[i] += (arr[i + 1][j] - av_x[i])**2
    D[i] /= n

print('г) построить графики остатков. Провести тестирование ошибок уравнения множественной регрессии на гетероскедастичность, применив тест Готфельда-Квандта')
print('Дисперсии: x1 = ' + str(round(D[0], 3)) + ', x2 = ' + str(round(D[1], 3)) + ', x3 = ' + str(round(D[2], 3)) + ', x4 = ' + str(round(D[3], 3)))
max_D = D[0]  # максимальная дисперсия
for i in range(1, len(D)):
    if D[i] > max_D:
        max_D = D[i]
        ind_D = i   # индекс макс. дисперсии

ind_D += 1
# ранжируем все наблюдения по возрастанию
for i in range(n):
    for j in range(i + 1, n):
        if arr[ind_D][i] > arr[ind_D][j]:
            for k in range(5):
                temp = arr[k][i]
                arr[k][i] = arr[k][j]
                arr[k][j] = temp

k = 25
arr_1, arr_3 = [[0]*k for i in range(5)], [[0]*k for i in range(5)]  # первая и третья подвыборка
for i in range(5):
    m = -1
    for j in range(k):
        arr_1[i][j] = arr[i][j]
    for j in range(n - k, n):
        m += 1
        arr_3[i][m] = arr[i][j]

x_1 = [1]*k
matrix_x_arr_1 = np.array([x_1, arr_1[1], arr_1[2], arr_1[3], arr_1[4]])
matrix_y_arr_1 = np.array([arr_1[0]])

def e2_sum(matrix_x, matrix_y, arr, k):
    y_x = [0]*k
    for i in range(k):
        y_x[i] = koeff_model(matrix_x, matrix_y)[0, 0] + koeff_model(matrix_x, matrix_y)[1, 0] * arr[1][i] + koeff_model(matrix_x, matrix_y)[2, 0] * arr[2][i] + \
            koeff_model(matrix_x, matrix_y)[3, 0] * arr[3][i] + koeff_model(matrix_x, matrix_y)[4, 0] * arr[4][i] # y от x (предсказанные)
    e = [0]*k   # остаток
    e2 = [0]*k  # квадрат остатка
    sum_e2 = 0
    for i in range(k):
        e[i] = arr[0][i] - y_x[i]
        e2[i] = e[i]**2
        sum_e2 += e2[i]
    return sum_e2

sum_e2_arr_1 = e2_sum(matrix_x_arr_1, matrix_y_arr_1, arr_1, k)

matrix_x_arr_3 = np.array([x_1, arr_3[1], arr_3[2], arr_3[3], arr_3[4]])
matrix_y_arr_3 = np.array([arr_3[0]])

sum_e2_arr_3 = e2_sum(matrix_x_arr_3, matrix_y_arr_3, arr_3, k)

print('Сумма квадратов остатков отклонений для первой подвыборки ESS1 = ' + str(round(sum_e2_arr_1, 3)))
print('Сумма квадратов остатков отклонений для третьей подвыборки ESS3 = ' + str(round(sum_e2_arr_3, 3)))

if sum_e2_arr_1 > sum_e2_arr_3:
    F = sum_e2_arr_1 / sum_e2_arr_3    # F-статистика
else:
    F = sum_e2_arr_3 / sum_e2_arr_1

print('F-статистика = ' + str(round(F, 3)))

F_crit = 2.1242

if F > F_crit:
    print('Fнабл > Fкр => в модели имеется гетероскедастичность')
else:
    print('Fнабл < Fкр => в модели не имеется гетероскедастичность')

