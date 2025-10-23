# Лабораторная работа №3
# Однофакторный регрессионный анализ
# Вариант №4

import numpy as np
import matplotlib.pyplot as plot

np.set_printoptions(suppress=True)

# Экспериментальные данные части 1
x = np.array([-2, 0, 2, 3, 4])
y = np.array([14, 1, -3, -2, 1])

# левая часть СЛАУ
A = np.array([[
    sum(x ** 4), sum(x ** 3), sum(x ** 2)],
    [sum(x ** 3), sum(x ** 2), sum(x)],
    [sum(x ** 2), sum(x), len(x)]
])

# правая часть СЛАУ
b = np.array([sum((x ** 2) * y), sum(x * y), sum(y)]) 

print(f"A={A}")
print(f"b={b}")


# решение в матричной форме
F = np.array([
    [x[0] ** 2, x[0], 1],
    [x[1] ** 2, x[1], 1],
    [x[2] ** 2, x[2], 1],
    [x[3] ** 2, x[3], 1],
    [x[4] ** 2, x[4], 1]
])

F_T = F.transpose()
F_T_F = F_T.dot(F)
F_T_F_1 = np.linalg.inv(F_T_F)
FTY = F_T.dot(y)
A_ = F_T_F_1.dot(FTY)

print(f"F =\n{F}")
print(f"F траспонированная =\n{F_T}")
print(f"Произведение F на FT =\n{F_T_F}")
print(f"Обратная матрица (FT*F)^-1 =\n{F_T_F_1}" )
print(f"Произведение FT на Y = FTY = \n{FTY}")
print(f"(FT*F)^-1 * FTY = Вектор A =\n{A_}")

# проверка адекватности по критерию фишера


def f1(x):
    """Функция линейной регрессии части 1"""
    return 1.061 * (x**2) - 4.265*x + 1.166

def f1_(x):
    """Скорректированная функция линейной регрессии части 1"""
    return 1.061 * (x**2) - 4.265*x


def D(y):
    """Оценка дисперсии выходной переменной"""
    mean = np.average(y)
    d = 0
    for i in range(len(y)):
        d += (y[i] - mean) ** 2
    d = d / (len(y) - 1)
    return d

def D1(y, f, x, k):
    """Оценка остаточной дисперсии"""
    d = 0
    for i in range(len(y)):
        d += (y[i] - f(x[i])) ** 2
    d = d / (len(y) - 1 - k)    
    return d

F_sogl = D(y) / D1(y, f1_, x, 2)

print(f"Оценка дисперсии выходной переменной: {D(y)}")
print(f"Оценка остаточной дисперсии: {D1(y, f1_, x, 2)}")
print(f"Показатель согласованности по критерию Фишера: {F_sogl}")



# проверка согласованности по критерию Стьюдента
a = [1.061, 4.265, 1.166]

sigma_1 = D1(y, f1, x, 2)

K_A = F_T_F_1 * sigma_1
diag = np.array([K_A[i,i] for i in range(len(K_A))])
t = a / (diag ** (1/2))

print(f"Корреляционная матрица:\n{K_A}")
print(f"Главная диагональ: {diag}")
print(f"СКО отклонения aj: {diag ** (1/2)}")
print(f"Вектор коэффициентов |a|: {a}")
print(f"Значения показателя согласованности: f{t}")


def draw_compare_plot(X,Y,f1):
    """Нарисовать сравнение экспериментальных данных и регрессии"""    
    x = np.arange(min(X), max(X), 0.01)
    y = [f1(x_val) for x_val in x]
    plot.scatter(x, y, label="Регрессия")
    plot.scatter(X, Y, label="Экспериментальные данные")

 

# Экспериментальные данные части 2
x = np.array([-3, -2, 0, 0.5, 1, 2, 4])
y = np.array([-59, -22, -6, -5, -4, 10, 119])

# левая часть СЛАУ
A = np.array([
    [sum(x ** 6), sum(x ** 5), sum(x ** 4), sum(x ** 3)],
    [sum(x ** 5), sum(x ** 4), sum(x ** 3), sum(x ** 2)],
    [sum(x ** 4), sum(x ** 3), sum(x ** 2), sum(x)],
    [sum(x ** 3), sum(x ** 2), sum(x), len(x)]
])

# правая часть СЛАУ
b = np.array([sum((x ** 3) * y), sum((x ** 2) * y), sum(x * y), sum(y)]) 

print(f"A={A}")
print(f"b={b}")

def f2(x):
    """Часть 2: функция регрессии 3 степени"""
    return 1.940 * (x ** 3) - 0.017 * (x ** 2) + 0.224 * x -5.862

def f2_(x):
    """Часть 2: функция регрессии 3 степени"""
    return 1.940 * (x ** 3) -5.862

# расчет 
F2_sogl = D(y) / D1(y, f2_, x, 3)

print(f"Оценка дисперсии выходной переменной: {D(y)}")
print(f"Оценка остаточной дисперсии: {D1(y, f2_, x, 3)}")
print(f"Показатель согласованности по критерию Фишера: {F2_sogl}")


# проверка согласованности по критерию Стьюдента
a = [1.940, 0.017, 0.224, 5.862]

sigma_1 = D1(y, f2, x, 3)

F = np.array([
    [x[0] ** 3, x[0] ** 2, x[0], 1],
    [x[1] ** 3, x[1] ** 2, x[1], 1],
    [x[2] ** 3, x[2] ** 2, x[2], 1],
    [x[3] ** 3, x[3] ** 2, x[3], 1],
    [x[4] ** 3, x[4] ** 2, x[4], 1],
    [x[5] ** 3, x[5] ** 2, x[5], 1],
    [x[6] ** 3, x[6] ** 2, x[6], 1]
])

F_T = F.transpose()
F_T_F = F_T.dot(F)
F_T_F_1 = np.linalg.inv(F_T_F)

K_A = F_T_F_1 * sigma_1
diag = np.array([K_A[i,i] for i in range(len(K_A))])
t = a / (diag ** (1/2))

print(f"Корреляционная матрица:\n{K_A}")
print(f"Главная диагональ: {diag}")
print(f"СКО отклонения aj: {diag ** (1/2)}")
print(f"Вектор коэффициентов |a|: {a}")
print(f"Значения показателя согласованности: {t}")


# FTY = F_T.dot(y)
# A_ = F_T_F_1.dot(FTY)



draw_compare_plot(x, y, f2_)
plot.legend()
plot.show()

exit()








draw_compare_plot(x, y, f1)
plot.legend()
plot.show()



