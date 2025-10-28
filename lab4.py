# Лабораторная работа №4
# Многофакторный регрессионный анализ
# Вариант №4
import numpy as np
np.set_printoptions(suppress=True)

x1 = np.array([0.5, 1, 2, 4, 3.5, -1])
x2 = np.array([0, 1, 1.5, 2, 2.5, 1])
y = [0.2, 5, 8.8, 15.2, 17, 1]

# центрирование факторов
def avg(n):
    s=0
    for x in n:
        s+=x
    return np.float64(s / len(n))

def center(n):
    average = avg(n)
    res=[]
    for x in n:
        res += [x - average]
    return np.array(res)

x1_ = center(x1)
x2_ = center(x2)

print(f"Исходные:\n{x1}, мат.ожид:{round(avg(x1), 2)}\n{x2}, мат.ожид:{round(avg(x2), 2)}")
print(f"Центрированные:\n{x1_}, мат.ожид:{round(avg(x1_), 2)}\n{x2_}, мат.ожид:{round(avg(x2_), 2)}")

# 2. Построение матричного уравнения для 
# нахождения коэффициентов регрессии

X_6_3 = np.array([ [1] * len(x1_), x1_, x2_ ]).T
X_T_X = X_6_3.T.dot(X_6_3)
X_T_X_1 = np.linalg.inv(X_T_X)
X_T_Y = X_6_3.T.dot(y)

print(f"X:\n{X_6_3.round(2)}") # матрица центрированных факторов
print(f"X_T*X:\n{X_T_X.round(2)}") # транспонированная X на X
print(f"(X_T*X)^-1:\n{X_T_X_1.round(2)}") # обратная матрица
print(f"X_T * Y:\n{X_T_Y.round(2)}") #транспонированная на вектор Y

print(f"Матричное уравнение:\nB={X_T_X_1.round(2)}*{X_T_Y.round(2)}")


# 3. Нахождение решения матричного уравнения:

B = X_T_X_1.dot(X_T_Y)
print(f"B={B.round(2)}")

# 4. Проверка адекватности по критерию Фишера

def f(x1, x2):
    """Получившаяся функция регрессии"""
    return B[0] + x1 * B[1] + x2 * B[2]

def D(Y):
    """Оценка дисперсии выходной переменной"""
    average = avg(y)
    n = np.array([y - average for y in Y])
    n = n ** 2
    res = sum(n) / (len(Y) - 1)
    return res

def D1(Y):
    """Оценка остаточной дисперсии"""
    average = avg(y)
    n = np.array([y - f(x1_[i], x2_[i]) for i, y in enumerate(Y)])
    n = n ** 2
    res = sum(n) / (len(Y) - 2 - 1)
    return res

d = D(y)
d1 = D1(y)
F = d/d1

print(f"Оценка дисперсии выходной переменной = {d}")
print(f"Оценка остаточной дисперсии = {d1}")
print(f"Показатель согласованности = {F}")


# Селекция факторов по критерию Стьюдента
K = d1 * X_T_X_1
diag = [K[i,i] for i in range(len(K))]
sigma = np.array(diag) ** (1/2)
t = abs(np.array(B)) / sigma


print(f"Показатели согласованности факторов: {t}")
