import matplotlib.pyplot as plt
import math
import numpy as np

def P(M):
    """Получить вероятность попадания величины в диапазон"""
    return [m / sum(M) for m in M]

def draw_hist(J, P_):
    """Построить график в plt"""
    vals = []
    x_bar_positions = []
    # расчет высот столбцов на основе вероятностей
    for i, p in enumerate(P_):
        vals += [p / (J[i+1] - J[i])]
        x_bar_positions += [J[i] + (J[i+1] - J[i]) / 2]
    plt.bar(
        x_bar_positions, 
        vals, 
        width=J[1]-J[0], # одинаковая ширина столбцов
        edgecolor = "black")

# интервалы
J = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# количество попаданий в каждый интервал
M_ = [3, 16, 22, 27, 15, 11, 6]

N = sum(M_)

# найти статистические вероятности попадания 
# значений случайной величины в интервалы 
P_ = P(M_)

print(f"Интервалы: {J}")
print(f"Кол-во попаданий: {M_}")
print(f"Вероятности: {P_}")


draw_hist(J, P_)


def M(J, P_):
    """Вычислить мат. ожидание
    Интервального статистического ряда"""
    m = 0
    for i, p in enumerate(P_):
        m += 0.5 * (J[i] + J[i+1]) * p
    return m

def D(J, P_):
    """Вычислить дисперсию ИСР"""
    m = M(J, P_)
    d = 0    
    for i, p in enumerate(P_):
        d += ((0.5 * (J[i] + J[i+1])) ** 2) * p
    d -= m ** 2
    return d

def SKO(J, P_):
    """Вычислить СКО ИСР"""
    return math.sqrt(D(J, P_))


m = M(J,P_)
d = D(J, P_)
sko = SKO(J, P_)

print(f"Мат. ожидание: {m}")
print(f"Дисперсия: {d}")
print(f"СКО: {sko}")

def f_approx(x, m, sko):
    """Функция аппроксимирующей кривой 
    плотности распределения вероятностей"""
    f = 1 / (sko * math.sqrt(2 * math.pi))
    power = - ((x - m) ** 2)/(2 * (sko ** 2))
    f = f * (math.e ** power)
    return f

def draw_normal(m, sko, start, end):
    """Нарисовать выравнивающий график плотности распределения"""
    X = [x for x in np.arange(start, end, 0.01)]
    Y = [f_approx(x, m, sko) for x in X]
    plt.plot(X, Y, color="red")

draw_normal(m, sko, -0.1, 0.8)

#plt.show()


import pandas as pd
np.set_printoptions(suppress=True)

def prepare_f1():
    """Подготовить функцию F1 из приложения"""
    df = pd.read_excel("./Таблицыприложения.xlsx", sheet_name="Приложение3")
    values = {}
    for v in zip(df.iloc[:, 1], df.iloc[:, 2]):
        values[v[0]]=v[1]
    def f(x):
        """Функция Ф1"""
        x=round(x, 2)
        if x not in values:
            x=round(x, 1)
        return values[x] if x in values else None
    return f


f1 = prepare_f1()

def calc_p(J, m, sko):
    """Вычислить вероятности попадания 
    случайной величины в диапазоны
    по нормальному закону"""
    p_list = []
    for i in range(len(J) - 1):
        p = f1((J[i+1] - m)/sko) - f1((J[i] - m)/sko)
        p_list += [p]
    return p_list

p_n = calc_p(J, m, sko)

# вывести вероятности из статистического ряда 
# и из нормального распределения
print(f"p*l: {P_}")
print(f"pl: {np.round(p_n, 4)}")

# разность между вероятностями
diff = np.array(P_, np.float64) - np.array(p_n, np.float64)
print(f"p*l-pl: {diff}")

# квадрат разностей
powered = np.pow(diff, 2)
print(f"(p*l-pl)^2: {np.round(powered, 4)}")

# умножение на кол-во измерений
# и деление на теоретическую вероятность
result = (powered / np.array(p_n, np.float64)) * N #* N
print(f"n(p*l-pl)^2/pl: {np.round(result, 4)}")

# сумма
u = sum(result)

print(f"u = {u}")
