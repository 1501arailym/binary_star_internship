# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Гравитационная постоянная
G = 6.67430e-11  # м³/(кг·с²)

# Массы звезд (в кг)
M_sun = 1.989e30  # масса Солнца
m1 = 1.10 * M_sun  # Alpha Centauri A
m2 = 0.123 * M_sun  # Alpha Centauri B

# Большая полуось (в метрах) и эксцентриситет
a = 23.64 * 1.496e11  # 23.64 а.е. в метрах
e = 0.52  # Эксцентриситет орбиты

# Вычисляем период обращения по 3-му закону Кеплера
mu = G * (m1 + m2)  # Приведённый гравитационный параметр
T = 2 * np.pi * np.sqrt(a**3 / mu)  # Период обращения

# Начальные условия (в перицентре)
r_peri = a * (1 - e)  # Расстояние в перицентре
v_peri = np.sqrt(mu * (1 + e) / (a * (1 - e)))  # Скорость в перицентре

# В начальный момент m1 и m2 находятся в перицентре и движутся в противоположные стороны
x1_0, y1_0 = -m2 / (m1 + m2) * r_peri, 0  # Положение m1
x2_0, y2_0 = m1 / (m1 + m2) * r_peri, 0  # Положение m2
vx1_0, vy1_0 = 0, -m2 / (m1 + m2) * v_peri  # Скорость m1
vx2_0, vy2_0 = 0, m1 / (m1 + m2) * v_peri  # Скорость m2

# Функция для интегрирования
def two_body(t, y):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = y

    r = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    ax1 = G * m2 * (x2 - x1) / r**3
    ay1 = G * m2 * (y2 - y1) / r**3
    ax2 = -G * m1 * (x2 - x1) / r**3
    ay2 = -G * m1 * (y2 - y1) / r**3

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

# Время интегрирования (увеличено до 10 периодов)
t_span = (0, 10 * T)
t_eval = np.linspace(0, 10 * T, 10000)  # Больше точек

# Начальные условия
y0 = [x1_0, y1_0, vx1_0, vy1_0, x2_0, y2_0, vx2_0, vy2_0]

# Решаем систему уравнений (точный метод DOP853)
sol = solve_ivp(two_body, t_span, y0, method='DOP853', t_eval=t_eval)

# Получаем координаты
x1, y1 = sol.y[0], sol.y[1]
x2, y2 = sol.y[4], sol.y[5]

# Строим орбиты
plt.figure(figsize=(8, 8))
plt.plot(x1, y1, label="Alpha Centauri A (m1)", color="blue")
plt.plot(x2, y2, label="Alpha Centauri B (m2)", color="red")
plt.plot(0, 0, 'yo', markersize=8, label="Центр масс")

# Оформление
plt.xlabel("X (м)")
plt.ylabel("Y (м)")
plt.legend()
plt.title("Полные орбиты Alpha Centauri A и B (10 орбит)")
plt.axis("equal")
plt.grid()
plt.show()


